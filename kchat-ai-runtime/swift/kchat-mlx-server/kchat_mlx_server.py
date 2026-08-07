#!/usr/bin/env python3
"""kchat-mlx-server: MLX inference server for kchat eval.

Provides a llama-server-compatible /completion endpoint using mlx-lm.
This is the fallback when the Swift kchat-mlx-server cannot build (e.g.,
no Xcode/Metal compiler available).
"""

import argparse
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# Global model state
MODEL = None
TOKENIZER = None


def load_model(model_path: str):
    global MODEL, TOKENIZER
    print(f"kchat-mlx-server (python): loading model from {model_path}...")
    MODEL, TOKENIZER = load(model_path)
    print(f"kchat-mlx-server (python): model loaded successfully")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress default logging

    def _send_json(self, status: int, body: dict):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": []})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/completion":
            self._handle_completion()
        elif self.path == "/v1/chat/completions":
            self._handle_chat_completion()
        else:
            self._send_json(404, {"error": "not found"})

    def _read_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        return json.loads(body) if body else {}

    def _handle_completion(self):
        try:
            req = self._read_body()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        prompt = req.get("prompt", "")
        max_tokens = req.get("n_predict", 128)
        temperature = req.get("temperature", 0.7)
        top_p = req.get("top_p", 0.9)
        seed = req.get("seed", 42)

        mx.random.seed(seed)

        prompt_start = time.time()
        # Generate
        sampler = make_sampler(temp=temperature, top_p=top_p)
        decode_start = time.time()
        response = generate(
            MODEL,
            TOKENIZER,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        decode_ms = (time.time() - decode_start) * 1000.0

        # Estimate token counts
        prompt_tokens = len(TOKENIZER.encode(prompt))
        output_tokens = len(TOKENIZER.encode(response))

        self._send_json(200, {
            "content": response,
            "tokens_predicted": output_tokens,
            "tokens_evaluated": prompt_tokens,
            "prompt_ms": 0.0,
            "predicted_ms": decode_ms,
            "predicted_per_token_ms": decode_ms / max(output_tokens, 1),
            "prompt_per_token_ms": 0.0,
        })

    def _handle_chat_completion(self):
        try:
            req = self._read_body()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        messages = req.get("messages", [])
        max_tokens = req.get("max_tokens", 128)
        temperature = req.get("temperature", 0.7)
        top_p = req.get("top_p", 0.9)
        seed = req.get("seed", 42)

        mx.random.seed(seed)

        # Convert messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)

        sampler = make_sampler(temp=temperature, top_p=top_p)
        decode_start = time.time()
        response = generate(
            MODEL,
            TOKENIZER,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        decode_ms = (time.time() - decode_start) * 1000.0

        prompt_tokens = len(TOKENIZER.encode(prompt))
        output_tokens = len(TOKENIZER.encode(response))

        self._send_json(200, {
            "id": "chatcmpl-pymlx",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,
            },
        })


def main():
    parser = argparse.ArgumentParser(description="kchat-mlx-server (Python fallback)")
    parser.add_argument("--model", "-m", required=True, help="Path to MLX model directory")
    parser.add_argument("--port", "-p", type=int, default=18888, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    load_model(args.model)

    server = HTTPServer((args.host, args.port), Handler)
    print(f"kchat-mlx-server (python) listening on http://{args.host}:{args.port}")
    print("  GET  /health              — health check")
    print("  POST /completion          — completion API")
    print("  POST /v1/chat/completions — chat completion API")
    import sys
    sys.stdout.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()

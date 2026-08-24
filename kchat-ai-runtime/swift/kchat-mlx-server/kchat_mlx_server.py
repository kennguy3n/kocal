#!/usr/bin/env python3
"""kchat-mlx-server: MLX inference server for kchat eval.

Provides a llama-server-compatible /completion endpoint using mlx-lm.
This is the fallback when the Swift kchat-mlx-server cannot build (e.g.,
no Xcode/Metal compiler available).

Supports LoRA adapter hot-swap via --adapter and POST /lora/load.
"""

import argparse
import json
import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# Global model state
MODEL = None
TOKENIZER = None
MODEL_PATH = None
ADAPTER_PATH = None
LOCK = threading.Lock()


def load_model(model_path: str, adapter_path: str = None):
    global MODEL, TOKENIZER, MODEL_PATH, ADAPTER_PATH
    print(f"kchat-mlx-server (python): loading model from {model_path}...")
    if adapter_path:
        print(f"  with adapter: {adapter_path}")
        MODEL, TOKENIZER = load(model_path, adapter_path=adapter_path)
    else:
        MODEL, TOKENIZER = load(model_path)
    MODEL_PATH = model_path
    ADAPTER_PATH = adapter_path
    print(f"kchat-mlx-server (python): model loaded successfully")


def swap_adapter(adapter_path: str):
    """Hot-swap LoRA adapter by reloading model with new adapter."""
    global MODEL, TOKENIZER, ADAPTER_PATH
    with LOCK:
        if adapter_path and not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"adapter dir not found: {adapter_path}")
        print(f"kchat-mlx-server (python): swapping adapter to {adapter_path}...")
        if adapter_path:
            MODEL, TOKENIZER = load(MODEL_PATH, adapter_path=adapter_path)
        else:
            MODEL, TOKENIZER = load(MODEL_PATH)
        ADAPTER_PATH = adapter_path
        print(f"kchat-mlx-server (python): adapter swapped")


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
        elif self.path == "/lora/load":
            self._handle_lora_load()
        elif self.path == "/lora/detach":
            self._handle_lora_detach()
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

        # Convert messages to prompt using chat template if available
        if hasattr(TOKENIZER, "apply_chat_template"):
            prompt = TOKENIZER.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
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

    def _handle_lora_load(self):
        try:
            req = self._read_body()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return
        adapter_path = req.get("adapter_path", "")
        if not adapter_path:
            self._send_json(400, {"error": "adapter_path required"})
            return
        try:
            swap_adapter(adapter_path)
            self._send_json(200, {"status": "ok", "adapter": adapter_path})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def _handle_lora_detach(self):
        try:
            swap_adapter(None)
            self._send_json(200, {"status": "ok", "adapter": None})
        except Exception as e:
            self._send_json(500, {"error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="kchat-mlx-server (Python fallback)")
    parser.add_argument("--model", "-m", required=True, help="Path to MLX model directory")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--port", "-p", type=int, default=18888, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    load_model(args.model, adapter_path=args.adapter)

    server = HTTPServer((args.host, args.port), Handler)
    print(f"kchat-mlx-server (python) listening on http://{args.host}:{args.port}")
    print("  GET  /health              - health check")
    print("  POST /completion          - completion API")
    print("  POST /v1/chat/completions - chat completion API")
    print("  POST /lora/load           - hot-swap LoRA adapter")
    print("  POST /lora/detach         - detach LoRA adapter")
    import sys
    sys.stdout.flush()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()

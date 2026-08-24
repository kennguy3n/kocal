#!/usr/bin/env python3
"""Lightweight MLX inference server with LoRA adapter support.

Provides a llama-server compatible API (/completion, /health) so the
kchat-task-suite per-device eval can test MLX models with LoRA adapters.

Usage:
    python3 mlx_server_with_lora.py --model <path> [--port <port>] [--lora <adapter_path>]
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import mlx.core as mx

# Thinking tokens for Qwen3 (defined via hex to avoid encoding issues)
THINK_START = bytes.fromhex('3c7468696e6b3e').decode('utf-8')     # <think>
THINK_END = bytes.fromhex('3c2f7468696e6b3e').decode('utf-8')     # </think>

# Globals (set during startup)
MODEL = None
TOKENIZER = None
ADAPTER_LOADED = False
_model_loaded = False


def load_model(model_path: str, adapter_path: str | None = None):
    """Load MLX model and optionally apply LoRA adapter."""
    global MODEL, TOKENIZER, ADAPTER_LOADED, _model_loaded
    from mlx_lm import load
    MODEL, TOKENIZER = load(str(model_path))

    if adapter_path:
        from mlx_lm.tuner.utils import load_adapters
        from pathlib import Path
        adapter_dir = Path(adapter_path)
        if adapter_dir.is_file():
            adapter_dir = adapter_dir.parent
        load_adapters(MODEL, str(adapter_dir))
        ADAPTER_LOADED = True
        print(f"  LoRA adapter loaded from {adapter_dir}", file=sys.stderr)
    else:
        ADAPTER_LOADED = False

    _model_loaded = True


def strip_thinking(text: str) -> str:
    """Strip Qwen3 thinking tokens from output.

    Handles two cases:
    1. Model outputs <think>...</think> - strip everything up to </think>
    2. Model outputs <think> without </think> - strip the <think> token
    """
    if THINK_END in text:
        parts = text.split(THINK_END, 1)
        if len(parts) > 1:
            return parts[1].strip()
    if THINK_START in text:
        idx = text.find(THINK_START)
        after = text[idx + len(THINK_START):]
        if after.startswith('\n'):
            after = after[1:]
        return after.strip()
    return text


def generate_text(prompt: str, max_tokens: int = 256, temperature: float = 0.7,
                  top_p: float = 0.9, seed: int = 42) -> dict:
    """Generate text using MLX model."""
    from mlx_lm import generate
    from mlx_lm.generate import make_sampler

    chat_prompt = prompt
    if "<|im_start|>" not in prompt:
        if "JSON" in prompt or "json" in prompt:
            chat_prompt = (
                "<|im_start|>system\nYou are a JSON generator. "
                "Output ONLY valid JSON. No explanation, no markdown, "
                "no thinking, no extra text. Start with { and end with }.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    mx.random.seed(seed)

    t0 = time.time()
    sampler = make_sampler(temp=temperature, top_p=top_p)
    output = generate(
        MODEL, TOKENIZER, chat_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )
    predicted_ms = (time.time() - t0) * 1000.0

    final_text = strip_thinking(output)
    estimated_tokens = max(len(final_text) // 4, 1)

    return {
        "content": final_text,
        "tokens_predicted": estimated_tokens,
        "tokens_evaluated": 0,
        "timings": {
            "prompt_ms": 0,
            "predicted_ms": predicted_ms,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            if _model_loaded:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            else:
                self.send_response(503)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return

        if self.path == "/completion":
            prompt = data.get("prompt", "")
            max_tokens = data.get("n_predict", 256)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)
            seed = data.get("seed", 42)

            result = generate_text(prompt, max_tokens, temperature, top_p, seed)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        elif self.path == "/v1/chat/completions":
            messages = data.get("messages", [])
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)

            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            prompt_parts.append("<|im_start|>assistant\n")
            prompt = "\n".join(prompt_parts)

            result = generate_text(prompt, max_tokens, temperature, 0.9, 42)

            response = {
                "choices": [{
                    "message": {"role": "assistant", "content": result["content"]},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": result["tokens_evaluated"],
                    "completion_tokens": result["tokens_predicted"],
                },
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser(description="MLX inference server with LoRA support")
    ap.add_argument("--model", "-m", required=True, help="Path to MLX model directory")
    ap.add_argument("--port", "-p", type=int, default=18888, help="Port (default: 18888)")
    ap.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    ap.add_argument("--lora", default=None, help="Path to LoRA adapter directory or safetensors file")
    args = ap.parse_args()

    # Load model synchronously in main thread (MLX Metal streams are thread-local)
    print(f"kchat-mlx-server (Python) starting on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"  Loading model from {args.model}...", file=sys.stderr)
    try:
        load_model(args.model, args.lora)
        lora_status = f" + LoRA: {args.lora}" if ADAPTER_LOADED else " (no LoRA)"
        print(f"  Model loaded{lora_status} - ready", file=sys.stderr)
    except Exception as e:
        print(f"  FATAL: failed to load model: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Start server (single-threaded - all requests handled in main thread)
    server = HTTPServer((args.host, args.port), Handler)

    print(f"  GET  /health              - health check", file=sys.stderr)
    print(f"  POST /completion          - completion API", file=sys.stderr)
    print(f"  POST /v1/chat/completions - chat completion API", file=sys.stderr)

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    server.serve_forever()


if __name__ == "__main__":
    main()

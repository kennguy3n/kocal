# kchat-ai-runtime

> KChat H2 2026 On-Device AI Runtime — a Rust workspace implementing the
> deterministic-first, privacy-first, tier-aware AI runtime for KChat.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)]()
[![Rust](https://img.shields.io/badge/rust-2021-orange)]()
[![Tests](https://img.shields.io/badge/tests-987%20unit%20%2B%203644%20realworld%20eval-brightgreen)]()

## Overview

kchat-ai-runtime is a complete on-device AI runtime designed for KChat, a
privacy-first messaging platform. The runtime executes AI workloads entirely
on the user's device — no cloud round-trips required for safety classification,
text generation, context retrieval, or action validation.

The system is built around four core principles:

1. **Deterministic-first** — Safety classification works on ALL devices without
   a generative model. NFKC normalization, PII detection, scam/URL detectors,
   prompt-injection detection, and signed policy packs operate even on 4GB
   low-tier phones and WASM.
2. **Tier-aware** — Devices are classified into Low/Medium/High tiers at runtime
   based on safe allocatable memory and thermal state. All tiers use the same
   1.7B base generative model family with LoRA adapters; tiers differ in
   context window size, output budget, and performance targets.
3. **Privacy-first** — Per-scope XChaCha20-Poly1305 encryption, append-only
   evidence chains, no raw content in telemetry. All inference is local.
4. **Signed distribution** — Ed25519-signed manifests and policy packs with
   pinned keys prevent tampering with model weights or safety rules.

## Quick Start

```bash
# Build all crates
cargo build --workspace

# Run all unit tests (987 tests)
cargo test --workspace

# Run the standard eval harness (369 cases)
cargo run -p kchat-task-suite

# Run the real-world eval (3,644 cases with real model inference)
cargo run -p kchat-task-suite -- --realworld

# Run the per-device eval (12 profiles × 251 tasks = 3,012 task runs)
cargo run -p kchat-task-suite -- --perdevice

# Run the red-team eval suite (36 attack cases)
cargo run -p kchat-task-suite -- --redteam

# Run the slides eval (2,171 mock cases; add --realworld for real model)
cargo run -p kchat-task-suite -- --slides

# Run the skills eval (307 cases) and device simulator (12 profiles)
cargo run -p kchat-task-suite -- --skills
cargo run -p kchat-task-suite -- --simulate
```

## Workspace Structure

```
kchat-ai-runtime/
├── crates/
│   ├── kchat-core/          # Capability probe, tier selection, registry, model manager
│   ├── kchat-encoder/       # mmBERT-small GGUF encoder: safety head + embeddings + reranker
│   ├── kchat-safety/        # Deterministic safety plane, policy packs, vision encoder
│   ├── kchat-asr/           # Whisper ASR: ONNX Runtime + whisper.cpp GGML backends
│   ├── kchat-context/       # SQLCipher store, FTS5 retrieval, embeddings, reranker
│   ├── kchat-generation/    # Grammar-constrained generation, llama.cpp/MLX backends
│   ├── kchat-action/        # Artifact AST, ToolPlan validation, RBAC, audit log
│   ├── kchat-image/         # Image search: Pexels/Pixabay/Unsplash/Shutterstock
│   ├── kchat-runtime/       # Orchestrator, SkillRouter, ConversationMemory
│   ├── kchat-bindings/      # UniFFI (Swift/Kotlin) + N-API (Node.js) FFI
│   └── kchat-wasm/          # WebAssembly safety plane (~2.1MB)
├── eval/
│   └── kchat-task-suite/    # Eval harness: safety, context, generation, action, red-team,
│                            #   slides, skills, per-device, device simulator
├── sidecars/
│   ├── kchat-server-offload/      # Go: server-side offload service
│   └── kchat-generative-sidecar/  # Rust: generative serving sidecar (stub)
├── swift/
│   └── kchat-mlx-server/    # Swift MLX server for Apple Silicon inference
├── manifest/
│   └── packs/               # Downloaded model packs (GGUF, MLX, ONNX, GGML)
└── docs/                    # Documentation
```

## Device Tiers

| Tier | Mobile RAM | Desktop RAM | Context (all platforms) | Output | Peak Memory | TTFT P95 |
|------|-----------|-----------|---------|--------|------------|----------|
| **Low** | 4–6 GB | 8 GB | 4,096 tok | 64–192 tok | 750 MB (mobile) / 2 GB (desktop) | 2,500 ms |
| **Medium** | 6–8 GB | 16–24 GB | 8,192 tok | 256–512 tok | 1,700 MB (iOS) / 1,800 MB (Android) / 4 GB (desktop) | 1,500 ms |
| **High** | 8 GB+ | 32 GB+ | 16,384 tok | 512–1,024 tok | 3,100 MB (iOS) / 3,200 MB (Android) / 8 GB (desktop) | 1,000 ms |

### Tier Selection Thresholds

| Platform | High | Medium | Low |
|----------|------|--------|-----|
| iOS / Android | ≥ 6,000 MB safe | ≥ 3,500 MB safe | < 3,500 MB |
| macOS / Windows | ≥ 20,000 MB safe | ≥ 10,000 MB safe | < 10,000 MB |

Thermal downgrade: Serious → drop one tier; Critical → force Low.

## Model Registry (13 packs)

### Generative Models (7 packs)

All tiers share the 1.7B Bonsai family (ternary-quantized Qwen3-1.7B) with two
quality modes — **Fast** (1-bit) and **Quality** (2-bit) — plus a Qwen3 Q4_K_M
lineup as standard-quant alternates. LoRA adapters (75 family-based:
5 task-families × 15 language slots; plus 270 task-based legacy adapters)
provide task/language specialization.

| Pack ID | Min Tier | Size | Quant | Backend | Platform |
|---------|----------|------|-------|---------|----------|
| `bonsai-1.7b-mlx-1bit` | Low | 269 MB | 1bit-MLX | MLX | iOS/macOS (Apple Silicon) |
| `bonsai-1.7b-q1_0` | Low | 248 MB | Q1_0 | llama.cpp Vulkan/CPU | Android/Windows/Intel Mac |
| `bonsai-1.7b-mlx-2bit` | Low | 484 MB | 2bit-MLX | MLX | iOS/macOS (Apple Silicon) |
| `bonsai-1.7b-q2_0` | Low | 442 MB | Q2_0 | llama.cpp Vulkan/CPU | Android/Windows/Intel Mac |
| `qwen3-0.6b-q4_k_m` | Low | 484 MB | Q4_K_M | llama.cpp | all |
| `qwen3-1.7b-q4_k_m` | Medium | 1.28 GB | Q4_K_M | llama.cpp | all |
| `qwen3-4b-instruct-2507-q4_k_m` | High | 2.5 GB | Q4_K_M | llama.cpp | all |

### Non-Generative Models

| Pack ID | Type | Min Tier | Size | Quant | Tasks | SHA-256 |
|---------|------|----------|------|-------|-------|---------|
| `mmbert-safety-q4_k_m` | encoder | Low | 145 MB | Q4_K_M GGUF | safety, embed, rerank | placeholder |
| `mobileclip-s2-int8` | vision | Low | 102 MB | INT8 | image_classify, image_embed, video_classify | ✅ real |
| `whisper-tiny` | asr | Low | 33 MB | ONNX (FP32) | transcribe (multilingual) | ✅ real |
| `whisper-base` | asr | Medium | 82 MB | ONNX (FP32) | transcribe (multilingual) | ✅ real |
| `whisper-tiny-ggml` | asr | Low | 78 MB | GGML (F32) | transcribe (whisper.cpp, mobile) | ✅ real |
| `whisper-base-ggml` | asr | Medium | 148 MB | GGML (F32) | transcribe (whisper.cpp, mobile) | ✅ real |

9/13 packs have real SHA-256 hashes (the 4 Bonsai generative packs are pending
final export).

> **Note**: Whisper ONNX files are FP32 (not INT8-quantized). Base models are `nb-whisper-tiny` and
> `nb-whisper-base` from NbAiLab (Norwegian fine-tunes of OpenAI Whisper). Despite the
> Norwegian fine-tuning, both models retain full multilingual capability
> (en, vi, zh, ja, ko, es, fr, de, ar, hi, th). The GGML packs are the
> standard whisper.cpp F32 files for the in-process mobile backend.

## Performance Targets

| Metric | Low | Medium | High |
|--------|-----|--------|------|
| Context cap (all platforms) | 4,096 tokens | 8,192 tokens | 16,384 tokens |
| TTFT P95 | 2,500 ms | 1,500 ms | 1,000 ms |
| Decode P50 (mobile) | 8 tok/s | 15 tok/s | 25 tok/s |
| Decode P50 (desktop) | 10 tok/s | 20 tok/s | 35 tok/s |
| Max perf cores | 2 | 3 | 4 |
| Idle unload (mobile) | 45 s | 45 s | 45 s |
| Idle unload (desktop) | 300 s | 300 s | 300 s |

## Eval Coverage

| Suite | Cases | Status |
|-------|-------|--------|
| Unit tests | 987 | All passing |
| Standard eval | 369 | All passing |
| Red-team eval | 36 | 100% (7 attack categories) |
| Real-world eval | 3,644 | Safety 95.8%, Guardrail 100%, Context 87.6%, Generation 98.3%, Action 100% |
| Skills eval | 307 | Document/chat skill coverage |
| Slides mock eval | 2,171 | 100% (12 skills × 210 templates) |
| Image search eval | 170 | Real provider APIs (requires keys) |
| Per-device eval | 3,012 | 12 profiles × 251 tasks × 4 generative models |
| Device simulator | 12 profiles × full decision tree | All passing |

## Platform Support

| Platform | Backend | Generative | Safety | Vision | ASR |
|----------|---------|-----------|--------|--------|-----|
| iOS (Apple Silicon) | MLX | Bonsai MLX | GGUF Q4_K_M | ONNX INT8 | whisper.cpp GGML |
| macOS (Apple Silicon) | MLX | Bonsai MLX | GGUF Q4_K_M | ONNX INT8 | ONNX / GGML |
| macOS (Intel) | llama.cpp CPU | Bonsai GGUF | GGUF Q4_K_M | ONNX INT8 | ONNX |
| Android | llama.cpp Vulkan | Bonsai GGUF | GGUF Q4_K_M | ONNX INT8 | whisper.cpp GGML |
| Windows | llama.cpp Vulkan | Bonsai GGUF | GGUF Q4_K_M | ONNX INT8 | ONNX |
| Web (WASM) | — | — | Deterministic only | — | — |

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Detailed architecture, crate interactions, data flow
- [MODEL.md](MODEL.md) — Complete model registry, device profiles, memory budgets, selection logic
- [AGENTS.md](AGENTS.md) — AI agent guide with build commands and test counts

## License

Apache-2.0

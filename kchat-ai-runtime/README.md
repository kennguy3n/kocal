# kchat-ai-runtime

> KChat H2 2026 On-Device AI Runtime — a Rust workspace implementing the
> deterministic-first, privacy-first, tier-aware AI runtime for KChat.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)]()
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)]()
[![Rust](https://img.shields.io/badge/rust-2021-orange)]()
[![Tests](https://img.shields.io/badge/tests-685%20unit%20%2B%202267%20eval-brightgreen)]()

## Overview

kchat-ai-runtime is a complete on-device AI runtime designed for KChat, a
privacy-first messaging platform. The runtime executes AI workloads entirely
on the user's device — no cloud round-trips required for safety classification,
text generation, context retrieval, or action validation.

The system is built around four core principles:

1. **Deterministic-first** — Safety classification works on ALL devices without
   a generative model. NFKC normalization, PII detection, scam/URL detectors,
   and signed policy packs operate even on 4GB low-tier phones and WASM.
2. **Tier-aware** — Devices are classified into Low/Medium/High tiers at runtime
   based on safe allocatable memory and thermal state. Each tier gets
   appropriately sized models, context windows, and performance budgets.
3. **Privacy-first** — Per-scope XChaCha20-Poly1305 encryption, append-only
   evidence chains, no raw content in telemetry. All inference is local.
4. **Signed distribution** — Ed25519-signed manifests and policy packs with
   pinned keys prevent tampering with model weights or safety rules.

## Quick Start

```bash
# Build all crates
cargo build --workspace

# Run all unit tests (685 tests)
cargo test --workspace

# Run the standard eval harness (233 cases)
cargo run -p kchat-task-suite

# Run the per-device eval (12 profiles × 150 tasks = 1800 task runs)
cargo run -p kchat-task-suite -- --perdevice

# Run the real-world eval (2267 cases with real model inference)
cargo run -p kchat-task-suite -- --realworld

# Run the red-team eval suite (36 attack cases)
cargo run -p kchat-task-suite -- --redteam

# Run the device simulator (12 profiles × 138 checks each)
cargo run -p kchat-task-suite -- --simulate
```

## Workspace Structure

```
kchat-ai-runtime/
├── crates/
│   ├── kchat-core/          # Capability probe, tier selection, registry, model manager
│   ├── kchat-safety/        # Deterministic safety plane, policy packs, vision encoder
│   ├── kchat-context/       # SQLCipher store, FTS5 retrieval, embeddings, reranker
│   ├── kchat-generation/    # Grammar-constrained generation, llama.cpp/MLX backends
│   ├── kchat-action/        # Artifact AST, ToolPlan validation, RBAC, audit log
│   ├── kchat-bindings/      # UniFFI (Swift/Kotlin) + N-API (Node.js) FFI
│   └── kchat-wasm/          # WebAssembly safety plane (~2.1MB)
├── eval/
│   └── kchat-task-suite/    # Eval harness: safety, context, generation, action, red-team
├── sidecars/
│   ├── kchat-server-offload/  # Go: server-side offload service
│   └── kchat-generative-sidecar/  # Rust: generative sidecar
├── swift/
│   └── kchat-mlx-server/    # Swift MLX server for Apple Silicon inference
├── manifest/
│   └── packs/               # Downloaded model packs (GGUF, MLX, ONNX)
└── docs/                    # Documentation
```

## Device Tiers

| Tier | Mobile RAM | Desktop RAM | Context | Output | Peak Memory | TTFT P95 |
|------|-----------|-----------|---------|--------|------------|----------|
| **Low** | 4–6 GB | 8 GB | 2,048 tok | 64–192 tok | 750 MB (mobile) / 2 GB (desktop) | 2,500 ms |
| **Medium** | 6–8 GB | 16–24 GB | 4,096 tok | 256–512 tok | 1,700 MB (iOS) / 1,800 MB (Android) / 4 GB (desktop) | 1,500 ms |
| **High** | 8 GB+ | 32 GB+ | 8,192 tok (mobile) / 16,384 tok (desktop) | 512–1,024 tok | 3,100 MB (iOS) / 3,200 MB (Android) / 8 GB (desktop) | 1,000 ms |

### Tier Selection Thresholds

| Platform | High | Medium | Low |
|----------|------|--------|-----|
| iOS / Android | ≥ 6,000 MB safe | ≥ 3,500 MB safe | < 3,500 MB |
| macOS / Windows | ≥ 20,000 MB safe | ≥ 10,000 MB safe | < 10,000 MB |

Thermal downgrade: Serious → drop one tier; Critical → force Low.

## Model Registry (11 packs)

### Generative Models

| Pack ID | Min Tier | Size | Quant | Backend | Platform |
|---------|----------|------|-------|---------|----------|
| `ternary-bonsai-1.7b-mlx-2bit` | Low | 472 MB | 2bit-MLX | MLX | iOS/macOS (Apple Silicon) |
| `ternary-bonsai-1.7b-q2_0` | Low | 442 MB | Q2_0 | llama.cpp Vulkan/CPU | Android/Windows/Intel Mac |
| `ternary-bonsai-4b-mlx-2bit` | Medium | 1,132 MB | 2bit-MLX | MLX | iOS/macOS (Apple Silicon) |
| `ternary-bonsai-4b-q2_0` | Medium | 1,075 MB | Q2_0 | llama.cpp Vulkan | Android |
| `ternary-bonsai-8b-mlx-2bit` | High | 2,304 MB | 2bit-MLX | MLX | iOS/macOS (Apple Silicon) |
| `ternary-bonsai-8b-q2_0` | High | 2,182 MB | Q2_0 | llama.cpp Vulkan | Android/Windows |

### Non-Generative Models

| Pack ID | Type | Min Tier | Size | Quant | Tasks | SHA-256 |
|---------|------|----------|------|-------|-------|---------|
| `kchat-encoder-int8` | encoder | High | 270 MB | INT8 | safety, embed, rerank | ⏳ placeholder |
| `kchat-encoder-int4` | encoder | Low | 90 MB | INT4 | safety, embed, rerank | ⏳ placeholder |
| `mobileclip-s2-int8` | vision | Low | 70 MB | INT8 | image_classify, image_embed, video_classify | ⏳ placeholder |
| `whisper-tiny-int8` | asr | Low | 33 MB | ONNX | transcribe | ✅ real |
| `whisper-base-int8` | asr | Medium | 82 MB | ONNX | transcribe | ✅ real |

8/11 packs have real SHA-256 hashes. 3 remaining placeholders require ONNX export.

## Performance Targets

| Metric | Low | Medium | High |
|--------|-----|--------|------|
| TTFT P95 | 2,500 ms | 1,500 ms | 1,000 ms |
| Decode P50 (mobile) | 8 tok/s | 15 tok/s | 25 tok/s |
| Decode P50 (desktop) | 10 tok/s | 20 tok/s | 35 tok/s |
| Max perf cores | 2 | 3 | 4 |
| Idle unload (mobile) | 45 s | 45 s | 45 s |
| Idle unload (desktop) | 300 s | 300 s | 300 s |

## Eval Coverage

| Suite | Cases | Status |
|-------|-------|--------|
| Unit tests | 685 | All passing |
| Standard eval | 233 | All passing |
| Red-team eval | 36 | 100% (7 attack categories) |
| Real-world eval | 2,267 | Safety 100%, Context 100%, Generation 82%, Action 100% |
| Per-device eval | 1,800 | 12 profiles × 150 tasks × 7 unique models |
| Device simulator | 138 checks × 12 profiles | All passing |

## Platform Support

| Platform | Backend | Generative | Safety | Vision | ASR |
|----------|---------|-----------|--------|--------|-----|
| iOS (Apple Silicon) | MLX | Bonsai MLX | ONNX INT4/INT8 | ONNX INT8 | ONNX |
| macOS (Apple Silicon) | MLX | Bonsai MLX | ONNX INT4/INT8 | ONNX INT8 | ONNX |
| macOS (Intel) | llama.cpp CPU | Bonsai GGUF | ONNX INT4/INT8 | ONNX INT8 | ONNX |
| Android | llama.cpp Vulkan | Bonsai GGUF | ONNX INT4/INT8 | ONNX INT8 | ONNX |
| Windows | llama.cpp Vulkan | Bonsai GGUF | ONNX INT4/INT8 | ONNX INT8 | ONNX |
| Web (WASM) | — | — | Deterministic only | — | — |

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Detailed architecture, crate interactions, data flow
- [MODEL.md](MODEL.md) — Complete model registry, device profiles, memory budgets, selection logic
- [AGENTS.md](AGENTS.md) — AI agent guide with build commands and test counts

## License

Apache-2.0

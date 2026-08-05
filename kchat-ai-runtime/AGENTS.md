# kchat-ai-runtime

KChat H2 2026 On-Device AI Runtime — a Rust workspace implementing the
deterministic-first, privacy-first, tier-aware AI runtime for KChat.

## Build & Test Commands

```bash
# Build all crates
cargo build --workspace

# Run all unit tests
cargo test --workspace

# Run the standard eval harness (synthetic unit-level evals)
cargo run -p kchat-task-suite

# Run the real-world eval harness (comprehensive datasets + real model inference)
# Requires llama-server running with a GGUF model, or it will auto-start one
cargo run -p kchat-task-suite -- --realworld

# Build with mobile bindings (UniFFI)
cargo build -p kchat-bindings --features mobile

# Build with desktop bindings (N-API)
cargo build -p kchat-bindings --features desktop

# Build WASM bindings for web (deterministic safety plane)
cargo build -p kchat-wasm --target wasm32-unknown-unknown --release
# Output: target/wasm32-unknown-unknown/release/kchat_wasm.wasm (~2.1MB)

# Run the red-team eval suite (36 attack cases)
cargo run -p kchat-task-suite -- --redteam

# Build the Go server-side offload service
cd sidecars/kchat-server-offload && go build && ./kchat-server-offload
```

### Real-World Eval Setup

The `--realworld` mode loads JSON datasets from `eval/kchat-task-suite/datasets/`
and runs comprehensive tests with real model inference:

- **Safety**: 50 cases (benign, PII, harmful, scam, URL risk, obfuscation, injection, multilingual)
  with per-class precision/recall/F1 and latency P50/P95/P99
- **Context**: 12 documents, 12 queries (multilingual, ACL tests) with recall@10 and MRR
- **Generation**: 10 prompts with real Qwen3.5-0.8B inference via llama-server,
  measuring TTFT, decode rate (tok/s), and JSON schema compliance
- **Action**: 16 cases (tool plans, artifact ops, commit tokens, formula injection)

To run generation tests, either:
1. Start llama-server manually: `llama-server -m manifest/packs/Qwen3.5-0.8B-Q4_K_M.gguf --port 18888 -ngl 99`
2. Or let the harness auto-start it (requires llama-server on PATH and model in manifest/packs/)
3. Or set `LLAMA_SERVER_URL` to point to an existing server

## Architecture

The workspace is organized into 8 crates + 1 Go sidecar following the 4-plane architecture:

- **kchat-core**: Capability probe (real OS APIs via sysctl/procfs/Win32),
  device tier selection, scheduler, signed manifest manager, telemetry,
  model manager (CDN download, LRU cache, mmap), resource governor,
  model registry (registry.toml). Foundation for all other crates.
- **kchat-safety**: Deterministic safety plane — NFKC normalization, PII/scam/
  URL detectors, signed policy packs (Ed25519), encoder/SLM escalation,
  ONNX Runtime safety encoder (INT8). Works on ALL devices including
  low-tier (no generative model) and WASM (deterministic only).
- **kchat-context**: Private context plane — SQLCipher encrypted store, FTS5
  BM25 retrieval, per-scope XChaCha20-Poly1305 encryption, provenance bundles,
  dense embeddings (e5-small ONNX + fallback), cross-encoder reranker.
- **kchat-generation**: Grammar-constrained generative plane — prompt templates,
  JSON Schema/regex/Lark grammar validation (real Lark parser), backend
  adapters (llama.cpp via llama-cpp-2 with Metal/Vulkan/Cuda), model lifecycle
  with idle unload, token streaming with safety cancellation, LoRA hot-swap
  (30 adapters: 5 tasks × 6 languages), swarm inference (multi-peer consensus).
- **kchat-action**: Action plane — artifact AST (typed operations, no arbitrary
  code), ToolPlan validation against signed manifests, RBAC authorization
  broker, commit tokens, audit log.
- **kchat-bindings**: FFI surface — UniFFI for Swift/Kotlin (mobile), N-API
  for Node.js (desktop). High-level KChatAiRuntime facade with real
  capability probing and tier-based config selection.
- **kchat-wasm**: WebAssembly bindings for web browsers — exposes the
  deterministic safety plane (classification, PII detection, normalization)
  as a ~2.1MB WASM module. No server-side model required.
- **kchat-task-suite**: Eval harness — safety, context, generation, action,
  integration test suites, and red-team eval suite (36 attack cases across
  7 categories: prompt injection, jailbreak, PII extraction, encoding attacks,
  obfuscation, social engineering, multi-turn).
- **kchat-server-offload** (Go): Server-side offload service — handles AI
  inference when on-device runtime can't (low tier, thermal, battery).
  Gin-based HTTP API with auth, rate limiting, and safety classification.

## Key Design Principles

1. **Deterministic-first**: Safety works on ALL devices without a generative model
2. **Tier-aware**: Low/Medium/High tiers with memory, thermal, and battery downgrades
3. **Privacy-first**: Per-scope encryption, append-only evidence, no raw content in telemetry
4. **Signed distribution**: Ed25519-signed manifests and policy packs with pinned keys
5. **Grammar-constrained**: Model output is always constrained to JSON Schema/regex/Lark
6. **No arbitrary code**: Artifact operations are typed (replace_range, insert_slide, etc.)
7. **Three-step authorization**: Before search, during search, before prompt construction

## Test Counts

- kchat-core: 81 tests (capability probe, model manager, governor, registry)
- kchat-safety: 78 tests (deterministic pipeline, encoder, policy packs)
- kchat-action: 31 tests
- kchat-context: 41 tests (FTS, embeddings, reranker, provenance)
- kchat-generation: 75 tests (llama.cpp backend, LoRA, swarm, Lark grammar)
- kchat-bindings: 12 tests (FFI facade, capability probing, tier selection)
- kchat-wasm: 10 tests (WASM safety classification)
- kchat-task-suite: 8 unit tests + 204 standard eval + 36 red-team cases
  - Standard eval: 43 synthetic + 161 device profile = 204 cases
  - Device profile suite: 12 profiles × 11 test categories + 9 standalone tests = 161 cases
- **Unit total: 358 tests, all passing**
- **Standard eval: 204 cases, all passing**
- **Red-team eval: 36/36 cases (100%) across 7 attack categories**
- **Real-world eval: 2005 safety + 13 context + 11 generation + 17 action = 2046 cases**
  - Safety: 2005/2005 (100%), Context: 13/13 (100%), Generation: 9/11 (82%), Action: 17/17 (100%)
  - Safety dataset v2: 14 languages (en, vi, zh, ja, ko, es, fr, de, ar, hi, th, id, pt, tl) + 13 mixed-lingual code-switch combos
  - Real model: Qwen3.5-0.8B Q4_K_M via llama-server (Metal), ~130 tok/s, 30ms TTFT
- **Go server offload: 7 tests, all passing**

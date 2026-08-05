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

The workspace is organized into 7 crates following the 4-plane architecture:

- **kchat-core**: Capability probe, device tier selection, scheduler, signed
  manifest manager, telemetry. Foundation for all other crates.
- **kchat-safety**: Deterministic safety plane — NFKC normalization, PII/scam/
  URL detectors, signed policy packs (Ed25519), encoder/SLM escalation.
  Works on ALL devices including low-tier (no generative model).
- **kchat-context**: Private context plane — SQLCipher encrypted store, FTS5
  BM25 retrieval, per-scope XChaCha20-Poly1305 encryption, provenance bundles.
- **kchat-generation**: Grammar-constrained generative plane — prompt templates,
  JSON Schema/regex/Lark grammar validation, backend adapters (llama.cpp),
  model lifecycle with idle unload, token streaming with safety cancellation.
- **kchat-action**: Action plane — artifact AST (typed operations, no arbitrary
  code), ToolPlan validation against signed manifests, RBAC authorization
  broker, commit tokens, audit log.
- **kchat-bindings**: FFI surface — UniFFI for Swift/Kotlin (mobile), N-API
  for Node.js (desktop). High-level KChatAiRuntime facade.
- **kchat-task-suite**: Eval harness — safety, context, generation, action,
  and integration test suites with required pass rates.

## Key Design Principles

1. **Deterministic-first**: Safety works on ALL devices without a generative model
2. **Tier-aware**: Low/Medium/High tiers with memory, thermal, and battery downgrades
3. **Privacy-first**: Per-scope encryption, append-only evidence, no raw content in telemetry
4. **Signed distribution**: Ed25519-signed manifests and policy packs with pinned keys
5. **Grammar-constrained**: Model output is always constrained to JSON Schema/regex/Lark
6. **No arbitrary code**: Artifact operations are typed (replace_range, insert_slide, etc.)
7. **Three-step authorization**: Before search, during search, before prompt construction

## Test Counts

- kchat-core: 23 tests
- kchat-safety: 31 tests
- kchat-action: 30 tests
- kchat-context: 19 tests
- kchat-generation: 29 tests
- kchat-bindings: 4 tests
- kchat-task-suite: 43 standard eval cases
- **Unit total: 136 tests, all passing**
- **Standard eval: 43 cases, all passing**
- **Real-world eval: 50 safety + 12 context + 10 generation + 16 action = 88 cases**
  - Safety: 47/50 (94%), Context: 13/13 (100%), Generation: 9/11 (82%), Action: 17/17 (100%)
  - Real model: Qwen3.5-0.8B Q4_K_M via llama-server (Metal), ~130 tok/s, 30ms TTFT

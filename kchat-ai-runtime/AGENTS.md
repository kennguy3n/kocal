# kchat-ai-runtime

KChat H2 2026 On-Device AI Runtime — a Rust workspace implementing the
deterministic-first, privacy-first, tier-aware AI runtime for KChat.

## Build & Test Commands

```bash
# Build all crates
cargo build --workspace

# Run all unit tests
cargo test --workspace

# Run the eval harness
cargo run -p kchat-task-suite

# Build with mobile bindings (UniFFI)
cargo build -p kchat-bindings --features mobile

# Build with desktop bindings (N-API)
cargo build -p kchat-bindings --features desktop
```

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
- kchat-task-suite: 43 eval cases
- **Total: 179 tests, all passing**

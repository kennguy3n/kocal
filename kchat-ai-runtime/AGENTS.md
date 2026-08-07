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

# Run the per-device real-world eval (12 profiles × 150 tasks × real model inference)
# Tests each device profile's assigned model with 150 tasks across 15 categories
# GGUF models use llama-server; MLX models use kchat-mlx-server (Swift or Python fallback)
cargo run -p kchat-task-suite -- --perdevice

# Build the Go server-side offload service
cd sidecars/kchat-server-offload && go build && ./kchat-server-offload

# Build/test with skill-pack feature (overlay-aware policy system)
cargo build -p kchat-safety --features skill-pack
cargo test -p kchat-safety --features skill-pack
```

### Real-World Eval Setup

The `--realworld` mode loads JSON datasets from `eval/kchat-task-suite/datasets/`
and runs comprehensive tests with real model inference:

- **Safety**: 2005 JSON cases (benign, PII, harmful, scam, URL risk, obfuscation, injection, multilingual)
  with per-class precision/recall/F1 and latency P50/P95/P99
- **Guardrail**: 221 YAML cases from `guardrail/text_sample/sample_messages.yaml` using the
  full 17-category taxonomy (0-16), severity rubric (0-5), jurisdiction codes, community
  overlays, and locale tags — tests harmonized classification against `kchat.guardrail.taxonomy.v1`
- **Context**: 12 documents, 12 queries (multilingual, ACL tests) with recall@10 and MRR
- **Generation**: 10 prompts with real Qwen3.5-0.8B inference via llama-server,
  measuring TTFT, decode rate (tok/s), and JSON schema compliance
- **Action**: 16 cases (tool plans, artifact ops, commit tokens, formula injection)

To run generation tests, either:
1. Start llama-server manually: `llama-server -m manifest/packs/Qwen3.5-0.8B-Q4_K_M.gguf --port 18888 -ngl 99`
2. Or let the harness auto-start it (requires llama-server on PATH and model in manifest/packs/)
3. Or set `LLAMA_SERVER_URL` to point to an existing server

### Per-Device Eval Setup

The `--perdevice` mode tests each of the 12 device profiles against its assigned
real model (4 unique models), running 150 tasks across 15 categories:

- **15 task categories**: summarization, translation, structured output, tool use,
  multi-turn conversation, code generation, reasoning, instruction following,
  safety, context retrieval, action, core, generation, WASM, bindings
- **4 unique models**: ternary-bonsai-1.7b-mlx-2bit, ternary-bonsai-1.7b-q2_0,
  ternary-bonsai-4b-q2_0, ternary-bonsai-8b-q2_0
- **Multilingual coverage**: English, Vietnamese, Japanese, Korean, Chinese, Spanish,
  Arabic, Hindi, Thai + mixed-language code-switching scenarios
- **Judgment criteria**: task success rate (50%), quality (35% blended: 15% pass rate + 20% avg quality score),
  TTFT P95 vs tier target (10%), decode P50 vs tier target (5%)
- **Quality scoring**: Each task's quality check returns a 0.0-1.0 score (not just pass/fail).
  Quality pass threshold is ≥0.7. Composite checks use `multi_check` to average sub-checks.
  Check types: min_length, max_length, contains_keyword, contains_all, not_contains,
  exact_match, json_schema_valid, regex_match, coherent, sentence_count, language_script,
  min_words, multi_check
- **Judgment**: Pass (≥75%), Marginal (50-74%), Fail (<50%)

GGUF models run via `llama-server` subprocess. MLX models run via `kchat-mlx-server`
(Swift binary preferred, Python `mlx-lm` fallback if Swift not built).

To build the Swift MLX server:
```bash
cd swift/kchat-mlx-server && swift build -c release
```

If the Swift binary is not available, the harness automatically falls back to
`swift/kchat-mlx-server/kchat_mlx_server.py` (requires `pip install mlx-lm`).

## Architecture

The workspace is organized into 8 crates + 1 Go sidecar following the 4-plane architecture:

- **kchat-core**: Capability probe (real OS APIs via sysctl/procfs/Win32),
  device tier selection, scheduler, signed manifest manager, telemetry,
  model manager (CDN download, LRU cache, mmap), resource governor,
  model registry (registry.toml). Foundation for all other crates.
- **kchat-safety**: Deterministic safety plane — NFKC normalization, PII/scam/
  URL detectors, signed policy packs (Ed25519), encoder/SLM escalation,
  ONNX Runtime safety encoder (INT8/INT4), media descriptor pipeline (10 score
  fields), MobileCLIP-S2 vision encoder (ONNX, feature-gated behind
  `onnx-runtime-vision`), video frame aggregation, vision bridge.
  Skill-pack system (feature-gated behind `skill-pack`): overlay-aware policy
  with 17-category taxonomy, 0-5 severity rubric, community/jurisdiction
  overlays, threshold policy (0.45/0.62/0.78/0.85), policy interpreter with
  SLM rate limiting, canonical JSON, revocation lists, anti-misuse validation.
  kchat-skills/ data tree: global baselines, 38 communities, 62 jurisdictions,
  prompts, transliteration maps, vision prototypes, eval datasets, regulatory docs.
  Works on ALL devices including low-tier (no generative model) and WASM (deterministic only).
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

- kchat-core: 97 tests (capability probe, model manager, governor, registry)
- kchat-safety: 389 tests (deterministic pipeline, encoder, policy packs, vision module)
  - 575 tests with `--features skill-pack` (adds skillpack loader, overlay merge, verifier,
    policy interpreter, threshold policy, revocation, anti-misuse, canonical JSON)
- kchat-action: 31 tests
- kchat-context: 41 tests (FTS, embeddings, reranker, provenance)
- kchat-generation: 80 tests (llama.cpp backend, LoRA, swarm, Lark grammar, MLX)
- kchat-bindings: 12 tests (FFI facade, capability probing, tier selection)
- kchat-wasm: 10 tests (WASM safety classification)
- kchat-task-suite: 8 unit tests + 205 standard eval + 36 red-team cases
  - Standard eval: 44 synthetic + 161 device profile = 205 cases
  - Device profile suite: 12 profiles × 15 test categories + 9 standalone tests = 189 cases
  - Device simulator: `--simulate` flag runs 12 profiles × full decision tree (138 checks)
- **Unit total: 680 tests, all passing**
- **Standard eval: 233 cases, all passing**
- **Red-team eval: 36/36 cases (100%) across 7 attack categories**
- **Real-world eval: 2005 safety + 221 guardrail + 13 context + 11 generation + 17 action = 2267 cases**
  - Safety: 2005/2005 (100%), Guardrail: 73/221 (33.0% — remaining gaps are missing lexicon rules for harassment/hate/extremism/drugs + vision encoder cases), Context: 13/13 (100%), Generation: 9/11 (82%), Action: 17/17 (100%)
  - Safety dataset v2: 14 languages (en, vi, zh, ja, ko, es, fr, de, ar, hi, th, id, pt, tl) + 13 mixed-lingual code-switch combos
  - Guardrail corpus: 221 YAML cases from `sample_messages.yaml` with 17-category taxonomy (0-16), severity rubric (0-5), jurisdiction codes, community overlays, locale tags
  - Real model: Qwen3.5-0.8B Q4_K_M via llama-server (Metal), ~130 tok/s, 30ms TTFT
- **Go server offload: 7 tests, all passing**
- **Per-device eval: 12 profiles × 150 tasks = 1800 task runs (4 unique models)**
  - 15 task categories: summarization, translation, structured output, tool use,
    multi-turn, code generation, reasoning, instruction following, safety,
    context retrieval, action, core, generation, WASM, bindings
  - Multilingual: EN, VI, JA, KO, ZH, ES, AR, HI, TH + mixed-language
  - Judgment: Pass (≥75%), Marginal (50-74%), Fail (<50%)
  - GGUF via llama-server, MLX via kchat-mlx-server (Swift or Python fallback)

## Model Registry (16 packs)

| Pack ID | Type | Min Tier | Size | Quant | Backend | Platform |
|---------|------|----------|------|-------|---------|----------|
| ternary-bonsai-1.7b-mlx-2bit | generative | Low | 472 MB | 2bit-MLX | MLX | ios/macos |
| ternary-bonsai-1.7b-q2_0 | generative | Low | 442 MB | Q2_0 | llama.cpp Vulkan | android/windows |
| qwen3.5-0.8b-q4 | generative | Medium | 500 MB | Q4_K_M | llama.cpp | all |
| ternary-bonsai-4b-q2_0 | generative | High | 1.0 GB | Q2_0 | llama.cpp Vulkan | android |
| macaw-4bit-mlx | generative | High | 1.5 GB | 4bit-MLX | MLX | ios/macos |
| ternary-bonsai-8b-q2_0 | generative | High | 2.1 GB | Q2_0 | llama.cpp Vulkan | windows |
| qwen3.5-0.8b-q8 | generative | High | 850 MB | Q8_0 | llama.cpp | fallback |
| multilingual-e5-small-int8 | embedding | Medium | 45 MB | INT8 | ONNX | all |
| safety-classifier-int8 | safety | Medium | 25 MB | INT8 | ONNX | all |
| safety-classifier-int4 | safety | Low | 15 MB | INT4 | ONNX | all |
| cross-encoder-miniLM-int8 | reranker | High | 25 MB | INT8 | ONNX | all |
| mobileclip-s2-image-int8 | vision | Low | 70 MB | INT8 | ONNX | all |
| mobileclip-s2-image-fp32 | vision | Medium | 137 MB | FP32 | ONNX | all |
| mobileclip-s2-video-int8 | vision | Medium | 70 MB | INT8 | ONNX | all |
| whisper-tiny-int8 | asr | Low | 40 MB | INT8 | ONNX | all |
| whisper-base-int8 | asr | Medium | 90 MB | INT8 | ONNX | all |

### Model selection by tier and platform

- **Low tier**:
  - Generative: iOS/macOS: `ternary-bonsai-1.7b-mlx-2bit` via **MLX** (472MB) / Android/Windows: `ternary-bonsai-1.7b-q2_0` via **llama.cpp Vulkan** (442MB)
  - Vision: `mobileclip-s2-image-int8` (70MB, INT8, 17 categories)
  - Safety: `safety-classifier-int4` (15MB, INT4)
  - ASR: `whisper-tiny-int8` (40MB, INT8)
  - Video: none (deterministic media descriptors only)
- **Medium tier**:
  - Generative: `qwen3.5-0.8b-q4` via **llama.cpp** (500MB)
  - Vision: `mobileclip-s2-image-fp32` (137MB, FP32)
  - Safety: `safety-classifier-int8` (25MB, INT8)
  - ASR: `whisper-base-int8` (90MB, INT8)
  - Video: `mobileclip-s2-video-int8` (70MB, INT8)
- **High tier**:
  - Generative: iOS/macOS: `macaw-4bit-mlx` / Android: `ternary-bonsai-4b-q2_0` / Windows: `ternary-bonsai-8b-q2_0` / Fallback: `qwen3.5-0.8b-q8`
  - Vision: `mobileclip-s2-image-fp32` (137MB, FP32)
  - Safety: `safety-classifier-int8` (25MB, INT8)
  - ASR: `whisper-base-int8` (90MB, INT8)
  - Video: `mobileclip-s2-video-int8` (70MB, INT8)

All generative models support `tool_use`. The "deterministic-first" principle is preserved —
safety works on ALL devices without a generative model. Vision and ASR run on ALL tiers.
Low-tier devices use INT8/INT4 quantized models to fit within memory budget.

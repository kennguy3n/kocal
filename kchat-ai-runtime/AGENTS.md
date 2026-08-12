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
real model (6 unique generative models), running 150 tasks across 15 categories:

- **15 task categories**: summarization, translation, structured output, tool use,
  multi-turn conversation, code generation, reasoning, instruction following,
  safety, context retrieval, action, core, generation, WASM, bindings
- **6 unique generative models**: ternary-bonsai-1.7b-mlx-2bit, ternary-bonsai-1.7b-q2_0,
  ternary-bonsai-4b-mlx-2bit, ternary-bonsai-4b-q2_0, ternary-bonsai-8b-mlx-2bit,
  ternary-bonsai-8b-q2_0
- **Non-generative models per profile**: vision (mobileclip-s2-int8),
  safety encoder (kchat-encoder-int4), ASR (whisper-tiny/base),
  video (mobileclip-s2-int8, same model as vision)
- **Multilingual coverage**: English, Vietnamese, Japanese, Korean, Chinese, Spanish,
  Arabic, German, Hindi, French + mixed-language code-switching scenarios
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

The workspace is organized into 9 crates + 1 Go sidecar following the 4-plane architecture:

- **kchat-core**: Capability probe (real OS APIs via sysctl/procfs/Win32),
  device tier selection, scheduler, signed manifest manager, telemetry,
  model manager (CDN download, LRU cache, mmap), resource governor,
  model registry (registry.toml). Foundation for all other crates.
- **kchat-encoder**: Unified multi-task ONNX encoder — XLM-RoBERTa-base model
  shared across safety classification (10 classes), text embedding (768-dim),
  and cross-encoder reranking. Single model replaces separate safety-classifier,
  e5-small embedding, and MiniLM reranker packs. Feature-gated behind
  `onnx-runtime`; includes mock implementations for testing without ONNX.
- **kchat-safety**: Deterministic safety plane — NFKC normalization, PII/scam/
  URL detectors, signed policy packs (Ed25519), encoder/SLM escalation,
  unified kchat-encoder for safety classification (INT8/INT4), media descriptor
  pipeline (10 score fields), MobileCLIP-S2 vision encoder (ONNX, feature-gated
  behind `onnx-runtime-vision`), video frame aggregation, vision bridge.
  Skill-pack system (feature-gated behind `skill-pack`): overlay-aware policy
  with 17-category taxonomy, 0-5 severity rubric, community/jurisdiction
  overlays, threshold policy (0.45/0.62/0.78/0.85), policy interpreter with
  SLM rate limiting, canonical JSON, revocation lists, anti-misuse validation.
  Embedded skill-pack data (include_str!): global baselines, 38 communities,
  62 jurisdictions, prompts, transliteration maps, vision prototypes, adversarial corpus.
  Works on ALL devices including low-tier (no generative model) and WASM (deterministic only).
- **kchat-context**: Private context plane — SQLCipher encrypted store, FTS5
  BM25 retrieval, per-scope XChaCha20-Poly1305 encryption, provenance bundles,
  dense embeddings (kchat-encoder 768-dim ONNX + fallback), cross-encoder reranker (kchat-encoder).
- **kchat-generation**: Grammar-constrained generative plane — prompt templates,
  JSON Schema/regex/Lark grammar validation (real Lark parser), backend
  adapters (llama.cpp via llama-cpp-2 with Metal/Vulkan/Cuda), model lifecycle
  with idle unload, token streaming with safety cancellation, LoRA hot-swap
  (50 adapters: 5 tasks × 10 languages), swarm inference (multi-peer consensus).
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

- kchat-core: 94 tests (capability probe, model manager, governor, registry)
- kchat-safety: 389 tests (deterministic pipeline, encoder, policy packs, vision module)
  - 927 tests with `--features skill-pack` (adds skillpack loader, overlay merge, verifier,
    policy interpreter, threshold policy, revocation, anti-misuse, canonical JSON,
    jurisdiction tests, community overlay tests, adversarial corpus tests)
- kchat-action: 37 tests
- kchat-context: 44 tests (FTS, embeddings, reranker, provenance, cache invalidation)
- kchat-generation: 84 tests (llama.cpp backend, LoRA, swarm, Lark grammar, MLX)
- kchat-bindings: 12 tests (FFI facade, capability probing, tier selection)
- kchat-wasm: 10 tests (WASM safety classification)
- kchat-encoder: 5 tests (mock safety, embed, rerank, session)
- kchat-task-suite: 8 unit tests + 205 standard eval + 36 red-team cases
  - Standard eval: 44 synthetic + 161 device profile = 205 cases
  - Device profile suite: 12 profiles × 15 test categories + 9 standalone tests = 189 cases
  - Device simulator: `--simulate` flag runs 12 profiles × full decision tree (138 checks)
- **Unit total: 685 tests, all passing**
- **Standard eval: 233 cases, all passing**
- **Red-team eval: 36/36 cases (100%) across 7 attack categories**
- **Real-world eval: 2005 safety + 221 guardrail + 13 context + 11 generation + 17 action = 2267 cases**
  - Safety: 2005/2005 (100%), Guardrail: 216/220 (98.2% — remaining gaps are 3 malware-vs-scam priority cases + 1 community-rule overlay reclassification), Context: 13/13 (100%), Generation: 9/11 (82%), Action: 17/17 (100%)
  - Safety dataset v2: 14 languages (en, vi, zh, ja, ko, es, fr, de, ar, hi, th, id, pt, tl) + 13 mixed-lingual code-switch combos
  - Guardrail corpus: 221 YAML cases from `sample_messages.yaml` with 17-category taxonomy (0-16), severity rubric (0-5), jurisdiction codes, community overlays, locale tags
  - Real model: Qwen3.5-0.8B Q4_K_M via llama-server (Metal), ~130 tok/s, 30ms TTFT
- **Go server offload: 7 tests, all passing**
- **Per-device eval: 12 profiles × 150 tasks = 1800 task runs (6 unique generative models)**
  - 15 task categories: summarization, translation, structured output, tool use,
    multi-turn, code generation, reasoning, instruction following, safety,
    context retrieval, action, core, generation, WASM, bindings
  - Multilingual: EN, VI, JA, KO, ZH, ES, AR, DE, HI, FR + mixed-language
  - Judgment: Pass (≥75%), Marginal (50-74%), Fail (<50%)
  - 6 unique generative models: ternary-bonsai-1.7b-mlx-2bit, ternary-bonsai-1.7b-q2_0,
    ternary-bonsai-4b-mlx-2bit, ternary-bonsai-4b-q2_0, ternary-bonsai-8b-mlx-2bit,
    ternary-bonsai-8b-q2_0
  - Also tracks per-profile: vision (mobileclip-s2-int8), safety encoder (INT4),
    ASR (whisper-tiny/base), and video (mobileclip-s2-int8, same as vision) model assignments

## Model Registry (11 packs)

| Pack ID | Type | Min Tier | Size | Quant | Backend | Platform | SHA-256 |
|---------|------|----------|------|-------|---------|----------|---------|
| ternary-bonsai-1.7b-mlx-2bit | generative | Low | 472 MB | 2bit-MLX | MLX | ios/macos | ✅ real |
| ternary-bonsai-1.7b-q2_0 | generative | Low | 442 MB | Q2_0 | llama.cpp Vulkan | android/windows | ✅ real |
| ternary-bonsai-4b-q2_0 | generative | Medium | 1,075 MB | Q2_0 | llama.cpp Vulkan | android | ✅ real |
| ternary-bonsai-4b-mlx-2bit | generative | Medium | 1,132 MB | 2bit-MLX | MLX | ios/macos | ✅ real |
| ternary-bonsai-8b-mlx-2bit | generative | High | 2,304 MB | 2bit-MLX | MLX | ios/macos | ✅ real |
| ternary-bonsai-8b-q2_0 | generative | High | 2,182 MB | Q2_0 | llama.cpp Vulkan | windows | ✅ real |
| kchat-encoder-int8 | encoder | High | 266 MB | INT8 | ONNX | all | ✅ real |
| kchat-encoder-int4 | encoder | Low | 143 MB | INT4 | ONNX | all | ✅ real |
| mobileclip-s2-int8 | vision | Low | 97 MB | INT8 | ONNX | all | ✅ real |
| whisper-tiny | asr | Low | 33 MB | ONNX (FP32) | ONNX | all | ✅ real |
| whisper-base | asr | Medium | 82 MB | ONNX (FP32) | ONNX | all | ✅ real |

11/11 packs have real SHA-256 hashes.

> **Note**: Whisper ONNX files are FP32 (not INT8-quantized). Base models: `nb-whisper-tiny` and `nb-whisper-base` from NbAiLab
> (Norwegian fine-tunes of OpenAI Whisper). Both retain full multilingual capability
> (en, vi, zh, ja, ko, es, fr, de, ar, hi, th).

### Model selection by tier and platform

- **Low tier**:
  - Generative: iOS/macOS: `ternary-bonsai-1.7b-mlx-2bit` via **MLX** (472MB) / Android/Windows: `ternary-bonsai-1.7b-q2_0` via **llama.cpp Vulkan** (442MB)
  - Vision: `mobileclip-s2-int8` (37MB runtime, INT8, 17 categories, image + video)
  - Encoder: `kchat-encoder-int4` (143MB, INT4) — safety + embedding + reranking
  - ASR: `whisper-tiny` (33MB, ONNX FP32, nb-whisper-tiny, multilingual)
  - Video: `mobileclip-s2-int8` (same model as vision)
  - **Total footprint**: ~685MB all loaded / ~472MB effective (Apple Silicon) / ~442MB effective (GGUF)
  - Context cap: 1,024 tokens (iOS) / 2,048 (Android) / 2,048 (desktop)
- **Medium tier**:
  - Generative: iOS/macOS: `ternary-bonsai-4b-mlx-2bit` via **MLX** (1,132MB) / Android: `ternary-bonsai-4b-q2_0` via **llama.cpp Vulkan** (1,075MB)
  - Vision: `mobileclip-s2-int8` (37MB runtime, INT8, image + video)
  - Encoder: `kchat-encoder-int4` (143MB, INT4) — safety + embedding + reranking
  - ASR: `whisper-base` (82MB, ONNX FP32, nb-whisper-base, multilingual)
  - Video: `mobileclip-s2-int8` (same model as vision)
  - **Total footprint**: ~1,394MB all loaded / ~1,132MB effective (Apple Silicon) / ~1,075MB effective (Android)
  - Context cap: 2,048 tokens (iOS) / 4,096 (Android) / 4,096 (desktop)
- **High tier**:
  - Generative: iOS/macOS: `ternary-bonsai-8b-mlx-2bit` via **MLX** (2,304MB) / Android: `ternary-bonsai-8b-q2_0` via **llama.cpp Vulkan** (2,182MB) / Windows: `ternary-bonsai-8b-q2_0` via **llama.cpp Vulkan** (2,182MB)
  - Vision: `mobileclip-s2-int8` (37MB runtime, INT8, image + video)
  - Encoder: `kchat-encoder-int4` (143MB, INT4) — safety + embedding + reranking
  - ASR: `whisper-base` (82MB, ONNX FP32, nb-whisper-base, multilingual)
  - Video: `mobileclip-s2-int8` (same model as vision)
  - **Total footprint**: ~2,566MB all loaded (Apple Silicon) / ~2,444MB all loaded (Android/Windows) / ~2,304MB effective (Apple Silicon) / ~2,182MB effective (Android/Windows)
  - Context cap: 4,096 tokens (iOS) / 8,192 (Android) / 16,384 (desktop)

All generative models support `tool_use`. The "deterministic-first" principle is preserved —
safety works on ALL devices without a generative model. Vision and ASR run on ALL tiers.
Low-tier devices use INT4 quantized encoder to fit within memory budget.
All tiers use INT4 encoder for consistency and efficiency.
Vision, ASR, and safety encoder models are lazy-loaded on-demand (not co-resident with generative model).
During generation, only the generative model is resident. All tiers use kchat-encoder-int4 (143MB) for efficiency.
KV cache: Q8_0 quantized for llama.cpp (Android/Windows/Intel Mac), FP16 for MLX (Apple Silicon).
Context caps: iOS 1K/2K/4K (FP16 KV cache), Android 2K/4K/8K (Q8 KV cache), desktop 2K/4K/16K.
No budget increases needed — all profiles fit with 163+ MB headroom on mobile.
The unified kchat-encoder replaces 4 separate model packs (e5-small, safety-int8,
safety-int4, cross-encoder-miniLM) with 2 multi-task packs (INT8 + INT4).
The unified mobileclip-s2-int8 replaces 3 separate vision packs (image-int8,
image-fp32, video-int8) with 1 multi-task pack handling both image and video.

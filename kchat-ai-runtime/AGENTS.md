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

# Run the slides AI skill eval (mock mode — 880 test cases, 12 skills × 210 templates)
cargo run -p kchat-task-suite -- --slides

# Run the slides AI skill eval (real model mode — requires llama-server)
cargo run -p kchat-task-suite -- --slides --realworld

# Run the image search eval (always real — requires PEXELS_API_KEY, PIXABAY_API_KEY,
# UNSPLASH_ACCESS_KEY, and/or SHUTTERSTOCK_API_TOKEN env vars)
cargo run -p kchat-task-suite -- --slides-images

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

# Build/run with full pipeline (skill-pack + ONNX encoder + vision)
# Requires libonnxruntime.dylib on system path or KCHAT_ONNX_LIB env var
cargo run -p kchat-task-suite --features full-pipeline -- --realworld

# Build/run with ONNX encoder only (no vision)
cargo run -p kchat-task-suite --features onnx-runtime -- --realworld

# Build/run with skill-pack overlays only (no ONNX)
cargo run -p kchat-task-suite --features skill-pack -- --realworld
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
- **Generation**: 10 prompts with real Ternary-Bonsai-1.7B inference via llama-server,
  measuring TTFT, decode rate (tok/s), and JSON schema compliance
- **Action**: 16 cases (tool plans, artifact ops, commit tokens, formula injection)

To run generation tests, either:
1. Start llama-server manually: `llama-server -m manifest/packs/Ternary-Bonsai-1.7B-Q2_0.gguf --port 18888 -ngl 99`
2. Or let the harness auto-start it (requires llama-server on PATH and model in manifest/packs/)
3. Or set `LLAMA_SERVER_URL` to point to an existing server

### Per-Device Eval Setup

The `--perdevice` mode tests each of the 12 device profiles against its assigned
real model (2 generative models, unified across all tiers), running 150 tasks across 15 categories:

- **15 task categories**: summarization, translation, structured output, tool use,
  multi-turn conversation, code generation, reasoning, instruction following,
  safety, context retrieval, action, core, generation, WASM, bindings
- **2 generative models**: bonsai-1.7b-mlx-1bit (Apple Silicon, 269MB),
  bonsai-1.7b-q1_0 (Android/Windows/Intel, 248MB)
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

The Swift MLX server uses the PrismML mlx-swift fork (branch: `v0.31.6_prism`)
which adds 1-bit quantization Metal kernels for Bonsai models. The metallib
(Metal shader library) must be built separately with CMake and copied next to
the binary:
```bash
# Build metallib (one-time, after swift package resolve)
cd .build/checkouts/mlx-swift/Source/Cmlx/mlx
mkdir -p build && cd build
cmake .. -DMLX_METAL_JIT=ON -DMACOS_VERSION=14.0
make -j10 mlx-metallib
cp mlx/backend/metal/kernels/mlx.metallib ../../../../../../.build/release/
```

The Swift server supports 1-bit (`bonsai-1.7b-mlx-1bit`) and 2-bit
(`Ternary-Bonsai-1.7B-mlx-2bit`) MLX models. The 1-bit model runs at ~22 tok/s
and the 2-bit at ~11 tok/s on M5.

If the Swift binary is not available, the harness automatically falls back to
`swift/kchat-mlx-server/kchat_mlx_server.py` (requires `pip install mlx-lm`).
Note: the Python fallback uses official Apple MLX which does NOT support 1-bit
quantization — only the Swift server with the PrismML fork can run 1-bit models.

## Architecture

The workspace is organized into 10 crates + 1 Go sidecar following the 4-plane architecture:

- **kchat-core**: Capability probe (real OS APIs via sysctl/procfs/Win32),
  device tier selection, scheduler, signed manifest manager, telemetry,
  model manager (CDN download, LRU cache, mmap), resource governor,
  model registry (registry.toml). Foundation for all other crates.
- **kchat-encoder**: Unified multi-task encoder — supports both ONNX (XLM-RoBERTa-base)
  and GGUF (mmBERT-small) backends. Shared across safety classification (17 classes
  with GGUF, 10 with ONNX), text embedding (384-dim GGUF, 768-dim ONNX), and
  cross-encoder reranking. GGUF backend uses llama-server --embedding HTTP API.
  Feature-gated behind `onnx-runtime` and `gguf-runtime`; includes mock
  implementations for testing without either backend.
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
  Works on ALL devices including low-tier and WASM (deterministic only).
- **kchat-context**: Private context plane — SQLCipher encrypted store, FTS5
  BM25 retrieval, per-scope XChaCha20-Poly1305 encryption, provenance bundles,
  dense embeddings (kchat-encoder 384-dim GGUF or 768-dim ONNX + fallback), cross-encoder reranker (kchat-encoder).
- **kchat-generation**: Grammar-constrained generative plane — prompt templates,
  JSON Schema/regex/Lark grammar validation (real Lark parser), backend
  adapters (llama.cpp via llama-cpp-2 with Metal/Vulkan/Cuda), model lifecycle
  with idle unload, token streaming with safety cancellation, LoRA hot-swap
  (50 adapters: 5 tasks × 10 languages), swarm inference (multi-peer consensus).
- **kchat-action**: Action plane — artifact AST (typed operations, no arbitrary
  code), ToolPlan validation against signed manifests, RBAC authorization
  broker, commit tokens, audit log. Slide operations (InsertSlide, UpdateSlide,
  ReorderSlide, SetSlideTemplate) validate template_id against the
  SlidesTemplateRegistry and enforce image-slot query-only constraints.
- **kchat-image**: Image search plane — unified image search across Pexels,
  Pixabay, Unsplash, and Shutterstock. ImageSearchProvider trait with 4 adapter
  implementations, ImageSearchRegistry with fallback/dedup/safety-filter/cache,
  MockProvider for offline testing. 21 unit tests.
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
8. **Smart templates**: 210 declarative slide templates across 12 families (Title, Agenda,
   Bullet, Quote, Comparison, Timeline, Image, Chart, Diagram, Team, Media, Section)
   with JSON Schema slot validation, image orientation hints, and chart support
9. **Image search**: Unified image search across 4 providers (Pexels, Pixabay, Unsplash,
   Shutterstock) with fallback, dedup, safety filtering, and in-memory cache

## Slides AI Skills (12 skills, 210 templates)

The `SkillSurface::Slides` surface adds 12 slides-specific skills to the existing
33 document skills (45 total):

| Skill ID | Label | Mode | Grammar | Tier | Description |
|----------|-------|------|---------|------|-------------|
| slides_generate_deck | Generate Deck | MultiStep | JsonSchema | High | Full deck from brief |
| slides_generate_slide | Generate Slide | PromptInput | JsonSchema | Medium | Single slide for template |
| slides_suggest_template | Suggest Template | OneClick | JsonSchema | Low | Pick best template |
| slides_suggest_outline | Suggest Outline | PromptInput | JsonSchema | Medium | Deck structure + template picks |
| slides_rewrite_slide | Rewrite Slide | PromptInput | JsonSchema | Medium | Rewrite for clarity/impact |
| slides_improve_slide | Improve Slide | OneClick | JsonSchema | Low | Improve conciseness |
| slides_add_image | Add Image | PromptInput | JsonSchema | Low | Derive image search query |
| slides_summarize_deck | Summarize Deck | OneClick | FreeText | Medium | Bullet-point summary |
| slides_extract_speaker_notes | Speaker Notes | OneClick | FreeText | Medium | Per-slide speaker notes |
| slides_translate_deck | Translate Deck | FormInput | FreeText | Medium | Translate all slide text |
| slides_suggest_title | Suggest Title | OneClick | Regex | Low | Concise deck title |
| slides_key_takeaways | Key Takeaways | OneClick | FreeText | Low | 3-5 key takeaways |

### Smart Template Registry (210 templates)

Templates are defined in `crates/kchat-generation/src/slides_templates.rs` and organized
into 12 families:

| Family | Count | Examples |
|--------|-------|----------|
| Title | 12 | title, title_subtitle, cover_hero, title_event |
| Agenda | 16 | agenda, table_of_contents, roadmap_agenda, agenda_track |
| Bullet | 20 | bullet, numbered_list, checklist, callout_box |
| Quote | 10 | quote, pull_quote, testimonial_quote, quote_with_background |
| Comparison | 18 | comparison_two_col, pros_cons, versus, pricing_comparison |
| Timeline | 16 | timeline_horizontal, milestone, roadmap, gantt_summary |
| Image | 24 | image_full_bleed, image_grid_2x2, hero_image, photo_collage |
| Chart | 22 | bar_chart, line_chart, pie_chart, kpi_dashboard, funnel |
| Diagram | 24 | flowchart, pyramid, venn_diagram, swot, cycle_4 |
| Team | 14 | team_grid, org_chart, person_card, testimonial |
| Media | 12 | video_embed, icon_list, word_cloud, map, qr_code |
| Section | 22 | section_break, divider_quote, recap, thank_you, contact |

Each template declares:
- **Slots**: Typed placeholders (TitleText, BodyText, BulletList, ImageQuery, ChartSeries, etc.)
- **Layout hint**: Renderer directive (centered, split_right, grid_2x2, etc.)
- **Image orientation hint**: landscape/portrait/square (for image-bearing templates)
- **JSON Schema**: Auto-generated slot-fill schema for grammar-constrained generation
- **Icon**: Lucide icon name for UI display

### Image Search Providers (kchat-image crate)

| Provider | Endpoint | Auth | License | Rate Limit |
|----------|----------|------|---------|------------|
| Pexels | api.pexels.com/v1/search | Authorization header | Free, no attribution | 200/hr |
| Pixabay | pixabay.com/api/ | key query param | Free, attribution required | 100/hr |
| Unsplash | api.unsplash.com/search/photos | Client-ID header | Free, attribution required | 50/hr |
| Shutterstock | api.shutterstock.com/v2/images/search | Bearer header | Commercial | 100/hr |

The `ImageSearchRegistry` merges results across providers with:
- **Fallback**: Tries providers in priority order, falls back on failure
- **Deduplication**: By URL across providers
- **Safety filtering**: HTTPS-only, alt-text blocklist, attribution check, zero-dimension check
- **Re-ranking**: Orientation match first
- **In-memory cache**: 256 entries, 10-minute TTL, SHA-256 cache keys

## Test Counts

- kchat-core: 94 tests (capability probe, model manager, governor, registry)
- kchat-safety: 389 tests (deterministic pipeline, encoder, policy packs, vision module)
  - 927 tests with `--features skill-pack` (adds skillpack loader, overlay merge, verifier,
    policy interpreter, threshold policy, revocation, anti-misuse, canonical JSON,
    jurisdiction tests, community overlay tests, adversarial corpus tests)
- kchat-action: 47 tests (artifact AST, ToolPlan, commit tokens, slide ops, image search tools)
- kchat-context: 44 tests (FTS, embeddings, reranker, provenance, cache invalidation)
- kchat-image: 21 tests (cache, safety, mock provider, registry, dedup, orientation rerank)
- kchat-generation: 171 tests (llama.cpp backend, LoRA, swarm, Lark grammar, MLX,
  45 skills, 210 slide templates, prompt construction, grammar schemas)
- kchat-bindings: 12 tests (FFI facade, capability probing, tier selection)
- kchat-wasm: 10 tests (WASM safety classification)
- kchat-encoder: 5 tests (mock safety, embed, rerank, session)
- kchat-task-suite: 24 unit tests + 205 standard eval + 36 red-team cases
  + 880 slides mock eval cases (12 skills × 210 templates) + 80 image search eval cases
  - Standard eval: 160 synthetic (64 safety + 33 context + 46 generation + 11 action + 6 integration) + 209 device profile = 369 cases
  - Safety eval: 64 cases covering all 6 PII types, 8 scam families, URL risk, obfuscation resistance, multilingual, false positive resistance, latency percentiles, per-category F1
  - Context eval: 33 cases with multi-doc retrieval quality (MRR, recall@k, MAP, NDCG), cross-language, ACL enforcement, encryption integrity, scale performance
  - Generation eval: 46 cases with grammar edge cases, prompt injection resistance, token budget, backend selection, model lifecycle
  - Device profile suite: 12 profiles × 15 test categories + 9 standalone tests = 189 cases
  - Device simulator: `--simulate` flag runs 12 profiles × full decision tree (138 checks)
- **Unit total: 902 tests, all passing**
- **Standard eval: 369 cases, all passing**
- **Red-team eval: 36/36 cases (100%) across 7 attack categories**
- **Real-world eval: 2005 safety + 221 guardrail + 13 context + 11 generation + 17 action = 2267 cases**
  - Safety: 2005/2005 (100%), Guardrail: 220/220 (100%), Context: 13/13 (100%), Generation: 9/11 (82%), Action: 17/17 (100%)
  - Safety dataset v2: 14 languages (en, vi, zh, ja, ko, es, fr, de, ar, hi, th, id, pt, tl) + 13 mixed-lingual code-switch combos
  - Guardrail corpus: 221 YAML cases from `sample_messages.yaml` with 17-category taxonomy (0-16), severity rubric (0-5), jurisdiction codes, community overlays, locale tags
  - Guardrail eval supports tier-aware execution: Deterministic (default), WithEncoder (ONNX INT4), FullPipeline (encoder + MobileCLIP-S2 vision)
  - Guardrail eval reports per-category breakdown, per-path latency (det vs enc), and applies jurisdiction severity floors (skill-pack overlay)
  - Real model: Ternary-Bonsai-1.7B Q2_0 via llama-server (Metal), ~130 tok/s, 30ms TTFT
- **Go server offload: 7 tests, all passing**
- **Per-device eval: 12 profiles × 150 tasks = 1800 task runs (2 generative models, unified)**
  - 15 task categories: summarization, translation, structured output, tool use,
    multi-turn, code generation, reasoning, instruction following, safety,
    context retrieval, action, core, generation, WASM, bindings
  - Multilingual: EN, VI, JA, KO, ZH, ES, AR, DE, HI, FR + mixed-language
  - Judgment: Pass (≥75%), Marginal (50-74%), Fail (<50%)
  - 2 generative models: bonsai-1.7b-mlx-1bit (Apple Silicon, 269MB),
    bonsai-1.7b-q1_0 (Android/Windows/Intel, 248MB)
  - Also tracks per-profile: vision (mobileclip-s2-int8), safety encoder (INT4),
    ASR (whisper-tiny/base), and video (mobileclip-s2-int8, same as vision) model assignments

## Model Registry (13 packs)

| Pack ID | Type | Min Tier | Size | Quant | Backend | Platform | SHA-256 |
|---------|------|----------|------|-------|---------|----------|---------|
| bonsai-1.7b-mlx-1bit | generative | Low | 269 MB | 1bit-MLX | MLX | ios/macos (Apple Silicon) | placeholder |
| bonsai-1.7b-q1_0 | generative | Low | 248 MB | Q1_0 | llama.cpp Vulkan/CPU | android/windows/intel | placeholder |
| qwen35-2b-mlx-4bit | generative | Low | 1,060 MB | 4bit-MLX | MLX | ios/macos (Apple Silicon) | placeholder |
| kchat-encoder-int8 | encoder | High | 266 MB | INT8 | ONNX | all | ✅ real |
| kchat-encoder-int4 | encoder | Low | 143 MB | INT4 | ONNX | all | ✅ real |
| mmbert-safety-q4_k_m | encoder | Low | 145 MB | Q4_K_M | GGUF (llama.cpp) | all | ✅ trained |
| mmbert-safety-q5_k_m | encoder | Medium | ~170 MB | Q5_K_M | GGUF (llama.cpp) | all | ⏳ pending export |
| mobileclip-s2-int8 | vision | Low | 97 MB | INT8 | ONNX | all | ✅ real |
| whisper-tiny | asr | Low | 33 MB | ONNX (FP32) | ONNX | all | ✅ real |
| whisper-base | asr | Medium | 82 MB | ONNX (FP32) | ONNX | all | ✅ real |

11/13 packs have real SHA-256 hashes. 2 new mmBERT GGUF encoders are pending training.

### mmBERT-small GGUF Encoder (v2.0)

The new `mmbert-safety-q4_k_m` and `mmbert-safety-q5_k_m` packs replace the
legacy XLM-RoBERTa ONNX encoder with a smaller, faster, more multilingual model:

- **Base**: mmBERT-small (140M params, 384-dim, 22 layers, 256K vocab, 1800+ languages)
- **Taxonomy**: 17 categories (kchat.guardrail.taxonomy.v1) vs legacy 10
- **Embedding dim**: 384 (vs legacy 768)
- **Backend**: GGUF via llama-server `--embedding` endpoint
- **Task heads**: Loaded separately from `classifier_weights.safetensors`
- **Size**: ~90MB (Q4_K_M) vs 143MB (INT4) / 266MB (INT8)
- **Feature flag**: `gguf-runtime` in kchat-encoder crate
- **Training**: `/Users/Ken/workspaces/mmbert-safety/` workspace

> **Note**: Whisper ONNX files are FP32 (not INT8-quantized). Base models: `nb-whisper-tiny` and `nb-whisper-base` from NbAiLab
> (Norwegian fine-tunes of OpenAI Whisper). Both retain full multilingual capability
> (en, vi, zh, ja, ko, es, fr, de, ar, hi, th).

### Model selection by tier and platform

- **Low tier**:
  - Generative: iOS/macOS: `bonsai-1.7b-mlx-1bit` via **MLX** (269MB) / Android/Windows: `bonsai-1.7b-q1_0` via **llama.cpp Vulkan** (248MB)
  - Vision: `mobileclip-s2-int8` (37MB runtime, INT8, 17 categories, image + video)
  - Encoder: `kchat-encoder-int4` (143MB, INT4) — safety + embedding + reranking
  - ASR: `whisper-tiny` (33MB, ONNX FP32, nb-whisper-tiny, multilingual)
  - Video: `mobileclip-s2-int8` (same model as vision)
  - **Total footprint**: ~482MB all loaded (Apple Silicon) / ~461MB all loaded (GGUF)
  - Context cap: 1,024 tokens (iOS) / 2,048 (Android) / 2,048 (desktop)
- **Medium tier**:
  - Generative: iOS/macOS: `bonsai-1.7b-mlx-1bit` via **MLX** (269MB) / Android: `bonsai-1.7b-q1_0` via **llama.cpp Vulkan** (248MB)
  - Vision: `mobileclip-s2-int8` (37MB runtime, INT8, image + video)
  - Encoder: `kchat-encoder-int4` (143MB, INT4) — safety + embedding + reranking
  - ASR: `whisper-base` (82MB, ONNX FP32, nb-whisper-base, multilingual)
  - Video: `mobileclip-s2-int8` (same model as vision)
  - **Total footprint**: ~531MB all loaded (Apple Silicon) / ~510MB all loaded (GGUF)
  - Context cap: 2,048 tokens (iOS) / 4,096 (Android) / 4,096 (desktop)
- **High tier**:
  - Generative: iOS/macOS: `bonsai-1.7b-mlx-1bit` via **MLX** (269MB) / Android/Windows: `bonsai-1.7b-q1_0` via **llama.cpp Vulkan** (248MB)
  - Vision: `mobileclip-s2-int8` (37MB runtime, INT8, image + video)
  - Encoder: `kchat-encoder-int4` (143MB, INT4) — safety + embedding + reranking
  - ASR: `whisper-base` (82MB, ONNX FP32, nb-whisper-base, multilingual)
  - Video: `mobileclip-s2-int8` (same model as vision)
  - **Total footprint**: ~531MB all loaded (Apple Silicon) / ~510MB all loaded (Android/Windows)
  - Context cap: 4,096 tokens (iOS) / 8,192 (Android) / 16,384 (desktop)

All tiers use the same 1.7B base generative model (bonsai-1.7b-mlx-1bit or bonsai-1.7b-q1_0)
with task-specialized LoRA adapters. Tier differences are handled via context window size,
output budget, and performance targets — not different model sizes.
All generative models support `tool_use`. The "deterministic-first" principle is preserved —
safety works on ALL devices without a generative model. Vision and ASR run on ALL tiers.
All tiers use INT4 encoder for consistency and efficiency.
Vision, ASR, and safety encoder models are lazy-loaded on-demand (not co-resident with generative model).
During generation, only the generative model is resident. All tiers use kchat-encoder-int4 (143MB) for efficiency.
KV cache: Q8_0 quantized for llama.cpp (Android/Windows/Intel Mac), FP16 for MLX (Apple Silicon).
Context caps: iOS 1K/2K/4K (FP16 KV cache), Android 2K/4K/8K (Q8 KV cache), desktop 2K/4K/16K.
No budget increases needed — all profiles fit with 268+ MB headroom on mobile.
The unified kchat-encoder replaces 4 separate model packs (e5-small, safety-int8,
safety-int4, cross-encoder-miniLM) with 2 multi-task packs (INT8 + INT4).
The unified mobileclip-s2-int8 replaces 3 separate vision packs (image-int8,
image-fp32, video-int8) with 1 multi-task pack handling both image and video.

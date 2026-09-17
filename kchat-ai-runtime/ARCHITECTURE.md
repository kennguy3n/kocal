# Architecture

> Detailed technical architecture for kchat-ai-runtime — the deterministic-first,
> privacy-first, tier-aware on-device AI runtime for KChat.

## Design Philosophy

### 1. Deterministic-First

Safety classification operates on ALL devices without requiring a generative
model. The deterministic safety plane uses:

- **NFKC Unicode normalization** — canonicalizes text before analysis
- **Regex-based PII detection** — emails, phone numbers, SSN, credit cards, addresses
- **Scam/URL detectors** — known scam patterns, suspicious URL heuristics
- **Signed policy packs (Ed25519)** — community/jurisdiction-specific rules
- **mmBERT safety encoder (GGUF)** — mmBERT-small Q4_K_M (~145MB) via llama.cpp, 17-category taxonomy
- **Skill-pack system** — 17-category taxonomy, 0-5 severity rubric, 38 communities,
  62 jurisdictions, threshold policy (0.45/0.62/0.78/0.85)

This means a 4GB low-tier phone or a WASM browser tab can classify messages
for safety without any LLM inference. The generative plane is only needed for
summarization, translation, and tool use.

### 2. Tier-Aware

Every device is classified at runtime into one of three tiers. Tier is not a
static label — it is re-evaluated before each job based on current memory
pressure and thermal state.

```
┌─────────────────────────────────────────────────────────────┐
│                    TierSelection::select()                   │
│                                                              │
│  1. Probe hardware (sysctl/procfs/Win32)                     │
│  2. Compute safe_allocatable_memory                          │
│  3. Memory gate:                                             │
│     Mobile:  ≥6000MB→High  ≥3500MB→Medium  <3500MB→Low      │
│     Desktop: ≥20000MB→High ≥10000MB→Medium <10000MB→Low     │
│  4. Thermal downgrade:                                       │
│     Serious → drop one tier                                  │
│     Critical → force Low                                     │
│  5. Enterprise policy may cap but not elevate                │
└─────────────────────────────────────────────────────────────┘
```

### 3. Privacy-First

- **Per-scope encryption**: XChaCha20-Poly1305 for context store
- **Append-only evidence**: No mutation of stored evidence, only tombstones
- **No raw content in telemetry**: Only hashes and metadata
- **Local inference**: All model inference happens on-device
- **SQLCipher**: FTS5 full-text search over encrypted store

### 4. Signed Distribution

- **Ed25519-signed manifests**: Model packs verified against pinned public keys
- **Policy pack signing**: Safety rules cannot be tampered with
- **Kill switch**: Compromised packs can be revoked via signed manifest updates
- **SHA-256 verification**: Every downloaded pack is hash-verified

### 5. Grammar-Constrained Generation

Model output is always constrained to a valid format:

- **JSON Schema validation** — structured output enforced at decode time
- **Regex constraints** — pattern-matched output for simple fields
- **Lark grammar** — full context-free grammar support for complex formats
- **No arbitrary code** — Artifact operations are typed AST nodes, not eval'd code

## Four-Plane Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        kchat-ai-runtime                              │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │  Safety      │  │  Context    │  │  Generation  │  │  Action   │ │
│  │  Plane       │  │  Plane      │  │  Plane       │  │  Plane    │ │
│  │              │  │             │  │              │  │           │ │
│  │ kchat-safety │  │ kchat-      │  │ kchat-       │  │ kchat-    │ │
│  │              │  │ context     │  │ generation   │  │ action    │ │
│  │              │  │             │  │              │  │           │ │
│  │ • NFKC norm  │  │ • SQLCipher │  │ • llama.cpp  │  │ • Artifact│ │
│  │ • PII detect │  │ • FTS5 BM25 │  │ • MLX backend│  │   AST    │ │
│  │ • Scam/URL   │  │ • Encrypted │  │ • LoRA swap  │  │ • ToolPlan│ │
│  │ • Policy pkgs│  │   store     │  │ • Swarm inf  │  │ • RBAC    │ │
│  │ • GGUF encdr │  │ • Embeddings│  │ • Grammar    │  │ • Commit │ │
│  │ • Vision     │  │ • Reranker  │  │   validation │  │   tokens │ │
│  │ • Skill-pack │  │ • Provenance│  │ • Streaming  │  │ • Audit   │ │
│  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘ │
│         │                 │                │                │       │
│         └────────────────┬┴────────────────┴────────────────┘       │
│                          │                                           │
│              ┌───────────┴───────────┐                               │
│              │    kchat-core          │                               │
│              │    (Foundation)        │                               │
│              │                        │                               │
│              │ • Capability probe     │                               │
│              │ • Tier selection       │                               │
│              │ • Model registry       │                               │
│              │ • Model manager (CDN)  │                               │
│              │ • Scheduler            │                               │
│              │ • Resource governor    │                               │
│              │ • Signed manifests     │                               │
│              │ • Telemetry            │                               │
│              └────────────────────────┘                               │
│                                                                      │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │ kchat-bindings  │  │ kchat-wasm   │  │ kchat-server-offload   │  │
│  │ (FFI surface)   │  │ (Web safety) │  │ (Go sidecar)           │  │
│  │                  │  │              │  │                        │  │
│  │ • UniFFI Swift   │  │ • ~2.1MB     │  │ • Gin HTTP API         │  │
│  │ • UniFFI Kotlin  │  │ • Safety cls │  │ • Auth + rate limit    │  │
│  │ • N-API Node.js  │  │ • PII detect │  │ • Safety classification│  │
│  │ • KChatAiRuntime │  │ • Normalize  │  │ • Cloud inference      │  │
│  │   facade         │  │ • No model   │  │                        │  │
│  └─────────────────┘  └──────────────┘  └────────────────────────┘  │
│                                                                      │
│  Shared services: kchat-encoder (mmBERT embeddings/safety head),       │
│  kchat-asr (Whisper ONNX + whisper.cpp GGML), kchat-image (stock       │
│  photo search), kchat-runtime (Orchestrator/SkillRouter/Memory),       │
│  kchat-generative-sidecar (Rust, stub)                                 │
└──────────────────────────────────────────────────────────────────────┘
```

## Crate Details

### kchat-core (Foundation)

The foundation crate that all other crates depend on.

**Capability Probe** (`capability.rs`):
- Detects platform (iOS, macOS, Android, Windows, Linux)
- Reads physical memory via sysctl (macOS/iOS), procfs (Linux), Win32 (Windows)
- Detects CPU architecture (aarch64, x86_64), core count, performance cores
- ISA features (NEON, FP16, AVX2, AVX512)
- Battery level, charger state, thermal state
- GPU backend (Metal, Vulkan, CUDA, None)
- NPU provider (Apple Neural Engine, NNAPI, Windows NPU, None)
- Free storage space
- Safe allocatable memory (60-83% of physical, platform-dependent)

**Tier Selection** (`tier.rs`):
- `TierSelection::select(&DeviceCapabilities) -> Result<DeviceTier>`
- Memory-based initial gate with platform-specific thresholds
- Thermal downgrade (Serious → drop one tier, Critical → force Low)
- Enterprise policy cap (can lower but not elevate)
- Per-tier budgets: context cap (uniform across platforms: Low 4K / Medium 8K / High 16K tokens),
  output token range, peak memory, TTFT target, decode rate target, max perf cores,
  idle unload timeout
- Memory optimization: Q8_0 KV cache for llama.cpp, FP16 for MLX, lazy-loaded encoder/vision/ASR
  (only generative model resident during inference), all tiers use mmbert-safety-q4_k_m encoder,
  all profiles fit within peak memory budgets with 163+ MB mobile headroom

**Model Registry** (`registry.rs`):
- 13 model packs (7 generative, 1 encoder, 1 vision, 4 ASR)
- Generative lineup: 4 Bonsai 1.7B packs (1-bit/2-bit × MLX/GGUF, Fast/Quality modes)
  plus Qwen3 Q4_K_M alternates (0.6B Low, 1.7B Medium, 4B-Instruct-2507 High)
- Encoder and vision models are unified across all tiers; ASR differentiates by
  tier (whisper-tiny Low, whisper-base Medium/High) with both ONNX and
  whisper.cpp GGML pack variants
- `RegistryEntry` with pack_id, version, pack_type, download_url, sha256,
  size_bytes, min_tier, task_capabilities, languages, quantization
- 9/13 packs have real SHA-256 hashes (4 pending: the Bonsai generative packs)
- `find_for_task(task, tier)` — filter by capability and tier eligibility
- `find_for_language(lang, tier)` — filter by language and tier
- `default_registry()` — hardcoded built-in catalog
- Supports loading from TOML files for custom registries

**Model Manager** (`model_manager.rs`):
- CDN download with SHA-256 verification
- LRU cache with tier-based limits
- Memory-mapped model loading (mmap)
- Pack ID validation (rejects path traversal, length limits)
- Cache scanning with safe filename validation
- Eviction when cache exceeds tier limit

**Scheduler** (`scheduler.rs`):
- Job queue with priority and preemption
- Memory budget enforcement (rejects oversized jobs)
- Kill switch support (revoked packs rejected)
- Background task blocking on mobile
- Idle unload with configurable timeout

**Resource Governor** (`governor.rs`):
- Per-tier resource limits
- Thermal throttling (blocks generation on Critical)
- Battery enforcement (blocks on low battery unless charging)
- Background app state restrictions on mobile
- Timeout enforcement

**Signed Manifests** (`manifest.rs`):
- Ed25519 signature verification
- Kill switch for revoked packs
- SHA-256 chunk verification
- Null digest rejection
- Manifest round-trips through TOML

**Telemetry** (`telemetry.rs`):
- Ring buffer with overflow dropping
- No raw content in events (only hashes/metadata)
- Record and drain pattern

### kchat-safety (Safety Plane)

**Deterministic Pipeline**:
- NFKC Unicode normalization
- PII detection: email, phone, SSN, credit card, address patterns
- Scam detection: known patterns and heuristics
- URL risk scoring: suspicious domains, redirect chains
- Obfuscation detection: homoglyphs, zero-width chars, encoding attacks
- Injection detection: prompt injection, jailbreak attempts
- Multilingual support: 14 languages + 13 mixed-lingual code-switch combos

**GGUF Safety Encoder** (mmBERT-small):
- Q4_K_M quantized (~145MB, all tiers)
- llama.cpp embedding backend (`gguf-runtime` feature); subprocess on
  desktop, in-process engine on mobile
- Escalation from deterministic → encoder → SLM

**Skill-Pack System** (feature: `skill-pack`):
- 17-category taxonomy (0-16): safe, harassment, hate, extremism, drugs,
  adult, violence, self-harm, PII, scam, URL risk, injection, obfuscation,
  multilingual, vision, code-switch
- 0-5 severity rubric with disposition thresholds (0.45/0.62/0.78/0.85)
- 38 community overlays
- 62 jurisdiction overlays
- Policy interpreter with SLM rate limiting
- Canonical JSON serialization
- Revocation lists
- Anti-misuse validation
- Embedded data via `include_str!`

**Vision Module** (feature: `onnx-runtime-vision`):
- MobileCLIP-S2 unified image + video encoder (ONNX)
- INT8 (102MB pack, all tiers) — single model for both image and video
- 512-dim embeddings, 17 categories
- Video frame aggregation with temporal smoothing
- Vision bridge connecting to safety pipeline

**Test Coverage**: 393 tests (more with `--features skill-pack`)

### kchat-context (Context Plane)

**Encrypted Store**:
- SQLCipher with per-scope XChaCha20-Poly1305 encryption
- FTS5 BM25 full-text search over encrypted content
- Append-only evidence (no mutation, only tombstones)
- Scope-based access control (user/role authorization)

**Retrieval Pipeline**:
- Dense embeddings: kchat-encoder (mmBERT-small) GGUF Q4_K_M (~145MB, 384-dim) on all tiers
- Fallback overlap scoring when embeddings unavailable
- Cross-encoder reranker: kchat-encoder shared session (all tiers)
- Recency boost for recent results
- Recall@10 and MRR metrics

**Provenance Bundles**:
- Cryptographic evidence chains
- Per-scope isolation
- Forget/delete with tombstone preservation

**Test Coverage**: 54 tests

### kchat-generation (Generation Plane)

**Backend Adapters**:
- `BackendType::select(platform, tier, cpu_arch)` — arch-aware selection
  - Apple Silicon (aarch64, iOS/macOS): MLX
  - Intel Macs (x86_64, macOS): llama.cpp CPU
  - Android/Windows: llama.cpp Vulkan
  - Other: llama.cpp CPU
- llama.cpp via llama-cpp-2 crate (Metal, Vulkan, CUDA backends)
- MLX via kchat-mlx-server (Swift binary or Python mlx-lm fallback)
- Mock backend for testing (no model required)

**Model Lifecycle**:
- Idle unload with tier-aware timeout (45s mobile, 300s desktop)
- Memory-mapped model loading
- GPU layer offload (-1 = all layers)
- Thread count based on tier (Low: 2, Medium: 3, High: 4 perf cores)

**Grammar Constraints**:
- JSON Schema validation (real parser, not regex)
- Regex pattern constraints
- Lark grammar support (context-free grammars)
- Validation at decode time, not post-hoc

**LoRA Hot-Swap**:
- 75 family-based adapters: 5 task-families × 15 language slots, plus
  270 task-based adapters (18 tasks × 15 slots, legacy layout)
- Runtime adapter swapping without model reload
- Task families: summarize, translate, generate, tool_use, code
- Language slots: en, es, fr, de, ja, ko, zh, ar, hi, pt, ru, it, th, vi, id
- Shared between the 1-bit and 2-bit Bonsai packs (same architecture)

**Swarm Inference**:
- Multi-peer consensus for high-stakes generation
- Configurable peer count and agreement threshold
- Fallback to single-peer when swarm unavailable

**Streaming**:
- Token-by-token streaming with safety cancellation
- Safety plane can abort generation mid-stream
- Backpressure-aware

**Slides Engine**:
- 12 slides-specific skills layered on 39 document/chat skills (51 total)
- 210 declarative slide templates across 12 families (Title, Agenda, Bullet,
  Quote, Comparison, Timeline, Image, Chart, Diagram, Team, Media, Section)
- JSON Schema slot-fill grammars auto-generated per template
- Image orientation hints and chart-series slots

**Test Coverage**: 197 tests (192 lib + 5 llamacpp integration)

### kchat-action (Action Plane)

**Artifact AST**:
- Typed operations: `replace_range`, `insert_slide`, `delete_range`, etc.
- No arbitrary code execution — only validated AST nodes
- Type-safe operation dispatch

**ToolPlan Validation**:
- Plans validated against signed manifests
- Tool ID verification
- Parameter schema validation
- RBAC authorization before execution

**Authorization**:
- Three-step authorization: before search, during search, before prompt
- RBAC broker with role-based permissions
- Commit tokens for atomic operations
- Audit log for all actions

**Test Coverage**: 47 tests

### kchat-encoder (Unified Encoder)

- mmBERT-small multi-task encoder (140M params, 384-dim, 22 layers, 256K vocab)
- GGUF Q4_K_M backend via llama-server `--embedding` endpoint
- Shared across safety classification (17-class head), text embedding, and
  cross-encoder reranking
- Task heads loaded separately from `classifier_weights.safetensors`
- Feature-gated behind `gguf-runtime`; mock implementations for testing
- Pack: `mmbert-safety-q4_k_m` (~145MB)

**Test Coverage**: 5 tests

### kchat-asr (Speech-to-Text)

- Whisper ASR with two backends:
  - ONNX Runtime (desktop, feature `onnx-runtime`): whisper-tiny (33MB),
    whisper-base (82MB), FP32 NbAiLab exports
  - whisper.cpp GGML (mobile, feature `whispercpp`): whisper-tiny-ggml (78MB),
    whisper-base-ggml (148MB), F32 GGML format
- Multilingual transcription (whisper multilingual vocab)
- Chunking and voice-activity detection for long audio

**Test Coverage**: 67 tests

### kchat-image (Image Search Plane)

- Unified image search across 4 providers: Pexels, Pixabay, Unsplash, Shutterstock
- `ImageSearchProvider` trait with per-provider adapters
- `ImageSearchRegistry`: fallback ordering, URL dedup, safety filtering,
  in-memory cache (256 entries, 10-minute TTL, SHA-256 keys)
- `MockProvider` for offline testing

**Test Coverage**: 25 tests (24 unit + 1 doctest)

### kchat-runtime (Orchestration)

- `Orchestrator` — wires safety → context → generation → action pipeline
- `SkillRouter` — dispatches requests to the 51-skill registry
- `ConversationMemory` — tier-scaled sliding window + RAG compression
- Criterion benches for orchestrator overhead

**Test Coverage**: 10 tests (orchestrator pipeline)

### kchat-bindings (FFI Surface)

**UniFFI (Mobile)**:
- Swift bindings for iOS/macOS
- Kotlin bindings for Android
- `KChatAiRuntime` facade with high-level API
- Real capability probing at startup
- Tier-based configuration selection

**N-API (Desktop)**:
- Node.js bindings for Windows/Linux/macOS
- Same `KChatAiRuntime` facade
- Async operations via tokio

**Test Coverage**: 12 tests

### kchat-wasm (WebAssembly)

- Exposes deterministic safety plane only (no generative)
- ~2.1MB compiled WASM module
- Safety classification, PII detection, NFKC normalization
- No server-side model required
- Works in any WebAssembly-compatible browser

**Test Coverage**: 10 tests

### kchat-server-offload (Go Sidecar)

- Gin-based HTTP API
- Handles AI inference when on-device runtime can't (low tier, thermal, battery)
- Auth with API key + rate limiting
- Safety classification (same taxonomy as on-device)
- Cloud model inference proxy

**Test Coverage**: 7 tests

### kchat-task-suite (Eval Harness)

**Standard Eval** (369 cases):
- 160 synthetic tests (64 safety + 33 context + 46 generation + 11 action + 6 integration)
- 209 device profile tests (12 profiles × ~16 categories + standalone)
- 36 red-team attack cases (7 categories) via `--redteam`

**Real-World Eval** (3,644 cases):
- 3,447 safety cases (3,142 core JSON + 220 guardrail YAML + held-out corpora;
  14 languages + mixed-lingual code-switch)
- 89 context cases (multilingual, ACL tests)
- 59 generation cases (real Bonsai-1.7B Q1_0 inference via llama-server)
- 49 action cases (tool plans, artifact ops, commit tokens)

**Additional Suites**:
- Skills eval: 307 cases across 39 document/chat skills (`--skills`)
- Slides mock eval: 2,171 cases (12 skills × 210 templates, `--slides`)
- Image search eval: 170 cases, real provider APIs (`--slides-images`)
- Per-device eval: 12 profiles × 251 tasks × 4 generative models (`--perdevice`)
- Red-team encoder mode: `--redteam-encoder` (known-gap escalation)
- Baseline regression: `--compare-old`, `--write-baseline` for smoke eval

**Per-Device Eval** (3,012 task runs):
- 12 device profiles × 251 tasks × generative model assigned per profile
- 15 task categories per profile
- Quality scoring (0.0-1.0 per task, ≥0.7 pass threshold)
- Judgment: Pass (≥75%), Marginal (50-74%), Fail (<50%)

**Device Simulator** (`--simulate`):
- 12 profiles × full decision tree
- Capability probe → tier selection → model selection → backend selection →
  memory budget → model fit → registry lookup → non-generative model availability

## Data Flow

### Message Classification Flow

```
Input Message
    │
    ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  NFKC           │────▶│  Deterministic   │────▶│  Safety Encoder │
│  Normalization  │     │  Detectors       │     │  (mmBERT GGUF)  │
│                 │     │  • PII           │     │                 │
│                 │     │  • Scam          │     │  Escalation     │
│                 │     │  • URL risk      │     │  from det → enc │
│                 │     │  • Obfuscation   │     │                 │
│                 │     │  • Injection     │     │                 │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │  Skill-Pack     │
                                                │  Policy         │
                                                │  Interpreter    │
                                                │                 │
                                                │  • Taxonomy     │
                                                │  • Severity     │
                                                │  • Thresholds   │
                                                │  • Community    │
                                                │  • Jurisdiction │
                                                └────────┬────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │  Safety Action  │
                                                │  (allow/warn/   │
                                                │   redact/block) │
                                                └─────────────────┘
```

### Generation Flow

```
User Request
    │
    ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Safety Check   │────▶│  Context         │────▶│  Prompt Template│
│  (pre-generation)│    │  Retrieval       │     │  Construction   │
│  Three-step auth│     │  • FTS5 BM25     │     │  • Tier-aware   │
│  1. Before search│    │  • Embeddings    │     │  • Grammar      │
│  2. During search│    │  • Reranker      │     │    constraints  │
│  3. Before prompt│    │  • Scope filter  │     │  • LoRA adapter │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │  Backend        │
                                                │  Selection      │
                                                │  • MLX (Apple)  │
                                                │  • Vulkan (Win) │
                                                │  • CPU (Intel)  │
                                                └────────┬────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │  Generation     │
                                                │  + Streaming    │
                                                │  + Safety cancel│
                                                │  + Grammar valid│
                                                └────────┬────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │  Safety Check   │
                                                │  (post-generation│
                                                │   output scan)  │
                                                └─────────────────┘
```

## Build Configuration

### Release Profile

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
strip = true
```

### Feature Flags

| Crate | Feature | Description |
|-------|---------|-------------|
| kchat-safety | `skill-pack` | Overlay-aware policy system (community/jurisdiction overlays) |
| kchat-safety | `encoder` / `gguf-encoder` | mmBERT GGUF safety encoder escalation |
| kchat-safety | `onnx-runtime` / `onnx-runtime-vision` | ONNX classifier / MobileCLIP-S2 vision encoder |
| kchat-encoder | `gguf-runtime` | GGUF backend via llama-server `--embedding` |
| kchat-encoder | `onnx-runtime` / `domain-adapters` | ONNX backend / domain adapter heads |
| kchat-context | `lexical` (default) / `embeddings` | BM25-only vs dense retrieval |
| kchat-context | `connectors` / `reranker` / `gguf-embeddings` | Source connectors, cross-encoder rerank, GGUF embedding |
| kchat-generation | `llamacpp` / `llamacpp-metal` / `llamacpp-vulkan` / `llamacpp-cuda` | llama.cpp backend variants |
| kchat-generation | `mlx` / `litert` | MLX server backend / LiteRT (Android NPU) |
| kchat-asr | `onnx-runtime` / `whispercpp` | ONNX Whisper / whisper.cpp GGML backend |
| kchat-bindings | `mobile` / `mobile-full` / `desktop` | UniFFI / UniFFI+encoder / N-API |
| kchat-bindings | `gguf-encoder` / `llamacpp` | Encoder attach / generation backend in FFI |
| kchat-task-suite | `full-pipeline` / `gguf-runtime` / `skill-pack` | Eval pipeline tiers |
| kchat-task-suite | `smoke` / `smoke-metal` | Real-model smoke eval (in-process llama.cpp) |

### WASM Build

```bash
cargo build -p kchat-wasm --target wasm32-unknown-unknown --release
# Output: target/wasm32-unknown-unknown/release/kchat_wasm.wasm (~2.1MB)
```

## Dependencies

### Core Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| serde | 1.0 | Serialization (derive) |
| serde_json | 1.0 | JSON handling |
| ed25519-dalek | 2.1 | Ed25519 signature verification |
| sha2 | 0.10 | SHA-256 hashing |
| hkdf | 0.12 | Key derivation |
| tokio | 1.40 | Async runtime |
| chrono | 0.4 | Time handling |
| uuid | 1.10 | ID generation |
| parking_lot | 0.12 | Synchronization primitives |
| tracing | 0.1 | Structured logging |

### Platform-Specific

| Platform | Backend | Key Dependencies |
|----------|---------|-----------------|
| iOS/macOS (Apple Silicon) | MLX | kchat-mlx-server (Swift), mlx-lm (Python fallback) |
| macOS (Intel) | llama.cpp CPU | llama-cpp-2 |
| Android | llama.cpp Vulkan | llama-cpp-2 |
| Windows | llama.cpp Vulkan/CUDA | llama-cpp-2 |
| Web | WASM | wasm-bindgen, js-sys |
| Server offload | Go | Gin, ed25519 |

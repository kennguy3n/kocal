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
- **ONNX Runtime safety encoder** — INT8 (High tier, 270MB) or INT4 (Low/Medium, 90MB) quantized
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
│  │ • ONNX encdr │  │ • Embeddings│  │ • Grammar    │  │ • Commit │ │
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
- Per-tier budgets: context cap, output token range, peak memory, TTFT target,
  decode rate target, max perf cores, idle unload timeout

**Model Registry** (`registry.rs`):
- 11 model packs (6 generative, 2 encoder, 1 vision, 2 ASR)
- `RegistryEntry` with pack_id, version, pack_type, download_url, sha256,
  size_bytes, min_tier, task_capabilities, languages, quantization
- 10/11 packs have real SHA-256 hashes (mobileclip-s2-int8 is the remaining placeholder)
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

**ONNX Runtime Safety Encoder**:
- INT8 quantized (High tier, 270MB)
- INT4 quantized (Low/Medium tier, 90MB)
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
- INT8 (70MB, all tiers) — single model for both image and video
- 512-dim embeddings, 17 categories
- Video frame aggregation with temporal smoothing
- Vision bridge connecting to safety pipeline

**Test Coverage**: 389 tests (927 with `--features skill-pack`)

### kchat-context (Context Plane)

**Encrypted Store**:
- SQLCipher with per-scope XChaCha20-Poly1305 encryption
- FTS5 BM25 full-text search over encrypted content
- Append-only evidence (no mutation, only tombstones)
- Scope-based access control (user/role authorization)

**Retrieval Pipeline**:
- Dense embeddings: kchat-encoder (XLM-RoBERTa-base) ONNX INT4 (90MB, 768-dim) on Low/Medium, INT8 (270MB) on High
- Fallback overlap scoring when embeddings unavailable
- Cross-encoder reranker: kchat-encoder shared session (all tiers)
- Recency boost for recent results
- Recall@10 and MRR metrics

**Provenance Bundles**:
- Cryptographic evidence chains
- Per-scope isolation
- Forget/delete with tombstone preservation

**Test Coverage**: 44 tests

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
- 30 adapters: 5 tasks × 6 languages
- Runtime adapter swapping without model reload
- Task: summarize, translate, generate, tool_use, code
- Languages: en, vi, zh, ja, ko, es

**Swarm Inference**:
- Multi-peer consensus for high-stakes generation
- Configurable peer count and agreement threshold
- Fallback to single-peer when swarm unavailable

**Streaming**:
- Token-by-token streaming with safety cancellation
- Safety plane can abort generation mid-stream
- Backpressure-aware

**Test Coverage**: 84 tests

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

**Test Coverage**: 37 tests

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

**Standard Eval** (233 cases):
- 44 synthetic tests (safety, context, generation, action)
- 161 device profile tests (12 profiles × 15 categories + 9 standalone)
- 36 red-team attack cases (7 categories)

**Real-World Eval** (2,267 cases):
- 2,005 safety cases (14 languages + 13 code-switch combos)
- 221 guardrail cases (17-category taxonomy, YAML from sample_messages.yaml)
- 13 context cases (multilingual, ACL tests)
- 11 generation cases (real Qwen3.5-0.8B inference via llama-server)
- 17 action cases (tool plans, artifact ops, commit tokens)

**Per-Device Eval** (1,800 task runs):
- 12 device profiles × 150 tasks × 7 unique generative models
- 15 task categories per profile
- Quality scoring (0.0-1.0 per task, ≥0.7 pass threshold)
- Judgment: Pass (≥75%), Marginal (50-74%), Fail (<50%)

**Device Simulator**:
- 12 profiles × 138 checks each
- Full decision tree: capability probe → tier selection → model selection →
  backend selection → memory budget → model fit → registry lookup →
  non-generative model availability

## Data Flow

### Message Classification Flow

```
Input Message
    │
    ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  NFKC           │────▶│  Deterministic   │────▶│  Safety Encoder │
│  Normalization  │     │  Detectors       │     │  (ONNX INT4/8)  │
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
| kchat-safety | `skill-pack` | Overlay-aware policy system (927 tests) |
| kchat-safety | `onnx-runtime-vision` | MobileCLIP-S2 vision encoder (ONNX) |
| kchat-generation | `llamacpp` | llama.cpp backend (Metal/Vulkan/CUDA) |
| kchat-bindings | `mobile` | UniFFI bindings for Swift/Kotlin |
| kchat-bindings | `desktop` | N-API bindings for Node.js |

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

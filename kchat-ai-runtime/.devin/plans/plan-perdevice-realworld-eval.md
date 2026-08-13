# Plan: Per-Device Real-World Eval with Real Models

## Goal

Create a comprehensive task dataset and eval runner that tests each of the 12 device
profiles against its assigned real model, measures performance + quality, and produces
a judgment report on whether each model is "good enough" for its tier.

## Architecture: Single Distributed App

```
┌──────────────────────────────────────────────────────────┐
│  kchat app (single binary, App Store)                     │
│                                                          │
│  ┌─────────────┐    ┌─────────────────────┐              │
│  │  Rust core   │◄──►│  Swift MLX layer    │              │
│  │  (kchat-*)   │    │  (mlx-swift)        │              │
│  │              │    │                     │              │
│  │  - scheduler │    │  - MLX inference    │              │
│  │  - safety    │    │  - model loading    │              │
│  │  - context   │    │  - Metal compute    │              │
│  │  - registry  │    │  - memory mapping   │              │
│  └─────────────┘    └─────────────────────┘              │
│         ▲                     ▲                          │
│         └────── UniFFI ───────┘                          │
│              (already in project)                        │
└──────────────────────────────────────────────────────────┘
```

### Model Inference Paths

| Model Format | Backend | Runtime | Works on iOS? |
|-------------|---------|---------|---------------|
| GGUF (Q2_0, Q4_K_M, Q8_0) | `llama-cpp-2` Rust crate | In-process | Yes (Metal) |
| MLX safetensors (2bit, 4bit) | `mlx-swift` via Swift CLI | Subprocess (eval) / UniFFI (prod) | Yes |

- **GGUF**: `llama-cpp-2` is already integrated in `kchat-generation` crate. In-process,
  no subprocess, no external dependencies. Works on iOS, macOS, Android, Windows.
- **MLX**: `mlx-swift` is Apple's official MLX Swift package. For eval, we build a small
  Swift CLI (`kchat-mlx-server`) that uses the same `mlx-swift` code as production would.
  For production, the iOS/macOS app embeds the Swift MLX layer via UniFFI.

### No Python anywhere

- No `mlx_lm.server`, no Python runtime
- Swift CLI uses `mlx-swift` directly — same code path as the shipping app
- GGUF uses `llama-cpp-2` Rust crate — already in the project

## Model-to-Device Mapping (6 unique models, 12 profiles)

| Model | Format | Size | Device Profiles | Eval Backend |
|-------|--------|------|-----------------|--------------|
| ternary-bonsai-1.7b-mlx-2bit | MLX | 472MB | iPhone SE, MacBook Air, Intel NUC | kchat-mlx-server |
| ternary-bonsai-1.7b-q2_0 | GGUF | 442MB | Galaxy A14, Windows Surface, Windows Legacy | llama-cpp-2 |
| ternary-bonsai-4b-mlx-2bit | MLX | 1.08GB | iPhone 14, Pixel 7a | kchat-mlx-server |
| ternary-bonsai-4b-q2_0 | GGUF Q2_0 | 1.0GB | Pixel 8 Pro | llama-cpp-2 |
| macaw-4bit-mlx | MLX | 1.5GB | iPhone 15 Pro, MacBook Pro M3 Max | kchat-mlx-server |
| ternary-bonsai-8b-q2_0 | GGUF Q2_0 | 2.1GB | Windows RTX 4090 | llama-cpp-2 |

**Unique models to test: 6** (each tested once, results applied to all profiles using it)

## Phase 1: Download Models (~5GB)

Download all 5 new models to `manifest/packs/`:

```bash
# GGUF models (via huggingface-cli)
huggingface-cli download prism-ml/Ternary-Bonsai-1.7B-gguf \
  Ternary-Bonsai-1.7B-Q2_0.gguf --local-dir manifest/packs/
huggingface-cli download prism-ml/Ternary-Bonsai-4B-gguf \
  Ternary-Bonsai-4B-Q2_0.gguf --local-dir manifest/packs/
huggingface-cli download prism-ml/Ternary-Bonsai-8B-gguf \
  Ternary-Bonsai-8B-Q2_0.gguf --local-dir manifest/packs/

# MLX models (via huggingface-cli — safetensors, not Python)
huggingface-cli download prism-ml/Ternary-Bonsai-1.7B-mlx-2bit \
  --local-dir manifest/packs/ternary-bonsai-1.7b-mlx-2bit/
huggingface-cli download badtheorylabs/Macaw-4bit-MLX \
  --local-dir manifest/packs/macaw-4bit-mlx/

# Already have: Ternary-Bonsai-1.7B-Q2_0.gguf in manifest/packs/
```

## Phase 2: Build Swift MLX CLI (kchat-mlx-server)

**Location**: `swift/kchat-mlx-server/`

A minimal Swift CLI that:
1. Loads an MLX model from a directory path using `mlx-swift`
2. Exposes an HTTP API compatible with llama-server:
   - `GET /health` → returns 200
   - `POST /completion` → accepts `{prompt, n_predict, temperature, top_p, seed}`, returns
     `{content, tokens_predicted, tokens_evaluated, prompt_ms, predicted_ms}`
3. Measures timing (prompt processing, decode) and returns in response

### Swift Package Structure

```
swift/kchat-mlx-server/
├── Package.swift
├── Sources/
│   └── kchat-mlx-server/
│       ├── main.swift         # CLI entry, arg parsing
│       ├── ModelServer.swift  # HTTP server + MLX inference
│       └── Types.swift        # Request/response types
```

### Package.swift

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "kchat-mlx-server",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-extras", from: "0.2.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.10.0"),
    ],
    targets: [
        .executableTarget(
            name: "kchat-mlx-server",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-extras"),
            ]
        )
    ]
)
```

### Key Design Decisions

- **HTTP server**: Use Swift's built-in `NWListener` (Network framework) — no external
  HTTP dependency, keeps the package minimal
- **API compatibility**: Same JSON shapes as llama-server's `/completion` endpoint so the
  eval runner can use both backends interchangeably
- **Model loading**: Use `MLXLMCommon.ModelFactory` to load from directory (handles
  config.json, tokenizer, safetensors)
- **Timing**: Measure prompt processing time and decode time separately, return in response

## Phase 3: Create Comprehensive Task Dataset

**File**: `eval/kchat-task-suite/datasets/multitask/multitask_dataset_v1.json`

**8 task categories, 40 tasks total**:

### 3.1 Summarization (5 tasks)
- News article summary (200→50 words)
- Meeting transcript summary (500→100 words)
- Product review summary (150→30 words)
- Technical doc summary (300→60 words)
- Multi-paragraph email summary (400→80 words)

### 3.2 Translation (5 tasks)
- EN→VI: Common phrases
- EN→ZH: Business email
- EN→JA: Greeting + introduction
- EN→KO: Question about schedule
- VI→EN: Casual message

### 3.3 JSON/Structured Output (5 tasks)
- User profile JSON (schema: name, age, email, active)
- Calendar event JSON (schema: title, date, time, attendees[])
- Search query JSON (schema: query, filters{}, limit)
- Tool call JSON (schema: tool_name, arguments{})
- Config JSON (schema: nested objects with arrays)

### 3.4 Tool Use (5 tasks)
- "What's the weather?" → should call weather tool
- "Send a message to Ada" → should call send_message tool
- "Search for Q3 reports" → should call search tool
- "Create a reminder for 3pm" → should call create_reminder tool
- "Read the battery status" → should call battery_status tool

### 3.5 Multi-turn Conversation (5 tasks)
- 3-turn: greeting → question → follow-up
- 4-turn: task assignment → clarification → confirmation → execution
- 3-turn: error report → diagnosis → fix suggestion
- 5-turn: planning discussion with constraints
- 3-turn: code review with feedback

### 3.6 Code Generation (5 tasks)
- Simple function (add two numbers)
- FizzBuzz implementation
- JSON parser helper
- SQL query builder
- React component skeleton

### 3.7 Reasoning (5 tasks)
- Math word problem
- Logical deduction (if A then B)
- Cause and effect analysis
- Comparison of two options
- Step-by-step troubleshooting

### 3.8 Instruction Following (5 tasks)
- "Write exactly 3 sentences about X"
- "List 5 items, numbered, no extra text"
- "Respond with only 'YES' or 'NO'"
- "Format as a table with 3 columns"
- "Write a haiku about autumn"

### Dataset Structure
```json
{
  "name": "kchat-multitask-v1",
  "version": "1.0.0",
  "tasks": [
    {
      "id": "summarize_001",
      "category": "summarization",
      "prompt": "...",
      "max_tokens": 128,
      "expected_min_tokens": 20,
      "grammar": null,
      "quality_check": {
        "type": "min_length",
        "min_chars": 50
      },
      "description": "News article summary"
    }
  ],
  "performance_targets": {
    "low_tier":    { "ttft_p95_ms": 2500, "decode_p50_tps": 8,  "max_memory_gb": 0.75 },
    "medium_tier": { "ttft_p95_ms": 1500, "decode_p50_tps": 15, "max_memory_gb": 1.5  },
    "high_tier":   { "ttft_p95_ms": 1000, "decode_p50_tps": 25, "max_memory_gb": 3.0  }
  }
}
```

## Phase 4: Build Per-Device Eval Runner

**File**: `eval/kchat-task-suite/src/eval_perdevice.rs`

**New CLI flag**: `--perdevice` — runs per-device real-world eval

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  Per-Device Eval Runner                             │
│                                                     │
│  1. Group 12 profiles by unique model (6 groups)    │
│  2. For each unique model:                          │
│     a. GGUF → load via llama-cpp-2 in-process       │
│     b. MLX  → spawn kchat-mlx-server subprocess     │
│  3. Run all 40 tasks against the model              │
│  4. Measure: TTFT, decode rate, quality             │
│  5. Apply results to all profiles using this model  │
│  6. Stop server / unload model                      │
│  7. Produce per-device judgment                     │
│                                                     │
│  Output: Comprehensive judgment report              │
└─────────────────────────────────────────────────────┘
```

### Key Structs

```rust
enum InferenceBackend {
    LlamaCppInProcess,  // llama-cpp-2, loaded directly in eval binary
    MlxServer(PathBuf), // path to kchat-mlx-server binary
}

struct ModelConfig {
    pack_id: String,
    model_path: PathBuf,
    backend: InferenceBackend,
    context_size: usize,
    server_port: Option<u16>,  // only for MLX
}

struct TaskResult {
    task_id: String,
    category: String,
    success: bool,
    quality_pass: bool,
    ttft_ms: u64,
    decode_rate_tps: f64,
    output_tokens: u32,
    output_text: String,
    error: Option<String>,
}

struct DeviceJudgment {
    profile_name: String,
    tier: DeviceTier,
    model: String,
    tasks_total: usize,
    tasks_passed: usize,
    quality_passed: usize,
    ttft_p50_ms: u64,
    ttft_p95_ms: u64,
    decode_p50_tps: f64,
    decode_p95_tps: f64,
    perf_target_met: bool,
    quality_target_met: bool,
    judgment: Judgment,  // Pass, Marginal, Fail
    judgment_reason: String,
}
```

### GGUF: In-Process via llama-cpp-2

The eval binary links `kchat-generation` with the `llamacpp` feature. For each GGUF model:
1. Create `LlamaCppBackend::new()`
2. Load model with `BackendConfig::for_tier(...)`
3. Call `backend.generate(prompt, params)` for each task
4. Measure timing from the generation call
5. Unload model when done

**Advantages**:
- No subprocess management, no port conflicts
- Direct timing measurement (no HTTP overhead)
- Same code path as production on Android/Windows

### MLX: Subprocess via kchat-mlx-server

For each MLX model:
1. Spawn `kchat-mlx-server --model <path> --port <port>`
2. Wait for `/health` to return 200
3. Send completion requests via HTTP (same as existing llama-server approach)
4. Kill process when done

**Why subprocess for MLX in eval?**
- mlx-swift is Swift, not Rust — can't link directly into the Rust eval binary
- The subprocess uses the same mlx-swift code that ships in the iOS/macOS app
- In production, the app embeds the Swift MLX layer via UniFFI (no subprocess)

### Quality Checks

| Check Type | Logic |
|-----------|-------|
| `min_length` | Output has at least N characters |
| `max_length` | Output has at most N characters |
| `json_schema_valid` | Output is valid JSON matching schema |
| `contains_tool_call` | Output contains a tool call for the expected tool |
| `contains_keyword` | Output contains expected keyword(s) |
| `regex_match` | Output matches regex pattern |
| `exact_match` | Output exactly matches expected string |
| `coherent` | Output is coherent and on-topic (heuristic) |

### Judgment Criteria

| Criterion | Weight | Pass Threshold |
|-----------|--------|----------------|
| Task success rate | 40% | ≥ 80% of tasks produce valid output |
| Quality pass rate | 30% | ≥ 70% of tasks pass quality check |
| TTFT P95 vs target | 15% | P95 ≤ tier target (2500/1500/1000ms) |
| Decode P50 vs target | 15% | P50 ≥ tier target (8/15/25 tok/s) |

**Judgment**:
- **Pass**: Overall score ≥ 75%
- **Marginal**: Overall score 50-74%
- **Fail**: Overall score < 50%

### Output Report

```
╔══════════════════════════════════════════════════════════════════════════════╗
║  PER-DEVICE REAL-WORLD EVAL REPORT                                           ║
║  12 Profiles × 40 Tasks × Real Model Inference                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ [1/12] iPhone 15 Pro (8GB, A17 Pro) — High tier
│  Model: macaw-4bit-mlx (1.5GB, MLX)
│  Backend: kchat-mlx-server :18890
├──────────────────────────────────────────────────────────────────────────────┤
│  TASK RESULTS                                                                │
│    Summarization:        5/5 passed (100%)
│    Translation:          4/5 passed (80%)
│    Structured Output:    5/5 passed (100%)
│    Tool Use:             3/5 passed (60%)
│    Multi-turn:           4/5 passed (80%)
│    Code Generation:      3/5 passed (60%)
│    Reasoning:            4/5 passed (80%)
│    Instruction Follow:   5/5 passed (100%)
│  PERFORMANCE                                                                │
│    TTFT P50:    850ms    P95: 1200ms  (target: 1000ms) ✓
│    Decode P50:  32 tps   P95: 28 tps  (target: 25 tps)  ✓
│  JUDGMENT: PASS (87%)
│    Task success: 33/40 (82%) ✓
│    Quality pass: 31/40 (77%) ✓
│    Performance:  both targets met ✓
└──────────────────────────────────────────────────────────────────────────────┘

... (11 more profiles) ...

╔══════════════════════════════════════════════════════════════════════════════╗
║  JUDGMENT SUMMARY                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Device                       Tier   Model                    Judge   Score  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ iPhone 15 Pro                High   macaw-4bit-mlx           PASS    87%    ║
║ iPhone 14                    Med    ternary-bonsai-4b-mlx     PASS    82%    ║
║ iPhone SE 2022               Low    bonsai-1.7b-mlx-2bit     MARG    65%    ║
║ ...                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

## Phase 5: Implementation Steps

1. **Download models** (~5GB, ~10 min) — start in background
2. **Build Swift MLX CLI** — `swift/kchat-mlx-server/`
3. **Create dataset** — 40 tasks across 8 categories
4. **Build eval runner** — `eval_perdevice.rs`
5. **Wire up CLI** — `--perdevice` flag in `main.rs`
6. **Run the eval** — observe results, adjust if needed
7. **Update AGENTS.md**

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `swift/kchat-mlx-server/Package.swift` | Create | Swift package manifest |
| `swift/kchat-mlx-server/Sources/kchat-mlx-server/main.swift` | Create | CLI entry point |
| `swift/kchat-mlx-server/Sources/kchat-mlx-server/ModelServer.swift` | Create | HTTP server + MLX inference |
| `swift/kchat-mlx-server/Sources/kchat-mlx-server/Types.swift` | Create | Request/response types |
| `eval/kchat-task-suite/datasets/multitask/multitask_dataset_v1.json` | Create | 40-task dataset |
| `eval/kchat-task-suite/src/eval_perdevice.rs` | Create | Per-device eval runner |
| `eval/kchat-task-suite/src/main.rs` | Modify | Add --perdevice flag |
| `eval/kchat-task-suite/Cargo.toml` | Modify | Add kchat-generation dep with llamacpp |
| `manifest/packs/` | Download | 5 new model files |
| `AGENTS.md` | Update | Document new eval mode |

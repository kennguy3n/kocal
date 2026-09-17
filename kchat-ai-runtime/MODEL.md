# Model Registry & Device Profiles

> Complete technical reference for all model packs, device profiles, memory
> budgets, and model selection logic in kchat-ai-runtime.

## Model Registry (13 packs)

The model registry is the canonical catalog of all downloadable model packs.
It is defined in `crates/kchat-core/src/registry.rs` as
`ModelRegistry::default_registry()` and can also be loaded from TOML files.

Each `RegistryEntry` contains:
- `pack_id` — unique identifier (e.g. `bonsai-1.7b-mlx-1bit`)
- `version` — semantic version string
- `pack_type` — category: `generative`, `encoder`, `vision`, `asr`
- `download_url` — CDN or HuggingFace URL
- `sha256` — content hash for verification (Ed25519-signed manifest)
- `size_bytes` — exact download size
- `min_tier` — minimum device tier required (`Low`, `Medium`, `High`)
- `task_capabilities` — supported tasks (e.g. `summarize`, `translate`, `generate`, `tool_use`)
- `languages` — supported language codes (e.g. `en`, `vi`, `zh`, `ja`, `ko`, `es`, `ar`, `de`, `hi`, `fr`)
- `quantization` — quantization recipe (e.g. `Q2_0`, `Q4_K_M`, `Q8_0`, `2bit-MLX`, `INT8`, `INT4`)

### Generative Models (7 packs)

The generative plane centers on the Bonsai-1.7B family (ternary-quantized
Qwen3-1.7B) with two quality modes per platform backend — **Fast** (1-bit) and
**Quality** (2-bit) — plus a Qwen3 Q4_K_M lineup as standard-quant alternates
per tier. Tier differentiation within the Bonsai family is achieved through
LoRA adapters (75 family-based: 5 task-families × 15 language slots; plus 270
task-based legacy adapters) that are hot-swapped at runtime, not by loading
larger base models.

| Pack ID | Base Model | Params | Min Tier | Size | Quant | Backend | Platform | Capabilities | Languages |
|---------|-----------|--------|----------|------|-------|---------|----------|-------------|-----------|
| `bonsai-1.7b-mlx-1bit` | Qwen3-1.7B | 1.7B | Low | 269 MB | 1bit-MLX | MLX | iOS/macOS (aarch64) | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `bonsai-1.7b-q1_0` | Qwen3-1.7B | 1.7B | Low | 248 MB | Q1_0 | llama.cpp Vulkan/CPU | Android/Windows/Intel Mac | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `bonsai-1.7b-mlx-2bit` | Qwen3-1.7B | 1.7B | Low | 484 MB | 2bit-MLX | MLX | iOS/macOS (aarch64) | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `bonsai-1.7b-q2_0` | Qwen3-1.7B | 1.7B | Low | 442 MB | Q2_0 | llama.cpp Vulkan/CPU | Android/Windows/Intel Mac | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `qwen3-0.6b-q4_k_m` | Qwen3-0.6B | 0.6B | Low | 484 MB | Q4_K_M | llama.cpp | all | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `qwen3-1.7b-q4_k_m` | Qwen3-1.7B | 1.7B | Medium | 1.28 GB | Q4_K_M | llama.cpp | all | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `qwen3-4b-instruct-2507-q4_k_m` | Qwen3-4B-Instruct-2507 | 4B | High | 2.5 GB | Q4_K_M | llama.cpp | all | summarize, translate, generate, tool_use, reasoning | en, vi, zh, ja, ko, es, ar, de, hi, fr |

#### Ternary Bonsai Family

The Ternary Bonsai models are ternary-quantized variants of the Qwen3-1.7B
model. Ternary weights (-1, 0, +1) dramatically reduce model size while
maintaining reasonable quality. Each platform backend has a Fast (1-bit) and
Quality (2-bit) pack; the `ModelQuality` setting selects between them.

| Model | Base | Parameters | Bits/Weight | Download Size | Running Size (est.) |
|-------|------|-----------|-------------|--------------|-------------------|
| Bonsai-1.7B MLX (1-bit, Fast) | Qwen3-1.7B | 1.7B | 1.0 | 269 MB | ~400 MB |
| Bonsai-1.7B GGUF (Q1_0, Fast) | Qwen3-1.7B | 1.7B | 1.0 (Q1_0) | 248 MB | ~400 MB |
| Bonsai-1.7B MLX (2-bit, Quality) | Qwen3-1.7B | 1.7B | 2.0 | 484 MB | ~700 MB |
| Bonsai-1.7B GGUF (Q2_0, Quality) | Qwen3-1.7B | 1.7B | 2.0 (Q2_0) | 442 MB | ~650 MB |

#### Qwen3 Q4_K_M Lineup

Standard-quant (Q4_K_M) alternates with real SHA-256 digests, covering each
tier: 0.6B for Low, 1.7B for Medium (same architecture as Bonsai — all LoRA
adapters apply), and 4B-Instruct-2507 for High (adds `reasoning` capability).

#### Quantization Comparison

| Quant | Bits/Weight | Quality | Size Factor | Use Case |
|-------|------------|---------|------------|----------|
| 1bit-MLX | 1.0 | Good | 0.14× | Apple Silicon, MLX framework (Fast) |
| Q1_0 | 1.0 | Good | 0.13× | Non-Apple, llama.cpp (Fast) |
| 2bit-MLX | 2.0 | Better | 0.26× | Apple Silicon, MLX framework (Quality) |
| Q2_0 | 2.0 | Better | 0.23× | Non-Apple, llama.cpp (Quality) |
| Q4_K_M | ~4.8 | Best | 0.28× | Standard-quant alternates, all tiers |

### Encoder Model (1 pack)

Unified multi-task mmBERT GGUF model. Replaces separate embedding,
safety, and reranker packs with a single shared encoder session.

| Pack ID | Base Model | Min Tier | Size | Quant | Backend | Tasks | Languages | SHA-256 |
|---------|-----------|----------|------|-------|---------|-------|-----------|---------|
| `mmbert-safety-q4_k_m` | mmBERT-small | Low | 145 MB | Q4_K_M GGUF | llama.cpp | safety, embed, rerank | 27 languages | ✅ real |

### Vision Model (1 pack)

Unified MobileCLIP-S2 INT8 ONNX model pack. Pack includes visual encoder (~37 MB)
and text encoder (~64 MB). Runtime loads only the visual encoder for image
classification/embedding and video frame classification.

| Pack ID | Base Model | Min Tier | Pack Size | Runtime | Quant | Backend | Tasks | Embedding Dim |
|---------|-----------|----------|-----------|---------|-------|---------|-------|--------------|
| `mobileclip-s2-int8` | MobileCLIP-S2 | Low | 102 MB | 37 MB | INT8 | ONNX Runtime | image_classify, image_embed, video_classify | 512 |

### ASR Models (4 packs)

Two backend families:

**ONNX Runtime** (desktop path) — NbAiLab `nb-whisper-tiny`/`nb-whisper-base`
(Norwegian fine-tunes of OpenAI Whisper that retain full multilingual
capability). ONNX files are **FP32 (not INT8-quantized)**; packs include
encoder + decoder + decoder_with_past.

**whisper.cpp GGML** (mobile-safe in-process path, feature `whispercpp`) —
standard `ggml-tiny.bin`/`ggml-base.bin` F32 files from the whisper.cpp
project, for iOS/Android where ONNX/subprocess paths cannot ship.

| Pack ID | Base Model | Params | Min Tier | Size | Quant | Backend | Languages | SHA-256 |
|---------|-----------|--------|----------|------|-------|---------|-----------|---------|
| `whisper-tiny` | nb-whisper-tiny | 39M | Low | 33 MB | ONNX (FP32) | ONNX Runtime | en, vi, zh, ja, ko, es, fr, de, ar, hi, th | ✅ real |
| `whisper-base` | nb-whisper-base | 74M | Medium | 82 MB | ONNX (FP32) | ONNX Runtime | en, vi, zh, ja, ko, es, fr, de, ar, hi, th | ✅ real |
| `whisper-tiny-ggml` | whisper-tiny | 39M | Low | 78 MB | GGML (F32) | whisper.cpp | en, vi, zh, ja, ko, es, fr, de, ar, hi, th | ✅ real |
| `whisper-base-ggml` | whisper-base | 74M | Medium | 148 MB | GGML (F32) | whisper.cpp | en, vi, zh, ja, ko, es, fr, de, ar, hi, th | ✅ real |

## Device Tiers

### Tier Definitions

| Tier | Mobile RAM | Desktop RAM | Description |
|------|-----------|-----------|-------------|
| **Low** | 4–6 GB | 8 GB | 1.7B generative model (LoRA-adapted), Q4_K_M encoder, INT8 vision, whisper-tiny ASR |
| **Medium** | 6–8 GB | 16–24 GB | 1.7B generative model (LoRA-adapted), Q4_K_M encoder, INT8 vision, whisper-base ASR |
| **High** | 8 GB+ | 32 GB+ | 1.7B generative model (LoRA-adapted), Q4_K_M encoder, INT8 vision, video classification, whisper-base ASR |

### Tier Selection Thresholds

Tier is determined by `safe_allocatable_memory` (not total RAM). Safe allocatable
is typically 60-83% of physical memory, depending on platform.

| Platform | High (≥) | Medium (≥) | Low (<) |
|----------|---------|-----------|--------|
| iOS | 6,000 MB | 3,500 MB | 3,500 MB |
| Android | 6,000 MB | 3,500 MB | 3,500 MB |
| macOS | 20,000 MB | 10,000 MB | 10,000 MB |
| Windows | 20,000 MB | 10,000 MB | 10,000 MB |

**Thermal downgrade**:
- `ThermalState::Serious` → drop one tier (High→Medium, Medium→Low)
- `ThermalState::Critical` → force Low

**Enterprise policy**: May cap tier but cannot elevate it.

### Per-Tier Resource Budgets

| Resource | Low | Medium | High |
|----------|-----|--------|------|
| Context window (all platforms) | 4,096 tok | 8,192 tok | 16,384 tok |
| Output tokens | 64–192 | 256–512 | 512–1,024 |
| Peak memory (iOS) | 750 MB | 1,700 MB | 3,100 MB |
| Peak memory (Android) | 750 MB | 1,800 MB | 3,200 MB |
| Peak memory (macOS) | 2,000 MB | 4,000 MB | 8,000 MB |
| Peak memory (Windows) | 2,000 MB | 4,000 MB | 8,000 MB |
| TTFT P95 target | 2,500 ms | 1,500 ms | 1,000 ms |
| Decode P50 min (mobile) | 8 tok/s | 15 tok/s | 25 tok/s |
| Decode P50 min (desktop) | 10 tok/s | 20 tok/s | 35 tok/s |
| Max perf cores | 2 | 3 | 4 |
| Idle unload (mobile) | 45 s | 45 s | 45 s |
| Idle unload (desktop) | 300 s | 300 s | 300 s |

## Model Selection Logic

### Generative Model Selection

`select_model_for_tier_platform(tier, platform, cpu_arch)` in
`eval/kchat-task-suite/src/eval_device_profile.rs` selects the default
(Fast-quality) pack:

```
is_apple_silicon = (platform == "ios" OR platform == "macos") AND cpu_arch == "aarch64"

All tiers (Low/Medium/High) use the same 1.7B base family:
  Apple Silicon  → bonsai-1.7b-mlx-1bit  (269 MB, MLX, Fast)
  Other          → bonsai-1.7b-q1_0       (248 MB, GGUF, Fast)

Quality mode (user-selectable via ModelQuality::Quality):
  Apple Silicon  → bonsai-1.7b-mlx-2bit  (484 MB, MLX)
  Other          → bonsai-1.7b-q2_0       (442 MB, GGUF)

Tier differentiation is achieved via LoRA adapters (75 family-based: 5
task-families × 15 language slots; plus 270 task-based legacy adapters)
hot-swapped at runtime. The Qwen3 Q4_K_M lineup (0.6B/1.7B/4B) provides
standard-quant alternates per tier.
```

### Backend Selection

`BackendType::select(platform, tier, cpu_arch)` in
`crates/kchat-generation/src/backend.rs`:

```
iOS                    → MLX           (all tiers, aarch64 only)
macOS + aarch64        → MLX           (Apple Silicon)
macOS + x86_64         → llama.cpp CPU (Intel Macs, no MLX)
Android                → llama.cpp Vulkan
Windows                → llama.cpp Vulkan
Other                  → llama.cpp CPU
```

### Non-Generative Model Selection

| Model Type | Low Tier | Medium Tier | High Tier |
|-----------|----------|-------------|-----------|
| Vision (image+video) | mobileclip-s2-int8 (102 MB pack / 37 MB runtime) | mobileclip-s2-int8 | mobileclip-s2-int8 |
| Encoder | mmbert-safety-q4_k_m (145 MB) | mmbert-safety-q4_k_m (145 MB) | mmbert-safety-q4_k_m (145 MB) |
| ASR (ONNX, desktop) | whisper-tiny (33 MB) | whisper-base (82 MB) | whisper-base (82 MB) |
| ASR (GGML, mobile) | whisper-tiny-ggml (78 MB) | whisper-base-ggml (148 MB) | whisper-base-ggml (148 MB) |
| Video | mobileclip-s2-int8 (same as vision) | mobileclip-s2-int8 (same as vision) | mobileclip-s2-int8 (same as vision) |

> **Lazy-loading**: Vision, ASR, and safety encoder models are loaded on-demand for
> their specific task and unloaded after use. During generation, only the generative
> model is resident in memory. This reduces effective memory footprint by 215–264 MB.
> All tiers use mmbert-safety-q4_k_m (145 MB) for memory efficiency.

## Device Profiles (12 profiles)

All profiles are defined in `eval/kchat-task-suite/src/eval_device_profile.rs`
and mirrored in `eval/kchat-task-suite/src/eval_perdevice.rs`. The tables below
show the **Fast-quality** generative assignment; each profile also has a
Quality-mode variant (`bonsai-1.7b-mlx-2bit` on Apple Silicon,
`bonsai-1.7b-q2_0` elsewhere).

### Mobile: iOS

#### 1. iPhone 15 Pro (8GB, A17 Pro)

| Property | Value |
|----------|-------|
| Platform | iOS |
| CPU Arch | aarch64 |
| Physical RAM | 8,192 MB |
| Safe AI Budget | 6,800 MB (83%) |
| Storage | 128 GB |
| CPU Cores | 6 (2 performance) |
| GPU | Metal |
| NPU | Apple Neural Engine |
| ISA Features | NEON, FP16 |
| Battery | 85% (not charging) |
| Tier | **High** |
| Generative | `bonsai-1.7b-mlx-1bit` (269 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~533 MB** |

#### 2. iPhone 14 (6GB, A15)

| Property | Value |
|----------|-------|
| Platform | iOS |
| CPU Arch | aarch64 |
| Physical RAM | 6,144 MB |
| Safe AI Budget | 4,000 MB (65%) |
| Storage | 64 GB |
| CPU Cores | 6 (2 performance) |
| GPU | Metal |
| NPU | Apple Neural Engine |
| ISA Features | NEON |
| Battery | 70% (not charging) |
| Tier | **Medium** |
| Generative | `bonsai-1.7b-mlx-1bit` (269 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~533 MB** |

#### 3. iPhone SE 2022 (4GB, A15)

| Property | Value |
|----------|-------|
| Platform | iOS |
| CPU Arch | aarch64 |
| Physical RAM | 4,096 MB |
| Safe AI Budget | 2,500 MB (61%) |
| Storage | 32 GB |
| CPU Cores | 6 (2 performance) |
| GPU | Metal |
| NPU | Apple Neural Engine |
| ISA Features | NEON |
| Battery | 60% (not charging) |
| Tier | **Low** |
| Generative | `bonsai-1.7b-mlx-1bit` (269 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~484 MB** |

### Mobile: Android

#### 4. Pixel 8 Pro (12GB, Tensor G3)

| Property | Value |
|----------|-------|
| Platform | Android |
| CPU Arch | aarch64 |
| Physical RAM | 12,288 MB |
| Safe AI Budget | 7,000 MB (57%) |
| Storage | 128 GB |
| CPU Cores | 9 (1 performance) |
| GPU | Vulkan |
| NPU | NNAPI |
| ISA Features | NEON |
| Battery | 80% (not charging) |
| Tier | **High** |
| Generative | `bonsai-1.7b-q1_0` (248 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~512 MB** |

#### 5. Pixel 7a (8GB, Tensor G2)

| Property | Value |
|----------|-------|
| Platform | Android |
| CPU Arch | aarch64 |
| Physical RAM | 8,192 MB |
| Safe AI Budget | 3,800 MB (46%) |
| Storage | 64 GB |
| CPU Cores | 8 (2 performance) |
| GPU | Vulkan |
| NPU | NNAPI |
| ISA Features | NEON |
| Battery | 65% (not charging) |
| Tier | **Medium** |
| Generative | `bonsai-1.7b-q1_0` (248 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~512 MB** |

#### 6. Galaxy A14 (4GB, Helio G80)

| Property | Value |
|----------|-------|
| Platform | Android |
| CPU Arch | aarch64 |
| Physical RAM | 4,096 MB |
| Safe AI Budget | 1,800 MB (44%) |
| Storage | 16 GB |
| CPU Cores | 8 (2 performance) |
| GPU | Vulkan |
| NPU | None |
| ISA Features | NEON |
| Battery | 50% (not charging) |
| Network | Metered |
| Tier | **Low** |
| Generative | `bonsai-1.7b-q1_0` (248 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~463 MB** |

### Desktop: macOS

#### 7. MacBook Pro M3 Max (36GB)

| Property | Value |
|----------|-------|
| Platform | macOS |
| CPU Arch | aarch64 (Apple Silicon) |
| Physical RAM | 36,864 MB |
| Safe AI Budget | 22,000 MB (60%) |
| Storage | 512 GB |
| CPU Cores | 12 (4 performance) |
| GPU | Metal |
| NPU | Apple Neural Engine |
| ISA Features | NEON, FP16 |
| Power | Plugged in |
| Tier | **High** |
| Generative | `bonsai-1.7b-mlx-1bit` (269 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~533 MB** |

#### 8. MacBook Air M2 (8GB)

| Property | Value |
|----------|-------|
| Platform | macOS |
| CPU Arch | aarch64 (Apple Silicon) |
| Physical RAM | 8,192 MB |
| Safe AI Budget | 4,900 MB (60%) |
| Storage | 256 GB |
| CPU Cores | 8 (4 performance) |
| GPU | Metal |
| NPU | Apple Neural Engine |
| ISA Features | NEON |
| Power | Plugged in |
| Tier | **Low** (< 10,000 MB desktop threshold) |
| Generative | `bonsai-1.7b-mlx-1bit` (269 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~484 MB** |

#### 9. Intel NUC (8GB, i3)

| Property | Value |
|----------|-------|
| Platform | macOS |
| CPU Arch | x86_64 (Intel, no MLX) |
| Physical RAM | 8,192 MB |
| Safe AI Budget | 4,900 MB (60%) |
| Storage | 128 GB |
| CPU Cores | 4 (no performance cores) |
| GPU | None |
| NPU | None |
| ISA Features | AVX2 |
| Power | Plugged in |
| Tier | **Low** (< 10,000 MB desktop threshold) |
| Generative | `bonsai-1.7b-q1_0` (248 MB, GGUF — no MLX on x86_64) |
| Backend | llama.cpp CPU |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~463 MB** |

### Desktop: Windows

#### 10. Windows RTX 4090 (32GB)

| Property | Value |
|----------|-------|
| Platform | Windows |
| CPU Arch | x86_64 |
| Physical RAM | 32,768 MB |
| Safe AI Budget | 22,000 MB (67%) |
| Storage | 1,024 GB |
| CPU Cores | 16 (8 performance) |
| GPU | CUDA (RTX 4090) |
| NPU | Windows NPU |
| ISA Features | AVX2, AVX512 |
| Power | Plugged in |
| Tier | **High** |
| Generative | `bonsai-1.7b-q1_0` (248 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~512 MB** |

#### 11. Windows Surface 8 (16GB)

| Property | Value |
|----------|-------|
| Platform | Windows |
| CPU Arch | aarch64 (Windows on ARM) |
| Physical RAM | 16,384 MB |
| Safe AI Budget | 9,800 MB (60%) |
| Storage | 256 GB |
| CPU Cores | 8 (4 performance) |
| GPU | Vulkan |
| NPU | Windows NPU |
| ISA Features | NEON |
| Battery | 75% (not charging) |
| Tier | **Low** (< 10,000 MB desktop threshold) |
| Generative | `bonsai-1.7b-q1_0` (248 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~463 MB** |

#### 12. Windows Legacy (8GB, i5)

| Property | Value |
|----------|-------|
| Platform | Windows |
| CPU Arch | x86_64 |
| Physical RAM | 8,192 MB |
| Safe AI Budget | 4,900 MB (60%) |
| Storage | 64 GB |
| CPU Cores | 4 (no performance cores) |
| GPU | None |
| NPU | None |
| ISA Features | AVX2 |
| Power | Plugged in |
| Tier | **Low** (< 10,000 MB desktop threshold) |
| Generative | `bonsai-1.7b-q1_0` (248 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `mmbert-safety-q4_k_m` (145 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~463 MB** |

## Summary Tables

### Model Footprint by Tier

**All models loaded** (worst case, all co-resident):

| Tier | Generative | Vision (image+video) | Encoder | ASR | **Total** |
|------|-----------|----------------------|---------|-----|-----------|
| **Low (Apple Silicon)** | 269 MB | 37 MB | 145 MB | 33 MB | **484 MB** |
| **Low (GGUF)** | 248 MB | 37 MB | 145 MB | 33 MB | **463 MB** |
| **Medium (Apple Silicon)** | 269 MB | 37 MB | 145 MB | 82 MB | **533 MB** |
| **Medium (Android)** | 248 MB | 37 MB | 145 MB | 82 MB | **512 MB** |
| **High (Apple Silicon)** | 269 MB | 37 MB | 145 MB | 82 MB | **533 MB** |
| **High (Android/Windows)** | 248 MB | 37 MB | 145 MB | 82 MB | **512 MB** |

**Effective footprint** (generative model only; encoder, vision, and ASR all lazy-loaded on demand):

| Tier | Generative | **Effective Footprint** |
|------|-----------|------------------------|
| **Low (Apple Silicon)** | 269 MB | **269 MB** |
| **Low (GGUF)** | 248 MB | **248 MB** |
| **Medium (Apple Silicon)** | 269 MB | **269 MB** |
| **Medium (Android)** | 248 MB | **248 MB** |
| **High (Apple Silicon)** | 269 MB | **269 MB** |
| **High (Android/Windows)** | 248 MB | **248 MB** |

> **Lazy-loading**: Vision (mobileclip-s2-int8, 37 MB runtime), ASR (whisper-tiny/base, 33/82 MB),
> and safety encoder (mmbert-safety-q4_k_m, 145 MB) are loaded on-demand for their specific
> task and unloaded after use. During generation, only the generative model is resident.
> This reduces effective memory footprint by 215–264 MB.
> All tiers use mmbert-safety-q4_k_m (145 MB) for memory efficiency.

### KV Cache Estimates

KV cache type depends on the inference backend:
- **llama.cpp** (Android/Windows/Intel Mac): **Q8_0 quantized** (1 byte/element)
  — configured via `with_type_k(KvCacheType::Q8_0)` and `with_type_v(KvCacheType::Q8_0)`
  in `llamacpp.rs`. Halves cache memory vs FP16.
- **MLX** (iOS/macOS Apple Silicon): **FP16** (2 bytes/element)
  — MLX's Swift library does not expose KV cache quantization. Uses FP16 by default.

All Bonsai models use GQA with 8 KV heads and 128 head dimension.

| Model | Layers | KV Heads | Head Dim | Per-Token (Q8) | Per-Token (FP16) |
|-------|--------|----------|----------|----------------|------------------|
| Bonsai-1.7B | 28 | 8 | 128 | ~56 KB | ~115 KB |

Context caps are uniform across platforms: Low 4,096 / Medium 8,192 / High
16,384 tokens. Worst-case KV cache (High, 16K ctx): ~896 MB FP16 / ~448 MB Q8.
At the Low-tier 4K cap: ~230 MB FP16 / ~115 MB Q8.

### Memory Budget vs Effective Footprint + KV Cache

| Tier | Platform | Backend | Peak Budget | Effective (Gen) | KV Cache | **Total** | **Headroom** |
|------|----------|---------|-------------|-----------------|----------|-----------|--------------|
| Low | iOS | MLX (FP16) | 750 MB | 269 MB | 115 MB | **384 MB** | **366 MB** |
| Low | Android | llama.cpp (Q8) | 750 MB | 248 MB | 115 MB | **363 MB** | **387 MB** |
| Low | macOS | MLX (FP16) | 2,000 MB | 269 MB | 115 MB | **384 MB** | **1,616 MB** |
| Low | Windows | llama.cpp (Q8) | 2,000 MB | 248 MB | 115 MB | **363 MB** | **1,637 MB** |
| Medium | iOS | MLX (FP16) | 1,700 MB | 269 MB | 115 MB | **384 MB** | **1,316 MB** |
| Medium | Android | llama.cpp (Q8) | 1,800 MB | 248 MB | 115 MB | **363 MB** | **1,437 MB** |
| Medium | macOS | MLX (FP16) | 4,000 MB | 269 MB | 115 MB | **384 MB** | **3,616 MB** |
| Medium | Windows | llama.cpp (Q8) | 4,000 MB | 248 MB | 115 MB | **363 MB** | **3,637 MB** |
| High | iOS | MLX (FP16) | 3,100 MB | 269 MB | 115 MB | **384 MB** | **2,716 MB** |
| High | Android | llama.cpp (Q8) | 3,200 MB | 248 MB | 115 MB | **363 MB** | **2,837 MB** |
| High | macOS | MLX (FP16) | 8,000 MB | 269 MB | 115 MB | **384 MB** | **7,616 MB** |
| High | Windows | llama.cpp (Q8) | 8,000 MB | 248 MB | 115 MB | **363 MB** | **7,637 MB** |

> **All profiles fit within their peak memory budgets** with per-backend KV cache
> (Q8 for llama.cpp, FP16 for MLX), uniform context caps (4K/8K/16K by tier), and
> lazy-loaded encoder/vision/ASR models. No budget increases needed.
> The unified 1.7B model leaves substantial headroom on all profiles.

### Unique Generative Models per Profile (Fast mode)

| Model | Size | Profiles Using It |
|-------|------|-------------------|
| `bonsai-1.7b-mlx-1bit` | 269 MB | iPhone 15 Pro, iPhone 14, iPhone SE 2022, MacBook Pro M3 Max, MacBook Air M2 |
| `bonsai-1.7b-q1_0` | 248 MB | Pixel 8 Pro, Pixel 7a, Galaxy A14, Intel NUC, Windows RTX 4090, Windows Surface 8, Windows Legacy |

**4 generative Bonsai packs** (2 Fast + 2 Quality) across 12 device profiles,
plus the Qwen3 Q4_K_M alternates (0.6B/1.7B/4B) available per tier.
Tier differentiation is achieved via LoRA adapters hot-swapped at runtime.

### Backend Distribution

| Backend | Profiles | Platforms |
|---------|----------|-----------|
| MLX | 5 | iOS (aarch64), macOS (aarch64) |
| llama.cpp Vulkan | 6 | Android, Windows |
| llama.cpp CPU | 1 | macOS (x86_64, Intel NUC) |

## Model Download URLs

### Generative Models

| Pack ID | Download URL |
|---------|-------------|
| `bonsai-1.7b-mlx-1bit` | `https://huggingface.co/prism-ml/Bonsai-1.7B-mlx-1bit/resolve/main/model.safetensors` |
| `bonsai-1.7b-q1_0` | `https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf` |
| `bonsai-1.7b-mlx-2bit` | `https://huggingface.co/prism-ml/Bonsai-1.7B-mlx-2bit/resolve/main/model.safetensors` |
| `bonsai-1.7b-q2_0` | `https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q2_0.gguf` |
| `qwen3-0.6b-q4_k_m` | `https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf` |
| `qwen3-1.7b-q4_k_m` | `https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf` |
| `qwen3-4b-instruct-2507-q4_k_m` | `https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |

### Non-Generative Models

| Pack ID | Download URL |
|---------|-------------|
| `mmbert-safety-q4_k_m` | `https://cdn.kchat.dev/models/mmbert-safety-q4_k_m/1.0.0/model.gguf` |
| `mobileclip-s2-int8` | `https://cdn.kchat.dev/models/mobileclip-s2-int8/1.0.0/visual_encoder_int8.onnx` |
| `whisper-tiny` | `https://huggingface.co/NbAiLabBeta/nb-whisper-tiny/resolve/main/onnx/encoder_model.onnx` |
| `whisper-base` | `https://huggingface.co/NbAiLabBeta/nb-whisper-base/resolve/main/onnx/encoder_model.onnx` |
| `whisper-tiny-ggml` | `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin` |
| `whisper-base-ggml` | `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin` |

## Inference Servers

### GGUF Models (llama-server)

GGUF models run via `llama-server` subprocess:

```bash
llama-server -m <model.gguf> --port <port> -ngl 99 -c <context_size>
```

| Model | Context Size | Server Type |
|-------|-------------|-------------|
| `bonsai-1.7b-q1_0` / `bonsai-1.7b-q2_0` | per-tier cap (4K/8K/16K) | LlamaServer |
| `qwen3-*-q4_k_m` | per-tier cap (4K/8K/16K) | LlamaServer |

### MLX Models (kchat-mlx-server)

MLX models run via `kchat-mlx-server` (Swift binary preferred, Python fallback):

```bash
# Swift binary (preferred)
cd swift/kchat-mlx-server && swift build -c release

# Python fallback (requires pip install mlx-lm)
swift/kchat-mlx-server/kchat_mlx_server.py
```

| Model | Context Size | Server Type |
|-------|-------------|-------------|
| `bonsai-1.7b-mlx-1bit` / `bonsai-1.7b-mlx-2bit` | per-tier cap (4K/8K/16K) | MlxServer |

## Language Coverage

### Generative Models

| Language | Bonsai Models |
|----------|--------------|
| English (en) | ✅ |
| Vietnamese (vi) | ✅ |
| Chinese (zh) | ✅ |
| Japanese (ja) | ✅ |
| Korean (ko) | ✅ |
| Spanish (es) | ✅ |
| Arabic (ar) | ✅ |
| German (de) | ✅ |
| Hindi (hi) | ✅ |
| French (fr) | ✅ |

### ASR Models (Whisper)

| Language | Whisper Tiny | Whisper Base |
|----------|-------------|-------------|
| English, Vietnamese, Chinese, Japanese, Korean, Spanish, French, German, Arabic, Hindi, Thai | ✅ | ✅ |

### Encoder (Unified)

| Language | mmbert-safety-q4_k_m |
|----------|----------------------|
| 27 languages (en, vi, zh, ja, ko, es, ar, de, hi, fr, + 17 more) | ✅ |

### Eval Multilingual Coverage

Per-device eval tests across 10 languages + mixed-language code-switching:
English, Vietnamese, Japanese, Korean, Chinese, Spanish, Arabic, German, Hindi, French +
mixed-language scenarios.

## File Locations

| File | Purpose |
|------|---------|
| `crates/kchat-core/src/registry.rs` | Model registry definition (13 packs) |
| `crates/kchat-core/src/tier.rs` | Tier selection logic and resource budgets |
| `crates/kchat-core/src/capability.rs` | Device capability probe |
| `crates/kchat-generation/src/backend.rs` | Backend type selection (MLX/Vulkan/CPU) |
| `eval/kchat-task-suite/src/eval_device_profile.rs` | 12 device profiles + model selection |
| `eval/kchat-task-suite/src/eval_perdevice.rs` | Per-device eval harness |
| `eval/kchat-task-suite/src/device_simulator.rs` | Device simulator with model fit checks |
| `manifest/packs/` | Downloaded model packs (GGUF, MLX, ONNX) |

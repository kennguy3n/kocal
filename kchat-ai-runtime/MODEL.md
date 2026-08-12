# Model Registry & Device Profiles

> Complete technical reference for all model packs, device profiles, memory
> budgets, and model selection logic in kchat-ai-runtime.

## Model Registry (11 packs)

The model registry is the canonical catalog of all downloadable model packs.
It is defined in `crates/kchat-core/src/registry.rs` as
`ModelRegistry::default_registry()` and can also be loaded from TOML files.

Each `RegistryEntry` contains:
- `pack_id` — unique identifier (e.g. `ternary-bonsai-8b-mlx-2bit`)
- `version` — semantic version string
- `pack_type` — category: `generative`, `encoder`, `vision`, `asr`
- `download_url` — CDN or HuggingFace URL
- `sha256` — content hash for verification (Ed25519-signed manifest)
- `size_bytes` — exact download size
- `min_tier` — minimum device tier required (`Low`, `Medium`, `High`)
- `task_capabilities` — supported tasks (e.g. `summarize`, `translate`, `generate`, `tool_use`)
- `languages` — supported language codes (e.g. `en`, `vi`, `zh`, `ja`, `ko`, `es`, `ar`, `de`, `hi`, `fr`)
- `quantization` — quantization recipe (e.g. `Q2_0`, `Q4_K_M`, `Q8_0`, `2bit-MLX`, `INT8`, `INT4`)

### Generative Models (6 packs)

| Pack ID | Base Model | Params | Min Tier | Size | Quant | Backend | Platform | Context | Capabilities | Languages |
|---------|-----------|--------|----------|------|-------|---------|----------|---------|-------------|-----------|
| `ternary-bonsai-1.7b-mlx-2bit` | Qwen3-1.7B | 1.7B | Low | 472 MB | 2bit-MLX | MLX | iOS/macOS (aarch64) | 1,024 | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `ternary-bonsai-1.7b-q2_0` | Qwen3-1.7B | 1.7B | Low | 442 MB | Q2_0 | llama.cpp Vulkan/CPU | Android/Windows/Intel Mac | 1,024 | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `ternary-bonsai-4b-mlx-2bit` | Qwen3-4B | 4B | Medium | 1,132 MB | 2bit-MLX | MLX | iOS/macOS (aarch64) | 2,048 | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `ternary-bonsai-4b-q2_0` | Qwen3-4B | 4B | Medium | 1,075 MB | Q2_0 | llama.cpp Vulkan | Android | 2,048 | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `ternary-bonsai-8b-mlx-2bit` | Qwen3-8B | 8B | High | 2,304 MB | 2bit-MLX | MLX | iOS/macOS (aarch64) | 4,096 | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |
| `ternary-bonsai-8b-q2_0` | Qwen3-8B | 8B | High | 2,182 MB | Q2_0 | llama.cpp Vulkan | Android/Windows | 4,096 | summarize, translate, generate, tool_use | en, vi, zh, ja, ko, es, ar, de, hi, fr |

#### Ternary Bonsai Family

The Ternary Bonsai models are 1.58-bit ternary quantized variants of the Qwen3
model family. They use ternary weights (-1, 0, +1) which dramatically reduces
model size while maintaining reasonable quality.

| Model | Base | Parameters | Bits/Weight | Download Size | Running Size (est.) |
|-------|------|-----------|-------------|--------------|-------------------|
| Bonsai-1.7B MLX | Qwen3-1.7B | 1.7B | 1.58 | 472 MB | ~600 MB |
| Bonsai-1.7B GGUF | Qwen3-1.7B | 1.7B | 1.58 (Q2_0) | 442 MB | ~600 MB |
| Bonsai-4B MLX | Qwen3-4B | 4B | 1.58 | 1,132 MB | ~1.5 GB |
| Bonsai-4B GGUF | Qwen3-4B | 4B | 1.58 (Q2_0) | 1,075 MB | ~1.4 GB |
| Bonsai-8B MLX | Qwen3-8B | 8B | 1.58 | 2,304 MB | ~2.9 GB |
| Bonsai-8B GGUF | Qwen3-8B | 8B | 1.58 (Q2_0) | 2,182 MB | ~2.8 GB |

#### Quantization Comparison

| Quant | Bits/Weight | Quality | Size Factor | Use Case |
|-------|------------|---------|------------|----------|
| 2bit-MLX | 1.58 | Good | 0.25× | Apple Silicon, MLX framework |
| Q2_0 | 2.0 | Good | 0.28× | Non-Apple, llama.cpp |

### Encoder Models (2 packs)

Unified multi-task XLM-RoBERTa-base ONNX model. Replaces separate embedding,
safety, and reranker packs with a single shared encoder session.
Source: `models/quantized_models/onnx_int8/` and `models/quantized_models/onnx_int4/`.

| Pack ID | Base Model | Min Tier | Size | Quant | Backend | Tasks | Languages | SHA-256 |
|---------|-----------|----------|------|-------|---------|-------|-----------|---------|
| `kchat-encoder-int4` | XLM-RoBERTa-base | Low | 143 MB | INT4 | ONNX Runtime | safety, embed, rerank | en, vi, zh, ja, ko, es, ar, de, hi, fr | ✅ real |
| `kchat-encoder-int8` | XLM-RoBERTa-base | High | 266 MB | INT8 | ONNX Runtime | safety, embed, rerank | en, vi, zh, ja, ko, es, ar, de, hi, fr | ✅ real |

### Vision Model (1 pack)

Unified MobileCLIP-S2 INT8 ONNX model pack. Pack includes visual encoder (~37 MB)
and text encoder (~64 MB). Runtime loads only the visual encoder for image
classification/embedding and video frame classification.

| Pack ID | Base Model | Min Tier | Pack Size | Runtime | Quant | Backend | Tasks | Embedding Dim |
|---------|-----------|----------|-----------|---------|-------|---------|-------|--------------|
| `mobileclip-s2-int8` | MobileCLIP-S2 | Low | 97 MB | 37 MB | INT8 | ONNX Runtime | image_classify, image_embed, video_classify | 512 |

### ASR Models (2 packs)

Whisper ONNX models from NbAiLab (Norwegian Language Technology Lab). These are
Norwegian fine-tunes of OpenAI's Whisper models — `nb-whisper-tiny` and `nb-whisper-base`.
Despite the Norwegian fine-tuning, the models retain full multilingual capability
inherited from the original Whisper models. The ONNX files are **FP32 (not INT8-quantized)**.

Full pack includes encoder + decoder + decoder_with_past ONNX files.

| Pack ID | Base Model | Params | Min Tier | Size | Quant | Backend | Languages | SHA-256 |
|---------|-----------|--------|----------|------|-------|---------|-----------|---------|
| `whisper-tiny` | nb-whisper-tiny | 39M | Low | 33 MB | ONNX (FP32) | ONNX Runtime | en, vi, zh, ja, ko, es, fr, de, ar, hi, th | ✅ real |
| `whisper-base` | nb-whisper-base | 74M | Medium | 82 MB | ONNX (FP32) | ONNX Runtime | en, vi, zh, ja, ko, es, fr, de, ar, hi, th | ✅ real |

## Device Tiers

### Tier Definitions

| Tier | Mobile RAM | Desktop RAM | Description |
|------|-----------|-----------|-------------|
| **Low** | 4–6 GB | 8 GB | Minimal generative model, INT4/INT8 non-generative |
| **Medium** | 6–8 GB | 16–24 GB | Default generative pack, INT8 non-generative, FP32 vision |
| **High** | 8 GB+ | 32 GB+ | Large generative pack, FP32 vision, video classification |

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
| Context window | 2,048 tok | 4,096 tok | 8,192 tok (mobile) / 16,384 tok (desktop) |
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
`eval/kchat-task-suite/src/eval_device_profile.rs`:

```
is_apple_silicon = (platform == "ios" OR platform == "macos") AND cpu_arch == "aarch64"

Low tier:
  Apple Silicon  → ternary-bonsai-1.7b-mlx-2bit  (472 MB, MLX)
  Other          → ternary-bonsai-1.7b-q2_0       (442 MB, GGUF)

Medium tier:
  Apple Silicon  → ternary-bonsai-4b-mlx-2bit     (1,132 MB, MLX)
  Other          → ternary-bonsai-4b-q2_0          (1,075 MB, GGUF)

High tier:
  Apple Silicon  → ternary-bonsai-8b-mlx-2bit     (2,304 MB, MLX)
  Android/Win    → ternary-bonsai-8b-q2_0          (2,182 MB, GGUF)
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
| Vision (image+video) | mobileclip-s2-int8 (37 MB) | mobileclip-s2-int8 (37 MB) | mobileclip-s2-int8 (37 MB) |
| Encoder | kchat-encoder-int4 (143 MB) | kchat-encoder-int4 (143 MB) | kchat-encoder-int4 (143 MB) |
| ASR | whisper-tiny (33 MB) | whisper-base (82 MB) | whisper-base (82 MB) |
| Video | mobileclip-s2-int8 (same as vision) | mobileclip-s2-int8 (same as vision) | mobileclip-s2-int8 (same as vision) |

> **Lazy-loading**: Vision, ASR, and safety encoder models are loaded on-demand for
> their specific task and unloaded after use. During generation, only the generative
> model is resident in memory. This reduces effective memory footprint by 213–262 MB.
> All tiers use kchat-encoder-int4 (143 MB) for memory efficiency.

## Device Profiles (12 profiles)

All profiles are defined in `eval/kchat-task-suite/src/eval_device_profile.rs`
and mirrored in `eval/kchat-task-suite/src/eval_perdevice.rs`.

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
| Generative | `ternary-bonsai-8b-mlx-2bit` (2,304 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~2,566 MB** |

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
| Generative | `ternary-bonsai-4b-mlx-2bit` (1,132 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~1,394 MB** |

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
| Generative | `ternary-bonsai-1.7b-mlx-2bit` (472 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~685 MB** |

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
| Generative | `ternary-bonsai-8b-q2_0` (2,182 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~2,444 MB** |

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
| Generative | `ternary-bonsai-4b-q2_0` (1,075 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~1,337 MB** |

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
| Generative | `ternary-bonsai-1.7b-q2_0` (442 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~655 MB** |

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
| Generative | `ternary-bonsai-8b-mlx-2bit` (2,304 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~2,566 MB** |

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
| Generative | `ternary-bonsai-1.7b-mlx-2bit` (472 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~685 MB** |

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
| Generative | `ternary-bonsai-1.7b-q2_0` (442 MB, GGUF — no MLX on x86_64) |
| Backend | llama.cpp CPU |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~655 MB** |

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
| Generative | `ternary-bonsai-8b-q2_0` (2,182 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-base` (82 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~2,444 MB** |

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
| Generative | `ternary-bonsai-1.7b-q2_0` (442 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~655 MB** |

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
| Generative | `ternary-bonsai-1.7b-q2_0` (442 MB) |
| Backend | llama.cpp Vulkan |
| Vision | `mobileclip-s2-int8` (37 MB) |
| Encoder | `kchat-encoder-int4` (143 MB) |
| ASR | `whisper-tiny` (33 MB) |
| Video | `mobileclip-s2-int8` (same as vision) |
| **Total model footprint** | **~655 MB** |

## Summary Tables

### Model Footprint by Tier

**All models loaded** (worst case, all co-resident):

| Tier | Generative | Vision (image+video) | Encoder | ASR | **Total** |
|------|-----------|----------------------|---------|-----|-----------|
| **Low (Apple Silicon)** | 472 MB | 37 MB | 143 MB | 33 MB | **685 MB** |
| **Low (GGUF)** | 442 MB | 37 MB | 143 MB | 33 MB | **655 MB** |
| **Medium (Apple Silicon)** | 1,132 MB | 37 MB | 143 MB | 82 MB | **1,394 MB** |
| **Medium (Android)** | 1,075 MB | 37 MB | 143 MB | 82 MB | **1,337 MB** |
| **High (Apple Silicon)** | 2,304 MB | 37 MB | 143 MB | 82 MB | **2,566 MB** |
| **High (Android/Windows)** | 2,182 MB | 37 MB | 143 MB | 82 MB | **2,444 MB** |

**Effective footprint** (generative model only; encoder, vision, and ASR all lazy-loaded on demand):

| Tier | Generative | **Effective Footprint** |
|------|-----------|------------------------|
| **Low (Apple Silicon)** | 472 MB | **472 MB** |
| **Low (GGUF)** | 442 MB | **442 MB** |
| **Medium (Apple Silicon)** | 1,132 MB | **1,132 MB** |
| **Medium (Android)** | 1,075 MB | **1,075 MB** |
| **High (Apple Silicon)** | 2,304 MB | **2,304 MB** |
| **High (Android/Windows)** | 2,182 MB | **2,182 MB** |

> **Lazy-loading**: Vision (mobileclip-s2-int8, 37 MB runtime), ASR (whisper-tiny/base, 33/82 MB),
> and safety encoder (kchat-encoder-int4, 143 MB) are loaded on-demand for their specific
> task and unloaded after use. During generation, only the generative model is resident.
> This reduces effective memory footprint by 213–262 MB.
> All tiers use kchat-encoder-int4 (143 MB) for memory efficiency.

### KV Cache Estimates

KV cache type depends on the inference backend:
- **llama.cpp** (Android/Windows/Intel Mac): **Q8_0 quantized** (1 byte/element)
  — configured via `with_type_k(KvCacheType::Q8_0)` and `with_type_v(KvCacheType::Q8_0)`
  in `llamacpp.rs`. Halves cache memory vs FP16.
- **MLX** (iOS/macOS Apple Silicon): **FP16** (2 bytes/element)
  — MLX's Swift library does not expose KV cache quantization. Uses FP16 by default.

All Bonsai models use GQA with 8 KV heads and 128 head dimension.

| Model | Layers | KV Heads | Head Dim | Per-Token (Q8) | Per-Token (FP16) | iOS Ctx | Android Ctx | Q8 @ Android | FP16 @ iOS |
|-------|--------|----------|----------|----------------|------------------|---------|-------------|--------------|------------|
| Bonsai-1.7B | 28 | 8 | 128 | ~56 KB | ~115 KB | 1,024 | 2,048 | ~115 MB | ~115 MB |
| Bonsai-4B | 36 | 8 | 128 | ~72 KB | ~147 KB | 2,048 | 4,096 | ~295 MB | ~295 MB |
| Bonsai-8B | 36 | 8 | 128 | ~72 KB | ~147 KB | 4,096 | 8,192 | ~590 MB | ~590 MB |

Desktop context caps: Low 2K, Medium 4K, High 16K (generous memory budgets).

### Memory Budget vs Effective Footprint + KV Cache

| Tier | Platform | Backend | Peak Budget | Effective (Gen) | KV Cache | **Total** | **Headroom** |
|------|----------|---------|-------------|-----------------|----------|-----------|--------------|
| Low | iOS | MLX (FP16) | 750 MB | 472 MB | 115 MB | **587 MB** | **163 MB** |
| Low | Android | llama.cpp (Q8) | 750 MB | 442 MB | 115 MB | **557 MB** | **193 MB** |
| Low | macOS | MLX (FP16) | 2,000 MB | 472 MB | 115 MB | **587 MB** | **1,413 MB** |
| Low | Windows | llama.cpp (Q8) | 2,000 MB | 442 MB | 115 MB | **557 MB** | **1,443 MB** |
| Medium | iOS | MLX (FP16) | 1,700 MB | 1,132 MB | 295 MB | **1,427 MB** | **273 MB** |
| Medium | Android | llama.cpp (Q8) | 1,800 MB | 1,075 MB | 295 MB | **1,370 MB** | **430 MB** |
| Medium | macOS | MLX (FP16) | 4,000 MB | 1,132 MB | 590 MB | **1,722 MB** | **2,278 MB** |
| Medium | Windows | llama.cpp (Q8) | 4,000 MB | 1,075 MB | 590 MB | **1,665 MB** | **2,335 MB** |
| High | iOS | MLX (FP16) | 3,100 MB | 2,304 MB | 590 MB | **2,894 MB** | **206 MB** |
| High | Android | llama.cpp (Q8) | 3,200 MB | 2,182 MB | 590 MB | **2,772 MB** | **428 MB** |
| High | macOS | MLX (FP16) | 8,000 MB | 2,304 MB | 2,360 MB | **4,664 MB** | **3,336 MB** |
| High | Windows | llama.cpp (Q8) | 8,000 MB | 2,182 MB | 2,360 MB | **4,542 MB** | **3,458 MB** |

> **All profiles fit within their peak memory budgets** with per-backend KV cache
> (Q8 for llama.cpp, FP16 for MLX), platform-aware context caps, and lazy-loaded
> encoder/vision/ASR models. No budget increases needed.
> Tightest profile: Low/iOS with 163 MB headroom. Android retains original 2K/4K/8K
> context caps thanks to Q8 KV cache efficiency.

### Unique Generative Models per Profile

| Model | Size | Profiles Using It |
|-------|------|-------------------|
| `ternary-bonsai-1.7b-mlx-2bit` | 472 MB | iPhone SE 2022, MacBook Air M2 |
| `ternary-bonsai-1.7b-q2_0` | 442 MB | Galaxy A14, Intel NUC, Windows Surface 8, Windows Legacy |
| `ternary-bonsai-4b-mlx-2bit` | 1,132 MB | iPhone 14 |
| `ternary-bonsai-4b-q2_0` | 1,075 MB | Pixel 7a |
| `ternary-bonsai-8b-mlx-2bit` | 2,304 MB | iPhone 15 Pro, MacBook Pro M3 Max |
| `ternary-bonsai-8b-q2_0` | 2,182 MB | Pixel 8 Pro, Windows RTX 4090 |

**6 unique generative models** across 12 device profiles.

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
| `ternary-bonsai-1.7b-mlx-2bit` | `https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit/resolve/main/model.safetensors` |
| `ternary-bonsai-1.7b-q2_0` | `https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/main/Ternary-Bonsai-1.7B-Q2_0.gguf` |
| `ternary-bonsai-4b-q2_0` | `https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf/resolve/main/Ternary-Bonsai-4B-Q2_0.gguf` |
| `ternary-bonsai-4b-mlx-2bit` | `https://huggingface.co/prism-ml/Ternary-Bonsai-4B-mlx-2bit/resolve/main/model.safetensors` |
| `ternary-bonsai-8b-mlx-2bit` | `https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit/resolve/main/model.safetensors` |
| `ternary-bonsai-8b-q2_0` | `https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/resolve/main/Ternary-Bonsai-8B-Q2_0.gguf` |

### Non-Generative Models

| Pack ID | Download URL |
|---------|-------------|
| `kchat-encoder-int8` | `https://cdn.kchat.dev/models/kchat-encoder-int8/1.0.0/model_quantized.onnx` |
| `kchat-encoder-int4` | `https://cdn.kchat.dev/models/kchat-encoder-int4/1.0.0/model_quantized_int4.onnx` |
| `mobileclip-s2-int8` | `https://cdn.kchat.dev/models/mobileclip-s2-int8/1.0.0/visual_encoder_int8.onnx` |
| `whisper-tiny` | `https://huggingface.co/NbAiLabBeta/nb-whisper-tiny/resolve/main/onnx/encoder_model.onnx` |
| `whisper-base` | `https://huggingface.co/NbAiLabBeta/nb-whisper-base/resolve/main/onnx/encoder_model.onnx` |

## Inference Servers

### GGUF Models (llama-server)

GGUF models run via `llama-server` subprocess:

```bash
llama-server -m <model.gguf> --port <port> -ngl 99 -c <context_size>
```

| Model | Context Size | Server Type |
|-------|-------------|-------------|
| `ternary-bonsai-1.7b-q2_0` | 2,048 | LlamaServer |
| `ternary-bonsai-4b-q2_0` | 8,192 | LlamaServer |
| `ternary-bonsai-8b-q2_0` | 8,192 | LlamaServer |

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
| `ternary-bonsai-1.7b-mlx-2bit` | 2,048 | MlxServer |
| `ternary-bonsai-4b-mlx-2bit` | 4,096 | MlxServer |
| `ternary-bonsai-8b-mlx-2bit` | 8,192 | MlxServer |

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

| Language | kchat-encoder INT8/INT4 |
|----------|------------------------|
| English, Vietnamese, Chinese, Japanese, Korean, Spanish, Arabic, German, Hindi, French | ✅ |

### Eval Multilingual Coverage

Per-device eval tests across 10 languages + mixed-language code-switching:
English, Vietnamese, Japanese, Korean, Chinese, Spanish, Arabic, German, Hindi, French +
mixed-language scenarios.

## File Locations

| File | Purpose |
|------|---------|
| `crates/kchat-core/src/registry.rs` | Model registry definition (11 packs) |
| `crates/kchat-core/src/tier.rs` | Tier selection logic and resource budgets |
| `crates/kchat-core/src/capability.rs` | Device capability probe |
| `crates/kchat-generation/src/backend.rs` | Backend type selection (MLX/Vulkan/CPU) |
| `eval/kchat-task-suite/src/eval_device_profile.rs` | 12 device profiles + model selection |
| `eval/kchat-task-suite/src/eval_perdevice.rs` | Per-device eval harness |
| `eval/kchat-task-suite/src/device_simulator.rs` | Device simulator with model fit checks |
| `manifest/packs/` | Downloaded model packs (GGUF, MLX, ONNX) |

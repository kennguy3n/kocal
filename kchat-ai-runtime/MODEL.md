# Model Registry & Device Profiles

> Complete technical reference for all model packs, device profiles, memory
> budgets, and model selection logic in kchat-ai-runtime.

## Model Registry (18 packs)

The model registry is the canonical catalog of all downloadable model packs.
It is defined in `crates/kchat-core/src/registry.rs` as
`ModelRegistry::default_registry()` and can also be loaded from TOML files.

Each `RegistryEntry` contains:
- `pack_id` — unique identifier (e.g. `ternary-bonsai-8b-mlx-2bit`)
- `version` — semantic version string
- `pack_type` — category: `generative`, `embedding`, `safety`, `reranker`, `vision`, `asr`
- `download_url` — CDN or HuggingFace URL
- `sha256` — content hash for verification (Ed25519-signed manifest)
- `size_bytes` — exact download size
- `min_tier` — minimum device tier required (`Low`, `Medium`, `High`)
- `task_capabilities` — supported tasks (e.g. `summarize`, `translate`, `generate`, `tool_use`)
- `languages` — supported language codes (e.g. `en`, `vi`, `zh`, `ja`, `ko`, `es`)
- `quantization` — quantization recipe (e.g. `Q2_0`, `Q4_K_M`, `Q8_0`, `2bit-MLX`, `INT8`, `INT4`)

### Generative Models (9 packs)

| Pack ID | Base Model | Params | Min Tier | Size | Quant | Backend | Platform | Context | Capabilities | Languages |
|---------|-----------|--------|----------|------|-------|---------|----------|---------|-------------|-----------|
| `ternary-bonsai-1.7b-mlx-2bit` | Qwen3-1.7B | 1.7B | Low | 472 MB | 2bit-MLX | MLX | iOS/macOS (aarch64) | 2,048 | summarize, translate, generate, tool_use | en |
| `ternary-bonsai-1.7b-q2_0` | Qwen3-1.7B | 1.7B | Low | 442 MB | Q2_0 | llama.cpp Vulkan/CPU | Android/Windows/Intel Mac | 2,048 | summarize, translate, generate, tool_use | en |
| `qwen3.5-0.8b-q4` | Qwen3.5-0.8B | 0.8B | Medium | 500 MB | Q4_K_M | llama.cpp | All (fallback) | 4,096 | summarize, translate, generate | en, vi, zh, ja, ko, es |
| `ternary-bonsai-4b-mlx-2bit` | Qwen3-4B | 4B | Medium | 1,000 MB | 2bit-MLX | MLX | iOS/macOS (aarch64) | 4,096 | summarize, translate, generate, tool_use | en |
| `ternary-bonsai-4b-q2_0` | Qwen3-4B | 4B | Medium | 1,075 MB | Q2_0 | llama.cpp Vulkan | Android | 4,096 | summarize, translate, generate, tool_use | en |
| `ternary-bonsai-8b-mlx-2bit` | Qwen3-8B | 8B | High | 2,100 MB | 2bit-MLX | MLX | iOS/macOS (aarch64) | 8,192 | summarize, translate, generate, tool_use | en |
| `macaw-4bit-mlx` | Macaw | — | High | 1,500 MB | 4bit-MLX | MLX | iOS/macOS (aarch64) | 8,192 | summarize, translate, generate, tool_use | en |
| `ternary-bonsai-8b-q2_0` | Qwen3-8B | 8B | High | 2,182 MB | Q2_0 | llama.cpp Vulkan | Android/Windows | 8,192 | summarize, translate, generate, tool_use | en |
| `qwen3.5-0.8b-q8` | Qwen3.5-0.8B | 0.8B | High | 850 MB | Q8_0 | llama.cpp | All (fallback) | 4,096 | summarize, translate, generate | en, vi, zh, ja, ko, es |

#### Ternary Bonsai Family

The Ternary Bonsai models are 1.58-bit ternary quantized variants of the Qwen3
model family. They use ternary weights (-1, 0, +1) which dramatically reduces
model size while maintaining reasonable quality.

| Model | Base | Parameters | Bits/Weight | Download Size | Running Size (est.) |
|-------|------|-----------|-------------|--------------|-------------------|
| Bonsai-1.7B MLX | Qwen3-1.7B | 1.7B | 1.58 | 472 MB | ~600 MB |
| Bonsai-1.7B GGUF | Qwen3-1.7B | 1.7B | 1.58 (Q2_0) | 442 MB | ~600 MB |
| Bonsai-4B MLX | Qwen3-4B | 4B | 1.58 | 1,000 MB | ~1.3 GB |
| Bonsai-4B GGUF | Qwen3-4B | 4B | 1.58 (Q2_0) | 1,075 MB | ~1.4 GB |
| Bonsai-8B MLX | Qwen3-8B | 8B | 1.58 | 2,100 MB | ~2.7 GB |
| Bonsai-8B GGUF | Qwen3-8B | 8B | 1.58 (Q2_0) | 2,182 MB | ~2.8 GB |

#### Quantization Comparison

| Quant | Bits/Weight | Quality | Size Factor | Use Case |
|-------|------------|---------|------------|----------|
| 2bit-MLX | 1.58 | Good | 0.25× | Apple Silicon, MLX framework |
| Q2_0 | 2.0 | Good | 0.28× | Non-Apple, llama.cpp |
| Q4_K_M | 4.0 | Better | 0.50× | Fallback, multilingual |
| 4bit-MLX | 4.0 | Better | 0.50× | Apple Silicon, high quality |
| Q8_0 | 8.0 | Best | 1.0× | Fallback, highest precision |

### Embedding Model (1 pack)

| Pack ID | Base Model | Min Tier | Size | Quant | Backend | Tasks | Languages |
|---------|-----------|----------|------|-------|---------|-------|-----------|
| `multilingual-e5-small-int8` | multilingual-e5-small | Medium | 45 MB | INT8 | ONNX Runtime | embed | en, vi, zh, ja, ko, es |

### Safety Classifiers (2 packs)

| Pack ID | Min Tier | Size | Quant | Backend | Tasks | Languages |
|---------|----------|------|-------|---------|-------|-----------|
| `safety-classifier-int4` | Low | 15 MB | INT4 | ONNX Runtime | safety | en, vi, zh, ja, ko, es |
| `safety-classifier-int8` | Medium | 25 MB | INT8 | ONNX Runtime | safety | en, vi, zh, ja, ko, es |

### Reranker (1 pack)

| Pack ID | Base Model | Min Tier | Size | Quant | Backend | Tasks | Languages |
|---------|-----------|----------|------|-------|---------|-------|-----------|
| `cross-encoder-miniLM-int8` | cross-encoder-miniLM | High | 25 MB | INT8 | ONNX Runtime | rerank | en, vi, zh, ja, ko, es |

### Vision Models (3 packs)

| Pack ID | Base Model | Min Tier | Size | Quant | Backend | Tasks | Embedding Dim |
|---------|-----------|----------|------|-------|---------|-------|--------------|
| `mobileclip-s2-image-int8` | MobileCLIP-S2 | Low | 70 MB | INT8 | ONNX Runtime | image_classify, image_embed | 512 |
| `mobileclip-s2-image-fp32` | MobileCLIP-S2 | Medium | 137 MB | FP32 | ONNX Runtime | image_classify, image_embed | 512 |
| `mobileclip-s2-video-int8` | MobileCLIP-S2 | Medium | 70 MB | INT8 | ONNX Runtime | video_classify | 512 |

### ASR Models (2 packs)

| Pack ID | Base Model | Params | Min Tier | Size | Quant | Backend | Languages |
|---------|-----------|--------|----------|------|-------|---------|-----------|
| `whisper-tiny-int8` | Whisper Tiny | 39M | Low | 40 MB | INT8 | ONNX Runtime | en, vi, zh, ja, ko, es, fr, de, ar, hi, th |
| `whisper-base-int8` | Whisper Base | 74M | Medium | 90 MB | INT8 | ONNX Runtime | en, vi, zh, ja, ko, es, fr, de, ar, hi, th |

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
| Peak memory (iOS) | 750 MB | 1,400 MB | 2,500 MB |
| Peak memory (Android) | 750 MB | 1,500 MB | 3,000 MB |
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
  Apple Silicon  → ternary-bonsai-4b-mlx-2bit     (1,000 MB, MLX)
  Other          → ternary-bonsai-4b-q2_0          (1,075 MB, GGUF)

High tier:
  Apple Silicon  → ternary-bonsai-8b-mlx-2bit     (2,100 MB, MLX)
  Android/Win    → ternary-bonsai-8b-q2_0          (2,182 MB, GGUF)
  Other          → qwen3.5-0.8b-q8                 (850 MB, GGUF fallback)
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
| Vision (image) | mobileclip-s2-image-int8 (70 MB) | mobileclip-s2-image-fp32 (137 MB) | mobileclip-s2-image-fp32 (137 MB) |
| Safety encoder | safety-classifier-int4 (15 MB) | safety-classifier-int8 (25 MB) | safety-classifier-int8 (25 MB) |
| ASR | whisper-tiny-int8 (40 MB) | whisper-base-int8 (90 MB) | whisper-base-int8 (90 MB) |
| Video | — (none) | mobileclip-s2-video-int8 (70 MB) | mobileclip-s2-video-int8 (70 MB) |
| Embedding | — (none) | multilingual-e5-small-int8 (45 MB) | multilingual-e5-small-int8 (45 MB) |
| Reranker | — (none) | — (none) | cross-encoder-miniLM-int8 (25 MB) |

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
| Generative | `ternary-bonsai-8b-mlx-2bit` (2,100 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-image-fp32` (137 MB) |
| Safety | `safety-classifier-int8` (25 MB) |
| ASR | `whisper-base-int8` (90 MB) |
| Video | `mobileclip-s2-video-int8` (70 MB) |
| **Total model footprint** | **~2,422 MB** |

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
| Generative | `ternary-bonsai-4b-mlx-2bit` (1,000 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-image-fp32` (137 MB) |
| Safety | `safety-classifier-int8` (25 MB) |
| ASR | `whisper-base-int8` (90 MB) |
| Video | `mobileclip-s2-video-int8` (70 MB) |
| **Total model footprint** | **~1,322 MB** |

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
| Vision | `mobileclip-s2-image-int8` (70 MB) |
| Safety | `safety-classifier-int4` (15 MB) |
| ASR | `whisper-tiny-int8` (40 MB) |
| Video | — (none, Low tier) |
| **Total model footprint** | **~597 MB** |

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
| Vision | `mobileclip-s2-image-fp32` (137 MB) |
| Safety | `safety-classifier-int8` (25 MB) |
| ASR | `whisper-base-int8` (90 MB) |
| Video | `mobileclip-s2-video-int8` (70 MB) |
| **Total model footprint** | **~2,504 MB** |

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
| Vision | `mobileclip-s2-image-fp32` (137 MB) |
| Safety | `safety-classifier-int8` (25 MB) |
| ASR | `whisper-base-int8` (90 MB) |
| Video | `mobileclip-s2-video-int8` (70 MB) |
| **Total model footprint** | **~1,397 MB** |

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
| Vision | `mobileclip-s2-image-int8` (70 MB) |
| Safety | `safety-classifier-int4` (15 MB) |
| ASR | `whisper-tiny-int8` (40 MB) |
| Video | — (none, Low tier) |
| **Total model footprint** | **~567 MB** |

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
| Generative | `ternary-bonsai-8b-mlx-2bit` (2,100 MB) |
| Backend | MLX |
| Vision | `mobileclip-s2-image-fp32` (137 MB) |
| Safety | `safety-classifier-int8` (25 MB) |
| ASR | `whisper-base-int8` (90 MB) |
| Video | `mobileclip-s2-video-int8` (70 MB) |
| **Total model footprint** | **~2,422 MB** |

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
| Vision | `mobileclip-s2-image-int8` (70 MB) |
| Safety | `safety-classifier-int4` (15 MB) |
| ASR | `whisper-tiny-int8` (40 MB) |
| Video | — (none, Low tier) |
| **Total model footprint** | **~597 MB** |

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
| Vision | `mobileclip-s2-image-int8` (70 MB) |
| Safety | `safety-classifier-int4` (15 MB) |
| ASR | `whisper-tiny-int8` (40 MB) |
| Video | — (none, Low tier) |
| **Total model footprint** | **~567 MB** |

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
| Vision | `mobileclip-s2-image-fp32` (137 MB) |
| Safety | `safety-classifier-int8` (25 MB) |
| ASR | `whisper-base-int8` (90 MB) |
| Video | `mobileclip-s2-video-int8` (70 MB) |
| **Total model footprint** | **~2,504 MB** |

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
| Vision | `mobileclip-s2-image-int8` (70 MB) |
| Safety | `safety-classifier-int4` (15 MB) |
| ASR | `whisper-tiny-int8` (40 MB) |
| Video | — (none, Low tier) |
| **Total model footprint** | **~567 MB** |

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
| Vision | `mobileclip-s2-image-int8` (70 MB) |
| Safety | `safety-classifier-int4` (15 MB) |
| ASR | `whisper-tiny-int8` (40 MB) |
| Video | — (none, Low tier) |
| **Total model footprint** | **~567 MB** |

## Summary Tables

### Model Footprint by Tier

| Tier | Generative | Vision | Safety | ASR | Video | **Total** |
|------|-----------|--------|--------|-----|-------|-----------|
| **Low (Apple Silicon)** | 472 MB | 70 MB | 15 MB | 40 MB | — | **597 MB** |
| **Low (GGUF)** | 442 MB | 70 MB | 15 MB | 40 MB | — | **567 MB** |
| **Medium (Apple Silicon)** | 1,000 MB | 137 MB | 25 MB | 90 MB | 70 MB | **1,322 MB** |
| **Medium (Android)** | 1,075 MB | 137 MB | 25 MB | 90 MB | 70 MB | **1,397 MB** |
| **High (Apple Silicon)** | 2,100 MB | 137 MB | 25 MB | 90 MB | 70 MB | **2,422 MB** |
| **High (Android/Windows)** | 2,182 MB | 137 MB | 25 MB | 90 MB | 70 MB | **2,504 MB** |

### Memory Budget vs Model Footprint

| Tier | Platform | Peak Memory Budget | Model Footprint | KV Cache Headroom |
|------|----------|-------------------|-----------------|-------------------|
| Low | iOS | 750 MB | 597 MB | 153 MB |
| Low | Android | 750 MB | 567 MB | 183 MB |
| Low | macOS | 2,000 MB | 597 MB | 1,403 MB |
| Low | Windows | 2,000 MB | 567 MB | 1,433 MB |
| Medium | iOS | 1,400 MB | 1,322 MB | 78 MB |
| Medium | Android | 1,500 MB | 1,397 MB | 103 MB |
| Medium | macOS | 4,000 MB | 1,322 MB | 2,678 MB |
| Medium | Windows | 4,000 MB | 1,397 MB | 2,603 MB |
| High | iOS | 2,500 MB | 2,422 MB | 78 MB |
| High | Android | 3,000 MB | 2,504 MB | 496 MB |
| High | macOS | 8,000 MB | 2,422 MB | 5,578 MB |
| High | Windows | 8,000 MB | 2,504 MB | 5,496 MB |

### Unique Generative Models per Profile

| Model | Size | Profiles Using It |
|-------|------|-------------------|
| `ternary-bonsai-1.7b-mlx-2bit` | 472 MB | iPhone SE 2022, MacBook Air M2 |
| `ternary-bonsai-1.7b-q2_0` | 442 MB | Galaxy A14, Intel NUC, Windows Surface 8, Windows Legacy |
| `ternary-bonsai-4b-mlx-2bit` | 1,000 MB | iPhone 14 |
| `ternary-bonsai-4b-q2_0` | 1,075 MB | Pixel 7a |
| `ternary-bonsai-8b-mlx-2bit` | 2,100 MB | iPhone 15 Pro, MacBook Pro M3 Max |
| `ternary-bonsai-8b-q2_0` | 2,182 MB | Pixel 8 Pro, Windows RTX 4090 |
| `macaw-4bit-mlx` | 1,500 MB | (In registry, not assigned to any profile) |

**7 unique generative models** across 12 device profiles.

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
| `qwen3.5-0.8b-q4` | `https://cdn.kchat.dev/models/qwen3.5-0.8b-q4/1.0.0/qwen3.5-0.8b-q4.gguf` |
| `ternary-bonsai-4b-q2_0` | `https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf/resolve/main/Ternary-Bonsai-4B-Q2_0.gguf` |
| `ternary-bonsai-4b-mlx-2bit` | `https://huggingface.co/prism-ml/Ternary-Bonsai-4B-mlx-2bit/resolve/main/model.safetensors` |
| `ternary-bonsai-8b-mlx-2bit` | `https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit/resolve/main/model.safetensors` |
| `macaw-4bit-mlx` | `https://huggingface.co/badtheorylabs/Macaw-4bit-MLX/resolve/main/model.safetensors` |
| `ternary-bonsai-8b-q2_0` | `https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/resolve/main/Ternary-Bonsai-8B-Q2_0.gguf` |
| `qwen3.5-0.8b-q8` | `https://cdn.kchat.dev/models/qwen3.5-0.8b-q8/1.0.0/qwen3.5-0.8b-q8.gguf` |

### Non-Generative Models

| Pack ID | Download URL |
|---------|-------------|
| `multilingual-e5-small-int8` | `https://cdn.kchat.dev/models/multilingual-e5-small-int8/1.0.0/multilingual-e5-small-int8.onnx` |
| `safety-classifier-int8` | `https://cdn.kchat.dev/models/safety-classifier-int8/1.0.0/safety-classifier-int8.onnx` |
| `safety-classifier-int4` | `https://cdn.kchat.dev/models/safety-classifier-int4/1.0.0/safety-classifier-int4.onnx` |
| `cross-encoder-miniLM-int8` | `https://cdn.kchat.dev/models/cross-encoder-miniLM-int8/1.0.0/cross-encoder-miniLM-int8.onnx` |
| `mobileclip-s2-image-int8` | `https://cdn.kchat.dev/models/mobileclip-s2-image-int8/1.0.0/mobileclip-s2-image-int8.onnx` |
| `mobileclip-s2-image-fp32` | `https://cdn.kchat.dev/models/mobileclip-s2-image-fp32/1.0.0/mobileclip-s2-image-fp32.onnx` |
| `mobileclip-s2-video-int8` | `https://cdn.kchat.dev/models/mobileclip-s2-video-int8/1.0.0/mobileclip-s2-video-int8.onnx` |
| `whisper-tiny-int8` | `https://cdn.kchat.dev/models/whisper-tiny-int8/1.0.0/whisper-tiny-int8.onnx` |
| `whisper-base-int8` | `https://cdn.kchat.dev/models/whisper-base-int8/1.0.0/whisper-base-int8.onnx` |

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
| `qwen3.5-0.8b-q4` | 4,096 | LlamaServer |
| `qwen3.5-0.8b-q8` | 4,096 | LlamaServer |

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
| `macaw-4bit-mlx` | 8,192 | MlxServer |

## Language Coverage

### Generative Models

| Language | Bonsai Models | Qwen Models | Macaw |
|----------|--------------|-------------|-------|
| English (en) | ✅ | ✅ | ✅ |
| Vietnamese (vi) | — | ✅ | — |
| Chinese (zh) | — | ✅ | — |
| Japanese (ja) | — | ✅ | — |
| Korean (ko) | — | ✅ | — |
| Spanish (es) | — | ✅ | — |

### ASR Models (Whisper)

| Language | Whisper Tiny | Whisper Base |
|----------|-------------|-------------|
| English, Vietnamese, Chinese, Japanese, Korean, Spanish, French, German, Arabic, Hindi, Thai | ✅ | ✅ |

### Embedding & Safety

| Language | e5-small | Safety INT8/INT4 | Reranker |
|----------|---------|-----------------|----------|
| English, Vietnamese, Chinese, Japanese, Korean, Spanish | ✅ | ✅ | ✅ |

### Eval Multilingual Coverage

Per-device eval tests across 10 languages + mixed-language code-switching:
English, Vietnamese, Japanese, Korean, Chinese, Spanish, Arabic, Hindi, Thai +
mixed-language scenarios.

## File Locations

| File | Purpose |
|------|---------|
| `crates/kchat-core/src/registry.rs` | Model registry definition (18 packs) |
| `crates/kchat-core/src/tier.rs` | Tier selection logic and resource budgets |
| `crates/kchat-core/src/capability.rs` | Device capability probe |
| `crates/kchat-generation/src/backend.rs` | Backend type selection (MLX/Vulkan/CPU) |
| `eval/kchat-task-suite/src/eval_device_profile.rs` | 12 device profiles + model selection |
| `eval/kchat-task-suite/src/eval_perdevice.rs` | Per-device eval harness |
| `eval/kchat-task-suite/src/device_simulator.rs` | Device simulator with model fit checks |
| `manifest/packs/` | Downloaded model packs (GGUF, MLX, ONNX) |

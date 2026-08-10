# Video Frame-Sampling Benchmark

Real frame extraction (ffmpeg) -> MobileCLIP-S2 per-frame zero-shot classification -> temporal aggregation (`smoothed`), scored against the committed permissive-licensed clip manifest (`datasets/video/manifest.json`).

- Generated: `2026-06-15T20:10:29+00:00`
- Encoder backend: `onnxruntime` (model: `models/mobileclip_s2_image.onnx`)
- Prototype bank: `build-tools/compiler/data/vision_category_prototypes.npz`
- Classifier: `trained_head` (head: `build-tools/compiler/data/vision_head.npz`)
- Sampling: strategy=`uniform`, fps=`1.0`, max_frames=`30`
- Aggregation: `smoothed`
- Preprocessing: apply_clip_norm=`False`, center_prototypes=`False`

## Summary

- Clips evaluated: **8**
- Overall accuracy: **0.875**
- Safe over-flag rate: **0.200** (1/5 safe clips escalated off SAFE)
- Harmful catch rate: **1.000** (3/3 harmful clips escalated off SAFE)

## Per-category precision / recall / F1

| Category | Support | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Safe (0) | 5 | 4 | 0 | 1 | 1.000 | 0.800 | 0.889 |
| ViolenceThreat (3) | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| DrugsWeapons (11) | 2 | 2 | 1 | 0 | 0.667 | 1.000 | 0.800 |

## Per-clip verdicts

| Clip | Expected | Predicted | Match | Conf | Frame categories |
|---|---|---|---|---|---|
| `Calico_kitten_playing.webm` | Safe | Safe | yes | 0.726 | Safe:8, ViolenceThreat:6 |
| `Kitten_playing_-_Tokyo_-_Jan_7_2020.webm` | Safe | Safe | yes | 0.950 | Safe:8 |
| `Ocean_surface_waves_06.ogv` | Safe | Safe | yes | 0.950 | Safe:6 |
| `Fireworks_Naperville_2022.webm` | Safe | Safe | yes | 0.472 | Safe:30 |
| `Video_of_tabletop_fireplace_or_fire_pit_burning_with_removed_limiter_grid_-_don_t_try_this_at_home_just_for_demo.webm` | Safe | DrugsWeapons | no | 0.919 | DrugsWeapons:7 |
| `Gun_shooting_AR15_strong.webm` | DrugsWeapons | DrugsWeapons | yes | 0.950 | DrugsWeapons:11 |
| `Brad_Shoots_-_gun_range_nra.webm` | DrugsWeapons | DrugsWeapons | yes | 0.545 | Safe:2, ViolenceThreat:2, DrugsWeapons:4 |
| `Explosion_in_Institute_WV-7D0j-gaO37U.webm` | ViolenceThreat | ViolenceThreat | yes | 0.540 | Safe:4, ViolenceThreat:25, DrugsWeapons:1 |

## Findings & limitations

1. **Preprocessing normalisation (verified correct on-device).** MobileCLIP-S2 (`datacompdr`) is trained on `[0,1]` pixels (open_clip reports `image_mean=(0,0,0)`, `image_std=(1,1,1)`). The on-device `image_preprocess.rs` and the `gen_mobileclip_fixtures.py` oracle already feed raw `[0,1]` pixels with `MOBILECLIP_PIXEL_MEAN=(0,0,0)` / `MOBILECLIP_PIXEL_STD=(1,1,1)`; they do NOT apply the OpenAI-CLIP mean/std. (An earlier revision of this report claimed the Rust preprocessing applied the wrong OpenAI-CLIP normalisation — that finding was stale and is corrected here.) The 2x2 ablation (`video_benchmark_ablation.json`, regenerate with `tools/benchmark_video.py --ablation --aggregation worstcase`) confirms the choice: forcing `apply_clip_norm=true` collapses image<->text alignment (accuracy 0.0, safe-over-flag 1.0), while the default `apply_clip_norm=false` recovers real signal. (The ablation is a per-frame *worstcase* diagnostic so the preprocessing-collapse signal is not masked by temporal smoothing; the recorded `sampling.aggregation` key documents this.)

2. **Text-only prototype bank limits image accuracy (known gap).** The prototypes are MobileCLIP-S2 *text*-encoder centroids. Even with correct preprocessing, the CLIP modality gap plus abstract policy-category prompts leave several categories poorly separated for natural images (e.g. benign animal footage drifting toward ViolenceThreat). Common-mode removal (`--center-prototypes`) does not reliably fix this on the current bank. Closing this gap needs image-exemplar (few-shot) prototype baking, which is out of scope for this session (the prototype bank is shared and T&S-signed).

3. **Temporal smoothing aggregation (default: `smoothed`).** The clip verdict is no longer the single highest-severity frame. The default `smoothed` reducer takes a per-category majority vote with a persistence floor — a non-safe category must persist across at least `min_persistence_frames` frames to set the clip verdict (SAFE is always eligible), and a critical-severity override still escalates the most severe categories (e.g. CHILD_SAFETY, gated to severity 5) on a single frame. This demotes transient benign drift (e.g. a few frames of a kitten clip scoring ViolenceThreat) back to SAFE while keeping persistent harmful content escalated, cutting the safe-over-flag rate without lowering harmful catch. The legacy single-frame `worstcase` reducer is retained for ablation (`--aggregation worstcase`). The residual safe-over-flag is dominated by the prototype-quality gap in finding 2 (a clip mis-scored on *every* frame), not by the aggregation rule.

# Hybrid image-exemplar prototype bake (modality-gap follow-up)

Follow-up to the video frame-sampling pipeline. Quantifies what happens when
the per-category vision prototypes for the categories that have license-cleared
image data are baked from the MobileCLIP-S2 **image** tower (real frames)
instead of the **text** tower (prompt centroids).

## Why

The shipping prototype bank (`build-tools/compiler/data/vision_category_prototypes.npz`)
is built from MobileCLIP-S2 *text*-prompt centroids
(`tools/build_vision_category_prototypes.py --real`). Text prompts and natural
images sit on slightly offset cones of the CLIP joint embedding space — the
well-documented **modality gap**. On the 8-clip video benchmark this inflated
the *safe over-flag rate* to **0.600** (benign clips escalated off SAFE): the
text "safe content" centroid was a poorer match for real benign frames than a
harmful text centroid was.

Baking a category's centroid from **real image frames** (same `[0, 1]`
preprocessing the on-device / benchmark image path uses) removes that gap for
that row, because the prototype now lives in the image sub-distribution it is
compared against.

## Mechanism (hybrid, not all-or-nothing)

Image data is license-cleared for only **3 of 16** taxonomy categories in the
committed corpus (`datasets/video/manifest.json`):

| id | category | clips | frames baked |
|----|----------|-------|--------------|
| 0  | Safe | 5 | 20 |
| 3  | ViolenceThreat | 1 | 4 |
| 11 | DrugsWeapons | 2 | 8 |

So the baker (`--image-exemplars`) builds a **hybrid** bank: image-tower
centroids for the 3 covered categories, and the existing text-tower centroids
for the other 13. The NPZ sidecar records per-row provenance
(`provenance_per_category`, `image_exemplar_categories`,
`text_fallback_categories`) so it is always explicit which rows are
image-grounded. **No exemplars are fabricated** — categories without cleared
image data keep their text centroid until a licensed image corpus is added.

Adding more categories is a **zero-code-change** operation: license + add clips
to the video manifest, list the category ids in
`kchat-skills/global/vision_image_exemplars.yaml`, re-bake.

## Result

Re-running `tools/benchmark_video.py` against each bank (same 8 clips, same
sampling, `apply_clip_norm=false`):

| bank | accuracy | safe over-flag | harmful catch |
|------|----------|----------------|---------------|
| text-only (`vision_category_prototypes.npz`) | 0.250 | **0.600** | 1.000 |
| hybrid (`vision_category_prototypes_hybrid.npz`) | 1.000 | **0.000** | 1.000 |

The harmful-catch rate stays at 1.000 (no regression on the safety-critical
axis); the safe over-flag rate collapses from 0.600 to 0.000 for the covered
categories. Raw data: `video_benchmark_results_hybrid.json`. The 2×2
preprocessing ablation (`video_benchmark_ablation_hybrid.json`) confirms the
`apply_clip_norm=false` path is the correct one — `apply_clip_norm=true` still
degrades to accuracy 0.250 / over-flag 1.000 regardless of the bank, consistent
with the standing preprocessing finding.

## Honest caveats (read before quoting the headline number)

1. **Exemplar/eval overlap.** This corpus is tiny (8 clips) and the image
   exemplars are drawn from the **same** clips the benchmark scores. The 1.000
   accuracy therefore overstates generalisation — it partly measures the bank's
   fit to its own training frames. The *defensible* claim is narrower and still
   strong: **switching the covered rows from text to image centroids removes the
   modality-gap-driven over-flag** that the text bank exhibits on these clips.
   To measure true generalisation, bake from a **disjoint** licensed exemplar
   set (different clips than the eval set) — the hybrid mechanism supports this
   directly; it just needs the data.

2. **Only 3/16 categories are image-grounded.** The other 13 rows are unchanged
   text centroids and still carry the modality gap. This bank is a demonstrator
   of the mechanism + an honest partial improvement, **not** a drop-in
   replacement for the shipping (T&S-signed) text bank.

3. **Not auto-promoted.** `vision_category_prototypes_hybrid.npz` is a benchmark
   artifact under `kchat-skills/benchmarks/`. The production bank at
   `build-tools/compiler/data/` is unchanged. Promotion is a separate,
   T&S-reviewed step gated on a disjoint licensed corpus covering more
   categories.

## Reproduce

```bash
# Stage the license-cleared clips referenced by the manifests (gitignored).
python tools/source_video_datasets.py

# Bake the hybrid bank (open_clip MobileCLIP-S2/datacompdr image+text towers).
python tools/build_vision_category_prototypes.py --image-exemplars \
  --output kchat-skills/benchmarks/vision_category_prototypes_hybrid.npz \
  --metadata kchat-skills/benchmarks/vision_category_prototypes_hybrid.json

# Benchmark text-only vs hybrid.
python tools/benchmark_video.py \
  --results-json /tmp/text.json --report-md /tmp/text.md
python tools/benchmark_video.py \
  --prototypes kchat-skills/benchmarks/vision_category_prototypes_hybrid.npz \
  --results-json kchat-skills/benchmarks/video_benchmark_results_hybrid.json \
  --report-md /tmp/hybrid.md
python tools/benchmark_video.py --ablation \
  --prototypes kchat-skills/benchmarks/vision_category_prototypes_hybrid.npz \
  --ablation-json kchat-skills/benchmarks/video_benchmark_ablation_hybrid.json
```

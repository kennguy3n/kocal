# Sample Messages

A curated, privacy-safe corpus used by
[`tools/run_guardrail_demo.py`](../../tools/run_guardrail_demo.py) and
the test suite to drive the full hybrid local pipeline through every
taxonomy category and every protected-speech context.

## What's in here

- [`sample_messages.yaml`](sample_messages.yaml) — 206 cases (68 text +
  138 vision/image/video). Each case is a fixture that maps directly
  onto
  [`kchat.guardrail.local_signal.v1`](../global/local_signal_schema.json)
  fields plus an `expected_category` / `expected_severity` /
  `description` field used by the demo and tests.

The text corpus covers:

- **Safe / benign** — greetings, logistics, multi-language samples
  (English, Vietnamese, Spanish, German, en↔vi code-switching).
- **Scam / fraud** — fake giveaway, credential harvest, advance fee.
  Each case uses high-risk-TLD URLs and scam-keyword combinations the
  deterministic detectors react to (no live phishing domains).
- **PII exposure** — email-, phone-, and credit-card-shaped patterns.
  No real PII; the digits in the credit-card sample are the canonical
  test value `4111 1111 1111 1111`.
- **Child safety** — discussions *about* safeguarding training in a
  minors-aware group. The corpus deliberately contains no
  CSAM-adjacent text; the encoder classifier must not produce a
  CHILD_SAFETY false positive on these.
- **Hate / harassment** — counterspeech and discussions *about*
  harassment. These exercise the protected-speech contexts.
- **Health misinformation** — quoted health claims with explicit
  refutation. Exercises `NEWS_CONTEXT` / `QUOTED_SPEECH_CONTEXT`.
- **Marketplace violations** — admin reminding members of restricted
  goods rules in a marketplace community.
- **Sexual / adult** — explicit adult-only channel with NSFW image
  descriptor. Tests `group_age_mode: adult_only` gating.
- **Extremism** — news reporting on banned organisations (NEWS_CONTEXT).
- **Self-harm** — supportive resource sharing in a mental-health
  community (resource_link_id surfacing).
- **Drugs / weapons** — public-health bulletin (EDUCATION_CONTEXT).
- **Community rule** — mild off-topic in a workplace overlay.

## Vision / image / video corpus

The 138 `vision-*` cases extend the corpus into the image and video
modality. Every case carries a `media_descriptors` block (the
structured on-device signal — `kind`, `nsfw_score`, `violence_score`,
`face_count`) and is grouped into four blocks delimited by
`# >>> BEGIN generated vision corpus` / `# <<< END generated vision
corpus` markers:

- **Per-category coverage** — ≥ 6 cases for each of taxonomy
  categories 1–15, spread across 8+ locales (en-US, de-DE, ja-JP,
  pt-BR, ar-SA, ko-KR, hi-IN, vi-VN, …). Each case pairs a plausible
  media descriptor with the jurisdiction/community overlay that
  governs it.
- **Video-specific (21 cases)** — `kind: video` descriptors populating
  both `nsfw_score` and `violence_score`, including minor_present
  child-safety escalation, across school / gaming / workplace /
  adult_only overlays and US / DE / JP / BR / IN / SA / KR
  jurisdictions.
- **Cross-cultural sensitivity (15 cases)** — religious imagery,
  political symbols banned in specific jurisdictions (e.g. StGB §86a in
  DE/AT), region-specific nudity / violence / alcohol-tobacco norms.
- **Age-mode escalation matrix (12 cases)** — the same NSFW descriptor
  evaluated across `minor_present` / `mixed_age` / `adult_only` × four
  jurisdictions, verifying the CHILD_SAFETY-floor escalation rule.

Only the structured-signal categories are deterministically decided by
`MockEncoderAdapter` (NSFW → SEXUAL_ADULT, NSFW + minor_present →
CHILD_SAFETY floor, high `violence_score` → VIOLENCE_THREAT). The
remaining vision categories (e.g. HATE / EXTREMISM / DRUGS_WEAPONS
imagery) are decided by the MobileCLIP-S2 prototype classifier in a
full-stack deployment; in the deterministic demo they surface as SAFE
and are retained as coverage for the held-out vision eval.

### Regenerating the vision corpus

The vision cases are produced deterministically (idempotent) by
[`tools/gen_vision_corpus.py`](../../tools/gen_vision_corpus.py), which
rewrites the marked block in `sample_messages.yaml` and the companion
[`eval/held_out_vision.yaml`](../eval/held_out_vision.yaml):

```bash
python tools/gen_vision_corpus.py
```

## File format

```yaml
- case_id: "safe-greeting-01"          # stable string id (used in benchmark reports)
  message:                              # `kchat.guardrail.local_signal.v1.message`
    text: "Hey everyone, what time is the meeting tomorrow?"
    lang_hint: "en"
    has_attachment: false
    attachment_kinds: []
    quoted_from_user: false
    is_outbound: false
    # Optional: media_descriptors are moved into `local_signals` by
    # the pipeline (see build-tools/compiler/pipeline/).
    # media_descriptors:
    #   - kind: image
    #     nsfw_score: 0.05
    #     violence_score: 0.0
    #     face_count: 4
  context:                              # `kchat.guardrail.local_signal.v1.context`
    group_kind: "small_group"
    group_age_mode: "mixed_age"
    user_role: "member"
    relationship_known: true
    locale: "en-US"
    jurisdiction_id: null               # set when a jurisdiction overlay is active
    community_overlay_id: null          # set when a community overlay is active
    is_offline: false
  expected_category: 0                  # 0..16 from `taxonomy.yaml`
  expected_severity: 0                  # 0..5 from `severity.yaml`
  description: "Benign scheduling message — should classify as SAFE."
```

`expected_category` / `expected_severity` describe the *deterministic*
verdict the demo expects from `MockEncoderAdapter`. A real encoder
classifier (XLM-R via `XLMRAdapter`, ONNX Runtime) may produce a
different but still schema-conformant output; the demo prints both so
divergence is visible.

## Privacy contract

All fixtures comply with
[`privacy_contract.yaml`](../global/privacy_contract.yaml):

- No real PII (the credit-card number in `pii-credit-card-01` is the
  industry-standard test value `4111 1111 1111 1111`).
- No live phishing domains. URLs use synthetic `*.click` / `*.top` host
  names that the deterministic URL detector flags via the high-risk-TLD
  list in `build-tools/compiler/pipeline/url.py`.
- No CSAM-adjacent text. The CHILD_SAFETY-relevant case is administrative
  language about safeguarding training; the lack of a CHILD_SAFETY
  output is deliberate.

## How to use

### Run the demo against a local XLM-R

```bash
# 1. Make the XLM-R ONNX model + tokenizer available locally (see
#    "Running with XLM-R" in the top-level README). Then:
python tools/run_guardrail_demo.py
```

### Run the demo with the deterministic mock adapter

```bash
python tools/run_guardrail_demo.py --mock
```

### Run with overlays

```bash
python tools/run_guardrail_demo.py \
  --jurisdiction us \
  --community workplace \
  --mock
```

### Benchmark + commit results

```bash
# Runs PipelineBenchmark over the corpus and writes
# kchat-skills/benchmarks/xlmr_results.json (or _mock_*.json
# when --mock is set).
python tools/run_guardrail_demo.py --benchmark --commit-results
```

See [`kchat-skills/benchmarks/README.md`](../benchmarks/README.md) for
the benchmark methodology.

## Extending the corpus

When adding a case:

1. Use a stable `case_id` matching `^[a-z0-9-]+$`.
2. Keep `text` short (the demo prints it; long lines wrap in terminals).
3. Match `expected_category` to one of the 17 taxonomy ids in
   `kchat-skills/global/taxonomy.yaml` (ids 0..16, where 16 is
   `DEEPFAKE_SYNTHETIC`).
   - Deepfake / synthetic-media cases are **encoder-required**: the
     structured `media_descriptor` cannot decide them on its own, so
     they exist to exercise the vision encoder path. Disclosed,
     non-deceptive AI-art must stay `expected_category: 0` (SAFE) —
     see the `vision-deepfake-neg-*` negatives.
4. Pin `expected_severity` to the deterministic detector + threshold
   policy outcome — not the encoder's output. The detector behaviour
   is stable across classifier-adapter swaps.
5. Avoid literal harm content. The detectors react to *shapes* (URL
   TLDs, keyword combinations, PII patterns); descriptive language
   gives the encoder classifier room to reason.
6. Run `pytest build-tools/tests/test_sample_messages.py` to
   confirm the case is well-formed.

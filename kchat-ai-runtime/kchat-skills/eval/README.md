# Held-out evaluation set

This directory holds the held-out evaluation set used by the pack
lifecycle gate (compiler — see `ARCHITECTURE.md`). It is
**not** part of the training corpus and must not be used to fit
any model component.

## Why a held-out set is non-negotiable

Accuracy on the training corpus
(`build-tools/compiler/training_data.py`) is not a meaningful safety
metric — the encoder head was fit on those exact strings and will
trivially memorise them. The numbers in the skill passport's
`test_results` block come from this directory.

## Methodology requirements

Cases here must obey the following rules. The pack lifecycle gate
inspects every new PR that touches this directory and rejects the
PR if any rule is violated.

1. **No overlap with training data.** Every case in this directory
   must be absent (verbatim and at the embedding-near-duplicate
   level) from `training_data.py`. The compiler runs a similarity
   check at pack-build time.
2. **Multilingual coverage.** At minimum: English, Vietnamese,
   Spanish, German, Japanese, Bengali, and Arabic must each be
   represented on both the benign and the adversarial sides. The
   `language` field on each case is canonical (BCP-47).
3. **Adversarial cases for every harm category.** Each non-SAFE
   taxonomy category must have at least three adversarial cases
   (leetspeak, homoglyph, zero-width, code-switch, euphemism) so
   the calibration report can surface category-specific weaknesses.
4. **Benign false-positive cases.** News coverage of harm,
   educational discussion, counterspeech, reclaimed language,
   identity discussion, health discussion, and political discourse
   each contribute at least five cases. These are the cases the
   pipeline is most likely to over-trigger on and they are tracked
   separately in the report.
5. **Closed labels.** `expected_category` is the integer id from
   `kchat-skills/global/taxonomy.yaml`. `expected_severity` is on
   the 0–5 rubric. `expected_actions` is the action dict; an empty
   `actions` block means "no action allowed".
6. **Privacy contract.** No real user data, no real PII, no live
   URLs. Cases must paraphrase realistic harms without producing
   exploit-ready content.

## Files

* `eval_template.yaml` — schema for a single eval case.
* `held_out_benign.yaml` — benign cases the pipeline must NOT
  flag. Minimum: 50 cases.
* `held_out_adversarial.yaml` — adversarial cases the pipeline
  must catch. Minimum: 30 cases.
* `eval_runner.py` — runs `GuardrailPipeline` end-to-end on every
  case, produces a calibration report with per-category precision /
  recall, false-positive rate on the benign set, and expected
  calibration error (ECE).

## Running the calibration report

```bash
python kchat-skills/eval/eval_runner.py \
    --benign kchat-skills/eval/held_out_benign.yaml \
    --adversarial kchat-skills/eval/held_out_adversarial.yaml \
    --output /tmp/calibration.json
```

The runner exits non-zero if any of the following pack-passport
gates fail:

* `child_safety_recall < 0.95`
* `privacy_leak_precision < 0.85`
* `protected_speech_false_positive > 0.05`
* `minority_language_false_positive > 0.10`

These match the thresholds in
`build-tools/compiler/metric_validator.py` and the pack passport's
`test_results` block.

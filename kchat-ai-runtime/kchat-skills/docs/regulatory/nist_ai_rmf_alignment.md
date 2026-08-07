# NIST AI Risk Management Framework (AI RMF 1.0) — Alignment Mapping

This document maps each function of the NIST AI Risk Management
Framework (NIST AI 100-1) to the corresponding concrete artefact in the
KChat on-device guardrail skill-pack system. The AI RMF is a
voluntary, rights-preserving framework organised around four core
functions: **Govern, Map, Measure, Manage**.

## Trustworthy AI characteristics (AI RMF § 3)

The AI RMF identifies seven characteristics of trustworthy AI. The
mapping below is the load-bearing cross-reference used by the
regulatory gate; every characteristic must be addressed by at
least one named artefact.

| Characteristic | Artefact |
| --- | --- |
| **Valid & Reliable** | `build-tools/compiler/metric_validator.py` pins targets for `minority_language_false_positive <= 0.07` and the deterministic-detector precision metrics. The adversarial corpus (`kchat-skills/tests/adversarial/corpus.yaml`) exercises robustness under homoglyph, leetspeak, unicode-trick, whitespace-insertion and image-text-evasion attacks. |
| **Safe** | `build-tools/compiler/anti_misuse.py` — bright-line validators: no overlay may relax category 1 (child safety) below floor 5, redefine the 17-category taxonomy, or redefine the 8 baseline privacy rules. |
| **Secure & Resilient** | On-device pipeline with no network dependencies (`build-tools/compiler/pipeline/`). The baseline declares `offline_capable: true` for every pack kind. |
| **Accountable & Transparent** | Every pack carries `signers`, `expires_on`, and a `skill_passport` entry (`build-tools/compiler/skill_passport.py`). The compiled-prompt format (`kchat-skills/prompts/compiled_prompt_format.md`) exposes the exact rules to downstream reviewers. |
| **Explainable & Interpretable** | The classifier output contract (`kchat-skills/global/output_schema.json`) carries a stable `rationale_id` pointing to a human-readable catalogue entry. Pack summaries are surfaced via `user_notice.visible_pack_summary`. |
| **Privacy-Enhanced** | The baseline's 8 privacy rules (`kchat-skills/global/baseline.yaml` `privacy_rules`) are immutable — `anti_misuse.assert_privacy_rules_not_redefined` rejects any overlay that touches them. The appeal flow (`build-tools/compiler/appeal_flow.py`) records only metadata, never content. |
| **Fair (Harmful bias managed)** | `build-tools/compiler/bias_audit.py` runs per-minority-language-target audits; `kchat-skills/tests/jurisdictions/test_minority_language_fp.py` exercises 62 archetypes × ≥4 minority-language / code-switching false-positive cases. |

---

## 1. Govern (AI RMF § 5.1)

**GOVERN 1 — Policies, processes, procedures, and practices.**

- `kchat-skills/docs/regulatory/` (this directory) aligns policies with DSA, NIST AI RMF, and UNICEF/ITU COP.
- `README.md` and `ARCHITECTURE.md` are the repo-canonical documents describing the system.

**GOVERN 2 — Accountability structures.**

- `signers: [trust_and_safety, legal_review, cultural_review]` on every jurisdiction pack names accountable roles. `anti_misuse.assert_required_signers` refuses to sign packs without them.

**GOVERN 3 — Workforce diversity, equity, inclusion, and accessibility.**

- Protected-class enumerations per jurisdiction in `local_definitions.protected_classes` ground the pack in the specific protected-class list of the jurisdiction's anti-discrimination statute.

**GOVERN 4 — Team competencies.**

- Pack authors are identified by the `signers` tuple; each signer role maps to a documented competency profile.

**GOVERN 5 — Engagement with stakeholders.**

- `user_notice.opt_out_allowed` records whether the pack supports user opt-out in the jurisdiction. The appeal flow provides a structured stakeholder-feedback channel without violating privacy.

**GOVERN 6 — Third-party risk.**

- Every external lexicon reference (`local_language_assets.lexicons`) declares `provenance`. `anti_misuse.assert_lexicons_have_provenance` rejects unattributed lexicons.

---

## 2. Map (AI RMF § 5.2)

**MAP 1 — Context established and understood.**

- The compiled prompt (`kchat-skills/prompts/compiled_prompt_format.md`) captures the full decision context (`[GLOBAL_BASELINE]`, `[JURISDICTION_OVERLAY]`, `[COMMUNITY_OVERLAY]`, `[INPUT]`, `[OUTPUT]`).

**MAP 2 — Categorization of the AI system.**

- 17-category closed-enum taxonomy in `kchat-skills/global/taxonomy.yaml`. `anti_misuse.assert_no_vague_categories` refuses out-of-range categories; `anti_misuse.assert_no_invented_categories` refuses overlay-local taxonomies.

**MAP 3 — AI capabilities, targeted usage, goals, and expected benefits.**

- Per-pack `purpose` fields declare intent; the compiled prompt's `[INSTRUCTION]` section is the operative capability description.

**MAP 4 — Risks and benefits mapped for all components.**

- Each jurisdiction overlay's docstring names the legal frameworks it encodes (e.g. "LGPNNA child protection, Ley Federal contra la Delincuencia Organizada").

**MAP 5 — Impacts to individuals, groups, communities, organizations, and society are characterized.**

- `allowed_contexts` (QUOTED_SPEECH_CONTEXT, NEWS_CONTEXT, EDUCATION_CONTEXT, COUNTERSPEECH_CONTEXT) enumerate the counter-balancing protected expressions. `anti_misuse.assert_protected_contexts_for_strict_floors` mandates all four are declared for any strict floor.

---

## 3. Measure (AI RMF § 5.3)

**MEASURE 1 — Appropriate methods and metrics are identified and applied.**

- `build-tools/compiler/metric_validator.py` declares and enforces the numeric targets. The suite template (`kchat-skills/tests/test_suite_template.yaml`) pins every metric and threshold.

**MEASURE 2 — Trustworthy characteristics evaluated.**

- The minority-language false-positive corpus (`test_minority_language_fp.py`) and the adversarial corpus (`test_adversarial_corpus.py`) evaluate fairness and robustness respectively.

**MEASURE 3 — Mechanisms for tracking identified risks are in place.**

- The appeal flow (`appeal_flow.py`) aggregates per-category appeal rates on-device.

**MEASURE 4 — Feedback about efficacy is gathered and assessed.**

- `AppealAggregator.aggregate` emits `AppealReport.recommendation` ∈ {`no_action`, `review_suggested`, `urgent_review`} — a structured efficacy signal.

---

## 4. Manage (AI RMF § 5.4)

**MANAGE 1 — Risks are prioritized and acted on based on impact.**

- Severity floors per category are the primary prioritization signal. `severity_floor: 5` triggers `critical_intervention`; `severity_floor: 4` triggers `strong_warn`.

**MANAGE 2 — Strategies to maximize benefits and minimize negative impacts.**

- Community overlays (`kchat-skills/communities/*.yaml`) let host applications opt into tighter constraints per context (`school`, `family`, `workplace`, `adult_only`, `marketplace`, `health_support`, `political`, `gaming`) without changing the jurisdiction floor.

**MANAGE 3 — Third-party risk is regularly monitored and managed.**

- Lexicon provenance is re-verified every review cycle; expiry dates force re-signing.

**MANAGE 4 — Risk treatments are documented, monitored, and communicated.**

- `skill_passport.py` emits a signed passport for each pack version; host applications surface `visible_pack_summary` to end-users.

---

## Performance benchmarking (AI RMF MEASURE 1)

The benchmark harness (`build-tools/compiler/benchmark.py`) provides a
structured `PipelineBenchmark` that measures p50/p95/p99/mean/max
latency against the full pack set, and a `BenchmarkReport` that
records whether the `p95 <= 250ms` target is met. The target
mirrors the NIST AI RMF MEASURE 1 expectation that performance
metrics are identified and continuously applied.

---

## Source citations

- NIST AI 100-1, Artificial Intelligence Risk Management Framework (AI RMF 1.0), January 2023.
- `kchat-skills/global/baseline.yaml`
- `build-tools/compiler/anti_misuse.py`
- `build-tools/compiler/bias_audit.py`
- `build-tools/compiler/benchmark.py`
- `build-tools/compiler/appeal_flow.py`
- `build-tools/compiler/metric_validator.py`
- `build-tools/compiler/skill_passport.py`

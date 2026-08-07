# Regulatory Alignment Index

This directory collects the concrete mappings between each
regulatory instrument that applies to the KChat on-device guardrail
system and the specific artefacts (YAML overlays, compiler modules,
tests, docs) that satisfy each obligation.

All three alignment documents follow the same structure:

1. A table per obligation family mapping requirements to artefacts.
2. Named `kchat-skills/` file references on the right-hand column.
3. A "gaps and mitigations" section calling out what is out of scope
   for an on-device layer and how the host application closes the gap.

## Index

| Instrument | Document | Scope |
| --- | --- | --- |
| EU Digital Services Act (Regulation (EU) 2022/2065) | [`eu_dsa_alignment.md`](./eu_dsa_alignment.md) | Transparency, notice-and-action, risk assessment, protection of minors, transparency reporting. |
| NIST AI Risk Management Framework 1.0 (NIST AI 100-1) | [`nist_ai_rmf_alignment.md`](./nist_ai_rmf_alignment.md) | Govern, Map, Measure, Manage — plus trustworthy-AI characteristics (validity, safety, privacy, fairness, etc.). |
| UNICEF / ITU Child Online Protection Guidelines (2020) | [`unicef_itu_cop_alignment.md`](./unicef_itu_cop_alignment.md) | Child-rights due diligence, safer environments, privacy by design, crisis resources, per-jurisdiction statutory grounding for 59 packs. |

## How to use these documents

- **Legal/compliance review:** every obligation has a cell pointing to
  a file path. Follow the path to see the exact YAML or Python code
  that satisfies the obligation.
- **Pack authors:** when adding a new jurisdiction overlay, cross-reference the
  UNICEF/ITU document's per-jurisdiction table to ensure the child-safety
  statute is cited in the overlay's header docstring.
- **Auditors:** `build-tools/tests/test_regulatory_docs.py`
  verifies each of these documents exists, is non-empty, and references
  the named source artefacts.

## Source artefacts referenced across this directory

- `kchat-skills/global/baseline.yaml`
- `build-tools/compiler/anti_misuse.py`
- `build-tools/compiler/appeal_flow.py`
- `build-tools/compiler/benchmark.py`
- `build-tools/compiler/bias_audit.py`
- `build-tools/compiler/metric_validator.py`
- `build-tools/compiler/skill_passport.py`
- `kchat-skills/tests/adversarial/corpus.yaml`
- `kchat-skills/tests/jurisdictions/test_minority_language_fp.py`

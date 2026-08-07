# EU Digital Services Act (DSA) — Alignment Mapping

This document maps each material obligation of Regulation (EU) 2022/2065
(the Digital Services Act, "DSA") to the concrete artefact of the KChat
on-device guardrail skill-pack system that satisfies it. Where a DSA
obligation is partially or indirectly addressed, the gap and
mitigation are noted.

The DSA sits alongside the GDPR (Regulation (EU) 2016/679), the
ePrivacy Directive (2002/58/EC), and — for child-safety content —
the proposed CSAM Regulation. Nothing in this skill-pack system
overrides or relaxes any of those instruments; it is a
**minimum-safety layer** run on-device, under user control.

---

## 1. Transparency of content moderation (DSA Art. 14 — Terms and Conditions, Art. 17 — Statement of Reasons)

| Obligation | Artefact |
| --- | --- |
| Providers must set out in their terms and conditions information on any restrictions imposed in relation to information provided by users. | Every skill pack is human-readable YAML (`kchat-skills/global/baseline.yaml`, `kchat-skills/jurisdictions/<cc>/overlay.yaml`, `kchat-skills/communities/*.yaml`) with explicit `severity_floor`, `allowed_contexts`, and `user_notice.visible_pack_summary` fields. The user-facing summary is surfaced in UI via `user_notice.visible_pack_summary`. |
| The information shall be set out in clear, plain, intelligible, user-friendly language. | `user_notice.visible_pack_summary` is constrained to plain-language prose. The compiled-prompt bundles under `kchat-skills/prompts/compiled_examples/` expose the exact rules that will apply, section by section (`[GLOBAL_BASELINE]`, `[JURISDICTION_OVERLAY]`, `[COMMUNITY_OVERLAY]`). |
| Providers must include a statement of reasons when they impose a restriction. | The pipeline's per-message output (`kchat-skills/global/output_schema.json`) carries a `rationale_id` that references a stable catalogue entry; the UI renders the catalogue entry plus the pack's `visible_pack_summary`. |

**Source artefacts:**
- `kchat-skills/global/baseline.yaml` — single source of truth for the 16 taxonomy categories.
- `kchat-skills/jurisdictions/_template/overlay.yaml` — template every regional overlay follows.
- `build-tools/compiler/skill_compiler.py` — produces the compiled prompt surfaced to users.

---

## 2. Notice-and-action / complaint handling (DSA Art. 16, 20)

| Obligation | Artefact |
| --- | --- |
| Providers must put in place mechanisms to allow any individual or entity to notify them of the presence on their service of specific items of information that the individual or entity considers to be illegal content. | Per-pack `user_notice.appeal_resource_id` points to a stable identifier the host application must resolve to a notice-and-action endpoint. The skill-pack system itself stays on-device; notice-and-action is the host application's responsibility, but the pack guarantees a stable anchor. |
| Internal complaint-handling systems. | `build-tools/compiler/appeal_flow.py` (`AppealCase`, `AppealAggregator`, `AppealReport`) captures appeals on-device with a closed enum of `user_context` rationales (`disagree_category`, `disagree_severity`, `false_positive`, `missing_context`). Content text is never uploaded — the privacy contract is enforced by design. |

**Source artefacts:**
- `build-tools/compiler/appeal_flow.py` — on-device appeal-flow spec.
- `build-tools/tests/test_appeal_flow.py` — contract tests pinning the privacy invariant.

---

## 3. Risk assessment and mitigation (DSA Art. 34, 35 — very large online platforms)

| Obligation | Artefact |
| --- | --- |
| VLOPs must diligently identify, analyse and assess any systemic risks. | The baseline pack enumerates the full 17-category taxonomy; jurisdiction overlays encode region-specific systemic risks (election integrity, lèse-majesté, terrorism apologia, etc.). Every overlay has a documented legal reference in its header docstring. |
| Mitigation measures must be reasonable, proportionate and effective. | `build-tools/compiler/anti_misuse.py` refuses to sign any overlay that relaxes the child-safety floor, privacy rules, or the closed 0..16 category enum — the bright-line guarantees that mitigation never weakens baseline safety. |
| Providers must carry out risk assessments at least once a year. | `expires_on` on every pack is capped at 18 months by the compiler (see `_country_pack_assertions.assert_expiry_within_18_months`). Packs become unsigned after expiry, forcing re-review. |

**Source artefacts:**
- `build-tools/compiler/anti_misuse.py` — bright-line validators.
- `build-tools/compiler/skill_passport.py` — record of review, signature and expiry.
- `build-tools/compiler/bias_audit.py` — periodic bias auditing per minority-language target.

---

## 4. Protection of minors (DSA Art. 28)

| Obligation | Artefact |
| --- | --- |
| Providers of online platforms accessible to minors shall put in place appropriate and proportionate measures to ensure a high level of privacy, safety, and security of minors. | `kchat-skills/global/baseline.yaml` `child_safety_policy` declares category 1 at `severity_floor: 5` and the baseline prohibits any overlay from lowering it. Every jurisdiction overlay re-affirms category 1 floor 5 (enforced by `assert_no_relaxed_child_safety`). |
| No profiling of minors for targeted advertising. | The privacy contract forbids embedding device-held PII in telemetry; the pack system has no advertising surface. |

**Source artefacts:**
- `kchat-skills/global/baseline.yaml` — child-safety invariants.
- `kchat-skills/docs/regulatory/unicef_itu_cop_alignment.md` — COP alignment detail.

---

## 5. Transparency reporting (DSA Art. 24)

| Obligation | Artefact |
| --- | --- |
| Providers shall publish at least once a year reports on any content moderation that they engaged in. | `build-tools/compiler/metric_validator.py` + `kchat-skills/tests/test_suite_template.yaml` pin the target metrics (`minority_language_false_positive <= 0.07`, etc.) and the CI runs them on every change. Aggregated, device-local appeal data from `appeal_flow.py` produces the structured input for such a report without any message content leaving the device. |

**Source artefacts:**
- `build-tools/compiler/metric_validator.py`
- `kchat-skills/tests/test_suite_template.yaml`

---

## 6. Known gaps and mitigations

| Gap | Mitigation |
| --- | --- |
| Out-of-court dispute settlement (Art. 21) is out of scope for an on-device safety layer. | `user_notice.appeal_resource_id` provides the pointer host applications use to route to their Art. 21 workflow. |
| Trusted flaggers (Art. 22) rely on host-application infrastructure. | The skill-pack signing workflow (`skill_passport.py`) accepts trusted-flagger signatures as an additional reviewer; the schema already permits arbitrary `signers` entries. |
| Researchers' data access (Art. 40) is fundamentally incompatible with a privacy-preserving, on-device layer that does not store message content. | Aggregated, per-rationale appeal counts from `appeal_flow.py` may be exported with opt-in only; no text leaves the device. |

---

## Source citations

- Regulation (EU) 2022/2065 of the European Parliament and of the Council
  of 19 October 2022 on a Single Market For Digital Services ("DSA").
- `kchat-skills/global/baseline.yaml`
- `build-tools/compiler/anti_misuse.py`
- `build-tools/compiler/appeal_flow.py`
- `build-tools/compiler/skill_passport.py`

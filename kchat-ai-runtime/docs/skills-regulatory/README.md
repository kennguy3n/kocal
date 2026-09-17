# Documentation Index — Regulatory Alignment

> **Provenance note:** These documents were imported from the `kchat-skills`
> source repository and map regulatory obligations onto that repo's artefact
> paths (`kchat-skills/`, `build-tools/compiler/`, `tools/`). In this
> repository the equivalent runtime artefacts live under
> `crates/kchat-safety/src/skillpack/` (embedded data: communities,
> jurisdictions, prompts, lexicons) and `eval/kchat-task-suite/datasets/`.

Documentation that lives alongside the skill packs and the compiler.

## Regulatory Alignment

[`regulatory/`](regulatory/) — maps each obligation of the EU
Digital Services Act, the NIST AI Risk Management Framework 1.0,
and the UNICEF / ITU Child Online Protection Guidelines onto the
concrete artefacts in this repository.

- [`regulatory/README.md`](regulatory/README.md) — index of the
  three alignment documents.
- [`regulatory/eu_dsa_alignment.md`](regulatory/eu_dsa_alignment.md)
  — EU DSA obligations (transparency, notice-and-action, risk
  assessment, protection of minors, transparency reporting).
- [`regulatory/nist_ai_rmf_alignment.md`](regulatory/nist_ai_rmf_alignment.md)
  — NIST AI RMF 1.0 core functions (Govern, Map, Measure, Manage)
  and the seven trustworthy-AI characteristics.
- [`regulatory/unicef_itu_cop_alignment.md`](regulatory/unicef_itu_cop_alignment.md)
  — UNICEF / ITU child-rights due diligence plus per-jurisdiction
  statutory grounding for all 59 country packs.

## Top-Level Project Documents

The canonical project documents live at the repository root:

- [`../../README.md`](../../README.md) — project overview, quick
  start, and documentation map.
- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) — technical
  reference: crate layering, four-plane design, data flow.
- [`../../MODEL.md`](../../MODEL.md) — model registry, device
  profiles, memory budgets, selection logic.
- [`../../AGENTS.md`](../../AGENTS.md) — build/test commands and
  workspace conventions.

The kchat-skills-side references (`docs/COMPILER.md`,
`docs/MODEL_LIFECYCLE.md`, `docs/RUNNING_XLMR.md`,
`docs/SUPPORTED_REGIONS.md`, `docs/CHANGELOG.md`) are not present in
this repository — see the source repo for the skill-pack compiler,
signing, lifecycle, and regional-roster documentation.

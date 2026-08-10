# archetype-strict-hate — lexicons

Per-language lexicon bundles for the
`kchat.jurisdiction.archetype-strict-hate.guardrail.v1` overlay.

Each lexicon file declares:

- `lexicon_id` — globally unique, referenced from `overlay.yaml`
  under `local_language_assets.lexicons`.
- `language` — IETF BCP 47 code.
- `categories` — taxonomy ids (see `kchat-skills/global/taxonomy.yaml`).
- `provenance` — reviewer / source.

The archetype ships with English placeholder lexicons:

- `archetype_strict_hate_extremism_v1` — category 4 (EXTREMISM).
- `archetype_strict_hate_hate_v1`      — category 6 (HATE).

Concrete country packs copy this structure and add per-language
lexicons under their own jurisdiction directory, including
minority-language and code-switching false-positive corpora.

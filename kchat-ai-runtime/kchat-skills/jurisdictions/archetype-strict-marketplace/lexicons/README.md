# archetype-strict-marketplace — lexicons

Per-language lexicon bundles for the
`kchat.jurisdiction.archetype-strict-marketplace.guardrail.v1` overlay.

Each lexicon file declares:

- `lexicon_id` — globally unique, referenced from `overlay.yaml`
  under `local_language_assets.lexicons`.
- `language` — IETF BCP 47 code.
- `categories` — taxonomy ids (see `kchat-skills/global/taxonomy.yaml`).
- `provenance` — reviewer / source.

The archetype ships with English placeholder lexicons:

- `archetype_strict_marketplace_drugs_weapons_v1` — category 11
  (DRUGS_WEAPONS).
- `archetype_strict_marketplace_illegal_goods_v1` — category 12
  (ILLEGAL_GOODS).

Concrete country packs copy this structure and add per-language
lexicons under their own jurisdiction directory, including
minority-language and code-switching false-positive corpora.

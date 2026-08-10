# archetype-strict-adult — lexicons

Per-language lexicon bundles for the
`kchat.jurisdiction.archetype-strict-adult.guardrail.v1` overlay.

Each lexicon file in this directory declares:

- `lexicon_id` — globally unique, referenced from `overlay.yaml`
  under `local_language_assets.lexicons`.
- `language` — IETF BCP 47 code.
- `categories` — taxonomy ids (see `kchat-skills/global/taxonomy.yaml`).
- `provenance` — reviewer / source for the lexicon entries.

Because this is an archetype, only the English placeholder lexicon
`archetype_strict_adult_lexicon_v1` is declared. Concrete country
packs copy this structure and add per-language bundles under their
own jurisdiction directory.

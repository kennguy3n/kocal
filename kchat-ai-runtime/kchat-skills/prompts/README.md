# Policy manifests, not LLM prompts

> **TL;DR:** Files in this directory are **policy manifests** —
> structured, reviewer-readable records of the compiled skill bundle.
> They are **not** generative-model system prompts. The on-device
> encoder backend (XLM-R classifier head, see
> `build-tools/compiler/xlmr_adapter.py`) is a frozen classifier;
> nothing in the runtime ever consumes these files as an LLM prompt.

## Why the name "prompt" persists

The format here was originally drafted when the planned backend was
a small generative classifier, and the per-section header tokens
(`[INSTRUCTION]`, `[GLOBAL_BASELINE]`, `[JURISDICTION_OVERLAY]`,
`[COMMUNITY_OVERLAY]`, `[INPUT]`, `[OUTPUT]`) were sized so the
artefact could be fed verbatim into a chat-style model. The backend
was later refactored to the encoder-classifier path described in
`ARCHITECTURE.md` §"Hybrid Local Pipeline", and the artefact's role
collapsed to two purposes:

1. **Auditor-readable record.** The compiled bundle is the single
   place where reviewers, legal, cultural reviewers, and Trust &
   Safety can read the union of the global baseline, the
   jurisdiction overlay, and the community overlay that will be
   active on a device. The 10-rule `[INSTRUCTION]` block is a
   human-readable policy summary — every numeric / structural
   constraint it states is also enforced in code (encoder head,
   deterministic detectors, `threshold_policy.py`).

2. **Compiler validation input.** The compiler
   (`build-tools/compiler/skill_compiler.py`) consumes the manifest's
   structure during merge to assert no overlay reintroduces a
   reserved privacy rule, no jurisdiction overlay relaxes the
   child-safety floor, etc.

The file names retain `compiled_prompt_format.md` because public
tests pin the filename. New code, docstrings, and ARCHITECTURE.md
sections use the **policy manifest** terminology (see P1-2 in the
hardening notes); the Python entry point is
`compile_policy_manifest()` and `compile_prompt()` is an alias kept
for backwards compatibility.

## File layout

| File | Purpose |
| --- | --- |
| `runtime_instruction.txt` | 10-rule policy record. Human-readable; encoder backend does **not** consume it as an LLM prompt. |
| `compiled_prompt_format.md` | Format spec for the assembled policy manifest. Read by reviewers and by `tests/global/test_prompts.py`. Despite the historic filename, it documents a policy manifest, not a generative prompt. |
| `compiled_examples/` | Example manifests for the regression tests in `tests/global/test_compiled_examples.py`. |

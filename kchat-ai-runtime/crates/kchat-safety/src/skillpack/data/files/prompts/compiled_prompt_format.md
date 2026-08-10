# Compiled Prompt Format

The on-device guardrail compiler takes the **active skill bundle** —
global baseline + jurisdiction overlays + community overlay + runtime
context — and emits a single compact text prompt that records the
classifier-bundle configuration. The compiled prompt is pinned to
the instruction budget (**< 1800 tokens**, output budget **< 600
tokens**) so the bundle stays compatible with future
generative-classifier backends. The reference XLM-R encoder
consumes the bundle directly (no chat-completion call,
deterministic argmax over fixed prototypes — "temperature **0.0**"
is kept in the schema for backward compatibility but is not used by
the encoder). See ARCHITECTURE.md.

## Sections

Every compiled prompt is composed of six sections, in this exact order:

### `[INSTRUCTION]`
The runtime classifier-bundle instruction (the 10-rule block from
[`runtime_instruction.txt`](./runtime_instruction.txt)). Always
present, byte-for-byte identical, never paraphrased. The encoder
backend does not consume this block at inference time, but the
compiler still pins it into the bundle as a human- and
reviewer-readable record of the classifier's allowed actions.

### `[GLOBAL_BASELINE]`
Compact summary of the global baseline:

- `taxonomy: 17-category v1`
- `severity: 0..5 v1`
- `privacy_rules: v1 (immutable)`
- `output_schema: kchat.guardrail.output.v1`
- `thresholds: label_only=0.45 warn=0.62 strong_warn=0.78 critical=0.85`

### `[JURISDICTION_OVERLAY]`
Empty if no jurisdiction overlay is active. When present, lists the
overlay id and the per-category `severity_floor` overrides plus the
allowed protected-context reason codes.

### `[COMMUNITY_OVERLAY]`
Empty if no community overlay is active. When present, lists the
overlay id, declared `age_mode`, per-category `action`s, and any
device-local expiring counters configured by the overlay.

### `[INPUT]`
A single instance of the structured input contract defined by
[`local_signal_schema.json`](../global/local_signal_schema.json) — the
`message`, `context`, `local_signals`, and `constraints` blocks for
the message currently being evaluated.

### `[OUTPUT]`
Reserved for the encoder classifier. The model emits exactly one JSON
object conforming to `kchat.guardrail.output.v1`. The runtime rejects
any output that does not validate.

## Budget

| Section                    | Typical tokens |
| -------------------------- | -------------- |
| `[INSTRUCTION]`            | ~180           |
| `[GLOBAL_BASELINE]`        | ~80            |
| `[JURISDICTION_OVERLAY]`   | ~120 (when present) |
| `[COMMUNITY_OVERLAY]`      | ~120 (when present) |
| `[INPUT]`                  | ~600 (varies with message) |
| `[OUTPUT]`                 | ≤ 600 (model output) |

Total compiled instruction budget: **< 1800 tokens**. Output budget:
**< 600 tokens**. Temperature: **0.0**.

## Examples

See [`compiled_examples/`](./compiled_examples/) for ready-to-feed
compiled prompts. The current reference example is:

- [`strict_marketplace_workplace.txt`](./compiled_examples/strict_marketplace_workplace.txt)
  — workplace community overlay over the strict-marketplace
  jurisdiction archetype.

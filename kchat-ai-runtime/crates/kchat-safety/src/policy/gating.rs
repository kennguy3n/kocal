//! Single-sourced derivation of the FFI / napi `(allow_reveal,
//! allow_forward)` gating bits.
//!
//! Both binding crates (`kchat-safety-ffi`, `kchat-safety-napi`)
//! project a core [`Verdict`] onto a smaller language-native
//! shape at the `classify_text` boundary. The boolean gating
//! bits the renderer needs (`allow_reveal`, `allow_forward`)
//! must agree byte-for-byte across both bindings, and must also
//! agree with `policy_decide`'s rubric-based gating for any
//! given severity — otherwise a host running both surfaces
//! against the same skill pack sees contradictory UX gating.
//!
//! Historically the two binding crates each carried a copy of
//! the action-flag derivation, and `classify_text` diverged
//! from `policy_decide` at severities 3 and 4 (warn /
//! strong_warn — the rubric defaults forbid forward / reveal
//! respectively, but the old derivation permitted both). The
//! gating routing was then unified through the loaded skill
//! pack's [`SeverityRubric`], but each binding crate still
//! carried its own copy of the routing function. This module
//! collapses both copies onto a single core implementation so
//! the two bindings physically cannot drift.
//!
//! ## Contract
//!
//! [`derive_gating`] is the canonical projection both bindings
//! call. The semantics:
//!
//! 1. **Rubric path (`rubric = Some(_)`).** The skill pack's
//!    [`SeverityRubric`] is consulted via
//!    [`SeverityMapper::disposition`]. Whatever
//!    `(allow_reveal, allow_forward)` the rubric assigns to the
//!    verdict's severity is the answer. This matches
//!    `policy_decide`'s behaviour exactly.
//! 2. **Defensive fallback (`rubric = Some(_)` + lookup error).**
//!    `mapper.disposition` can only error if the rubric was
//!    constructed bypassing validation OR the severity is `> 5`.
//!    Both are schema-impossible through the public API (skill
//!    pack loader rejects an incomplete rubric; the
//!    `output_schema.json` constrains severity to `0..=5`); the
//!    function still falls back to the action-flag derivation
//!    instead of panicking, so an out-of-band combination
//!    degrades gracefully.
//! 3. **No-rubric path (`rubric = None`).** Fall back to the
//!    action-flag derivation —
//!    `critical_intervention` blocks both reveal and forward,
//!    `strong_warn` permits reveal but blocks forward, `warn`
//!    permits both, and `label_only` / `suggest_redact` are
//!    non-gating decorations. This preserves the pre-PR2
//!    behaviour for callers that classify before a pack is
//!    loaded (the only place that path runs in production is
//!    the `attach_onnx_encoder` happy-path tests).
//!
//! The function is intentionally pure (`&Verdict +
//! Option<&SeverityRubric> → (bool, bool)`); no allocation, no
//! interior mutability, no I/O. It's safe to call from any
//! binding context.
//!
//! [`Verdict`]: super::Verdict
//! [`SeverityRubric`]: crate::policy_interpreter::SeverityRubric
//! [`SeverityMapper::disposition`]: crate::policy_interpreter::SeverityMapper::disposition

use crate::policy::Verdict;
use crate::policy_interpreter::{SeverityMapper, SeverityRubric};

/// Compute the FFI / napi gating bits `(allow_reveal,
/// allow_forward)` from a core [`Verdict`], preferring the
/// loaded skill pack's [`SeverityRubric`] over the action-flag
/// fallback when one is supplied.
///
/// See the module-level doc for the full contract, fallback
/// semantics, and rationale for collapsing the two bindings'
/// previous copies onto this single core function.
///
/// [`Verdict`]: super::Verdict
/// [`SeverityRubric`]: crate::policy_interpreter::SeverityRubric
pub fn derive_gating(verdict: &Verdict, rubric: Option<&SeverityRubric>) -> (bool, bool) {
    if let Some(rubric) = rubric {
        let mapper = SeverityMapper::new(rubric);
        if let Ok(disposition) = mapper.disposition(verdict.severity) {
            return (disposition.allow_reveal, disposition.allow_forward);
        }
        // Defensive fallback: rubric loaded but
        // `disposition()` returned an out-of-range error. The
        // schema and skill-pack loader make this unreachable
        // through the public API; fall back to action-flag
        // derivation rather than panic so an out-of-band call
        // still produces a deterministic answer.
    }
    action_flag_gating(verdict)
}

/// Action-flag-only gating derivation. The no-rubric fallback
/// path of [`derive_gating`] — kept as a separate function so
/// the two code paths are easy to read.
///
/// `critical_intervention` blocks reveal AND forward;
/// `strong_warn` permits reveal but blocks forward; `warn`
/// permits both. `label_only` and `suggest_redact` are
/// non-gating decorations.
fn action_flag_gating(verdict: &Verdict) -> (bool, bool) {
    let allow_reveal = !verdict.actions.critical_intervention;
    let allow_forward = !(verdict.actions.strong_warn || verdict.actions.critical_intervention);
    (allow_reveal, allow_forward)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Actions, Verdict};
    use crate::policy_interpreter::severity::default_rubric;

    fn verdict_at(severity: u8, actions: Actions) -> Verdict {
        Verdict {
            severity,
            category: 0,
            confidence: 0.5,
            actions,
            reason_codes: Vec::new(),
            rationale_id: String::from("test"),
            resource_link_id: None,
            counter_updates: None,
            model_health: None,
        }
    }

    #[test]
    fn no_rubric_critical_intervention_blocks_both() {
        let verdict = verdict_at(
            5,
            Actions {
                critical_intervention: true,
                ..Actions::blank()
            },
        );
        assert_eq!(derive_gating(&verdict, None), (false, false));
    }

    #[test]
    fn no_rubric_strong_warn_permits_reveal_blocks_forward() {
        let verdict = verdict_at(
            4,
            Actions {
                strong_warn: true,
                ..Actions::blank()
            },
        );
        assert_eq!(derive_gating(&verdict, None), (true, false));
    }

    #[test]
    fn no_rubric_warn_permits_both() {
        let verdict = verdict_at(
            3,
            Actions {
                warn: true,
                ..Actions::blank()
            },
        );
        assert_eq!(derive_gating(&verdict, None), (true, true));
    }

    #[test]
    fn no_rubric_blank_permits_both() {
        let verdict = verdict_at(0, Actions::blank());
        assert_eq!(derive_gating(&verdict, None), (true, true));
    }

    #[test]
    fn rubric_severity_3_blocks_forward() {
        // Canonical rubric defaults forbid forward at severity
        // 3 (warn band), regardless of action flags. The rubric
        // is the source of truth — even a verdict with no
        // action flags set still gets `allow_forward = false`.
        let rubric = default_rubric();
        let verdict = verdict_at(3, Actions::blank());
        assert_eq!(derive_gating(&verdict, Some(&rubric)), (true, false));
    }

    #[test]
    fn rubric_severity_4_blocks_reveal_and_forward() {
        let rubric = default_rubric();
        let verdict = verdict_at(4, Actions::blank());
        assert_eq!(derive_gating(&verdict, Some(&rubric)), (false, false));
    }

    #[test]
    fn rubric_severity_5_blocks_reveal_and_forward() {
        let rubric = default_rubric();
        let verdict = verdict_at(5, Actions::blank());
        assert_eq!(derive_gating(&verdict, Some(&rubric)), (false, false));
    }

    #[test]
    fn rubric_severity_0_permits_both() {
        let rubric = default_rubric();
        let verdict = verdict_at(0, Actions::blank());
        assert_eq!(derive_gating(&verdict, Some(&rubric)), (true, true));
    }

    #[test]
    fn rubric_overrides_action_flags() {
        // Even if the action flags would say "block both" (via
        // critical_intervention), the rubric for severity 1
        // permits both. This proves the rubric path is taken
        // when one is supplied — the action-flag fallback never
        // executes.
        let rubric = default_rubric();
        let verdict = verdict_at(
            1,
            Actions {
                critical_intervention: true,
                ..Actions::blank()
            },
        );
        assert_eq!(derive_gating(&verdict, Some(&rubric)), (true, true));
    }
}

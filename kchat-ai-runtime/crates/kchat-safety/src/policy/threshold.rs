//! [`ThresholdPolicy`] — hard-coded decision policy enforcer.
//!
//! Port of `build-tools/compiler/threshold_policy.py`. The four
//! canonical confidence thresholds (`label_only=0.45`,
//! `warn=0.62`, `strong_warn=0.78`,
//! `critical_intervention=0.85`) are **immutable for signed
//! packs** — the encoder classifier cannot override them.
//! [`ThresholdPolicy::apply`] re-coerces any classifier output
//! whose asserted action set is inconsistent with its confidence
//! before the verdict leaves the device.
//!
//! Child-safety floor: any positive `CHILD_SAFETY` signal at
//! confidence `>= 0.45` is pinned to severity 5 with
//! `critical_intervention=true` per ARCHITECTURE.md line 373.
//!
//! Protected-speech demotion: when `reason_codes` carries one of
//! `NEWS_CONTEXT` / `EDUCATION_CONTEXT` / `COUNTERSPEECH_CONTEXT`
//! / `QUOTED_SPEECH_CONTEXT` (forwarded from
//! `local_signals.context_hints` by the classifier) and the
//! verdict is non-SAFE, the output is demoted to SAFE —
//! protecting news coverage, education, and counterspeech from
//! false positives. A CHILD_SAFETY verdict at confidence `>= 0.45`
//! is fully handled by the child-safety floor (Rule 1) and never
//! reaches the protected-speech rule. A CHILD_SAFETY verdict at
//! `confidence < 0.45` DOES fall through here — matching the
//! Python reference — so a low-confidence child-safety signal
//! framed as news / education / quoted speech can still be
//! demoted. Low-confidence CHILD_SAFETY without a protected
//! reason code is caught by the uncertainty cutoff (Rule 3).
//! Production callers that need the floor to win unconditionally
//! must gate on `confidence >= 0.45` upstream of the policy.
//!
//! Experimental thresholds (research / calibration only) are
//! supported via [`ThresholdPolicy::experimental`], which
//! constructs an instance with `is_experimental=true` and
//! validates the ordering invariant
//! `0 < label_only < warn < strong_warn < critical_intervention <= 1`
//! instead of the canonical-value check. Production code paths
//! (a future `GuardrailPipeline`) must refuse experimental
//! policies — the Rust API exposes the `is_experimental` flag so
//! a consumer can refuse them explicitly.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use super::verdict::{Actions, RawClassifierOutput, Verdict};

/// Canonical taxonomy id for `CHILD_SAFETY`. Kept in sync with
/// `files/global/taxonomy.yaml`.
pub const CHILD_SAFETY_CATEGORY: u32 = 1;

/// Canonical taxonomy id for the SAFE category.
pub const SAFE_CATEGORY: u32 = 0;

/// Minimum context-hint confidence required to fully demote a
/// non-SAFE verdict to SAFE under the protected-speech rule.
/// Below this floor the pipeline downgrades the verdict to
/// *warn-with-context* instead of wiping it entirely. High-
/// confidence CHILD_SAFETY never reaches this rule because the
/// child-safety floor (Rule 1) returns first.
pub const CONTEXT_DEMOTION_CONFIDENCE_THRESHOLD: f64 = 0.5;

/// Sentinel value returned when the
/// `context_hint_confidences` map is absent entirely. Pipelines
/// that did not forward confidences keep the legacy
/// *always-demote* behaviour for back-compat.
pub const DEFAULT_CONTEXT_CONFIDENCE_WHEN_MISSING: f64 = 1.0;

pub const CANONICAL_LABEL_ONLY: f64 = 0.45;
pub const CANONICAL_WARN: f64 = 0.62;
pub const CANONICAL_STRONG_WARN: f64 = 0.78;
pub const CANONICAL_CRITICAL_INTERVENTION: f64 = 0.85;

/// Reason codes that mark the message as protected speech. Any
/// non-SAFE verdict carrying one of these is demoted to SAFE by
/// [`ThresholdPolicy::apply`]. High-confidence CHILD_SAFETY (>=
/// `label_only`) is fully handled by the child-safety floor
/// (Rule 1) before this rule fires, so it cannot be demoted.
/// Low-confidence CHILD_SAFETY (< `label_only`) DOES fall
/// through to this rule — matching the Python reference —
/// because Rule 1's confidence guard does not trip.
///
/// Kept in sync with the `local_signal_schema.json` `context_hints`
/// enum. The four codes are: `NEWS_CONTEXT`, `EDUCATION_CONTEXT`,
/// `COUNTERSPEECH_CONTEXT`, `QUOTED_SPEECH_CONTEXT`.
pub const PROTECTED_SPEECH_REASON_CODES: [&str; 4] = [
    "COUNTERSPEECH_CONTEXT",
    "EDUCATION_CONTEXT",
    "NEWS_CONTEXT",
    "QUOTED_SPEECH_CONTEXT",
];

/// Errors returned by [`ThresholdPolicy::from_thresholds`] /
/// [`ThresholdPolicy::experimental`] when invariants are violated.
#[derive(Clone, Debug, PartialEq)]
pub enum ThresholdPolicyError {
    /// Production constructor was called with thresholds that
    /// differ from the canonical values. The signed-pack contract
    /// forbids this; use [`ThresholdPolicy::experimental`] for
    /// research / calibration code instead.
    NonCanonical {
        label_only: f64,
        warn: f64,
        strong_warn: f64,
        critical_intervention: f64,
    },
    /// Experimental constructor was called with a threshold
    /// outside the legal `(0.0, 1.0]` range.
    OutOfRange { name: &'static str, value: f64 },
    /// Experimental constructor was called with thresholds that
    /// do not satisfy the strict-increase invariant
    /// `label_only < warn < strong_warn < critical_intervention`.
    /// The `apply()` `elif` chain requires strict ordering to
    /// emit deterministic severities at the threshold boundaries.
    NotStrictlyIncreasing {
        label_only: f64,
        warn: f64,
        strong_warn: f64,
        critical_intervention: f64,
    },
}

impl std::fmt::Display for ThresholdPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThresholdPolicyError::NonCanonical {
                label_only,
                warn,
                strong_warn,
                critical_intervention,
            } => write!(
                f,
                "ThresholdPolicy thresholds are hard-coded for signed \
                packs (label_only={label_only}, warn={warn}, \
                strong_warn={strong_warn}, \
                critical_intervention={critical_intervention}); use \
                ThresholdPolicy::experimental(...) for research / \
                calibration overrides"
            ),
            ThresholdPolicyError::OutOfRange { name, value } => write!(
                f,
                "experimental ThresholdPolicy.{name}={value} must lie in (0.0, 1.0]"
            ),
            ThresholdPolicyError::NotStrictlyIncreasing {
                label_only,
                warn,
                strong_warn,
                critical_intervention,
            } => write!(
                f,
                "experimental ThresholdPolicy thresholds must be \
                strictly increasing: label_only < warn < strong_warn \
                < critical_intervention; got {label_only} / {warn} / \
                {strong_warn} / {critical_intervention}"
            ),
        }
    }
}

impl std::error::Error for ThresholdPolicyError {}

/// Immutable decision-policy enforcer.
///
/// Construct via [`ThresholdPolicy::default`] for the canonical
/// signed-pack policy, or via [`ThresholdPolicy::experimental`]
/// for research / calibration. The struct is intentionally NOT
/// `Copy` so a consumer that accidentally pass-by-values it
/// doesn't lose the `is_experimental` provenance silently.
///
/// **Deserialization is validated.** `Deserialize` routes through
/// [`ThresholdPolicyRaw`] and dispatches on the
/// `is_experimental` flag: a non-experimental policy must carry
/// the canonical thresholds (signed-pack invariant), while an
/// experimental policy must satisfy the same ordering / range
/// invariants enforced by [`Self::experimental`]. A future
/// config-loading path (e.g. caching the policy in an audit
/// trail) therefore cannot smuggle a non-canonical
/// `is_experimental=false` policy past validation by going
/// through serde.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ThresholdPolicyRaw")]
pub struct ThresholdPolicy {
    label_only: f64,
    warn: f64,
    strong_warn: f64,
    critical_intervention: f64,
    is_experimental: bool,
}

/// Shadow type used for validated deserialization of
/// [`ThresholdPolicy`]. Field-for-field mirror that derives
/// `Deserialize` (with no `try_from`) so serde can populate it,
/// then routes through `TryFrom` to enforce the
/// signed-pack-canonical-or-strictly-increasing-experimental
/// invariants.
#[derive(Deserialize)]
struct ThresholdPolicyRaw {
    label_only: f64,
    warn: f64,
    strong_warn: f64,
    critical_intervention: f64,
    is_experimental: bool,
}

impl TryFrom<ThresholdPolicyRaw> for ThresholdPolicy {
    type Error = ThresholdPolicyError;

    fn try_from(raw: ThresholdPolicyRaw) -> Result<Self, Self::Error> {
        // Dispatch on `is_experimental`: production deserialization
        // must hit the same canonical-value gate as
        // `from_thresholds`; experimental deserialization must hit
        // the same ordering / range / NaN gate as `experimental`.
        if raw.is_experimental {
            Self::experimental(
                raw.label_only,
                raw.warn,
                raw.strong_warn,
                raw.critical_intervention,
            )
        } else {
            Self::from_thresholds(
                raw.label_only,
                raw.warn,
                raw.strong_warn,
                raw.critical_intervention,
            )
        }
    }
}

impl Default for ThresholdPolicy {
    /// The canonical, signed-pack-compatible policy.
    fn default() -> Self {
        Self {
            label_only: CANONICAL_LABEL_ONLY,
            warn: CANONICAL_WARN,
            strong_warn: CANONICAL_STRONG_WARN,
            critical_intervention: CANONICAL_CRITICAL_INTERVENTION,
            is_experimental: false,
        }
    }
}

impl ThresholdPolicy {
    /// Construct a canonical policy. Equivalent to
    /// [`ThresholdPolicy::default`] but reads more explicitly at
    /// call sites that pin the policy used by a pipeline.
    pub fn canonical() -> Self {
        Self::default()
    }

    /// Construct a production policy from explicit threshold
    /// values. Rejects anything other than the canonical four
    /// values — the signed-pack contract forbids smuggling a
    /// relaxed policy past review. For research / calibration use
    /// [`ThresholdPolicy::experimental`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`ThresholdPolicyError::NonCanonical`] when any of
    /// the four thresholds differs from the canonical value.
    pub fn from_thresholds(
        label_only: f64,
        warn: f64,
        strong_warn: f64,
        critical_intervention: f64,
    ) -> Result<Self, ThresholdPolicyError> {
        if label_only != CANONICAL_LABEL_ONLY
            || warn != CANONICAL_WARN
            || strong_warn != CANONICAL_STRONG_WARN
            || critical_intervention != CANONICAL_CRITICAL_INTERVENTION
        {
            return Err(ThresholdPolicyError::NonCanonical {
                label_only,
                warn,
                strong_warn,
                critical_intervention,
            });
        }
        Ok(Self::default())
    }

    /// Construct a research / calibration policy with custom
    /// thresholds. The resulting instance carries
    /// [`is_experimental`](Self::is_experimental) `= true` so
    /// production code paths can refuse it explicitly. The
    /// ordering invariant
    /// `0 < label_only < warn < strong_warn < critical_intervention <= 1`
    /// is validated; violators are returned as
    /// [`ThresholdPolicyError::OutOfRange`] or
    /// [`ThresholdPolicyError::NotStrictlyIncreasing`].
    ///
    /// Note: in the Python reference this method also fires a
    /// `ExperimentalPolicyWarning` to alert audit tooling. Rust
    /// has no warning channel, so production callers should
    /// inspect the [`is_experimental`](Self::is_experimental)
    /// flag and reject any policy whose flag is `true`. A
    /// future `GuardrailPipeline` analogue must enforce this
    /// gate by default.
    pub fn experimental(
        label_only: f64,
        warn: f64,
        strong_warn: f64,
        critical_intervention: f64,
    ) -> Result<Self, ThresholdPolicyError> {
        for (name, v) in [
            ("label_only", label_only),
            ("warn", warn),
            ("strong_warn", strong_warn),
            ("critical_intervention", critical_intervention),
        ] {
            // `v.is_nan()` short-circuits the range check —
            // NaN-vs-NaN comparisons are always false in
            // IEEE-754, so a NaN would pass `0.0 < v <= 1.0`
            // silently otherwise.
            if !(v.is_finite() && v > 0.0 && v <= 1.0) {
                return Err(ThresholdPolicyError::OutOfRange { name, value: v });
            }
        }
        if !(label_only < warn && warn < strong_warn && strong_warn < critical_intervention) {
            return Err(ThresholdPolicyError::NotStrictlyIncreasing {
                label_only,
                warn,
                strong_warn,
                critical_intervention,
            });
        }
        Ok(Self {
            label_only,
            warn,
            strong_warn,
            critical_intervention,
            is_experimental: true,
        })
    }

    pub fn label_only(&self) -> f64 {
        self.label_only
    }
    pub fn warn(&self) -> f64 {
        self.warn
    }
    pub fn strong_warn(&self) -> f64 {
        self.strong_warn
    }
    pub fn critical_intervention(&self) -> f64 {
        self.critical_intervention
    }
    pub fn is_experimental(&self) -> bool {
        self.is_experimental
    }

    /// Apply the four-rule decision-policy chain to a classifier
    /// output, returning a policy-enforced [`Verdict`].
    ///
    /// Rule order (per ARCHITECTURE.md):
    ///
    /// 1. **Child-safety floor.** A `CHILD_SAFETY` verdict with
    ///    `confidence >= label_only` is pinned to severity 5 and
    ///    `critical_intervention=true`. The reason code
    ///    `CHILD_SAFETY_FLOOR` is added. This rule wins over every
    ///    other rule — even a news quote of CSAM still surfaces
    ///    the floor.
    /// 2. **Protected-speech demotion.** A non-SAFE verdict
    ///    carrying one of
    ///    [`PROTECTED_SPEECH_REASON_CODES`] is demoted. If the
    ///    highest protected-hint confidence (looked up in
    ///    `raw.context_hint_confidences`) clears
    ///    [`CONTEXT_DEMOTION_CONFIDENCE_THRESHOLD`], the verdict
    ///    is fully demoted to SAFE with `rationale_id =
    ///    "safe_protected_speech_v1"`. Below the floor, the
    ///    category is preserved but the action set is downgraded
    ///    to *warn-with-context* (rationale
    ///    `"warn_low_confidence_context_v1"`, reason codes augmented
    ///    with `WARN_WITH_CONTEXT`). Non-protected reason codes
    ///    are dropped from the demoted output for review
    ///    traceability.
    /// 3. **Uncertainty handling.** A non-SAFE verdict with
    ///    `confidence < label_only` is coerced to SAFE (category
    ///    0, severity 0, blank actions, empty reason codes).
    /// 4. **Action re-derivation.** For non-SAFE verdicts the
    ///    confidence-driven action flags
    ///    (`label_only` / `warn` / `strong_warn` /
    ///    `critical_intervention`) are recomputed from the policy
    ///    thresholds and `confidence`. The `suggest_redact` flag
    ///    is preserved verbatim (content-type hint, not
    ///    confidence-driven). At most one of the four
    ///    confidence-driven flags is set.
    ///
    /// The `model_health` field is forwarded through every branch
    /// when present, so the UI can distinguish 'classifier produced
    /// SAFE' from 'classifier could not run, deterministic
    /// detectors only'.
    pub fn apply(&self, raw: &RawClassifierOutput) -> Verdict {
        let v = &raw.verdict;

        // Rule 1: Child-safety floor — wins over every other rule.
        if v.category == CHILD_SAFETY_CATEGORY && v.confidence >= self.label_only {
            let mut reasons: BTreeSet<String> = v.reason_codes.iter().cloned().collect();
            reasons.insert(String::from("CHILD_SAFETY_FLOOR"));
            return Verdict {
                severity: 5,
                category: v.category,
                confidence: v.confidence,
                actions: Actions {
                    critical_intervention: true,
                    ..Actions::blank()
                },
                reason_codes: reasons.into_iter().collect(),
                rationale_id: effective_rationale_id(&v.rationale_id),
                resource_link_id: v.resource_link_id.clone(),
                counter_updates: v.counter_updates.clone(),
                model_health: v.model_health,
            };
        }

        // Rule 2: Protected-speech demotion. Applies to every
        // non-SAFE category — including CHILD_SAFETY at
        // `confidence < label_only`, which did NOT trip Rule 1
        // and so falls through here. Matches the Python
        // reference's `category != SAFE_CATEGORY` guard exactly
        // (build-tools/compiler/threshold_policy.py).
        // High-confidence CHILD_SAFETY is fully handled by Rule
        // 1 above and cannot reach this branch.
        let protected_present: Vec<String> = v
            .reason_codes
            .iter()
            .filter(|r| PROTECTED_SPEECH_REASON_CODES.contains(&r.as_str()))
            .cloned()
            .collect();
        if v.category != SAFE_CATEGORY && !protected_present.is_empty() {
            let best =
                best_context_confidence(&protected_present, raw.context_hint_confidences.as_ref());
            if best >= CONTEXT_DEMOTION_CONFIDENCE_THRESHOLD {
                // Full demotion to SAFE. Drop all non-protected
                // reason codes for review traceability. `BTreeSet`
                // dedupes and sorts in one pass, matching the
                // Python oracle's `sorted(set(...))`.
                let dedup: BTreeSet<String> = protected_present.into_iter().collect();
                let reasons: Vec<String> = dedup.into_iter().collect();
                return Verdict {
                    severity: 0,
                    category: SAFE_CATEGORY,
                    confidence: v.confidence,
                    actions: Actions::blank(),
                    reason_codes: reasons,
                    rationale_id: String::from("safe_protected_speech_v1"),
                    resource_link_id: None,
                    counter_updates: None,
                    model_health: v.model_health,
                };
            }
            // Below the floor: keep the category but downgrade
            // the action set to at most `warn`. Add the
            // `WARN_WITH_CONTEXT` audit reason. `suggest_redact`
            // is preserved verbatim from the input.
            let suggest_redact = v.actions.suggest_redact;
            let warn_actions = Actions {
                warn: true,
                suggest_redact,
                ..Actions::blank()
            };
            let mut reasons: BTreeSet<String> = v.reason_codes.iter().cloned().collect();
            reasons.insert(String::from("WARN_WITH_CONTEXT"));
            return Verdict {
                severity: v.severity.min(2),
                category: v.category,
                confidence: v.confidence,
                actions: warn_actions,
                reason_codes: reasons.into_iter().collect(),
                rationale_id: String::from("warn_low_confidence_context_v1"),
                resource_link_id: None,
                counter_updates: None,
                model_health: v.model_health,
            };
        }

        // Rule 3: Uncertainty handling.
        if v.category != SAFE_CATEGORY && v.confidence < self.label_only {
            return Verdict {
                severity: 0,
                category: SAFE_CATEGORY,
                confidence: v.confidence,
                actions: Actions::blank(),
                reason_codes: Vec::new(),
                rationale_id: effective_rationale_id(&v.rationale_id),
                resource_link_id: None,
                counter_updates: None,
                model_health: v.model_health,
            };
        }

        // Rule 4: Re-derive action flags from confidence for
        // non-SAFE categories. SAFE category passes through with
        // its (already-blank or pinned) actions intact.
        if v.category != SAFE_CATEGORY {
            let suggest_redact = v.actions.suggest_redact;
            let mut actions = Actions::blank();
            if v.confidence >= self.critical_intervention {
                actions.critical_intervention = true;
            } else if v.confidence >= self.strong_warn {
                actions.strong_warn = true;
            } else if v.confidence >= self.warn {
                actions.warn = true;
            } else if v.confidence >= self.label_only {
                actions.label_only = true;
            }
            actions.suggest_redact = suggest_redact;
            return Verdict {
                severity: v.severity,
                category: v.category,
                confidence: v.confidence,
                actions,
                reason_codes: v.reason_codes.clone(),
                rationale_id: effective_rationale_id(&v.rationale_id),
                resource_link_id: v.resource_link_id.clone(),
                counter_updates: v.counter_updates.clone(),
                model_health: v.model_health,
            };
        }

        // SAFE category: pass through verbatim. The classifier
        // already emitted a SAFE verdict; the policy has nothing
        // to coerce. Copy through `model_health` /
        // `resource_link_id` / `counter_updates` so a SAFE
        // verdict with a `model_unavailable` health signal still
        // surfaces the deterministic-detector reason codes to
        // the UI.
        Verdict {
            severity: v.severity,
            category: v.category,
            confidence: v.confidence,
            actions: v.actions,
            reason_codes: v.reason_codes.clone(),
            rationale_id: effective_rationale_id(&v.rationale_id),
            resource_link_id: v.resource_link_id.clone(),
            counter_updates: v.counter_updates.clone(),
            model_health: v.model_health,
        }
    }

    /// Return the tie-break winner from `candidates`. The winner
    /// is the highest-severity candidate; ties on severity break
    /// in favour of the lower-numbered taxonomy category.
    ///
    /// # Errors
    ///
    /// Returns `None` when `candidates` is empty. Callers that
    /// need the historical "panic on empty" semantics from the
    /// Python `tie_break(...)` can `.expect("tie_break: candidates
    /// must be non-empty")` on the result.
    pub fn tie_break(candidates: &[Verdict]) -> Option<&Verdict> {
        candidates.iter().min_by(|a, b| {
            // Higher severity wins => negate severity for the
            // min sort. Tie-break on lower category number.
            let key_a = ((a.severity as i32).wrapping_neg(), a.category);
            let key_b = ((b.severity as i32).wrapping_neg(), b.category);
            key_a.cmp(&key_b)
        })
    }
}

/// Apply the `rationale_id or "safe_benign_v1"` fallback from the
/// Python `_deepcopy_output` helper. Every branch of `apply()`
/// that carries the input rationale_id through (rules 1 / 3 / 4
/// and the SAFE-fallthrough) must apply the same fallback so an
/// input with an empty `rationale_id` produces the canonical
/// default rather than an empty string. Rules 2a and 2b override
/// the field with an explicit literal, so they bypass this
/// helper.
fn effective_rationale_id(raw: &str) -> String {
    if raw.is_empty() {
        String::from(super::verdict::SAFE_BENIGN_RATIONALE)
    } else {
        String::from(raw)
    }
}

/// Return the highest reported context-hint confidence across
/// `hints`. When `confidences` is `None`, falls back to
/// [`DEFAULT_CONTEXT_CONFIDENCE_WHEN_MISSING`] so older pipelines
/// (which did not forward per-hint confidences) keep their
/// legacy always-demote behaviour.
///
/// When `confidences` is `Some(map)` but the map omits a listed
/// hint, that hint contributes `0.0` — a partial map cannot
/// accidentally re-enable always-demote.
fn best_context_confidence(
    hints: &[String],
    confidences: Option<&std::collections::BTreeMap<String, f64>>,
) -> f64 {
    let Some(map) = confidences else {
        return DEFAULT_CONTEXT_CONFIDENCE_WHEN_MISSING;
    };
    let mut best = 0.0_f64;
    for h in hints {
        let v = map.get(h).copied().unwrap_or(0.0);
        // Guard against NaN / non-finite confidences — those
        // would silently disable the floor under naive `>` /
        // `<` comparisons.
        let v = if v.is_finite() { v } else { 0.0 };
        if v > best {
            best = v;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::verdict::ModelHealth;

    fn raw_input(
        category: u32,
        severity: u8,
        confidence: f64,
        actions: Actions,
        reason_codes: Vec<&str>,
    ) -> RawClassifierOutput {
        RawClassifierOutput {
            verdict: Verdict {
                severity,
                category,
                confidence,
                actions,
                reason_codes: reason_codes.iter().map(|s| s.to_string()).collect(),
                rationale_id: String::from("test_v1"),
                resource_link_id: None,
                counter_updates: None,
                model_health: None,
            },
            context_hint_confidences: None,
        }
    }

    fn simple_raw(category: u32, severity: u8, confidence: f64) -> RawClassifierOutput {
        raw_input(category, severity, confidence, Actions::blank(), vec![])
    }

    // ----- Threshold constants -----
    #[test]
    fn thresholds_are_hard_coded() {
        let p = ThresholdPolicy::default();
        assert_eq!(p.label_only(), 0.45);
        assert_eq!(p.warn(), 0.62);
        assert_eq!(p.strong_warn(), 0.78);
        assert_eq!(p.critical_intervention(), 0.85);
    }

    #[test]
    fn thresholds_cannot_be_overridden() {
        let err = ThresholdPolicy::from_thresholds(0.5, 0.62, 0.78, 0.85).unwrap_err();
        assert!(matches!(err, ThresholdPolicyError::NonCanonical { .. }));
    }

    // ----- Uncertainty handling (< 0.45 => SAFE for non-zero category) -----
    #[test]
    fn confidence_044_with_non_zero_category_is_safe() {
        let p = ThresholdPolicy::default();
        let out = p.apply(&simple_raw(7, 3, 0.44));
        assert_eq!(out.category, SAFE_CATEGORY);
        assert_eq!(out.severity, 0);
        assert_eq!(out.actions, Actions::blank());
    }

    #[test]
    fn confidence_below_label_only_with_safe_category_stays_safe() {
        let p = ThresholdPolicy::default();
        let out = p.apply(&simple_raw(0, 0, 0.10));
        assert_eq!(out.category, SAFE_CATEGORY);
        assert_eq!(out.severity, 0);
    }

    // ----- Threshold boundaries -----
    #[test]
    fn threshold_boundaries_promote_to_each_tier() {
        let p = ThresholdPolicy::default();
        type AssertActionFn = fn(&Actions) -> bool;
        let cases: &[(f64, AssertActionFn)] = &[
            (0.45, |a| a.label_only),
            (0.62, |a| a.warn),
            (0.78, |a| a.strong_warn),
            (0.85, |a| a.critical_intervention),
        ];
        for (conf, expected) in cases {
            let out = p.apply(&simple_raw(7, 3, *conf));
            assert!(expected(&out.actions), "confidence {conf} did not promote");
            // At-most-one of the confidence-driven flags is set.
            let set_count = [
                out.actions.label_only,
                out.actions.warn,
                out.actions.strong_warn,
                out.actions.critical_intervention,
            ]
            .iter()
            .filter(|b| **b)
            .count();
            assert_eq!(set_count, 1, "exactly one tier should fire at {conf}");
        }
    }

    #[test]
    fn confidence_just_below_boundary_uses_previous_tier() {
        let p = ThresholdPolicy::default();
        let a = p.apply(&simple_raw(7, 3, 0.61)).actions;
        assert!(a.label_only && !a.warn);
        let b = p.apply(&simple_raw(7, 3, 0.77)).actions;
        assert!(b.warn && !b.strong_warn);
        let c = p.apply(&simple_raw(7, 3, 0.84)).actions;
        assert!(c.strong_warn && !c.critical_intervention);
    }

    // ----- Child-safety floor -----
    #[test]
    fn child_safety_at_confidence_045_pins_severity_5() {
        let p = ThresholdPolicy::default();
        let out = p.apply(&simple_raw(CHILD_SAFETY_CATEGORY, 2, 0.45));
        assert_eq!(out.severity, 5);
        assert!(out.actions.critical_intervention);
        assert!(out.reason_codes.iter().any(|r| r == "CHILD_SAFETY_FLOOR"));
    }

    #[test]
    fn child_safety_at_confidence_044_is_safe() {
        let p = ThresholdPolicy::default();
        let out = p.apply(&simple_raw(CHILD_SAFETY_CATEGORY, 2, 0.44));
        assert_eq!(out.category, SAFE_CATEGORY);
        assert_eq!(out.severity, 0);
    }

    #[test]
    fn child_safety_high_confidence_keeps_critical_intervention() {
        let p = ThresholdPolicy::default();
        let out = p.apply(&simple_raw(CHILD_SAFETY_CATEGORY, 5, 0.90));
        assert_eq!(out.severity, 5);
        assert!(out.actions.critical_intervention);
    }

    #[test]
    fn child_safety_floor_wins_over_news_context() {
        // Public-interest reporting of CSAM must still surface
        // the floor; rule 1 evaluates before rule 2.
        let p = ThresholdPolicy::default();
        let mut raw = simple_raw(CHILD_SAFETY_CATEGORY, 5, 0.90);
        raw.verdict.reason_codes = vec![
            String::from("NEWS_CONTEXT"),
            String::from("QUOTED_SPEECH_CONTEXT"),
        ];
        let out = p.apply(&raw);
        assert_eq!(out.category, CHILD_SAFETY_CATEGORY);
        assert_eq!(out.severity, 5);
        assert!(out.actions.critical_intervention);
        assert!(out.reason_codes.iter().any(|r| r == "CHILD_SAFETY_FLOOR"));
    }

    // ----- Tie-break -----
    #[test]
    fn tie_break_lower_numbered_category_wins() {
        let candidates = vec![
            Verdict {
                category: 6,
                severity: 3,
                confidence: 0.7,
                ..Verdict::default()
            },
            Verdict {
                category: 4,
                severity: 3,
                confidence: 0.7,
                ..Verdict::default()
            },
        ];
        let winner = ThresholdPolicy::tie_break(&candidates).unwrap();
        assert_eq!(winner.category, 4);
    }

    #[test]
    fn tie_break_prefers_higher_severity() {
        let candidates = vec![
            Verdict {
                category: 4,
                severity: 3,
                confidence: 0.7,
                ..Verdict::default()
            },
            Verdict {
                category: 6,
                severity: 4,
                confidence: 0.7,
                ..Verdict::default()
            },
        ];
        let winner = ThresholdPolicy::tie_break(&candidates).unwrap();
        assert_eq!(winner.category, 6);
    }

    #[test]
    fn tie_break_empty_returns_none() {
        assert!(ThresholdPolicy::tie_break(&[]).is_none());
    }

    // ----- Input does not override thresholds -----
    #[test]
    fn encoder_cannot_assert_warn_below_warn_threshold() {
        let p = ThresholdPolicy::default();
        let asserted = Actions {
            label_only: true,
            warn: true,
            ..Actions::blank()
        };
        let out = p.apply(&raw_input(7, 3, 0.10, asserted, vec![]));
        // Below label_only with non-SAFE category -> SAFE.
        assert_eq!(out.category, 0);
        assert_eq!(out.actions, Actions::blank());
    }

    #[test]
    fn encoder_cannot_assert_critical_intervention_at_low_confidence() {
        let p = ThresholdPolicy::default();
        let asserted = Actions {
            critical_intervention: true,
            ..Actions::blank()
        };
        let out = p.apply(&raw_input(7, 3, 0.50, asserted, vec![]));
        // At 0.50, only label_only should be set.
        assert!(out.actions.label_only);
        assert!(!out.actions.critical_intervention);
    }

    #[test]
    fn suggest_redact_is_preserved() {
        let p = ThresholdPolicy::default();
        let asserted = Actions {
            suggest_redact: true,
            ..Actions::blank()
        };
        let out = p.apply(&raw_input(9, 3, 0.70, asserted, vec![]));
        assert!(out.actions.suggest_redact);
        assert!(out.actions.warn);
    }

    // ----- Protected-speech demotion -----
    #[test]
    fn protected_speech_reason_codes_constant() {
        let codes: BTreeSet<&str> = PROTECTED_SPEECH_REASON_CODES.iter().copied().collect();
        let expected: BTreeSet<&str> = [
            "NEWS_CONTEXT",
            "EDUCATION_CONTEXT",
            "COUNTERSPEECH_CONTEXT",
            "QUOTED_SPEECH_CONTEXT",
        ]
        .iter()
        .copied()
        .collect();
        assert_eq!(codes, expected);
    }

    #[test]
    fn protected_speech_demotes_violence_threat_to_safe() {
        let p = ThresholdPolicy::default();
        for code in PROTECTED_SPEECH_REASON_CODES.iter() {
            let actions = Actions {
                label_only: true,
                ..Actions::blank()
            };
            let out = p.apply(&raw_input(3, 2, 0.50, actions, vec![code]));
            assert_eq!(out.category, SAFE_CATEGORY);
            assert_eq!(out.severity, 0);
            assert_eq!(out.actions, Actions::blank());
            assert!(out.reason_codes.iter().any(|r| r == code));
            assert_eq!(out.rationale_id, "safe_protected_speech_v1");
        }
    }

    #[test]
    fn protected_speech_preserves_multiple_protected_codes_drops_others() {
        let p = ThresholdPolicy::default();
        let out = p.apply(&raw_input(
            3,
            2,
            0.55,
            Actions::blank(),
            vec!["NEWS_CONTEXT", "QUOTED_SPEECH_CONTEXT", "LEXICON_HIT"],
        ));
        assert_eq!(out.category, SAFE_CATEGORY);
        assert!(out.reason_codes.iter().any(|r| r == "NEWS_CONTEXT"));
        assert!(out
            .reason_codes
            .iter()
            .any(|r| r == "QUOTED_SPEECH_CONTEXT"));
        // Non-protected codes are dropped from the demoted output.
        assert!(!out.reason_codes.iter().any(|r| r == "LEXICON_HIT"));
    }

    #[test]
    fn safe_with_protected_speech_stays_safe() {
        let p = ThresholdPolicy::default();
        let out = p.apply(&raw_input(
            SAFE_CATEGORY,
            0,
            0.20,
            Actions::blank(),
            vec!["NEWS_CONTEXT"],
        ));
        assert_eq!(out.category, SAFE_CATEGORY);
        assert_eq!(out.severity, 0);
    }

    #[test]
    fn non_protected_reason_code_does_not_demote() {
        let p = ThresholdPolicy::default();
        let out = p.apply(&raw_input(
            6,
            3,
            0.70,
            Actions::blank(),
            vec!["LEXICON_HIT"],
        ));
        assert_eq!(out.category, 6);
        assert!(out.actions.warn);
    }

    #[test]
    fn protected_speech_at_low_confidence_still_demotes_when_map_missing() {
        // No `context_hint_confidences` map => legacy fallback
        // (=1.0) => fully demote even at low classifier
        // confidence.
        let p = ThresholdPolicy::default();
        let out = p.apply(&raw_input(
            3,
            2,
            0.20,
            Actions::blank(),
            vec!["NEWS_CONTEXT"],
        ));
        assert_eq!(out.category, SAFE_CATEGORY);
        assert!(out.reason_codes.iter().any(|r| r == "NEWS_CONTEXT"));
        assert_eq!(out.rationale_id, "safe_protected_speech_v1");
    }

    #[test]
    fn protected_speech_below_floor_with_map_warns_with_context() {
        // Map present, NEWS_CONTEXT confidence below the
        // 0.5 floor => not fully demoted; warn-with-context.
        let p = ThresholdPolicy::default();
        let mut raw = raw_input(3, 4, 0.20, Actions::blank(), vec!["NEWS_CONTEXT"]);
        let mut map = std::collections::BTreeMap::new();
        map.insert(String::from("NEWS_CONTEXT"), 0.30);
        raw.context_hint_confidences = Some(map);
        let out = p.apply(&raw);
        assert_eq!(out.category, 3);
        // severity is clamped to min(orig, 2).
        assert_eq!(out.severity, 2);
        assert!(out.actions.warn);
        assert_eq!(out.rationale_id, "warn_low_confidence_context_v1");
        assert!(out.reason_codes.iter().any(|r| r == "WARN_WITH_CONTEXT"));
        assert!(out.reason_codes.iter().any(|r| r == "NEWS_CONTEXT"));
    }

    #[test]
    fn protected_speech_at_floor_with_map_fully_demotes() {
        // Boundary: floor=0.5 — exactly equal demotes.
        let p = ThresholdPolicy::default();
        let mut raw = raw_input(3, 4, 0.50, Actions::blank(), vec!["NEWS_CONTEXT"]);
        let mut map = std::collections::BTreeMap::new();
        map.insert(String::from("NEWS_CONTEXT"), 0.50);
        raw.context_hint_confidences = Some(map);
        let out = p.apply(&raw);
        assert_eq!(out.category, SAFE_CATEGORY);
        assert_eq!(out.rationale_id, "safe_protected_speech_v1");
    }

    #[test]
    fn protected_speech_partial_map_does_not_re_enable_always_demote() {
        // Map present but does NOT list NEWS_CONTEXT — that hint
        // contributes 0.0, so demotion threshold is not met.
        let p = ThresholdPolicy::default();
        let mut raw = raw_input(3, 4, 0.20, Actions::blank(), vec!["NEWS_CONTEXT"]);
        let mut map = std::collections::BTreeMap::new();
        map.insert(String::from("EDUCATION_CONTEXT"), 0.99);
        raw.context_hint_confidences = Some(map);
        let out = p.apply(&raw);
        // NEWS_CONTEXT is not in the map, scored 0.0 => below
        // floor => warn-with-context, not full demote.
        assert_eq!(out.category, 3);
        assert_eq!(out.rationale_id, "warn_low_confidence_context_v1");
    }

    #[test]
    fn protected_speech_nan_confidence_is_treated_as_zero() {
        // A pipeline that somehow forwards NaN as a hint
        // confidence must not silently disable the floor —
        // `best_context_confidence` collapses non-finite to 0.0.
        let p = ThresholdPolicy::default();
        let mut raw = raw_input(3, 4, 0.20, Actions::blank(), vec!["NEWS_CONTEXT"]);
        let mut map = std::collections::BTreeMap::new();
        map.insert(String::from("NEWS_CONTEXT"), f64::NAN);
        raw.context_hint_confidences = Some(map);
        let out = p.apply(&raw);
        assert_eq!(out.category, 3);
        assert_eq!(out.rationale_id, "warn_low_confidence_context_v1");
    }

    // ----- Experimental factory -----
    #[test]
    fn canonical_policy_is_not_experimental() {
        assert!(!ThresholdPolicy::default().is_experimental());
    }

    #[test]
    fn experimental_factory_returns_marked_instance() {
        let p = ThresholdPolicy::experimental(0.45, 0.55, 0.78, 0.85).unwrap();
        assert!(p.is_experimental());
        assert_eq!(p.warn(), 0.55);
    }

    #[test]
    fn experimental_factory_applies_with_overridden_thresholds() {
        let p = ThresholdPolicy::experimental(0.45, 0.50, 0.78, 0.85).unwrap();
        let above = p.apply(&simple_raw(7, 3, 0.51));
        assert!(above.actions.warn);
        assert!(!above.actions.label_only);
        let below = p.apply(&simple_raw(7, 3, 0.49));
        assert!(below.actions.label_only);
        assert!(!below.actions.warn);
    }

    #[test]
    fn experimental_factory_rejects_non_increasing_order() {
        let err = ThresholdPolicy::experimental(0.7, 0.5, 0.78, 0.85).unwrap_err();
        assert!(matches!(
            err,
            ThresholdPolicyError::NotStrictlyIncreasing { .. }
        ));
    }

    #[test]
    fn experimental_factory_rejects_equal_adjacent_thresholds() {
        let err = ThresholdPolicy::experimental(0.45, 0.62, 0.62, 0.85).unwrap_err();
        assert!(matches!(
            err,
            ThresholdPolicyError::NotStrictlyIncreasing { .. }
        ));
    }

    #[test]
    fn experimental_factory_rejects_out_of_range() {
        let err = ThresholdPolicy::experimental(0.0, 0.62, 0.78, 0.85).unwrap_err();
        assert!(matches!(err, ThresholdPolicyError::OutOfRange { .. }));
        let err = ThresholdPolicy::experimental(0.45, 0.62, 0.78, 1.5).unwrap_err();
        assert!(matches!(err, ThresholdPolicyError::OutOfRange { .. }));
    }

    #[test]
    fn experimental_factory_rejects_negative() {
        let err = ThresholdPolicy::experimental(-0.1, 0.62, 0.78, 0.85).unwrap_err();
        assert!(matches!(err, ThresholdPolicyError::OutOfRange { .. }));
    }

    #[test]
    fn experimental_factory_rejects_nan() {
        let err = ThresholdPolicy::experimental(f64::NAN, 0.62, 0.78, 0.85).unwrap_err();
        assert!(matches!(err, ThresholdPolicyError::OutOfRange { .. }));
        let err = ThresholdPolicy::experimental(0.45, f64::NAN, 0.78, 0.85).unwrap_err();
        assert!(matches!(err, ThresholdPolicyError::OutOfRange { .. }));
    }

    #[test]
    fn experimental_factory_accepts_full_confidence_at_critical() {
        let p = ThresholdPolicy::experimental(0.45, 0.62, 0.78, 1.0).unwrap();
        assert_eq!(p.critical_intervention(), 1.0);
    }

    #[test]
    fn experimental_policy_round_trips_apply_for_child_safety_floor() {
        // The child-safety floor uses `self.label_only`, not the
        // canonical 0.45 constant.
        let p = ThresholdPolicy::experimental(0.30, 0.50, 0.70, 0.90).unwrap();
        let out = p.apply(&simple_raw(CHILD_SAFETY_CATEGORY, 2, 0.31));
        assert_eq!(out.severity, 5);
        assert!(out.actions.critical_intervention);
        assert!(out.reason_codes.iter().any(|r| r == "CHILD_SAFETY_FLOOR"));
    }

    #[test]
    fn experimental_policy_round_trips_apply_for_uncertainty_cutoff() {
        // The uncertainty cutoff (< label_only -> SAFE) follows
        // the experimental label_only.
        let p = ThresholdPolicy::experimental(0.60, 0.62, 0.78, 0.85).unwrap();
        let out = p.apply(&simple_raw(7, 3, 0.50));
        assert_eq!(out.category, SAFE_CATEGORY);
    }

    // ----- Optional fields forward through -----
    #[test]
    fn model_health_forwards_through_apply() {
        let p = ThresholdPolicy::default();
        let mut raw = simple_raw(7, 3, 0.70);
        raw.verdict.model_health = Some(ModelHealth::ModelUnavailable);
        let out = p.apply(&raw);
        assert_eq!(out.model_health, Some(ModelHealth::ModelUnavailable));
        assert!(out.actions.warn);
    }

    #[test]
    fn model_health_forwards_through_child_safety_floor() {
        let p = ThresholdPolicy::default();
        let mut raw = simple_raw(CHILD_SAFETY_CATEGORY, 2, 0.90);
        raw.verdict.model_health = Some(ModelHealth::InferenceError);
        let out = p.apply(&raw);
        assert_eq!(out.model_health, Some(ModelHealth::InferenceError));
    }

    #[test]
    fn model_health_forwards_through_uncertainty_branch() {
        let p = ThresholdPolicy::default();
        let mut raw = simple_raw(7, 3, 0.10);
        raw.verdict.model_health = Some(ModelHealth::ModelUnavailable);
        let out = p.apply(&raw);
        assert_eq!(out.category, SAFE_CATEGORY);
        assert_eq!(out.model_health, Some(ModelHealth::ModelUnavailable));
    }

    // ----- Sanity: full chain wins over actions input -----
    #[test]
    fn classifier_action_set_is_recomputed_from_confidence() {
        let p = ThresholdPolicy::default();
        // Asserts critical_intervention but at confidence 0.62
        // only `warn` should fire.
        let asserted = Actions {
            critical_intervention: true,
            strong_warn: true,
            warn: true,
            label_only: true,
            ..Actions::blank()
        };
        let out = p.apply(&raw_input(7, 3, 0.62, asserted, vec![]));
        assert!(out.actions.warn);
        assert!(!out.actions.label_only);
        assert!(!out.actions.strong_warn);
        assert!(!out.actions.critical_intervention);
    }

    // -------- Validated deserialization --------

    #[test]
    fn deserialize_canonical_policy_round_trips() {
        let canonical = ThresholdPolicy::default();
        let json = serde_json::to_string(&canonical).expect("serialize canonical");
        let parsed: ThresholdPolicy = serde_json::from_str(&json).expect("deserialize canonical");
        assert_eq!(parsed, canonical);
        assert!(!parsed.is_experimental);
    }

    #[test]
    fn deserialize_experimental_policy_round_trips() {
        let exp = ThresholdPolicy::experimental(0.30, 0.50, 0.70, 0.90)
            .expect("build experimental for round-trip");
        let json = serde_json::to_string(&exp).expect("serialize experimental");
        let parsed: ThresholdPolicy =
            serde_json::from_str(&json).expect("deserialize experimental");
        assert_eq!(parsed, exp);
        assert!(parsed.is_experimental);
    }

    #[test]
    fn deserialize_rejects_non_canonical_production_policy() {
        // Crafted JSON claiming to be a signed-pack-canonical policy
        // but with relaxed thresholds — the very smuggling vector the
        // bot flagged. `try_from` must reject before the struct is
        // constructed.
        let smuggled = r#"{"label_only":0.10,"warn":0.20,"strong_warn":0.30,"critical_intervention":0.40,"is_experimental":false}"#;
        let result: Result<ThresholdPolicy, _> = serde_json::from_str(smuggled);
        assert!(
            result.is_err(),
            "non-canonical thresholds with is_experimental=false must not deserialize"
        );
    }

    #[test]
    fn deserialize_rejects_non_increasing_experimental_policy() {
        let bad = r#"{"label_only":0.60,"warn":0.50,"strong_warn":0.70,"critical_intervention":0.90,"is_experimental":true}"#;
        let result: Result<ThresholdPolicy, _> = serde_json::from_str(bad);
        assert!(result.is_err());
    }

    #[test]
    fn deserialize_rejects_out_of_range_experimental_policy() {
        let bad = r#"{"label_only":0.10,"warn":0.20,"strong_warn":0.30,"critical_intervention":1.50,"is_experimental":true}"#;
        let result: Result<ThresholdPolicy, _> = serde_json::from_str(bad);
        assert!(result.is_err());
    }
}

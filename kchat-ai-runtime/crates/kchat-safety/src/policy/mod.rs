//! Decision-policy enforcement.
//!
//! This module ports the hard-coded threshold policy from
//! `build-tools/compiler/threshold_policy.py`. It is the load-bearing
//! gate that re-coerces an encoder classifier's output before it
//! leaves the device — see ARCHITECTURE.md "Decision Policy"
//! (lines 353-373) for the canonical specification.
//!
//! ## What lives here
//!
//! * [`Verdict`] — the JSON-conformant output shape emitted by the
//!   guardrail pipeline. Matches `kchat-skills/global/output_schema.json`
//!   exactly (severity / category / confidence / actions / reason
//!   codes / rationale id + optional `resource_link_id` /
//!   `counter_updates` / `model_health`).
//! * [`Actions`] — the five boolean action flags
//!   (label_only / warn / strong_warn / critical_intervention /
//!   suggest_redact).
//! * [`ModelHealth`] — coarse safety-relevant status signal
//!   (`healthy` / `model_unavailable` / `inference_error`) the UI
//!   inspects to distinguish 'safe' from 'classifier could not run'.
//! * [`ThresholdPolicy`] — the immutable canonical policy plus the
//!   experimental factory ([`ThresholdPolicy::experimental`]) for
//!   research / calibration code.
//! * [`apply`](ThresholdPolicy::apply) — the four-rule chain
//!   (child-safety floor → protected-speech demotion → uncertainty
//!   cutoff → action re-derivation) that every classifier output
//!   passes through before the runtime emits it.
//! * [`tie_break`](ThresholdPolicy::tie_break) — the
//!   higher-severity-wins, lower-category-wins tie-break helper
//!   that the orchestrator uses when combining multiple candidate
//!   verdicts.
//!
//! ## Determinism and signed-pack invariants
//!
//! The canonical confidence thresholds
//! (`label_only=0.45`, `warn=0.62`, `strong_warn=0.78`,
//! `critical_intervention=0.85`) are **immutable for signed packs**.
//! A `ThresholdPolicy::default()` constructs the canonical policy;
//! any attempt to construct a non-canonical policy via the
//! validating builder
//! ([`ThresholdPolicy::from_thresholds`](ThresholdPolicy::from_thresholds))
//! is rejected with [`ThresholdPolicyError::NonCanonical`] unless
//! the caller explicitly opts into the experimental factory.
//! Production code paths (e.g. a future `GuardrailPipeline`)
//! refuse experimental policies by default.
//!
//! Experimental policies still validate the ordering invariant
//! `0 < label_only < warn < strong_warn < critical_intervention <= 1`,
//! so the apply()-chain `elif` ladder cannot emit nonsensical
//! severities.
//!
//! ## Cross-implementation parity
//!
//! The Python `compiler.threshold_policy.ThresholdPolicy.apply` is
//! the build-time oracle. `tests/parity.rs` feeds a corpus of
//! `RawClassifierOutput`s through both implementations and asserts
//! bit-identical `Verdict` JSON — the Rust port cannot drift
//! silently. See `tools/gen_parity_fixtures.py` for the fixture
//! generator.

// Gated on `text-pipeline` because the implementation routes
// through `policy_interpreter::{SeverityMapper, SeverityRubric}`
// — both of which live behind the same feature gate (see
// `lib.rs::87`). An encoder-only consumer
// (`--no-default-features --features encoder`) is documented as
// supported (`Cargo.toml` "Feature flag matrix"); leaving the
// module unconditional broke that contract because `gating.rs`
// fails to resolve `crate::policy_interpreter` when the feature
// is off. The two binding crates that ultimately call
// `derive_gating` (kchat-safety-ffi, kchat-safety-napi) only do
// so behind `onnx-runtime` which transitively requires
// `text-pipeline`, so no production caller is affected by the
// gating tightening.
pub mod compat;
#[cfg(feature = "text-pipeline")]
pub mod gating;
pub mod threshold;
pub mod verdict;

#[cfg(feature = "text-pipeline")]
pub use gating::derive_gating;
pub use threshold::{
    ThresholdPolicy, ThresholdPolicyError, CANONICAL_CRITICAL_INTERVENTION, CANONICAL_LABEL_ONLY,
    CANONICAL_STRONG_WARN, CANONICAL_WARN, CHILD_SAFETY_CATEGORY,
    CONTEXT_DEMOTION_CONFIDENCE_THRESHOLD, DEFAULT_CONTEXT_CONFIDENCE_WHEN_MISSING,
    PROTECTED_SPEECH_REASON_CODES, SAFE_CATEGORY,
};
pub use verdict::{
    severity_bucket_from_u8, Actions, CounterUpdate, ModelHealth, RawClassifierOutput,
    SeverityBucket, Verdict, SAFE_BENIGN_RATIONALE, SEVERITY_BUCKET_BORDERLINE_MIN,
    SEVERITY_BUCKET_SEVERE_MIN,
};

pub use compat::{
    compute_pack_digest, parse_action, severity_from_u8, verify_policy_pack, PolicyPack,
    PolicyPackError, PolicyPackManifest, PolicyRule, PolicyThresholds, RiskCategory,
};

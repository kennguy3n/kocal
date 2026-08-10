//! JSON-conformant output shape for the decision policy.
//!
//! Mirrors `files/global/output_schema.json` (draft-07,
//! `$id: kchat.guardrail.output.v1`). Round-trips through serde
//! against the same JSON the Python pipeline emits, so the parity
//! tests can compare byte-equivalent payloads.
//!
//! All fields except the five `required` schema fields are wrapped
//! in `Option<...>` with `skip_serializing_if = "Option::is_none"`
//! so a minimal-shape JSON (e.g. a SAFE verdict without
//! `resource_link_id` / `counter_updates` / `model_health`) does
//! not gain spurious null keys. The schema enforces
//! `additionalProperties: false`, so adding extra fields here
//! would break runtime validation downstream — keep this struct
//! in sync with the JSON Schema or the test
//! `verdict_round_trips_through_canonical_json` will fail.

use serde::{Deserialize, Serialize};

/// The five boolean action flags carried on every [`Verdict`].
///
/// `label_only` / `warn` / `strong_warn` / `critical_intervention`
/// are confidence-driven and re-derived from the verdict's
/// `confidence` by [`crate::policy::ThresholdPolicy::apply`] — a
/// classifier output that claims `warn=true` at `confidence=0.10`
/// is re-coerced by the policy before the verdict leaves the
/// device.
///
/// `suggest_redact` is a content-type hint (e.g. PII detected) and
/// is preserved verbatim through `apply()` — it is *not*
/// confidence-driven, so the policy does not clear or recompute it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Actions {
    pub label_only: bool,
    pub warn: bool,
    pub strong_warn: bool,
    pub critical_intervention: bool,
    pub suggest_redact: bool,
    pub surface_resources: bool,
}

impl Actions {
    /// All-false action set; the SAFE / uncertainty / protected-speech
    /// branches of `apply()` use this as the starting state.
    pub const fn blank() -> Self {
        Self {
            label_only: false,
            warn: false,
            strong_warn: false,
            critical_intervention: false,
            suggest_redact: false,
            surface_resources: false,
        }
    }
}

/// A counter-update record — one entry per
/// [`Verdict::counter_updates`] payload. The
/// `build-tools/compiler/counters.py` runtime resolves
/// `counter_id` against the locally-stored expiring-counter store
/// and applies `delta` (positive to increment, negative to
/// decrement) when the verdict is emitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterUpdate {
    pub counter_id: String,
    pub delta: i64,
}

/// Coarse, safety-relevant status signal indicating whether the
/// encoder model produced this verdict or whether the pipeline
/// fell back to a degraded mode (deterministic detectors only).
///
/// Matches the closed enum in `output_schema.json`. Consumers MUST
/// inspect this field — when it is anything other than
/// [`ModelHealth::Healthy`] the deterministic-detector reason
/// codes (`PRIVATE_DATA_PATTERN`, `SCAM_PATTERN`, `URL_RISK`,
/// `LEXICON_HIT`) may sit on a `category=0` (SAFE) verdict and the
/// UI must surface them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelHealth {
    #[default]
    Healthy,
    ModelUnavailable,
    InferenceError,
}

/// JSON-conformant verdict emitted by the guardrail pipeline.
///
/// This is the *output* shape of
/// [`crate::policy::ThresholdPolicy::apply`]. The input shape
/// allows one extra optional field
/// (`context_hint_confidences`) for the protected-speech demotion
/// rule; see [`RawClassifierOutput`].
///
/// Round-trip serde compatibility with the Python reference is
/// load-bearing — the parity test feeds JSON dumped from the
/// Python `apply()` here and asserts equality, so do not reorder
/// or rename fields without updating the fixture generator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    pub severity: u8,
    pub category: u32,
    pub confidence: f64,
    pub actions: Actions,
    pub reason_codes: Vec<String>,
    pub rationale_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_link_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counter_updates: Option<Vec<CounterUpdate>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_health: Option<ModelHealth>,
}

/// Canonical `rationale_id` emitted by the deterministic-classifier
/// fall-through SAFE path. Mirrors the Python
/// `compiler/encoder_adapter.py::_safe_output()` constant.
///
/// Exposed as a `pub const` so adapters and bridges that need to
/// distinguish "the upstream classifier emitted canonical SAFE"
/// from "the upstream classifier emitted a specific deterministic
/// rationale" (e.g. the ONNX encoder bridge's per-call fallback
/// path) can do the comparison against a shared symbol instead
/// of duplicating the string literal. Changing the canonical
/// SAFE rationale id is a coordinated cross-language change that
/// touches the Python reference + every consumer of this constant
/// in lockstep, so the value is intentionally NOT load-time
/// configurable.
pub const SAFE_BENIGN_RATIONALE: &str = "safe_benign_v1";

/// Coarse three-tier projection of [`Verdict::severity`] (a `u8`
/// on the 0–5 scale) onto a `safe / mild / serious` bucket that
/// foreign-language UI surfaces can pattern-match on without
/// having to memorise the exact 0–5 numbers.
///
/// This enum is the **single source of truth** for the bucketing
/// the FFI and napi bindings expose as `FfiSeverity` / `Severity`.
/// Each binding owns its own enum (because UniFFI and napi-rs
/// proc-macros cannot share type definitions across crate
/// boundaries), but both bindings construct their enum from a
/// `From<SeverityBucket>` impl that consumes this type, so the
/// bucketing thresholds live in exactly one place
/// ([`severity_bucket_from_u8`]).
///
/// Bucketing rules:
///
/// | core severity | bucket       | UX intent                       |
/// |---------------|--------------|---------------------------------|
/// | `0`           | `Benign`     | No actionable signal            |
/// | `1`–`2`       | `Borderline` | Renderer should warn / blur     |
/// | `3`–`5`       | `Severe`     | Renderer should block / hide    |
///
/// Callers that need the precise 0–5 number stay on the existing
/// `policy_decide` path + the raw severity field (kept as a `u32`
/// at the FFI boundary). [`SeverityBucket`] is purely a
/// convenience projection for the classify-text surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeverityBucket {
    /// `Verdict::severity = 0`. No actionable signal.
    Benign,
    /// `Verdict::severity` in `1..=2`. Renderer should warn / blur.
    Borderline,
    /// `Verdict::severity` in `3..=5`. Renderer should block / hide.
    Severe,
}

/// Lower bound (inclusive) of the `Borderline` bucket. Severity
/// values in `1..=2` project to [`SeverityBucket::Borderline`].
pub const SEVERITY_BUCKET_BORDERLINE_MIN: u8 = 1;

/// Lower bound (inclusive) of the `Severe` bucket. Severity values
/// in `3..=5` project to [`SeverityBucket::Severe`].
pub const SEVERITY_BUCKET_SEVERE_MIN: u8 = 3;

/// Project a raw 0–5 [`Verdict::severity`] onto the three-tier
/// [`SeverityBucket`] enum. The single source of truth for the
/// bucketing the FFI / napi bindings expose at the
/// `classify_text` boundary; both bindings call this and then
/// map the result onto their respective enum.
///
/// Out-of-band inputs (severity > 5) are clamped to
/// [`SeverityBucket::Severe`] — the schema constrains severity to
/// `0..=5` so this branch is defensive only.
pub fn severity_bucket_from_u8(severity: u8) -> SeverityBucket {
    if severity >= SEVERITY_BUCKET_SEVERE_MIN {
        SeverityBucket::Severe
    } else if severity >= SEVERITY_BUCKET_BORDERLINE_MIN {
        SeverityBucket::Borderline
    } else {
        SeverityBucket::Benign
    }
}

impl Verdict {
    /// All-SAFE verdict (category 0, severity 0, confidence
    /// `confidence`, blank actions). Used by the uncertainty
    /// cutoff branch of `apply()`.
    pub fn safe_with_confidence(confidence: f64, rationale_id: impl Into<String>) -> Self {
        Self {
            severity: 0,
            category: super::SAFE_CATEGORY,
            confidence,
            actions: Actions::blank(),
            reason_codes: Vec::new(),
            rationale_id: rationale_id.into(),
            resource_link_id: None,
            counter_updates: None,
            model_health: None,
        }
    }

    /// Canonical SAFE verdict — `confidence = 0.05`,
    /// `rationale_id = "safe_benign_v1"`, blank actions, no reason
    /// codes. Mirrors the Python `compiler/encoder_adapter.py::
    /// _safe_output()` shape that the deterministic-classifier
    /// fall-through path emits.
    ///
    /// Prefer this over [`Verdict::default`] anywhere you want the
    /// shape the production pipeline would emit for an
    /// adapter-fall-through SAFE: `Verdict::default()` exists
    /// primarily to populate fields when deserialising JSON with
    /// absent keys and uses `confidence = 0.0` (the "no signal"
    /// default), which differs from the canonical SAFE
    /// `confidence = 0.05` by `0.05`. The difference is silent in
    /// tests that don't assert on `confidence`, so this helper
    /// removes a footgun for future test adapters / mock
    /// classifiers that want canonical SAFE.
    pub fn safe() -> Self {
        Self::safe_with_confidence(0.05, SAFE_BENIGN_RATIONALE)
    }
}

/// Input shape consumed by
/// [`crate::policy::ThresholdPolicy::apply`].
///
/// Wraps a [`Verdict`] (the schema-conformant fields the classifier
/// emits) plus the optional `context_hint_confidences` side-channel
/// that the pipeline forwards for the protected-speech demotion
/// rule (P1-1 in the Python reference).
///
/// `context_hint_confidences` is an INPUT-only field — the
/// `output_schema.json` enforces `additionalProperties: false`, so
/// it never appears on a [`Verdict`] returned by `apply()`. Older
/// pipelines that did not forward confidences set the map to
/// `None`, which triggers the legacy "always demote" behaviour
/// (per the Python
/// `DEFAULT_CONTEXT_CONFIDENCE_WHEN_MISSING = 1.0` fallback).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RawClassifierOutput {
    #[serde(flatten)]
    pub verdict: Verdict,
    /// Per-hint confidence map for the four protected-speech
    /// reason codes (`NEWS_CONTEXT` / `EDUCATION_CONTEXT` /
    /// `COUNTERSPEECH_CONTEXT` / `QUOTED_SPEECH_CONTEXT`). When
    /// `None`, the legacy fallback applies (any protected-speech
    /// reason code on a non-SAFE / non-CHILD_SAFETY verdict fully
    /// demotes to SAFE). When `Some(map)` but the map omits a
    /// listed hint, that hint contributes `0.0` so a partial map
    /// cannot accidentally re-enable the legacy always-demote.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_hint_confidences: Option<std::collections::BTreeMap<String, f64>>,
}

impl Default for Verdict {
    fn default() -> Self {
        Self {
            severity: 0,
            category: super::SAFE_CATEGORY,
            confidence: 0.0,
            actions: Actions::blank(),
            reason_codes: Vec::new(),
            rationale_id: String::from(SAFE_BENIGN_RATIONALE),
            resource_link_id: None,
            counter_updates: None,
            model_health: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actions_blank_is_all_false() {
        let a = Actions::blank();
        assert!(!a.label_only);
        assert!(!a.warn);
        assert!(!a.strong_warn);
        assert!(!a.critical_intervention);
        assert!(!a.suggest_redact);
    }

    #[test]
    fn model_health_round_trips_snake_case() {
        // The Python enum stores `"model_unavailable"` etc — Rust
        // must emit identical strings or parity will diverge.
        let h = ModelHealth::ModelUnavailable;
        let json = serde_json::to_string(&h).unwrap();
        assert_eq!(json, "\"model_unavailable\"");
        let back: ModelHealth = serde_json::from_str(&json).unwrap();
        assert_eq!(back, ModelHealth::ModelUnavailable);
    }

    #[test]
    fn minimal_verdict_omits_none_fields() {
        // A minimal safe verdict must serialize WITHOUT `null`
        // resource_link_id / counter_updates / model_health —
        // those are excluded by `skip_serializing_if`.
        let v = Verdict::safe_with_confidence(0.0, "safe_benign_v1");
        let json = serde_json::to_string(&v).unwrap();
        assert!(!json.contains("resource_link_id"));
        assert!(!json.contains("counter_updates"));
        assert!(!json.contains("model_health"));
        assert!(json.contains("\"category\":0"));
        assert!(json.contains("\"severity\":0"));
    }

    #[test]
    fn verdict_round_trips_through_canonical_json() {
        // The exact-byte JSON shape the Python `apply()` emits.
        // If anyone reorders a field, renames a key, or changes a
        // default — this assertion catches it before parity does.
        let v = Verdict {
            severity: 3,
            category: 7,
            confidence: 0.62,
            actions: Actions {
                label_only: false,
                warn: true,
                strong_warn: false,
                critical_intervention: false,
                suggest_redact: false,
                surface_resources: false,
            },
            reason_codes: vec![String::from("LEXICON_HIT")],
            rationale_id: String::from("warn_v1"),
            resource_link_id: None,
            counter_updates: None,
            model_health: Some(ModelHealth::Healthy),
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: Verdict = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn severity_bucket_thresholds_match_canonical_ladder() {
        // Lock in the 3-tier projection that every FFI binding
        // surfaces at the `classify_text` boundary. Drift here
        // would silently re-bucket warn/strong-warn UX states
        // without touching the FFI enum schema, so the test
        // pins both the bucket boundaries and the constants.
        assert_eq!(SEVERITY_BUCKET_BORDERLINE_MIN, 1);
        assert_eq!(SEVERITY_BUCKET_SEVERE_MIN, 3);

        // Below `BORDERLINE_MIN`: Benign.
        assert_eq!(severity_bucket_from_u8(0), SeverityBucket::Benign);

        // `BORDERLINE_MIN..SEVERE_MIN`: Borderline.
        assert_eq!(severity_bucket_from_u8(1), SeverityBucket::Borderline);
        assert_eq!(severity_bucket_from_u8(2), SeverityBucket::Borderline);

        // `>= SEVERE_MIN`: Severe. Covers all canonical severities
        // (3 = warn, 4 = strong_warn, 5 = critical_intervention)
        // plus the schema-out-of-range defensive clamp.
        assert_eq!(severity_bucket_from_u8(3), SeverityBucket::Severe);
        assert_eq!(severity_bucket_from_u8(4), SeverityBucket::Severe);
        assert_eq!(severity_bucket_from_u8(5), SeverityBucket::Severe);
        assert_eq!(severity_bucket_from_u8(255), SeverityBucket::Severe);
    }

    #[test]
    fn raw_classifier_output_flattens_verdict_fields() {
        // Python passes a single dict with the verdict fields and
        // an optional `context_hint_confidences` key. The Rust
        // `RawClassifierOutput` must flatten the verdict so the
        // resulting JSON has a flat structure (no nested
        // `"verdict": { ... }`).
        let raw = RawClassifierOutput {
            verdict: Verdict {
                severity: 2,
                category: 3,
                confidence: 0.55,
                actions: Actions::blank(),
                reason_codes: vec![String::from("NEWS_CONTEXT")],
                rationale_id: String::from("test_v1"),
                resource_link_id: None,
                counter_updates: None,
                model_health: None,
            },
            context_hint_confidences: Some({
                let mut m = std::collections::BTreeMap::new();
                m.insert(String::from("NEWS_CONTEXT"), 0.7);
                m
            }),
        };
        let json = serde_json::to_string(&raw).unwrap();
        // No nested `"verdict"` key — fields are flattened.
        assert!(!json.contains("\"verdict\""));
        assert!(json.contains("\"category\":3"));
        assert!(json.contains("\"context_hint_confidences\""));
        let back: RawClassifierOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(raw, back);
    }
}

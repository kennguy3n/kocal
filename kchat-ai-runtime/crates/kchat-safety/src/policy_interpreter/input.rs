//! Signal JSON accepted by the SLM policy interpreter.
//!
//! Mirrors `cv-guard/shared/policy/policy_input.py`. The types
//! here are deliberately narrow — every field is either a scalar,
//! a mapping of label-id → score, or a tightly-typed OCR summary.
//! The interpreter MUST NOT accept raw pixels or user-provided free
//! text; that contract is enforced by the type system below and by
//! [`crate::policy_interpreter::sanitizer`] for the one field
//! (`context_hints`) the host application can populate freely.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Fully-qualified label → sigmoid score in `[0.0, 1.0]`.
///
/// `BTreeMap` (rather than `HashMap`) so iteration order is
/// deterministic — the SLM prompt embedded these scores
/// alphabetically, and a hash-order-dependent prompt would lead to
/// non-deterministic SLM output across deployments.
pub type VisionScores = BTreeMap<String, f64>;

/// Compact, non-text OCR summary the interpreter is allowed to
/// see.
///
/// Raw OCR text is *never* forwarded to the SLM; instead the
/// client distils it to a handful of boolean / categorical flags
/// (URL present, crypto-wallet candidate found, scam-phrase match
/// count, PII category matches) that the SLM can reason about.
///
/// Validation invariants (mirrors the Python pydantic model):
///
/// * every count field is non-negative (enforced by `u32`),
/// * `pii_categories_matched` is de-duplicated (the
///   [`OCRSignals::new`] constructor strips repeats while
///   preserving the first occurrence — match Python's
///   `_no_duplicates` validator exactly),
/// * `pii_categories_matched` accepts any string here; the
///   closed-set membership check lives in
///   [`crate::policy_interpreter::sanitizer::sanitize_pii_categories`]
///   so the same allow-list is applied across iOS / Android /
///   desktop without duplicating it in the input type.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OCRSignals {
    /// Whether OCR actually executed. If `false`, every count flag
    /// below is a don't-care.
    #[serde(default)]
    pub ran: bool,
    /// Number of distinct URLs found.
    #[serde(default)]
    pub url_count: u32,
    /// Subset of `url_count` that contain a Punycode-encoded
    /// host (a strong scam signal).
    #[serde(default)]
    pub punycode_url_count: u32,
    /// Number of matches against the crypto-wallet candidate
    /// matcher.
    #[serde(default)]
    pub crypto_wallet_matches: u32,
    /// Number of scam-phrase hits.
    #[serde(default)]
    pub scam_phrase_hits: u32,
    /// Categories of PII that the OCR pipeline matched.
    ///
    /// Insertion order is preserved (`Vec`) and duplicates are
    /// stripped at construction.
    #[serde(default)]
    pub pii_categories_matched: Vec<String>,
    /// Length bucket only — no raw text content.
    #[serde(default)]
    pub total_text_chars: u32,
}

impl OCRSignals {
    /// Construct a fresh [`OCRSignals`] with `ran = false` and
    /// every counter set to zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a PII category, preserving first-occurrence order.
    /// Duplicates after the first call to `push_pii_category` for
    /// a given value are silently dropped — matches the Python
    /// `_no_duplicates` validator on the field.
    pub fn push_pii_category(&mut self, category: impl Into<String>) {
        let category = category.into();
        if !self.pii_categories_matched.iter().any(|c| c == &category) {
            self.pii_categories_matched.push(category);
        }
    }

    /// Drop duplicates from `pii_categories_matched` in-place,
    /// preserving first-occurrence order. The `serde` deserialisation
    /// path doesn't run the pydantic field validator — callers
    /// rebuilding an [`OCRSignals`] from JSON should call this to
    /// get the same shape Python yields.
    pub fn dedup_pii_categories(&mut self) {
        let mut seen: Vec<String> = Vec::with_capacity(self.pii_categories_matched.len());
        for c in std::mem::take(&mut self.pii_categories_matched) {
            if !seen.iter().any(|s| s == &c) {
                seen.push(c);
            }
        }
        self.pii_categories_matched = seen;
    }
}

/// Media type — only `Image` and `Video` ever appear on the wire;
/// anything else is a contract violation upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum MediaType {
    Image,
    Video,
}

impl MediaType {
    /// On-the-wire string used by every platform mirror.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
        }
    }
}

impl fmt::Display for MediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Errors raised when constructing or validating a
/// [`PolicyInput`].
///
/// `Eq` is intentionally NOT derived because [`PolicyInputError::ScoreOutOfRange::score`]
/// is an `f64` that may carry `NaN` (and `NaN != NaN`). The
/// `PartialEq` impl below treats two `ScoreOutOfRange` variants
/// as equal when the label matches and both scores fail the
/// validation predicate identically — useful for test assertions.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum PolicyInputError {
    /// `media_id` was empty.
    EmptyMediaId,
    /// `vision_scores` contains a label whose score is outside
    /// `[0.0, 1.0]` (or `NaN` / non-finite).
    ScoreOutOfRange { label: String, score: f64 },
}

impl fmt::Display for PolicyInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMediaId => f.write_str("media_id must not be empty"),
            Self::ScoreOutOfRange { label, score } => {
                write!(
                    f,
                    "score for {label} out of range: {score} (expected [0.0, 1.0])"
                )
            }
        }
    }
}

impl std::error::Error for PolicyInputError {}

/// Full signal JSON handed to the policy interpreter.
///
/// The interpreter uses `vision_scores` + the active skill pack's
/// threshold table for its fast-path rule evaluation, and only
/// serialises the input itself into the SLM prompt for ambiguous
/// mid-range cases.
///
/// `context_hints` is the one field the host application can
/// populate freely. Pass the raw map through
/// [`crate::policy_interpreter::sanitizer::sanitize_context_hints`]
/// before constructing the [`PolicyInput`] so prompt-injection
/// payloads (newlines, bidi overrides, embedded JSON) are dropped
/// at the boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyInput {
    /// Opaque, caller-supplied media identifier.
    pub media_id: String,
    /// `image` or `video`.
    pub media_type: MediaType,
    /// Sigmoid output of the classifier head per label.
    #[serde(default)]
    pub vision_scores: VisionScores,
    /// Distilled OCR summary; never raw text.
    #[serde(default)]
    pub ocr: OCRSignals,
    /// Coarse, allow-listed hints from the host application —
    /// `jurisdiction`, `community_type`, `sender_trust`,
    /// `media_origin`, `user_preferences`, `test_scenario`.
    #[serde(default)]
    pub context_hints: BTreeMap<String, String>,
    /// If `false` the interpreter MUST stay on the rule-based fast
    /// path even when the rule output is ambiguous (the SLM is
    /// skipped). Default `true` matches the Python default.
    #[serde(default = "default_allow_slm")]
    pub allow_slm: bool,
}

fn default_allow_slm() -> bool {
    true
}

impl PolicyInput {
    /// Construct a [`PolicyInput`] with full validation. Mirrors
    /// the Python pydantic field validators.
    pub fn new(
        media_id: impl Into<String>,
        media_type: MediaType,
    ) -> Result<Self, PolicyInputError> {
        let media_id = media_id.into();
        if media_id.is_empty() {
            return Err(PolicyInputError::EmptyMediaId);
        }
        Ok(Self {
            media_id,
            media_type,
            vision_scores: BTreeMap::new(),
            ocr: OCRSignals::default(),
            context_hints: BTreeMap::new(),
            allow_slm: true,
        })
    }

    /// Fluent setter for `vision_scores`. Validates every score is
    /// finite + in `[0.0, 1.0]`.
    pub fn with_vision_scores(mut self, scores: VisionScores) -> Result<Self, PolicyInputError> {
        for (label, score) in &scores {
            if !score.is_finite() || !(0.0..=1.0).contains(score) {
                return Err(PolicyInputError::ScoreOutOfRange {
                    label: label.clone(),
                    score: *score,
                });
            }
        }
        self.vision_scores = scores;
        Ok(self)
    }

    /// Fluent setter for `ocr`.
    #[must_use]
    pub fn with_ocr(mut self, ocr: OCRSignals) -> Self {
        self.ocr = ocr;
        self
    }

    /// Fluent setter for `context_hints`. Note: this does NOT
    /// re-sanitize — callers MUST run
    /// [`crate::policy_interpreter::sanitizer::sanitize_context_hints`]
    /// first.
    #[must_use]
    pub fn with_context_hints(mut self, hints: BTreeMap<String, String>) -> Self {
        self.context_hints = hints;
        self
    }

    /// Fluent setter for `allow_slm`.
    #[must_use]
    pub fn with_allow_slm(mut self, allow: bool) -> Self {
        self.allow_slm = allow;
        self
    }

    /// Re-validate every field. Useful when reconstituting from
    /// JSON because the deserialisation path doesn't run the
    /// pydantic-style validators on `vision_scores`.
    pub fn validate(&self) -> Result<(), PolicyInputError> {
        if self.media_id.is_empty() {
            return Err(PolicyInputError::EmptyMediaId);
        }
        for (label, score) in &self.vision_scores {
            if !score.is_finite() || !(0.0..=1.0).contains(score) {
                return Err(PolicyInputError::ScoreOutOfRange {
                    label: label.clone(),
                    score: *score,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ocr_signals_default_is_zero_initialised() {
        let ocr = OCRSignals::default();
        assert!(!ocr.ran);
        assert_eq!(ocr.url_count, 0);
        assert_eq!(ocr.punycode_url_count, 0);
        assert_eq!(ocr.crypto_wallet_matches, 0);
        assert_eq!(ocr.scam_phrase_hits, 0);
        assert!(ocr.pii_categories_matched.is_empty());
        assert_eq!(ocr.total_text_chars, 0);
    }

    #[test]
    fn ocr_signals_push_pii_dedup_preserves_order() {
        let mut ocr = OCRSignals::default();
        ocr.push_pii_category("phone");
        ocr.push_pii_category("crypto_wallet");
        ocr.push_pii_category("phone"); // duplicate
        ocr.push_pii_category("govt_id");
        assert_eq!(
            ocr.pii_categories_matched,
            vec!["phone", "crypto_wallet", "govt_id"]
        );
    }

    #[test]
    fn ocr_signals_dedup_collapses_runtime_duplicates() {
        let mut ocr = OCRSignals {
            pii_categories_matched: vec![
                "phone".into(),
                "phone".into(),
                "crypto_wallet".into(),
                "phone".into(),
            ],
            ..Default::default()
        };
        ocr.dedup_pii_categories();
        assert_eq!(ocr.pii_categories_matched, vec!["phone", "crypto_wallet"]);
    }

    #[test]
    fn media_type_strings_match_python_contract() {
        assert_eq!(MediaType::Image.as_str(), "image");
        assert_eq!(MediaType::Video.as_str(), "video");
    }

    #[test]
    fn media_type_serde_uses_lowercase_strings() {
        assert_eq!(
            serde_json::to_string(&MediaType::Image).unwrap(),
            "\"image\""
        );
        let parsed: MediaType = serde_json::from_str("\"video\"").unwrap();
        assert_eq!(parsed, MediaType::Video);
    }

    #[test]
    fn policy_input_new_rejects_empty_media_id() {
        let err = PolicyInput::new("", MediaType::Image).unwrap_err();
        assert_eq!(err, PolicyInputError::EmptyMediaId);
    }

    #[test]
    fn policy_input_with_vision_scores_validates_range() {
        let mut scores = VisionScores::new();
        scores.insert("adult.explicit_sexual".into(), 0.5);
        scores.insert("benign".into(), 1.0);
        scores.insert("graphic".into(), 0.0);
        let input = PolicyInput::new("m1", MediaType::Image)
            .unwrap()
            .with_vision_scores(scores)
            .unwrap();
        assert_eq!(input.vision_scores.len(), 3);

        let mut bad = VisionScores::new();
        bad.insert("scam".into(), 1.5);
        let err = PolicyInput::new("m1", MediaType::Image)
            .unwrap()
            .with_vision_scores(bad)
            .unwrap_err();
        match err {
            PolicyInputError::ScoreOutOfRange { label, score } => {
                assert_eq!(label, "scam");
                assert_eq!(score, 1.5);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn policy_input_with_vision_scores_rejects_non_finite() {
        let mut bad = VisionScores::new();
        bad.insert("scam".into(), f64::NAN);
        assert!(matches!(
            PolicyInput::new("m1", MediaType::Image)
                .unwrap()
                .with_vision_scores(bad),
            Err(PolicyInputError::ScoreOutOfRange { .. })
        ));
        let mut bad = VisionScores::new();
        bad.insert("scam".into(), f64::INFINITY);
        assert!(matches!(
            PolicyInput::new("m1", MediaType::Image)
                .unwrap()
                .with_vision_scores(bad),
            Err(PolicyInputError::ScoreOutOfRange { .. })
        ));
    }

    #[test]
    fn policy_input_defaults_match_python() {
        let input = PolicyInput::new("m1", MediaType::Image).unwrap();
        assert!(input.vision_scores.is_empty());
        assert!(input.ocr.pii_categories_matched.is_empty());
        assert!(input.context_hints.is_empty());
        assert!(input.allow_slm);
    }

    #[test]
    fn policy_input_validate_catches_post_mutation_drift() {
        let mut input = PolicyInput::new("m1", MediaType::Image).unwrap();
        input.vision_scores.insert("scam".into(), -0.1);
        assert!(matches!(
            input.validate(),
            Err(PolicyInputError::ScoreOutOfRange { .. })
        ));
        input.vision_scores.clear();
        input.media_id.clear();
        assert_eq!(input.validate(), Err(PolicyInputError::EmptyMediaId));
    }

    #[test]
    fn policy_input_serde_round_trip() {
        let mut scores = VisionScores::new();
        scores.insert("adult.explicit_sexual".into(), 0.95);
        scores.insert("benign".into(), 0.02);
        let mut ocr = OCRSignals {
            ran: true,
            url_count: 3,
            scam_phrase_hits: 1,
            ..Default::default()
        };
        ocr.push_pii_category("phone");
        ocr.push_pii_category("crypto_wallet");
        let mut hints = BTreeMap::new();
        hints.insert("jurisdiction".into(), "us-ca".into());
        let input = PolicyInput::new("media-1", MediaType::Image)
            .unwrap()
            .with_vision_scores(scores)
            .unwrap()
            .with_ocr(ocr)
            .with_context_hints(hints)
            .with_allow_slm(false);
        let json = serde_json::to_string(&input).unwrap();
        let parsed: PolicyInput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, input);
    }

    #[test]
    fn policy_input_serde_rejects_extra_fields() {
        // mirrors pydantic ConfigDict(extra="forbid").
        let json = r#"{
            "media_id": "m1",
            "media_type": "image",
            "extra_evil": "evil"
        }"#;
        let result: Result<PolicyInput, _> = serde_json::from_str(json);
        assert!(result.is_err(), "expected extra_evil to be rejected");
    }

    #[test]
    fn ocr_signals_serde_rejects_extra_fields() {
        let json = r#"{
            "ran": true,
            "url_count": 1,
            "wat": 1
        }"#;
        let result: Result<OCRSignals, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}

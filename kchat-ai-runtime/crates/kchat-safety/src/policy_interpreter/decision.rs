//! Decision JSON returned by the SLM policy interpreter.
//!
//! Mirrors `cv-guard/shared/policy/policy_decision.py`. Every field
//! is either a scalar, an enum, or a stable rationale code — never
//! free-form text that could leak content out of the decision path.
//!
//! The interpreter writes this struct after either the rule-based
//! fast path or the SLM-backed slow path; the renderer / platform
//! UI reads it directly. The set of [`UXAction`] values is the
//! cross-platform contract — the iOS Swift enum, Android Kotlin
//! enum, and desktop TypeScript union mirror these exact strings,
//! and adding a fifth action requires updating all three mirrors in
//! lock-step with this file. The [`PolicyDecision::rationale_code`]
//! shape (`category.sub.severity` dotted snake_case) is also a
//! cross-platform contract — anything outside that shape is
//! rejected at construction time.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Stable set of UX actions the renderer is allowed to apply.
///
/// Matches the desktop TypeScript union and the iOS / Android
/// enums. Adding a value requires updating those mirrors in the
/// same change. The on-the-wire representation is the lowercase
/// snake_case string (see [`UXAction::as_str`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum UXAction {
    /// No friction — render the media normally.
    Clear,
    /// Blur the media; user taps to reveal.
    BlurTap,
    /// Persistently pixelate (irrecoverable preview).
    Pixelate,
    /// Block the media behind a card explaining the rejection.
    BlockedCard,
}

impl UXAction {
    /// On-the-wire string used by every platform mirror.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Clear => "clear",
            Self::BlurTap => "blur_tap",
            Self::Pixelate => "pixelate",
            Self::BlockedCard => "blocked_card",
        }
    }

    /// Parse the cross-platform string back into the enum. Returns
    /// [`UXActionParseError::Unknown`] for anything not in the
    /// closed set. The error is surfaced rather than mapped to a
    /// default because the renderer contract is closed — an
    /// unknown action almost certainly means a skill pack drift.
    pub fn from_str_strict(s: &str) -> Result<Self, UXActionParseError> {
        match s {
            "clear" => Ok(Self::Clear),
            "blur_tap" => Ok(Self::BlurTap),
            "pixelate" => Ok(Self::Pixelate),
            "blocked_card" => Ok(Self::BlockedCard),
            other => Err(UXActionParseError::Unknown(other.to_string())),
        }
    }
}

impl fmt::Display for UXAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error raised when [`UXAction::from_str_strict`] fails.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum UXActionParseError {
    /// String was not one of `clear` / `blur_tap` / `pixelate` /
    /// `blocked_card`.
    Unknown(String),
}

impl fmt::Display for UXActionParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(s) => write!(
                f,
                "ux_action {s:?} not in {{clear, blur_tap, pixelate, blocked_card}}"
            ),
        }
    }
}

impl std::error::Error for UXActionParseError {}

/// Origin of a [`PolicyDecision`] — useful for telemetry and for
/// gating UI behavior that depends on whether the SLM was
/// consulted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DecisionSource {
    /// Decided on the rule-based fast path; no SLM call.
    Rule,
    /// Decided by the SLM after rule-path ambiguity.
    Slm,
    /// Decided by the child-safety floor (severity-5 invariant);
    /// overrides any other source.
    ChildSafetyFloor,
}

impl DecisionSource {
    /// On-the-wire string used by every platform mirror.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rule => "rule",
            Self::Slm => "slm",
            Self::ChildSafetyFloor => "child_safety_floor",
        }
    }
}

impl fmt::Display for DecisionSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Errors raised when constructing a [`PolicyDecision`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PolicyDecisionError {
    /// `category` was empty or whitespace-only.
    EmptyCategory,
    /// `severity` was outside `0..=5`.
    SeverityOutOfRange(i32),
    /// `rationale_code` did not match the closed-set shape (dotted
    /// snake_case starting with an alpha character).
    InvalidRationaleCode(String),
}

impl fmt::Display for PolicyDecisionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCategory => f.write_str("category must not be empty"),
            Self::SeverityOutOfRange(n) => {
                write!(f, "severity must be in 0..=5 (got {n})")
            }
            Self::InvalidRationaleCode(s) => write!(
                f,
                "rationale_code {s:?} must be dotted snake_case (e.g. \"adult.explicit_sexual.severe\")"
            ),
        }
    }
}

impl std::error::Error for PolicyDecisionError {}

/// Concrete per-scan decision the interpreter returns to the
/// scanner.
///
/// Mirrors Python `shared.policy.policy_decision.PolicyDecision`
/// exactly. Construction goes through [`PolicyDecision::new`] so
/// every value passes the same validators the Python pydantic
/// model applies — this matters for the prompt-injection threat
/// model where a malicious SLM response could otherwise smuggle a
/// pathological rationale code through the decision path.
///
/// `#[serde(deny_unknown_fields)]` matches the Python
/// `ConfigDict(extra="forbid")` contract — every other
/// serialisable type in this module carries the same attribute,
/// and a malicious SLM response that smuggles in an unknown
/// field must surface as a deserialisation error rather than
/// being silently dropped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyDecision {
    /// Top-level category label (e.g. `"adult"`, `"graphic"`,
    /// `"scam"`, `"benign"`, `"child_safety"`).
    pub category: String,
    /// Severity rank `0..=5` (PROPOSAL.md §9 rubric).
    pub severity: u8,
    /// UX action the renderer must apply.
    pub ux_action: UXAction,
    /// Stable dotted snake_case identifier
    /// (e.g. `"adult.explicit_sexual.severe"`). No free-form text.
    pub rationale_code: String,
    /// Whether the renderer may offer a "tap to reveal" affordance.
    #[serde(default = "default_true")]
    pub allow_reveal: bool,
    /// Whether the user may forward the media.
    #[serde(default = "default_true")]
    pub allow_forward: bool,
    /// True if the SLM was invoked for this decision. Useful for
    /// telemetry; the interpreter is the only writer.
    #[serde(default)]
    pub used_slm: bool,
    /// Where the decision came from.
    #[serde(default = "default_source")]
    pub source: DecisionSource,
}

fn default_true() -> bool {
    true
}

fn default_source() -> DecisionSource {
    DecisionSource::Rule
}

impl PolicyDecision {
    /// Construct a [`PolicyDecision`] with full validation. Mirrors
    /// the Python pydantic field validators. Use this rather than
    /// the struct literal so the rationale-code shape is enforced
    /// at every construction site.
    pub fn new(
        category: impl Into<String>,
        severity: u8,
        ux_action: UXAction,
        rationale_code: impl Into<String>,
    ) -> Result<Self, PolicyDecisionError> {
        let category = category.into();
        let rationale_code = rationale_code.into();
        validate_category(&category)?;
        validate_severity(severity)?;
        validate_rationale_code(&rationale_code)?;
        Ok(Self {
            category,
            severity,
            ux_action,
            rationale_code,
            allow_reveal: true,
            allow_forward: true,
            used_slm: false,
            source: DecisionSource::Rule,
        })
    }

    /// Fluent setter for `allow_reveal`.
    #[must_use]
    pub fn with_allow_reveal(mut self, allow: bool) -> Self {
        self.allow_reveal = allow;
        self
    }

    /// Fluent setter for `allow_forward`.
    #[must_use]
    pub fn with_allow_forward(mut self, allow: bool) -> Self {
        self.allow_forward = allow;
        self
    }

    /// Fluent setter for `used_slm`.
    #[must_use]
    pub fn with_used_slm(mut self, used: bool) -> Self {
        self.used_slm = used;
        self
    }

    /// Fluent setter for `source`.
    #[must_use]
    pub fn with_source(mut self, source: DecisionSource) -> Self {
        self.source = source;
        self
    }

    /// Re-validate every field (after manual mutation via the
    /// `pub` fields). Useful when reconstituting from JSON when
    /// the deserialisation path skipped pydantic-style validators.
    pub fn validate(&self) -> Result<(), PolicyDecisionError> {
        validate_category(&self.category)?;
        validate_severity(self.severity)?;
        validate_rationale_code(&self.rationale_code)?;
        Ok(())
    }
}

fn validate_category(category: &str) -> Result<(), PolicyDecisionError> {
    if category.trim().is_empty() {
        Err(PolicyDecisionError::EmptyCategory)
    } else {
        Ok(())
    }
}

fn validate_severity(severity: u8) -> Result<(), PolicyDecisionError> {
    if severity > 5 {
        Err(PolicyDecisionError::SeverityOutOfRange(severity as i32))
    } else {
        Ok(())
    }
}

/// Validate a rationale-code against the cross-platform shape.
///
/// Dotted snake_case, lowercase, starting with an alphabetic
/// character. Mirrors the Python regex
/// `r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$"`.
fn validate_rationale_code(code: &str) -> Result<(), PolicyDecisionError> {
    if code.is_empty() {
        return Err(PolicyDecisionError::InvalidRationaleCode(code.to_string()));
    }
    let mut chars = code.chars().peekable();
    // First char must be a-z.
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return Err(PolicyDecisionError::InvalidRationaleCode(code.to_string())),
    }
    // Subsequent: each segment is [a-z0-9_]+, segments separated
    // by literal '.'. Track whether we just saw a '.' so we can
    // reject leading/empty segments.
    let mut last_was_dot = false;
    for c in chars {
        if c == '.' {
            if last_was_dot {
                return Err(PolicyDecisionError::InvalidRationaleCode(code.to_string()));
            }
            last_was_dot = true;
        } else if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' {
            // After a dot, the segment must start with a-z|0-9|_.
            // (Python's `[a-z0-9_]+` allows digits / underscores
            // to start a segment.) The first character of the
            // *first* segment is enforced separately above.
            last_was_dot = false;
        } else {
            return Err(PolicyDecisionError::InvalidRationaleCode(code.to_string()));
        }
    }
    if last_was_dot {
        // Trailing dot.
        return Err(PolicyDecisionError::InvalidRationaleCode(code.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ux_action_round_trip_matches_python_strings() {
        assert_eq!(UXAction::Clear.as_str(), "clear");
        assert_eq!(UXAction::BlurTap.as_str(), "blur_tap");
        assert_eq!(UXAction::Pixelate.as_str(), "pixelate");
        assert_eq!(UXAction::BlockedCard.as_str(), "blocked_card");
        for variant in [
            UXAction::Clear,
            UXAction::BlurTap,
            UXAction::Pixelate,
            UXAction::BlockedCard,
        ] {
            assert_eq!(UXAction::from_str_strict(variant.as_str()), Ok(variant));
        }
    }

    #[test]
    fn ux_action_rejects_unknown_strings() {
        assert!(matches!(
            UXAction::from_str_strict("blur"),
            Err(UXActionParseError::Unknown(_))
        ));
        assert!(matches!(
            UXAction::from_str_strict("Clear"),
            Err(UXActionParseError::Unknown(_))
        ));
        assert!(matches!(
            UXAction::from_str_strict(""),
            Err(UXActionParseError::Unknown(_))
        ));
    }

    #[test]
    fn decision_source_strings_match_python_contract() {
        assert_eq!(DecisionSource::Rule.as_str(), "rule");
        assert_eq!(DecisionSource::Slm.as_str(), "slm");
        assert_eq!(
            DecisionSource::ChildSafetyFloor.as_str(),
            "child_safety_floor"
        );
    }

    #[test]
    fn ux_action_serde_uses_snake_case_strings() {
        let json = serde_json::to_string(&UXAction::BlurTap).unwrap();
        assert_eq!(json, "\"blur_tap\"");
        let parsed: UXAction = serde_json::from_str("\"pixelate\"").unwrap();
        assert_eq!(parsed, UXAction::Pixelate);
    }

    #[test]
    fn decision_source_serde_uses_snake_case_strings() {
        let json = serde_json::to_string(&DecisionSource::ChildSafetyFloor).unwrap();
        assert_eq!(json, "\"child_safety_floor\"");
    }

    #[test]
    fn policy_decision_new_validates_category() {
        let err = PolicyDecision::new("", 1, UXAction::Clear, "x.y").unwrap_err();
        assert_eq!(err, PolicyDecisionError::EmptyCategory);
        let err = PolicyDecision::new("   ", 1, UXAction::Clear, "x.y").unwrap_err();
        assert_eq!(err, PolicyDecisionError::EmptyCategory);
    }

    #[test]
    fn policy_decision_new_validates_severity_range() {
        let err = PolicyDecision::new("a", 6, UXAction::Clear, "x.y").unwrap_err();
        assert!(matches!(err, PolicyDecisionError::SeverityOutOfRange(6)));
    }

    #[test]
    fn policy_decision_new_validates_rationale_code_shape() {
        // Valid examples mirror cv-guard test fixtures.
        for code in [
            "benign",
            "adult.explicit_sexual.severe",
            "scam.crypto_wallet",
            "child_safety.csam",
            "a.b.c.d",
            "x123",
            "x_y",
            "x.y0",
        ] {
            assert!(
                PolicyDecision::new("c", 0, UXAction::Clear, code).is_ok(),
                "expected {code:?} to be a valid rationale_code"
            );
        }
        for code in [
            "",
            ".",
            "..",
            ".leading",
            "trailing.",
            "double..dot",
            "Uppercase",
            "0starts_with_digit",
            "has space",
            "has-dash",
            "ünîcødé",
            "_leading_underscore",
        ] {
            assert!(
                PolicyDecision::new("c", 0, UXAction::Clear, code).is_err(),
                "expected {code:?} to be rejected"
            );
        }
    }

    #[test]
    fn policy_decision_defaults_match_python() {
        let d = PolicyDecision::new("benign", 0, UXAction::Clear, "benign").unwrap();
        assert!(d.allow_reveal);
        assert!(d.allow_forward);
        assert!(!d.used_slm);
        assert_eq!(d.source, DecisionSource::Rule);
    }

    #[test]
    fn policy_decision_builders_override_defaults() {
        let d = PolicyDecision::new("scam", 3, UXAction::BlurTap, "scam.crypto_wallet")
            .unwrap()
            .with_allow_reveal(false)
            .with_allow_forward(false)
            .with_used_slm(true)
            .with_source(DecisionSource::Slm);
        assert!(!d.allow_reveal);
        assert!(!d.allow_forward);
        assert!(d.used_slm);
        assert_eq!(d.source, DecisionSource::Slm);
    }

    #[test]
    fn policy_decision_validate_catches_post_mutation_drift() {
        let mut d = PolicyDecision::new("benign", 0, UXAction::Clear, "benign").unwrap();
        d.severity = 9;
        assert!(matches!(
            d.validate(),
            Err(PolicyDecisionError::SeverityOutOfRange(9))
        ));
        d.severity = 0;
        d.category = "".to_string();
        assert_eq!(d.validate(), Err(PolicyDecisionError::EmptyCategory));
        d.category = "benign".to_string();
        d.rationale_code = "".to_string();
        assert!(matches!(
            d.validate(),
            Err(PolicyDecisionError::InvalidRationaleCode(_))
        ));
    }

    #[test]
    fn policy_decision_round_trips_through_serde_json() {
        let d = PolicyDecision::new(
            "child_safety",
            5,
            UXAction::BlockedCard,
            "child_safety.csam",
        )
        .unwrap()
        .with_allow_reveal(false)
        .with_allow_forward(false)
        .with_used_slm(false)
        .with_source(DecisionSource::ChildSafetyFloor);
        let json = serde_json::to_string(&d).unwrap();
        let parsed: PolicyDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, d);
    }

    #[test]
    fn policy_decision_serde_defaults_apply_on_missing_fields() {
        // Python pydantic ConfigDict(extra="forbid") would reject
        // extra fields but allow missing optional ones with the
        // declared default. Our `serde(default = ...)` attributes
        // mirror this — `allow_reveal` / `allow_forward` default
        // to true and `used_slm` to false and `source` to "rule".
        let json =
            r#"{"category":"benign","severity":0,"ux_action":"clear","rationale_code":"benign"}"#;
        let parsed: PolicyDecision = serde_json::from_str(json).unwrap();
        assert!(parsed.allow_reveal);
        assert!(parsed.allow_forward);
        assert!(!parsed.used_slm);
        assert_eq!(parsed.source, DecisionSource::Rule);
    }

    #[test]
    fn policy_decision_rejects_unknown_fields_on_deserialise() {
        // Python `ConfigDict(extra="forbid")` rejects any unknown
        // field; the Rust port mirrors that with
        // `#[serde(deny_unknown_fields)]`. The threat model is a
        // malicious SLM response (or compromised skill pack)
        // smuggling in an unrecognised key — without the deny
        // attribute, `from_str` silently drops the extra field and
        // returns a syntactically-valid decision, hiding the
        // injection from downstream observers.
        let json = r#"{
            "category": "benign",
            "severity": 0,
            "ux_action": "clear",
            "rationale_code": "benign",
            "injected_field": "smuggled_payload"
        }"#;
        let result: Result<PolicyDecision, _> = serde_json::from_str(json);
        let err = result.expect_err(
            "deny_unknown_fields must reject `injected_field` and surface a serde error",
        );
        // Sanity-check that the error names the offending field
        // so observers can attribute the rejection.
        assert!(
            err.to_string().contains("injected_field"),
            "expected serde error to mention the unknown field, got: {err}",
        );
    }

    #[test]
    fn policy_decision_rejects_unknown_fields_alongside_known_optional_defaults() {
        // Confirm `deny_unknown_fields` does not regress the
        // documented `serde(default = ...)` behaviour for the
        // four optional fields. The JSON below should still error
        // (because of the extra), proving that the deny attribute
        // is what triggers the rejection — not the missing
        // optional fields.
        let json = r#"{
            "category": "scam",
            "severity": 3,
            "ux_action": "blur_tap",
            "rationale_code": "scam.payment_request.suspected",
            "allow_reveal": true,
            "x_unknown": 1
        }"#;
        assert!(serde_json::from_str::<PolicyDecision>(json).is_err());
    }
}

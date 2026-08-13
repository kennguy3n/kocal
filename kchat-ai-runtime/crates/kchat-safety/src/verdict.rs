//! Verdict types — the output of the safety classification pipeline.
//!
//! The verdict is a structured result with stable reason codes. It never
//! contains raw message text. The deterministic path returns allow, warn,
//! block, redact, or require-consent.

use serde::{Deserialize, Serialize};

/// Action the application should take based on the verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Content is safe to process
    Allow,
    /// Content is borderline — show a warning but proceed
    Warn,
    /// Content violates policy — block and show reason
    Block,
    /// Content contains sensitive data — redact before processing
    Redact,
    /// Content requires user consent before processing (e.g. sensitive topic)
    RequireConsent,
}

/// The source of the verdict — which layer produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerdictSource {
    /// Deterministic rules only (fastest, always available)
    Deterministic,
    /// Compact encoder classifier (ONNX INT8/INT4)
    Encoder,
    /// SLM adjudication (llama.cpp, medium/high tier only)
    Slm,
    /// Degraded mode — deterministic fallback when encoder/SLM unavailable
    Degraded,
}

/// Severity level 1-5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Severity(pub u8);

impl Severity {
    pub const SAFE: Severity = Severity(0);
    pub const BENIGN: Severity = Severity(1);
    pub const BORDERLINE: Severity = Severity(2);
    pub const SEVERE: Severity = Severity(3);
    pub const HIGH: Severity = Severity(4);
    pub const CRITICAL: Severity = Severity(5);

    /// Maximum valid severity value.
    pub const MAX: u8 = 5;

    /// Create a Severity from a u8, returning an error if out of range (0-5).
    pub fn from_u8(value: u8) -> Result<Self, &'static str> {
        if value > Self::MAX {
            Err("severity out of range (0-5)")
        } else {
            Ok(Severity(value))
        }
    }

    /// Returns the raw u8 value.
    pub fn raw(self) -> u8 {
        self.0
    }

    pub fn is_safe(self) -> bool {
        self.0 <= 1
    }
}

/// A structured safety verdict with stable reason codes.
///
/// This struct never contains raw message text. Reason codes are stable
/// identifiers that the application can use for UI, logging, and telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    /// Action to take
    pub action: Action,
    /// Severity level (0=safe, 5=critical)
    pub severity: Severity,
    /// Risk category (numeric ID from taxonomy)
    pub category: u32,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Stable reason codes for UI, logging, and telemetry
    pub reason_codes: Vec<String>,
    /// Which layer produced this verdict
    pub source: VerdictSource,
    /// Whether the encoder was consulted
    pub used_encoder: bool,
    /// Whether the SLM was consulted
    pub used_slm: bool,
    /// Rationale ID for appeal/support flows
    pub rationale_id: String,
    /// Optional resource link for user education
    pub resource_link_id: Option<String>,
}

impl Verdict {
    /// Create an "allow" verdict from deterministic rules.
    pub fn allow() -> Self {
        Self {
            action: Action::Allow,
            severity: Severity::SAFE,
            category: 0,
            confidence: 1.0,
            reason_codes: vec!["deterministic_pass".into()],
            source: VerdictSource::Deterministic,
            used_encoder: false,
            used_slm: false,
            rationale_id: String::new(),
            resource_link_id: None,
        }
    }

    /// Create a verdict from the builder.
    pub fn builder() -> VerdictBuilder {
        VerdictBuilder::default()
    }
}

/// Builder for constructing verdicts in the pipeline.
#[derive(Debug, Clone, Default)]
pub struct VerdictBuilder {
    action: Option<Action>,
    severity: Option<Severity>,
    category: Option<u32>,
    confidence: Option<f64>,
    reason_codes: Vec<String>,
    source: Option<VerdictSource>,
    used_encoder: bool,
    used_slm: bool,
    rationale_id: String,
    resource_link_id: Option<String>,
}

impl VerdictBuilder {
    pub fn action(mut self, action: Action) -> Self {
        self.action = Some(action);
        self
    }

    pub fn severity(mut self, severity: Severity) -> Self {
        self.severity = Some(severity);
        self
    }

    pub fn category(mut self, category: u32) -> Self {
        self.category = Some(category);
        self
    }

    pub fn confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn reason_code(mut self, code: impl Into<String>) -> Self {
        self.reason_codes.push(code.into());
        self
    }

    pub fn source(mut self, source: VerdictSource) -> Self {
        self.source = Some(source);
        self
    }

    pub fn used_encoder(mut self, used: bool) -> Self {
        self.used_encoder = used;
        self
    }

    pub fn used_slm(mut self, used: bool) -> Self {
        self.used_slm = used;
        self
    }

    pub fn rationale_id(mut self, id: impl Into<String>) -> Self {
        self.rationale_id = id.into();
        self
    }

    pub fn build(self) -> Verdict {
        Verdict {
            action: self.action.unwrap_or(Action::Allow),
            severity: self.severity.unwrap_or(Severity::SAFE),
            category: self.category.unwrap_or(0),
            confidence: self.confidence.unwrap_or(1.0),
            reason_codes: self.reason_codes,
            source: self.source.unwrap_or(VerdictSource::Deterministic),
            used_encoder: self.used_encoder,
            used_slm: self.used_slm,
            rationale_id: self.rationale_id,
            resource_link_id: self.resource_link_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allow_verdict() {
        let v = Verdict::allow();
        assert_eq!(v.action, Action::Allow);
        assert!(v.severity.is_safe());
        assert_eq!(v.source, VerdictSource::Deterministic);
    }

    #[test]
    fn test_builder() {
        let v = Verdict::builder()
            .action(Action::Block)
            .severity(Severity::CRITICAL)
            .category(1)
            .confidence(0.95)
            .reason_code("child_safety")
            .source(VerdictSource::Deterministic)
            .build();
        assert_eq!(v.action, Action::Block);
        assert_eq!(v.severity, Severity::CRITICAL);
        assert_eq!(v.category, 1);
        assert!(!v.reason_codes.is_empty());
    }

    #[test]
    fn test_severity_from_u8() {
        assert_eq!(Severity::from_u8(0).unwrap(), Severity::SAFE);
        assert_eq!(Severity::from_u8(5).unwrap(), Severity::CRITICAL);
        assert!(Severity::from_u8(6).is_err());
        assert!(Severity::from_u8(255).is_err());
    }

    #[test]
    fn test_severity_raw() {
        assert_eq!(Severity::CRITICAL.raw(), 5);
        assert_eq!(Severity::SAFE.raw(), 0);
    }
}

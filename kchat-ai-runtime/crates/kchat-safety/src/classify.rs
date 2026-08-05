//! Safety classifier — the main entry point for the safety plane.
//!
//! Implements the 6-step guardrail flow:
//! 1. Normalize text locally after decryption
//! 2. Apply signed deterministic policy, allowlists, blocklists, rate signals
//! 3. If confidence is insufficient, run the compact encoder
//! 4. On eligible medium/high devices, invoke the SLM for ambiguous cases
//! 5. Apply deterministic policy to the structured result
//! 6. Return allow, warn, block, redact, or require-consent with reason codes
//!
//! The deterministic path P95 target is <5ms. The encoder path P95 target
//! is <150ms on qualified devices.

use crate::detectors;
use crate::normalize;
use crate::policy::{PolicyPack, PolicyThresholds};
use crate::verdict::{Action, Severity, Verdict, VerdictBuilder, VerdictSource};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::Instant;

/// Request for safety classification.
#[derive(Debug, Clone)]
pub struct ClassifyRequest {
    /// Message text (already decrypted)
    pub text: String,
    /// Whether this is a group conversation
    pub is_group: bool,
    /// Age mode if applicable (e.g. "minor", "adult")
    pub age_mode: Option<String>,
    /// Relationship context if known
    pub relationship: Option<String>,
    /// Whether the encoder is available (medium+ tier)
    pub encoder_available: bool,
    /// Whether the SLM is available (medium+ tier)
    pub slm_available: bool,
}

impl ClassifyRequest {
    /// Simple request from just text (deterministic-only).
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_group: false,
            age_mode: None,
            relationship: None,
            encoder_available: false,
            slm_available: false,
        }
    }

    /// Enable encoder and SLM (medium/high tier).
    pub fn with_encoder(mut self) -> Self {
        self.encoder_available = true;
        self.slm_available = true;
        self
    }
}

/// Result of safety classification.
#[derive(Debug, Clone)]
pub struct ClassifyResult {
    /// The final verdict
    pub verdict: Verdict,
    /// Time taken in microseconds (for telemetry, not content)
    pub duration_us: u64,
}

/// The safety classifier — owns loaded policy packs and optional encoder/SLM.
pub struct SafetyClassifier {
    policy_packs: RwLock<Vec<Arc<PolicyPack>>>,
    /// Optional encoder (ONNX classifier) — set on medium+ devices
    encoder: RwLock<Option<Box<dyn EncoderAdapter>>>,
    /// Optional SLM adjudicator — set on medium+ devices
    slm: RwLock<Option<Box<dyn SlmAdjudicator>>>,
}

/// Trait for encoder-based classification (ONNX INT8/INT4).
pub trait EncoderAdapter: Send + Sync {
    /// Classify text and return a verdict.
    fn classify(&self, text: &str) -> Result<EncoderVerdict, EncoderError>;
}

/// Verdict from the encoder.
#[derive(Debug, Clone)]
pub struct EncoderVerdict {
    pub category: u32,
    pub confidence: f64,
}

/// Error from the encoder.
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("encoder inference failed: {0}")]
    InferenceFailed(String),
    #[error("encoder not loaded")]
    NotLoaded,
}

/// Trait for SLM-based adjudication (llama.cpp with grammar-constrained JSON).
pub trait SlmAdjudicator: Send + Sync {
    /// Adjudicate an ambiguous case and return a structured decision.
    fn adjudicate(&self, text: &str, signal_json: &str) -> Result<SlmDecision, SlmError>;
}

/// SLM decision (closed JSON grammar output).
#[derive(Debug, Clone)]
pub struct SlmDecision {
    pub category: u32,
    pub severity: u8,
    pub action: Action,
    pub confidence: f64,
    pub rationale_code: String,
}

/// Error from the SLM.
#[derive(Debug, thiserror::Error)]
pub enum SlmError {
    #[error("SLM inference failed: {0}")]
    InferenceFailed(String),
    #[error("SLM not loaded")]
    NotLoaded,
    #[error("SLM output invalid: {0}")]
    InvalidOutput(String),
}

impl SafetyClassifier {
    /// Create a new classifier with no policy packs (deterministic-only mode).
    pub fn new() -> Self {
        Self {
            policy_packs: RwLock::new(Vec::new()),
            encoder: RwLock::new(None),
            slm: RwLock::new(None),
        }
    }

    /// Load a signed policy pack.
    pub fn load_policy_pack(&self, pack: Arc<PolicyPack>) {
        self.policy_packs.write().push(pack);
    }

    /// Attach an encoder (ONNX classifier) — medium+ tier only.
    pub fn attach_encoder(&self, encoder: Box<dyn EncoderAdapter>) {
        *self.encoder.write() = Some(encoder);
    }

    /// Attach an SLM adjudicator — medium+ tier only.
    pub fn attach_slm(&self, slm: Box<dyn SlmAdjudicator>) {
        *self.slm.write() = Some(slm);
    }

    /// Classify a message through the full guardrail pipeline.
    ///
    /// This is the main entry point. It runs the 6-step flow:
    /// 1. Normalize
    /// 2. Deterministic rules
    /// 3. Encoder (if needed and available)
    /// 4. SLM (if needed and available)
    /// 5. Deterministic policy on result
    /// 6. Return verdict
    pub fn classify(&self, request: &ClassifyRequest) -> ClassifyResult {
        let start = Instant::now();

        // Step 1: Normalize — two levels for different detector types
        let pattern_text = normalize::normalize_for_patterns(&request.text);
        let lexicon_text = normalize::normalize(&request.text);

        // Step 2: Run deterministic detectors
        let lexicon = self.build_lexicon();
        let signals = detectors::run_all_detectors(&pattern_text, &lexicon_text, &lexicon);

        // Resolve deterministic verdict
        let verdict = if let Some(signal) = detectors::resolve_priority_chain(&signals) {
            // Deterministic match found
            let mut builder = VerdictBuilder::default()
                .action(signal.action)
                .severity(signal.severity)
                .category(signal.category)
                .confidence(signal.confidence)
                .reason_code(&signal.reason_code)
                .source(VerdictSource::Deterministic);

            // Step 3: If confidence is below the encoder escalation threshold,
            // and encoder is available, escalate
            let thresholds = self.get_thresholds();
            if signal.confidence < thresholds.encoder_escalation_threshold
                && request.encoder_available
            {
                if let Some(encoder) = self.encoder.read().as_ref() {
                    if let Ok(enc_verdict) = encoder.classify(&pattern_text) {
                        builder = builder
                            .used_encoder(true)
                            .confidence(enc_verdict.confidence)
                            .category(enc_verdict.category);

                        // Step 4: If still ambiguous and SLM is available, adjudicate
                        if enc_verdict.confidence < thresholds.warn_threshold
                            && request.slm_available
                        {
                            if let Some(slm) = self.slm.read().as_ref() {
                                let signal_json = serde_json::json!({
                                    "category": enc_verdict.category,
                                    "confidence": enc_verdict.confidence,
                                    "is_group": request.is_group,
                                    "age_mode": request.age_mode,
                                })
                                .to_string();

                                if let Ok(slm_decision) = slm.adjudicate(&pattern_text, &signal_json) {
                                    builder = builder
                                        .used_slm(true)
                                        .action(slm_decision.action)
                                        .severity(Severity(slm_decision.severity))
                                        .confidence(slm_decision.confidence)
                                        .reason_code(&slm_decision.rationale_code)
                                        .source(VerdictSource::Slm);
                                }
                            }
                        } else {
                            builder = builder.source(VerdictSource::Encoder);
                        }
                    }
                }
            }

            builder.build()
        } else {
            // No deterministic match — check if we need encoder for safety
            let mut degraded = false;
            let thresholds = self.get_thresholds();

            // For certain contexts (group, minor), always run encoder if available
            let needs_encoder = request.encoder_available
                && (request.is_group
                    || request.age_mode.as_deref() == Some("minor")
                    || self.has_high_risk_indicators(&lexicon_text));

            if needs_encoder {
                if let Some(encoder) = self.encoder.read().as_ref() {
                    match encoder.classify(&pattern_text) {
                        Ok(enc_verdict) => {
                            let action = if enc_verdict.confidence >= thresholds.block_threshold {
                                Action::Block
                            } else if enc_verdict.confidence >= thresholds.warn_threshold {
                                Action::Warn
                            } else {
                                Action::Allow
                            };

                            let severity = if enc_verdict.confidence >= thresholds.block_threshold {
                                Severity::SEVERE
                            } else if enc_verdict.confidence >= thresholds.warn_threshold {
                                Severity::BORDERLINE
                            } else {
                                Severity::SAFE
                            };

                            return ClassifyResult {
                                verdict: VerdictBuilder::default()
                                    .action(action)
                                    .severity(severity)
                                    .category(enc_verdict.category)
                                    .confidence(enc_verdict.confidence)
                                    .reason_code("encoder_classification")
                                    .source(VerdictSource::Encoder)
                                    .used_encoder(true)
                                    .build(),
                                duration_us: start.elapsed().as_micros() as u64,
                            };
                        }
                        Err(_) => {
                            // Encoder failed — mark as degraded and fall through
                            // to deterministic verdict
                            degraded = true;
                        }
                    }
                }
            }

            // No match and no encoder needed → allow (or degraded if encoder failed)
            if degraded {
                VerdictBuilder::default()
                    .action(Action::Allow)
                    .source(VerdictSource::Degraded)
                    .build()
            } else {
                Verdict::allow()
            }
        };

        ClassifyResult {
            verdict,
            duration_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Build a lexicon from all loaded policy packs.
    /// Deduplicates terms across packs (first occurrence wins).
    fn build_lexicon(&self) -> Vec<(String, u32, Severity)> {
        let packs = self.policy_packs.read();
        let mut seen = std::collections::HashSet::new();
        let mut lexicon = Vec::new();

        for pack in packs.iter() {
            for rule in &pack.rules {
                let cat = rule.category.as_u32();
                let sev = crate::policy::severity_from_u8(rule.severity);
                for term in &rule.lexicon {
                    // Normalize to lowercase for consistent matching
                    let lower_term = term.to_lowercase();
                    if seen.insert(lower_term.clone()) {
                        lexicon.push((lower_term, cat, sev));
                    }
                }
            }
        }

        lexicon
    }

    /// Get the thresholds from loaded policy packs.
    /// Merges by taking the most conservative (highest) thresholds across all packs.
    fn get_thresholds(&self) -> PolicyThresholds {
        let packs = self.policy_packs.read();
        if packs.is_empty() {
            return PolicyThresholds::default();
        }
        // Merge: take the most conservative threshold from all packs.
        // For warn/block, higher = more conservative (trigger more often).
        // For encoder_escalation, lower = more conservative (escalate more often).
        packs.iter().skip(1).fold(packs[0].thresholds.clone(), |acc, pack| {
            PolicyThresholds {
                warn_threshold: acc.warn_threshold.max(pack.thresholds.warn_threshold),
                block_threshold: acc.block_threshold.max(pack.thresholds.block_threshold),
                encoder_escalation_threshold: acc.encoder_escalation_threshold.min(pack.thresholds.encoder_escalation_threshold),
            }
        })
    }

    /// Check for high-risk indicators that warrant encoder escalation.
    fn has_high_risk_indicators(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        // Simple heuristic: certain keywords suggest encoder review is warranted
        let indicators = [
            "kill", "hurt", "die", "suicide", "self-harm",
            "nude", "nsfw", "sexual",
            "weapon", "gun", "bomb",
            "drug", "illegal",
        ];
        indicators.iter().any(|i| lower.contains(i))
    }

    /// Check if the classifier is in deterministic-only mode (no encoder/SLM).
    pub fn is_deterministic_only(&self) -> bool {
        self.encoder.read().is_none() && self.slm.read().is_none()
    }

    /// Number of loaded policy packs.
    pub fn pack_count(&self) -> usize {
        self.policy_packs.read().len()
    }
}

impl Default for SafetyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detectors::categories;

    struct MockEncoder {
        category: u32,
        confidence: f64,
    }

    impl EncoderAdapter for MockEncoder {
        fn classify(&self, _text: &str) -> Result<EncoderVerdict, EncoderError> {
            Ok(EncoderVerdict {
                category: self.category,
                confidence: self.confidence,
            })
        }
    }

    #[test]
    fn test_deterministic_only_allows_safe_text() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("Hello, how are you?");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Allow);
        assert_eq!(result.verdict.source, VerdictSource::Deterministic);
    }

    #[test]
    fn test_deterministic_blocks_pii() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("my card is 4111 1111 1111 1111");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Redact);
        assert_eq!(result.verdict.source, VerdictSource::Deterministic);
    }

    #[test]
    fn test_deterministic_warns_scam() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("URGENT! Send money via bitcoin immediately!");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Warn);
    }

    #[test]
    fn test_encoder_escalation_for_group() {
        let classifier = SafetyClassifier::new();
        classifier.attach_encoder(Box::new(MockEncoder {
            category: categories::SAFE,
            confidence: 0.95,
        }));

        let req = ClassifyRequest {
            text: "Hello everyone".into(),
            is_group: true,
            age_mode: None,
            relationship: None,
            encoder_available: true,
            slm_available: false,
        };

        let result = classifier.classify(&req);
        assert!(result.verdict.used_encoder);
        assert_eq!(result.verdict.source, VerdictSource::Encoder);
    }

    #[test]
    fn test_encoder_escalation_for_minor() {
        let classifier = SafetyClassifier::new();
        classifier.attach_encoder(Box::new(MockEncoder {
            category: categories::SAFE,
            confidence: 0.90,
        }));

        let req = ClassifyRequest {
            text: "What is the meaning of life?".into(),
            is_group: false,
            age_mode: Some("minor".into()),
            relationship: None,
            encoder_available: true,
            slm_available: false,
        };

        let result = classifier.classify(&req);
        assert!(result.verdict.used_encoder);
    }

    #[test]
    fn test_deterministic_only_on_low_tier() {
        let classifier = SafetyClassifier::new();
        assert!(classifier.is_deterministic_only());

        let req = ClassifyRequest::from_text("Hello");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.source, VerdictSource::Deterministic);
        assert!(!result.verdict.used_encoder);
        assert!(!result.verdict.used_slm);
    }

    #[test]
    fn test_latency_target_deterministic() {
        // Deterministic path P95 target: <5ms = <5000us
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("Hello, how are you today?");
        let result = classifier.classify(&req);
        // On a fast machine this should be well under 5ms
        // (On CI it might be slower, so we use a generous bound)
        assert!(
            result.duration_us < 50_000,
            "deterministic path took {}us, expected <50000us",
            result.duration_us
        );
    }
}

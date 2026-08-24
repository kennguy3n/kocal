//! Safety encoder — ONNX Runtime-based ML classifier for ambiguous cases.
//!
//! The encoder provides ML-based escalation for cases that deterministic
//! detectors can't confidently classify. It's used on Medium+ tier devices
//! when the deterministic pipeline returns Allow/Warn for group/minor contexts.
//!
//! The encoder uses the unified kchat-encoder crate (XLM-RoBERTa-base)
//! which provides a shared ONNX session for safety classification, text
//! embedding, and cross-encoder reranking.
//!
//! Categories (matching the deterministic pipeline):
//! - 0: SAFE
//! - 1: HARASSMENT
//! - 2: HATE_SPEECH
//! - 3: SELF_HARM
//! - 4: VIOLENCE
//! - 5: SEXUAL_CONTENT
//! - 6: CHILD_SAFETY
//! - 7: SCAM
//! - 8: PII
//! - 9: URL_RISK

use crate::classify::{EncoderAdapter, EncoderError, EncoderVerdict};

/// Re-export category constants from kchat-encoder for convenience.
pub use kchat_encoder::categories;

/// ONNX Runtime safety encoder using the unified kchat-encoder.
///
/// Wraps a shared `kchat_encoder::EncoderSession` and delegates
/// classification to `kchat_encoder::SafetyHead`.
#[cfg(feature = "onnx-runtime")]
pub struct OnnxEncoder {
    session: std::sync::Arc<kchat_encoder::EncoderSession>,
}

#[cfg(feature = "onnx-runtime")]
impl OnnxEncoder {
    /// Create a new ONNX encoder from a shared encoder session.
    pub fn new(session: std::sync::Arc<kchat_encoder::EncoderSession>) -> Self {
        Self { session }
    }

    /// Create a new ONNX encoder by loading a model from file.
    ///
    /// `intra_threads` controls ONNX Runtime intra-op parallelism (2 for low, 3 for medium, 4+ for high).
    pub fn from_files(
        model_path: &str,
        tokenizer_path: &str,
        quantization: kchat_encoder::Quantization,
        intra_threads: usize,
    ) -> Result<Self, EncoderError> {
        let session = kchat_encoder::EncoderSession::new(model_path, tokenizer_path, quantization, intra_threads)
            .map_err(|e| EncoderError::InferenceFailed(format!("encoder session: {e}")))?;
        Ok(Self {
            session: std::sync::Arc::new(session),
        })
    }
}

#[cfg(feature = "onnx-runtime")]
impl EncoderAdapter for OnnxEncoder {
    fn classify(&self, text: &str) -> Result<EncoderVerdict, EncoderError> {
        let head = kchat_encoder::SafetyHead::new(&self.session);
        let verdict = head
            .classify(text)
            .map_err(|e| EncoderError::InferenceFailed(format!("safety head: {e}")))?;
        Ok(EncoderVerdict {
            category: verdict.category,
            confidence: verdict.confidence,
        })
    }
}

/// Mock encoder for testing — returns a fixed category and confidence.
pub struct MockEncoder {
    category: u32,
    confidence: f64,
}

impl MockEncoder {
    pub fn new(category: u32, confidence: f64) -> Self {
        Self { category, confidence }
    }

    /// Create a mock encoder that always returns SAFE.
    pub fn safe() -> Self {
        Self::new(categories::SAFE, 0.95)
    }

    /// Create a mock encoder that always returns a specific harm category.
    pub fn harmful(category: u32) -> Self {
        Self::new(category, 0.90)
    }
}

impl EncoderAdapter for MockEncoder {
    fn classify(&self, _text: &str) -> Result<EncoderVerdict, EncoderError> {
        Ok(EncoderVerdict {
            category: self.category,
            confidence: self.confidence,
        })
    }
}

/// SLM (Small Language Model) adjudicator — uses a generative model for
/// ambiguous cases that the encoder can't confidently classify.
///
/// The SLM is called with a structured prompt and constrained to output
/// JSON with category, severity, action, and confidence fields.
pub trait SlmAdjudicator: Send + Sync {
    /// Adjudicate a text using the SLM.
    ///
    /// `signal_json` contains the deterministic detector signals for context.
    fn adjudicate(&self, text: &str, signal_json: &str) -> Result<SlmDecision, EncoderError>;
}

/// Decision from the SLM adjudicator.
#[derive(Debug, Clone)]
pub struct SlmDecision {
    pub category: u32,
    pub severity: u32,
    pub action: String,
    pub confidence: f64,
    pub reasoning: String,
}

/// Mock SLM adjudicator for testing.
pub struct MockSlmAdjudicator {
    decision: SlmDecision,
}

impl MockSlmAdjudicator {
    pub fn new(decision: SlmDecision) -> Self {
        Self { decision }
    }
}

impl SlmAdjudicator for MockSlmAdjudicator {
    fn adjudicate(&self, _text: &str, _signal_json: &str) -> Result<SlmDecision, EncoderError> {
        Ok(self.decision.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_encoder_safe() {
        let encoder = MockEncoder::safe();
        let verdict = encoder.classify("hello world").unwrap();
        assert_eq!(verdict.category, categories::SAFE);
        assert!(verdict.confidence > 0.9);
    }

    #[test]
    fn test_mock_encoder_harmful() {
        let encoder = MockEncoder::harmful(categories::VIOLENCE);
        let verdict = encoder.classify("harmful text").unwrap();
        assert_eq!(verdict.category, categories::VIOLENCE);
    }

    #[test]
    fn test_category_names() {
        assert_eq!(categories::name(categories::SAFE), "safe");
        assert_eq!(categories::name(categories::HARASSMENT), "harassment");
        assert_eq!(categories::name(categories::CHILD_SAFETY), "child_safety");
        assert_eq!(categories::name(999), "unknown");
    }

    #[test]
    fn test_num_categories() {
        assert_eq!(categories::NUM_CATEGORIES, 17);
    }

    #[test]
    fn test_mock_slm_adjudicator() {
        let decision = SlmDecision {
            category: categories::HATE_SPEECH,
            severity: 2,
            action: "block".into(),
            confidence: 0.92,
            reasoning: "contains slur".into(),
        };
        let slm = MockSlmAdjudicator::new(decision);

        let result = slm.adjudicate("hateful text", "{}").unwrap();
        assert_eq!(result.category, categories::HATE_SPEECH);
        assert_eq!(result.action, "block");
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_slm_decision_fields() {
        let decision = SlmDecision {
            category: categories::SCAM,
            severity: 1,
            action: "warn".into(),
            confidence: 0.75,
            reasoning: "suspicious link".into(),
        };
        assert_eq!(decision.category, categories::SCAM);
        assert_eq!(decision.severity, 1);
        assert_eq!(decision.action, "warn");
    }
}

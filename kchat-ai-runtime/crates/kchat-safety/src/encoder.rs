//! Safety encoder — ONNX Runtime-based ML classifier for ambiguous cases.
//!
//! The encoder provides ML-based escalation for cases that deterministic
//! detectors can't confidently classify. It's used on Medium+ tier devices
//! when the deterministic pipeline returns Allow/Warn for group/minor contexts.
//!
//! The encoder loads an INT8 quantized ONNX model (~45MB) and tokenizes text
//! using the model's tokenizer. It runs inference and maps the output logits
//! to safety categories.
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

/// Category labels for the safety encoder.
pub mod categories {
    pub const SAFE: u32 = 0;
    pub const HARASSMENT: u32 = 1;
    pub const HATE_SPEECH: u32 = 2;
    pub const SELF_HARM: u32 = 3;
    pub const VIOLENCE: u32 = 4;
    pub const SEXUAL_CONTENT: u32 = 5;
    pub const CHILD_SAFETY: u32 = 6;
    pub const SCAM: u32 = 7;
    pub const PII: u32 = 8;
    pub const URL_RISK: u32 = 9;

    pub const NUM_CATEGORIES: usize = 10;

    /// Get the category name for a category ID.
    pub fn name(id: u32) -> &'static str {
        match id {
            SAFE => "safe",
            HARASSMENT => "harassment",
            HATE_SPEECH => "hate_speech",
            SELF_HARM => "self_harm",
            VIOLENCE => "violence",
            SEXUAL_CONTENT => "sexual_content",
            CHILD_SAFETY => "child_safety",
            SCAM => "scam",
            PII => "pii",
            URL_RISK => "url_risk",
            _ => "unknown",
        }
    }
}

/// ONNX Runtime safety encoder.
///
/// Loads an INT8 quantized ONNX model and runs classification on input text.
/// The model should have a single output with logits for each category.
#[cfg(feature = "onnx-runtime")]
pub struct OnnxEncoder {
    session: parking_lot::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    model_name: String,
    max_length: usize,
}

#[cfg(feature = "onnx-runtime")]
impl OnnxEncoder {
    /// Create a new ONNX encoder from a model file and tokenizer.
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self, EncoderError> {
        let session = ort::session::Session::builder()
            .map_err(|e| EncoderError::InferenceFailed(format!("session builder: {e}")))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| EncoderError::InferenceFailed(format!("optimization level: {e}")))?
            .with_intra_threads(2)
            .map_err(|e| EncoderError::InferenceFailed(format!("intra threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| EncoderError::InferenceFailed(format!("load model: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EncoderError::InferenceFailed(format!("tokenizer: {e}")))?;

        Ok(Self {
            session: parking_lot::Mutex::new(session),
            tokenizer,
            model_name: "safety-classifier-int8".into(),
            max_length: 512,
        })
    }

    /// Softmax function to convert logits to probabilities.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        if sum == 0.0 {
            return vec![1.0 / logits.len() as f32; logits.len()];
        }
        exp.iter().map(|e| e / sum).collect()
    }
}

#[cfg(feature = "onnx-runtime")]
impl EncoderAdapter for OnnxEncoder {
    fn classify(&self, text: &str) -> Result<EncoderVerdict, EncoderError> {
        use ndarray::Array2;

        // Truncate text to max_length tokens
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EncoderError::InferenceFailed(format!("tokenize: {e}")))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Truncate to max_length
        let seq_len = input_ids.len().min(self.max_length);
        let input_ids = &input_ids[..seq_len];
        let attention_mask = &attention_mask[..seq_len];

        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids.iter().map(|&v| v as i64).collect())
            .map_err(|e| EncoderError::InferenceFailed(format!("array: {e}")))?;
        let attention_arr = Array2::from_shape_vec((1, seq_len), attention_mask.iter().map(|&v| v as i64).collect())
            .map_err(|e| EncoderError::InferenceFailed(format!("array: {e}")))?;

        let input_ids_tensor = ort::value::Tensor::from_array(input_ids_arr)
            .map_err(|e| EncoderError::InferenceFailed(format!("input_ids tensor: {e}")))?;
        let attention_tensor = ort::value::Tensor::from_array(attention_arr)
            .map_err(|e| EncoderError::InferenceFailed(format!("attention tensor: {e}")))?;

        let inputs = ort::inputs! {
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_tensor,
        };

        let mut session = self.session.lock();
        let outputs = session
            .run(inputs)
            .map_err(|e| EncoderError::InferenceFailed(format!("run: {e}")))?;

        // Extract logits — ort 2.0.0-rc.10 returns (Shape, &[f32])
        let (logits_shape, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| EncoderError::InferenceFailed(format!("extract: {e}")))?;

        let dims: &[i64] = logits_shape;
        let _num_classes = dims.last().copied().unwrap_or(0) as usize;

        // Get logits for the first (and only) sequence
        let logits_slice: Vec<f32> = if dims.len() == 2 {
            // [1, num_classes]
            logits_data.to_vec()
        } else if dims.len() == 3 {
            // [batch, seq, hidden] → take last token
            let hidden = dims[2] as usize;
            let offset = (seq_len - 1) * hidden;
            logits_data[offset..offset + hidden].to_vec()
        } else {
            return Err(EncoderError::InferenceFailed(format!(
                "unexpected output shape: {dims:?}"
            )));
        };

        if logits_slice.len() < categories::NUM_CATEGORIES {
            return Err(EncoderError::InferenceFailed(format!(
                "expected >= {} classes, got {}",
                categories::NUM_CATEGORIES,
                logits_slice.len()
            )));
        }

        // Softmax to get probabilities
        let probs = Self::softmax(&logits_slice[..categories::NUM_CATEGORIES]);

        // Find the category with highest probability
        let (best_idx, best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, p)| (i as u32, *p as f64))
            .unwrap_or((0, 0.0));

        Ok(EncoderVerdict {
            category: best_idx,
            confidence: best_prob,
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
        assert_eq!(categories::NUM_CATEGORIES, 10);
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

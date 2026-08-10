//! Safety classification head — 10-class safety category classifier.
//!
//! Uses the shared encoder session's forward pass. If the ONNX model exports
//! `safety_logits` directly, uses those. Otherwise, falls back to
//! attention-mask-weighted mean pooling over hidden states and takes the
//! first NUM_CATEGORIES dimensions as a projection.

use crate::categories;
use crate::{EncoderError, EncoderResult, SafetyVerdict};

/// Safety classification head — wraps a shared encoder session.
pub struct SafetyHead<'a> {
    session: &'a crate::session::EncoderSession,
}

impl<'a> SafetyHead<'a> {
    /// Create a new safety head borrowing a shared encoder session.
    pub fn new(session: &'a crate::session::EncoderSession) -> Self {
        Self { session }
    }

    /// Classify text into one of 10 safety categories.
    pub fn classify(&self, text: &str) -> EncoderResult<SafetyVerdict> {
        let output = self.session.forward(text)?;

        // If the ONNX model exports safety_logits directly, use them.
        let logits = if let Some(ref safety_logits) = output.safety_logits {
            if safety_logits.len() < categories::NUM_CATEGORIES {
                return Err(EncoderError::InferenceFailed(format!(
                    "safety_logits has {} values, expected >= {}",
                    safety_logits.len(),
                    categories::NUM_CATEGORIES
                )));
            }
            safety_logits[..categories::NUM_CATEGORIES].to_vec()
        } else {
            // Fall back: CLS-pool hidden states (matches ONNX export's pooling strategy)
            // and use first N as logits.
            let pooled = self.session.cls_pool(&output.hidden);
            if pooled.len() < categories::NUM_CATEGORIES {
                return Err(EncoderError::InferenceFailed(format!(
                    "pooled output has {} values, expected >= {}",
                    pooled.len(),
                    categories::NUM_CATEGORIES
                )));
            }
            pooled[..categories::NUM_CATEGORIES].to_vec()
        };

        let probs = crate::session::EncoderSession::softmax(&logits);

        let (best_idx, best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, p)| (i as u32, *p as f64))
            .unwrap_or((0, 0.0));

        Ok(SafetyVerdict {
            category: best_idx,
            confidence: best_prob,
        })
    }
}

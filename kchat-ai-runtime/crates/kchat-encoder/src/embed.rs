//! Text embedding head — 768-dim dense vectors for retrieval.
//!
//! Uses the shared encoder session's forward pass. If the ONNX model exports
//! `embedding` directly, uses that. Otherwise, falls back to attention-mask-
//! weighted mean pooling over hidden states and L2 normalization.

use crate::EncoderResult;
use crate::session::EncoderSession;

/// Embedding head — wraps a shared encoder session.
pub struct EmbedHead<'a> {
    session: &'a EncoderSession,
}

impl<'a> EmbedHead<'a> {
    /// Create a new embedding head borrowing a shared encoder session.
    pub fn new(session: &'a EncoderSession) -> Self {
        Self { session }
    }

    /// Embed text into a 768-dim L2-normalized vector.
    pub fn embed(&self, text: &str) -> EncoderResult<Vec<f32>> {
        let output = self.session.forward(text)?;

        // If the ONNX model exports embedding directly, use it.
        if let Some(ref embedding) = output.embedding {
            let mut emb = embedding.clone();
            EncoderSession::l2_normalize(&mut emb);
            return Ok(emb);
        }

        // Fall back: CLS-pool hidden states and L2 normalize.
        // This matches the ONNX export's pooling strategy (last_hidden_state[:, 0, :]).
        let mut pooled = self.session.cls_pool(&output.hidden);
        EncoderSession::l2_normalize(&mut pooled);
        Ok(pooled)
    }

    /// Get the embedding dimension (768).
    pub fn dimension(&self) -> usize {
        crate::EMBEDDING_DIM
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        self.session.model_name()
    }
}

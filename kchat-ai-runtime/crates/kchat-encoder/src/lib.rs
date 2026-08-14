//! Unified multi-task encoder — XLM-RoBERTa-base ONNX model.
//!
//! A single shared encoder backbone that handles three tasks via different
//! output processing:
//! - **Safety classification**: mean-pool → linear → 10-class softmax
//! - **Text embedding**: mean-pool → L2 normalize → 768-dim vector
//! - **Cross-encoder reranking**: pair encoding → [CLS] logit → relevance score
//!
//! The encoder is loaded once and shared across all three task heads,
//! avoiding duplicate model loads and reducing memory footprint.

pub mod mock;

#[cfg(feature = "domain-adapters")]
pub mod domain_adapters;

#[cfg(feature = "onnx-runtime")]
pub mod session;

#[cfg(feature = "onnx-runtime")]
pub mod safety;

#[cfg(feature = "onnx-runtime")]
pub mod embed;

#[cfg(feature = "onnx-runtime")]
pub mod rerank;

// Re-export public types
pub use mock::{MockEncoderSession, MockSafetyHead, MockEmbedHead, MockRerankHead};

#[cfg(feature = "onnx-runtime")]
pub use session::{EncoderSession, ForwardOutput};

#[cfg(feature = "onnx-runtime")]
pub use safety::SafetyHead;

#[cfg(feature = "onnx-runtime")]
pub use embed::EmbedHead;

#[cfg(feature = "onnx-runtime")]
pub use rerank::RerankHead;

/// Embedding dimension for XLM-RoBERTa-base.
pub const EMBEDDING_DIM: usize = 768;

/// Maximum sequence length for the encoder.
pub const MAX_SEQ_LENGTH: usize = 512;

/// Number of safety classification categories.
pub const NUM_SAFETY_CATEGORIES: usize = 10;

/// Error type for encoder operations (always available, even without onnx-runtime).
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("encoder inference failed: {0}")]
    InferenceFailed(String),

    #[error("tokenizer error: {0}")]
    TokenizerError(String),

    #[error("session error: {0}")]
    SessionError(String),

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

pub type EncoderResult<T> = std::result::Result<T, EncoderError>;

/// Safety classification verdict (always available for mock use).
#[derive(Debug, Clone)]
pub struct SafetyVerdict {
    pub category: u32,
    pub confidence: f64,
}

/// Quantization level for the encoder model (always available).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    Int8,
    Int4,
}

impl Quantization {
    pub fn as_str(&self) -> &'static str {
        match self {
            Quantization::Int8 => "int8",
            Quantization::Int4 => "int4",
        }
    }
}

/// Safety category labels.
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

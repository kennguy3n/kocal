//! Unified multi-task encoder — mmBERT-small (GGUF) or XLM-RoBERTa-base (ONNX).
//!
//! A single shared encoder backbone that handles three tasks via different
//! output processing:
//! - **Safety classification**: CLS-pool → MLP → 17-class softmax
//! - **Text embedding**: CLS-pool → MLP → L2 normalize → 384-dim vector
//! - **Cross-encoder reranking**: pair encoding → MLP → relevance score
//!
//! The encoder is loaded once and shared across all three task heads,
//! avoiding duplicate model loads and reducing memory footprint.
//!
//! # Backends
//!
//! - **GGUF backend** (`gguf-runtime` feature): Uses llama-server's embedding
//!   endpoint with mmBERT-small-Q4_K_M.gguf (~90MB). Task heads are loaded
//!   from a separate safetensors file. Supports 17-category taxonomy.
//! - **ONNX backend** (`onnx-runtime` feature): Uses ONNX Runtime with
//!   XLM-RoBERTa-base INT8/INT4. Legacy 10-category taxonomy.

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

#[cfg(feature = "gguf-runtime")]
pub mod gguf_session;

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

#[cfg(feature = "gguf-runtime")]
pub use gguf_session::{GgufEncoderSession, ClassifierHeads};

/// Embedding dimension for mmBERT-small (384).
///
/// Note: The legacy ONNX backend (XLM-RoBERTa-base) used 768-dim embeddings.
/// The new GGUF backend (mmBERT-small) uses 384-dim. Code that depends on
/// this constant should be aware that the dimension changed in v2.0.
pub const EMBEDDING_DIM: usize = 384;

/// Maximum sequence length for the encoder.
pub const MAX_SEQ_LENGTH: usize = 512;

/// Number of safety classification categories (17, kchat.guardrail.taxonomy.v1).
///
/// Note: The legacy ONNX backend used 10 categories. The new GGUF backend
/// uses the full 17-category guardrail taxonomy.
pub const NUM_SAFETY_CATEGORIES: usize = 17;

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

/// Safety category labels — 17-category guardrail taxonomy (kchat.guardrail.taxonomy.v1).
///
/// Categories 0-16 match the deterministic detector taxonomy in kchat-safety.
/// The legacy 10-category names are kept as aliases for backward compatibility.
pub mod categories {
    pub const SAFE: u32 = 0;
    pub const CHILD_SAFETY: u32 = 1;
    pub const SELF_HARM: u32 = 2;
    pub const VIOLENCE_THREAT: u32 = 3;
    pub const EXTREMISM: u32 = 4;
    pub const HARASSMENT: u32 = 5;
    pub const HATE: u32 = 6;
    pub const SCAM_FRAUD: u32 = 7;
    pub const MALWARE_LINK: u32 = 8;
    pub const PRIVATE_DATA: u32 = 9;
    pub const SEXUAL_ADULT: u32 = 10;
    pub const DRUGS_WEAPONS: u32 = 11;
    pub const ILLEGAL_GOODS: u32 = 12;
    pub const MISINFORMATION_HEALTH: u32 = 13;
    pub const MISINFORMATION_CIVIC: u32 = 14;
    pub const COMMUNITY_RULE: u32 = 15;
    pub const DEEPFAKE_SYNTHETIC: u32 = 16;

    pub const NUM_CATEGORIES: usize = 17;

    // Backward-compatible aliases for the legacy 10-category taxonomy.
    // These map old names to the new 17-category IDs.
    pub const HATE_SPEECH: u32 = HATE;
    pub const VIOLENCE: u32 = VIOLENCE_THREAT;
    pub const SEXUAL_CONTENT: u32 = SEXUAL_ADULT;
    pub const SCAM: u32 = SCAM_FRAUD;
    pub const PII: u32 = PRIVATE_DATA;
    pub const URL_RISK: u32 = MALWARE_LINK;

    pub fn name(id: u32) -> &'static str {
        match id {
            SAFE => "safe",
            CHILD_SAFETY => "child_safety",
            SELF_HARM => "self_harm",
            VIOLENCE_THREAT => "violence_threat",
            EXTREMISM => "extremism",
            HARASSMENT => "harassment",
            HATE => "hate",
            SCAM_FRAUD => "scam_fraud",
            MALWARE_LINK => "malware_link",
            PRIVATE_DATA => "private_data",
            SEXUAL_ADULT => "sexual_adult",
            DRUGS_WEAPONS => "drugs_weapons",
            ILLEGAL_GOODS => "illegal_goods",
            MISINFORMATION_HEALTH => "misinformation_health",
            MISINFORMATION_CIVIC => "misinformation_civic",
            COMMUNITY_RULE => "community_rule",
            DEEPFAKE_SYNTHETIC => "deepfake_synthetic",
            _ => "unknown",
        }
    }
}

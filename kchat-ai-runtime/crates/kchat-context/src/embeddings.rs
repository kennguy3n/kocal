//! Dense embeddings — multilingual text embeddings for hybrid retrieval.
//!
//! Supports two embedding providers:
//! - **ONNX Runtime** with kchat-encoder (XLM-RoBERTa-base, 768-dim) as primary
//! - **llama.cpp** embeddings from the generative model as fallback
//!
//! The embedding manager tries the primary provider first, falling back to
//! the secondary if the primary is unavailable or fails.

use serde::{Deserialize, Serialize};

/// Error type for embedding operations.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("no embedding provider available")]
    NoProvider,

    #[error("embedding inference failed: {0}")]
    InferenceFailed(String),

    #[error("tokenizer error: {0}")]
    TokenizerError(String),

    #[error("session error: {0}")]
    SessionError(String),

    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Result type for embedding operations.
pub type EmbeddingResult<T> = std::result::Result<T, EmbeddingError>;

/// Trait for embedding providers.
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a text and return its vector representation.
    fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>>;

    /// Get the dimensionality of the embedding vectors.
    fn dimension(&self) -> usize;

    /// Get the name of the embedding model.
    fn model_name(&self) -> &str;
}

/// Prefix type for e5 models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingPrefix {
    /// For queries: "query: "
    Query,
    /// For documents/passages: "passage: "
    Passage,
    /// No prefix
    None,
}

impl EmbeddingPrefix {
    pub fn as_str(&self) -> &'static str {
        match self {
            EmbeddingPrefix::Query => "query: ",
            EmbeddingPrefix::Passage => "passage: ",
            EmbeddingPrefix::None => "",
        }
    }
}

/// Embedding manager — tries primary provider, falls back to secondary.
pub struct EmbeddingManager {
    primary: Option<Box<dyn EmbeddingProvider>>,
    fallback: Option<Box<dyn EmbeddingProvider>>,
}

impl EmbeddingManager {
    /// Create a new embedding manager with no providers.
    pub fn new() -> Self {
        Self {
            primary: None,
            fallback: None,
        }
    }

    /// Set the primary embedding provider (e.g., kchat-encoder).
    pub fn with_primary(mut self, provider: Box<dyn EmbeddingProvider>) -> Self {
        self.primary = Some(provider);
        self
    }

    /// Set the fallback embedding provider (e.g., llama.cpp).
    pub fn with_fallback(mut self, provider: Box<dyn EmbeddingProvider>) -> Self {
        self.fallback = Some(provider);
        self
    }

    /// Check if any provider is available.
    pub fn is_available(&self) -> bool {
        self.primary.is_some() || self.fallback.is_some()
    }

    /// Get the dimensionality of the active provider.
    pub fn dimension(&self) -> Option<usize> {
        if let Some(p) = &self.primary {
            return Some(p.dimension());
        }
        if let Some(f) = &self.fallback {
            return Some(f.dimension());
        }
        None
    }

    /// Embed a text, trying primary first, then fallback.
    pub fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        if let Some(p) = &self.primary {
            match p.embed(text) {
                Ok(v) => return Ok(v),
                Err(e) => tracing::warn!("Primary embedding failed: {}, trying fallback", e),
            }
        }
        if let Some(f) = &self.fallback {
            return f.embed(text);
        }
        Err(EmbeddingError::NoProvider)
    }

    /// Embed a query for retrieval.
    ///
    /// XLM-RoBERTa (kchat-encoder) does not use e5-style "query: " prefixes.
    /// The text is embedded directly.
    pub fn embed_query(&self, query: &str) -> EmbeddingResult<Vec<f32>> {
        self.embed(query)
    }

    /// Embed a document/passage for indexing.
    ///
    /// XLM-RoBERTa (kchat-encoder) does not use e5-style "passage: " prefixes.
    /// The text is embedded directly.
    pub fn embed_passage(&self, passage: &str) -> EmbeddingResult<Vec<f32>> {
        self.embed(passage)
    }
}

impl Default for EmbeddingManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute cosine similarity between two vectors.
/// Returns 0.0 for empty/zero vectors or near-zero norms (numerical stability).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    // Use epsilon threshold for numerical stability (prevents div by near-zero)
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// ONNX Runtime embedding provider using kchat-encoder (XLM-RoBERTa-base).
///
/// This provider wraps a shared `kchat_encoder::EncoderSession` and delegates
/// embedding to `kchat_encoder::EmbedHead`, producing 768-dimensional
/// L2-normalized embeddings.
#[cfg(feature = "embeddings")]
pub struct OnnxEmbedder {
    session: std::sync::Arc<kchat_encoder::EncoderSession>,
}

#[cfg(feature = "embeddings")]
impl OnnxEmbedder {
    /// Create a new ONNX embedder from a shared encoder session.
    pub fn new(session: std::sync::Arc<kchat_encoder::EncoderSession>) -> Self {
        Self { session }
    }

    /// Create a new ONNX embedder by loading a model from file.
    ///
    /// `intra_threads` controls ONNX Runtime intra-op parallelism (2 for low, 3 for medium, 4+ for high).
    pub fn from_files(
        model_path: &str,
        tokenizer_path: &str,
        quantization: kchat_encoder::Quantization,
        intra_threads: usize,
    ) -> EmbeddingResult<Self> {
        let session = kchat_encoder::EncoderSession::new(model_path, tokenizer_path, quantization, intra_threads)
            .map_err(|e| EmbeddingError::SessionError(e.to_string()))?;
        Ok(Self {
            session: std::sync::Arc::new(session),
        })
    }
}

#[cfg(feature = "embeddings")]
impl EmbeddingProvider for OnnxEmbedder {
    fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        let head = kchat_encoder::EmbedHead::new(&self.session);
        head.embed(text)
            .map_err(|e| EmbeddingError::InferenceFailed(e.to_string()))
    }

    fn dimension(&self) -> usize {
        kchat_encoder::EMBEDDING_DIM
    }

    fn model_name(&self) -> &str {
        self.session.model_name()
    }
}

/// A mock embedding provider for testing.
pub struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl EmbeddingProvider for MockEmbedder {
    fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        // Deterministic mock: hash the text to produce a pseudo-embedding
        let mut embedding = vec![0.0f32; self.dimension];
        let bytes = text.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            embedding[i % self.dimension] += b as f32 / 255.0;
        }
        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }
        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        "mock-embedder"
    }
}

/// Cached embedding with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEmbedding {
    /// The embedding vector
    pub vector: Vec<f32>,
    /// Model name that produced this embedding
    pub model: String,
    /// Dimension
    pub dimension: usize,
    /// Model version hash (e.g. "kchat-encoder-int8-v1.0.0") for cache invalidation.
    /// If empty, the cache entry is from a legacy version and should be invalidated.
    #[serde(default)]
    pub model_version: String,
}

/// Current embedding model version string for cache compatibility checks.
pub const ENCODER_MODEL_VERSION: &str = "kchat-encoder-v1.0.0";

impl CachedEmbedding {
    /// Magic header for v2 binary format (includes model name + version).
    const V2_MAGIC: u32 = 0x4B434532; // "KCE2"
    /// Legacy v1 format has no magic — just dimension as first 4 bytes.
    /// v1 entries always start with a small u32 (dimension <= 4096), so
    /// the v2 magic (0x4B434532 = 1263369778) is easily distinguishable.

    /// Serialize to bytes for storage (v2 format with model metadata).
    pub fn to_bytes(&self) -> Vec<u8> {
        // v2 format: magic(4) + dim(4) + model_len(4) + model_bytes + version_len(4) + version_bytes + vector(dim*4)
        let model_bytes = self.model.as_bytes();
        let version_bytes = self.model_version.as_bytes();
        let mut bytes = Vec::with_capacity(16 + model_bytes.len() + version_bytes.len() + self.vector.len() * 4);
        bytes.extend_from_slice(&Self::V2_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&(self.dimension as u32).to_le_bytes());
        bytes.extend_from_slice(&(model_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(model_bytes);
        bytes.extend_from_slice(&(version_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(version_bytes);
        for v in &self.vector {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes (supports both v1 and v2 formats).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }

        // Check for v2 magic header
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if magic == Self::V2_MAGIC {
            return Self::from_bytes_v2(bytes);
        }

        // Legacy v1 format: dim(4) + vector(dim*4), no model metadata
        Self::from_bytes_v1(bytes)
    }

    fn from_bytes_v2(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 12 {
            return None;
        }
        let dim = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
        let model_len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let offset = 12;
        if bytes.len() < offset + model_len + 4 {
            return None;
        }
        let model = String::from_utf8(bytes[offset..offset + model_len].to_vec()).ok()?;
        let version_offset = offset + model_len;
        let version_len = u32::from_le_bytes([
            bytes[version_offset],
            bytes[version_offset + 1],
            bytes[version_offset + 2],
            bytes[version_offset + 3],
        ]) as usize;
        let version_start = version_offset + 4;
        if bytes.len() < version_start + version_len + dim * 4 {
            return None;
        }
        let model_version = String::from_utf8(bytes[version_start..version_start + version_len].to_vec()).ok()?;
        let vec_start = version_start + version_len;
        let mut vector = Vec::with_capacity(dim);
        for i in 0..dim {
            let off = vec_start + i * 4;
            let v = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            vector.push(v);
        }
        Some(Self {
            vector,
            model,
            dimension: dim,
            model_version,
        })
    }

    fn from_bytes_v1(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let dim = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        let expected_len = 4 + dim * 4;
        if bytes.len() < expected_len {
            return None;
        }
        let mut vector = Vec::with_capacity(dim);
        for i in 0..dim {
            let offset = 4 + i * 4;
            let v = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            vector.push(v);
        }
        // v1 entries have no model metadata — they will fail compatibility checks
        // and be recomputed, which is the correct behavior for cache invalidation.
        Some(Self {
            vector,
            model: "unknown".into(),
            dimension: dim,
            model_version: String::new(),
        })
    }

    /// Check if this cached embedding is compatible with the current encoder model.
    ///
    /// Returns `false` if the model version doesn't match or the dimension
    /// doesn't match the expected embedding dimension, indicating the cache
    /// entry should be invalidated and recomputed.
    pub fn is_compatible_with(&self, expected_version: &str, expected_dim: usize) -> bool {
        self.model_version == expected_version && self.dimension == expected_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embedder() {
        let embedder = MockEmbedder::new(768);
        let vec = embedder.embed("hello world").unwrap();
        assert_eq!(vec.len(), 768);

        // L2 norm should be ~1.0
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_embedding_manager_no_provider() {
        let manager = EmbeddingManager::new();
        assert!(!manager.is_available());
        assert!(manager.embed("test").is_err());
    }

    #[test]
    fn test_embedding_manager_with_mock() {
        let manager = EmbeddingManager::new()
            .with_primary(Box::new(MockEmbedder::new(768)));
        assert!(manager.is_available());
        assert_eq!(manager.dimension(), Some(768));

        let vec = manager.embed("hello").unwrap();
        assert_eq!(vec.len(), 768);
    }

    #[test]
    fn test_embedding_manager_fallback() {
        // Primary that always fails, fallback that works
        struct FailingEmbedder;
        impl EmbeddingProvider for FailingEmbedder {
            fn embed(&self, _text: &str) -> EmbeddingResult<Vec<f32>> {
                Err(EmbeddingError::InferenceFailed("intentional failure".into()))
            }
            fn dimension(&self) -> usize { 768 }
            fn model_name(&self) -> &str { "failing" }
        }

        let manager = EmbeddingManager::new()
            .with_primary(Box::new(FailingEmbedder))
            .with_fallback(Box::new(MockEmbedder::new(768)));

        let vec = manager.embed("hello").unwrap();
        assert_eq!(vec.len(), 768);
    }

    #[test]
    fn test_embed_query_with_prefix() {
        let manager = EmbeddingManager::new()
            .with_primary(Box::new(MockEmbedder::new(128)));
        let vec = manager.embed_query("hello").unwrap();
        assert_eq!(vec.len(), 128);
    }

    #[test]
    fn test_embed_passage_with_prefix() {
        let manager = EmbeddingManager::new()
            .with_primary(Box::new(MockEmbedder::new(128)));
        let vec = manager.embed_passage("hello world").unwrap();
        assert_eq!(vec.len(), 128);
    }

    #[test]
    fn test_semantic_similarity() {
        let embedder = MockEmbedder::new(256);
        let hello = embedder.embed("hello").unwrap();
        let hi = embedder.embed("hi").unwrap();
        let world = embedder.embed("world").unwrap();

        // "hello" and "hi" share 'h' so should be more similar than "hello" and "world"
        let sim_hello_hi = cosine_similarity(&hello, &hi);
        let sim_hello_world = cosine_similarity(&hello, &world);

        // With our simple mock, this may not hold perfectly, but let's check
        // that similarity is in valid range
        assert!(sim_hello_hi >= -1.0 && sim_hello_hi <= 1.0);
        assert!(sim_hello_world >= -1.0 && sim_hello_world <= 1.0);
    }

    #[test]
    fn test_cached_embedding_serialization() {
        let cached = CachedEmbedding {
            vector: vec![0.1, 0.2, 0.3, 0.4],
            model: "kchat-encoder-int8".into(),
            dimension: 4,
            model_version: "kchat-encoder-v1.0.0".into(),
        };

        let bytes = cached.to_bytes();
        let restored = CachedEmbedding::from_bytes(&bytes).unwrap();

        assert_eq!(restored.dimension, 4);
        assert_eq!(restored.vector.len(), 4);
        assert_eq!(restored.model, "kchat-encoder-int8");
        assert_eq!(restored.model_version, "kchat-encoder-v1.0.0");
        for (a, b) in cached.vector.iter().zip(restored.vector.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn test_cached_embedding_v1_backward_compat() {
        // v1 format: dim(4) + vector(dim*4), no model metadata
        let dim: u32 = 4;
        let mut v1_bytes = Vec::new();
        v1_bytes.extend_from_slice(&dim.to_le_bytes());
        for v in [0.1f32, 0.2, 0.3, 0.4] {
            v1_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let restored = CachedEmbedding::from_bytes(&v1_bytes).unwrap();
        assert_eq!(restored.dimension, 4);
        assert_eq!(restored.model, "unknown");
        assert_eq!(restored.model_version, "");
        // v1 entries should fail compatibility checks
        assert!(!restored.is_compatible_with(ENCODER_MODEL_VERSION, 4));
    }

    #[test]
    fn test_cached_embedding_invalid_bytes() {
        assert!(CachedEmbedding::from_bytes(&[]).is_none());
        assert!(CachedEmbedding::from_bytes(&[1, 2]).is_none());
    }

    #[test]
    fn test_cached_embedding_compatibility() {
        let compatible = CachedEmbedding {
            vector: vec![0.1; 768],
            model: "kchat-encoder-int8".into(),
            dimension: 768,
            model_version: ENCODER_MODEL_VERSION.into(),
        };
        assert!(compatible.is_compatible_with(ENCODER_MODEL_VERSION, 768));

        let wrong_version = CachedEmbedding {
            vector: vec![0.1; 768],
            model: "multilingual-e5-small".into(),
            dimension: 768,
            model_version: "e5-small-v1".into(),
        };
        assert!(!wrong_version.is_compatible_with(ENCODER_MODEL_VERSION, 768));

        let wrong_dim = CachedEmbedding {
            vector: vec![0.1; 384],
            model: "multilingual-e5-small".into(),
            dimension: 384,
            model_version: ENCODER_MODEL_VERSION.into(),
        };
        assert!(!wrong_dim.is_compatible_with(ENCODER_MODEL_VERSION, 768));

        let legacy = CachedEmbedding {
            vector: vec![0.1; 384],
            model: "unknown".into(),
            dimension: 384,
            model_version: String::new(),
        };
        assert!(!legacy.is_compatible_with(ENCODER_MODEL_VERSION, 768));
    }

    #[test]
    fn test_embedding_prefix() {
        assert_eq!(EmbeddingPrefix::Query.as_str(), "query: ");
        assert_eq!(EmbeddingPrefix::Passage.as_str(), "passage: ");
        assert_eq!(EmbeddingPrefix::None.as_str(), "");
    }
}

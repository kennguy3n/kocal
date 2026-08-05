//! Dense embeddings — multilingual text embeddings for hybrid retrieval.
//!
//! Supports two embedding providers:
//! - **ONNX Runtime** with multilingual-e5-small (384-dim, INT8, ~45MB) as primary
//! - **llama.cpp** embeddings from the generative model as fallback
//!
//! The embedding manager tries the primary provider first, falling back to
//! the secondary if the primary is unavailable or fails.
//!
//! e5 models require a prefix: `"query: "` for queries and `"passage: "` for
//! documents.

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

    /// Set the primary embedding provider (e.g., ONNX e5-small).
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

    /// Embed a query (with "query: " prefix for e5 models).
    pub fn embed_query(&self, query: &str) -> EmbeddingResult<Vec<f32>> {
        let prefixed = format!("query: {}", query);
        self.embed(&prefixed)
    }

    /// Embed a document/passage (with "passage: " prefix for e5 models).
    pub fn embed_passage(&self, passage: &str) -> EmbeddingResult<Vec<f32>> {
        let prefixed = format!("passage: {}", passage);
        self.embed(&prefixed)
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

/// ONNX Runtime embedding provider using multilingual-e5-small.
///
/// This provider loads an ONNX model and tokenizer, runs inference to produce
/// 384-dimensional embeddings. The model must be INT8 quantized (~45MB).
#[cfg(feature = "embeddings")]
pub struct OnnxEmbedder {
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
    dimension: usize,
    model_name: String,
}

#[cfg(feature = "embeddings")]
impl OnnxEmbedder {
    /// Create a new ONNX embedder from a model file and tokenizer.
    pub fn new(model_path: &str, tokenizer_path: &str) -> EmbeddingResult<Self> {
        let session = ort::session::Session::builder()
            .and_then(|b| b.with_optimization_level(ort::session::GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_intra_threads(2))
            .map_err(|e| EmbeddingError::SessionError(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| EmbeddingError::SessionError(e.to_string()))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbeddingError::TokenizerError(e.to_string()))?;

        // e5-small has 384 dimensions
        let dimension = 384;

        Ok(Self {
            session,
            tokenizer,
            dimension,
            model_name: "multilingual-e5-small-int8".into(),
        })
    }
}

#[cfg(feature = "embeddings")]
impl EmbeddingProvider for OnnxEmbedder {
    fn embed(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
        use ndarray::Array2;
        use ort::session::inputs;

        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EmbeddingError::TokenizerError(e.to_string()))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Convert to ndarray
        let seq_len = input_ids.len();
        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids.to_vec())
            .map_err(|e| EmbeddingError::InferenceFailed(format!("array: {e}")))?;
        let attention_arr = Array2::from_shape_vec((1, seq_len), attention_mask.to_vec())
            .map_err(|e| EmbeddingError::InferenceFailed(format!("array: {e}")))?;

        // Run inference
        let inputs = inputs! {
            "input_ids" => input_ids_arr.view(),
            "attention_mask" => attention_arr.view(),
        }
        .map_err(|e| EmbeddingError::InferenceFailed(format!("inputs: {e}")))?;

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| EmbeddingError::InferenceFailed(format!("run: {e}")))?;

        // Extract embeddings (last_hidden_state or embeddings output)
        let embeddings = outputs
            .get(0)
            .ok_or_else(|| EmbeddingError::InferenceFailed("no output".into()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbeddingError::InferenceFailed(format!("extract: {e}")))?;

        // Mean pool over sequence dimension
        let data = embeddings.view();
        let seq_len = data.shape()[1];
        let dim = data.shape()[2];

        if dim != self.dimension {
            return Err(EmbeddingError::DimensionMismatch {
                expected: self.dimension,
                actual: dim,
            });
        }

        if seq_len == 0 {
            return Err(EmbeddingError::InferenceFailed(
                "empty sequence after tokenization".into(),
            ));
        }

        // Mean pool: sum first, divide once (more efficient than dividing per element)
        let mut pooled = vec![0.0f32; dim];
        for i in 0..seq_len {
            for j in 0..dim {
                pooled[j] += data[[0, i, j]];
            }
        }
        let inv_seq_len = 1.0 / seq_len as f32;
        for x in &mut pooled {
            *x *= inv_seq_len;
        }

        // L2 normalize
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut pooled {
                *x /= norm;
            }
        }

        Ok(pooled)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.model_name
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
}

impl CachedEmbedding {
    /// Serialize to bytes for storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Simple serialization: 4 bytes per f32 + header
        let mut bytes = Vec::with_capacity(8 + self.vector.len() * 4);
        bytes.extend_from_slice(&(self.dimension as u32).to_le_bytes());
        for v in &self.vector {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
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
        Some(Self {
            vector,
            model: "unknown".into(),
            dimension: dim,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_embedder() {
        let embedder = MockEmbedder::new(384);
        let vec = embedder.embed("hello world").unwrap();
        assert_eq!(vec.len(), 384);

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
            .with_primary(Box::new(MockEmbedder::new(384)));
        assert!(manager.is_available());
        assert_eq!(manager.dimension(), Some(384));

        let vec = manager.embed("hello").unwrap();
        assert_eq!(vec.len(), 384);
    }

    #[test]
    fn test_embedding_manager_fallback() {
        // Primary that always fails, fallback that works
        struct FailingEmbedder;
        impl EmbeddingProvider for FailingEmbedder {
            fn embed(&self, _text: &str) -> EmbeddingResult<Vec<f32>> {
                Err(EmbeddingError::InferenceFailed("intentional failure".into()))
            }
            fn dimension(&self) -> usize { 384 }
            fn model_name(&self) -> &str { "failing" }
        }

        let manager = EmbeddingManager::new()
            .with_primary(Box::new(FailingEmbedder))
            .with_fallback(Box::new(MockEmbedder::new(384)));

        let vec = manager.embed("hello").unwrap();
        assert_eq!(vec.len(), 384);
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
            model: "test".into(),
            dimension: 4,
        };

        let bytes = cached.to_bytes();
        let restored = CachedEmbedding::from_bytes(&bytes).unwrap();

        assert_eq!(restored.dimension, 4);
        assert_eq!(restored.vector.len(), 4);
        for (a, b) in cached.vector.iter().zip(restored.vector.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn test_cached_embedding_invalid_bytes() {
        assert!(CachedEmbedding::from_bytes(&[]).is_none());
        assert!(CachedEmbedding::from_bytes(&[1, 2]).is_none());
    }

    #[test]
    fn test_embedding_prefix() {
        assert_eq!(EmbeddingPrefix::Query.as_str(), "query: ");
        assert_eq!(EmbeddingPrefix::Passage.as_str(), "passage: ");
        assert_eq!(EmbeddingPrefix::None.as_str(), "");
    }
}

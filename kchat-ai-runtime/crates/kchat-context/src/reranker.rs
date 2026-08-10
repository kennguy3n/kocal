//! Reranker — cross-encoder reranking for retrieval precision.
//!
//! A cross-encoder reranker takes a query-document pair as input and produces
//! a relevance score. Unlike bi-encoders (embeddings), the cross-encoder
//! processes both texts together, allowing it to capture fine-grained
//! interactions. This gives better precision but is more expensive (O(n)
//! model calls for n documents).
//!
//! Uses the unified kchat-encoder (XLM-RoBERTa-base) shared session.
//! On Medium+ tier devices, the reranker re-ranks the top-N (e.g. 20) retrieval
//! results to produce a more accurate final ranking.

use serde::{Deserialize, Serialize};

/// Error type for reranker operations.
#[derive(Debug, thiserror::Error)]
pub enum RerankerError {
    #[error("reranker inference failed: {0}")]
    InferenceFailed(String),

    #[error("reranker not loaded")]
    NotLoaded,

    #[error("tokenizer error: {0}")]
    TokenizerError(String),
}

/// Result type for reranker operations.
pub type RerankerResult<T> = std::result::Result<T, RerankerError>;

/// Trait for reranker implementations.
pub trait Reranker: Send + Sync {
    /// Rerank documents by relevance to the query.
    ///
    /// Returns a list of (document_index, score) pairs sorted by score
    /// descending, truncated to `top_k`.
    fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> RerankerResult<Vec<(usize, f64)>>;

    /// Get the model name.
    fn model_name(&self) -> &str;
}

/// ONNX Runtime cross-encoder reranker using kchat-encoder (XLM-RoBERTa-base).
///
/// Wraps a shared `kchat_encoder::EncoderSession` and delegates reranking
/// to `kchat_encoder::RerankHead`.
#[cfg(feature = "reranker")]
pub struct CrossEncoderReranker {
    session: std::sync::Arc<kchat_encoder::EncoderSession>,
}

#[cfg(feature = "reranker")]
impl CrossEncoderReranker {
    /// Create a new cross-encoder reranker from a shared encoder session.
    pub fn new(session: std::sync::Arc<kchat_encoder::EncoderSession>) -> Self {
        Self { session }
    }

    /// Create a new cross-encoder reranker by loading a model from file.
    ///
    /// `intra_threads` controls ONNX Runtime intra-op parallelism (2 for low, 3 for medium, 4+ for high).
    pub fn from_files(
        model_path: &str,
        tokenizer_path: &str,
        quantization: kchat_encoder::Quantization,
        intra_threads: usize,
    ) -> RerankerResult<Self> {
        let session = kchat_encoder::EncoderSession::new(model_path, tokenizer_path, quantization, intra_threads)
            .map_err(|e| RerankerError::InferenceFailed(format!("encoder session: {e}")))?;
        Ok(Self {
            session: std::sync::Arc::new(session),
        })
    }
}

#[cfg(feature = "reranker")]
impl Reranker for CrossEncoderReranker {
    fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> RerankerResult<Vec<(usize, f64)>> {
        let head = kchat_encoder::RerankHead::new(&self.session);
        head.rerank(query, documents, top_k)
            .map_err(|e| RerankerError::InferenceFailed(e.to_string()))
    }

    fn model_name(&self) -> &str {
        self.session.model_name()
    }
}

/// Mock reranker for testing — uses simple text overlap scoring.
pub struct MockReranker {
    model_name: String,
}

impl MockReranker {
    pub fn new() -> Self {
        Self {
            model_name: "mock-reranker".into(),
        }
    }

    /// Simple word overlap score.
    fn overlap_score(query: &str, document: &str) -> f64 {
        let query_words: std::collections::HashSet<&str> = query.split_whitespace().collect();
        let doc_words: std::collections::HashSet<&str> = document.split_whitespace().collect();
        if query_words.is_empty() || doc_words.is_empty() {
            return 0.0;
        }
        let overlap = query_words.intersection(&doc_words).count();
        overlap as f64 / query_words.len().max(doc_words.len()) as f64
    }
}

impl Default for MockReranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for MockReranker {
    fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> RerankerResult<Vec<(usize, f64)>> {
        let mut scored: Vec<(usize, f64)> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| (i, Self::overlap_score(query, doc)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Reranking result with score and document index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    pub document_index: usize,
    pub score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_reranker_basic() {
        let reranker = MockReranker::new();
        let docs = vec![
            "The quick brown fox jumps".to_string(),
            "Hello world greeting".to_string(),
            "Fox animal wildlife nature".to_string(),
        ];

        let results = reranker.rerank("fox animal", &docs, 2).unwrap();
        assert_eq!(results.len(), 2);
        // "Fox animal wildlife nature" should score higher than "Hello world"
        assert_eq!(results[0].0, 2); // index 2 = "Fox animal wildlife nature"
    }

    #[test]
    fn test_mock_reranker_top_k() {
        let reranker = MockReranker::new();
        let docs = vec![
            "apple banana".to_string(),
            "cherry date".to_string(),
            "elderberry fig".to_string(),
        ];

        let results = reranker.rerank("apple", &docs, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0); // "apple banana" matches "apple"
    }

    #[test]
    fn test_mock_reranker_no_overlap() {
        let reranker = MockReranker::new();
        let docs = vec!["completely different text".to_string()];
        let results = reranker.rerank("hello world", &docs, 1).unwrap();
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn test_mock_reranker_empty_query() {
        let reranker = MockReranker::new();
        let docs = vec!["some text".to_string()];
        let results = reranker.rerank("", &docs, 1).unwrap();
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn test_mock_reranker_empty_docs() {
        let reranker = MockReranker::new();
        let docs: Vec<String> = vec![];
        let results = reranker.rerank("query", &docs, 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_reranker_model_name() {
        let reranker = MockReranker::new();
        assert_eq!(reranker.model_name(), "mock-reranker");
    }

    #[test]
    fn test_overlap_score_identical() {
        let score = MockReranker::overlap_score("hello world", "hello world");
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_overlap_score_partial() {
        let score = MockReranker::overlap_score("hello world foo", "hello bar");
        // 1 word overlap out of 3+2=5 unique words
        assert!(score > 0.0 && score < 1.0);
    }
}

//! Mock implementations for testing — no ONNX Runtime required.

use crate::categories;

/// Mock encoder session for testing.
///
/// Produces deterministic outputs based on input text hashing,
/// simulating the behavior of a real encoder session.
pub struct MockEncoderSession {
    model_name: String,
    dimension: usize,
}

impl MockEncoderSession {
    pub fn new(model_name: &str, dimension: usize) -> Self {
        Self {
            model_name: model_name.into(),
            dimension,
        }
    }

    pub fn int8() -> Self {
        Self::new("kchat-encoder-int8", crate::EMBEDDING_DIM)
    }

    pub fn int4() -> Self {
        Self::new("kchat-encoder-int4", crate::EMBEDDING_DIM)
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

/// Mock safety head — returns a fixed category and confidence.
pub struct MockSafetyHead {
    pub category: u32,
    pub confidence: f64,
}

impl MockSafetyHead {
    pub fn new(category: u32, confidence: f64) -> Self {
        Self { category, confidence }
    }

    pub fn safe() -> Self {
        Self::new(categories::SAFE, 0.95)
    }

    pub fn harmful(category: u32) -> Self {
        Self::new(category, 0.90)
    }

    pub fn classify(&self, _text: &str) -> Result<crate::SafetyVerdict, crate::EncoderError> {
        Ok(crate::SafetyVerdict {
            category: self.category,
            confidence: self.confidence,
        })
    }
}

/// Mock embedding head — deterministic hash-based embeddings.
pub struct MockEmbedHead {
    dimension: usize,
}

impl MockEmbedHead {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, crate::EncoderError> {
        let mut embedding = vec![0.0f32; self.dimension];
        let bytes = text.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            embedding[i % self.dimension] += b as f32 / 255.0;
        }
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }
        Ok(embedding)
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn model_name(&self) -> &str {
        "mock-encoder"
    }
}

/// Mock reranking head — simple text overlap scoring.
pub struct MockRerankHead {
    model_name: String,
}

impl MockRerankHead {
    pub fn new() -> Self {
        Self {
            model_name: "mock-encoder".into(),
        }
    }

    pub fn score_pair(&self, query: &str, document: &str) -> Result<f64, crate::EncoderError> {
        let query_lower = query.to_lowercase();
        let doc_lower = document.to_lowercase();
        let overlap = query_lower
            .split_whitespace()
            .filter(|w| !w.is_empty() && doc_lower.contains(w))
            .count();
        Ok(overlap as f64 / query_lower.split_whitespace().count().max(1) as f64)
    }

    pub fn rerank(
        &self,
        query: &str,
        documents: &[String],
        top_k: usize,
    ) -> Result<Vec<(usize, f64)>, crate::EncoderError> {
        let mut scored: Vec<(usize, f64)> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let score = self.score_pair(query, doc).unwrap_or(0.0);
                (i, score)
            })
            .collect();

        scored.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl Default for MockRerankHead {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_safety_head_safe() {
        let head = MockSafetyHead::safe();
        let verdict = head.classify("hello world").unwrap();
        assert_eq!(verdict.category, categories::SAFE);
        assert!(verdict.confidence > 0.9);
    }

    #[test]
    fn test_mock_safety_head_harmful() {
        let head = MockSafetyHead::harmful(categories::VIOLENCE);
        let verdict = head.classify("harmful text").unwrap();
        assert_eq!(verdict.category, categories::VIOLENCE);
    }

    #[test]
    fn test_mock_embed_head() {
        let head = MockEmbedHead::new(crate::EMBEDDING_DIM);
        let vec = head.embed("hello world").unwrap();
        assert_eq!(vec.len(), crate::EMBEDDING_DIM);
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mock_rerank_head_basic() {
        let head = MockRerankHead::new();
        let docs = vec![
            "The quick brown fox jumps".to_string(),
            "Hello world greeting".to_string(),
            "Fox animal wildlife nature".to_string(),
        ];
        let results = head.rerank("fox animal", &docs, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_mock_encoder_session() {
        let session = MockEncoderSession::int8();
        assert_eq!(session.model_name(), "kchat-encoder-int8");
        assert_eq!(session.dimension(), crate::EMBEDDING_DIM);
    }
}

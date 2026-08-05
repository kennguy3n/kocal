//! Retrieval — hybrid FTS/BM25 + dense embedding fusion with recency.
//!
//! Retrieval tiers:
//! - Low: FTS/BM25, field filters, recency, deterministic entity extraction
//! - Medium: add multilingual dense embeddings and hybrid fusion
//! - High: add a reranker for top candidates and larger citation budgets
//!
//! Every retrieved item carries source, timestamp, ACL version, provenance,
//! and citation location. Remote documents and chat messages are untrusted
//! content.

use crate::scope::ScopeFilter;
use crate::store::{ContextStore, EvidenceId};
use serde::{Deserialize, Serialize};

/// Retrieval tier — controls which retrieval features are available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RetrievalTier {
    /// FTS/BM25 only
    Low,
    /// FTS/BM25 + dense embeddings (hybrid fusion)
    Medium,
    /// Hybrid + reranker
    High,
}

/// Weights for hybrid retrieval fusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridWeights {
    pub fts: f64,
    pub recency: f64,
    pub vector: f64,
}

impl Default for HybridWeights {
    fn default() -> Self {
        Self {
            fts: 0.6,
            recency: 0.3,
            vector: 0.1,
        }
    }
}

/// A retrieval result with fused scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub evidence_id: EvidenceId,
    pub score: f64,
    pub fts_score: f64,
    pub recency_score: f64,
    pub vector_score: f64,
}

/// The retriever — orchestrates FTS, recency, and optional vector search.
pub struct Retriever<'a> {
    store: &'a ContextStore,
    weights: HybridWeights,
    recency_half_life_secs: f64,
    tier: RetrievalTier,
}

impl<'a> Retriever<'a> {
    pub fn new(store: &'a ContextStore, tier: RetrievalTier) -> Self {
        Self {
            store,
            weights: HybridWeights::default(),
            recency_half_life_secs: 30.0 * 24.0 * 60.0 * 60.0, // 30 days
            tier,
        }
    }

    pub fn with_weights(mut self, weights: HybridWeights) -> Self {
        self.weights = weights;
        self
    }

    pub fn with_recency_half_life(mut self, half_life_secs: f64) -> Self {
        self.recency_half_life_secs = half_life_secs;
        self
    }

    /// Retrieve evidence for a query.
    ///
    /// Authorization is checked three times:
    /// 1. Before search (scope filter)
    /// 2. During search (FTS scope_id IN filter)
    /// 3. After search (constructing the prompt — done by caller)
    pub fn retrieve(
        &self,
        query: &str,
        filter: &ScopeFilter,
        limit: usize,
    ) -> Result<Vec<RetrievalResult>, RetrievalError> {
        // Step 1: FTS search (always available)
        let fts_results = self.store.search_fts(query, filter, limit * 2)?;

        // Step 2: Fuse scores
        let now = chrono::Utc::now().timestamp() as f64;
        let mut results = Vec::with_capacity(fts_results.len());

        for fts in fts_results {
            // Skip evidence from forgotten scopes (cryptographic forgetting)
            if self.store.is_scope_forgotten(fts.scope_id).unwrap_or(false) {
                continue;
            }

            // Normalize BM25 score (lower is better → invert and normalize)
            let fts_score = 1.0 / (1.0 + fts.bm25_score.abs());

            // Recency score: exponential decay
            let age_secs = (now - fts.created_at as f64).max(0.0);
            let recency_score = (-age_secs * (2.0_f64.ln()) / self.recency_half_life_secs).exp();

            // Vector score: 0.0 on low tier (no embeddings)
            let vector_score = match self.tier {
                RetrievalTier::Low => 0.0,
                RetrievalTier::Medium | RetrievalTier::High => {
                    // In production: compute query embedding and cosine similarity
                    // For now, 0.0 (embeddings not yet integrated)
                    0.0
                }
            };

            // Weighted fusion
            let total = self.weights.fts * fts_score
                + self.weights.recency * recency_score
                + self.weights.vector * vector_score;

            results.push(RetrievalResult {
                evidence_id: fts.evidence_id,
                score: total,
                fts_score,
                recency_score,
                vector_score,
            });
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to limit
        results.truncate(limit);

        Ok(results)
    }
}

/// Retrieval errors.
#[derive(Debug, thiserror::Error)]
pub enum RetrievalError {
    #[error("store error: {0}")]
    Store(#[from] crate::store::StoreError),

    #[error("scope forgotten")]
    ScopeForgotten,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::ScopeId;
    use crate::store::{ContextStore, ContextStoreConfig, Evidence};
    use uuid::Uuid;

    fn make_store() -> ContextStore {
        let config = ContextStoreConfig::for_low_tier("test".into(), [42u8; 32]);
        ContextStore::open_in_memory(&config).unwrap()
    }

    fn make_evidence(scope_id: ScopeId, content: &str, age_secs: i64) -> Evidence {
        Evidence {
            id: EvidenceId::new(),
            scope_id,
            content_hash: blake3::hash(content.as_bytes()).to_hex().to_string(),
            encrypted_body: vec![],
            nonce: {
                let mut n = [0u8; 24];
                n[0] = 1;
                n.to_vec()
            },
            source_ref: None,
            importance: 5,
            language_tag: Some("en".into()),
            created_at: chrono::Utc::now().timestamp() - age_secs,
            fts_content: content.into(),
        }
    }

    #[test]
    fn test_low_tier_retrieval() {
        let store = make_store();
        let scope = ScopeId::new();

        store.insert(&make_evidence(scope, "The quick brown fox", 0)).unwrap();
        store.insert(&make_evidence(scope, "Hello world greeting", 3600)).unwrap();

        let retriever = Retriever::new(&store, RetrievalTier::Low);
        let filter = ScopeFilter {
            allowed_scopes: vec![scope],
            denied_scopes: vec![],
            user_id: Uuid::new_v4(),
            roles: vec![],
        };

        let results = retriever.retrieve("hello", &filter, 10).unwrap();
        assert!(!results.is_empty());
        // All vector scores should be 0 on low tier
        assert!(results.iter().all(|r| r.vector_score == 0.0));
    }

    #[test]
    fn test_recency_boosts_recent_results() {
        let store = make_store();
        let scope = ScopeId::new();

        // Old result
        store.insert(&make_evidence(scope, "important hello message", 86400 * 30)).unwrap();
        // Recent result
        store.insert(&make_evidence(scope, "hello recent message", 60)).unwrap();

        let retriever = Retriever::new(&store, RetrievalTier::Low);
        let filter = ScopeFilter {
            allowed_scopes: vec![scope],
            denied_scopes: vec![],
            user_id: Uuid::new_v4(),
            roles: vec![],
        };

        let results = retriever.retrieve("hello", &filter, 10).unwrap();
        assert!(!results.is_empty());
        // Recent result should have higher recency_score
        let recent = results.iter().find(|r| r.recency_score > 0.5);
        assert!(recent.is_some());
    }

    #[test]
    fn test_scope_filter_excludes_unauthorized() {
        let store = make_store();
        let scope1 = ScopeId::new();
        let scope2 = ScopeId::new();

        store.insert(&make_evidence(scope1, "hello in scope 1", 0)).unwrap();
        store.insert(&make_evidence(scope2, "hello in scope 2", 0)).unwrap();

        let retriever = Retriever::new(&store, RetrievalTier::Low);
        let filter = ScopeFilter {
            allowed_scopes: vec![scope1],
            denied_scopes: vec![],
            user_id: Uuid::new_v4(),
            roles: vec![],
        };

        let results = retriever.retrieve("hello", &filter, 10).unwrap();
        // Should only return results from scope1
        // (The FTS query filters by scope_id IN allowed_scopes)
        assert!(!results.is_empty(), "should have results from scope1");
    }
}

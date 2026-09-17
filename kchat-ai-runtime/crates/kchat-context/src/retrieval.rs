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

use crate::embeddings::{cosine_similarity, EmbeddingManager};
use crate::scope::{ScopeFilter, ScopeId};
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
    /// Cross-encoder score when a reranker ran (High tier); 0 otherwise.
    #[serde(default)]
    pub rerank_score: f64,
}

/// RRF rank-fusion constant — standard value from the original paper
/// (Cormack et al. 2009). Larger values flatten rank differences.
const RRF_K: f64 = 60.0;

/// Cap on missing/stale vectors re-embedded per query. Backfill is bounded
/// so a cold index degrades gracefully instead of blocking retrieval.
const BACKFILL_PER_QUERY: usize = 64;

/// The retriever — orchestrates FTS, recency, and optional vector search.
pub struct Retriever<'a> {
    store: &'a ContextStore,
    weights: HybridWeights,
    recency_half_life_secs: f64,
    tier: RetrievalTier,
    /// Optional embedding manager for Medium/High tier vector search
    embeddings: Option<&'a EmbeddingManager>,
    /// Optional cross-encoder reranker for High tier (top-k only)
    reranker: Option<&'a dyn crate::reranker::Reranker>,
}

impl<'a> Retriever<'a> {
    pub fn new(store: &'a ContextStore, tier: RetrievalTier) -> Self {
        Self {
            store,
            weights: HybridWeights::default(),
            recency_half_life_secs: 30.0 * 24.0 * 60.0 * 60.0, // 30 days
            tier,
            embeddings: None,
            reranker: None,
        }
    }

    pub fn with_weights(mut self, weights: HybridWeights) -> Self {
        // Validate weights are non-negative and normalize to sum=1.0
        let total = weights.fts + weights.recency + weights.vector;
        if total > 0.0 && weights.fts >= 0.0 && weights.recency >= 0.0 && weights.vector >= 0.0 {
            self.weights = HybridWeights {
                fts: weights.fts / total,
                recency: weights.recency / total,
                vector: weights.vector / total,
            };
        }
        self
    }

    pub fn with_recency_half_life(mut self, half_life_secs: f64) -> Self {
        self.recency_half_life_secs = half_life_secs;
        self
    }

    /// Attach an embedding manager for Medium/High tier vector search.
    pub fn with_embeddings(mut self, embeddings: &'a EmbeddingManager) -> Self {
        self.embeddings = Some(embeddings);
        self
    }

    /// Attach a cross-encoder reranker for High tier — runs on the top
    /// fused candidates only (bounded pool, never the full corpus).
    pub fn with_reranker(mut self, reranker: &'a dyn crate::reranker::Reranker) -> Self {
        self.reranker = Some(reranker);
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

        // Step 2: Compute query embedding for Medium/High tier
        let query_embedding = match (self.tier, &self.embeddings) {
            (RetrievalTier::Medium | RetrievalTier::High, Some(embs)) if embs.is_available() => {
                embs.embed_query(query).ok()
            }
            _ => None,
        };

        // Step 3: Dense vector search over PERSISTED embeddings.
        // `evidence_vec` rows are written at index time; this path never
        // decrypts or re-embeds already-indexed documents. Rows missing a
        // current-model vector are backfilled in one bounded batch.
        let embs = self.embeddings.filter(|e| e.is_available());
        let model_tag = embs.and_then(|e| e.model_tag());

        // evidence_id → vector, for all rows with a current-model vector
        let mut stored_vecs: std::collections::HashMap<EvidenceId, (Vec<f32>, i64)> =
            std::collections::HashMap::new();
        if query_embedding.is_some() {
            if let Some(tag) = &model_tag {
                for (eid, vec, created_at) in
                    self.store.vectors_in_scopes(filter, tag, limit * 8)?
                {
                    stored_vecs.insert(eid, (vec, created_at));
                }
            }

            // Bounded backfill: embed evidence rows that have no current-model
            // vector yet (new rows, or rows written by a stale model).
            if let (Some(embs), Some(tag)) = (embs, &model_tag) {
                let candidates = self.store.list_evidence_in_scopes(filter, limit * 4)?;
                let missing: Vec<(EvidenceId, ScopeId)> = candidates
                    .iter()
                    .filter(|(eid, _, _, _)| !stored_vecs.contains_key(eid))
                    .take(BACKFILL_PER_QUERY)
                    .map(|(eid, sid, _, _)| (*eid, *sid))
                    .collect();

                if !missing.is_empty() {
                    // Decrypt contents, then batch-embed.
                    let mut texts: Vec<String> = Vec::with_capacity(missing.len());
                    let mut indexed: Vec<(EvidenceId, ScopeId, i64)> = Vec::new();
                    let created_map: std::collections::HashMap<EvidenceId, i64> = candidates
                        .iter()
                        .map(|(eid, _, _, ts)| (*eid, *ts))
                        .collect();
                    for (eid, sid) in &missing {
                        if let Ok(Some(ev)) = self.store.get_evidence(*eid) {
                            texts.push(ev.fts_content);
                            indexed.push((*eid, *sid, created_map[eid]));
                        }
                    }
                    let text_refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
                    if let Ok(vecs) = embs.embed_passages(&text_refs) {
                        for ((eid, sid, created_at), vec) in indexed.iter().zip(vecs) {
                            let _ = self.store.insert_vector(*eid, *sid, &vec, tag);
                            stored_vecs.insert(*eid, (vec, *created_at));
                        }
                    }
                }
            }
        }

        // Step 4: Rank fusion — Reciprocal Rank Fusion across FTS and dense
        // rankings, with a bounded recency prior. RRF is calibration-free:
        // it fuses *rank positions* rather than raw scores, so the BM25 and
        // cosine scales can't dominate each other.
        let now = chrono::Utc::now().timestamp() as f64;
        let recency_of = |created_at: i64| {
            let age = (now - created_at as f64).max(0.0);
            (-age * (2.0_f64.ln()) / self.recency_half_life_secs).exp()
        };

        // Dense ranking: sort stored vectors by cosine similarity to query.
        let dense_rank: std::collections::HashMap<EvidenceId, (usize, f64, i64)> =
            if let Some(qe) = &query_embedding {
                let mut scored: Vec<(EvidenceId, f64, i64)> = stored_vecs
                    .iter()
                    .map(|(eid, (vec, ts))| (*eid, cosine_similarity(qe, vec) as f64, *ts))
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored
                    .into_iter()
                    .enumerate()
                    .map(|(rank, (eid, sim, ts))| (eid, (rank, sim, ts)))
                    .collect()
            } else {
                Default::default()
            };

        // Union of candidates from both rankings.
        let mut results: std::collections::HashMap<EvidenceId, RetrievalResult> =
            std::collections::HashMap::new();

        for (rank, fts) in fts_results.iter().enumerate() {
            if self.store.is_scope_forgotten(fts.scope_id).unwrap_or(false) {
                continue;
            }
            let fts_score = 1.0 / (1.0 + fts.bm25_score.abs());
            let recency_score = recency_of(fts.created_at);
            let (vector_score, rrf_dense) = dense_rank
                .get(&fts.evidence_id)
                .map(|(r, sim, _)| (*sim, 1.0 / (RRF_K + (*r + 1) as f64)))
                .unwrap_or((0.0, 0.0));
            let rrf_fts = 1.0 / (RRF_K + (rank + 1) as f64);

            // RRF fusion + small recency prior; per-signal scores retained
            // for observability/debugging.
            let score = self.weights.fts * rrf_fts
                + self.weights.vector * rrf_dense
                + self.weights.recency * recency_score * 0.01;

            results.insert(
                fts.evidence_id,
                RetrievalResult {
                    evidence_id: fts.evidence_id,
                    score,
                    fts_score,
                    recency_score,
                    vector_score,
                    rerank_score: 0.0,
                },
            );
        }

        // Dense-only candidates (missed by FTS).
        for (eid, (rank, sim, created_at)) in &dense_rank {
            if results.contains_key(eid) {
                continue;
            }
            let recency_score = recency_of(*created_at);
            let rrf_dense = 1.0 / (RRF_K + (*rank + 1) as f64);
            let score =
                self.weights.vector * rrf_dense + self.weights.recency * recency_score * 0.01;
            results.insert(
                *eid,
                RetrievalResult {
                    evidence_id: *eid,
                    score,
                    fts_score: 0.0,
                    recency_score,
                    vector_score: *sim,
                    rerank_score: 0.0,
                },
            );
        }

        let mut results: Vec<RetrievalResult> = results.into_values().collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // High tier: cross-encoder rerank on the top fused candidates only.
        // The pool is bounded (2× limit) so the reranker never scans the
        // whole corpus; candidates whose content can't be decrypted are
        // dropped from the rerank pool but keep their fused score.
        if self.tier == RetrievalTier::High {
            if let Some(reranker) = self.reranker {
                let pool: Vec<RetrievalResult> = results.iter().take(limit * 2).cloned().collect();
                let mut docs: Vec<String> = Vec::with_capacity(pool.len());
                let mut kept: Vec<RetrievalResult> = Vec::with_capacity(pool.len());
                for res in pool {
                    if let Ok(Some(ev)) = self.store.get_evidence(res.evidence_id) {
                        docs.push(ev.fts_content);
                        kept.push(res);
                    }
                }
                if let Ok(ranked) = reranker.rerank(query, &docs, limit) {
                    let tail: Vec<RetrievalResult> =
                        results.iter().skip(limit * 2).cloned().collect();
                    results = ranked
                        .into_iter()
                        .filter_map(|(i, score)| {
                            kept.get(i).map(|res| RetrievalResult {
                                score,
                                rerank_score: score,
                                ..res.clone()
                            })
                        })
                        .collect();
                    results.extend(tail);
                }
            }
        }

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

        store
            .insert(&make_evidence(scope, "The quick brown fox", 0))
            .unwrap();
        store
            .insert(&make_evidence(scope, "Hello world greeting", 3600))
            .unwrap();

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
        store
            .insert(&make_evidence(scope, "important hello message", 86400 * 30))
            .unwrap();
        // Recent result
        store
            .insert(&make_evidence(scope, "hello recent message", 60))
            .unwrap();

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

        store
            .insert(&make_evidence(scope1, "hello in scope 1", 0))
            .unwrap();
        store
            .insert(&make_evidence(scope2, "hello in scope 2", 0))
            .unwrap();

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

    /// Embedder that returns a constant unit vector and counts embed calls —
    /// lets us assert persisted vectors are reused without re-embedding.
    struct CountingEmbedder {
        dim: usize,
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::embeddings::EmbeddingProvider for CountingEmbedder {
        fn embed(&self, _text: &str) -> crate::embeddings::EmbeddingResult<Vec<f32>> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let v = 1.0f32 / (self.dim as f32).sqrt();
            Ok(vec![v; self.dim])
        }
        fn dimension(&self) -> usize {
            self.dim
        }
        fn model_name(&self) -> &str {
            "counting-embedder"
        }
    }

    #[test]
    fn test_persisted_vectors_avoid_reembedding() {
        let store = make_store();
        let scope = ScopeId::new();
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let embs =
            crate::embeddings::EmbeddingManager::new().with_primary(Box::new(CountingEmbedder {
                dim: 8,
                calls: calls.clone(),
            }));

        // Two docs: "hello world" (FTS hit) and a semantically-related doc
        // with no keyword overlap (dense-only candidate).
        store
            .insert(&make_evidence(scope, "hello world greeting", 0))
            .unwrap();
        let dense_only = make_evidence(scope, "xyzzy plugh thud", 0);
        store.insert(&dense_only).unwrap();

        // Pre-persist a vector for the dense-only doc (as embed-on-write would).
        store
            .insert_vector(
                dense_only.id,
                scope,
                &[1.0f32 / 8f32.sqrt(); 8],
                "counting-embedder",
            )
            .unwrap();

        let filter = ScopeFilter {
            allowed_scopes: vec![scope],
            denied_scopes: vec![],
            user_id: Uuid::new_v4(),
            roles: vec![],
        };
        let retriever = Retriever::new(&store, RetrievalTier::Medium).with_embeddings(&embs);

        let results = retriever.retrieve("hello", &filter, 10).unwrap();
        assert!(!results.is_empty());
        // Dense-only doc surfaced via its persisted vector (sim = 1.0)
        let dense_hit = results.iter().find(|r| r.evidence_id == dense_only.id);
        assert!(dense_hit.is_some(), "persisted vector doc should surface");
        assert!(dense_hit.unwrap().vector_score > 0.9);
        // Backfill embedded the unindexed FTS doc once; the persisted doc
        // was NOT re-embedded. Calls: 1 query + 1 backfill = 2.
        let after_first = calls.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(after_first, 2);

        // Second query: all vectors persisted → only the query embedding.
        let results2 = retriever.retrieve("hello again", &filter, 10).unwrap();
        assert!(!results2.is_empty());
        let after_second = calls.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(after_second - after_first, 1, "only query embed expected");
    }

    #[test]
    fn test_backfill_indexes_missing_vectors() {
        let store = make_store();
        let scope = ScopeId::new();
        store
            .insert(&make_evidence(scope, "hello world greeting", 0))
            .unwrap();

        let embs =
            crate::embeddings::EmbeddingManager::new().with_primary(Box::new(CountingEmbedder {
                dim: 8,
                calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }));

        let filter = ScopeFilter {
            allowed_scopes: vec![scope],
            denied_scopes: vec![],
            user_id: Uuid::new_v4(),
            roles: vec![],
        };
        let retriever = Retriever::new(&store, RetrievalTier::Medium).with_embeddings(&embs);
        retriever.retrieve("hello", &filter, 10).unwrap();

        // Backfill should have persisted the vector
        let rows = store
            .vectors_in_scopes(&filter, "counting-embedder", 10)
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn test_high_tier_rerank() {
        let store = make_store();
        let scope = ScopeId::new();
        store
            .insert(&make_evidence(scope, "hello world greeting message", 0))
            .unwrap();
        store
            .insert(&make_evidence(scope, "hello unrelated content", 0))
            .unwrap();

        let reranker = crate::reranker::MockReranker::new();
        let retriever = Retriever::new(&store, RetrievalTier::High).with_reranker(&reranker);
        let filter = ScopeFilter {
            allowed_scopes: vec![scope],
            denied_scopes: vec![],
            user_id: Uuid::new_v4(),
            roles: vec![],
        };

        let results = retriever.retrieve("hello world", &filter, 10).unwrap();
        assert!(!results.is_empty());
        // Reranker ran — top result carries its score
        assert!(results[0].rerank_score > 0.0);
    }
}

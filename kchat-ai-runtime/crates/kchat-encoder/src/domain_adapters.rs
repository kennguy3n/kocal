//! On-device adapter layers for domain-specific embeddings.
//!
//! The base XLM-R model is small and CPU-friendly, but it doesn't
//! specialise for high-value domains (crypto, sports, regional
//! dialects). Rather than shipping a separate model per domain —
//! which would blow the on-device storage budget — we add lightweight
//! **adapter layers** that transform the base embedding into a
//! domain-specific one.
//!
//! # How it works
//!
//! 1. Each [`DomainAdapter`] is a rank-`r` matrix `A ∈ R^{d×r}` and
//!    `B ∈ R^{r×d}` where `d = EMBEDDING_DIM` and `r` is small
//!    (typically 8-16). The adapted embedding is:
//!    ```text
//!    e_adapted = e_base + A · B · e_base
//!    ```
//!    This is a **residual** adapter — the base embedding is always
//!    preserved, so the adapter can only refine, not replace.
//!
//! 2. Per-domain **recall metrics** are logged to detect drift. If
//!    recall drops below a threshold, the adapter is flagged for
//!    retraining.
//!
//! # Privacy
//!
//! - Adapter weights are stored locally in the SQLCipher DB.
//! - No training data or embeddings leave the device.
//! - The adapter is deterministic given the weights.
//!
//! Gated behind the `domain-adapters` cargo feature.

#![cfg(feature = "domain-adapters")]

use crate::EMBEDDING_DIM;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Default adapter rank (bottleneck dimension).
pub const DEFAULT_ADAPTER_RANK: usize = 16;

/// Minimum recall threshold before an adapter is flagged for retraining.
pub const RECALL_DRIFT_THRESHOLD: f32 = 0.70;

/// SQL to create the adapter tables.
pub const MIGRATION_V10_SQL: &str = r#"
-- On-device domain adapter weights.
-- Each row stores a domain's A and B matrices as f32-LE blobs.
CREATE TABLE IF NOT EXISTS domain_adapter (
    domain_id    TEXT PRIMARY KEY,
    adapter_rank INTEGER NOT NULL,
    matrix_a     BLOB NOT NULL,
    matrix_b     BLOB NOT NULL,
    trained_at_ms INTEGER NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0
);

-- Per-domain recall metrics logged after each evaluation.
CREATE TABLE IF NOT EXISTS adapter_recall_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    domain_id    TEXT NOT NULL REFERENCES domain_adapter(domain_id),
    recall       REAL NOT NULL,
    evaluated_at_ms INTEGER NOT NULL,
    sample_count INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_adapter_recall_domain
    ON adapter_recall_log(domain_id, evaluated_at_ms);
"#;

/// A domain adapter with rank-r bottleneck matrices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAdapter {
    pub domain_id: String,
    pub adapter_rank: usize,
    /// `A ∈ R^{d×r}` stored row-major.
    pub matrix_a: Vec<f32>,
    /// `B ∈ R^{r×d}` stored row-major.
    pub matrix_b: Vec<f32>,
    pub trained_at_ms: i64,
    pub sample_count: usize,
}

impl DomainAdapter {
    /// Create a new adapter with identity-like initialisation
    /// (A = 0, B = 0 → adapted = base).
    pub fn new(domain_id: &str, rank: usize, now_ms: i64) -> Self {
        let d = EMBEDDING_DIM;
        Self {
            domain_id: domain_id.to_string(),
            adapter_rank: rank,
            matrix_a: vec![0.0; d * rank],
            matrix_b: vec![0.0; rank * d],
            trained_at_ms: now_ms,
            sample_count: 0,
        }
    }

    /// Apply the adapter to a base embedding.
    ///
    /// `e_adapted = e_base + A · (B · e_base)`
    ///
    /// If the embedding dimension doesn't match `EMBEDDING_DIM`,
    /// the base embedding is returned unchanged.
    pub fn adapt(&self, base_embedding: &[f32]) -> Vec<f32> {
        let d = EMBEDDING_DIM;
        if base_embedding.len() != d {
            return base_embedding.to_vec();
        }

        let r = self.adapter_rank;
        if r == 0 || self.matrix_a.is_empty() || self.matrix_b.is_empty() {
            return base_embedding.to_vec();
        }

        // Compute B · e_base ∈ R^r
        let mut bottleneck = vec![0.0f32; r];
        for (i, dot) in bottleneck.iter_mut().enumerate() {
            let row = &self.matrix_b[i * d..(i + 1) * d];
            *dot = row
                .iter()
                .zip(base_embedding.iter())
                .map(|(b, e)| b * e)
                .sum();
        }

        // Compute A · bottleneck ∈ R^d, add to base
        let mut result = base_embedding.to_vec();
        for (i, result_i) in result.iter_mut().enumerate() {
            let col = &self.matrix_a[i * r..(i + 1) * r];
            let dot: f32 = col.iter().zip(bottleneck.iter()).map(|(a, b)| a * b).sum();
            *result_i += dot;
        }

        result
    }

    /// Encode matrices as f32-LE bytes for storage.
    pub fn encode_weights(&self) -> Result<(Vec<u8>, Vec<u8>), String> {
        let a = encode_f32_slice(&self.matrix_a);
        let b = encode_f32_slice(&self.matrix_b);
        Ok((a, b))
    }

    /// Decode matrices from f32-LE storage blobs.
    pub fn decode_weights(
        domain_id: &str,
        rank: usize,
        a_blob: &[u8],
        b_blob: &[u8],
        trained_at_ms: i64,
        sample_count: usize,
    ) -> Result<Self, String> {
        let matrix_a = decode_f32_slice(a_blob)?;
        let matrix_b = decode_f32_slice(b_blob)?;
        Ok(Self {
            domain_id: domain_id.to_string(),
            adapter_rank: rank,
            matrix_a,
            matrix_b,
            trained_at_ms,
            sample_count,
        })
    }
}

/// A recall measurement for a domain adapter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallMeasurement {
    pub domain_id: String,
    pub recall: f32,
    pub evaluated_at_ms: i64,
    pub sample_count: usize,
}

/// Whether an adapter needs retraining based on recent recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterStatus {
    /// Recall is above threshold — adapter is healthy.
    Healthy,
    /// Recall has dropped below threshold — retraining recommended.
    DriftDetected,
    /// No recall data available — adapter has never been evaluated.
    NotEvaluated,
}

/// Persistence layer for domain adapters.
#[derive(Debug)]
pub struct DomainAdapterStore;

impl DomainAdapterStore {
    /// Apply the adapter schema migration.
    pub fn apply_schema(conn: &Connection) -> Result<(), String> {
        conn.execute_batch(MIGRATION_V10_SQL)
            .map_err(|e| format!("domain_adapter schema migration failed: {e}"))
    }

    /// Save or update an adapter.
    pub fn upsert_adapter(conn: &Connection, adapter: &DomainAdapter) -> Result<(), String> {
        let (a_blob, b_blob) = adapter.encode_weights()?;
        conn.execute(
            "INSERT OR REPLACE INTO domain_adapter
                (domain_id, adapter_rank, matrix_a, matrix_b, trained_at_ms, sample_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                adapter.domain_id,
                adapter.adapter_rank as i64,
                a_blob,
                b_blob,
                adapter.trained_at_ms,
                adapter.sample_count as i64,
            ],
        )
        .map_err(|e| format!("upsert adapter: {e}"))?;
        Ok(())
    }

    /// Load an adapter by domain ID.
    pub fn get_adapter(
        conn: &Connection,
        domain_id: &str,
    ) -> Result<Option<DomainAdapter>, String> {
        let result = conn.query_row(
            "SELECT domain_id, adapter_rank, matrix_a, matrix_b, trained_at_ms, sample_count
             FROM domain_adapter WHERE domain_id = ?1",
            params![domain_id],
            |row| {
                let domain_id: String = row.get(0)?;
                let rank: i64 = row.get(1)?;
                let a_blob: Vec<u8> = row.get(2)?;
                let b_blob: Vec<u8> = row.get(3)?;
                let trained_at_ms: i64 = row.get(4)?;
                let sample_count: i64 = row.get(5)?;
                Ok(DomainAdapter::decode_weights(
                    &domain_id,
                    rank as usize,
                    &a_blob,
                    &b_blob,
                    trained_at_ms,
                    sample_count as usize,
                ))
            },
        );

        match result {
            Ok(adapter) => Ok(Some(adapter.map_err(|e| format!("decode: {e}"))?)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("get adapter: {e}")),
        }
    }

    /// List all domain IDs that have adapters.
    pub fn list_domains(conn: &Connection) -> Result<Vec<String>, String> {
        let mut stmt = conn
            .prepare("SELECT domain_id FROM domain_adapter ORDER BY domain_id")
            .map_err(|e| format!("prepare domains: {e}"))?;
        let rows = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| format!("query domains: {e}"))?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| format!("row: {e}"))?);
        }
        Ok(results)
    }

    /// Delete an adapter.
    pub fn delete_adapter(conn: &Connection, domain_id: &str) -> Result<(), String> {
        conn.execute(
            "DELETE FROM domain_adapter WHERE domain_id = ?1",
            params![domain_id],
        )
        .map_err(|e| format!("delete adapter: {e}"))?;
        Ok(())
    }

    /// Log a recall measurement for a domain.
    pub fn log_recall(conn: &Connection, measurement: &RecallMeasurement) -> Result<(), String> {
        conn.execute(
            "INSERT INTO adapter_recall_log (domain_id, recall, evaluated_at_ms, sample_count)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                measurement.domain_id,
                measurement.recall,
                measurement.evaluated_at_ms,
                measurement.sample_count as i64,
            ],
        )
        .map_err(|e| format!("log recall: {e}"))?;
        Ok(())
    }

    /// Get the most recent recall measurement for a domain.
    pub fn get_latest_recall(
        conn: &Connection,
        domain_id: &str,
    ) -> Result<Option<RecallMeasurement>, String> {
        let result = conn.query_row(
            "SELECT domain_id, recall, evaluated_at_ms, sample_count
             FROM adapter_recall_log
             WHERE domain_id = ?1
             ORDER BY evaluated_at_ms DESC LIMIT 1",
            params![domain_id],
            |row| {
                Ok(RecallMeasurement {
                    domain_id: row.get(0)?,
                    recall: row.get(1)?,
                    evaluated_at_ms: row.get(2)?,
                    sample_count: row.get::<_, i64>(3)? as usize,
                })
            },
        );

        match result {
            Ok(m) => Ok(Some(m)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("get latest recall: {e}")),
        }
    }

    /// Get recall history for a domain (most recent first).
    pub fn get_recall_history(
        conn: &Connection,
        domain_id: &str,
        limit: usize,
    ) -> Result<Vec<RecallMeasurement>, String> {
        let mut stmt = conn
            .prepare(
                "SELECT domain_id, recall, evaluated_at_ms, sample_count
                 FROM adapter_recall_log
                 WHERE domain_id = ?1
                 ORDER BY evaluated_at_ms DESC LIMIT ?2",
            )
            .map_err(|e| format!("prepare history: {e}"))?;
        let rows = stmt
            .query_map(params![domain_id, limit as i64], |row| {
                Ok(RecallMeasurement {
                    domain_id: row.get(0)?,
                    recall: row.get(1)?,
                    evaluated_at_ms: row.get(2)?,
                    sample_count: row.get::<_, i64>(3)? as usize,
                })
            })
            .map_err(|e| format!("query history: {e}"))?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| format!("row: {e}"))?);
        }
        Ok(results)
    }

    /// Check adapter status based on latest recall.
    pub fn adapter_status(conn: &Connection, domain_id: &str) -> Result<AdapterStatus, String> {
        match Self::get_latest_recall(conn, domain_id)? {
            None => Ok(AdapterStatus::NotEvaluated),
            Some(m) => {
                if m.recall < RECALL_DRIFT_THRESHOLD {
                    Ok(AdapterStatus::DriftDetected)
                } else {
                    Ok(AdapterStatus::Healthy)
                }
            }
        }
    }

    /// Get status for all domains.
    pub fn all_adapter_statuses(
        conn: &Connection,
    ) -> Result<HashMap<String, AdapterStatus>, String> {
        let domains = Self::list_domains(conn)?;
        let mut statuses = HashMap::new();
        for domain in domains {
            let status = Self::adapter_status(conn, &domain)?;
            statuses.insert(domain, status);
        }
        Ok(statuses)
    }
}

/// Compute recall@k for a set of query-result pairs.
pub fn compute_recall_at_k(
    queries: &[(Vec<f32>, Vec<usize>)],
    corpus: &[Vec<f32>],
    k: usize,
) -> f32 {
    if queries.is_empty() {
        return 0.0;
    }

    let mut hits = 0usize;
    let mut total = 0usize;

    for (query_emb, relevant) in queries {
        let mut scores: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let sim = cosine(query_emb, doc);
                (i, sim)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k: Vec<usize> = scores.iter().take(k).map(|(i, _)| *i).collect();
        for &rel_idx in relevant {
            if top_k.contains(&rel_idx) {
                hits += 1;
            }
            total += 1;
        }
    }

    if total == 0 {
        return 0.0;
    }
    hits as f32 / total as f32
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Encode a `&[f32]` as little-endian bytes.
fn encode_f32_slice(slice: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(slice.len() * 4);
    for &x in slice {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Decode a `Vec<f32>` from little-endian bytes. A trailing
/// partial lane (corrupt blob) is ignored.
fn decode_f32_slice(blob: &[u8]) -> Result<Vec<f32>, String> {
    Ok(blob
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        DomainAdapterStore::apply_schema(&conn).unwrap();
        conn
    }

    #[test]
    fn adapter_identity_when_untrained() {
        let adapter = DomainAdapter::new("crypto", 16, 1000);
        let base = vec![0.5; EMBEDDING_DIM];
        let adapted = adapter.adapt(&base);
        assert_eq!(adapted, base);
    }

    #[test]
    fn adapter_preserves_dimension() {
        let adapter = DomainAdapter::new("crypto", 8, 1000);
        let base = vec![0.3; EMBEDDING_DIM];
        let adapted = adapter.adapt(&base);
        assert_eq!(adapted.len(), EMBEDDING_DIM);
    }

    #[test]
    fn adapter_wrong_dim_returns_base() {
        let adapter = DomainAdapter::new("crypto", 8, 1000);
        let base = vec![0.3; 128];
        let adapted = adapter.adapt(&base);
        assert_eq!(adapted, base);
    }

    #[test]
    fn adapter_nonzero_matrices_modify_embedding() {
        let mut adapter = DomainAdapter::new("crypto", 4, 1000);
        let d = EMBEDDING_DIM;
        for i in 0..d {
            for j in 0..4 {
                adapter.matrix_a[i * 4 + j] = 0.1;
                adapter.matrix_b[j * d + i] = 0.01;
            }
        }
        let base = vec![1.0; d];
        let adapted = adapter.adapt(&base);
        assert_ne!(adapted, base);
        assert_eq!(adapted.len(), d);
    }

    #[test]
    fn store_round_trip() {
        let conn = open_db();
        let adapter = DomainAdapter::new("crypto", 16, 1000);
        DomainAdapterStore::upsert_adapter(&conn, &adapter).unwrap();

        let loaded = DomainAdapterStore::get_adapter(&conn, "crypto").unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.domain_id, "crypto");
        assert_eq!(loaded.adapter_rank, 16);
        assert_eq!(loaded.matrix_a, adapter.matrix_a);
        assert_eq!(loaded.matrix_b, adapter.matrix_b);
    }

    #[test]
    fn store_get_missing_returns_none() {
        let conn = open_db();
        let result = DomainAdapterStore::get_adapter(&conn, "nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn store_list_domains() {
        let conn = open_db();
        DomainAdapterStore::upsert_adapter(&conn, &DomainAdapter::new("crypto", 8, 1000)).unwrap();
        DomainAdapterStore::upsert_adapter(&conn, &DomainAdapter::new("sports", 8, 2000)).unwrap();

        let domains = DomainAdapterStore::list_domains(&conn).unwrap();
        assert_eq!(domains.len(), 2);
        assert!(domains.contains(&"crypto".to_string()));
        assert!(domains.contains(&"sports".to_string()));
    }

    #[test]
    fn store_delete_adapter() {
        let conn = open_db();
        DomainAdapterStore::upsert_adapter(&conn, &DomainAdapter::new("crypto", 8, 1000)).unwrap();
        DomainAdapterStore::delete_adapter(&conn, "crypto").unwrap();
        assert!(DomainAdapterStore::get_adapter(&conn, "crypto")
            .unwrap()
            .is_none());
    }

    #[test]
    fn recall_logging_and_retrieval() {
        let conn = open_db();
        DomainAdapterStore::upsert_adapter(&conn, &DomainAdapter::new("crypto", 8, 1000)).unwrap();
        let m1 = RecallMeasurement {
            domain_id: "crypto".to_string(),
            recall: 0.85,
            evaluated_at_ms: 1000,
            sample_count: 100,
        };
        let m2 = RecallMeasurement {
            domain_id: "crypto".to_string(),
            recall: 0.72,
            evaluated_at_ms: 2000,
            sample_count: 120,
        };
        DomainAdapterStore::log_recall(&conn, &m1).unwrap();
        DomainAdapterStore::log_recall(&conn, &m2).unwrap();

        let latest = DomainAdapterStore::get_latest_recall(&conn, "crypto").unwrap();
        assert!(latest.is_some());
        let latest = latest.unwrap();
        assert_eq!(latest.recall, 0.72);
        assert_eq!(latest.evaluated_at_ms, 2000);

        let history = DomainAdapterStore::get_recall_history(&conn, "crypto", 10).unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].evaluated_at_ms, 2000);
        assert_eq!(history[1].evaluated_at_ms, 1000);
    }

    #[test]
    fn adapter_status_healthy() {
        let conn = open_db();
        DomainAdapterStore::upsert_adapter(&conn, &DomainAdapter::new("crypto", 8, 1000)).unwrap();
        DomainAdapterStore::log_recall(
            &conn,
            &RecallMeasurement {
                domain_id: "crypto".to_string(),
                recall: 0.90,
                evaluated_at_ms: 1000,
                sample_count: 100,
            },
        )
        .unwrap();
        assert_eq!(
            DomainAdapterStore::adapter_status(&conn, "crypto").unwrap(),
            AdapterStatus::Healthy
        );
    }

    #[test]
    fn adapter_status_drift() {
        let conn = open_db();
        DomainAdapterStore::upsert_adapter(&conn, &DomainAdapter::new("crypto", 8, 1000)).unwrap();
        DomainAdapterStore::log_recall(
            &conn,
            &RecallMeasurement {
                domain_id: "crypto".to_string(),
                recall: 0.50,
                evaluated_at_ms: 1000,
                sample_count: 100,
            },
        )
        .unwrap();
        assert_eq!(
            DomainAdapterStore::adapter_status(&conn, "crypto").unwrap(),
            AdapterStatus::DriftDetected
        );
    }

    #[test]
    fn adapter_status_not_evaluated() {
        let conn = open_db();
        assert_eq!(
            DomainAdapterStore::adapter_status(&conn, "crypto").unwrap(),
            AdapterStatus::NotEvaluated
        );
    }

    #[test]
    fn compute_recall_at_k_simple() {
        let corpus = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let queries = vec![
            (vec![1.0, 0.1, 0.0], vec![0]),
            (vec![0.0, 1.0, 0.1], vec![1]),
        ];
        let recall = compute_recall_at_k(&queries, &corpus, 1);
        assert!((recall - 1.0).abs() < 1e-6);
    }

    #[test]
    fn compute_recall_at_k_partial() {
        let corpus = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let queries = vec![(vec![1.0, 0.0, 0.0], vec![2])];
        let recall = compute_recall_at_k(&queries, &corpus, 1);
        assert!((recall - 0.0).abs() < 1e-6);
    }

    #[test]
    fn compute_recall_empty_queries() {
        let recall = compute_recall_at_k(&[], &[], 5);
        assert_eq!(recall, 0.0);
    }
}

//! Cross-pipeline inference deduplication cache.
//!
//! Lets the guardrail and search pipelines share one inference
//! result per message. Generalises to all on-device ML pipelines
//! whose output is expensive to recompute:
//!
//! * OCR text extraction (`~200 ms`),
//! * Whisper transcription (`~2000 ms`),
//! * MobileCLIP-S2 image embedding (`~50 ms`).
//!
//! If the guardrail already ran OCR on an image to screen it for
//! policy violations, the search-indexer reads the cached text
//! back through [`InferenceCache::get_ocr`] instead of re-running
//! the OCR model — and vice versa. The cache is keyed purely by
//! `message_id`, so any pipeline that observes a message first
//! wins.
//!
//! The backing store is SQLCipher. A configurable byte budget
//! (default 100 MB) bounds growth: every write enforces an LRU
//! eviction so the cache can never grow without limit.
//!
//! Gated behind the `sqlcipher` cargo feature.

#![cfg(feature = "sqlcipher")]

use crate::error::{CoreError, Result};
use rusqlite::{params, Connection, OptionalExtension};

/// Default cache ceiling: 100 MB of payload bytes across all kinds.
pub const DEFAULT_MAX_BYTES: u64 = 100 * 1024 * 1024;

/// Cache configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceCacheConfig {
    /// Maximum total payload bytes retained before LRU eviction
    /// kicks in.
    pub max_bytes: u64,
}

impl Default for InferenceCacheConfig {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

/// The ML pipelines whose results the cache stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CacheKind {
    Ocr,
    Transcript,
    ImageEmbedding,
}

impl CacheKind {
    fn as_str(self) -> &'static str {
        match self {
            CacheKind::Ocr => "ocr",
            CacheKind::Transcript => "transcript",
            CacheKind::ImageEmbedding => "image_embedding",
        }
    }
}

/// Cross-pipeline inference cache.
///
/// All lookups are keyed by `message_id`; the cache does not care
/// which pipeline produced the entry, only which kind of result it
/// is. A `get` is also a "touch" that refreshes the entry's LRU
/// recency so frequently-read entries survive eviction.
pub trait InferenceCache: std::fmt::Debug {
    /// Cached OCR text for `message_id`, or `None` on a miss.
    fn get_ocr(&self, message_id: &str) -> Result<Option<String>>;
    /// Cached transcript for `message_id`, or `None` on a miss.
    fn get_transcript(&self, message_id: &str) -> Result<Option<String>>;
    /// Cached MobileCLIP image embedding for `message_id`, or
    /// `None` on a miss.
    fn get_image_embedding(&self, message_id: &str) -> Result<Option<Vec<f32>>>;

    /// Store OCR `text` for `message_id` (upsert).
    fn put_ocr(&self, message_id: &str, text: &str) -> Result<()>;
    /// Store transcript `text` for `message_id` (upsert).
    fn put_transcript(&self, message_id: &str, text: &str) -> Result<()>;
    /// Store image `embedding` for `message_id` (upsert).
    fn put_image_embedding(&self, message_id: &str, embedding: &[f32]) -> Result<()>;
}

/// Cache-disabled [`InferenceCache`] used when no SQLCipher-backed
/// store is wired in.
///
/// Every `get` misses and every `put` is silently dropped with
/// `Ok(())`. Deduplication is an optimisation, so disabling it
/// must never turn a successful media-ingest into an error.
#[derive(Debug, Default, Clone, Copy)]
pub struct DisabledInferenceCache;

impl InferenceCache for DisabledInferenceCache {
    fn get_ocr(&self, _message_id: &str) -> Result<Option<String>> {
        Ok(None)
    }
    fn get_transcript(&self, _message_id: &str) -> Result<Option<String>> {
        Ok(None)
    }
    fn get_image_embedding(&self, _message_id: &str) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }
    fn put_ocr(&self, _message_id: &str, _text: &str) -> Result<()> {
        Ok(())
    }
    fn put_transcript(&self, _message_id: &str, _text: &str) -> Result<()> {
        Ok(())
    }
    fn put_image_embedding(&self, _message_id: &str, _embedding: &[f32]) -> Result<()> {
        Ok(())
    }
}

/// SQLCipher-backed [`InferenceCache`].
///
/// Wraps a [`Connection`] so cached results inherit the same
/// encryption as the rest of the local store. The backing table
/// is created on construction with `CREATE TABLE IF NOT EXISTS`.
///
/// On-disk layout (`inference_cache`):
///
/// ```text
/// message_id       TEXT    -- owning message
/// kind             TEXT    -- 'ocr' | 'transcript' | 'image_embedding'
/// payload          BLOB    -- UTF-8 text, or f32-LE embedding lanes
/// byte_len         INTEGER -- payload.len(), for the budget tally
/// last_accessed_ms INTEGER -- LRU recency, refreshed on every get
/// access_seq       INTEGER -- monotonic LRU ordering
/// PRIMARY KEY (message_id, kind)
/// ```
pub struct LocalStoreInferenceCache<'a> {
    conn: &'a Connection,
    config: InferenceCacheConfig,
}

impl std::fmt::Debug for LocalStoreInferenceCache<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalStoreInferenceCache")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<'a> LocalStoreInferenceCache<'a> {
    /// Wrap `conn` with the default [`InferenceCacheConfig`],
    /// creating the backing table if it does not yet exist.
    pub fn new(conn: &'a Connection) -> Result<Self> {
        Self::with_config(conn, InferenceCacheConfig::default())
    }

    /// Wrap `conn` with an explicit cache configuration.
    pub fn with_config(conn: &'a Connection, config: InferenceCacheConfig) -> Result<Self> {
        Self::ensure_schema(conn)?;
        Ok(Self { conn, config })
    }

    /// Idempotently create the backing table.
    pub fn ensure_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS inference_cache (
                message_id       TEXT    NOT NULL,
                kind             TEXT    NOT NULL,
                payload          BLOB    NOT NULL,
                byte_len         INTEGER NOT NULL,
                last_accessed_ms INTEGER NOT NULL,
                access_seq       INTEGER NOT NULL,
                PRIMARY KEY (message_id, kind)
            );
            CREATE INDEX IF NOT EXISTS idx_inference_cache_lru
                ON inference_cache(access_seq);",
        )
        .map_err(|e| CoreError::Storage(format!("inference_cache schema: {e}")))?;
        Ok(())
    }

    fn now_ms() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis().min(i64::MAX as u128) as i64)
    }

    fn next_seq(&self) -> Result<i64> {
        self.conn
            .query_row(
                "SELECT COALESCE(MAX(access_seq), 0) + 1 FROM inference_cache",
                [],
                |row| row.get(0),
            )
            .map_err(|e| CoreError::Storage(format!("inference_cache seq: {e}")))
    }

    fn get_payload(&self, message_id: &str, kind: CacheKind) -> Result<Option<Vec<u8>>> {
        let blob: Option<Vec<u8>> = self
            .conn
            .query_row(
                "SELECT payload FROM inference_cache
                 WHERE message_id = ?1 AND kind = ?2",
                params![message_id, kind.as_str()],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()
            .map_err(|e| CoreError::Storage(format!("inference_cache lookup: {e}")))?;

        if blob.is_some() {
            let seq = self.next_seq()?;
            self.conn
                .execute(
                    "UPDATE inference_cache SET last_accessed_ms = ?3, access_seq = ?4
                     WHERE message_id = ?1 AND kind = ?2",
                    params![message_id, kind.as_str(), Self::now_ms(), seq],
                )
                .map_err(|e| CoreError::Storage(format!("inference_cache touch: {e}")))?;
        }
        Ok(blob)
    }

    fn put_payload(&self, message_id: &str, kind: CacheKind, payload: Vec<u8>) -> Result<()> {
        let byte_len = payload.len() as i64;
        let seq = self.next_seq()?;
        self.conn
            .execute(
                "INSERT INTO inference_cache(message_id, kind, payload, byte_len, last_accessed_ms, access_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(message_id, kind)
                 DO UPDATE SET payload = excluded.payload,
                               byte_len = excluded.byte_len,
                               last_accessed_ms = excluded.last_accessed_ms,
                               access_seq = excluded.access_seq",
                params![message_id, kind.as_str(), payload, byte_len, Self::now_ms(), seq],
            )
            .map_err(|e| CoreError::Storage(format!("inference_cache upsert: {e}")))?;
        self.enforce_budget()?;
        Ok(())
    }

    /// Total stored payload bytes across all entries.
    pub fn total_bytes(&self) -> Result<u64> {
        let total: i64 = self
            .conn
            .query_row(
                "SELECT COALESCE(SUM(byte_len), 0) FROM inference_cache",
                [],
                |row| row.get(0),
            )
            .map_err(|e| CoreError::Storage(format!("inference_cache tally: {e}")))?;
        Ok(total.max(0) as u64)
    }

    fn enforce_budget(&self) -> Result<()> {
        let mut total = self.total_bytes()?;
        if total <= self.config.max_bytes {
            return Ok(());
        }
        let rows: Vec<(String, String, i64)> = {
            let mut stmt = self
                .conn
                .prepare(
                    "SELECT message_id, kind, byte_len FROM inference_cache
                     ORDER BY access_seq ASC, message_id ASC, kind ASC",
                )
                .map_err(|e| CoreError::Storage(format!("inference_cache scan: {e}")))?;
            let mapped = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                })
                .map_err(|e| CoreError::Storage(format!("inference_cache scan: {e}")))?;
            let mut out = Vec::new();
            for r in mapped {
                out.push(r.map_err(|e| {
                    CoreError::Storage(format!("inference_cache scan: {e}"))
                })?);
            }
            out
        };

        for (message_id, kind, byte_len) in rows {
            if total <= self.config.max_bytes {
                break;
            }
            self.conn
                .execute(
                    "DELETE FROM inference_cache WHERE message_id = ?1 AND kind = ?2",
                    params![message_id, kind],
                )
                .map_err(|e| CoreError::Storage(format!("inference_cache evict: {e}")))?;
            total = total.saturating_sub(byte_len.max(0) as u64);
        }
        Ok(())
    }
}

fn encode_embedding(embedding: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(embedding.len() * 4);
    for &x in embedding {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn decode_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl InferenceCache for LocalStoreInferenceCache<'_> {
    fn get_ocr(&self, message_id: &str) -> Result<Option<String>> {
        Ok(self
            .get_payload(message_id, CacheKind::Ocr)?
            .map(|b| String::from_utf8_lossy(&b).into_owned()))
    }

    fn get_transcript(&self, message_id: &str) -> Result<Option<String>> {
        Ok(self
            .get_payload(message_id, CacheKind::Transcript)?
            .map(|b| String::from_utf8_lossy(&b).into_owned()))
    }

    fn get_image_embedding(&self, message_id: &str) -> Result<Option<Vec<f32>>> {
        Ok(self
            .get_payload(message_id, CacheKind::ImageEmbedding)?
            .map(|b| decode_embedding(&b)))
    }

    fn put_ocr(&self, message_id: &str, text: &str) -> Result<()> {
        self.put_payload(message_id, CacheKind::Ocr, text.as_bytes().to_vec())
    }

    fn put_transcript(&self, message_id: &str, text: &str) -> Result<()> {
        self.put_payload(message_id, CacheKind::Transcript, text.as_bytes().to_vec())
    }

    fn put_image_embedding(&self, message_id: &str, embedding: &[f32]) -> Result<()> {
        self.put_payload(
            message_id,
            CacheKind::ImageEmbedding,
            encode_embedding(embedding),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_cache(max_bytes: u64) -> (Connection, InferenceCacheConfig) {
        let conn = Connection::open_in_memory().expect("open_in_memory");
        LocalStoreInferenceCache::ensure_schema(&conn).expect("schema");
        (conn, InferenceCacheConfig { max_bytes })
    }

    #[test]
    fn ocr_round_trips() {
        let (conn, cfg) = fresh_cache(DEFAULT_MAX_BYTES);
        let cache = LocalStoreInferenceCache::with_config(&conn, cfg).unwrap();
        assert!(cache.get_ocr("m-1").unwrap().is_none());
        cache.put_ocr("m-1", "hello receipt $4.99").unwrap();
        assert_eq!(
            cache.get_ocr("m-1").unwrap().as_deref(),
            Some("hello receipt $4.99")
        );
    }

    #[test]
    fn transcript_round_trips() {
        let (conn, cfg) = fresh_cache(DEFAULT_MAX_BYTES);
        let cache = LocalStoreInferenceCache::with_config(&conn, cfg).unwrap();
        cache.put_transcript("m-2", "the quick brown fox").unwrap();
        assert_eq!(
            cache.get_transcript("m-2").unwrap().as_deref(),
            Some("the quick brown fox")
        );
        assert!(cache.get_ocr("m-2").unwrap().is_none());
    }

    #[test]
    fn image_embedding_round_trips() {
        let (conn, cfg) = fresh_cache(DEFAULT_MAX_BYTES);
        let cache = LocalStoreInferenceCache::with_config(&conn, cfg).unwrap();
        let v = vec![0.1_f32, -0.5, 0.25, 1.0, -1.0];
        cache.put_image_embedding("m-3", &v).unwrap();
        let got = cache.get_image_embedding("m-3").unwrap().unwrap();
        assert_eq!(got, v, "f32-LE codec is lossless");
    }

    #[test]
    fn put_overwrites_same_key() {
        let (conn, cfg) = fresh_cache(DEFAULT_MAX_BYTES);
        let cache = LocalStoreInferenceCache::with_config(&conn, cfg).unwrap();
        cache.put_ocr("m-4", "first").unwrap();
        cache.put_ocr("m-4", "second").unwrap();
        assert_eq!(cache.get_ocr("m-4").unwrap().as_deref(), Some("second"));
    }

    #[test]
    fn cross_pipeline_sharing() {
        let (conn, cfg) = fresh_cache(DEFAULT_MAX_BYTES);
        let guardrail = LocalStoreInferenceCache::with_config(&conn, cfg).unwrap();
        guardrail.put_ocr("shared-msg", "policy text").unwrap();

        let search = LocalStoreInferenceCache::with_config(&conn, cfg).unwrap();
        assert_eq!(
            search.get_ocr("shared-msg").unwrap().as_deref(),
            Some("policy text")
        );
    }

    #[test]
    fn lru_evicts_oldest_at_boundary() {
        let (conn, cfg) = fresh_cache(30);
        let cache = LocalStoreInferenceCache::with_config(&conn, cfg).unwrap();

        cache.put_ocr("a", "0123456789").unwrap();
        cache.put_ocr("b", "0123456789").unwrap();
        cache.put_ocr("c", "0123456789").unwrap();
        assert_eq!(cache.total_bytes().unwrap(), 30);

        assert!(cache.get_ocr("a").unwrap().is_some());

        cache.put_ocr("d", "0123456789").unwrap();
        assert!(cache.total_bytes().unwrap() <= 30, "cache stays bounded");
        assert!(cache.get_ocr("b").unwrap().is_none(), "LRU entry evicted");
        assert!(cache.get_ocr("a").unwrap().is_some(), "touched entry kept");
        assert!(cache.get_ocr("d").unwrap().is_some(), "newest entry kept");
    }

    #[test]
    fn cache_never_exceeds_budget_under_churn() {
        let (conn, cfg) = fresh_cache(50);
        let cache = LocalStoreInferenceCache::with_config(&conn, cfg).unwrap();
        for i in 0..200 {
            cache
                .put_transcript(&format!("msg-{i}"), "0123456789")
                .unwrap();
            assert!(
                cache.total_bytes().unwrap() <= 50,
                "exceeded budget at iteration {i}"
            );
        }
    }

    #[test]
    fn decode_embedding_ignores_trailing_partial_lane() {
        let decoded = decode_embedding(&[0, 0, 0x80, 0x3f, 0xaa, 0xbb]);
        assert_eq!(decoded, vec![1.0_f32]);
    }

    #[test]
    fn disabled_cache_misses_and_drops_writes() {
        let cache = DisabledInferenceCache;
        assert!(cache.get_ocr("x").unwrap().is_none());
        assert!(cache.get_transcript("x").unwrap().is_none());
        assert!(cache.get_image_embedding("x").unwrap().is_none());
        cache.put_ocr("x", "y").expect("disabled put_ocr is Ok");
        cache
            .put_transcript("x", "y")
            .expect("disabled put_transcript is Ok");
        cache
            .put_image_embedding("x", &[0.1, 0.2])
            .expect("disabled put_image_embedding is Ok");
    }
}

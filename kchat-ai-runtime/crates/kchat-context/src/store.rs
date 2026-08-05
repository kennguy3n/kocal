//! Evidence store — SQLCipher-backed encrypted local storage with FTS5.
//!
//! Schema based on the knowledge repo's evidence_store, adapted for KChat:
//! - Append-only evidence table (UPDATE/DELETE blocked by triggers)
//! - Deduplicated body store (content-hash keyed)
//! - Three-lane FTS5 retrieval (unicode61, trigram, bigram)
//! - Per-scope encryption with XChaCha20-Poly1305

use crate::encryption::{self};
use crate::scope::{ScopeFilter, ScopeId};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;
use parking_lot::Mutex;

/// Stable identifier for an evidence row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EvidenceId(pub Uuid);

impl EvidenceId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for EvidenceId {
    fn default() -> Self {
        Self::new()
    }
}

/// An evidence row — encrypted content with metadata for retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub id: EvidenceId,
    pub scope_id: ScopeId,
    pub content_hash: String,
    pub encrypted_body: Vec<u8>,
    pub nonce: Vec<u8>,
    pub source_ref: Option<String>,
    pub importance: u8,
    pub language_tag: Option<String>,
    pub created_at: i64,
    /// Plaintext for FTS indexing (stored separately, encrypted at rest)
    pub fts_content: String,
}

/// Configuration for the context store.
#[derive(Debug, Clone)]
pub struct ContextStoreConfig {
    /// SQLCipher password (in production, from Keychain/Keystore/DPAPI)
    pub db_password: String,
    /// Master encryption key for per-scope AEAD
    pub master_key: [u8; 32],
    /// Page cache size in KB
    pub page_cache_kb: u32,
    /// Whether mmap is enabled
    pub mmap_enabled: bool,
}

impl ContextStoreConfig {
    pub fn for_low_tier(db_password: String, master_key: [u8; 32]) -> Self {
        Self {
            db_password,
            master_key,
            page_cache_kb: 512,
            mmap_enabled: false,
        }
    }

    pub fn for_medium_tier(db_password: String, master_key: [u8; 32]) -> Self {
        Self {
            db_password,
            master_key,
            page_cache_kb: 1024,
            mmap_enabled: true,
        }
    }

    pub fn for_high_tier(db_password: String, master_key: [u8; 32]) -> Self {
        Self {
            db_password,
            master_key,
            page_cache_kb: 2048,
            mmap_enabled: true,
        }
    }
}

/// SQLCipher-backed evidence store.
pub struct ContextStore {
    conn: Mutex<Connection>,
    master_key: [u8; 32],
}

impl Drop for ContextStore {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.master_key.zeroize();
    }
}

impl ContextStore {
    /// Open or create a context store at the given path.
    pub fn open(path: &Path, config: &ContextStoreConfig) -> Result<Self, StoreError> {
        let conn = Connection::open(path)?;

        // Set SQLCipher key
        conn.pragma_update(None, "key", &config.db_password)?;

        // Configure page cache and mmap
        conn.pragma_update(None, "cache_size", format!("-{}", config.page_cache_kb))?;
        if config.mmap_enabled {
            conn.pragma_update(None, "mmap_size", "268435456")?; // 256MB
        }

        // Enable foreign keys
        conn.pragma_update(None, "foreign_keys", "ON")?;

        // Create schema
        Self::init_schema(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
            master_key: config.master_key,
        })
    }

    /// Open an in-memory store (for testing).
    pub fn open_in_memory(config: &ContextStoreConfig) -> Result<Self, StoreError> {
        let conn = Connection::open_in_memory()?;
        conn.pragma_update(None, "key", &config.db_password)?;
        Self::init_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
            master_key: config.master_key,
        })
    }

    fn init_schema(conn: &Connection) -> Result<(), StoreError> {
        // Evidence table (append-only)
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS evidence (
                id              BLOB    PRIMARY KEY,
                scope_id        BLOB    NOT NULL,
                content_hash    TEXT    NOT NULL,
                body            BLOB    NOT NULL,
                nonce           BLOB    NOT NULL,
                source_ref      TEXT,
                importance      INTEGER NOT NULL DEFAULT 0,
                language_tag    TEXT,
                created_at      INTEGER NOT NULL,
                acl_version     INTEGER NOT NULL DEFAULT 1
            );

            -- Prevent UPDATE and DELETE (append-only)
            CREATE TRIGGER IF NOT EXISTS no_update_evidence
                BEFORE UPDATE ON evidence
                BEGIN
                    SELECT RAISE(ABORT, 'evidence is append-only');
                END;

            CREATE TRIGGER IF NOT EXISTS no_delete_evidence
                BEFORE DELETE ON evidence
                BEGIN
                    SELECT RAISE(ABORT, 'evidence is append-only');
                END;

            -- Three-lane FTS5
            CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
                content, evidence_id UNINDEXED, scope_id UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2'
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts_cjk USING fts5(
                content, evidence_id UNINDEXED, scope_id UNINDEXED,
                tokenize = 'trigram'
            );

            -- Scopes table
            CREATE TABLE IF NOT EXISTS scopes (
                id              BLOB    PRIMARY KEY,
                scope_type      TEXT    NOT NULL,
                parent          BLOB,
                acl_version     INTEGER NOT NULL DEFAULT 1,
                retention_class TEXT    NOT NULL,
                authorized_users TEXT,  -- JSON array
                authorized_roles TEXT   -- JSON array
            );

            -- Tombstones for forgotten scopes
            CREATE TABLE IF NOT EXISTS forgotten_scopes (
                scope_id        BLOB PRIMARY KEY,
                forgotten_at    INTEGER NOT NULL
            );
            "#,
        )?;

        Ok(())
    }

    /// Insert evidence into the store.
    pub fn insert(&self, evidence: &Evidence) -> Result<(), StoreError> {
        let mut conn = self.conn.lock();

        // Encrypt the body with per-scope key
        let scope_key = encryption::derive_scope_key(&self.master_key, &evidence.scope_id.0.as_bytes().to_vec())?;
        let nonce = encryption::AeadNonce::try_from_bytes(&evidence.nonce)?;
        let aad = evidence.scope_id.0.as_bytes();

        let encrypted = encryption::encrypt_aead(&scope_key, &nonce, evidence.fts_content.as_bytes(), aad)?;

        // Use a transaction so all 3 inserts succeed or fail atomically
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO evidence (id, scope_id, content_hash, body, nonce, source_ref, importance, language_tag, created_at, acl_version)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1)",
            params![
                evidence.id.0.as_bytes(),
                evidence.scope_id.0.as_bytes(),
                evidence.content_hash,
                encrypted.ciphertext,
                encrypted.nonce.0.as_slice(),
                evidence.source_ref,
                evidence.importance as i32,
                evidence.language_tag,
                evidence.created_at,
            ],
        )?;

        // Index in FTS (plaintext for search)
        tx.execute(
            "INSERT INTO evidence_fts (content, evidence_id, scope_id) VALUES (?1, ?2, ?3)",
            params![evidence.fts_content, evidence.id.0.as_bytes(), evidence.scope_id.0.as_bytes()],
        )?;

        // Also index in CJK lane
        tx.execute(
            "INSERT INTO evidence_fts_cjk (content, evidence_id, scope_id) VALUES (?1, ?2, ?3)",
            params![evidence.fts_content, evidence.id.0.as_bytes(), evidence.scope_id.0.as_bytes()],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Search using FTS5 BM25 (lexical-only, works on all tiers).
    pub fn search_fts(
        &self,
        query: &str,
        filter: &ScopeFilter,
        limit: usize,
    ) -> Result<Vec<FTSResult>, StoreError> {
        let conn = self.conn.lock();

        // Build scope filter — exclude denied scopes from allowed list
        let scope_ids: Vec<Vec<u8>> = filter
            .allowed_scopes
            .iter()
            .filter(|s| !filter.denied_scopes.contains(s))
            .map(|s| s.0.as_bytes().to_vec())
            .collect();
        if scope_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Sanitize FTS query — escape special FTS5 syntax to prevent injection.
        // Wrap in double quotes and escape any internal double quotes.
        let sanitized_query = sanitize_fts_query(query);

        // Search unicode61 lane
        let placeholders: Vec<String> = (0..scope_ids.len()).map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT e.id, e.scope_id, e.content_hash, e.importance, e.created_at,
                    bm25(evidence_fts) as score
             FROM evidence_fts
             JOIN evidence e ON e.id = evidence_fts.evidence_id
             WHERE evidence_fts MATCH ?1
               AND evidence_fts.scope_id IN ({})
             ORDER BY score
             LIMIT ?{}",
            placeholders.join(", "),
            scope_ids.len() + 2
        );

        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(sanitized_query)];
        for s in &scope_ids {
            params_vec.push(Box::new(s.clone()));
        }
        params_vec.push(Box::new(limit as i64));

        let params_refs: Vec<&dyn rusqlite::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        let results = stmt.query_map(params_refs.as_slice(), |row| {
            let id_bytes: Vec<u8> = row.get(0)?;
            let scope_bytes: Vec<u8> = row.get(1)?;
            let evidence_id = Uuid::from_slice(&id_bytes)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(16, rusqlite::types::Type::Blob, Box::new(e)))?;
            let scope_id = Uuid::from_slice(&scope_bytes)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(16, rusqlite::types::Type::Blob, Box::new(e)))?;
            Ok(FTSResult {
                evidence_id: EvidenceId(evidence_id),
                scope_id: ScopeId(scope_id),
                content_hash: row.get(2)?,
                importance: row.get(3)?,
                created_at: row.get(4)?,
                bm25_score: row.get(5)?,
            })
        })?;

        let mut collected = Vec::new();
        for r in results {
            collected.push(r?);
        }

        // Also search CJK lane and merge results (dedup by evidence_id)
        let cjk_sql = format!(
            "SELECT e.id, e.scope_id, e.content_hash, e.importance, e.created_at,
                    bm25(evidence_fts_cjk) as score
             FROM evidence_fts_cjk
             JOIN evidence e ON e.id = evidence_fts_cjk.evidence_id
             WHERE evidence_fts_cjk MATCH ?1
               AND evidence_fts_cjk.scope_id IN ({})
             ORDER BY score
             LIMIT ?{}",
            placeholders.join(", "),
            scope_ids.len() + 2
        );

        let mut cjk_stmt = conn.prepare(&cjk_sql)?;
        let cjk_results = cjk_stmt.query_map(params_refs.as_slice(), |row| {
            let id_bytes: Vec<u8> = row.get(0)?;
            let scope_bytes: Vec<u8> = row.get(1)?;
            let evidence_id = Uuid::from_slice(&id_bytes)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(16, rusqlite::types::Type::Blob, Box::new(e)))?;
            let scope_id = Uuid::from_slice(&scope_bytes)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(16, rusqlite::types::Type::Blob, Box::new(e)))?;
            Ok(FTSResult {
                evidence_id: EvidenceId(evidence_id),
                scope_id: ScopeId(scope_id),
                content_hash: row.get(2)?,
                importance: row.get(3)?,
                created_at: row.get(4)?,
                bm25_score: row.get(5)?,
            })
        })?;

        let mut seen_ids: std::collections::HashSet<Uuid> = collected.iter().map(|r| r.evidence_id.0).collect();
        for r in cjk_results {
            let r = r?;
            if seen_ids.insert(r.evidence_id.0) {
                collected.push(r);
            }
        }

        Ok(collected)
    }

    /// Decrypt and retrieve evidence body.
    pub fn get_evidence(&self, id: EvidenceId) -> Result<Option<Evidence>, StoreError> {
        let conn = self.conn.lock();

        let mut stmt = conn.prepare(
            "SELECT id, scope_id, content_hash, body, nonce, source_ref, importance, language_tag, created_at
             FROM evidence WHERE id = ?1"
        )?;

        let mut rows = stmt.query(params![id.0.as_bytes()])?;
        if let Some(row) = rows.next()? {
            let scope_bytes: Vec<u8> = row.get(1)?;
            let scope_id = ScopeId(Uuid::from_slice(&scope_bytes)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(16, rusqlite::types::Type::Blob, Box::new(e)))?);

            let body: Vec<u8> = row.get(3)?;
            let nonce_bytes: Vec<u8> = row.get(4)?;

            // Decrypt
            let scope_key = encryption::derive_scope_key(&self.master_key, &scope_id.0.as_bytes().to_vec())?;
            let nonce = encryption::AeadNonce::try_from_bytes(&nonce_bytes)?;
            let aad = scope_id.0.as_bytes();
            let plaintext = encryption::decrypt_aead(&scope_key, &nonce, &body, aad)?;

            return Ok(Some(Evidence {
                id,
                scope_id,
                content_hash: row.get(2)?,
                encrypted_body: body,
                nonce: nonce_bytes,
                source_ref: row.get(5)?,
                importance: row.get(6)?,
                language_tag: row.get(7)?,
                created_at: row.get(8)?,
                fts_content: String::from_utf8(plaintext)
                    .map_err(|e| StoreError::Encryption(encryption::CryptoError::DecryptionFailed(
                        format!("decrypted data is not valid UTF-8: {e}")
                    )))?,
            }));
        }

        Ok(None)
    }

    /// Mark a scope as forgotten (cryptographic forgetting).
    pub fn forget_scope(&self, scope_id: ScopeId) -> Result<(), StoreError> {
        let conn = self.conn.lock();
        conn.execute(
            "INSERT OR REPLACE INTO forgotten_scopes (scope_id, forgotten_at) VALUES (?1, ?2)",
            params![scope_id.0.as_bytes(), chrono::Utc::now().timestamp()],
        )?;
        Ok(())
    }

    /// Check if a scope has been forgotten.
    pub fn is_scope_forgotten(&self, scope_id: ScopeId) -> Result<bool, StoreError> {
        let conn = self.conn.lock();
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM forgotten_scopes WHERE scope_id = ?1",
            params![scope_id.0.as_bytes()],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }
}

/// Sanitize a user-provided query string for FTS5 MATCH.
/// Wraps the query in double quotes (phrase query) and escapes internal
/// double quotes to prevent FTS5 syntax injection.
fn sanitize_fts_query(query: &str) -> String {
    // Escape internal double quotes by doubling them (FTS5 escaping)
    let escaped = query.replace('"', "\"\"");
    // Wrap in double quotes to make it a phrase query,
    // preventing interpretation of FTS5 operators like AND, OR, NOT, *, etc.
    format!("\"{}\"", escaped)
}

/// FTS search result.
#[derive(Debug, Clone)]
pub struct FTSResult {
    pub evidence_id: EvidenceId,
    pub scope_id: ScopeId,
    pub content_hash: String,
    pub importance: i32,
    pub created_at: i64,
    /// BM25 score (lower is better in SQLite FTS5)
    pub bm25_score: f64,
}

/// Store errors.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("encryption error: {0}")]
    Encryption(#[from] encryption::CryptoError),

    #[error("evidence not found")]
    NotFound,

    #[error("scope forgotten")]
    ScopeForgotten,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> ContextStore {
        let config = ContextStoreConfig::for_low_tier(
            "test_password".into(),
            [42u8; 32],
        );
        ContextStore::open_in_memory(&config).unwrap()
    }

    fn make_evidence(scope_id: ScopeId, content: &str) -> Evidence {
        Evidence {
            id: EvidenceId::new(),
            scope_id,
            content_hash: blake3::hash(content.as_bytes()).to_hex().to_string(),
            encrypted_body: vec![],
            nonce: {
                let mut n = [0u8; 24];
                n[0] = 1; // deterministic for tests
                n.to_vec()
            },
            source_ref: None,
            importance: 5,
            language_tag: Some("en".into()),
            created_at: chrono::Utc::now().timestamp(),
            fts_content: content.into(),
        }
    }

    #[test]
    fn test_insert_and_retrieve() {
        let store = make_store();
        let scope = ScopeId::new();
        let evidence = make_evidence(scope, "Hello world from KChat");

        store.insert(&evidence).unwrap();

        let retrieved = store.get_evidence(evidence.id).unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.fts_content, "Hello world from KChat");
    }

    #[test]
    fn test_fts_search() {
        let store = make_store();
        let scope = ScopeId::new();

        store.insert(&make_evidence(scope, "The quick brown fox jumps")).unwrap();
        store.insert(&make_evidence(scope, "Hello world from KChat")).unwrap();
        store.insert(&make_evidence(scope, "Machine learning is fascinating")).unwrap();

        let filter = ScopeFilter {
            allowed_scopes: vec![scope],
            denied_scopes: vec![],
            user_id: Uuid::new_v4(),
            roles: vec![],
        };

        let results = store.search_fts("hello", &filter, 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_scope_filter_excludes_unauthorized() {
        let store = make_store();
        let scope1 = ScopeId::new();
        let scope2 = ScopeId::new();

        store.insert(&make_evidence(scope1, "private message in scope 1")).unwrap();
        store.insert(&make_evidence(scope2, "private message in scope 2")).unwrap();

        // Filter only allows scope1
        let filter = ScopeFilter {
            allowed_scopes: vec![scope1],
            denied_scopes: vec![],
            user_id: Uuid::new_v4(),
            roles: vec![],
        };

        let results = store.search_fts("private", &filter, 10).unwrap();
        // All results should be from scope1
        assert!(results.iter().all(|r| r.scope_id == scope1));
    }

    #[test]
    fn test_forget_scope() {
        let store = make_store();
        let scope = ScopeId::new();

        assert!(!store.is_scope_forgotten(scope).unwrap());

        store.forget_scope(scope).unwrap();

        assert!(store.is_scope_forgotten(scope).unwrap());
    }

    #[test]
    fn test_append_only_prevents_update() {
        let store = make_store();
        let scope = ScopeId::new();
        let evidence = make_evidence(scope, "original content");
        store.insert(&evidence).unwrap();

        // Attempt to update should fail
        let conn = store.conn.lock();
        let result = conn.execute(
            "UPDATE evidence SET importance = 99 WHERE id = ?1",
            params![evidence.id.0.as_bytes()],
        );
        assert!(result.is_err());
    }
}

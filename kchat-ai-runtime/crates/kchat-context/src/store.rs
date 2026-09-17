//! Evidence store — SQLCipher-backed encrypted local storage with FTS5.
//!
//! Schema based on the knowledge repo's evidence_store, adapted for KChat:
//! - Append-only evidence table (UPDATE/DELETE blocked by triggers)
//! - Deduplicated body store (content-hash keyed)
//! - Three-lane FTS5 retrieval (unicode61, trigram, bigram)
//! - Per-scope encryption with XChaCha20-Poly1305

use crate::encryption::{self};
use crate::scope::{ScopeFilter, ScopeId};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;

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

            -- Tombstones for forgotten scopes (must exist before trigger)
            CREATE TABLE IF NOT EXISTS forgotten_scopes (
                scope_id        BLOB PRIMARY KEY,
                forgotten_at    INTEGER NOT NULL
            );

            -- Prevent UPDATE and DELETE (append-only).
            -- DELETE is allowed only when the scope has been marked as forgotten
            -- (cryptographic forgetting) — the trigger checks forgotten_scopes.
            CREATE TRIGGER IF NOT EXISTS no_update_evidence
                BEFORE UPDATE ON evidence
                BEGIN
                    SELECT RAISE(ABORT, 'evidence is append-only');
                END;

            CREATE TRIGGER IF NOT EXISTS no_delete_evidence
                BEFORE DELETE ON evidence
                FOR EACH ROW
                WHEN NOT EXISTS (
                    SELECT 1 FROM forgotten_scopes WHERE scope_id = OLD.scope_id
                )
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

            -- Persisted dense embeddings — one vector per evidence row,
            -- AEAD-encrypted with the per-scope key. Unlike `evidence`,
            -- this is derived cache data, so it is NOT append-only:
            -- vectors may be refreshed when the encoder model changes.
            -- Rows are hard-deleted when a scope is forgotten.
            CREATE TABLE IF NOT EXISTS evidence_vec (
                evidence_id     BLOB    PRIMARY KEY,
                scope_id        BLOB    NOT NULL,
                dim             INTEGER NOT NULL,
                vec             BLOB    NOT NULL,
                vec_nonce       BLOB    NOT NULL,
                model_tag       TEXT    NOT NULL,
                created_at      INTEGER NOT NULL
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
            "#,
        )?;

        Ok(())
    }

    /// Insert evidence into the store.
    pub fn insert(&self, evidence: &Evidence) -> Result<(), StoreError> {
        let mut conn = self.conn.lock();

        // Encrypt the body with per-scope key
        let scope_key = encryption::derive_scope_key(
            &self.master_key,
            evidence.scope_id.0.as_bytes().as_ref(),
        )?;
        let nonce = encryption::AeadNonce::try_from_bytes(&evidence.nonce)?;
        let aad = evidence.scope_id.0.as_bytes();

        let encrypted =
            encryption::encrypt_aead(&scope_key, &nonce, evidence.fts_content.as_bytes(), aad)?;

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
            params![
                evidence.fts_content,
                evidence.id.0.as_bytes(),
                evidence.scope_id.0.as_bytes()
            ],
        )?;

        // Also index in CJK lane
        tx.execute(
            "INSERT INTO evidence_fts_cjk (content, evidence_id, scope_id) VALUES (?1, ?2, ?3)",
            params![
                evidence.fts_content,
                evidence.id.0.as_bytes(),
                evidence.scope_id.0.as_bytes()
            ],
        )?;

        tx.commit()?;
        Ok(())
    }

    /// Insert evidence and index its embedding in one call — the
    /// embed-on-write path. If no embedding provider is available, the
    /// evidence is still inserted; the vector is backfilled lazily by the
    /// retriever on the next query.
    pub fn insert_indexed(
        &self,
        evidence: &Evidence,
        embeddings: &crate::embeddings::EmbeddingManager,
    ) -> Result<(), StoreError> {
        self.insert(evidence)?;
        if let (Some(tag), Ok(vec)) = (
            embeddings.model_tag(),
            embeddings.embed_passage(&evidence.fts_content),
        ) {
            self.insert_vector(evidence.id, evidence.scope_id, &vec, &tag)?;
        }
        Ok(())
    }

    /// Insert a batch of evidence rows, embedding them in one provider call.
    /// Falls back to per-row lazy backfill when no provider is available.
    pub fn insert_batch_indexed(
        &self,
        evidence: &[Evidence],
        embeddings: &crate::embeddings::EmbeddingManager,
    ) -> Result<(), StoreError> {
        for ev in evidence {
            self.insert(ev)?;
        }
        if let Some(tag) = embeddings.model_tag() {
            let texts: Vec<&str> = evidence.iter().map(|e| e.fts_content.as_str()).collect();
            if let Ok(vecs) = embeddings.embed_passages(&texts) {
                for (ev, vec) in evidence.iter().zip(vecs.iter()) {
                    self.insert_vector(ev.id, ev.scope_id, vec, &tag)?;
                }
            }
        }
        Ok(())
    }

    /// Persist a dense embedding for an evidence row.
    ///
    /// The vector is serialized as little-endian f32 and AEAD-encrypted with
    /// the per-scope key — a forgotten scope's vectors become unrecoverable
    /// (and are hard-deleted by `forget_scope`). `model_tag` records which
    /// encoder produced the vector so callers can detect stale vectors after
    /// a model upgrade.
    pub fn insert_vector(
        &self,
        id: EvidenceId,
        scope_id: ScopeId,
        vector: &[f32],
        model_tag: &str,
    ) -> Result<(), StoreError> {
        let conn = self.conn.lock();
        let scope_key =
            encryption::derive_scope_key(&self.master_key, scope_id.0.as_bytes().as_ref())?;
        let nonce = encryption::AeadNonce::random()?;
        let aad = scope_id.0.as_bytes();

        let mut bytes = Vec::with_capacity(vector.len() * 4);
        for v in vector {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let encrypted = encryption::encrypt_aead(&scope_key, &nonce, &bytes, aad)?;

        conn.execute(
            "INSERT OR REPLACE INTO evidence_vec
             (evidence_id, scope_id, dim, vec, vec_nonce, model_tag, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                id.0.as_bytes(),
                scope_id.0.as_bytes(),
                vector.len() as i64,
                encrypted.ciphertext,
                encrypted.nonce.0.as_slice(),
                model_tag,
                chrono::Utc::now().timestamp(),
            ],
        )?;
        Ok(())
    }

    /// Load the persisted vector for an evidence row, if any.
    /// Returns `(vector, model_tag)`.
    pub fn get_vector(
        &self,
        id: EvidenceId,
        scope_id: ScopeId,
    ) -> Result<Option<(Vec<f32>, String)>, StoreError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT dim, vec, vec_nonce, model_tag FROM evidence_vec WHERE evidence_id = ?1",
        )?;
        let row = stmt
            .query_row(params![id.0.as_bytes()], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .optional()?;

        let Some((dim, blob, nonce_bytes, model_tag)) = row else {
            return Ok(None);
        };

        let scope_key =
            encryption::derive_scope_key(&self.master_key, scope_id.0.as_bytes().as_ref())?;
        let nonce = encryption::AeadNonce::try_from_bytes(&nonce_bytes)?;
        let aad = scope_id.0.as_bytes();
        let plain = encryption::decrypt_aead(&scope_key, &nonce, &blob, aad)?;

        let dim = dim as usize;
        if plain.len() != dim * 4 {
            return Err(StoreError::Encryption(
                encryption::CryptoError::DecryptionFailed("vector blob length mismatch".into()),
            ));
        }
        let mut vec = Vec::with_capacity(dim);
        for chunk in plain.chunks_exact(4) {
            vec.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Some((vec, model_tag)))
    }

    /// Load all persisted vectors for the allowed scopes that were produced
    /// by `model_tag`. Rows written by a different embedding model are
    /// treated as stale and invisible — the caller re-embeds and upserts.
    ///
    /// Returns `(evidence_id, vector, created_at)` rows. Rows whose scope is
    /// forgotten are hard-deleted, so this never returns them.
    pub fn vectors_in_scopes(
        &self,
        filter: &ScopeFilter,
        model_tag: &str,
        limit: usize,
    ) -> Result<Vec<(EvidenceId, Vec<f32>, i64)>, StoreError> {
        let conn = self.conn.lock();

        let scope_ids: Vec<Vec<u8>> = filter
            .allowed_scopes
            .iter()
            .filter(|s| !filter.denied_scopes.contains(s))
            .map(|s| s.0.as_bytes().to_vec())
            .collect();
        if scope_ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders: Vec<String> = (0..scope_ids.len()).map(|_| "?".to_string()).collect();
        let sql = format!(
            "SELECT evidence_id, scope_id, dim, vec, vec_nonce, created_at
             FROM evidence_vec
             WHERE scope_id IN ({})
               AND model_tag = ?{}
             LIMIT ?{}",
            placeholders.join(", "),
            scope_ids.len() + 1,
            scope_ids.len() + 2
        );

        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        for s in &scope_ids {
            params_vec.push(Box::new(s.clone()));
        }
        params_vec.push(Box::new(model_tag.to_string()));
        params_vec.push(Box::new(limit as i64));
        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        // Collect rows, then decrypt after releasing the statement (each
        // scope key derivation is cheap; decryption errors on a single row
        // are skipped rather than failing the whole scan).
        let mut stmt = conn.prepare(&sql)?;
        /// Raw row: (evidence_id, scope_id, dim, ciphertext, nonce, created_at)
        type RawVecRow = (Vec<u8>, Vec<u8>, i64, Vec<u8>, Vec<u8>, i64);
        let rows: Vec<RawVecRow> = stmt
            .query_map(params_refs.as_slice(), |row| {
                Ok((
                    row.get::<_, Vec<u8>>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Vec<u8>>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            })?
            .collect::<Result<_, _>>()?;
        drop(stmt);
        drop(conn);

        let mut out = Vec::with_capacity(rows.len());
        for (id_bytes, scope_bytes, dim, blob, nonce_bytes, created_at) in rows {
            let Ok(eid) = Uuid::from_slice(&id_bytes) else {
                continue;
            };
            let Ok(sid) = Uuid::from_slice(&scope_bytes) else {
                continue;
            };
            let Ok(scope_key) = encryption::derive_scope_key(&self.master_key, &sid.as_bytes()[..])
            else {
                continue;
            };
            let Ok(nonce) = encryption::AeadNonce::try_from_bytes(&nonce_bytes) else {
                continue;
            };
            let Ok(plain) = encryption::decrypt_aead(&scope_key, &nonce, &blob, sid.as_bytes())
            else {
                continue;
            };
            let dim = dim as usize;
            if plain.len() != dim * 4 {
                continue;
            }
            let mut vec = Vec::with_capacity(dim);
            for chunk in plain.chunks_exact(4) {
                vec.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            out.push((EvidenceId(eid), vec, created_at));
        }
        Ok(out)
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

        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        let results = stmt.query_map(params_refs.as_slice(), |row| {
            let id_bytes: Vec<u8> = row.get(0)?;
            let scope_bytes: Vec<u8> = row.get(1)?;
            let evidence_id = Uuid::from_slice(&id_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    16,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?;
            let scope_id = Uuid::from_slice(&scope_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    16,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?;
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
            let evidence_id = Uuid::from_slice(&id_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    16,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?;
            let scope_id = Uuid::from_slice(&scope_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    16,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?;
            Ok(FTSResult {
                evidence_id: EvidenceId(evidence_id),
                scope_id: ScopeId(scope_id),
                content_hash: row.get(2)?,
                importance: row.get(3)?,
                created_at: row.get(4)?,
                bm25_score: row.get(5)?,
            })
        })?;

        let mut seen_ids: std::collections::HashSet<Uuid> =
            collected.iter().map(|r| r.evidence_id.0).collect();
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
            let scope_id = ScopeId(Uuid::from_slice(&scope_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    16,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?);

            let body: Vec<u8> = row.get(3)?;
            let nonce_bytes: Vec<u8> = row.get(4)?;

            // Decrypt
            let scope_key =
                encryption::derive_scope_key(&self.master_key, scope_id.0.as_bytes().as_ref())?;
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
                fts_content: String::from_utf8(plaintext).map_err(|e| {
                    StoreError::Encryption(encryption::CryptoError::DecryptionFailed(format!(
                        "decrypted data is not valid UTF-8: {e}"
                    )))
                })?,
            }));
        }

        Ok(None)
    }

    /// Mark a scope as forgotten (cryptographic forgetting).
    /// Inserts the tombstone FIRST (so the append-only trigger allows deletion),
    /// then deletes all evidence and FTS entries for the scope in a single transaction.
    /// This ensures the data is actually unrecoverable, not just hidden.
    pub fn forget_scope(&self, scope_id: ScopeId) -> Result<(), StoreError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction()?;

        // Insert tombstone FIRST — the no_delete_evidence trigger checks this table
        tx.execute(
            "INSERT OR REPLACE INTO forgotten_scopes (scope_id, forgotten_at) VALUES (?1, ?2)",
            params![scope_id.0.as_bytes(), chrono::Utc::now().timestamp()],
        )?;

        // Delete FTS index entries for this scope (both tokenizers)
        tx.execute(
            "DELETE FROM evidence_fts WHERE scope_id = ?1",
            params![scope_id.0.as_bytes()],
        )?;
        tx.execute(
            "DELETE FROM evidence_fts_cjk WHERE scope_id = ?1",
            params![scope_id.0.as_bytes()],
        )?;

        // Delete persisted vectors (allowed because tombstone now exists)
        tx.execute(
            "DELETE FROM evidence_vec WHERE scope_id = ?1",
            params![scope_id.0.as_bytes()],
        )?;

        // Delete encrypted evidence data (allowed because tombstone now exists)
        tx.execute(
            "DELETE FROM evidence WHERE scope_id = ?1",
            params![scope_id.0.as_bytes()],
        )?;

        tx.commit()?;
        tracing::info!(
            "Scope {} forgotten — evidence and FTS entries deleted",
            scope_id.0
        );
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

    /// List all evidence IDs and their FTS content in the given scopes.
    ///
    /// This is used by the dense vector search path in the retriever to
    /// scan all documents when FTS/BM25 returns insufficient results
    /// (e.g., cross-language queries with no keyword overlap).
    pub fn list_evidence_in_scopes(
        &self,
        filter: &ScopeFilter,
        limit: usize,
    ) -> Result<Vec<(EvidenceId, ScopeId, String, i64)>, StoreError> {
        let conn = self.conn.lock();

        let scope_ids: Vec<Vec<u8>> = filter
            .allowed_scopes
            .iter()
            .filter(|s| !filter.denied_scopes.contains(s))
            .map(|s| s.0.as_bytes().to_vec())
            .collect();
        if scope_ids.is_empty() {
            return Ok(Vec::new());
        }

        let placeholders: Vec<String> = (0..scope_ids.len()).map(|_| "?".to_string()).collect();
        // Query the evidence table for IDs, scopes, and created_at.
        // We don't get the plaintext content here (it's encrypted in the body
        // column); the caller can use get_evidence() to decrypt individual
        // documents as needed. This avoids slow FTS5 virtual table scans.
        let sql = format!(
            "SELECT id, scope_id, created_at
             FROM evidence
             WHERE scope_id IN ({})
             LIMIT ?{}",
            placeholders.join(", "),
            scope_ids.len() + 1
        );

        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        for s in &scope_ids {
            params_vec.push(Box::new(s.clone()));
        }
        params_vec.push(Box::new(limit as i64));

        let params_refs: Vec<&dyn rusqlite::ToSql> =
            params_vec.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn.prepare(&sql)?;
        let results = stmt.query_map(params_refs.as_slice(), |row| {
            let id_bytes: Vec<u8> = row.get(0)?;
            let scope_bytes: Vec<u8> = row.get(1)?;
            let evidence_id = Uuid::from_slice(&id_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    16,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?;
            let scope_id = ScopeId(Uuid::from_slice(&scope_bytes).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    16,
                    rusqlite::types::Type::Blob,
                    Box::new(e),
                )
            })?);
            let created_at: i64 = row.get(2)?;
            Ok((EvidenceId(evidence_id), scope_id, String::new(), created_at))
        })?;

        // Collect all results first (the conn lock is still held).
        // We check forgotten scopes after releasing the lock to avoid
        // a deadlock (is_scope_forgotten also locks self.conn).
        let mut all_results = Vec::new();
        for r in results {
            all_results.push(r?);
        }
        drop(stmt);
        drop(conn);

        // Now filter out forgotten scopes (lock is released, safe to re-lock)
        let mut collected = Vec::new();
        for r in all_results {
            if !self.is_scope_forgotten(r.1).unwrap_or(false) {
                collected.push(r);
            }
        }
        Ok(collected)
    }
}

/// Sanitize a user-provided query string for FTS5 MATCH.
/// Wraps the query in double quotes (phrase query) and escapes internal
/// double quotes to prevent FTS5 syntax injection.
fn sanitize_fts_query(query: &str) -> String {
    // Split the query into individual terms and sanitize each one.
    // This allows FTS5 to match documents containing any of the terms
    // (using OR) rather than requiring an exact phrase match.
    let terms: Vec<&str> = query.split_whitespace().filter(|t| !t.is_empty()).collect();
    if terms.is_empty() {
        return "\"\"".to_string();
    }
    // Quote each term individually and join with OR for broader recall
    let quoted: Vec<String> = terms
        .iter()
        .map(|t| {
            // Escape internal double quotes by doubling them (FTS5 escaping)
            let escaped = t.replace('"', "\"\"");
            format!("\"{}\"", escaped)
        })
        .collect();
    quoted.join(" OR ")
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
        let config = ContextStoreConfig::for_low_tier("test_password".into(), [42u8; 32]);
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

        store
            .insert(&make_evidence(scope, "The quick brown fox jumps"))
            .unwrap();
        store
            .insert(&make_evidence(scope, "Hello world from KChat"))
            .unwrap();
        store
            .insert(&make_evidence(scope, "Machine learning is fascinating"))
            .unwrap();

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

        store
            .insert(&make_evidence(scope1, "private message in scope 1"))
            .unwrap();
        store
            .insert(&make_evidence(scope2, "private message in scope 2"))
            .unwrap();

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

    #[test]
    fn test_forget_scope_deletes_evidence() {
        let store = make_store();
        let scope = ScopeId::new();
        let evidence = make_evidence(scope, "sensitive content");
        store.insert(&evidence).unwrap();

        // Verify evidence exists
        assert!(store.get_evidence(evidence.id).unwrap().is_some());

        // Forget the scope — should delete the evidence
        store.forget_scope(scope).unwrap();

        // Evidence should be gone
        assert!(store.get_evidence(evidence.id).unwrap().is_none());
        assert!(store.is_scope_forgotten(scope).unwrap());
    }

    #[test]
    fn test_append_only_still_blocks_delete_without_tombstone() {
        let store = make_store();
        let scope = ScopeId::new();
        let evidence = make_evidence(scope, "content");
        store.insert(&evidence).unwrap();

        // Direct DELETE should fail (no tombstone)
        let conn = store.conn.lock();
        let result = conn.execute(
            "DELETE FROM evidence WHERE id = ?1",
            params![evidence.id.0.as_bytes()],
        );
        assert!(result.is_err(), "DELETE should be blocked by trigger");
    }

    #[test]
    fn test_vector_persist_roundtrip() {
        let store = make_store();
        let scope = ScopeId::new();
        let evidence = make_evidence(scope, "vector test content");
        store.insert(&evidence).unwrap();

        let vec = vec![0.5f32, -1.25, 3.75, 42.0];
        store
            .insert_vector(evidence.id, scope, &vec, "test-model-v1")
            .unwrap();

        let (loaded, tag) = store.get_vector(evidence.id, scope).unwrap().unwrap();
        assert_eq!(loaded, vec);
        assert_eq!(tag, "test-model-v1");
    }

    #[test]
    fn test_vector_model_tag_staleness() {
        let store = make_store();
        let scope = ScopeId::new();
        let evidence = make_evidence(scope, "content");
        store.insert(&evidence).unwrap();

        let filter = ScopeFilter {
            allowed_scopes: vec![scope],
            denied_scopes: vec![],
            user_id: Uuid::new_v4(),
            roles: vec![],
        };

        store
            .insert_vector(evidence.id, scope, &[1.0, 0.0], "model-v1")
            .unwrap();

        // Current-model query sees it; other-model query does not
        assert_eq!(
            store
                .vectors_in_scopes(&filter, "model-v1", 10)
                .unwrap()
                .len(),
            1
        );
        assert!(store
            .vectors_in_scopes(&filter, "model-v2", 10)
            .unwrap()
            .is_empty());

        // Model upgrade: OR REPLACE rewrites the row under the new tag
        store
            .insert_vector(evidence.id, scope, &[0.0, 1.0], "model-v2")
            .unwrap();
        assert!(store
            .vectors_in_scopes(&filter, "model-v1", 10)
            .unwrap()
            .is_empty());
        let rows = store.vectors_in_scopes(&filter, "model-v2", 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1, vec![0.0, 1.0]);
    }

    #[test]
    fn test_forget_scope_deletes_vectors() {
        let store = make_store();
        let scope = ScopeId::new();
        let evidence = make_evidence(scope, "sensitive");
        store.insert(&evidence).unwrap();
        store
            .insert_vector(evidence.id, scope, &[1.0, 2.0], "m")
            .unwrap();

        store.forget_scope(scope).unwrap();

        assert!(store.get_vector(evidence.id, scope).unwrap().is_none());
        let filter = ScopeFilter {
            allowed_scopes: vec![scope],
            denied_scopes: vec![],
            user_id: Uuid::new_v4(),
            roles: vec![],
        };
        assert!(store
            .vectors_in_scopes(&filter, "m", 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn test_insert_indexed_embeds_on_write() {
        let store = make_store();
        let scope = ScopeId::new();
        let embs = crate::embeddings::EmbeddingManager::new()
            .with_primary(Box::new(crate::embeddings::MockEmbedder::new(8)));

        let evidence = make_evidence(scope, "indexed content");
        store.insert_indexed(&evidence, &embs).unwrap();

        let (vec, tag) = store.get_vector(evidence.id, scope).unwrap().unwrap();
        assert_eq!(vec.len(), 8);
        assert_eq!(tag, "mock-embedder");
    }
}

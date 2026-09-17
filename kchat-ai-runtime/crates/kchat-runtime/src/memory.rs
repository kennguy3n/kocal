//! Rolling conversation memory — summarizes turns and writes them back
//! into the encrypted context store so later retrieval can surface them.
//!
//! Turns accumulate in a pending buffer; when it exceeds
//! `FLUSH_CHAR_THRESHOLD`, the buffer is persisted via `insert_indexed` so
//! the memory is searchable immediately. When a generation backend is
//! attached, [`Self::flush_summarized`] additionally writes an LLM summary
//! (additive — evidence is append-only, so raw rows remain).

use kchat_context::embeddings::EmbeddingManager;
use kchat_context::scope::ScopeId;
use kchat_context::store::{ContextStore, Evidence, EvidenceId};
use kchat_generation::backend::BackendAdapter;
use kchat_generation::stream::StreamHandle;
use parking_lot::Mutex;
use std::sync::Arc;

/// Flush when the pending buffer exceeds this many chars (~1-2k tokens).
const FLUSH_CHAR_THRESHOLD: usize = 4000;

/// Hard cap on pending buffer size — a failed/absent backend must not let
/// the buffer grow unboundedly; oldest turns are dropped first.
const PENDING_CAP: usize = 16384;

/// A recorded conversation turn (truncated per-field to bound memory).
#[derive(Debug, Clone)]
struct Turn {
    user: String,
    assistant: String,
}

/// Rolling conversation memory — buffers turns, persists them into the
/// context store, optionally summarizes via the generation backend.
pub struct ConversationMemory {
    pending: Mutex<Vec<Turn>>,
    store: Option<Arc<ContextStore>>,
    embeddings: Option<Arc<EmbeddingManager>>,
    /// Scope that memory evidence is written under.
    scope: Option<ScopeId>,
}

impl ConversationMemory {
    /// Inert memory — `record_turn` buffers but nothing persists.
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(Vec::new()),
            store: None,
            embeddings: None,
            scope: None,
        }
    }

    /// Attach persistence: encrypted store + embedding manager + scope.
    pub fn with_store(
        mut self,
        store: Arc<ContextStore>,
        embeddings: Option<Arc<EmbeddingManager>>,
        scope: ScopeId,
    ) -> Self {
        self.store = Some(store);
        self.embeddings = embeddings;
        self.scope = Some(scope);
        self
    }

    /// The active memory scope, if configured.
    pub fn scope(&self) -> Option<ScopeId> {
        self.scope
    }

    /// Number of pending (unflushed) turns.
    pub fn pending_turns(&self) -> usize {
        self.pending.lock().len()
    }

    /// Take another memory's pending turns into this one — used when the
    /// orchestrator is rebuilt (e.g. a component is attached) so buffered
    /// turns are not silently dropped.
    pub fn adopt_from(&self, other: &ConversationMemory) {
        let mut incoming = std::mem::take(&mut *other.pending.lock());
        if incoming.is_empty() {
            return;
        }
        let mut pending = self.pending.lock();
        incoming.append(&mut pending);
        *pending = incoming;
    }

    /// Record a conversation turn. When the pending buffer crosses the
    /// flush threshold, it is persisted immediately (verbatim — no model
    /// required, so nothing is ever lost). Returns `true` on flush.
    pub fn record_turn(&self, user: &str, assistant: &str) -> bool {
        {
            let mut pending = self.pending.lock();
            pending.push(Turn {
                user: user.chars().take(2000).collect(),
                assistant: assistant.chars().take(4000).collect(),
            });
            let mut len: usize = pending
                .iter()
                .map(|t| t.user.len() + t.assistant.len())
                .sum();
            while len > PENDING_CAP && !pending.is_empty() {
                let dropped = pending.remove(0);
                len -= dropped.user.len() + dropped.assistant.len();
            }
            if len < FLUSH_CHAR_THRESHOLD {
                return false;
            }
        }
        self.flush_pending();
        true
    }

    /// Persist all pending turns verbatim as one evidence row and clear
    /// the buffer. Lossless — requires no model.
    pub fn flush_pending(&self) {
        let (Some(store), Some(scope)) = (&self.store, self.scope) else {
            self.pending.lock().clear();
            return;
        };
        let turns = std::mem::take(&mut *self.pending.lock());
        if turns.is_empty() {
            return;
        }
        let mut content = String::from("Conversation log:\n");
        for t in &turns {
            content.push_str(&format!("User: {}\nAssistant: {}\n", t.user, t.assistant));
        }
        let evidence = make_memory_evidence(scope, &content, "memory.writeback", 4);
        let res = match &self.embeddings {
            Some(e) => store.insert_indexed(&evidence, e),
            None => store.insert(&evidence),
        };
        if let Err(err) = res {
            tracing::warn!("memory write-back failed: {}", err);
            // Re-queue the turns ahead of anything recorded meanwhile so a
            // transient store error does not lose conversation history.
            let mut pending = self.pending.lock();
            let mut restored = turns;
            restored.append(&mut pending);
            *pending = restored;
        }
    }

    /// Generate an LLM summary of the pending turns and persist it as an
    /// additional evidence row (additive — raw rows remain). The pending
    /// buffer is NOT cleared; call [`Self::flush_pending`] to persist+clear.
    /// Returns `true` when a summary evidence row was written.
    pub fn flush_summarized(&self, backend: &Arc<dyn BackendAdapter>) -> bool {
        let (Some(store), Some(scope)) = (&self.store, self.scope) else {
            return false;
        };
        let transcript = {
            let pending = self.pending.lock();
            if pending.is_empty() {
                return false;
            }
            pending
                .iter()
                .map(|t| format!("User: {}\nAssistant: {}\n", t.user, t.assistant))
                .collect::<String>()
        };

        let prompt = format!(
            "<|im_start|>system\nSummarize this conversation in 2-4 bullet points capturing decisions, tasks, and facts worth remembering. Output only the bullets.\n<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
            transcript.chars().take(8000).collect::<String>()
        );
        let config = kchat_generation::backend::GenerationConfig {
            max_tokens: 192,
            temperature: 0.3,
            ..Default::default()
        };
        let handle = StreamHandle::new();
        let Ok(result) = backend.generate_stream(&prompt, &config, &handle) else {
            return false;
        };
        if result.text.trim().is_empty() {
            return false;
        }

        let content = format!("Conversation summary:\n{}", result.text.trim());
        let evidence = make_memory_evidence(scope, &content, "memory.summary", 6);
        match &self.embeddings {
            Some(e) => store.insert_indexed(&evidence, e).is_ok(),
            None => store.insert(&evidence).is_ok(),
        }
    }
}

impl Default for ConversationMemory {
    fn default() -> Self {
        Self::new()
    }
}

fn make_memory_evidence(
    scope: ScopeId,
    content: &str,
    source_ref: &str,
    importance: u8,
) -> Evidence {
    Evidence {
        id: EvidenceId::new(),
        scope_id: scope,
        content_hash: blake3::hash(content.as_bytes()).to_hex().to_string(),
        encrypted_body: vec![],
        nonce: vec![0u8; 24],
        source_ref: Some(source_ref.into()),
        importance,
        language_tag: None,
        created_at: chrono::Utc::now().timestamp(),
        fts_content: content.into(),
    }
}

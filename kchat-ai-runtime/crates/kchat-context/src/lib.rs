//! kchat-context: Private context plane — encrypted local knowledge store,
//! FTS/BM25 retrieval, embeddings, connectors, and provenance.
//!
//! Local chat and artifacts are indexed under explicit scopes: user, account,
//! workspace, conversation, participant, source, record ACL, retention class,
//! and time. Retrieval checks authorization before search, filters candidates
//! during search, and checks again when constructing the prompt.
//!
//! Retrieval tiers:
//! - Low: FTS/BM25, field filters, recency, deterministic entity extraction
//! - Medium: add multilingual dense embeddings and hybrid fusion
//! - High: add a reranker for top candidates and larger citation budgets

pub mod encryption;
pub mod retrieval;
pub mod scope;
pub mod store;
pub mod provenance;

pub use encryption::{AeadKey, AeadNonce, decrypt_aead, encrypt_aead};
pub use retrieval::{RetrievalResult, Retriever, RetrievalTier, HybridWeights};
pub use scope::{Scope, ScopeId, ScopeFilter};
pub use store::{ContextStore, ContextStoreConfig, Evidence, EvidenceId};
pub use provenance::{ProvenanceBundle, ProvenanceAgent, AgentKind, Citation, CitationLocation};

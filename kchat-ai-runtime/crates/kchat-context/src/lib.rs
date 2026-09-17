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

pub mod embeddings;
pub mod encryption;
pub mod provenance;
pub mod reranker;
pub mod retrieval;
pub mod scope;
pub mod store;

#[cfg(feature = "gguf-embeddings")]
pub use embeddings::GgufEmbedder;
#[cfg(feature = "embeddings")]
pub use embeddings::LlamaServerEmbedder;
#[cfg(feature = "embeddings")]
pub use embeddings::OnnxEmbedder;
pub use embeddings::{
    cosine_similarity, EmbeddingError, EmbeddingManager, EmbeddingProvider, MockEmbedder,
};
pub use encryption::{decrypt_aead, encrypt_aead, AeadKey, AeadNonce};
pub use provenance::{AgentKind, Citation, CitationLocation, ProvenanceAgent, ProvenanceBundle};
#[cfg(feature = "reranker")]
pub use reranker::CrossEncoderReranker;
#[cfg(feature = "gguf-embeddings")]
pub use reranker::GgufReranker;
pub use reranker::{MockReranker, Reranker, RerankerError};
pub use retrieval::{HybridWeights, RetrievalResult, RetrievalTier, Retriever};
pub use scope::{Scope, ScopeFilter, ScopeId};
pub use store::{ContextStore, ContextStoreConfig, Evidence, EvidenceId};

//! Error types for kchat-core.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("capability probe failed: {0}")]
    CapabilityProbeFailed(String),

    #[error("manifest verification failed: {0}")]
    ManifestVerificationFailed(String),

    #[error("manifest signature invalid: {0}")]
    ManifestSignatureInvalid(String),

    #[error("pack download failed: {0}")]
    PackDownloadFailed(String),

    #[error("pack chunk hash mismatch: expected {expected}, got {actual}")]
    ChunkHashMismatch { expected: String, actual: String },

    #[error("pack not found: {0}")]
    PackNotFound(String),

    #[error("invalid pack id: {0}")]
    InvalidPackId(String),

    #[error("tier downgrade required: {reason}")]
    TierDowngradeRequired { reason: String },

    #[error("memory budget exceeded: requested {requested_mb} MB, safe budget {safe_mb} MB")]
    MemoryBudgetExceeded { requested_mb: u64, safe_mb: u64 },

    #[error("thermal state critical: cannot run generative workload")]
    ThermalCritical,

    #[error("scheduler cancelled: {0}")]
    SchedulerCancelled(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

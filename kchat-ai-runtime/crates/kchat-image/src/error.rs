//! Error types for image search.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ImageError {
    #[error("API key missing for provider '{0}' — set the required environment variable")]
    KeyMissing(&'static str),

    #[error("authentication failed for provider '{0}': {1}")]
    AuthFailed(&'static str, String),

    #[error("rate limited by provider '{0}' (retry after {1}s)")]
    RateLimited(&'static str, u64),

    #[error("network error calling provider '{0}': {1}")]
    Network(&'static str, String),

    #[error("response parse error from provider '{0}': {1}")]
    Parse(&'static str, String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("provider '{0}' returned HTTP {1}: {2}")]
    HttpStatus(&'static str, u16, String),

    #[error("no providers configured")]
    NoProviders,

    #[error("all providers failed ({0} errors)")]
    AllProvidersFailed(usize),
}

impl From<reqwest::Error> for ImageError {
    fn from(e: reqwest::Error) -> Self {
        ImageError::Network("unknown", e.to_string())
    }
}

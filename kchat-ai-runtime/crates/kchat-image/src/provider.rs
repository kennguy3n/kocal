//! Provider trait — all image search backends implement this.

use crate::error::ImageError;
use crate::types::{ApiKey, ImageSearchRequest, ImageSearchResponse, RateLimit};
use async_trait::async_trait;
use std::future::Future;
use std::time::Duration;

/// A single image search provider (Pexels, Pixabay, Unsplash, Shutterstock).
#[async_trait]
pub trait ImageSearchProvider: Send + Sync {
    /// Provider identifier (e.g. "pexels", "pixabay").
    fn id(&self) -> &'static str;

    /// Human-readable provider name.
    fn name(&self) -> &'static str;

    /// Required environment variable name for the API key.
    fn env_var_name(&self) -> &'static str;

    /// Per-provider rate limit.
    fn rate_limit(&self) -> RateLimit;

    /// Execute a search against this provider.
    /// The key is provided by the registry; the provider should return
    /// `ImageError::KeyMissing` if the key is empty.
    async fn search(
        &self,
        req: &ImageSearchRequest,
        key: &ApiKey,
    ) -> Result<ImageSearchResponse, ImageError>;
}

/// Maximum number of retry attempts for transient failures.
const MAX_RETRIES: u32 = 2;

/// Initial backoff delay before the first retry.
const INITIAL_BACKOFF: Duration = Duration::from_millis(500);

/// Check whether an error is transient and worth retrying.
fn is_transient(err: &ImageError) -> bool {
    match err {
        ImageError::RateLimited(_, _) => true,
        ImageError::Network(_, _) => true,
        ImageError::HttpStatus(_, code, _) => {
            // Retry on server errors and 429.
            *code == 429 || *code >= 500
        }
        _ => false,
    }
}

/// Retry a fallible async operation with exponential backoff.
/// Retries up to `MAX_RETRIES` times for transient errors (429, 5xx, network).
/// Non-transient errors (auth, parse, key missing) are returned immediately.
pub async fn retry_with_backoff<F, Fut, T>(
    provider_id: &'static str,
    mut operation: F,
) -> Result<T, ImageError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ImageError>>,
{
    let mut last_err = None;
    let mut delay = INITIAL_BACKOFF;
    for attempt in 0..=MAX_RETRIES {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                if !is_transient(&e) || attempt == MAX_RETRIES {
                    return Err(e);
                }
                tracing::warn!(
                    provider = provider_id,
                    attempt = attempt + 1,
                    max_retries = MAX_RETRIES,
                    error = %e,
                    "transient error, retrying after {:?}",
                    delay
                );
                last_err = Some(e);
                tokio::time::sleep(delay).await;
                delay = delay.saturating_mul(2); // exponential backoff
            }
        }
    }
    Err(last_err.unwrap_or_else(|| ImageError::Network(provider_id, "retry loop exhausted".into())))
}

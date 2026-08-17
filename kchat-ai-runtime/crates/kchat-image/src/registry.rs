//! Image search registry — manages providers, API keys, cache, and fallback.
//!
//! The registry tries providers in priority order. If a provider fails
//! (rate limit, network error, missing key), it falls back to the next.
//! Results are deduplicated by URL and re-ranked by orientation match.

use crate::cache::ImageCache;
use crate::error::ImageError;
use crate::provider::ImageSearchProvider;
use crate::safety::filter_results;
use crate::types::{ApiKey, ImageSearchRequest, ImageSearchResponse, RateLimit};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Sliding-window rate limiter — tracks request count within a rolling
/// 1-hour window. Returns false when the provider's per-hour limit is
/// reached, preventing quota exhaustion from burst traffic.
struct RateLimiter {
    per_hour: u32,
    timestamps: Mutex<Vec<Instant>>,
}

impl RateLimiter {
    fn new(per_hour: u32) -> Self {
        Self {
            per_hour,
            timestamps: Mutex::new(Vec::with_capacity(per_hour as usize + 16)),
        }
    }

    /// Attempt to acquire a request slot. Returns true if allowed, false if
    /// the rate limit has been exceeded. Prunes expired timestamps on each call.
    fn try_acquire(&self) -> bool {
        let now = Instant::now();
        let window = Duration::from_secs(3600);
        let mut ts = self.timestamps.lock();
        // Prune timestamps older than 1 hour.
        ts.retain(|&t| now.duration_since(t) < window);
        if ts.len() >= self.per_hour as usize {
            return false;
        }
        ts.push(now);
        true
    }
}

/// A registered provider with its API key and rate limiter.
struct ProviderEntry {
    provider: Arc<dyn ImageSearchProvider>,
    key: ApiKey,
    limiter: RateLimiter,
}

/// The image search registry.
pub struct ImageSearchRegistry {
    providers: Mutex<Vec<ProviderEntry>>,
    cache: ImageCache,
}

impl ImageSearchRegistry {
    /// Create an empty registry with the default cache.
    pub fn new() -> Self {
        Self {
            providers: Mutex::new(Vec::new()),
            cache: ImageCache::default_cache(),
        }
    }

    /// Create a registry pre-loaded with all 4 real providers, reading keys
    /// from environment variables. Providers without keys are skipped.
    pub fn from_env() -> Self {
        let reg = Self::new();
        reg.add_from_env::<crate::providers::PexelsProvider>();
        reg.add_from_env::<crate::providers::PixabayProvider>();
        reg.add_from_env::<crate::providers::UnsplashProvider>();
        reg.add_from_env::<crate::providers::ShutterstockProvider>();
        reg
    }

    /// Add a provider, reading its API key from the environment variable
    /// declared by `provider.env_var_name()`. Skips silently if the env var
    /// is unset or empty.
    pub fn add_from_env<P>(&self)
    where
        P: ImageSearchProvider + Default + 'static,
    {
        let provider = P::default();
        let key_str = std::env::var(provider.env_var_name()).unwrap_or_default();
        if key_str.is_empty() {
            tracing::debug!(
                provider = provider.id(),
                env = provider.env_var_name(),
                "skipping provider — env var not set"
            );
            return;
        }
        let rate_limit = provider.rate_limit();
        self.add_with_limit(Arc::new(provider), ApiKey::new(key_str), rate_limit);
    }

    /// Add a provider with an explicit API key (uses the provider's declared rate limit).
    pub fn add(&self, provider: Arc<dyn ImageSearchProvider>, key: ApiKey) {
        let rate_limit = provider.rate_limit();
        self.add_with_limit(provider, key, rate_limit);
    }

    /// Add a provider with an explicit API key and rate limit.
    pub fn add_with_limit(
        &self,
        provider: Arc<dyn ImageSearchProvider>,
        key: ApiKey,
        rate_limit: RateLimit,
    ) {
        let limiter = RateLimiter::new(rate_limit.per_hour);
        self.providers.lock().push(ProviderEntry { provider, key, limiter });
    }

    /// Number of registered providers.
    pub fn provider_count(&self) -> usize {
        self.providers.lock().len()
    }

    /// List provider IDs.
    pub fn provider_ids(&self) -> Vec<String> {
        self.providers
            .lock()
            .iter()
            .map(|e| e.provider.id().to_string())
            .collect()
    }

    /// Search using all providers, with fallback. Results are merged,
    /// deduplicated by URL, safety-filtered, and re-ranked by orientation match.
    ///
    /// Providers are queried sequentially in priority order. If a provider
    /// fails (rate limit, network error, missing key), the registry falls
    /// back to the next provider. Results from all successful providers are
    /// merged.
    pub async fn search(
        &self,
        req: &ImageSearchRequest,
    ) -> Result<ImageSearchResponse, ImageError> {
        let cache_key = req.cache_key();
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached);
        }

        // Collect provider entries, checking rate limits under the lock.
        // Providers that are rate-limited are skipped (treated as fallback failure).
        let entries: Vec<(Arc<dyn ImageSearchProvider>, ApiKey)> = {
            let providers = self.providers.lock();
            if providers.is_empty() {
                return Err(ImageError::NoProviders);
            }
            providers
                .iter()
                .filter(|e| {
                    if !e.limiter.try_acquire() {
                        tracing::warn!(
                            provider = e.provider.id(),
                            "rate limit exceeded, skipping provider"
                        );
                        false
                    } else {
                        true
                    }
                })
                .map(|e| (e.provider.clone(), e.key.clone()))
                .collect()
        };

        if entries.is_empty() {
            return Err(ImageError::RateLimited("all", 3600));
        }

        let mut all_results: Vec<ImageResult> = Vec::new();
        let mut errors: Vec<ImageError> = Vec::new();
        let mut total: u64 = 0;
        let mut last_provider: String = String::new();

        for (provider, key) in &entries {
            match provider.search(req, key).await {
                Ok(resp) => {
                    total = total.max(resp.total);
                    last_provider = resp.provider.clone();
                    all_results.extend(resp.results);
                }
                Err(e) => {
                    tracing::warn!(
                        provider = provider.id(),
                        error = %e,
                        "provider failed, falling back"
                    );
                    errors.push(e);
                }
            }
        }

        if all_results.is_empty() && !errors.is_empty() {
            return Err(ImageError::AllProvidersFailed(errors.len()));
        }

        // Deduplicate by URL.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        all_results.retain(|r| seen.insert(r.url.clone()));

        // Safety filter.
        let (mut safe, _dropped) = filter_results(all_results);

        // Re-rank: orientation match first, then keep provider order.
        if let Some(o) = req.orientation {
            safe.sort_by_key(|r| r.orientation != o);
        }

        let response = ImageSearchResponse {
            results: safe,
            total,
            page: req.page,
            per_page: req.per_page,
            provider: if last_provider.is_empty() {
                "registry".into()
            } else {
                last_provider
            },
        };

        self.cache.put(cache_key, response.clone());
        Ok(response)
    }

    /// Search a single provider by ID.
    pub async fn search_provider(
        &self,
        provider_id: &str,
        req: &ImageSearchRequest,
    ) -> Result<ImageSearchResponse, ImageError> {
        let entry = {
            let providers = self.providers.lock();
            let e = providers
                .iter()
                .find(|e| e.provider.id() == provider_id);
            if let Some(e) = e {
                if !e.limiter.try_acquire() {
                    return Err(ImageError::RateLimited(e.provider.id(), 3600));
                }
                Some((e.provider.clone(), e.key.clone()))
            } else {
                None
            }
        };
        match entry {
            Some((provider, key)) => provider.search(req, &key).await,
            None => Err(ImageError::InvalidRequest(format!(
                "provider '{}' not registered",
                provider_id
            ))),
        }
    }

    /// Clear the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl Default for ImageSearchRegistry {
    fn default() -> Self {
        Self::new()
    }
}

use crate::types::ImageResult;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockProvider;
    use crate::types::{ImageOrientation, ImageResult, License, Attribution};

    fn fixture(id: &str, alt: &str, o: ImageOrientation) -> ImageResult {
        ImageResult {
            id: id.into(),
            provider: "mock".into(),
            url: format!("https://mock.example.com/{}.jpg", id),
            thumb_url: format!("https://mock.example.com/thumb_{}.jpg", id),
            width: 800,
            height: 800,
            orientation: o,
            alt_text: alt.into(),
            attribution: Attribution {
                photographer: "Mock".into(),
                photographer_url: "https://mock.example.com/p".into(),
                source_url: format!("https://mock.example.com/{}", id),
            },
            license: License::FreeNoAttribution,
            color: None,
        }
    }

    #[tokio::test]
    async fn test_registry_search_with_mock() {
        let reg = ImageSearchRegistry::new();
        let mut mock = MockProvider::new();
        mock.add_fixture(
            "forest",
            vec![
                fixture("1", "pine forest", ImageOrientation::Landscape),
                fixture("2", "rainforest", ImageOrientation::Portrait),
            ],
        );
        reg.add(Arc::new(mock), ApiKey::default());
        let req = ImageSearchRequest::new("forest");
        let resp = reg.search(&req).await.unwrap();
        assert_eq!(resp.results.len(), 2);
    }

    #[tokio::test]
    async fn test_registry_dedup_by_url() {
        let reg = ImageSearchRegistry::new();
        let mut m1 = MockProvider::new();
        m1.add_fixture(
            "tree",
            vec![fixture("1", "oak tree", ImageOrientation::Landscape)],
        );
        let mut m2 = MockProvider::new();
        // Same URL as m1's fixture → should be deduped.
        m2.add_fixture(
            "tree",
            vec![fixture("1", "oak tree", ImageOrientation::Landscape)],
        );
        reg.add(Arc::new(m1), ApiKey::default());
        reg.add(Arc::new(m2), ApiKey::default());
        let req = ImageSearchRequest::new("tree");
        let resp = reg.search(&req).await.unwrap();
        assert_eq!(resp.results.len(), 1, "duplicate URL should be deduped");
    }

    #[tokio::test]
    async fn test_registry_orientation_rerank() {
        let reg = ImageSearchRegistry::new();
        let mut mock = MockProvider::new();
        mock.add_fixture(
            "river",
            vec![
                fixture("1", "wide river", ImageOrientation::Portrait),
                fixture("2", "river valley", ImageOrientation::Landscape),
            ],
        );
        reg.add(Arc::new(mock), ApiKey::default());
        let req = ImageSearchRequest::new("river").with_orientation(ImageOrientation::Landscape);
        let resp = reg.search(&req).await.unwrap();
        assert_eq!(resp.results[0].orientation, ImageOrientation::Landscape);
    }

    #[tokio::test]
    async fn test_registry_no_providers() {
        let reg = ImageSearchRegistry::new();
        let req = ImageSearchRequest::new("test");
        let err = reg.search(&req).await.unwrap_err();
        assert!(matches!(err, ImageError::NoProviders));
    }

    #[tokio::test]
    async fn test_registry_cache_hit() {
        let reg = ImageSearchRegistry::new();
        let mut mock = MockProvider::new();
        mock.add_fixture(
            "ocean",
            vec![fixture("1", "blue ocean", ImageOrientation::Landscape)],
        );
        reg.add(Arc::new(mock), ApiKey::default());
        let req = ImageSearchRequest::new("ocean");
        let _ = reg.search(&req).await.unwrap();
        // Second call should hit the cache.
        let cached = reg.search(&req).await.unwrap();
        assert_eq!(cached.results.len(), 1);
    }

    #[tokio::test]
    async fn test_registry_search_provider_by_id() {
        let reg = ImageSearchRegistry::new();
        let mut mock = MockProvider::new();
        mock.add_fixture(
            "lake",
            vec![fixture("1", "calm lake", ImageOrientation::Landscape)],
        );
        reg.add(Arc::new(mock), ApiKey::default());
        let req = ImageSearchRequest::new("lake");
        let resp = reg.search_provider("mock", &req).await.unwrap();
        assert_eq!(resp.results.len(), 1);
    }

    #[tokio::test]
    async fn test_registry_search_provider_unknown_id() {
        let reg = ImageSearchRegistry::new();
        let req = ImageSearchRequest::new("test");
        let err = reg.search_provider("nonexistent", &req).await.unwrap_err();
        assert!(matches!(err, ImageError::InvalidRequest(_)));
    }
}

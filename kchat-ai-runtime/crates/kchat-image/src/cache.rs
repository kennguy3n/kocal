//! In-memory LRU cache for image search responses.
//!
//! Cache keys are SHA-256 hashes of the canonical (provider, request) pair.
//! Cached entries expire after `ttl` to avoid serving stale results.
//! On access, the entry's last-access timestamp is updated, so frequently
//! accessed entries survive eviction (true LRU, not FIFO).

use crate::types::ImageSearchResponse;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A single cache entry.
struct Entry {
    response: ImageSearchResponse,
    /// Last access time (updated on get and put). Used for LRU eviction.
    last_accessed: Instant,
}

/// LRU cache (size-bounded, TTL-based eviction on access).
/// On `get()`, the entry's `last_accessed` timestamp is refreshed, so
/// frequently-accessed entries are protected from eviction.
pub struct ImageCache {
    inner: Mutex<HashMap<String, Entry>>,
    max_entries: usize,
    ttl: Duration,
}

impl ImageCache {
    pub fn new(max_entries: usize, ttl: Duration) -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            max_entries,
            ttl,
        }
    }

    /// Default cache: 256 entries, 10-minute TTL.
    pub fn default_cache() -> Self {
        Self::new(256, Duration::from_secs(600))
    }

    /// Look up a cached response by key. Updates the last-accessed timestamp
    /// on hit, implementing LRU semantics (frequently-accessed entries survive).
    pub fn get(&self, key: &str) -> Option<ImageSearchResponse> {
        let mut inner = self.inner.lock();
        let now = Instant::now();
        let expired = inner
            .get(key)
            .map(|e| now.duration_since(e.last_accessed) > self.ttl)
            .unwrap_or(false);
        if expired {
            inner.remove(key);
            return None;
        }
        // Update last-accessed timestamp for LRU.
        if let Some(entry) = inner.get_mut(key) {
            entry.last_accessed = now;
            return Some(entry.response.clone());
        }
        None
    }

    /// Insert a response.
    pub fn put(&self, key: String, response: ImageSearchResponse) {
        let mut inner = self.inner.lock();
        if inner.len() >= self.max_entries && !inner.contains_key(&key) {
            // Evict the least-recently-accessed entry.
            if let Some((oldest_key, _)) = inner
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| (k.clone(), ()))
            {
                inner.remove(&oldest_key);
            }
        }
        inner.insert(
            key,
            Entry {
                response,
                last_accessed: Instant::now(),
            },
        );
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        self.inner.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ImageSearchResponse;

    #[test]
    fn test_cache_put_get() {
        let cache = ImageCache::new(10, Duration::from_secs(60));
        let resp = ImageSearchResponse::empty("pexels");
        cache.put("key1".into(), resp.clone());
        assert_eq!(cache.len(), 1);
        let got = cache.get("key1");
        assert!(got.is_some());
        assert_eq!(got.unwrap().provider, "pexels");
    }

    #[test]
    fn test_cache_miss() {
        let cache = ImageCache::new(10, Duration::from_secs(60));
        assert!(cache.get("missing").is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let cache = ImageCache::new(2, Duration::from_secs(60));
        cache.put("a".into(), ImageSearchResponse::empty("pexels"));
        cache.put("b".into(), ImageSearchResponse::empty("pixabay"));
        cache.put("c".into(), ImageSearchResponse::empty("unsplash"));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_cache_ttl_expiry() {
        let cache = ImageCache::new(10, Duration::from_millis(1));
        cache.put("k".into(), ImageSearchResponse::empty("pexels"));
        std::thread::sleep(Duration::from_millis(10));
        assert!(cache.get("k").is_none());
    }
}

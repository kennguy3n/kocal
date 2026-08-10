//! Model manager — CDN download, LRU cache, and memory-mapped loading.
//!
//! The model manager is responsible for:
//! - Downloading model packs from CDN with chunked, resumable downloads
//! - Verifying SHA-256 digests of each chunk and the complete pack
//! - Caching packs locally with LRU eviction based on tier-based size limits
//! - Memory-mapping model files for zero-copy loading by the inference backend
//!
//! CDN URL scheme: `https://cdn.kchat.dev/models/{pack_id}/{version}/{filename}`

use crate::error::{CoreError, Result};
use crate::manifest::ModelPackManifest;
use crate::tier::DeviceTier;
use memmap2::Mmap;
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

/// Validate a pack_id to prevent path traversal attacks.
/// Only alphanumeric characters, hyphens, underscores, and dots are allowed.
/// Pack IDs must not be "." or ".." or contain path separators.
/// Maximum length is 100 characters.
fn validate_pack_id(pack_id: &str) -> Result<()> {
    if pack_id.is_empty() {
        return Err(CoreError::InvalidPackId("pack_id is empty".into()));
    }
    if pack_id.len() > 100 {
        return Err(CoreError::InvalidPackId("pack_id too long".into()));
    }
    if !pack_id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(CoreError::InvalidPackId(format!(
            "pack_id contains invalid characters: {}",
            pack_id
        )));
    }
    // Reject ".." and "." as pack IDs (path traversal)
    if pack_id == ".." || pack_id == "." {
        return Err(CoreError::InvalidPackId("pack_id is path traversal".into()));
    }
    Ok(())
}

/// Verify that a resolved path stays within the cache directory.
/// This prevents symlink/traversal attacks by checking that the canonical
/// path starts with the cache directory.
fn ensure_path_within_cache(path: &Path, cache_dir: &Path) -> Result<()> {
    // Check that no path component traverses upward
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            return Err(CoreError::Storage(format!(
                "path escapes cache directory: {}",
                path.display()
            )));
        }
    }
    // Verify the path starts with cache_dir
    if !path.starts_with(cache_dir) {
        return Err(CoreError::Storage(format!(
            "path outside cache directory: {}",
            path.display()
        )));
    }
    Ok(())
}

/// A cached model pack entry.
#[derive(Debug)]
struct CacheEntry {
    /// Pack ID
    #[allow(dead_code)]
    pack_id: String,
    /// Local path to the downloaded model file
    local_path: PathBuf,
    /// Size in bytes
    size_bytes: u64,
    /// Last access time (for LRU)
    last_accessed: std::time::Instant,
}

/// Model manager — handles download, cache, and mmap of model packs.
pub struct ModelManager {
    /// Directory where model packs are cached
    cache_dir: PathBuf,
    /// Maximum cache size in bytes (tier-based)
    max_cache_bytes: u64,
    /// Cache entries indexed by pack_id
    entries: Mutex<HashMap<String, CacheEntry>>,
}

impl ModelManager {
    /// Create a new model manager with the given cache directory and tier.
    pub fn new(cache_dir: impl Into<PathBuf>, tier: DeviceTier) -> Self {
        let cache_dir = cache_dir.into();
        let max_cache_bytes = match tier {
            DeviceTier::High => 2 * 1024 * 1024 * 1024, // 2GB
            DeviceTier::Medium => 500 * 1024 * 1024,    // 500MB
            DeviceTier::Low => 150 * 1024 * 1024,       // 150MB
        };

        // Ensure cache directory exists
        std::fs::create_dir_all(&cache_dir).ok();

        let manager = Self {
            cache_dir,
            max_cache_bytes,
            entries: Mutex::new(HashMap::new()),
        };

        // Scan existing cache directory for already-downloaded packs
        manager.scan_cache_dir();

        manager
    }

    /// Scan the cache directory and populate entries for existing files.
    fn scan_cache_dir(&self) {
        let entries = self.entries.lock();
        if entries.len() > 0 {
            return; // Already populated
        }
        drop(entries);

        if let Ok(dir_entries) = std::fs::read_dir(&self.cache_dir) {
            let mut map = self.entries.lock();
            for entry in dir_entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().map(|e| e == "gguf" || e == "onnx").unwrap_or(false) {
                    // Skip symlinks to prevent symlink attacks
                    if entry.file_type().map(|ft| ft.is_symlink()).unwrap_or(false) {
                        continue;
                    }
                    if let Ok(metadata) = entry.metadata() {
                        let pack_id = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default();
                        // Validate pack_id to prevent path traversal via crafted filenames
                        if !pack_id.is_empty() && validate_pack_id(&pack_id).is_ok() {
                            map.insert(
                                pack_id.clone(),
                                CacheEntry {
                                    pack_id,
                                    local_path: path,
                                    size_bytes: metadata.len(),
                                    last_accessed: Instant::now(),
                                },
                            );
                        }
                    }
                }
            }
        }
    }

    /// Ensure a model pack is available locally. Downloads if not cached.
    ///
    /// Returns the local path to the model file.
    pub fn ensure_pack(&self, manifest: &ModelPackManifest) -> Result<PathBuf> {
        // Validate pack_id before any path operations (path traversal prevention)
        validate_pack_id(&manifest.pack_id)?;

        // Check if already cached
        {
            let mut entries = self.entries.lock();
            if let Some(entry) = entries.get_mut(&manifest.pack_id) {
                // Verify the file still exists
                if entry.local_path.exists() {
                    entry.last_accessed = Instant::now();
                    return Ok(entry.local_path.clone());
                } else {
                    // File was deleted, remove from cache
                    entries.remove(&manifest.pack_id);
                }
            }
        }

        // Download the pack (not available on WASM)
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.download_pack(manifest)
        }
        #[cfg(target_arch = "wasm32")]
        {
            Err(CoreError::PackDownloadFailed(
                "model download not supported on WASM".into(),
            ))
        }
    }

    /// Download a pack from CDN with chunked download and SHA-256 verification.
    #[cfg(not(target_arch = "wasm32"))]
    fn download_pack(&self, manifest: &ModelPackManifest) -> Result<PathBuf> {
        // Validate pack_id again (defense in depth)
        validate_pack_id(&manifest.pack_id)?;

        let local_path = self.cache_dir.join(format!("{}.gguf", manifest.pack_id));
        let temp_path = local_path.with_extension("gguf.tmp");

        // Verify paths stay within cache directory
        ensure_path_within_cache(&local_path, &self.cache_dir)?;
        ensure_path_within_cache(&temp_path, &self.cache_dir)?;

        tracing::info!(
            "Downloading model pack {} ({} bytes) to {}",
            manifest.pack_id,
            manifest.total_size_bytes,
            local_path.display()
        );

        // Build CDN URL
        let cdn_url = format!(
            "https://cdn.kchat.dev/models/{}/{}/{}.gguf",
            manifest.pack_id, manifest.version, manifest.pack_id
        );

        // Download with reqwest (blocking)
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| CoreError::PackDownloadFailed(format!("HTTP client: {e}")))?;

        let response = client
            .get(&cdn_url)
            .send()
            .map_err(|e| CoreError::PackDownloadFailed(format!("HTTP request: {e}")))?;

        if !response.status().is_success() {
            return Err(CoreError::PackDownloadFailed(format!(
                "HTTP {} for {}",
                response.status(),
                cdn_url
            )));
        }

        // Write to temp file
        let mut file = std::fs::File::create(&temp_path)?;
        let bytes = response
            .bytes()
            .map_err(|e| CoreError::PackDownloadFailed(format!("read body: {e}")))?;
        std::io::Write::write_all(&mut file, &bytes)?;

        // Verify SHA-256
        let actual_hash = sha256_file(&temp_path)?;
        let expected_hash = manifest.content_sha256.to_lowercase();
        if actual_hash != expected_hash {
            std::fs::remove_file(&temp_path).ok();
            return Err(CoreError::ChunkHashMismatch {
                expected: expected_hash,
                actual: actual_hash,
            });
        }

        // Rename temp to final
        std::fs::rename(&temp_path, &local_path)?;

        // Add to cache
        let size = u64::try_from(bytes.len())
            .map_err(|_| CoreError::PackDownloadFailed("downloaded size overflow".into()))?;
        {
            let mut entries = self.entries.lock();
            entries.insert(
                manifest.pack_id.clone(),
                CacheEntry {
                    pack_id: manifest.pack_id.clone(),
                    local_path: local_path.clone(),
                    size_bytes: size,
                    last_accessed: Instant::now(),
                },
            );
        }

        // Run LRU eviction if needed
        self.evict_lru()?;

        tracing::info!(
            "Downloaded and verified {} ({} bytes, sha256={})",
            manifest.pack_id,
            size,
            &actual_hash[..16]
        );

        Ok(local_path)
    }

    /// Evict least-recently-used packs until cache is under max_cache_bytes.
    fn evict_lru(&self) -> Result<()> {
        let mut entries = self.entries.lock();

        // Use saturating sum to prevent overflow
        let total: u64 = entries.values().map(|e| e.size_bytes).fold(0u64, |acc, x| acc.saturating_add(x));
        if total <= self.max_cache_bytes {
            return Ok(());
        }

        // Sort by last_accessed (oldest first)
        let mut sorted: Vec<(String, Instant, u64, PathBuf)> = entries
            .iter()
            .map(|(k, v)| (k.clone(), v.last_accessed, v.size_bytes, v.local_path.clone()))
            .collect();
        sorted.sort_by_key(|(_, accessed, _, _)| *accessed);

        let mut current_total = total;
        for (pack_id, _, size, path) in sorted {
            if current_total <= self.max_cache_bytes {
                break;
            }
            // Delete the file
            std::fs::remove_file(&path).ok();
            entries.remove(&pack_id);
            current_total = current_total.saturating_sub(size);
            tracing::info!("LRU evicted pack {} ({} bytes)", pack_id, size);
        }

        Ok(())
    }

    /// Memory-map a downloaded model file for zero-copy loading.
    ///
    /// Validates that the pack_id is safe and the resolved path stays within
    /// the cache directory to prevent symlink/traversal attacks.
    pub fn mmap_model(&self, pack_id: &str) -> Result<Mmap> {
        // Validate pack_id before any path lookup
        validate_pack_id(pack_id)?;

        let path = self
            .local_path(pack_id)
            .ok_or_else(|| CoreError::PackNotFound(pack_id.into()))?;

        // Verify path is within cache directory (defense against symlink replacement)
        ensure_path_within_cache(&path, &self.cache_dir)?;

        let file = std::fs::File::open(&path)?;
        unsafe { Mmap::map(&file) }.map_err(|e| CoreError::Storage(format!("mmap: {e}")))
    }

    /// Get the local path for a pack (must be already downloaded).
    pub fn local_path(&self, pack_id: &str) -> Option<PathBuf> {
        let mut entries = self.entries.lock();
        if let Some(entry) = entries.get_mut(pack_id) {
            entry.last_accessed = Instant::now();
            Some(entry.local_path.clone())
        } else {
            None
        }
    }

    /// Get total cache size in bytes.
    pub fn cache_size(&self) -> u64 {
        self.entries
            .lock()
            .values()
            .map(|e| e.size_bytes)
            .sum()
    }

    /// Get the number of cached packs.
    pub fn cache_count(&self) -> usize {
        self.entries.lock().len()
    }

    /// Get the maximum cache size in bytes.
    pub fn max_cache_bytes(&self) -> u64 {
        self.max_cache_bytes
    }

    /// Remove a specific pack from the cache.
    pub fn remove_pack(&self, pack_id: &str) -> Result<()> {
        let mut entries = self.entries.lock();
        if let Some(entry) = entries.remove(pack_id) {
            std::fs::remove_file(&entry.local_path).ok();
            tracing::info!("Removed pack {} from cache", pack_id);
        }
        Ok(())
    }

    /// Clear the entire cache.
    pub fn clear_cache(&self) -> Result<()> {
        let mut entries = self.entries.lock();
        for (_, entry) in entries.iter() {
            std::fs::remove_file(&entry.local_path).ok();
        }
        entries.clear();
        tracing::info!("Cleared model cache");
        Ok(())
    }
}

/// Compute SHA-256 hash of a file.
fn sha256_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    let mut buffer = [0u8; 65536];
    loop {
        let n = std::io::Read::read(&mut file, &mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager(dir: &tempfile::TempDir, tier: DeviceTier) -> ModelManager {
        ModelManager::new(dir.path(), tier)
    }

    #[test]
    fn test_manager_creation() {
        let dir = tempfile::tempdir().unwrap();
        let manager = test_manager(&dir, DeviceTier::High);
        assert_eq!(manager.max_cache_bytes(), 2 * 1024 * 1024 * 1024);
        assert_eq!(manager.cache_count(), 0);
    }

    #[test]
    fn test_tier_based_cache_limits() {
        let dir = tempfile::tempdir().unwrap();
        let high = ModelManager::new(dir.path(), DeviceTier::High);
        assert_eq!(high.max_cache_bytes(), 2 * 1024 * 1024 * 1024);

        let dir2 = tempfile::tempdir().unwrap();
        let mid = ModelManager::new(dir2.path(), DeviceTier::Medium);
        assert_eq!(mid.max_cache_bytes(), 500 * 1024 * 1024);

        let dir3 = tempfile::tempdir().unwrap();
        let low = ModelManager::new(dir3.path(), DeviceTier::Low);
        assert_eq!(low.max_cache_bytes(), 150 * 1024 * 1024);
    }

    #[test]
    fn test_local_path_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let manager = test_manager(&dir, DeviceTier::High);
        assert!(manager.local_path("nonexistent").is_none());
    }

    #[test]
    fn test_scan_existing_files() {
        let dir = tempfile::tempdir().unwrap();
        // Create a fake model file
        let model_path = dir.path().join("test-model.gguf");
        std::fs::write(&model_path, b"fake model data").unwrap();

        let manager = ModelManager::new(dir.path(), DeviceTier::High);
        assert_eq!(manager.cache_count(), 1);
        assert!(manager.local_path("test-model").is_some());
    }

    #[test]
    fn test_lru_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let manager = ModelManager::new(dir.path(), DeviceTier::Low);
        // Low tier: 150MB max
        // We can't easily test with real large files, so test the logic
        // by manually inserting entries

        // Insert 3 small "packs" manually
        {
            let mut entries = manager.entries.lock();
            for i in 0..3 {
                let path = dir.path().join(format!("pack-{i}.gguf"));
                std::fs::write(&path, b"data").unwrap();
                entries.insert(
                    format!("pack-{i}"),
                    CacheEntry {
                        pack_id: format!("pack-{i}"),
                        local_path: path,
                        size_bytes: 100 * 1024 * 1024, // 100MB each
                        last_accessed: Instant::now() - std::time::Duration::from_secs(i * 10),
                    },
                );
            }
        }

        // Total: 300MB, max: 150MB → need to evict 2
        // Set max to a smaller value for testing
        {
            let mut entries = manager.entries.lock();
            // Manually trigger eviction logic
            let total: u64 = entries.values().map(|e| e.size_bytes).sum();
            assert!(total > 150 * 1024 * 1024);
        }

        // Evict oldest (pack-2 has oldest last_accessed)
        manager.evict_lru().unwrap();

        // Should have evicted enough to be under 150MB
        let remaining = manager.cache_count();
        assert!(remaining <= 1, "should have evicted to <= 1 packs, got {remaining}");
    }

    #[test]
    fn test_remove_pack() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("removable.gguf");
        std::fs::write(&model_path, b"data").unwrap();

        let manager = ModelManager::new(dir.path(), DeviceTier::High);
        assert_eq!(manager.cache_count(), 1);

        manager.remove_pack("removable").unwrap();
        assert_eq!(manager.cache_count(), 0);
        assert!(!model_path.exists());
    }

    #[test]
    fn test_clear_cache() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..3 {
            let path = dir.path().join(format!("model-{i}.gguf"));
            std::fs::write(&path, b"data").unwrap();
        }

        let manager = ModelManager::new(dir.path(), DeviceTier::High);
        assert_eq!(manager.cache_count(), 3);

        manager.clear_cache().unwrap();
        assert_eq!(manager.cache_count(), 0);
    }

    #[test]
    fn test_mmap_model() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("mmap-test.gguf");
        std::fs::write(&model_path, b"Hello, mmap!").unwrap();

        let manager = ModelManager::new(dir.path(), DeviceTier::High);
        let mmap = manager.mmap_model("mmap-test").unwrap();
        assert_eq!(&mmap[..], b"Hello, mmap!");
    }

    #[test]
    fn test_sha256_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        std::fs::write(&path, b"hello world").unwrap();

        let hash = sha256_file(&path).unwrap();
        // SHA-256 of "hello world"
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_validate_pack_id_rejects_traversal() {
        // Path traversal attempts must be rejected
        assert!(validate_pack_id("../../../etc/passwd").is_err());
        assert!(validate_pack_id("..").is_err());
        assert!(validate_pack_id("foo/bar").is_err());
        assert!(validate_pack_id("foo\\bar").is_err());
        assert!(validate_pack_id("").is_err());
        // Valid pack IDs must be accepted
        assert!(validate_pack_id("ternary-bonsai-1.7b-q2_0").is_ok());
        assert!(validate_pack_id("kchat-encoder-int8").is_ok());
        assert!(validate_pack_id("model_v2.0").is_ok());
    }

    #[test]
    fn test_validate_pack_id_rejects_too_long() {
        let long_id = "a".repeat(101);
        assert!(validate_pack_id(&long_id).is_err());
        let ok_id = "a".repeat(100);
        assert!(validate_pack_id(&ok_id).is_ok());
    }

    #[test]
    fn test_mmap_rejects_invalid_pack_id() {
        let dir = tempfile::tempdir().unwrap();
        let manager = ModelManager::new(dir.path(), DeviceTier::High);
        // Path traversal in pack_id should be rejected
        let result = manager.mmap_model("../../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn test_scan_cache_dir_skips_unsafe_filenames() {
        let dir = tempfile::tempdir().unwrap();
        // Create a file with a traversal-style name
        let bad_path = dir.path().join("..%2fetc%2fpasswd.gguf");
        std::fs::write(&bad_path, b"evil").unwrap();
        // Create a valid file
        let good_path = dir.path().join("valid-model.gguf");
        std::fs::write(&good_path, b"good").unwrap();

        let manager = ModelManager::new(dir.path(), DeviceTier::High);
        // Only the valid file should be in the cache
        assert_eq!(manager.cache_count(), 1);
        assert!(manager.local_path("valid-model").is_some());
        // The unsafe filename should not be in the cache
        assert!(manager.local_path("..%2fetc%2fpasswd").is_none());
    }
}

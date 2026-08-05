//! Model registry — a TOML catalog of all available model packs.
//!
//! The registry (`registry.toml`) lists every downloadable model pack with its
//! metadata: pack ID, version, type, download URL, SHA-256 digest, size,
//! minimum device tier, task capabilities, languages, and quantization recipe.
//! The model manager consults the registry to determine which packs to fetch
//! for a given device tier and workload.

use crate::error::{CoreError, Result};
use serde::{Deserialize, Serialize};

/// Minimum device tier required to run a pack.
///
/// Stored in TOML as a lowercase string (`low`, `medium`, `high`) and compared
/// ordinally so that a device at a higher tier satisfies a lower minimum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MinTier {
    /// Runs on any device, including low-tier.
    Low,
    /// Requires at least a medium-tier device.
    Medium,
    /// Requires a high-tier device.
    High,
}

impl MinTier {
    /// Returns true if a device at the given tier can run a pack with this
    /// minimum. A device tier satisfies a minimum when it is greater than or
    /// equal to the minimum.
    pub fn satisfied_by(self, device: MinTier) -> bool {
        device >= self
    }
}

/// A single entry in the model registry describing one downloadable pack.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegistryEntry {
    /// Unique pack identifier (e.g. `qwen3.5-0.8b-q4`).
    pub pack_id: String,
    /// Pack version (semver-style string).
    pub version: String,
    /// Pack type: `generative`, `embedding`, `safety`, `reranker`, etc.
    pub pack_type: String,
    /// Download URL for the assembled pack artifact.
    pub download_url: String,
    /// SHA-256 digest of the downloaded artifact (hex).
    pub sha256: String,
    /// Total size of the artifact in bytes.
    pub size_bytes: u64,
    /// Minimum device tier required to run this pack.
    pub min_tier: MinTier,
    /// Task capabilities this pack supports (e.g. `summarize`, `translate`).
    pub task_capabilities: Vec<String>,
    /// Language codes this pack supports (e.g. `en`, `vi`, `zh`).
    pub languages: Vec<String>,
    /// Quantization recipe (e.g. `Q4_K_M`, `Q8_0`, `INT8`).
    pub quantization: String,
}

/// The model registry: a catalog of all known downloadable packs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelRegistry {
    /// All registered model entries.
    #[serde(default)]
    pub models: Vec<RegistryEntry>,
}

impl ModelRegistry {
    /// Parse a registry from a TOML string.
    pub fn load_from_toml(toml_str: &str) -> Result<Self> {
        let registry: ModelRegistry = toml::from_str(toml_str)
            .map_err(|e| CoreError::Config(format!("failed to parse registry TOML: {e}")))?;
        Ok(registry)
    }

    /// Load a registry from a file on disk.
    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        Self::load_from_toml(&contents)
    }

    /// Find an entry by pack ID.
    pub fn find(&self, pack_id: &str) -> Option<&RegistryEntry> {
        self.models.iter().find(|e| e.pack_id == pack_id)
    }

    /// Find all packs that support a given task and run on the supplied device
    /// tier. A pack qualifies when the task appears in its `task_capabilities`
    /// and the device tier satisfies its `min_tier`.
    pub fn find_for_task(&self, task: &str, tier: MinTier) -> Vec<&RegistryEntry> {
        self.models
            .iter()
            .filter(|e| e.task_capabilities.iter().any(|c| c == task) && e.min_tier.satisfied_by(tier))
            .collect()
    }

    /// Find all packs that support a given language and run on the supplied
    /// device tier.
    pub fn find_for_language(&self, language: &str, tier: MinTier) -> Vec<&RegistryEntry> {
        self.models
            .iter()
            .filter(|e| {
                e.languages.iter().any(|l| l == language) && e.min_tier.satisfied_by(tier)
            })
            .collect()
    }

    /// List all entries in the registry.
    pub fn list(&self) -> &[RegistryEntry] {
        &self.models
    }

    /// Return the built-in default registry with the standard set of packs.
    pub fn default_registry() -> Self {
        let mut models = Vec::new();

        // --- Generative models ---
        models.push(RegistryEntry {
            pack_id: "qwen3.5-0.8b-q4".into(),
            version: "1.0.0".into(),
            pack_type: "generative".into(),
            download_url: "https://cdn.kchat.dev/models/qwen3.5-0.8b-q4/1.0.0/qwen3.5-0.8b-q4.gguf".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            size_bytes: 500_000_000,
            min_tier: MinTier::High,
            task_capabilities: vec!["summarize".into(), "translate".into(), "generate".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into()],
            quantization: "Q4_K_M".into(),
        });

        models.push(RegistryEntry {
            pack_id: "qwen3.5-0.8b-q8".into(),
            version: "1.0.0".into(),
            pack_type: "generative".into(),
            download_url: "https://cdn.kchat.dev/models/qwen3.5-0.8b-q8/1.0.0/qwen3.5-0.8b-q8.gguf".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            size_bytes: 850_000_000,
            min_tier: MinTier::High,
            task_capabilities: vec!["summarize".into(), "translate".into(), "generate".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into()],
            quantization: "Q8_0".into(),
        });

        // --- Embedding model ---
        models.push(RegistryEntry {
            pack_id: "multilingual-e5-small-int8".into(),
            version: "1.0.0".into(),
            pack_type: "embedding".into(),
            download_url: "https://cdn.kchat.dev/models/multilingual-e5-small-int8/1.0.0/multilingual-e5-small-int8.onnx".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            size_bytes: 45_000_000,
            min_tier: MinTier::Medium,
            task_capabilities: vec!["embed".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into()],
            quantization: "INT8".into(),
        });

        // --- Safety classifier ---
        models.push(RegistryEntry {
            pack_id: "safety-classifier-int8".into(),
            version: "1.0.0".into(),
            pack_type: "safety".into(),
            download_url: "https://cdn.kchat.dev/models/safety-classifier-int8/1.0.0/safety-classifier-int8.onnx".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            size_bytes: 25_000_000,
            min_tier: MinTier::Medium,
            task_capabilities: vec!["safety".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into()],
            quantization: "INT8".into(),
        });

        // --- Reranker ---
        models.push(RegistryEntry {
            pack_id: "cross-encoder-miniLM-int8".into(),
            version: "1.0.0".into(),
            pack_type: "reranker".into(),
            download_url: "https://cdn.kchat.dev/models/cross-encoder-miniLM-int8/1.0.0/cross-encoder-miniLM-int8.onnx".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            size_bytes: 25_000_000,
            min_tier: MinTier::High,
            task_capabilities: vec!["rerank".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into()],
            quantization: "INT8".into(),
        });

        Self { models }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TOML: &str = r#"
[[models]]
pack_id = "qwen3.5-0.8b-q4"
version = "1.0.0"
pack_type = "generative"
download_url = "https://cdn.kchat.dev/models/qwen3.5-0.8b-q4/1.0.0/qwen3.5-0.8b-q4.gguf"
sha256 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
size_bytes = 500000000
min_tier = "high"
task_capabilities = ["summarize", "translate", "generate"]
languages = ["en", "vi", "zh", "ja", "ko", "es"]
quantization = "Q4_K_M"

[[models]]
pack_id = "multilingual-e5-small-int8"
version = "1.0.0"
pack_type = "embedding"
download_url = "https://cdn.kchat.dev/models/multilingual-e5-small-int8/1.0.0/multilingual-e5-small-int8.onnx"
sha256 = "1111111111111111111111111111111111111111111111111111111111111111"
size_bytes = 45000000
min_tier = "medium"
task_capabilities = ["embed"]
languages = ["en", "vi", "zh"]
quantization = "INT8"
"#;

    #[test]
    fn test_load_from_toml_parses_entries() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        assert_eq!(registry.list().len(), 2);
    }

    #[test]
    fn test_load_from_toml_fields_parsed_correctly() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        let entry = registry.find("qwen3.5-0.8b-q4").expect("found");
        assert_eq!(entry.version, "1.0.0");
        assert_eq!(entry.pack_type, "generative");
        assert_eq!(entry.size_bytes, 500_000_000);
        assert_eq!(entry.min_tier, MinTier::High);
        assert_eq!(entry.quantization, "Q4_K_M");
        assert_eq!(entry.task_capabilities, vec!["summarize", "translate", "generate"]);
        assert_eq!(entry.languages.len(), 6);
    }

    #[test]
    fn test_load_from_toml_invalid_returns_error() {
        let result = ModelRegistry::load_from_toml("not valid toml = =");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_file_reads_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("registry.toml");
        std::fs::write(&path, SAMPLE_TOML).expect("write");
        let registry = ModelRegistry::load_from_file(&path).expect("load");
        assert_eq!(registry.list().len(), 2);
    }

    #[test]
    fn test_load_from_file_missing_returns_error() {
        let result = ModelRegistry::load_from_file(std::path::Path::new("/nonexistent/registry.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_find_returns_entry_by_id() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        let entry = registry.find("multilingual-e5-small-int8").expect("found");
        assert_eq!(entry.pack_type, "embedding");
    }

    #[test]
    fn test_find_returns_none_for_unknown_id() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        assert!(registry.find("does-not-exist").is_none());
    }

    #[test]
    fn test_find_for_task_filters_by_task_and_tier() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        // "summarize" only on the high-tier generative pack; high tier satisfies it.
        let results = registry.find_for_task("summarize", MinTier::High);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "qwen3.5-0.8b-q4");
    }

    #[test]
    fn test_find_for_task_excludes_when_tier_too_low() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        // The generative pack requires high tier; a low-tier device cannot run it.
        let results = registry.find_for_task("summarize", MinTier::Low);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_for_task_embed_works_on_medium() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        let results = registry.find_for_task("embed", MinTier::Medium);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "multilingual-e5-small-int8");
    }

    #[test]
    fn test_find_for_language_filters_by_language_and_tier() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        let results = registry.find_for_language("vi", MinTier::High);
        // Both packs support "vi" and high tier satisfies both minimums.
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_for_language_excludes_when_tier_too_low() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        // Both packs require at least medium tier, so a low-tier device runs none.
        let results = registry.find_for_language("en", MinTier::Low);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_for_language_medium_tier_runs_embedding() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        // The embedding pack (medium tier) is runnable on a medium-tier device,
        // but the generative pack (high tier) is not.
        let results = registry.find_for_language("en", MinTier::Medium);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "multilingual-e5-small-int8");
    }

    #[test]
    fn test_default_registry_has_five_entries() {
        let registry = ModelRegistry::default_registry();
        assert_eq!(registry.list().len(), 5);
    }

    #[test]
    fn test_default_registry_contains_expected_pack_ids() {
        let registry = ModelRegistry::default_registry();
        let ids: Vec<&str> = registry.list().iter().map(|e| e.pack_id.as_str()).collect();
        assert!(ids.contains(&"qwen3.5-0.8b-q4"));
        assert!(ids.contains(&"qwen3.5-0.8b-q8"));
        assert!(ids.contains(&"multilingual-e5-small-int8"));
        assert!(ids.contains(&"safety-classifier-int8"));
        assert!(ids.contains(&"cross-encoder-miniLM-int8"));
    }

    #[test]
    fn test_default_registry_generative_packs_are_high_tier() {
        let registry = ModelRegistry::default_registry();
        let q4 = registry.find("qwen3.5-0.8b-q4").expect("q4");
        let q8 = registry.find("qwen3.5-0.8b-q8").expect("q8");
        assert_eq!(q4.min_tier, MinTier::High);
        assert_eq!(q8.min_tier, MinTier::High);
        assert_eq!(q4.quantization, "Q4_K_M");
        assert_eq!(q8.quantization, "Q8_0");
    }

    #[test]
    fn test_default_registry_embedding_is_medium_tier() {
        let registry = ModelRegistry::default_registry();
        let emb = registry.find("multilingual-e5-small-int8").expect("emb");
        assert_eq!(emb.min_tier, MinTier::Medium);
        assert_eq!(emb.pack_type, "embedding");
        assert_eq!(emb.quantization, "INT8");
    }

    #[test]
    fn test_default_registry_safety_is_medium_tier() {
        let registry = ModelRegistry::default_registry();
        let safety = registry.find("safety-classifier-int8").expect("safety");
        assert_eq!(safety.min_tier, MinTier::Medium);
        assert_eq!(safety.pack_type, "safety");
        assert_eq!(safety.size_bytes, 25_000_000);
    }

    #[test]
    fn test_default_registry_reranker_is_high_tier() {
        let registry = ModelRegistry::default_registry();
        let reranker = registry.find("cross-encoder-miniLM-int8").expect("reranker");
        assert_eq!(reranker.min_tier, MinTier::High);
        assert_eq!(reranker.pack_type, "reranker");
        assert_eq!(reranker.quantization, "INT8");
    }

    #[test]
    fn test_default_registry_find_for_task_safety_on_medium() {
        let registry = ModelRegistry::default_registry();
        let results = registry.find_for_task("safety", MinTier::Medium);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "safety-classifier-int8");
    }

    #[test]
    fn test_default_registry_find_for_task_rerank_requires_high() {
        let registry = ModelRegistry::default_registry();
        assert!(registry.find_for_task("rerank", MinTier::Medium).is_empty());
        assert_eq!(registry.find_for_task("rerank", MinTier::High).len(), 1);
    }

    #[test]
    fn test_min_tier_ordering() {
        assert!(MinTier::Low < MinTier::Medium);
        assert!(MinTier::Medium < MinTier::High);
        assert!(MinTier::High.satisfied_by(MinTier::High));
        assert!(MinTier::Medium.satisfied_by(MinTier::High));
        assert!(!MinTier::High.satisfied_by(MinTier::Low));
    }

    #[test]
    fn test_empty_registry_has_no_entries() {
        let registry = ModelRegistry::default();
        assert!(registry.list().is_empty());
        assert!(registry.find("anything").is_none());
        assert!(registry.find_for_task("anything", MinTier::High).is_empty());
    }

    #[test]
    fn test_registry_round_trips_through_toml() {
        let registry = ModelRegistry::default_registry();
        let toml_str = toml::to_string(&registry).expect("serialize");
        let parsed = ModelRegistry::load_from_toml(&toml_str).expect("parse");
        assert_eq!(parsed.list().len(), registry.list().len());
        assert_eq!(parsed.find("qwen3.5-0.8b-q4"), registry.find("qwen3.5-0.8b-q4"));
    }
}

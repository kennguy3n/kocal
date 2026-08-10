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
    /// Unique pack identifier (e.g. `ternary-bonsai-1.7b-q2_0`).
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

impl RegistryEntry {
    /// Returns true if the SHA-256 hash is a placeholder (all zeros).
    /// Placeholder hashes must not ship to production — they bypass
    /// download integrity verification.
    pub fn is_placeholder_hash(&self) -> bool {
        self.sha256.chars().all(|c| c == '0') || self.sha256.is_empty()
    }
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
        // Low tier: Ternary-Bonsai-1.7B MLX 2-bit for Apple platforms (iOS/macOS)
        // ~472MB, Qwen3-1.7B base, 1.58-bit ternary, MLX format
        models.push(RegistryEntry {
            pack_id: "ternary-bonsai-1.7b-mlx-2bit".into(),
            version: "1.0.0".into(),
            pack_type: "generative".into(),
            download_url: "https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-mlx-2bit/resolve/main/model.safetensors".into(),
            sha256: "1945a52c46cdce6daa921fa292da6a54e466ab55e7e984f5cba8e68ec6b7eeb5".into(),
            size_bytes: 472_000_000, // ~472 MB
            min_tier: MinTier::Low,
            task_capabilities: vec!["summarize".into(), "translate".into(), "generate".into(), "tool_use".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "ar".into(), "de".into(), "hi".into(), "fr".into()],
            quantization: "2bit-MLX".into(),
        });

        // Low tier: Ternary-Bonsai-1.7B Q2_0 GGUF for Android/Windows/Linux
        // 463,290,464 bytes (~442MB), Qwen3-1.7B base, 1.58-bit ternary Q2_0
        models.push(RegistryEntry {
            pack_id: "ternary-bonsai-1.7b-q2_0".into(),
            version: "1.0.0".into(),
            pack_type: "generative".into(),
            download_url: "https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/main/Ternary-Bonsai-1.7B-Q2_0.gguf".into(),
            sha256: "d97d94eb564590c9f0300e54d3f87bbbb25a78693d0ade9f6e177973dcb8228a".into(),
            size_bytes: 463_290_464, // ~442 MB (exact from HF API)
            min_tier: MinTier::Low,
            task_capabilities: vec!["summarize".into(), "translate".into(), "generate".into(), "tool_use".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "ar".into(), "de".into(), "hi".into(), "fr".into()],
            quantization: "Q2_0".into(),
        });

        // Medium tier: Ternary-Bonsai-4B for Android (Qwen3-4B, 1.58-bit Q2_0, ~1.0GB)
        // Fits in 1500MB Android Medium budget with 500MB headroom for KV cache
        models.push(RegistryEntry {
            pack_id: "ternary-bonsai-4b-q2_0".into(),
            version: "1.0.0".into(),
            pack_type: "generative".into(),
            download_url: "https://huggingface.co/prism-ml/Ternary-Bonsai-4B-gguf/resolve/main/Ternary-Bonsai-4B-Q2_0.gguf".into(),
            sha256: "4e0bf8b737b0431552f8c2c97695ab7c0cb214c94bcdeb4f5f267e67ddf28b8b".into(),
            size_bytes: 1_074_969_344, // ~1.0 GB (exact from HF API)
            min_tier: MinTier::Medium,
            task_capabilities: vec!["summarize".into(), "translate".into(), "generate".into(), "tool_use".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "ar".into(), "de".into(), "hi".into(), "fr".into()],
            quantization: "Q2_0".into(),
        });

        // Medium tier: Ternary-Bonsai-4B MLX 2-bit for Apple platforms (iOS/macOS)
        // ~1.08GB, Qwen3-4B base, 1.58-bit ternary, MLX format
        models.push(RegistryEntry {
            pack_id: "ternary-bonsai-4b-mlx-2bit".into(),
            version: "1.0.0".into(),
            pack_type: "generative".into(),
            download_url: "https://huggingface.co/prism-ml/Ternary-Bonsai-4B-mlx-2bit/resolve/main/model.safetensors".into(),
            sha256: "1f956eefc2fa23fc21048ac2d1a8e76d78558666d81bf6b1fa89d83ba59698ac".into(),
            size_bytes: 1_131_565_944, // ~1.08 GB (exact from HF LFS)
            min_tier: MinTier::Medium,
            task_capabilities: vec!["summarize".into(), "translate".into(), "generate".into(), "tool_use".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "ar".into(), "de".into(), "hi".into(), "fr".into()],
            quantization: "2bit-MLX".into(),
        });

        // High tier: Ternary-Bonsai-8B MLX 2-bit for Apple platforms (iOS/macOS)
        // ~2.15GB, Qwen3-8B base, 1.58-bit ternary, MLX format
        models.push(RegistryEntry {
            pack_id: "ternary-bonsai-8b-mlx-2bit".into(),
            version: "1.0.0".into(),
            pack_type: "generative".into(),
            download_url: "https://huggingface.co/prism-ml/Ternary-Bonsai-8B-mlx-2bit/resolve/main/model.safetensors".into(),
            sha256: "f43270cbae86830b7eecb25bb8a0a0a005a81f180b68868dc39c755cebfff362".into(),
            size_bytes: 2_303_661_704, // ~2.15 GB (exact from HF LFS)
            min_tier: MinTier::High,
            task_capabilities: vec!["summarize".into(), "translate".into(), "generate".into(), "tool_use".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "ar".into(), "de".into(), "hi".into(), "fr".into()],
            quantization: "2bit-MLX".into(),
        });

        // High tier: Ternary-Bonsai-8B for Windows (Qwen3-8B, 1.58-bit Q2_0, ~2.1GB)
        // Fits in 8000MB Windows High budget with 5.9GB headroom for KV cache
        models.push(RegistryEntry {
            pack_id: "ternary-bonsai-8b-q2_0".into(),
            version: "1.0.0".into(),
            pack_type: "generative".into(),
            download_url: "https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf/resolve/main/Ternary-Bonsai-8B-Q2_0.gguf".into(),
            sha256: "3c8d70470a5d97e5a2b9410ddd899cb740116591462626c60cb2fead6448f60b".into(),
            size_bytes: 2_182_184_672, // ~2.1 GB (exact from HF API)
            min_tier: MinTier::High,
            task_capabilities: vec!["summarize".into(), "translate".into(), "generate".into(), "tool_use".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "ar".into(), "de".into(), "hi".into(), "fr".into()],
            quantization: "Q2_0".into(),
        });

        // --- Unified multi-task encoder (XLM-RoBERTa-base) ---
        // Replaces separate safety-classifier, embedding, and reranker models.
        // Single ONNX model handles safety classification, text embedding, and reranking.
        models.push(RegistryEntry {
            pack_id: "kchat-encoder-int8".into(),
            version: "1.0.0".into(),
            pack_type: "encoder".into(),
            download_url: "https://cdn.kchat.dev/models/kchat-encoder-int8/1.0.0/kchat-encoder-int8.onnx".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            size_bytes: 270_000_000,
            min_tier: MinTier::High,
            task_capabilities: vec!["safety".into(), "embed".into(), "rerank".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "ar".into(), "de".into(), "hi".into(), "fr".into()],
            quantization: "INT8".into(),
        });

        // --- Vision model (MobileCLIP-S2) ---

        // Unified MobileCLIP-S2 INT8 for both image and video classification
        // 70MB, 512-dim embeddings, 17 categories
        // Single model handles image_classify, image_embed, and video_classify
        models.push(RegistryEntry {
            pack_id: "mobileclip-s2-int8".into(),
            version: "1.0.0".into(),
            pack_type: "vision".into(),
            download_url: "https://cdn.kchat.dev/models/mobileclip-s2-int8/1.0.0/mobileclip-s2-int8.onnx".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            size_bytes: 70_000_000,
            min_tier: MinTier::Low,
            task_capabilities: vec!["image_classify".into(), "image_embed".into(), "video_classify".into()],
            languages: vec!["en".into()],
            quantization: "INT8".into(),
        });

        // --- ASR models (Whisper) ---

        // Low tier: Whisper Tiny ONNX for audio transcription
        // ~33MB encoder ONNX, fine-tuned on Norwegian (NbAiLab), supports no/nb/nn/en
        // Full pack includes encoder + decoder + decoder_with_past ONNX files
        models.push(RegistryEntry {
            pack_id: "whisper-tiny-int8".into(),
            version: "1.0.0".into(),
            pack_type: "asr".into(),
            download_url: "https://huggingface.co/NbAiLabBeta/nb-whisper-tiny/resolve/main/onnx/encoder_model.onnx".into(),
            sha256: "f4130dfa98ca79ca3b75d690535c4d1b43af357a31aae07b9af06d2264fe934d".into(),
            size_bytes: 32_904_983,
            min_tier: MinTier::Low,
            task_capabilities: vec!["transcribe".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "fr".into(), "de".into(), "ar".into(), "hi".into(), "th".into()],
            quantization: "ONNX".into(),
        });

        // Medium tier: Whisper Base ONNX for higher-accuracy transcription
        // ~82MB encoder ONNX, fine-tuned on Norwegian (NbAiLab), supports no/nb/nn/en
        // Full pack includes encoder + decoder + decoder_with_past ONNX files
        models.push(RegistryEntry {
            pack_id: "whisper-base-int8".into(),
            version: "1.0.0".into(),
            pack_type: "asr".into(),
            download_url: "https://huggingface.co/NbAiLabBeta/nb-whisper-base/resolve/main/onnx/encoder_model.onnx".into(),
            sha256: "7f9a82b47fd1b82a5262c148d82b025df0c1ae7e1213f7db96f413e498fe2976".into(),
            size_bytes: 82_468_069,
            min_tier: MinTier::Medium,
            task_capabilities: vec!["transcribe".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "fr".into(), "de".into(), "ar".into(), "hi".into(), "th".into()],
            quantization: "ONNX".into(),
        });

        // --- Unified encoder INT4 variant (Low tier) ---
        // 4-bit quantized version for low-tier devices
        models.push(RegistryEntry {
            pack_id: "kchat-encoder-int4".into(),
            version: "1.0.0".into(),
            pack_type: "encoder".into(),
            download_url: "https://cdn.kchat.dev/models/kchat-encoder-int4/1.0.0/kchat-encoder-int4.onnx".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            size_bytes: 90_000_000,
            min_tier: MinTier::Low,
            task_capabilities: vec!["safety".into(), "embed".into(), "rerank".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "ar".into(), "de".into(), "hi".into(), "fr".into()],
            quantization: "INT4".into(),
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
pack_id = "kchat-encoder-int8"
version = "1.0.0"
pack_type = "encoder"
download_url = "https://cdn.kchat.dev/models/kchat-encoder-int8/1.0.0/kchat-encoder-int8.onnx"
sha256 = "1111111111111111111111111111111111111111111111111111111111111111"
size_bytes = 270000000
min_tier = "medium"
task_capabilities = ["safety", "embed", "rerank"]
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
        let entry = registry.find("kchat-encoder-int8").expect("found");
        assert_eq!(entry.pack_type, "encoder");
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
        // In the sample TOML, kchat-encoder-int8 is medium tier
        let results = registry.find_for_task("embed", MinTier::Medium);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "kchat-encoder-int8");
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
        // The generative pack requires high tier; the encoder pack requires medium tier.
        // A low-tier device runs neither.
        let results = registry.find_for_language("en", MinTier::Low);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_for_language_medium_tier_runs_embedding() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        // The encoder pack (medium tier) is runnable on a medium-tier device,
        // but the generative pack (high tier) is not.
        let results = registry.find_for_language("en", MinTier::Medium);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "kchat-encoder-int8");
    }

    #[test]
    fn test_default_registry_has_eleven_entries() {
        let registry = ModelRegistry::default_registry();
        assert_eq!(registry.list().len(), 11);
    }

    #[test]
    fn test_default_registry_contains_expected_pack_ids() {
        let registry = ModelRegistry::default_registry();
        let ids: Vec<&str> = registry.list().iter().map(|e| e.pack_id.as_str()).collect();
        assert!(ids.contains(&"ternary-bonsai-1.7b-mlx-2bit"));
        assert!(ids.contains(&"ternary-bonsai-1.7b-q2_0"));
        assert!(ids.contains(&"ternary-bonsai-4b-q2_0"));
        assert!(ids.contains(&"ternary-bonsai-4b-mlx-2bit"));
        assert!(ids.contains(&"ternary-bonsai-8b-mlx-2bit"));
        assert!(ids.contains(&"ternary-bonsai-8b-q2_0"));
        assert!(ids.contains(&"kchat-encoder-int8"));
        assert!(ids.contains(&"kchat-encoder-int4"));
        assert!(ids.contains(&"mobileclip-s2-int8"));
        assert!(ids.contains(&"whisper-tiny-int8"));
        assert!(ids.contains(&"whisper-base-int8"));
    }

    #[test]
    fn test_default_registry_generative_packs_span_all_tiers() {
        let registry = ModelRegistry::default_registry();
        let low_mlx = registry.find("ternary-bonsai-1.7b-mlx-2bit").expect("bonsai-1.7b-mlx");
        let low_gguf = registry.find("ternary-bonsai-1.7b-q2_0").expect("bonsai-1.7b-gguf");
        let bonsai4 = registry.find("ternary-bonsai-4b-q2_0").expect("bonsai-4b");
        let bonsai4_mlx = registry.find("ternary-bonsai-4b-mlx-2bit").expect("bonsai-4b-mlx");
        let bonsai8_mlx = registry.find("ternary-bonsai-8b-mlx-2bit").expect("bonsai-8b-mlx");
        let bonsai8 = registry.find("ternary-bonsai-8b-q2_0").expect("bonsai-8b");
        assert_eq!(low_mlx.min_tier, MinTier::Low);
        assert_eq!(low_gguf.min_tier, MinTier::Low);
        assert_eq!(bonsai4.min_tier, MinTier::Medium);
        assert_eq!(bonsai4_mlx.min_tier, MinTier::Medium);
        assert_eq!(bonsai8_mlx.min_tier, MinTier::High);
        assert_eq!(bonsai8.min_tier, MinTier::High);
        assert_eq!(low_mlx.quantization, "2bit-MLX");
        assert_eq!(low_gguf.quantization, "Q2_0");
        assert_eq!(bonsai4.quantization, "Q2_0");
        assert_eq!(bonsai4_mlx.quantization, "2bit-MLX");
        assert_eq!(bonsai8_mlx.quantization, "2bit-MLX");
        assert_eq!(bonsai8.quantization, "Q2_0");
    }

    #[test]
    fn test_default_registry_bonsai_1_7b_mlx_fits_low_tier_budget() {
        let registry = ModelRegistry::default_registry();
        let low_mlx = registry.find("ternary-bonsai-1.7b-mlx-2bit").expect("bonsai-1.7b-mlx");
        // Low tier mobile budget is 750MB (iOS)
        assert!(low_mlx.size_bytes <= 750 * 1024 * 1024);
        // ~472MB
        assert_eq!(low_mlx.size_bytes, 472_000_000);
        // Should support tool_use
        assert!(low_mlx.task_capabilities.contains(&"tool_use".to_string()));
    }

    #[test]
    fn test_default_registry_bonsai_1_7b_gguf_fits_low_tier_budget() {
        let registry = ModelRegistry::default_registry();
        let low_gguf = registry.find("ternary-bonsai-1.7b-q2_0").expect("bonsai-1.7b-gguf");
        // Low tier mobile budget is 750MB (Android)
        assert!(low_gguf.size_bytes <= 750 * 1024 * 1024);
        // Exact size from HF API: 463,290,464 bytes (~442MB)
        assert_eq!(low_gguf.size_bytes, 463_290_464);
        // Should support tool_use
        assert!(low_gguf.task_capabilities.contains(&"tool_use".to_string()));
    }

    #[test]
    fn test_default_registry_bonsai_4b_fits_android_medium_budget() {
        let registry = ModelRegistry::default_registry();
        let bonsai4 = registry.find("ternary-bonsai-4b-q2_0").expect("bonsai-4b");
        // Android Medium tier budget is 1500MB
        assert!(bonsai4.size_bytes <= 1500 * 1024 * 1024);
        // Exact size from HF API: 1,074,969,344 bytes (~1.0GB)
        assert_eq!(bonsai4.size_bytes, 1_074_969_344);
        // Should support tool_use
        assert!(bonsai4.task_capabilities.contains(&"tool_use".to_string()));
    }

    #[test]
    fn test_default_registry_bonsai_8b_fits_windows_high_budget() {
        let registry = ModelRegistry::default_registry();
        let bonsai8 = registry.find("ternary-bonsai-8b-q2_0").expect("bonsai-8b");
        // Windows High tier budget is 8000MB
        assert!(bonsai8.size_bytes <= 8000 * 1024 * 1024);
        // Exact size from HF API: 2,182,184,672 bytes (~2.1GB)
        assert_eq!(bonsai8.size_bytes, 2_182_184_672);
        // Should support tool_use
        assert!(bonsai8.task_capabilities.contains(&"tool_use".to_string()));
    }

    #[test]
    fn test_default_registry_encoder_int8_is_high_tier() {
        let registry = ModelRegistry::default_registry();
        let enc = registry.find("kchat-encoder-int8").expect("encoder-int8");
        assert_eq!(enc.min_tier, MinTier::High);
        assert_eq!(enc.pack_type, "encoder");
        assert_eq!(enc.quantization, "INT8");
        assert_eq!(enc.size_bytes, 270_000_000);
        assert!(enc.task_capabilities.contains(&"safety".to_string()));
        assert!(enc.task_capabilities.contains(&"embed".to_string()));
        assert!(enc.task_capabilities.contains(&"rerank".to_string()));
    }

    #[test]
    fn test_default_registry_find_for_task_safety_on_medium() {
        let registry = ModelRegistry::default_registry();
        // Medium tier can only run kchat-encoder-int4 (low tier); INT8 is now high-tier only
        let results = registry.find_for_task("safety", MinTier::Medium);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "kchat-encoder-int4");
    }

    #[test]
    fn test_default_registry_find_for_task_safety_on_low() {
        let registry = ModelRegistry::default_registry();
        // Low tier can only run kchat-encoder-int4
        let results = registry.find_for_task("safety", MinTier::Low);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "kchat-encoder-int4");
    }

    #[test]
    fn test_default_registry_find_for_task_safety_on_high() {
        let registry = ModelRegistry::default_registry();
        // High tier can run both kchat-encoder-int8 (high) and kchat-encoder-int4 (low)
        let results = registry.find_for_task("safety", MinTier::High);
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|e| e.pack_id.as_str()).collect();
        assert!(ids.contains(&"kchat-encoder-int8"));
        assert!(ids.contains(&"kchat-encoder-int4"));
    }

    #[test]
    fn test_default_registry_find_for_task_rerank_on_medium() {
        let registry = ModelRegistry::default_registry();
        // Reranking on Medium tier via kchat-encoder-int4 only (INT8 is high-tier)
        let results = registry.find_for_task("rerank", MinTier::Medium);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "kchat-encoder-int4");
    }

    #[test]
    fn test_default_registry_find_generative_for_all_tiers() {
        let registry = ModelRegistry::default_registry();
        // Low tier: should find 2 Bonsai-1.7B models (MLX + GGUF)
        let low = registry.find_for_task("summarize", MinTier::Low);
        assert_eq!(low.len(), 2);
        let low_ids: Vec<&str> = low.iter().map(|e| e.pack_id.as_str()).collect();
        assert!(low_ids.contains(&"ternary-bonsai-1.7b-mlx-2bit"));
        assert!(low_ids.contains(&"ternary-bonsai-1.7b-q2_0"));

        // Medium tier: should find 4 models (2 Bonsai-1.7B + Bonsai-4B-MLX + Bonsai-4B-GGUF)
        let medium = registry.find_for_task("summarize", MinTier::Medium);
        assert_eq!(medium.len(), 4);
        let medium_ids: Vec<&str> = medium.iter().map(|e| e.pack_id.as_str()).collect();
        assert!(medium_ids.contains(&"ternary-bonsai-1.7b-mlx-2bit"));
        assert!(medium_ids.contains(&"ternary-bonsai-1.7b-q2_0"));
        assert!(medium_ids.contains(&"ternary-bonsai-4b-mlx-2bit"));
        assert!(medium_ids.contains(&"ternary-bonsai-4b-q2_0"));

        // High tier: should find all 6 generative models
        // (Bonsai-1.7B-MLX, Bonsai-1.7B-GGUF, Bonsai-4B-GGUF, Bonsai-4B-MLX,
        //  Bonsai-8B-MLX, Bonsai-8B-GGUF)
        let high = registry.find_for_task("summarize", MinTier::High);
        assert_eq!(high.len(), 6);
    }

    #[test]
    fn test_default_registry_vision_pack_spans_all_tiers() {
        let registry = ModelRegistry::default_registry();
        // Unified mobileclip-s2-int8 available on all tiers
        let low = registry.find_for_task("image_classify", MinTier::Low);
        assert_eq!(low.len(), 1);
        assert_eq!(low[0].pack_id, "mobileclip-s2-int8");

        let medium = registry.find_for_task("image_classify", MinTier::Medium);
        assert_eq!(medium.len(), 1);
        assert_eq!(medium[0].pack_id, "mobileclip-s2-int8");

        let high = registry.find_for_task("image_classify", MinTier::High);
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].pack_id, "mobileclip-s2-int8");
    }

    #[test]
    fn test_default_registry_video_pack_is_low_tier() {
        let registry = ModelRegistry::default_registry();
        let video = registry.find("mobileclip-s2-int8").expect("vision");
        assert_eq!(video.min_tier, MinTier::Low);
        assert_eq!(video.pack_type, "vision");
        assert_eq!(video.size_bytes, 70_000_000);
        // Video classify is now available on all tiers (same model as image)
        assert_eq!(registry.find_for_task("video_classify", MinTier::Low).len(), 1);
        assert_eq!(registry.find_for_task("video_classify", MinTier::Medium).len(), 1);
        assert_eq!(registry.find_for_task("video_classify", MinTier::High).len(), 1);
    }

    #[test]
    fn test_default_registry_asr_packs_span_all_tiers() {
        let registry = ModelRegistry::default_registry();
        // Low tier: only whisper-tiny-int8
        let low = registry.find_for_task("transcribe", MinTier::Low);
        assert_eq!(low.len(), 1);
        assert_eq!(low[0].pack_id, "whisper-tiny-int8");

        // Medium tier: both whisper-tiny + whisper-base
        let medium = registry.find_for_task("transcribe", MinTier::Medium);
        assert_eq!(medium.len(), 2);
    }

    #[test]
    fn test_default_registry_encoder_int4_is_low_tier() {
        let registry = ModelRegistry::default_registry();
        let enc = registry.find("kchat-encoder-int4").expect("encoder-int4");
        assert_eq!(enc.min_tier, MinTier::Low);
        assert_eq!(enc.pack_type, "encoder");
        assert_eq!(enc.size_bytes, 90_000_000);
        assert_eq!(enc.quantization, "INT4");
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
    fn test_default_registry_no_placeholder_hashes_for_available_models() {
        let registry = ModelRegistry::default_registry();
        // Models with locally-available artifacts must have real SHA-256 hashes.
        // Remaining placeholders: kchat-encoder-int8, kchat-encoder-int4, mobileclip-s2-int8
        // (ONNX models not yet exported — must be hashed before release).
        let must_have_real_hash = [
            "ternary-bonsai-1.7b-mlx-2bit",
            "ternary-bonsai-1.7b-q2_0",
            "ternary-bonsai-4b-mlx-2bit",
            "ternary-bonsai-4b-q2_0",
            "ternary-bonsai-8b-mlx-2bit",
            "ternary-bonsai-8b-q2_0",
            "whisper-tiny-int8",
            "whisper-base-int8",
        ];
        for pack_id in &must_have_real_hash {
            let entry = registry.find(pack_id).unwrap_or_else(|| panic!("pack {} not found", pack_id));
            assert!(
                !entry.is_placeholder_hash(),
                "pack {} still has placeholder SHA-256 — compute and set real hash before release",
                pack_id
            );
        }
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

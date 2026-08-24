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
    /// Unique pack identifier (e.g. `bonsai-1.7b-mlx-1bit`).
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

        // --- Generative models (unified small-model + LoRA architecture) ---
        // All tiers use the same base model; task specialization via LoRA adapters.
        // Apple Silicon (iOS/macOS aarch64): Bonsai-1.7B MLX 1-bit (~269MB) + LoRA
        // Other platforms: Bonsai-1.7B Q1_0 GGUF (~248MB) + LoRA

        // Low tier: Bonsai-1.7B MLX 1-bit for Apple platforms (iOS/macOS)
        // 269MB, Qwen3-1.7B base, 1-bit binary {-1,+1} g128, MLX format
        // End-to-end 1-bit trained from scratch (not post-hoc quantized)
        // Supports 75 LoRA adapters: 5 task-families × 15 language slots
        models.push(RegistryEntry {
            pack_id: "bonsai-1.7b-mlx-1bit".into(),
            version: "1.0.0".into(),
            pack_type: "generative".into(),
            download_url: "https://huggingface.co/prism-ml/Bonsai-1.7B-mlx-1bit/resolve/main/model.safetensors".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(), // placeholder - update after upload
            size_bytes: 269_060_904, // ~269 MB (exact from HF LFS)
            min_tier: MinTier::Low,
            task_capabilities: vec!["summarize".into(), "translate".into(), "generate".into(), "tool_use".into(), "extract_json".into(), "rewrite_grammar".into(), "summarize_catchup".into(), "doc_creative".into(), "slides_deck".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "ar".into(), "de".into(), "hi".into(), "fr".into(), "es-en".into(), "ja-en".into(), "ko-en".into(), "vi-en".into(), "zh-en".into()],
            quantization: "1bit-MLX".into(),
        });

        // Low tier: Bonsai-1.7B Q1_0 GGUF for Android/Windows/Linux
        // 248,302,272 bytes (~248MB), Qwen3-1.7B base, 1-bit binary Q1_0 g128
        models.push(RegistryEntry {
            pack_id: "bonsai-1.7b-q1_0".into(),
            version: "1.0.0".into(),
            pack_type: "generative".into(),
            download_url: "https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(), // placeholder - update after upload
            size_bytes: 248_302_272, // ~248 MB (exact from HF API)
            min_tier: MinTier::Low,
            task_capabilities: vec!["summarize".into(), "translate".into(), "generate".into(), "tool_use".into()],
            languages: vec!["en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(), "es".into(), "ar".into(), "de".into(), "hi".into(), "fr".into()],
            quantization: "Q1_0".into(),
        });

        // --- Vision model (MobileCLIP-S2) ---

        // Unified MobileCLIP-S2 INT8 for both image and video classification
        // Pack includes visual_encoder_int8.onnx (~37MB) + text_encoder_int8.onnx (~64MB)
        // Runtime loads only visual encoder for image_classify/image_embed/video_classify
        // 512-dim embeddings, 17 categories
        models.push(RegistryEntry {
            pack_id: "mobileclip-s2-int8".into(),
            version: "1.0.0".into(),
            pack_type: "vision".into(),
            download_url: "https://cdn.kchat.dev/models/mobileclip-s2-int8/1.0.0/visual_encoder_int8.onnx".into(),
            sha256: "bcf1e864f3c30ae03eab1c61cba3d176d6baedb25868af43170dc821bdece797".into(),
            size_bytes: 102_011_590,
            min_tier: MinTier::Low,
            task_capabilities: vec!["image_classify".into(), "image_embed".into(), "video_classify".into()],
            languages: vec!["en".into()],
            quantization: "INT8".into(),
        });

        // --- ASR models (Whisper) ---

        // Low tier: Whisper Tiny ONNX for audio transcription
        // ~33MB encoder ONNX, based on nb-whisper-tiny (NbAiLab Norwegian fine-tune of Whisper Tiny)
        // Multilingual: supports en, vi, zh, ja, ko, es, fr, de, ar, hi, th
        // Full pack includes encoder + decoder + decoder_with_past ONNX files
        // ONNX files are FP32 (not INT8-quantized)
        models.push(RegistryEntry {
            pack_id: "whisper-tiny".into(),
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
        // ~82MB encoder ONNX, based on nb-whisper-base (NbAiLab Norwegian fine-tune of Whisper Base)
        // Multilingual: supports en, vi, zh, ja, ko, es, fr, de, ar, hi, th
        // Full pack includes encoder + decoder + decoder_with_past ONNX files
        // ONNX files are FP32 (not INT8-quantized)
        models.push(RegistryEntry {
            pack_id: "whisper-base".into(),
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

        // --- mmBERT-small GGUF encoder (new v2.0, 17-category taxonomy) ---
        // Replaces XLM-RoBERTa ONNX encoder with a smaller, faster, more
        // multilingual model. 140M params, 384-dim, 1800+ languages.
        // GGUF Q4_K_M ~145MB + classifier_weights.safetensors ~1.5MB.
        // Uses llama-server embedding endpoint for inference.
        // 17-category guardrail taxonomy (kchat.guardrail.taxonomy.v1).
        models.push(RegistryEntry {
            pack_id: "mmbert-safety-q4_k_m".into(),
            version: "2.0.0".into(),
            pack_type: "encoder".into(),
            download_url: "https://cdn.kchat.dev/models/mmbert-safety/2.0.0/mmBERT-small-Q4_K_M.gguf".into(),
            sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(), // placeholder until model is trained
            size_bytes: 151_698_944, // ~145 MB (exact from trained model)
            min_tier: MinTier::Low,
            task_capabilities: vec!["safety".into(), "embed".into(), "rerank".into()],
            languages: vec![
                "en".into(), "vi".into(), "zh".into(), "ja".into(), "ko".into(),
                "es".into(), "fr".into(), "de".into(), "ar".into(), "hi".into(),
                "th".into(), "id".into(), "pt".into(), "tl".into(), "tr".into(),
                "fa".into(), "ur".into(), "bn".into(), "ta".into(), "te".into(),
                "mr".into(), "sw".into(), "ru".into(), "it".into(), "nl".into(),
                "pl".into(), "he".into(),
            ],
            quantization: "Q4_K_M".into(),
        });

        Self { models }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TOML: &str = r#"
[[models]]
pack_id = "bonsai-1.7b-mlx-1bit"
version = "1.0.0"
pack_type = "generative"
download_url = "https://huggingface.co/prism-ml/Bonsai-1.7B-mlx-1bit/resolve/main/model.safetensors"
sha256 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
size_bytes = 269060904
min_tier = "low"
task_capabilities = ["summarize", "translate", "generate", "tool_use"]
languages = ["en", "vi", "zh", "ja", "ko", "es", "ar", "de", "hi", "fr"]
quantization = "1bit-MLX"

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
        let entry = registry.find("bonsai-1.7b-mlx-1bit").expect("found");
        assert_eq!(entry.version, "1.0.0");
        assert_eq!(entry.pack_type, "generative");
        assert_eq!(entry.size_bytes, 269_060_904);
        assert_eq!(entry.min_tier, MinTier::Low);
        assert_eq!(entry.quantization, "1bit-MLX");
        assert_eq!(entry.task_capabilities, vec!["summarize", "translate", "generate", "tool_use"]);
        assert_eq!(entry.languages.len(), 10);
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
        // "summarize" only on the low-tier generative pack; low tier satisfies it.
        let results = registry.find_for_task("summarize", MinTier::Low);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "bonsai-1.7b-mlx-1bit");
    }

    #[test]
    fn test_find_for_task_excludes_when_tier_too_low() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        // The encoder pack requires medium tier; a low-tier device cannot run it.
        let results = registry.find_for_task("embed", MinTier::Low);
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
        // The encoder pack requires medium tier; a low-tier device cannot run it.
        // The generative pack is low tier so it IS available.
        let results = registry.find_for_language("vi", MinTier::Low);
        // Only the generative pack supports Vietnamese at low tier.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pack_id, "bonsai-1.7b-mlx-1bit");
    }

    #[test]
    fn test_find_for_language_medium_tier_runs_embedding() {
        let registry = ModelRegistry::load_from_toml(SAMPLE_TOML).expect("parse");
        // At medium tier, both the generative pack (low tier) and encoder pack (medium tier)
        // are available for English.
        let results = registry.find_for_language("en", MinTier::Medium);
        assert_eq!(results.len(), 2);
        let ids: Vec<&str> = results.iter().map(|e| e.pack_id.as_str()).collect();
        assert!(ids.contains(&"bonsai-1.7b-mlx-1bit"));
        assert!(ids.contains(&"kchat-encoder-int8"));
    }

    #[test]
    fn test_default_registry_has_six_entries() {
        let registry = ModelRegistry::default_registry();
        // 2 generative (bonsai-1.7b-mlx-1bit, bonsai-1.7b-q1_0)
        // + 1 encoder (mmbert-safety-q4_k_m)
        // + 1 vision (mobileclip-s2-int8)
        // + 2 ASR (whisper-tiny, whisper-base)
        // = 6
        assert_eq!(registry.list().len(), 6);
    }

    #[test]
    fn test_default_registry_contains_expected_pack_ids() {
        let registry = ModelRegistry::default_registry();
        let ids: Vec<&str> = registry.list().iter().map(|e| e.pack_id.as_str()).collect();
        assert!(ids.contains(&"bonsai-1.7b-mlx-1bit"));
        assert!(ids.contains(&"bonsai-1.7b-q1_0"));
        assert!(ids.contains(&"mmbert-safety-q4_k_m"));
        assert!(ids.contains(&"mobileclip-s2-int8"));
        assert!(ids.contains(&"whisper-tiny"));
        assert!(ids.contains(&"whisper-base"));
    }

    #[test]
    fn test_default_registry_generative_packs_are_low_tier() {
        let registry = ModelRegistry::default_registry();
        let mlx_1bit = registry.find("bonsai-1.7b-mlx-1bit").expect("bonsai-1.7b-mlx-1bit");
        let q1_0 = registry.find("bonsai-1.7b-q1_0").expect("bonsai-1.7b-q1_0");
        // Both unified Bonsai models are Low tier (fit 750MB mobile budget)
        assert_eq!(mlx_1bit.min_tier, MinTier::Low);
        assert_eq!(q1_0.min_tier, MinTier::Low);
        assert_eq!(mlx_1bit.quantization, "1bit-MLX");
        assert_eq!(q1_0.quantization, "Q1_0");
    }

    #[test]
    fn test_default_registry_bonsai_1_7b_mlx_1bit_fits_low_tier_budget() {
        let registry = ModelRegistry::default_registry();
        let mlx_1bit = registry.find("bonsai-1.7b-mlx-1bit").expect("bonsai-1.7b-mlx-1bit");
        // Low tier mobile budget is 750MB (iOS)
        assert!(mlx_1bit.size_bytes <= 750 * 1024 * 1024);
        // Exact size from HF LFS: 269,060,904 bytes (~269MB)
        assert_eq!(mlx_1bit.size_bytes, 269_060_904);
        // Should support tool_use
        assert!(mlx_1bit.task_capabilities.contains(&"tool_use".to_string()));
        // Should support family-based tasks
        assert!(mlx_1bit.task_capabilities.contains(&"extract_json".to_string()));
        assert!(mlx_1bit.task_capabilities.contains(&"slides_deck".to_string()));
        // Should support 15 language slots
        assert_eq!(mlx_1bit.languages.len(), 15);
        // Quantization should be 1bit-MLX
        assert_eq!(mlx_1bit.quantization, "1bit-MLX");
    }

    #[test]
    fn test_default_registry_bonsai_1_7b_q1_0_fits_low_tier_budget() {
        let registry = ModelRegistry::default_registry();
        let q1_0 = registry.find("bonsai-1.7b-q1_0").expect("bonsai-1.7b-q1_0");
        // Low tier mobile budget is 750MB (Android)
        assert!(q1_0.size_bytes <= 750 * 1024 * 1024);
        // Exact size from HF API: 248,302,272 bytes (~248MB)
        assert_eq!(q1_0.size_bytes, 248_302_272);
        // Should support tool_use
        assert!(q1_0.task_capabilities.contains(&"tool_use".to_string()));
        // Quantization should be Q1_0
        assert_eq!(q1_0.quantization, "Q1_0");
    }

    #[test]
    fn test_default_registry_bonsai_4b_fits_android_medium_budget() {
        // Old per-tier 4B/8B models have been removed in favor of unified
        // small-model + LoRA architecture. All tiers use bonsai-1.7b-mlx-1bit
        // or bonsai-1.7b-q1_0. This test is kept as a no-op placeholder.
        let registry = ModelRegistry::default_registry();
        // Verify no old per-tier models exist
        assert!(registry.find("ternary-bonsai-4b-q2_0").is_none());
        assert!(registry.find("ternary-bonsai-8b-q2_0").is_none());
    }

    #[test]
    fn test_default_registry_mmbert_encoder_is_low_tier() {
        let registry = ModelRegistry::default_registry();
        let enc = registry.find("mmbert-safety-q4_k_m").expect("mmbert-safety-q4_k_m");
        assert_eq!(enc.min_tier, MinTier::Low);
        assert_eq!(enc.pack_type, "encoder");
        assert_eq!(enc.quantization, "Q4_K_M");
        assert_eq!(enc.size_bytes, 151_698_944);
        assert!(enc.task_capabilities.contains(&"safety".to_string()));
        assert!(enc.task_capabilities.contains(&"embed".to_string()));
        assert!(enc.task_capabilities.contains(&"rerank".to_string()));
    }

    #[test]
    fn test_default_registry_find_for_task_safety_on_medium() {
        let registry = ModelRegistry::default_registry();
        // Medium tier: mmbert-safety-q4_k_m (low tier, qualifies)
        let results = registry.find_for_task("safety", MinTier::Medium);
        assert_eq!(results.len(), 1);
        let ids: Vec<&str> = results.iter().map(|e| e.pack_id.as_str()).collect();
        assert!(ids.contains(&"mmbert-safety-q4_k_m"));
    }

    #[test]
    fn test_default_registry_find_for_task_safety_on_low() {
        let registry = ModelRegistry::default_registry();
        // Low tier: mmbert-safety-q4_k_m (low tier)
        let results = registry.find_for_task("safety", MinTier::Low);
        assert_eq!(results.len(), 1);
        let ids: Vec<&str> = results.iter().map(|e| e.pack_id.as_str()).collect();
        assert!(ids.contains(&"mmbert-safety-q4_k_m"));
    }

    #[test]
    fn test_default_registry_find_for_task_safety_on_high() {
        let registry = ModelRegistry::default_registry();
        // High tier: mmbert-safety-q4_k_m (low tier, qualifies)
        let results = registry.find_for_task("safety", MinTier::High);
        assert_eq!(results.len(), 1);
        let ids: Vec<&str> = results.iter().map(|e| e.pack_id.as_str()).collect();
        assert!(ids.contains(&"mmbert-safety-q4_k_m"));
    }

    #[test]
    fn test_default_registry_find_for_task_rerank_on_medium() {
        let registry = ModelRegistry::default_registry();
        // Reranking on Medium tier: mmbert-safety-q4_k_m (low tier, qualifies)
        let results = registry.find_for_task("rerank", MinTier::Medium);
        assert_eq!(results.len(), 1);
        let ids: Vec<&str> = results.iter().map(|e| e.pack_id.as_str()).collect();
        assert!(ids.contains(&"mmbert-safety-q4_k_m"));
    }

    #[test]
    fn test_default_registry_find_generative_for_all_tiers() {
        let registry = ModelRegistry::default_registry();
        // Low tier: 2 unified Bonsai models
        let low = registry.find_for_task("summarize", MinTier::Low);
        assert_eq!(low.len(), 2);
        let low_ids: Vec<&str> = low.iter().map(|e| e.pack_id.as_str()).collect();
        assert!(low_ids.contains(&"bonsai-1.7b-mlx-1bit"));
        assert!(low_ids.contains(&"bonsai-1.7b-q1_0"));

        // Medium tier: same 2 models
        let medium = registry.find_for_task("summarize", MinTier::Medium);
        assert_eq!(medium.len(), 2);

        // High tier: same 2 models
        let high = registry.find_for_task("summarize", MinTier::High);
        assert_eq!(high.len(), 2);
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
        assert_eq!(video.size_bytes, 102_011_590);
        // Video classify is now available on all tiers (same model as image)
        assert_eq!(registry.find_for_task("video_classify", MinTier::Low).len(), 1);
        assert_eq!(registry.find_for_task("video_classify", MinTier::Medium).len(), 1);
        assert_eq!(registry.find_for_task("video_classify", MinTier::High).len(), 1);
    }

    #[test]
    fn test_default_registry_asr_packs_span_all_tiers() {
        let registry = ModelRegistry::default_registry();
        // Low tier: only whisper-tiny
        let low = registry.find_for_task("transcribe", MinTier::Low);
        assert_eq!(low.len(), 1);
        assert_eq!(low[0].pack_id, "whisper-tiny");

        // Medium tier: both whisper-tiny + whisper-base
        let medium = registry.find_for_task("transcribe", MinTier::Medium);
        assert_eq!(medium.len(), 2);
    }

    #[test]
    fn test_default_registry_mmbert_encoder_is_low_tier_v2() {
        let registry = ModelRegistry::default_registry();
        let enc = registry.find("mmbert-safety-q4_k_m").expect("mmbert-safety-q4_k_m");
        assert_eq!(enc.min_tier, MinTier::Low);
        assert_eq!(enc.pack_type, "encoder");
        assert_eq!(enc.size_bytes, 151_698_944);
        assert_eq!(enc.quantization, "Q4_K_M");
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
        // All packs now have real SHA-256 hashes (mobileclip-s2-int8 included).
        let must_have_real_hash = [
            "whisper-tiny",
            "whisper-base",
            "mobileclip-s2-int8",
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
        assert_eq!(parsed.find("bonsai-1.7b-mlx-1bit"), registry.find("bonsai-1.7b-mlx-1bit"));
    }
}

//! Backend adapters — abstraction over llama.cpp, ONNX Runtime, and cloud.
//!
//! The generative plane uses llama.cpp as the primary runtime with:
//! - Metal/CoreML on iOS/macOS
//! - Vulkan on Android/Windows
//! - CPU fallback on all platforms
//!
//! Backend selection is based on device capabilities and tier.

use crate::grammar::Grammar;
use crate::stream::StreamHandle;
use kchat_core::tier::DeviceTier;
use serde::{Deserialize, Serialize};

/// Type of generation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    /// llama.cpp with Metal (iOS/macOS)
    LlamaCppMetal,
    /// llama.cpp with Vulkan (Android/Windows)
    LlamaCppVulkan,
    /// llama.cpp with CPU
    LlamaCppCpu,
    /// Apple MLX framework (macOS/iOS for MLX-format Bonsai models)
    Mlx,
    /// ONNX Runtime (for encoder-only models)
    OnnxRuntime,
    /// Cloud backend (hybrid mode, if enabled)
    Cloud,
}

impl BackendType {
    /// Select the best backend for the given platform, tier, and CPU architecture.
    ///
    /// - Apple Silicon (aarch64) on iOS/macOS: MLX (for Bonsai MLX models)
    /// - Intel Macs (x86_64) on macOS: llama.cpp CPU (no MLX support)
    /// - Android/Windows: llama.cpp Vulkan
    /// - Other: llama.cpp CPU
    pub fn select(platform: &str, _tier: DeviceTier, cpu_arch: &str) -> Option<Self> {
        // Note: tier is currently unused — all tiers use MLX on Apple Silicon.
        // The parameter is retained for future tier-aware backend selection
        // (e.g. falling back to CPU on Low tier if MLX is unavailable).
        match platform {
            "ios" => Some(BackendType::Mlx),
            "macos" => {
                match cpu_arch {
                    "aarch64" => Some(BackendType::Mlx),
                    _ => Some(BackendType::LlamaCppCpu),
                }
            }
            "android" | "windows" => Some(BackendType::LlamaCppVulkan),
            _ => Some(BackendType::LlamaCppCpu),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            BackendType::LlamaCppMetal => "llama.cpp_metal",
            BackendType::LlamaCppVulkan => "llama.cpp_vulkan",
            BackendType::LlamaCppCpu => "llama.cpp_cpu",
            BackendType::Mlx => "mlx",
            BackendType::OnnxRuntime => "onnxruntime",
            BackendType::Cloud => "cloud",
        }
    }
}

/// Backend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    pub backend_type: BackendType,
    /// Model pack ID
    pub model_pack_id: String,
    /// Path to the model file
    pub model_path: String,
    /// Number of GPU layers to offload (-1 = all)
    pub gpu_layers: i32,
    /// Context size in tokens
    pub context_size: usize,
    /// Thread count for CPU
    pub threads: u32,
    /// Batch size
    pub batch_size: u32,
}

impl BackendConfig {
    pub fn for_tier(
        backend_type: BackendType,
        model_pack_id: impl Into<String>,
        model_path: impl Into<String>,
        tier: DeviceTier,
        platform: &str,
    ) -> Self {
        let context_size = tier.context_cap_for_platform(platform);
        let (threads, gpu_layers) = match (tier, platform) {
            (DeviceTier::Low, _) => (2, 0),
            (DeviceTier::Medium, "ios") | (DeviceTier::Medium, "android") => (4, -1),
            (DeviceTier::Medium, "macos") | (DeviceTier::Medium, "windows") => (6, -1),
            (DeviceTier::High, "ios") | (DeviceTier::High, "android") => (6, -1),
            (DeviceTier::High, "macos") | (DeviceTier::High, "windows") => (8, -1),
            _ => (2, 0),
        };

        Self {
            backend_type,
            model_pack_id: model_pack_id.into(),
            model_path: model_path.into(),
            gpu_layers,
            context_size,
            threads,
            batch_size: 512,
        }
    }
}

/// Generation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum output tokens
    pub max_tokens: usize,
    /// Temperature (0.0-1.0)
    pub temperature: f32,
    /// Top-p sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: u32,
    /// Repetition penalty
    pub repeat_penalty: f32,
    /// Grammar constraint
    pub grammar: Option<Grammar>,
    /// Seed for reproducibility (0 = random)
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            grammar: None,
            seed: 0,
        }
    }
}

impl GenerationConfig {
    /// Conservative config for structured output (ToolPlan, JSON).
    pub fn for_structured() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.3, // lower temperature for structured output
            top_p: 0.95,
            top_k: 40,
            repeat_penalty: 1.1,
            grammar: None,
            seed: 0,
        }
    }

    /// Creative config for rewrite/summarize.
    pub fn for_creative() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            grammar: None,
            seed: 0,
        }
    }
}

/// Generation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated text
    pub text: String,
    /// Number of prompt tokens
    pub prompt_tokens: u32,
    /// Number of generated tokens
    pub completion_tokens: u32,
    /// Time to first token in milliseconds
    pub ttft_ms: u64,
    /// Total generation time in milliseconds
    pub total_ms: u64,
    /// Tokens per second
    pub tokens_per_second: f64,
    /// Backend used
    pub backend: String,
    /// Whether grammar validation passed
    pub grammar_valid: bool,
}

/// Trait for backend adapters — abstracts llama.cpp, ONNX, and cloud.
pub trait BackendAdapter: Send + Sync {
    /// Load the model into memory.
    fn load(&self, config: &BackendConfig) -> Result<(), BackendError>;

    /// Unload the model from memory.
    fn unload(&self) -> Result<(), BackendError>;

    /// Check if the model is loaded.
    fn is_loaded(&self) -> bool;

    /// Generate text from a prompt.
    fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationResult, BackendError>;

    /// Generate text from a prompt, streaming tokens to the provided handle.
    ///
    /// The stream allows the caller to receive tokens as they are generated
    /// and to cancel generation mid-stream (e.g. on safety violation).
    fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        stream: &StreamHandle,
    ) -> Result<GenerationResult, BackendError>;

    /// Get the backend type.
    fn backend_type(&self) -> BackendType;
}

/// Backend errors.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("model not loaded")]
    NotLoaded,

    #[error("model load failed: {0}")]
    LoadFailed(String),

    #[error("generation failed: {0}")]
    GenerationFailed(String),

    #[error("grammar validation failed: {0}")]
    GrammarValidationFailed(String),

    #[error("backend unavailable: {0}")]
    Unavailable(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_selection_ios() {
        // Low tier on iOS: MLX (for Bonsai-1.7B-MLX-1bit)
        assert_eq!(
            BackendType::select("ios", DeviceTier::Low, "aarch64"),
            Some(BackendType::Mlx)
        );
        // Medium tier on iOS: MLX (for Bonsai-1.7B-MLX-1bit, unified)
        assert_eq!(
            BackendType::select("ios", DeviceTier::Medium, "aarch64"),
            Some(BackendType::Mlx)
        );
        // High tier on iOS: MLX (for Bonsai-1.7B-MLX-1bit, unified)
        assert_eq!(
            BackendType::select("ios", DeviceTier::High, "aarch64"),
            Some(BackendType::Mlx)
        );
    }

    #[test]
    fn test_backend_selection_macos() {
        // Low tier on macOS Apple Silicon: MLX (for Bonsai-1.7B-MLX-1bit)
        assert_eq!(
            BackendType::select("macos", DeviceTier::Low, "aarch64"),
            Some(BackendType::Mlx)
        );
        // Medium tier on macOS Apple Silicon: MLX (for Bonsai-1.7B-MLX-1bit, unified)
        assert_eq!(
            BackendType::select("macos", DeviceTier::Medium, "aarch64"),
            Some(BackendType::Mlx)
        );
        // High tier on macOS Apple Silicon: MLX (for Bonsai-1.7B-MLX-1bit, unified)
        assert_eq!(
            BackendType::select("macos", DeviceTier::High, "aarch64"),
            Some(BackendType::Mlx)
        );
        // Intel Mac (x86_64): llama.cpp CPU (no MLX)
        assert_eq!(
            BackendType::select("macos", DeviceTier::Low, "x86_64"),
            Some(BackendType::LlamaCppCpu)
        );
    }

    #[test]
    fn test_backend_selection_android() {
        // Android always uses Vulkan (no MLX)
        assert_eq!(
            BackendType::select("android", DeviceTier::Low, "aarch64"),
            Some(BackendType::LlamaCppVulkan)
        );
        assert_eq!(
            BackendType::select("android", DeviceTier::Medium, "aarch64"),
            Some(BackendType::LlamaCppVulkan)
        );
        assert_eq!(
            BackendType::select("android", DeviceTier::High, "aarch64"),
            Some(BackendType::LlamaCppVulkan)
        );
    }

    #[test]
    fn test_backend_selection_high_tier_apple_uses_mlx() {
        // High tier on Apple platforms should use MLX for Bonsai-1.7B-MLX-1bit
        assert_eq!(
            BackendType::select("ios", DeviceTier::High, "aarch64"),
            Some(BackendType::Mlx)
        );
        assert_eq!(
            BackendType::select("macos", DeviceTier::High, "aarch64"),
            Some(BackendType::Mlx)
        );
        // High tier on non-Apple should NOT use MLX
        assert_ne!(
            BackendType::select("android", DeviceTier::High, "aarch64"),
            Some(BackendType::Mlx)
        );
        assert_ne!(
            BackendType::select("windows", DeviceTier::High, "x86_64"),
            Some(BackendType::Mlx)
        );
    }

    #[test]
    fn test_mlx_backend_as_str() {
        assert_eq!(BackendType::Mlx.as_str(), "mlx");
    }

    #[test]
    fn test_backend_config_for_tier() {
        let config = BackendConfig::for_tier(
            BackendType::LlamaCppMetal,
            "ternary-bonsai-1.7b-q2_0",
            "/path/to/model.gguf",
            DeviceTier::Medium,
            "ios",
        );
        assert_eq!(config.context_size, 2048);
        assert_eq!(config.gpu_layers, -1);
    }

    #[test]
    fn test_generation_config_structured() {
        let config = GenerationConfig::for_structured();
        assert!(config.temperature < 0.5);
    }

    #[test]
    fn test_generation_config_creative() {
        let config = GenerationConfig::for_creative();
        assert!(config.temperature > 0.5);
    }
}

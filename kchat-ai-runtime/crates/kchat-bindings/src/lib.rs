//! kchat-bindings: FFI bindings for Swift/Kotlin (UniFFI) and Node.js (N-API).
//!
//! This crate exposes the kchat-ai-runtime API surface to platform code:
//! - Mobile (iOS/Android): UniFFI generates Swift and Kotlin bindings
//! - Desktop (macOS/Windows): N-API generates Node.js native addons
//!
//! The public API surface is intentionally minimal:
//! - Safety classification (deterministic + encoder + SLM)
//! - Context retrieval (FTS + hybrid)
//! - Generation (grammar-constrained, streaming)
//! - Action validation (ToolPlan, artifact AST)
//! - Device capability probe and tier selection
//! - Model lifecycle management

// The actual FFI surface is defined here. UniFFI uses inline UDL (procmacro)
// and N-API uses the #[napi] attribute macro.

// ============================================================================
// Common types exposed across all platforms
// ============================================================================

use serde::{Deserialize, Serialize};

/// Device tier — returned by the capability probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FfiDeviceTier {
    Low,
    Medium,
    High,
}

impl From<kchat_core::tier::DeviceTier> for FfiDeviceTier {
    fn from(tier: kchat_core::tier::DeviceTier) -> Self {
        match tier {
            kchat_core::tier::DeviceTier::Low => FfiDeviceTier::Low,
            kchat_core::tier::DeviceTier::Medium => FfiDeviceTier::Medium,
            kchat_core::tier::DeviceTier::High => FfiDeviceTier::High,
        }
    }
}

/// Safety verdict action — returned by the safety classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfiSafetyAction {
    Allow,
    Warn,
    Block,
    Redact,
    RequireConsent,
}

impl From<kchat_safety::verdict::Action> for FfiSafetyAction {
    fn from(action: kchat_safety::verdict::Action) -> Self {
        match action {
            kchat_safety::verdict::Action::Allow => FfiSafetyAction::Allow,
            kchat_safety::verdict::Action::Warn => FfiSafetyAction::Warn,
            kchat_safety::verdict::Action::Block => FfiSafetyAction::Block,
            kchat_safety::verdict::Action::Redact => FfiSafetyAction::Redact,
            kchat_safety::verdict::Action::RequireConsent => FfiSafetyAction::RequireConsent,
        }
    }
}

/// Safety classification result — the main FFI return type for safety.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiSafetyResult {
    pub action: FfiSafetyAction,
    pub severity: u8,
    pub category: u32,
    pub confidence: f64,
    pub reason_codes: Vec<String>,
    pub used_encoder: bool,
    pub used_slm: bool,
    pub duration_us: u64,
}

impl From<kchat_safety::classify::ClassifyResult> for FfiSafetyResult {
    fn from(result: kchat_safety::classify::ClassifyResult) -> Self {
        let v = &result.verdict;
        Self {
            action: v.action.into(),
            severity: v.severity.0,
            category: v.category,
            confidence: v.confidence,
            reason_codes: v.reason_codes.clone(),
            used_encoder: v.used_encoder,
            used_slm: v.used_slm,
            duration_us: result.duration_us,
        }
    }
}

/// Retrieval result — the main FFI return type for context retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiRetrievalResult {
    pub evidence_id: String,
    pub score: f64,
    pub fts_score: f64,
    pub recency_score: f64,
}

/// Generation result — the main FFI return type for generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiGenerationResult {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub ttft_ms: u64,
    pub total_ms: u64,
    pub tokens_per_second: f64,
    pub grammar_valid: bool,
}

impl From<kchat_generation::backend::GenerationResult> for FfiGenerationResult {
    fn from(r: kchat_generation::backend::GenerationResult) -> Self {
        Self {
            text: r.text,
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            ttft_ms: r.ttft_ms,
            total_ms: r.total_ms,
            tokens_per_second: r.tokens_per_second,
            grammar_valid: r.grammar_valid,
        }
    }
}

/// Tool plan validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiValidationResult {
    pub valid: bool,
    pub step_count: usize,
    pub error: Option<String>,
}

// ============================================================================
// High-level FFI facade — the main entry point for platform code
// ============================================================================

/// The high-level KChat AI Runtime facade.
///
/// This struct is exposed via UniFFI (Swift/Kotlin) and N-API (Node.js).
/// Platform code creates an instance, configures it, and calls the methods.
pub struct KChatAiRuntime {
    safety: kchat_safety::classify::SafetyClassifier,
    tier: kchat_core::tier::DeviceTier,
    platform: String,
}

impl KChatAiRuntime {
    /// Create a new runtime instance for the given platform.
    pub fn new(platform: &str) -> Self {
        // In production, this would probe device capabilities
        let tier = kchat_core::tier::DeviceTier::Medium; // default
        Self {
            safety: kchat_safety::classify::SafetyClassifier::new(),
            tier,
            platform: platform.to_string(),
        }
    }

    /// Classify a message for safety.
    pub fn classify_safety(&self, text: &str, is_group: bool) -> FfiSafetyResult {
        let request = kchat_safety::classify::ClassifyRequest {
            text: text.to_string(),
            is_group,
            age_mode: None,
            relationship: None,
            encoder_available: self.tier != kchat_core::tier::DeviceTier::Low,
            slm_available: self.tier != kchat_core::tier::DeviceTier::Low,
        };
        self.safety.classify(&request).into()
    }

    /// Get the current device tier.
    pub fn device_tier(&self) -> FfiDeviceTier {
        self.tier.into()
    }

    /// Get the platform name.
    pub fn platform(&self) -> &str {
        &self.platform
    }

    /// Check if generative AI is available on this device.
    pub fn can_generate(&self) -> bool {
        self.tier != kchat_core::tier::DeviceTier::Low
    }
}

// ============================================================================
// UniFFI bindings (mobile: iOS/Android)
// ============================================================================

#[cfg(feature = "mobile")]
uniffi::setup_scaffolding!();

#[cfg(feature = "mobile")]
mod uniffi_bindings {
    use super::*;

    // UniFFI uses the `#[uniffi::export]` macro to generate bindings.
    // The callback interface allows platform code to provide implementations.

    /// Callback for platform-specific device capability probing.
    #[uniffi::export(callback_interface)]
    pub trait DeviceCapabilityProbe: Send + Sync {
        fn get_platform(&self) -> String;
        fn get_physical_memory(&self) -> u64;
        fn get_safe_allocatable_memory(&self) -> u64;
        fn get_cpu_arch(&self) -> String;
        fn get_cpu_cores(&self) -> u32;
        fn get_gpu_backend(&self) -> String;
        fn get_npu_provider(&self) -> String;
        fn get_thermal_state(&self) -> u8;
        fn get_battery_level(&self) -> Option<u8>;
        fn is_on_charger(&self) -> bool;
        fn get_app_state(&self) -> u8;
    }

    #[uniffi::export]
    impl KChatAiRuntime {
        /// Create a new runtime for the given platform.
        #[uniffi::constructor]
        pub fn uniffi_new(platform: String) -> Self {
            KChatAiRuntime::new(&platform)
        }

        /// Classify a message for safety.
        pub fn uniffi_classify_safety(&self, text: String, is_group: bool) -> FfiSafetyResult {
            self.classify_safety(&text, is_group)
        }

        /// Get the device tier.
        pub fn uniffi_device_tier(&self) -> FfiDeviceTier {
            self.device_tier()
        }

        /// Check if generation is available.
        pub fn uniffi_can_generate(&self) -> bool {
            self.can_generate()
        }
    }
}

// ============================================================================
// N-API bindings (desktop: macOS/Windows)
// ============================================================================

#[cfg(feature = "desktop")]
mod napi_bindings {
    use napi_derive::napi;

    use super::*;

    #[napi]
    pub struct KChatAiRuntimeNapi {
        inner: KChatAiRuntime,
    }

    #[napi]
    impl KChatAiRuntimeNapi {
        #[napi(constructor)]
        pub fn new(platform: String) -> Self {
            Self {
                inner: KChatAiRuntime::new(&platform),
            }
        }

        #[napi]
        pub fn classify_safety(&self, text: String, is_group: bool) -> FfiSafetyResult {
            self.inner.classify_safety(&text, is_group)
        }

        #[napi]
        pub fn device_tier(&self) -> FfiDeviceTier {
            self.inner.device_tier()
        }

        #[napi]
        pub fn can_generate(&self) -> bool {
            self.inner.can_generate()
        }

        #[napi]
        pub fn platform(&self) -> String {
            self.inner.platform().to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_creation() {
        let runtime = KChatAiRuntime::new("ios");
        assert_eq!(runtime.platform(), "ios");
        assert!(runtime.can_generate()); // default medium tier
    }

    #[test]
    fn test_classify_safety() {
        let runtime = KChatAiRuntime::new("ios");
        let result = runtime.classify_safety("Hello world", false);
        assert_eq!(result.action, FfiSafetyAction::Allow);
    }

    #[test]
    fn test_classify_safety_blocks_pii() {
        let runtime = KChatAiRuntime::new("ios");
        let result = runtime.classify_safety("my card is 4111 1111 1111 1111", false);
        assert_eq!(result.action, FfiSafetyAction::Redact);
    }

    #[test]
    fn test_device_tier_conversion() {
        let tier = kchat_core::tier::DeviceTier::High;
        let ffi_tier: FfiDeviceTier = tier.into();
        assert_eq!(ffi_tier, FfiDeviceTier::High);
    }
}

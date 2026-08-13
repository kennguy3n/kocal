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
    /// Cached device capabilities from the last probe
    caps: Option<kchat_core::capability::DeviceCapabilities>,
}

impl KChatAiRuntime {
    /// Create a new runtime instance for the given platform.
    ///
    /// In production, this probes device capabilities and selects a tier.
    /// If the probe fails, defaults to Low tier for safety (avoid overloading
    /// an unknown device).
    pub fn new(platform: &str) -> Self {
        // Probe real device capabilities
        match kchat_core::capability::CapabilityProbe::probe() {
            Ok(caps) => {
                let tier = select_tier(&caps);
                Self {
                    safety: kchat_safety::classify::SafetyClassifier::new(),
                    tier,
                    platform: platform.to_string(),
                    caps: Some(caps),
                }
            }
            Err(e) => {
                // Fail-safe: default to Low tier when hardware detection fails
                tracing::warn!("Capability probe failed, defaulting to Low tier: {}", e);
                Self {
                    safety: kchat_safety::classify::SafetyClassifier::new(),
                    tier: kchat_core::tier::DeviceTier::Low,
                    platform: platform.to_string(),
                    caps: None,
                }
            }
        }
    }

    /// Create a runtime with an explicit tier (for testing).
    pub fn with_tier(platform: &str, tier: kchat_core::tier::DeviceTier) -> Self {
        Self {
            safety: kchat_safety::classify::SafetyClassifier::new(),
            tier,
            platform: platform.to_string(),
            caps: None,
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
            quoted_from_user: false,
            community_overlay_id: None,
            jurisdiction: None,
            locale: None,
            media_descriptors: Vec::new(),
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
    ///
    /// All tiers now have tier-appropriate generative models:
    /// - Low: 1.7B Q2_0 (~442MB)
    /// - Medium: 4B Q2_0 (~1.0GB)
    /// - High: 8B Q2_0 (~2.1GB)
    pub fn can_generate(&self) -> bool {
        true
    }

    /// Probe device capabilities (real OS API calls).
    /// Re-evaluates dynamic state (thermal, battery, app state) on each call
    /// to avoid returning stale data on long-lived runtime instances.
    pub fn probe_capabilities(&self) -> FfiDeviceCapabilities {
        if let Some(caps) = &self.caps {
            // Clone and re-evaluate dynamic state (thermal, battery, app state)
            let mut refreshed = caps.clone();
            kchat_core::capability::CapabilityProbe::re_evaluate(&mut refreshed);
            (&refreshed).into()
        } else {
            kchat_core::capability::CapabilityProbe::probe()
                .map(|c| (&c).into())
                .unwrap_or_default()
        }
    }

    /// Get the safe AI memory budget in bytes.
    pub fn safe_ai_budget(&self) -> u64 {
        self.caps.as_ref().map(|c| c.safe_ai_budget()).unwrap_or(0)
    }

    /// Check if the device allows generative inference (thermal + app state).
    pub fn allows_generative(&self) -> bool {
        self.caps.as_ref().map(|c| c.allows_generative()).unwrap_or(false)
    }
}

/// Select device tier based on capabilities.
/// Thermal state is checked at every level to prevent overheating.
fn select_tier(caps: &kchat_core::capability::DeviceCapabilities) -> kchat_core::tier::DeviceTier {
    use kchat_core::capability::ThermalState;
    use kchat_core::tier::DeviceTier;

    // Critical thermal → always Low
    if matches!(caps.thermal_state, ThermalState::Critical) {
        return DeviceTier::Low;
    }

    // Memory-based selection, with thermal downgrade at each tier
    let budget = caps.safe_ai_budget();
    let high_allowed = caps.thermal_state == ThermalState::Nominal;
    // Serious/Fair thermal → cap at Medium
    let medium_allowed = matches!(
        caps.thermal_state,
        ThermalState::Nominal | ThermalState::Fair | ThermalState::Serious
    );

    match caps.platform.as_str() {
        "ios" | "android" => {
            if high_allowed && budget >= 4 * 1024 * 1024 * 1024 {
                DeviceTier::High
            } else if medium_allowed && budget >= 2 * 1024 * 1024 * 1024 {
                DeviceTier::Medium
            } else {
                DeviceTier::Low
            }
        }
        "macos" | "windows" | "linux" => {
            if high_allowed && budget >= 8 * 1024 * 1024 * 1024 {
                DeviceTier::High
            } else if medium_allowed && budget >= 4 * 1024 * 1024 * 1024 {
                DeviceTier::Medium
            } else {
                DeviceTier::Low
            }
        }
        _ => DeviceTier::Low,
    }
}

/// FFI-friendly device capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FfiDeviceCapabilities {
    pub platform: String,
    pub physical_memory: u64,
    pub safe_allocatable_memory: u64,
    pub cpu_arch: String,
    pub cpu_cores: u32,
    pub performance_cores: Option<u32>,
    pub isa_features: Vec<String>,
    pub gpu_backend: String,
    pub npu_provider: String,
    pub free_storage: u64,
    pub battery_level: Option<u8>,
    pub on_charger: bool,
    pub thermal_state: String,
    pub app_state: String,
    pub unmetered_network: bool,
}

impl From<&kchat_core::capability::DeviceCapabilities> for FfiDeviceCapabilities {
    fn from(c: &kchat_core::capability::DeviceCapabilities) -> Self {
        Self {
            platform: c.platform.clone(),
            physical_memory: c.physical_memory,
            safe_allocatable_memory: c.safe_allocatable_memory,
            cpu_arch: c.cpu_arch.clone(),
            cpu_cores: c.cpu_cores,
            performance_cores: c.performance_cores,
            isa_features: c.isa_features.clone(),
            gpu_backend: format!("{:?}", c.gpu_backend).to_lowercase(),
            npu_provider: format!("{:?}", c.npu_provider).to_lowercase(),
            free_storage: c.free_storage,
            battery_level: c.battery_level,
            on_charger: c.on_charger,
            thermal_state: format!("{:?}", c.thermal_state).to_lowercase(),
            app_state: format!("{:?}", c.app_state).to_lowercase(),
            unmetered_network: c.unmetered_network,
        }
    }
}

impl Default for FfiDeviceCapabilities {
    fn default() -> Self {
        Self {
            platform: "unknown".into(),
            physical_memory: 0,
            safe_allocatable_memory: 0,
            cpu_arch: "unknown".into(),
            cpu_cores: 1,
            performance_cores: None,
            isa_features: vec![],
            gpu_backend: "none".into(),
            npu_provider: "none".into(),
            free_storage: 0,
            battery_level: None,
            on_charger: true,
            thermal_state: "nominal".into(),
            app_state: "foreground".into(),
            unmetered_network: false,
        }
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
    }

    #[test]
    fn test_runtime_with_explicit_tier() {
        let runtime = KChatAiRuntime::with_tier("ios", kchat_core::tier::DeviceTier::Low);
        assert_eq!(runtime.device_tier(), FfiDeviceTier::Low);
        // Low tier now has a generative model (0.3B Q4)
        assert!(runtime.can_generate());
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

    #[test]
    fn test_probe_capabilities() {
        let runtime = KChatAiRuntime::new("macos");
        let caps = runtime.probe_capabilities();
        // On a real device, platform should be detected
        assert!(!caps.platform.is_empty());
        assert!(caps.physical_memory > 0);
        assert!(caps.cpu_cores > 0);
    }

    #[test]
    fn test_probe_capabilities_on_macos() {
        let runtime = KChatAiRuntime::new("macos");
        let caps = runtime.probe_capabilities();
        // macOS should detect Metal GPU
        assert_eq!(caps.gpu_backend, "metal");
        // And Apple NE
        assert_eq!(caps.npu_provider, "applene");
    }

    #[test]
    fn test_safe_ai_budget() {
        let runtime = KChatAiRuntime::new("macos");
        let budget = runtime.safe_ai_budget();
        // On a real device, budget should be > 0
        if runtime.caps.is_some() {
            assert!(budget > 0);
        }
    }

    #[test]
    fn test_allows_generative() {
        let runtime = KChatAiRuntime::new("macos");
        // On a nominal-thermal device in foreground, should allow
        if let Some(caps) = &runtime.caps {
            if caps.thermal_state == kchat_core::capability::ThermalState::Nominal {
                assert!(runtime.allows_generative());
            }
        }
    }

    #[test]
    fn test_select_tier_high() {
        let caps = kchat_core::capability::DeviceCapabilities {
            platform: "macos".into(),
            physical_memory: 32 * 1024 * 1024 * 1024,
            safe_allocatable_memory: 20 * 1024 * 1024 * 1024,
            cpu_arch: "aarch64".into(),
            cpu_cores: 10,
            performance_cores: Some(8),
            isa_features: vec![],
            gpu_backend: kchat_core::capability::GpuBackend::Metal,
            npu_provider: kchat_core::capability::NpuProvider::AppleNe,
            free_storage: 0,
            battery_level: None,
            on_charger: true,
            thermal_state: kchat_core::capability::ThermalState::Nominal,
            app_state: kchat_core::capability::AppState::Foreground,
            unmetered_network: true,
        };
        let tier = select_tier(&caps);
        assert_eq!(tier, kchat_core::tier::DeviceTier::High);
    }

    #[test]
    fn test_select_tier_thermal_critical_forces_low() {
        let caps = kchat_core::capability::DeviceCapabilities {
            platform: "macos".into(),
            physical_memory: 32 * 1024 * 1024 * 1024,
            safe_allocatable_memory: 20 * 1024 * 1024 * 1024,
            cpu_arch: "aarch64".into(),
            cpu_cores: 10,
            performance_cores: Some(8),
            isa_features: vec![],
            gpu_backend: kchat_core::capability::GpuBackend::Metal,
            npu_provider: kchat_core::capability::NpuProvider::AppleNe,
            free_storage: 0,
            battery_level: None,
            on_charger: true,
            thermal_state: kchat_core::capability::ThermalState::Critical,
            app_state: kchat_core::capability::AppState::Foreground,
            unmetered_network: true,
        };
        let tier = select_tier(&caps);
        assert_eq!(tier, kchat_core::tier::DeviceTier::Low);
    }

    #[test]
    fn test_ffi_device_capabilities_conversion() {
        let caps = kchat_core::capability::DeviceCapabilities {
            platform: "ios".into(),
            physical_memory: 8 * 1024 * 1024 * 1024,
            safe_allocatable_memory: 3 * 1024 * 1024 * 1024,
            cpu_arch: "aarch64".into(),
            cpu_cores: 6,
            performance_cores: Some(4),
            isa_features: vec!["neon".into()],
            gpu_backend: kchat_core::capability::GpuBackend::Metal,
            npu_provider: kchat_core::capability::NpuProvider::AppleNe,
            free_storage: 64 * 1024 * 1024 * 1024,
            battery_level: Some(85),
            on_charger: false,
            thermal_state: kchat_core::capability::ThermalState::Nominal,
            app_state: kchat_core::capability::AppState::Foreground,
            unmetered_network: true,
        };
        let ffi: FfiDeviceCapabilities = (&caps).into();
        assert_eq!(ffi.platform, "ios");
        assert_eq!(ffi.physical_memory, 8 * 1024 * 1024 * 1024);
        assert_eq!(ffi.cpu_cores, 6);
        assert_eq!(ffi.gpu_backend, "metal");
        assert_eq!(ffi.thermal_state, "nominal");
        assert_eq!(ffi.battery_level, Some(85));
    }
}

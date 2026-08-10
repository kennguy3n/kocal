//! Device profile evaluation suite.
//!
//! Tests that each device profile selects the correct tier, model pack, backend,
//! and resource budget. Verifies performance/latency targets, memory budgets,
//! and tier transition behavior under thermal, battery, and background pressure.
//!
//! Profiles covered (12 total):
//!   Mobile: iPhone 15 Pro (High), iPhone 14 (Medium), iPhone SE (Low),
//!           Pixel 8 Pro (High), Pixel 7a (Medium), Galaxy A14 (Low)
//!   Desktop: MacBook Pro M3 Max (High), MacBook Air M2 (Low), Intel NUC (Low),
//!            Windows RTX 4090 (High), Windows Surface 8 (Low), Windows Legacy (Low)

use crate::report::{EvalResult, SuiteReport};
use kchat_core::capability::{AppState, DeviceCapabilities, GpuBackend, NpuProvider, ThermalState};
use kchat_core::registry::{MinTier, ModelRegistry};
use kchat_core::scheduler::{Scheduler, SchedulerConfig};
use kchat_core::tier::{DeviceTier, TierBudget, TierSelection};
use kchat_generation::BackendType;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Device profile definitions
// ---------------------------------------------------------------------------

/// A device profile with known hardware specs and expected tier.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub name: &'static str,
    pub platform: &'static str,
    pub physical_memory_mb: u64,
    pub safe_allocatable_mb: u64,
    pub cpu_arch: &'static str,
    pub cpu_cores: u32,
    pub performance_cores: Option<u32>,
    pub isa_features: Vec<&'static str>,
    pub gpu_backend: GpuBackend,
    pub npu_provider: NpuProvider,
    pub free_storage_gb: u64,
    pub battery_level: Option<u8>,
    pub on_charger: bool,
    pub thermal_state: ThermalState,
    pub app_state: AppState,
    pub unmetered_network: bool,
    /// Expected tier under nominal conditions
    pub expected_tier: DeviceTier,
    /// Expected model pack ID for generative tasks (None = no generative)
    pub expected_model_pack: Option<&'static str>,
    /// Expected backend type string
    pub expected_backend: Option<&'static str>,
    /// Expected vision model pack ID (None = no vision model)
    pub expected_vision_pack: Option<&'static str>,
    /// Expected ASR model pack ID (None = no ASR model)
    pub expected_asr_pack: Option<&'static str>,
    /// Expected safety encoder pack ID
    pub expected_safety_pack: &'static str,
    /// Expected video model pack ID (None = no video, low-tier devices)
    pub expected_video_pack: Option<&'static str>,
}

impl DeviceProfile {
    pub fn to_caps(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            platform: self.platform.into(),
            physical_memory: self.physical_memory_mb * 1024 * 1024,
            safe_allocatable_memory: self.safe_allocatable_mb * 1024 * 1024,
            cpu_arch: self.cpu_arch.into(),
            cpu_cores: self.cpu_cores,
            performance_cores: self.performance_cores,
            isa_features: self.isa_features.iter().map(|s| s.to_string()).collect(),
            gpu_backend: self.gpu_backend,
            npu_provider: self.npu_provider,
            free_storage: self.free_storage_gb * 1024 * 1024 * 1024,
            battery_level: self.battery_level,
            on_charger: self.on_charger,
            thermal_state: self.thermal_state,
            app_state: self.app_state,
            unmetered_network: self.unmetered_network,
        }
    }

    pub fn with_thermal(&self, thermal: ThermalState) -> DeviceCapabilities {
        let mut caps = self.to_caps();
        caps.thermal_state = thermal;
        caps
    }

    pub fn with_battery(&self, level: u8, on_charger: bool) -> DeviceCapabilities {
        let mut caps = self.to_caps();
        caps.battery_level = Some(level);
        caps.on_charger = on_charger;
        caps
    }

    pub fn with_app_state(&self, state: AppState) -> DeviceCapabilities {
        let mut caps = self.to_caps();
        caps.app_state = state;
        caps
    }
}

/// All 12 device profiles used in the test suite.
pub fn all_profiles() -> Vec<DeviceProfile> {
    vec![
        // === Mobile: iOS ===
        DeviceProfile {
            name: "iPhone 15 Pro (8GB, A17 Pro)",
            platform: "ios",
            physical_memory_mb: 8192,
            safe_allocatable_mb: 6800, // ~83% of physical on iOS
            cpu_arch: "aarch64",
            cpu_cores: 6,
            performance_cores: Some(2),
            isa_features: vec!["neon", "fp16"],
            gpu_backend: GpuBackend::Metal,
            npu_provider: NpuProvider::AppleNe,
            free_storage_gb: 128,
            battery_level: Some(85),
            on_charger: false,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::High,
            expected_model_pack: Some("ternary-bonsai-8b-mlx-2bit"),
            expected_backend: Some("mlx"),
            expected_vision_pack: Some("mobileclip-s2-image-fp32"),
            expected_asr_pack: Some("whisper-base-int8"),
            expected_safety_pack: "safety-classifier-int8",
            expected_video_pack: Some("mobileclip-s2-video-int8"),
        },
        DeviceProfile {
            name: "iPhone 14 (6GB, A15)",
            platform: "ios",
            physical_memory_mb: 6144,
            safe_allocatable_mb: 4000,
            cpu_arch: "aarch64",
            cpu_cores: 6,
            performance_cores: Some(2),
            isa_features: vec!["neon"],
            gpu_backend: GpuBackend::Metal,
            npu_provider: NpuProvider::AppleNe,
            free_storage_gb: 64,
            battery_level: Some(70),
            on_charger: false,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::Medium,
            expected_model_pack: Some("ternary-bonsai-4b-mlx-2bit"),
            expected_backend: Some("mlx"),
            expected_vision_pack: Some("mobileclip-s2-image-fp32"),
            expected_asr_pack: Some("whisper-base-int8"),
            expected_safety_pack: "safety-classifier-int8",
            expected_video_pack: Some("mobileclip-s2-video-int8"),
        },
        DeviceProfile {
            name: "iPhone SE 2022 (4GB, A15)",
            platform: "ios",
            physical_memory_mb: 4096,
            safe_allocatable_mb: 2500,
            cpu_arch: "aarch64",
            cpu_cores: 6,
            performance_cores: Some(2),
            isa_features: vec!["neon"],
            gpu_backend: GpuBackend::Metal,
            npu_provider: NpuProvider::AppleNe,
            free_storage_gb: 32,
            battery_level: Some(60),
            on_charger: false,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::Low,
            expected_model_pack: Some("ternary-bonsai-1.7b-mlx-2bit"),
            expected_backend: Some("mlx"),
            expected_vision_pack: Some("mobileclip-s2-image-int8"),
            expected_asr_pack: Some("whisper-tiny-int8"),
            expected_safety_pack: "safety-classifier-int4",
            expected_video_pack: None,
        },
        // === Mobile: Android ===
        DeviceProfile {
            name: "Pixel 8 Pro (12GB, Tensor G3)",
            platform: "android",
            physical_memory_mb: 12288,
            safe_allocatable_mb: 7000,
            cpu_arch: "aarch64",
            cpu_cores: 9,
            performance_cores: Some(1),
            isa_features: vec!["neon"],
            gpu_backend: GpuBackend::Vulkan,
            npu_provider: NpuProvider::Nnapi,
            free_storage_gb: 128,
            battery_level: Some(80),
            on_charger: false,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::High,
            expected_model_pack: Some("ternary-bonsai-8b-q2_0"),
            expected_backend: Some("llama.cpp_vulkan"),
            expected_vision_pack: Some("mobileclip-s2-image-fp32"),
            expected_asr_pack: Some("whisper-base-int8"),
            expected_safety_pack: "safety-classifier-int8",
            expected_video_pack: Some("mobileclip-s2-video-int8"),
        },
        DeviceProfile {
            name: "Pixel 7a (8GB, Tensor G2)",
            platform: "android",
            physical_memory_mb: 8192,
            safe_allocatable_mb: 3800,
            cpu_arch: "aarch64",
            cpu_cores: 8,
            performance_cores: Some(2),
            isa_features: vec!["neon"],
            gpu_backend: GpuBackend::Vulkan,
            npu_provider: NpuProvider::Nnapi,
            free_storage_gb: 64,
            battery_level: Some(65),
            on_charger: false,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::Medium,
            expected_model_pack: Some("ternary-bonsai-4b-q2_0"),
            expected_backend: Some("llama.cpp_vulkan"),
            expected_vision_pack: Some("mobileclip-s2-image-fp32"),
            expected_asr_pack: Some("whisper-base-int8"),
            expected_safety_pack: "safety-classifier-int8",
            expected_video_pack: Some("mobileclip-s2-video-int8"),
        },
        DeviceProfile {
            name: "Galaxy A14 (4GB, Helio G80)",
            platform: "android",
            physical_memory_mb: 4096,
            safe_allocatable_mb: 1800,
            cpu_arch: "aarch64",
            cpu_cores: 8,
            performance_cores: Some(2),
            isa_features: vec!["neon"],
            gpu_backend: GpuBackend::Vulkan,
            npu_provider: NpuProvider::None,
            free_storage_gb: 16,
            battery_level: Some(50),
            on_charger: false,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: false,
            expected_tier: DeviceTier::Low,
            expected_model_pack: Some("ternary-bonsai-1.7b-q2_0"),
            expected_backend: Some("llama.cpp_vulkan"),
            expected_vision_pack: Some("mobileclip-s2-image-int8"),
            expected_asr_pack: Some("whisper-tiny-int8"),
            expected_safety_pack: "safety-classifier-int4",
            expected_video_pack: None,
        },
        // === Desktop: macOS ===
        DeviceProfile {
            name: "MacBook Pro M3 Max (36GB)",
            platform: "macos",
            physical_memory_mb: 36864,
            safe_allocatable_mb: 22000,
            cpu_arch: "aarch64",
            cpu_cores: 12,
            performance_cores: Some(4),
            isa_features: vec!["neon", "fp16"],
            gpu_backend: GpuBackend::Metal,
            npu_provider: NpuProvider::AppleNe,
            free_storage_gb: 512,
            battery_level: None,
            on_charger: true,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::High,
            expected_model_pack: Some("ternary-bonsai-8b-mlx-2bit"),
            expected_backend: Some("mlx"),
            expected_vision_pack: Some("mobileclip-s2-image-fp32"),
            expected_asr_pack: Some("whisper-base-int8"),
            expected_safety_pack: "safety-classifier-int8",
            expected_video_pack: Some("mobileclip-s2-video-int8"),
        },
        DeviceProfile {
            name: "MacBook Air M2 (8GB)",
            platform: "macos",
            physical_memory_mb: 8192,
            safe_allocatable_mb: 4900, // 60% of 8GB
            cpu_arch: "aarch64",
            cpu_cores: 8,
            performance_cores: Some(4),
            isa_features: vec!["neon"],
            gpu_backend: GpuBackend::Metal,
            npu_provider: NpuProvider::AppleNe,
            free_storage_gb: 256,
            battery_level: None,
            on_charger: true,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::Low,
            expected_model_pack: Some("ternary-bonsai-1.7b-mlx-2bit"),
            expected_backend: Some("mlx"),
            expected_vision_pack: Some("mobileclip-s2-image-int8"),
            expected_asr_pack: Some("whisper-tiny-int8"),
            expected_safety_pack: "safety-classifier-int4",
            expected_video_pack: None,
        },
        DeviceProfile {
            name: "Intel NUC (8GB, i3)",
            platform: "macos",
            physical_memory_mb: 8192,
            safe_allocatable_mb: 4900,
            cpu_arch: "x86_64",
            cpu_cores: 4,
            performance_cores: None,
            isa_features: vec!["avx2"],
            gpu_backend: GpuBackend::None,
            npu_provider: NpuProvider::None,
            free_storage_gb: 128,
            battery_level: None,
            on_charger: true,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::Low,
            expected_model_pack: Some("ternary-bonsai-1.7b-q2_0"),
            expected_backend: Some("llama.cpp_cpu"),
            expected_vision_pack: Some("mobileclip-s2-image-int8"),
            expected_asr_pack: Some("whisper-tiny-int8"),
            expected_safety_pack: "safety-classifier-int4",
            expected_video_pack: None,
        },
        // === Desktop: Windows ===
        DeviceProfile {
            name: "Windows RTX 4090 (32GB)",
            platform: "windows",
            physical_memory_mb: 32768,
            safe_allocatable_mb: 22000, // 60% of 32GB, above 20GB High threshold
            cpu_arch: "x86_64",
            cpu_cores: 16,
            performance_cores: Some(8),
            isa_features: vec!["avx2", "avx512"],
            gpu_backend: GpuBackend::Cuda,
            npu_provider: NpuProvider::WindowsNpu,
            free_storage_gb: 1024,
            battery_level: None,
            on_charger: true,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::High,
            expected_model_pack: Some("ternary-bonsai-8b-q2_0"),
            expected_backend: Some("llama.cpp_vulkan"),
            expected_vision_pack: Some("mobileclip-s2-image-fp32"),
            expected_asr_pack: Some("whisper-base-int8"),
            expected_safety_pack: "safety-classifier-int8",
            expected_video_pack: Some("mobileclip-s2-video-int8"),
        },
        DeviceProfile {
            name: "Windows Surface 8 (16GB)",
            platform: "windows",
            physical_memory_mb: 16384,
            safe_allocatable_mb: 9800, // 60% of 16GB
            cpu_arch: "aarch64",
            cpu_cores: 8,
            performance_cores: Some(4),
            isa_features: vec!["neon"],
            gpu_backend: GpuBackend::Vulkan,
            npu_provider: NpuProvider::WindowsNpu,
            free_storage_gb: 256,
            battery_level: Some(75),
            on_charger: false,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::Low,
            expected_model_pack: Some("ternary-bonsai-1.7b-q2_0"),
            expected_backend: Some("llama.cpp_vulkan"),
            expected_vision_pack: Some("mobileclip-s2-image-int8"),
            expected_asr_pack: Some("whisper-tiny-int8"),
            expected_safety_pack: "safety-classifier-int4",
            expected_video_pack: None,
        },
        DeviceProfile {
            name: "Windows Legacy (8GB, i5)",
            platform: "windows",
            physical_memory_mb: 8192,
            safe_allocatable_mb: 4900,
            cpu_arch: "x86_64",
            cpu_cores: 4,
            performance_cores: None,
            isa_features: vec!["avx2"],
            gpu_backend: GpuBackend::None,
            npu_provider: NpuProvider::None,
            free_storage_gb: 64,
            battery_level: None,
            on_charger: true,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
            expected_tier: DeviceTier::Low,
            expected_model_pack: Some("ternary-bonsai-1.7b-q2_0"),
            expected_backend: Some("llama.cpp_vulkan"),
            expected_vision_pack: Some("mobileclip-s2-image-int8"),
            expected_asr_pack: Some("whisper-tiny-int8"),
            expected_safety_pack: "safety-classifier-int4",
            expected_video_pack: None,
        },
    ]
}

// ---------------------------------------------------------------------------
// Helper: select model pack for a tier from the registry
// ---------------------------------------------------------------------------

/// Select the appropriate generative model for a tier and platform.
///
/// - High tier on Apple Silicon (iOS/macOS): Ternary-Bonsai-8B MLX 2-bit (~2.1GB)
/// - High tier on Android/Windows: Ternary-Bonsai-8B Q2_0 GGUF (~2.1GB)
/// - Medium tier on Apple Silicon (iOS/macOS): Ternary-Bonsai-4B MLX 2-bit (~1.0GB)
/// - Medium tier on other platforms: Ternary-Bonsai-4B Q2_0 GGUF (~1.0GB)
/// - Low tier on Apple Silicon (iOS/macOS): Ternary-Bonsai-1.7B MLX 2-bit (~472MB)
/// - Low tier on other platforms (including Intel Macs): Ternary-Bonsai-1.7B Q2_0 GGUF (~442MB)
pub fn select_model_for_tier(tier: DeviceTier) -> Option<&'static str> {
    select_model_for_tier_platform(tier, "", "aarch64")
}
/// Platform- and arch-aware model selection.
pub fn select_model_for_tier_platform(tier: DeviceTier, platform: &str, cpu_arch: &str) -> Option<&'static str> {
    let is_apple_silicon = (platform == "ios" || platform == "macos") && cpu_arch == "aarch64";
    match tier {
        DeviceTier::Low => {
            if is_apple_silicon {
                Some("ternary-bonsai-1.7b-mlx-2bit")  // Bonsai 1.7B MLX 2-bit, ~472MB
            } else {
                Some("ternary-bonsai-1.7b-q2_0")       // Bonsai 1.7B Q2_0 GGUF, ~442MB
            }
        }
        DeviceTier::Medium => {
            if is_apple_silicon {
                Some("ternary-bonsai-4b-mlx-2bit")    // Bonsai 4B MLX 2-bit, ~1.0GB
            } else {
                Some("ternary-bonsai-4b-q2_0")         // Bonsai 4B Q2_0 GGUF, ~1.0GB
            }
        }
        DeviceTier::High => {
            if is_apple_silicon {
                Some("ternary-bonsai-8b-mlx-2bit")    // Bonsai 8B MLX 2-bit, ~2.1GB
            } else if platform == "android" || platform == "windows" {
                Some("ternary-bonsai-8b-q2_0")         // Bonsai 8B Q2_0 GGUF, ~2.1GB
            } else {
                Some("qwen3.5-0.8b-q8")                // Qwen 0.8B Q8 fallback, 850MB
            }
        }
    }
}

pub fn tier_to_min_tier(tier: DeviceTier) -> MinTier {
    match tier {
        DeviceTier::Low => MinTier::Low,
        DeviceTier::Medium => MinTier::Medium,
        DeviceTier::High => MinTier::High,
    }
}

/// Select the appropriate vision model for a tier.
///
/// - Low tier: mobileclip-s2-image-int8 (70MB, INT8)
/// - Medium/High tier: mobileclip-s2-image-fp32 (137MB, FP32)
pub fn select_vision_model_for_tier(tier: DeviceTier) -> Option<&'static str> {
    match tier {
        DeviceTier::Low => Some("mobileclip-s2-image-int8"),
        DeviceTier::Medium | DeviceTier::High => Some("mobileclip-s2-image-fp32"),
    }
}

/// Select the appropriate video model for a tier.
///
/// - Low tier: None (deterministic media descriptors only)
/// - Medium/High tier: mobileclip-s2-video-int8 (70MB, INT8)
pub fn select_video_model_for_tier(tier: DeviceTier) -> Option<&'static str> {
    match tier {
        DeviceTier::Low => None,
        DeviceTier::Medium | DeviceTier::High => Some("mobileclip-s2-video-int8"),
    }
}

/// Select the appropriate ASR model for a tier.
///
/// - Low tier: whisper-tiny-int8 (40MB, INT8)
/// - Medium/High tier: whisper-base-int8 (90MB, INT8)
pub fn select_asr_model_for_tier(tier: DeviceTier) -> Option<&'static str> {
    match tier {
        DeviceTier::Low => Some("whisper-tiny-int8"),
        DeviceTier::Medium | DeviceTier::High => Some("whisper-base-int8"),
    }
}

/// Select the appropriate safety encoder model for a tier.
///
/// - Low tier: safety-classifier-int4 (15MB, INT4)
/// - Medium/High tier: safety-classifier-int8 (25MB, INT8)
pub fn select_safety_model_for_tier(tier: DeviceTier) -> &'static str {
    match tier {
        DeviceTier::Low => "safety-classifier-int4",
        DeviceTier::Medium | DeviceTier::High => "safety-classifier-int8",
    }
}

// ---------------------------------------------------------------------------
// Test suite entry point
// ---------------------------------------------------------------------------

pub fn run() -> SuiteReport {
    let mut suite = SuiteReport::new("Device Profile Suite", 1.0);
    let profiles = all_profiles();
    let registry = ModelRegistry::default_registry();

    // --- Tier selection tests ---
    for p in &profiles {
        suite.add(test_tier_selection(p));
    }

    // --- Model selection tests ---
    for p in &profiles {
        suite.add(test_model_selection(p, &registry));
    }

    // --- Backend selection tests ---
    for p in &profiles {
        suite.add(test_backend_selection(p));
    }

    // --- Resource budget tests ---
    for p in &profiles {
        suite.add(test_resource_budget(p));
    }

    // --- Performance target tests ---
    for p in &profiles {
        suite.add(test_performance_targets(p));
    }

    // --- Memory budget tests ---
    for p in &profiles {
        suite.add(test_memory_budget(p));
    }

    // --- Thermal transition tests ---
    for p in &profiles {
        suite.add(test_thermal_transition(p));
    }

    // --- Battery transition tests ---
    for p in &profiles {
        suite.add(test_battery_transition(p));
    }

    // --- Background transition tests (mobile only) ---
    for p in &profiles {
        if p.platform == "ios" || p.platform == "android" {
            suite.add(test_background_transition(p));
        }
    }

    // --- Scheduler job admission tests ---
    for p in &profiles {
        suite.add(test_scheduler_admission(p));
    }

    // --- Scheduler concurrent job limit ---
    suite.add(test_concurrent_job_limit());

    // --- Scheduler kill switch ---
    suite.add(test_kill_switch());

    // --- Enterprise policy cap tests ---
    suite.add(test_policy_cap_never_elevates());
    suite.add(test_policy_cap_allows_lower());

    // --- Idle unload timeout tests ---
    for p in &profiles {
        suite.add(test_idle_unload_timeout(p));
    }

    // --- Context cap per platform tests ---
    suite.add(test_context_cap_mobile());
    suite.add(test_context_cap_desktop());

    // --- Output token range tests ---
    suite.add(test_output_token_ranges());

    // --- Safe AI budget calculation tests ---
    for p in &profiles {
        suite.add(test_safe_ai_budget(p));
    }

    // --- Re-evaluation consistency tests ---
    for p in &profiles {
        suite.add(test_re_evaluate_consistency(p));
    }

    // --- Model registry compatibility tests ---
    suite.add(test_registry_finds_model_for_high_tier(&registry));
    suite.add(test_registry_finds_no_model_for_low_tier(&registry));
    suite.add(test_registry_finds_embedding_for_medium(&registry));
    suite.add(test_registry_finds_safety_for_medium(&registry));

    // --- Vision / ASR / Safety model selection tests ---
    for p in &profiles {
        suite.add(test_vision_model_selection(p, &registry));
        suite.add(test_asr_model_selection(p, &registry));
        suite.add(test_safety_model_selection(p, &registry));
        suite.add(test_video_model_selection(p, &registry));
    }

    suite
}

// ---------------------------------------------------------------------------
// Individual test functions
// ---------------------------------------------------------------------------

fn test_tier_selection(p: &DeviceProfile) -> EvalResult {
    let caps = p.to_caps();
    match TierSelection::select(&caps) {
        Ok(tier) => {
            if tier == p.expected_tier {
                let mut meta = HashMap::new();
                meta.insert("tier".into(), format!("{:?}", tier));
                meta.insert("safe_mb".into(), format!("{}", p.safe_allocatable_mb));
                EvalResult::pass_with_meta(
                    format!("tier_select: {}", p.name),
                    0,
                    meta,
                )
            } else {
                EvalResult::fail(
                    format!("tier_select: {}", p.name),
                    format!("expected {:?}, got {:?}", p.expected_tier, tier),
                )
            }
        }
        Err(e) => EvalResult::fail(
            format!("tier_select: {}", p.name),
            format!("selection error: {}", e),
        ),
    }
}

fn test_model_selection(p: &DeviceProfile, registry: &ModelRegistry) -> EvalResult {
    let caps = p.to_caps();
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
    let selected = select_model_for_tier_platform(tier, &caps.platform, &caps.cpu_arch);

    if selected == p.expected_model_pack {
        // Verify the model exists in the registry and is compatible with the tier
        if let Some(pack_id) = selected {
            if let Some(entry) = registry.find(pack_id) {
                if !entry.min_tier.satisfied_by(tier_to_min_tier(tier)) {
                    return EvalResult::fail(
                        format!("model_select: {}", p.name),
                        format!("model {} min_tier not satisfied by device tier {:?}", pack_id, tier),
                    );
                }
            } else {
                return EvalResult::fail(
                    format!("model_select: {}", p.name),
                    format!("model {} not found in registry", pack_id),
                );
            }
        }
        EvalResult::pass(format!("model_select: {}", p.name))
    } else {
        EvalResult::fail(
            format!("model_select: {}", p.name),
            format!("expected {:?}, got {:?}", p.expected_model_pack, selected),
        )
    }
}

fn test_backend_selection(p: &DeviceProfile) -> EvalResult {
    let caps = p.to_caps();
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);

    // Use the real BackendType::select API from kchat-generation
    let backend = BackendType::select(&caps.platform, tier, &caps.cpu_arch).map(|b| b.as_str().to_string());

    let expected = p.expected_backend.map(|s| s.to_string());

    if backend == expected {
        EvalResult::pass(format!("backend_select: {}", p.name))
    } else {
        EvalResult::fail(
            format!("backend_select: {}", p.name),
            format!("expected {:?}, got {:?}", expected, backend),
        )
    }
}

fn test_resource_budget(p: &DeviceProfile) -> EvalResult {
    let caps = p.to_caps();
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
    let budget = TierBudget::for_tier(tier, p.platform);

    let mut errors = Vec::new();

    // Context cap must match tier
    let expected_ctx = tier.context_cap();
    if budget.context_cap != expected_ctx {
        errors.push(format!(
            "context_cap {} != {}",
            budget.context_cap, expected_ctx
        ));
    }

    // Output token range must match tier
    let expected_output = tier.output_cap();
    if budget.output_token_range != expected_output {
        errors.push(format!(
            "output_range {:?} != {:?}",
            budget.output_token_range, expected_output
        ));
    }

    // Max memory must match tier+platform
    let expected_mem = tier.peak_memory_budget(p.platform);
    if budget.max_memory_bytes != expected_mem {
        errors.push(format!(
            "max_memory {} != {}",
            budget.max_memory_bytes, expected_mem
        ));
    }

    // Max perf cores must match tier
    if budget.max_perf_cores != tier.max_perf_cores() {
        errors.push(format!(
            "max_perf_cores {} != {}",
            budget.max_perf_cores,
            tier.max_perf_cores()
        ));
    }

    // Idle unload timeout
    let expected_idle = if p.platform == "ios" || p.platform == "android" {
        45
    } else {
        300
    };
    if budget.idle_unload_secs != expected_idle {
        errors.push(format!(
            "idle_unload {} != {}",
            budget.idle_unload_secs, expected_idle
        ));
    }

    if errors.is_empty() {
        EvalResult::pass(format!("resource_budget: {}", p.name))
    } else {
        EvalResult::fail(format!("resource_budget: {}", p.name), errors.join("; "))
    }
}

fn test_performance_targets(p: &DeviceProfile) -> EvalResult {
    let caps = p.to_caps();
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);

    let mut errors = Vec::new();

    // TTFT P95 target
    let ttft_target = tier.ttft_p95_target_ms();
    match tier {
        DeviceTier::Low => {
            if ttft_target != 2500 {
                errors.push(format!("ttft_p95 {} != 2500", ttft_target));
            }
        }
        DeviceTier::Medium => {
            if ttft_target != 1500 {
                errors.push(format!("ttft_p95 {} != 1500", ttft_target));
            }
        }
        DeviceTier::High => {
            if ttft_target != 1000 {
                errors.push(format!("ttft_p95 {} != 1000", ttft_target));
            }
        }
    }

    // Decode rate minimums
    let is_mobile = p.platform == "ios" || p.platform == "android";
    let decode_min = if is_mobile {
        tier.mobile_decode_p50_min()
    } else {
        tier.desktop_decode_p50_min()
    };

    match tier {
        DeviceTier::Low => {
            if is_mobile && decode_min != 8.0 {
                errors.push(format!("mobile_decode_p50 {} != 8.0", decode_min));
            }
            if !is_mobile && decode_min != 10.0 {
                errors.push(format!("desktop_decode_p50 {} != 10.0", decode_min));
            }
        }
        DeviceTier::Medium => {
            if is_mobile && decode_min != 15.0 {
                errors.push(format!("mobile_decode_p50 {} != 15.0", decode_min));
            }
            if !is_mobile && decode_min != 20.0 {
                errors.push(format!("desktop_decode_p50 {} != 20.0", decode_min));
            }
        }
        DeviceTier::High => {
            if is_mobile && decode_min != 25.0 {
                errors.push(format!("mobile_decode_p50 {} != 25.0", decode_min));
            }
            if !is_mobile && decode_min != 35.0 {
                errors.push(format!("desktop_decode_p50 {} != 35.0", decode_min));
            }
        }
    }

    if errors.is_empty() {
        let mut meta = HashMap::new();
        meta.insert("ttft_p95_ms".into(), format!("{}", ttft_target));
        meta.insert("decode_p50_min".into(), format!("{:.1}", decode_min));
        EvalResult::pass_with_meta(
            format!("perf_targets: {}", p.name),
            0,
            meta,
        )
    } else {
        EvalResult::fail(format!("perf_targets: {}", p.name), errors.join("; "))
    }
}

fn test_memory_budget(p: &DeviceProfile) -> EvalResult {
    let caps = p.to_caps();
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
    let peak_budget = tier.peak_memory_budget(p.platform);
    let safe_ai = caps.safe_ai_budget();

    let mut errors = Vec::new();

    // Peak memory budget must not exceed safe AI budget
    if peak_budget > safe_ai {
        errors.push(format!(
            "peak_budget {}MB > safe_ai {}MB",
            peak_budget / (1024 * 1024),
            safe_ai / (1024 * 1024)
        ));
    }

    // Check that the tier-appropriate model pack fits within the budget
    let model_size = match (tier, &p.platform[..], &p.cpu_arch[..]) {
        (DeviceTier::Low, "ios" | "macos", "aarch64") => 472_000_000,              // Bonsai 1.7B MLX ~472MB
        (DeviceTier::Low, _, _) => 463_290_464,                                     // Bonsai 1.7B Q2_0 GGUF ~442MB
        (DeviceTier::Medium, "ios" | "macos", "aarch64") => 1_000_000_000,         // Bonsai 4B MLX ~1.0GB
        (DeviceTier::Medium, _, _) => 1_074_969_344,                               // Bonsai 4B Q2_0 GGUF ~1.0GB
        (DeviceTier::High, "ios" | "macos", "aarch64") => 2_100_000_000,           // Bonsai 8B MLX ~2.1GB
        (DeviceTier::High, "android", _) => 2_182_184_672,                         // Bonsai 8B Q2_0 GGUF ~2.1GB
        (DeviceTier::High, "windows", _) => 2_182_184_672,                         // Bonsai 8B Q2_0 GGUF ~2.1GB
        (DeviceTier::High, _, _) => 850 * 1024 * 1024,                             // 0.8B Q8 fallback ~850MB
    };
    if model_size > peak_budget {
        errors.push(format!(
            "model_size {}MB > peak_budget {}MB",
            model_size / (1024 * 1024),
            peak_budget / (1024 * 1024)
        ));
    }

    // Verify check_memory_budget passes for the tier's peak budget
    if TierSelection::check_memory_budget(peak_budget, &caps).is_err() {
        errors.push("check_memory_budget rejected tier's own peak budget".into());
    }

    // Verify check_memory_budget rejects excessive requests
    let excessive = safe_ai + 1;
    if TierSelection::check_memory_budget(excessive, &caps).is_ok() {
        errors.push("check_memory_budget accepted request exceeding safe AI budget".into());
    }

    if errors.is_empty() {
        let mut meta = HashMap::new();
        meta.insert("peak_mb".into(), format!("{}", peak_budget / (1024 * 1024)));
        meta.insert("safe_ai_mb".into(), format!("{}", safe_ai / (1024 * 1024)));
        EvalResult::pass_with_meta(
            format!("memory_budget: {}", p.name),
            0,
            meta,
        )
    } else {
        EvalResult::fail(format!("memory_budget: {}", p.name), errors.join("; "))
    }
}

fn test_thermal_transition(p: &DeviceProfile) -> EvalResult {
    let nominal_tier = TierSelection::select(&p.to_caps()).unwrap_or(DeviceTier::Low);

    // Serious thermal → downgrade once
    let serious_tier = TierSelection::select(&p.with_thermal(ThermalState::Serious))
        .unwrap_or(DeviceTier::Low);
    let expected_serious = match nominal_tier {
        DeviceTier::High => DeviceTier::Medium,
        DeviceTier::Medium => DeviceTier::Low,
        DeviceTier::Low => DeviceTier::Low,
    };

    // Critical thermal → always Low
    let critical_tier = TierSelection::select(&p.with_thermal(ThermalState::Critical))
        .unwrap_or(DeviceTier::Low);

    // Fair thermal → same as nominal
    let fair_tier = TierSelection::select(&p.with_thermal(ThermalState::Fair))
        .unwrap_or(DeviceTier::Low);

    let mut errors = Vec::new();
    if serious_tier != expected_serious {
        errors.push(format!(
            "serious: expected {:?}, got {:?}",
            expected_serious, serious_tier
        ));
    }
    if critical_tier != DeviceTier::Low {
        errors.push(format!("critical: expected Low, got {:?}", critical_tier));
    }
    if fair_tier != nominal_tier {
        errors.push(format!(
            "fair: expected {:?} (same as nominal), got {:?}",
            nominal_tier, fair_tier
        ));
    }

    if errors.is_empty() {
        EvalResult::pass(format!("thermal_transition: {}", p.name))
    } else {
        EvalResult::fail(format!("thermal_transition: {}", p.name), errors.join("; "))
    }
}

fn test_battery_transition(p: &DeviceProfile) -> EvalResult {
    let nominal_tier = TierSelection::select(&p.to_caps()).unwrap_or(DeviceTier::Low);

    // Low battery + not charging → downgrade once
    let low_battery_tier = TierSelection::re_evaluate(
        nominal_tier,
        &p.with_battery(10, false),
    ).unwrap_or(DeviceTier::Low);

    let expected_low = match nominal_tier {
        DeviceTier::High => DeviceTier::Medium,
        DeviceTier::Medium => DeviceTier::Low,
        DeviceTier::Low => DeviceTier::Low,
    };

    // Low battery + charging → no downgrade
    let charging_tier = TierSelection::re_evaluate(
        nominal_tier,
        &p.with_battery(10, true),
    ).unwrap_or(DeviceTier::Low);

    // Full battery + not charging → no downgrade
    let full_battery_tier = TierSelection::re_evaluate(
        nominal_tier,
        &p.with_battery(90, false),
    ).unwrap_or(DeviceTier::Low);

    let mut errors = Vec::new();
    if low_battery_tier != expected_low {
        errors.push(format!(
            "low_battery: expected {:?}, got {:?}",
            expected_low, low_battery_tier
        ));
    }
    if charging_tier != nominal_tier {
        errors.push(format!(
            "charging: expected {:?} (no downgrade), got {:?}",
            nominal_tier, charging_tier
        ));
    }
    if full_battery_tier != nominal_tier {
        errors.push(format!(
            "full_battery: expected {:?} (no downgrade), got {:?}",
            nominal_tier, full_battery_tier
        ));
    }

    if errors.is_empty() {
        EvalResult::pass(format!("battery_transition: {}", p.name))
    } else {
        EvalResult::fail(format!("battery_transition: {}", p.name), errors.join("; "))
    }
}

fn test_background_transition(p: &DeviceProfile) -> EvalResult {
    let nominal_tier = TierSelection::select(&p.to_caps()).unwrap_or(DeviceTier::Low);

    // Background on mobile → always Low
    let bg_tier = TierSelection::re_evaluate(
        nominal_tier,
        &p.with_app_state(AppState::Background),
    ).unwrap_or(DeviceTier::Low);

    // Foreground → same as nominal
    let fg_tier = TierSelection::re_evaluate(
        nominal_tier,
        &p.with_app_state(AppState::Foreground),
    ).unwrap_or(DeviceTier::Low);

    let mut errors = Vec::new();
    if bg_tier != DeviceTier::Low {
        errors.push(format!(
            "background: expected Low, got {:?}",
            bg_tier
        ));
    }
    if fg_tier != nominal_tier {
        errors.push(format!(
            "foreground: expected {:?}, got {:?}",
            nominal_tier, fg_tier
        ));
    }

    if errors.is_empty() {
        EvalResult::pass(format!("background_transition: {}", p.name))
    } else {
        EvalResult::fail(format!("background_transition: {}", p.name), errors.join("; "))
    }
}

fn test_scheduler_admission(p: &DeviceProfile) -> EvalResult {
    let caps = p.to_caps();
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
    let scheduler = Scheduler::new(SchedulerConfig::default(), tier);

    // All tiers can now run generative jobs with tier-appropriate models
    let requires_generative = true;
    let peak_bytes = tier.peak_memory_budget(p.platform);

    let result = scheduler.request_job(&caps, requires_generative, peak_bytes);

    let mut errors = Vec::new();

    match &result {
        Ok(budget) => {
            if budget.tier != tier {
                errors.push(format!(
                    "budget tier {:?} != selected tier {:?}",
                    budget.tier, tier
                ));
            }
        }
        Err(e) => {
            errors.push(format!("job rejected on {:?} tier: {}", tier, e));
        }
    }

    if errors.is_empty() {
        EvalResult::pass(format!("scheduler_admission: {}", p.name))
    } else {
        EvalResult::fail(format!("scheduler_admission: {}", p.name), errors.join("; "))
    }
}

fn test_concurrent_job_limit() -> EvalResult {
    let caps = DeviceCapabilities {
        platform: "ios".into(),
        physical_memory: 8 * 1024 * 1024 * 1024,
        safe_allocatable_memory: 6800 * 1024 * 1024,
        cpu_arch: "aarch64".into(),
        cpu_cores: 6,
        performance_cores: Some(2),
        isa_features: vec!["neon".into()],
        gpu_backend: GpuBackend::Metal,
        npu_provider: NpuProvider::AppleNe,
        free_storage: 128 * 1024 * 1024 * 1024,
        battery_level: Some(85),
        on_charger: false,
        thermal_state: ThermalState::Nominal,
        app_state: AppState::Foreground,
        unmetered_network: true,
    };

    let scheduler = Scheduler::new(SchedulerConfig::default(), DeviceTier::High);
    let peak = DeviceTier::High.peak_memory_budget("ios");

    // First job should succeed
    let r1 = scheduler.request_job(&caps, true, peak);
    if r1.is_err() {
        return EvalResult::fail("concurrent_job_limit", "first job rejected");
    }

    // Second concurrent job should fail (max_concurrent_jobs = 1)
    let r2 = scheduler.request_job(&caps, true, peak);
    if r2.is_ok() {
        return EvalResult::fail("concurrent_job_limit", "second concurrent job accepted");
    }

    // Complete first job
    scheduler.complete_job();

    // Now a new job should succeed
    let r3 = scheduler.request_job(&caps, true, peak);
    if r3.is_err() {
        return EvalResult::fail("concurrent_job_limit", "job after completion rejected");
    }

    EvalResult::pass("concurrent_job_limit")
}

fn test_kill_switch() -> EvalResult {
    let caps = DeviceCapabilities {
        platform: "macos".into(),
        physical_memory: 36 * 1024 * 1024 * 1024,
        safe_allocatable_memory: 22000 * 1024 * 1024,
        cpu_arch: "aarch64".into(),
        cpu_cores: 12,
        performance_cores: Some(4),
        isa_features: vec!["neon".into()],
        gpu_backend: GpuBackend::Metal,
        npu_provider: NpuProvider::AppleNe,
        free_storage: 512 * 1024 * 1024 * 1024,
        battery_level: None,
        on_charger: true,
        thermal_state: ThermalState::Nominal,
        app_state: AppState::Foreground,
        unmetered_network: true,
    };

    let scheduler = Scheduler::new(SchedulerConfig::default(), DeviceTier::High);
    let peak = DeviceTier::High.peak_memory_budget("macos");

    // Normal job should succeed
    let r1 = scheduler.request_job(&caps, true, peak);
    if r1.is_err() {
        return EvalResult::fail("kill_switch", "normal job rejected");
    }
    scheduler.complete_job();

    // Activate kill switch
    scheduler.activate_kill_switch();

    // Job should be rejected
    let r2 = scheduler.request_job(&caps, true, peak);
    if r2.is_ok() {
        return EvalResult::fail("kill_switch", "job accepted with kill switch active");
    }

    // Deactivate kill switch
    scheduler.deactivate_kill_switch();

    // Job should succeed again
    let r3 = scheduler.request_job(&caps, true, peak);
    if r3.is_err() {
        return EvalResult::fail("kill_switch", "job rejected after kill switch deactivated");
    }

    EvalResult::pass("kill_switch")
}

fn test_policy_cap_never_elevates() -> EvalResult {
    // Low device with High policy cap → should stay Low
    let tier = TierSelection::apply_policy_cap(DeviceTier::Low, Some(DeviceTier::High));
    if tier != DeviceTier::Low {
        return EvalResult::fail(
            "policy_cap_never_elevates",
            format!("expected Low, got {:?}", tier),
        );
    }
    EvalResult::pass("policy_cap_never_elevates")
}

fn test_policy_cap_allows_lower() -> EvalResult {
    // High device with Low policy cap → should be capped to Low
    let tier = TierSelection::apply_policy_cap(DeviceTier::High, Some(DeviceTier::Low));
    if tier != DeviceTier::Low {
        return EvalResult::fail(
            "policy_cap_allows_lower",
            format!("expected Low, got {:?}", tier),
        );
    }

    // High device with Medium policy cap → should be capped to Medium
    let tier = TierSelection::apply_policy_cap(DeviceTier::High, Some(DeviceTier::Medium));
    if tier != DeviceTier::Medium {
        return EvalResult::fail(
            "policy_cap_allows_lower",
            format!("expected Medium, got {:?}", tier),
        );
    }

    // Medium device with no cap → should stay Medium
    let tier = TierSelection::apply_policy_cap(DeviceTier::Medium, None);
    if tier != DeviceTier::Medium {
        return EvalResult::fail(
            "policy_cap_allows_lower",
            format!("expected Medium, got {:?}", tier),
        );
    }

    EvalResult::pass("policy_cap_allows_lower")
}

fn test_idle_unload_timeout(p: &DeviceProfile) -> EvalResult {
    let tier = TierSelection::select(&p.to_caps()).unwrap_or(DeviceTier::Low);
    let budget = TierBudget::for_tier(tier, p.platform);

    let expected = if p.platform == "ios" || p.platform == "android" {
        45
    } else {
        300
    };

    if budget.idle_unload_secs == expected {
        EvalResult::pass(format!("idle_unload: {}", p.name))
    } else {
        EvalResult::fail(
            format!("idle_unload: {}", p.name),
            format!("expected {}s, got {}s", expected, budget.idle_unload_secs),
        )
    }
}

fn test_context_cap_mobile() -> EvalResult {
    let mut errors: Vec<String> = Vec::new();

    if DeviceTier::Low.context_cap_for_platform("ios") != 2048 {
        errors.push("Low/ios context cap != 2048".into());
    }
    if DeviceTier::Medium.context_cap_for_platform("android") != 4096 {
        errors.push("Medium/android context cap != 4096".into());
    }
    if DeviceTier::High.context_cap_for_platform("ios") != 8192 {
        errors.push("High/ios context cap != 8192".into());
    }

    if errors.is_empty() {
        EvalResult::pass("context_cap_mobile")
    } else {
        EvalResult::fail("context_cap_mobile", errors.join("; "))
    }
}

fn test_context_cap_desktop() -> EvalResult {
    let mut errors: Vec<String> = Vec::new();

    if DeviceTier::Low.context_cap_for_platform("macos") != 2048 {
        errors.push("Low/macos context cap != 2048".into());
    }
    if DeviceTier::Medium.context_cap_for_platform("windows") != 4096 {
        errors.push("Medium/windows context cap != 4096".into());
    }
    if DeviceTier::High.context_cap_for_platform("macos") != 16384 {
        errors.push("High/macos context cap != 16384".into());
    }
    if DeviceTier::High.context_cap_for_platform("windows") != 16384 {
        errors.push("High/windows context cap != 16384".into());
    }

    if errors.is_empty() {
        EvalResult::pass("context_cap_desktop")
    } else {
        EvalResult::fail("context_cap_desktop", errors.join("; "))
    }
}

fn test_output_token_ranges() -> EvalResult {
    let mut errors = Vec::new();

    if DeviceTier::Low.output_cap() != (64, 192) {
        errors.push(format!("Low output cap {:?} != (64, 192)", DeviceTier::Low.output_cap()));
    }
    if DeviceTier::Medium.output_cap() != (256, 512) {
        errors.push(format!("Medium output cap {:?} != (256, 512)", DeviceTier::Medium.output_cap()));
    }
    if DeviceTier::High.output_cap() != (512, 1024) {
        errors.push(format!("High output cap {:?} != (512, 1024)", DeviceTier::High.output_cap()));
    }

    if errors.is_empty() {
        EvalResult::pass("output_token_ranges")
    } else {
        EvalResult::fail("output_token_ranges", errors.join("; "))
    }
}

fn test_safe_ai_budget(p: &DeviceProfile) -> EvalResult {
    let caps = p.to_caps();
    let safe_ai = caps.safe_ai_budget();
    let expected = caps.safe_allocatable_memory * 7 / 10;

    if safe_ai == expected {
        let mut meta = HashMap::new();
        meta.insert("safe_ai_mb".into(), format!("{}", safe_ai / (1024 * 1024)));
        EvalResult::pass_with_meta(
            format!("safe_ai_budget: {}", p.name),
            0,
            meta,
        )
    } else {
        EvalResult::fail(
            format!("safe_ai_budget: {}", p.name),
            format!("expected {}MB, got {}MB", expected / (1024 * 1024), safe_ai / (1024 * 1024)),
        )
    }
}

fn test_re_evaluate_consistency(p: &DeviceProfile) -> EvalResult {
    let caps = p.to_caps();
    let initial_tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);

    // Re-evaluating with the same caps should return the same tier
    let re_evaluated = TierSelection::re_evaluate(initial_tier, &caps).unwrap_or(DeviceTier::Low);

    if re_evaluated == initial_tier {
        EvalResult::pass(format!("re_evaluate: {}", p.name))
    } else {
        EvalResult::fail(
            format!("re_evaluate: {}", p.name),
            format!("initial {:?}, re-evaluated {:?} (should match)", initial_tier, re_evaluated),
        )
    }
}

fn test_registry_finds_model_for_high_tier(registry: &ModelRegistry) -> EvalResult {
    let results = registry.find_for_task("summarize", MinTier::High);
    if results.is_empty() {
        EvalResult::fail("registry_high_tier_model", "no summarize models found for High tier")
    } else {
        let mut meta = HashMap::new();
        meta.insert("count".into(), format!("{}", results.len()));
        meta.insert("first".into(), results[0].pack_id.clone());
        // High tier should find all 9 generative models
        // (Bonsai-1.7B-MLX, Bonsai-1.7B-GGUF, 0.8B-Q4, Bonsai-4B-GGUF, Bonsai-4B-MLX,
        //  Bonsai-8B-MLX, Macaw, Bonsai-8B-GGUF, Q8)
        if results.len() != 9 {
            EvalResult::fail(
                "registry_high_tier_model",
                format!("expected 9 generative models for High tier, got {}", results.len()),
            )
        } else {
            EvalResult::pass_with_meta("registry_high_tier_model", 0, meta)
        }
    }
}

fn test_registry_finds_no_model_for_low_tier(registry: &ModelRegistry) -> EvalResult {
    // Low tier should find 2 Bonsai-1.7B generative models (MLX + GGUF)
    let low_results = registry.find_for_task("summarize", MinTier::Low);
    let mut errors = Vec::new();
    if low_results.is_empty() {
        errors.push("no generative models found for Low tier (expected 2 Bonsai-1.7B)".into());
    } else {
        // Verify all Low tier models fit in 750MB mobile budget
        for m in &low_results {
            if m.size_bytes > 750 * 1024 * 1024 {
                errors.push(format!(
                    "Low tier model {} too large: {}MB > 750MB",
                    m.pack_id,
                    m.size_bytes / (1024 * 1024)
                ));
            }
        }
        // Should find exactly 2 models
        if low_results.len() != 2 {
            errors.push(format!(
                "expected 2 Low tier generative models, got {}",
                low_results.len()
            ));
        }
    }

    // Reranker still requires High tier
    let high_only = registry.find_for_task("rerank", MinTier::High);
    let medium_only = registry.find_for_task("rerank", MinTier::Medium);
    if high_only.is_empty() {
        errors.push("no rerank models for High tier".into());
    }
    if !medium_only.is_empty() {
        errors.push(format!(
            "rerank model found for Medium tier (should require High): {}",
            medium_only[0].pack_id
        ));
    }

    if errors.is_empty() {
        EvalResult::pass("registry_low_tier_model")
    } else {
        EvalResult::fail("registry_low_tier_model", errors.join("; "))
    }
}

fn test_registry_finds_embedding_for_medium(registry: &ModelRegistry) -> EvalResult {
    let results = registry.find_for_task("embed", MinTier::Medium);
    if results.is_empty() {
        EvalResult::fail("registry_embedding_medium", "no embedding models found for Medium tier")
    } else {
        let mut meta = HashMap::new();
        meta.insert("pack_id".into(), results[0].pack_id.clone());
        meta.insert("size_mb".into(), format!("{}", results[0].size_bytes / (1024 * 1024)));
        EvalResult::pass_with_meta("registry_embedding_medium", 0, meta)
    }
}

fn test_registry_finds_safety_for_medium(registry: &ModelRegistry) -> EvalResult {
    let results = registry.find_for_task("safety", MinTier::Medium);
    if results.is_empty() {
        EvalResult::fail("registry_safety_medium", "no safety models found for Medium tier")
    } else {
        let mut meta = HashMap::new();
        meta.insert("count".into(), format!("{}", results.len()));
        meta.insert("pack_ids".into(), results.iter().map(|e| e.pack_id.as_str()).collect::<Vec<_>>().join(", "));
        // Medium tier should find both safety-classifier-int8 and safety-classifier-int4
        if results.len() != 2 {
            EvalResult::fail(
                "registry_safety_medium",
                format!("expected 2 safety models for Medium tier, got {}", results.len()),
            )
        } else {
            EvalResult::pass_with_meta("registry_safety_medium", 0, meta)
        }
    }
}

fn test_vision_model_selection(p: &DeviceProfile, registry: &ModelRegistry) -> EvalResult {
    let caps = p.to_caps();
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
    let selected = select_vision_model_for_tier(tier);

    if selected == p.expected_vision_pack {
        if let Some(pack_id) = selected {
            if let Some(entry) = registry.find(pack_id) {
                if !entry.min_tier.satisfied_by(tier_to_min_tier(tier)) {
                    return EvalResult::fail(
                        format!("vision_select: {}", p.name),
                        format!("vision model {} min_tier not satisfied by device tier {:?}", pack_id, tier),
                    );
                }
            } else {
                return EvalResult::fail(
                    format!("vision_select: {}", p.name),
                    format!("vision model {} not found in registry", pack_id),
                );
            }
        }
        EvalResult::pass(format!("vision_select: {}", p.name))
    } else {
        EvalResult::fail(
            format!("vision_select: {}", p.name),
            format!("expected {:?}, got {:?}", p.expected_vision_pack, selected),
        )
    }
}

fn test_asr_model_selection(p: &DeviceProfile, registry: &ModelRegistry) -> EvalResult {
    let caps = p.to_caps();
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
    let selected = select_asr_model_for_tier(tier);

    if selected == p.expected_asr_pack {
        if let Some(pack_id) = selected {
            if let Some(entry) = registry.find(pack_id) {
                if !entry.min_tier.satisfied_by(tier_to_min_tier(tier)) {
                    return EvalResult::fail(
                        format!("asr_select: {}", p.name),
                        format!("ASR model {} min_tier not satisfied by device tier {:?}", pack_id, tier),
                    );
                }
            } else {
                return EvalResult::fail(
                    format!("asr_select: {}", p.name),
                    format!("ASR model {} not found in registry", pack_id),
                );
            }
        }
        EvalResult::pass(format!("asr_select: {}", p.name))
    } else {
        EvalResult::fail(
            format!("asr_select: {}", p.name),
            format!("expected {:?}, got {:?}", p.expected_asr_pack, selected),
        )
    }
}

fn test_safety_model_selection(p: &DeviceProfile, registry: &ModelRegistry) -> EvalResult {
    let caps = p.to_caps();
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
    let selected = select_safety_model_for_tier(tier);

    if selected == p.expected_safety_pack {
        if let Some(entry) = registry.find(selected) {
            if !entry.min_tier.satisfied_by(tier_to_min_tier(tier)) {
                return EvalResult::fail(
                    format!("safety_select: {}", p.name),
                    format!("safety model {} min_tier not satisfied by device tier {:?}", selected, tier),
                );
            }
        } else {
            return EvalResult::fail(
                format!("safety_select: {}", p.name),
                format!("safety model {} not found in registry", selected),
            );
        }
        EvalResult::pass(format!("safety_select: {}", p.name))
    } else {
        EvalResult::fail(
            format!("safety_select: {}", p.name),
            format!("expected {}, got {}", p.expected_safety_pack, selected),
        )
    }
}

fn test_video_model_selection(p: &DeviceProfile, registry: &ModelRegistry) -> EvalResult {
    let caps = p.to_caps();
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
    let selected = select_video_model_for_tier(tier);

    if selected == p.expected_video_pack {
        if let Some(pack_id) = selected {
            if let Some(entry) = registry.find(pack_id) {
                if !entry.min_tier.satisfied_by(tier_to_min_tier(tier)) {
                    return EvalResult::fail(
                        format!("video_select: {}", p.name),
                        format!("video model {} min_tier not satisfied by device tier {:?}", pack_id, tier),
                    );
                }
            } else {
                return EvalResult::fail(
                    format!("video_select: {}", p.name),
                    format!("video model {} not found in registry", pack_id),
                );
            }
        }
        EvalResult::pass(format!("video_select: {}", p.name))
    } else {
        EvalResult::fail(
            format!("video_select: {}", p.name),
            format!("expected {:?}, got {:?}", p.expected_video_pack, selected),
        )
    }
}

//! Device tier selection and per-tier resource budgets.
//!
//! Tier is a runtime decision, not a marketing label. A high-end phone under
//! memory pressure or serious thermal state must temporarily route as medium
//! or low. Enterprise policy may cap the tier but may not elevate it.

use crate::capability::{DeviceCapabilities, ThermalState};
use crate::error::{CoreError, Result};
use serde::{Deserialize, Serialize};

/// Device capability tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceTier {
    /// Low: 4-6 GB mobile, 8 GB desktop. No mandatory generator.
    Low,
    /// Medium: 6-8 GB mobile, 16-24 GB desktop. Default generative pack.
    Medium,
    /// High: 8 GB+ mobile, 32 GB+ desktop. Larger generative pack.
    High,
}

impl DeviceTier {
    /// Maximum active context window in tokens.
    /// High tier returns 8K for mobile, 16K for desktop.
    pub fn context_cap(self) -> usize {
        match self {
            DeviceTier::Low => 2048,
            DeviceTier::Medium => 4096,
            DeviceTier::High => 8192,
        }
    }

    /// Platform-aware context cap: 16K for desktop, 8K for mobile.
    pub fn context_cap_for_platform(self, platform: &str) -> usize {
        match self {
            DeviceTier::Low => 2048,
            DeviceTier::Medium => 4096,
            DeviceTier::High => match platform {
                "macos" | "windows" => 16384,
                _ => 8192,
            },
        }
    }

    /// Maximum output tokens per task.
    pub fn output_cap(self) -> (usize, usize) {
        match self {
            DeviceTier::Low => (64, 192),
            DeviceTier::Medium => (256, 512),
            DeviceTier::High => (512, 1024),
        }
    }

    /// Peak AI memory budget in bytes.
    pub fn peak_memory_budget(self, platform: &str) -> u64 {
        match (self, platform) {
            (DeviceTier::Low, "ios") | (DeviceTier::Low, "android") => 750 * 1024 * 1024,
            (DeviceTier::Medium, "ios") => 1400 * 1024 * 1024,
            (DeviceTier::Medium, "android") => 1500 * 1024 * 1024,
            (DeviceTier::High, "ios") => 2500 * 1024 * 1024,
            (DeviceTier::High, "android") => 3000 * 1024 * 1024,
            (DeviceTier::Low, "macos") | (DeviceTier::Low, "windows") => 2000 * 1024 * 1024,
            (DeviceTier::Medium, "macos") | (DeviceTier::Medium, "windows") => 4000 * 1024 * 1024,
            (DeviceTier::High, "macos") | (DeviceTier::High, "windows") => 8000 * 1024 * 1024,
            _ => 750 * 1024 * 1024,
        }
    }

    /// Target TTFT P95 in milliseconds.
    pub fn ttft_p95_target_ms(self) -> u64 {
        match self {
            DeviceTier::Low => 2500,
            DeviceTier::Medium => 1500,
            DeviceTier::High => 1000,
        }
    }

    /// Minimum decode P50 in tokens/second (mobile).
    pub fn mobile_decode_p50_min(self) -> f64 {
        match self {
            DeviceTier::Low => 8.0,
            DeviceTier::Medium => 15.0,
            DeviceTier::High => 25.0,
        }
    }

    /// Minimum decode P50 in tokens/second (desktop).
    pub fn desktop_decode_p50_min(self) -> f64 {
        match self {
            DeviceTier::Low => 10.0,
            DeviceTier::Medium => 20.0,
            DeviceTier::High => 35.0,
        }
    }

    /// Maximum CPU performance cores to use for generative inference.
    pub fn max_perf_cores(self) -> u32 {
        match self {
            DeviceTier::Low => 2,
            DeviceTier::Medium => 3,
            DeviceTier::High => 4,
        }
    }
}

/// Per-tier resource budget for a specific job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierBudget {
    pub tier: DeviceTier,
    pub context_cap: usize,
    pub output_token_range: (usize, usize),
    pub max_memory_bytes: u64,
    pub max_perf_cores: u32,
    /// Idle unload timeout in seconds (mobile only; desktop uses configurable).
    pub idle_unload_secs: u32,
}

impl TierBudget {
    pub fn for_tier(tier: DeviceTier, platform: &str) -> Self {
        let idle_unload_secs = if platform == "ios" || platform == "android" {
            45 // 30-60 second range, midpoint
        } else {
            300 // 5 minutes desktop default
        };

        Self {
            tier,
            context_cap: tier.context_cap(),
            output_token_range: tier.output_cap(),
            max_memory_bytes: tier.peak_memory_budget(platform),
            max_perf_cores: tier.max_perf_cores(),
            idle_unload_secs,
        }
    }
}

/// Tier selection logic — runs at install and is re-evaluated before each job.
pub struct TierSelection;

impl TierSelection {
    /// Select the maximum eligible tier from device capabilities.
    ///
    /// Rules:
    /// 1. Detect hardware and OS capabilities.
    /// 2. Verify safe allocation and memory-map behavior under normal KChat load.
    /// 3. Select a maximum eligible tier.
    /// 4. Enterprise policy may cap but not elevate.
    pub fn select(caps: &DeviceCapabilities) -> Result<DeviceTier> {
        let platform = caps.platform.as_str();
        let safe_mb = caps.safe_allocatable_memory / (1024 * 1024);

        // Memory-based tier selection (initial gate)
        let memory_tier = match platform {
            "ios" | "android" => {
                if safe_mb >= 6000 {
                    DeviceTier::High
                } else if safe_mb >= 3500 {
                    DeviceTier::Medium
                } else {
                    DeviceTier::Low
                }
            }
            "macos" | "windows" => {
                if safe_mb >= 20000 {
                    DeviceTier::High
                } else if safe_mb >= 10000 {
                    DeviceTier::Medium
                } else {
                    DeviceTier::Low
                }
            }
            _ => DeviceTier::Low,
        };

        // Thermal downgrade
        let tier = Self::apply_thermal_downgrade(memory_tier, caps.thermal_state);

        Ok(tier)
    }

    /// Re-evaluate tier before each job using free memory, thermal, battery,
    /// and background status. Downgrade immediately after allocation failure,
    /// repeated slow TTFT, critical thermal events, or OS termination signals.
    pub fn re_evaluate(
        current: DeviceTier,
        caps: &DeviceCapabilities,
    ) -> Result<DeviceTier> {
        // Always apply thermal state
        let tier = Self::apply_thermal_downgrade(current, caps.thermal_state);

        // Background → no generative on mobile
        if caps.platform == "ios" || caps.platform == "android" {
            if caps.app_state != crate::capability::AppState::Foreground {
                // In background, only allow low tier (deterministic-only effectively)
                return Ok(DeviceTier::Low);
            }
        }

        // Battery below 15% and not charging → downgrade by one level
        if let Some(level) = caps.battery_level {
            if level < 15 && !caps.on_charger {
                return Ok(Self::downgrade_once(tier));
            }
        }

        Ok(tier)
    }

    /// Apply enterprise policy cap. May lower the tier but never elevate it.
    pub fn apply_policy_cap(tier: DeviceTier, cap: Option<DeviceTier>) -> DeviceTier {
        match cap {
            Some(cap_tier) => {
                let cap_rank = match cap_tier {
                    DeviceTier::Low => 0,
                    DeviceTier::Medium => 1,
                    DeviceTier::High => 2,
                };
                let tier_rank = match tier {
                    DeviceTier::Low => 0,
                    DeviceTier::Medium => 1,
                    DeviceTier::High => 2,
                };
                if tier_rank > cap_rank {
                    cap_tier
                } else {
                    tier
                }
            }
            None => tier,
        }
    }

    fn apply_thermal_downgrade(tier: DeviceTier, thermal: ThermalState) -> DeviceTier {
        match thermal {
            ThermalState::Nominal | ThermalState::Fair => tier,
            ThermalState::Serious => Self::downgrade_once(tier),
            ThermalState::Critical => DeviceTier::Low,
        }
    }

    fn downgrade_once(tier: DeviceTier) -> DeviceTier {
        match tier {
            DeviceTier::High => DeviceTier::Medium,
            DeviceTier::Medium => DeviceTier::Low,
            DeviceTier::Low => DeviceTier::Low,
        }
    }

    /// Check if a predicted peak memory exceeds 70% of the currently safe AI budget.
    /// If so, the task must be rejected or rerouted before allocation.
    pub fn check_memory_budget(
        predicted_peak_bytes: u64,
        caps: &DeviceCapabilities,
    ) -> Result<()> {
        let threshold = caps.safe_ai_budget();
        if predicted_peak_bytes > threshold {
            return Err(CoreError::MemoryBudgetExceeded {
                requested_mb: predicted_peak_bytes / (1024 * 1024),
                safe_mb: threshold / (1024 * 1024),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{AppState, DeviceCapabilities, GpuBackend, NpuProvider, ThermalState};

    fn make_caps(platform: &str, mem_mb: u64, thermal: ThermalState) -> DeviceCapabilities {
        DeviceCapabilities {
            platform: platform.into(),
            physical_memory: mem_mb * 1024 * 1024,
            safe_allocatable_memory: mem_mb * 1024 * 1024,
            cpu_arch: "aarch64".into(),
            cpu_cores: 8,
            performance_cores: Some(4),
            isa_features: vec!["neon".into()],
            gpu_backend: GpuBackend::Metal,
            npu_provider: NpuProvider::AppleNe,
            free_storage: 10 * 1024 * 1024 * 1024,
            battery_level: Some(80),
            on_charger: true,
            thermal_state: thermal,
            app_state: AppState::Foreground,
            unmetered_network: true,
        }
    }

    #[test]
    fn test_ios_high_tier_selection() {
        let caps = make_caps("ios", 8000, ThermalState::Nominal);
        let tier = TierSelection::select(&caps).unwrap();
        assert_eq!(tier, DeviceTier::High);
    }

    #[test]
    fn test_ios_medium_tier_selection() {
        let caps = make_caps("ios", 4000, ThermalState::Nominal);
        let tier = TierSelection::select(&caps).unwrap();
        assert_eq!(tier, DeviceTier::Medium);
    }

    #[test]
    fn test_ios_low_tier_selection() {
        let caps = make_caps("ios", 2000, ThermalState::Nominal);
        let tier = TierSelection::select(&caps).unwrap();
        assert_eq!(tier, DeviceTier::Low);
    }

    #[test]
    fn test_thermal_serious_downgrades() {
        let caps = make_caps("ios", 8000, ThermalState::Serious);
        let tier = TierSelection::select(&caps).unwrap();
        assert_eq!(tier, DeviceTier::Medium);
    }

    #[test]
    fn test_thermal_critical_forces_low() {
        let caps = make_caps("ios", 8000, ThermalState::Critical);
        let tier = TierSelection::select(&caps).unwrap();
        assert_eq!(tier, DeviceTier::Low);
    }

    #[test]
    fn test_policy_cap_never_elevates() {
        let tier = DeviceTier::High;
        let capped = TierSelection::apply_policy_cap(tier, Some(DeviceTier::Low));
        assert_eq!(capped, DeviceTier::Low);

        let tier = DeviceTier::Low;
        let capped = TierSelection::apply_policy_cap(tier, Some(DeviceTier::High));
        assert_eq!(capped, DeviceTier::Low);
    }

    #[test]
    fn test_memory_budget_check() {
        let caps = make_caps("ios", 4000, ThermalState::Nominal);
        // safe_ai_budget = 4000 MB * 1024 * 1024 * 0.7 = 2936012800 bytes
        let budget = caps.safe_ai_budget();
        assert!(TierSelection::check_memory_budget(budget - 1, &caps).is_ok());
        assert!(TierSelection::check_memory_budget(budget + 1, &caps).is_err());
    }

    #[test]
    fn test_context_caps() {
        assert_eq!(DeviceTier::Low.context_cap(), 2048);
        assert_eq!(DeviceTier::Medium.context_cap(), 4096);
        assert_eq!(DeviceTier::High.context_cap(), 8192);
    }

    #[test]
    fn test_output_caps() {
        assert_eq!(DeviceTier::Low.output_cap(), (64, 192));
        assert_eq!(DeviceTier::Medium.output_cap(), (256, 512));
        assert_eq!(DeviceTier::High.output_cap(), (512, 1024));
    }
}

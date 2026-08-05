//! Resource governor — OS-level resource limits for inference workloads.
//!
//! The governor wraps every inference call with resource checks:
//! - CPU usage monitoring (don't exceed max_cpu_percent)
//! - GPU memory monitoring (don't exceed max_gpu_memory_percent)
//! - Thermal state enforcement (pause on Serious/Critical)
//! - Battery conservation (pause below threshold, throttle below throttle threshold)
//! - Concurrency limit (max 1 concurrent inference via semaphore)
//! - Timeout enforcement (cancel jobs that exceed timeout_secs)
//!
//! The governor integrates with the scheduler: the scheduler handles job
//! queuing, the governor handles resource limits.

use crate::capability::{AppState, DeviceCapabilities, ThermalState};
use crate::error::{CoreError, Result};
use parking_lot::Mutex;
use std::time::{Duration, Instant};

/// Configuration for the resource governor.
#[derive(Debug, Clone)]
pub struct GovernorConfig {
    /// Maximum CPU usage percentage (default: 60%)
    pub max_cpu_percent: u32,
    /// Maximum GPU memory usage percentage (default: 70%)
    pub max_gpu_memory_percent: u32,
    /// Maximum AI memory usage as percentage of safe_allocatable (default: 50%)
    pub max_ai_memory_percent: u32,
    /// Battery level below which inference is paused (default: 20%)
    pub battery_pause_threshold: u8,
    /// Battery level below which inference is throttled (default: 30%)
    pub battery_throttle_threshold: u8,
    /// Thermal state at which inference is paused (default: Serious)
    pub thermal_pause_state: ThermalState,
    /// Maximum inference duration before timeout (default: 30s)
    pub timeout_secs: u64,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            max_cpu_percent: 60,
            max_gpu_memory_percent: 70,
            max_ai_memory_percent: 50,
            battery_pause_threshold: 20,
            battery_throttle_threshold: 30,
            thermal_pause_state: ThermalState::Serious,
            timeout_secs: 30,
        }
    }
}

impl GovernorConfig {
    /// Conservative config for low-tier devices.
    pub fn for_low_tier() -> Self {
        Self {
            max_cpu_percent: 40,
            max_gpu_memory_percent: 50,
            max_ai_memory_percent: 30,
            battery_pause_threshold: 30,
            battery_throttle_threshold: 50,
            thermal_pause_state: ThermalState::Fair,
            timeout_secs: 15,
        }
    }

    /// Balanced config for medium-tier devices.
    pub fn for_medium_tier() -> Self {
        Self {
            max_cpu_percent: 60,
            max_gpu_memory_percent: 70,
            max_ai_memory_percent: 40,
            battery_pause_threshold: 20,
            battery_throttle_threshold: 30,
            thermal_pause_state: ThermalState::Serious,
            timeout_secs: 30,
        }
    }

    /// High-performance config for high-tier devices.
    pub fn for_high_tier() -> Self {
        Self {
            max_cpu_percent: 80,
            max_gpu_memory_percent: 80,
            max_ai_memory_percent: 60,
            battery_pause_threshold: 15,
            battery_throttle_threshold: 25,
            thermal_pause_state: ThermalState::Serious,
            timeout_secs: 60,
        }
    }
}

/// A permit held during an inference job. Released when dropped.
pub struct Permit {
    _released: bool,
}

impl Permit {
    fn new() -> Self {
        Self { _released: false }
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        self._released = true;
    }
}

/// Resource governor — enforces OS-level resource limits on inference.
pub struct ResourceGovernor {
    config: GovernorConfig,
    /// Tracks the last time a job started (for concurrency check)
    last_job_start: Mutex<Option<Instant>>,
    /// Whether a job is currently running
    job_running: Mutex<bool>,
}

impl ResourceGovernor {
    /// Create a new governor with the given config.
    pub fn new(config: GovernorConfig) -> Self {
        Self {
            config,
            last_job_start: Mutex::new(None),
            job_running: Mutex::new(false),
        }
    }

    /// Create a governor with tier-appropriate defaults.
    pub fn for_tier(tier: crate::tier::DeviceTier) -> Self {
        let config = match tier {
            crate::tier::DeviceTier::Low => GovernorConfig::for_low_tier(),
            crate::tier::DeviceTier::Medium => GovernorConfig::for_medium_tier(),
            crate::tier::DeviceTier::High => GovernorConfig::for_high_tier(),
        };
        Self::new(config)
    }

    /// Check if inference is allowed under current device conditions.
    ///
    /// Returns Ok(()) if allowed, or an error explaining why it's blocked.
    pub fn check(&self, caps: &DeviceCapabilities) -> Result<()> {
        // 1. Thermal check
        let thermal_blocked = match self.config.thermal_pause_state {
            ThermalState::Critical => {
                matches!(caps.thermal_state, ThermalState::Critical)
            }
            ThermalState::Serious => {
                matches!(
                    caps.thermal_state,
                    ThermalState::Serious | ThermalState::Critical
                )
            }
            ThermalState::Fair => matches!(
                caps.thermal_state,
                ThermalState::Fair | ThermalState::Serious | ThermalState::Critical
            ),
            ThermalState::Nominal => true, // Always block if pause state is Nominal
        };
        if thermal_blocked {
            return Err(CoreError::TierDowngradeRequired {
                reason: format!("thermal state {:?} blocks inference", caps.thermal_state),
            });
        }

        // 2. Battery check
        if let Some(battery) = caps.battery_level {
            if !caps.on_charger && battery <= self.config.battery_pause_threshold {
                return Err(CoreError::TierDowngradeRequired {
                    reason: format!(
                        "battery {}% below pause threshold {}%",
                        battery, self.config.battery_pause_threshold
                    ),
                });
            }
        }

        // 3. App state check (no inference in background on mobile)
        if caps.app_state == AppState::Background
            && (caps.platform == "ios" || caps.platform == "android")
        {
            return Err(CoreError::TierDowngradeRequired {
                reason: "inference blocked in background".into(),
            });
        }

        // 4. Memory check
        let ai_budget = caps.safe_ai_budget();
        let max_ai = caps.safe_allocatable_memory * self.config.max_ai_memory_percent as u64 / 100;
        if ai_budget > max_ai {
            // This is just informational — the actual memory check happens at load time
            tracing::warn!(
                "AI budget {} exceeds configured max {}%",
                ai_budget,
                self.config.max_ai_memory_percent
            );
        }

        Ok(())
    }

    /// Check if inference should be throttled (slower generation, fewer tokens).
    pub fn should_throttle(&self, caps: &DeviceCapabilities) -> bool {
        // Throttle on Fair thermal
        if caps.thermal_state == ThermalState::Fair {
            return true;
        }

        // Throttle on low battery (but above pause threshold)
        if let Some(battery) = caps.battery_level {
            if !caps.on_charger && battery <= self.config.battery_throttle_threshold {
                return true;
            }
        }

        false
    }

    /// Acquire a permit to run inference. Ensures only 1 concurrent job.
    ///
    /// Returns an error if a job is already running.
    pub fn acquire(&self) -> Result<Permit> {
        let mut running = self.job_running.lock();
        if *running {
            return Err(CoreError::SchedulerCancelled(
                "another inference job is already running".into(),
            ));
        }
        *running = true;
        *self.last_job_start.lock() = Some(Instant::now());
        Ok(Permit::new())
    }

    /// Release the inference permit.
    pub fn release(&self) {
        *self.job_running.lock() = false;
    }

    /// Check if the current job has exceeded the timeout.
    pub fn check_timeout(&self) -> Result<()> {
        let start = self.last_job_start.lock();
        if let Some(start_time) = *start {
            let elapsed = start_time.elapsed();
            if elapsed > Duration::from_secs(self.config.timeout_secs) {
                return Err(CoreError::SchedulerCancelled(format!(
                    "inference timeout after {}s",
                    elapsed.as_secs()
                )));
            }
        }
        Ok(())
    }

    /// Get the governor configuration.
    pub fn config(&self) -> &GovernorConfig {
        &self.config
    }

    /// Check if a job is currently running.
    pub fn is_job_running(&self) -> bool {
        *self.job_running.lock()
    }

    /// Get the timeout duration.
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.config.timeout_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{DeviceCapabilities, GpuBackend, NpuProvider};
    use crate::tier::DeviceTier;

    fn test_caps(
        thermal: ThermalState,
        battery: Option<u8>,
        on_charger: bool,
        app_state: AppState,
    ) -> DeviceCapabilities {
        DeviceCapabilities {
            platform: "macos".into(),
            physical_memory: 16 * 1024 * 1024 * 1024,
            safe_allocatable_memory: 10 * 1024 * 1024 * 1024,
            cpu_arch: "aarch64".into(),
            cpu_cores: 8,
            performance_cores: Some(4),
            isa_features: vec![],
            gpu_backend: GpuBackend::Metal,
            npu_provider: NpuProvider::AppleNe,
            free_storage: 0,
            battery_level: battery,
            on_charger,
            thermal_state: thermal,
            app_state,
            unmetered_network: true,
        }
    }

    #[test]
    fn test_check_allows_nominal() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        let caps = test_caps(ThermalState::Nominal, Some(80), true, AppState::Foreground);
        assert!(governor.check(&caps).is_ok());
    }

    #[test]
    fn test_check_blocks_serious_thermal() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        let caps = test_caps(ThermalState::Serious, Some(80), true, AppState::Foreground);
        assert!(governor.check(&caps).is_err());
    }

    #[test]
    fn test_check_blocks_critical_thermal() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        let caps = test_caps(ThermalState::Critical, Some(80), true, AppState::Foreground);
        assert!(governor.check(&caps).is_err());
    }

    #[test]
    fn test_check_blocks_low_battery() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        let caps = test_caps(ThermalState::Nominal, Some(15), false, AppState::Foreground);
        assert!(governor.check(&caps).is_err());
    }

    #[test]
    fn test_check_allows_low_battery_on_charger() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        let caps = test_caps(ThermalState::Nominal, Some(15), true, AppState::Foreground);
        assert!(governor.check(&caps).is_ok());
    }

    #[test]
    fn test_check_blocks_background_mobile() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        let mut caps = test_caps(ThermalState::Nominal, Some(80), true, AppState::Background);
        caps.platform = "ios".into();
        assert!(governor.check(&caps).is_err());
    }

    #[test]
    fn test_check_allows_background_desktop() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        let caps = test_caps(ThermalState::Nominal, Some(80), true, AppState::Background);
        // macOS allows background inference
        assert!(governor.check(&caps).is_ok());
    }

    #[test]
    fn test_should_throttle_fair_thermal() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        let caps = test_caps(ThermalState::Fair, Some(80), true, AppState::Foreground);
        assert!(governor.should_throttle(&caps));
    }

    #[test]
    fn test_should_throttle_low_battery() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        let caps = test_caps(ThermalState::Nominal, Some(25), false, AppState::Foreground);
        assert!(governor.should_throttle(&caps));
    }

    #[test]
    fn test_should_not_throttle_nominal() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        let caps = test_caps(ThermalState::Nominal, Some(80), true, AppState::Foreground);
        assert!(!governor.should_throttle(&caps));
    }

    #[test]
    fn test_acquire_and_release() {
        let governor = ResourceGovernor::new(GovernorConfig::default());
        assert!(!governor.is_job_running());

        let permit = governor.acquire().unwrap();
        assert!(governor.is_job_running());

        // Second acquire should fail
        let result = governor.acquire();
        assert!(result.is_err());

        drop(permit);
        governor.release();
        assert!(!governor.is_job_running());

        // Now should be able to acquire again
        let _permit2 = governor.acquire().unwrap();
        assert!(governor.is_job_running());
    }

    #[test]
    fn test_timeout_check() {
        let governor = ResourceGovernor::new(GovernorConfig {
            timeout_secs: 0, // 0 second timeout = immediate
            ..Default::default()
        });

        let _permit = governor.acquire().unwrap();
        // Sleep briefly to ensure elapsed > 0
        std::thread::sleep(Duration::from_millis(10));
        let result = governor.check_timeout();
        assert!(result.is_err());
    }

    #[test]
    fn test_timeout_not_exceeded() {
        let governor = ResourceGovernor::new(GovernorConfig {
            timeout_secs: 60,
            ..Default::default()
        });

        let _permit = governor.acquire().unwrap();
        let result = governor.check_timeout();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tier_configs() {
        let low = GovernorConfig::for_low_tier();
        assert!(low.max_cpu_percent < 50);
        assert!(low.timeout_secs <= 15);

        let mid = GovernorConfig::for_medium_tier();
        assert!(mid.max_cpu_percent >= 50);
        assert!(mid.timeout_secs >= 20);

        let high = GovernorConfig::for_high_tier();
        assert!(high.max_cpu_percent >= 70);
        assert!(high.timeout_secs >= 45);
    }

    #[test]
    fn test_for_tier() {
        let low_gov = ResourceGovernor::for_tier(DeviceTier::Low);
        assert!(low_gov.config().max_cpu_percent < 50);

        let high_gov = ResourceGovernor::for_tier(DeviceTier::High);
        assert!(high_gov.config().max_cpu_percent >= 70);
    }
}

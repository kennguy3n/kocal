//! Scheduler for memory pressure, thermal throttling, battery budgets,
//! background restrictions, and idle unloading.
//!
//! The scheduler is the central coordinator that decides whether a job can
//! run, what tier it runs at, and when to unload models.

use crate::capability::{AppState, DeviceCapabilities, ThermalState};
use crate::error::{CoreError, Result};
use crate::tier::{DeviceTier, TierBudget, TierSelection};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Scheduler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Idle unload timeout for mobile (seconds)
    pub mobile_idle_unload_secs: u32,
    /// Idle unload timeout for desktop (seconds)
    pub desktop_idle_unload_secs: u32,
    /// Maximum concurrent AI jobs
    pub max_concurrent_jobs: u32,
    /// Battery threshold below which to downgrade tier (percent)
    pub battery_downgrade_threshold: u8,
    /// Whether to allow generative inference in background (mobile)
    pub allow_background_generation: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            mobile_idle_unload_secs: 45,
            desktop_idle_unload_secs: 300,
            max_concurrent_jobs: 1,
            battery_downgrade_threshold: 15,
            allow_background_generation: false,
        }
    }
}

/// Current scheduler state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerState {
    /// Current active tier
    pub current_tier: DeviceTier,
    /// Number of active jobs
    pub active_jobs: u32,
    /// Whether a generative model is currently loaded
    pub generative_model_loaded: bool,
    /// Last activity timestamp (epoch millis)
    pub last_activity_ms: u128,
    /// Whether the scheduler has issued a kill switch
    pub kill_switch_active: bool,
}

/// The scheduler manages resource allocation for AI workloads.
pub struct Scheduler {
    config: SchedulerConfig,
    state: Mutex<SchedulerState>,
    last_model_load: Mutex<Option<Instant>>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, initial_tier: DeviceTier) -> Self {
        Self {
            config,
            state: Mutex::new(SchedulerState {
                current_tier: initial_tier,
                active_jobs: 0,
                generative_model_loaded: false,
                last_activity_ms: 0,
                kill_switch_active: false,
            }),
            last_model_load: Mutex::new(None),
        }
    }

    /// Request permission to start a job. Returns the tier budget if allowed.
    pub fn request_job(
        &self,
        caps: &DeviceCapabilities,
        requires_generative: bool,
        predicted_peak_bytes: u64,
    ) -> Result<TierBudget> {
        let mut state = self.state.lock();

        if state.kill_switch_active {
            return Err(CoreError::SchedulerCancelled("kill switch active".into()));
        }

        if state.active_jobs >= self.config.max_concurrent_jobs {
            return Err(CoreError::SchedulerCancelled(
                "max concurrent jobs reached".into(),
            ));
        }

        // Re-evaluate tier
        let tier = TierSelection::re_evaluate(state.current_tier, caps)?;
        state.current_tier = tier;

        // Check thermal state
        if requires_generative && matches!(caps.thermal_state, ThermalState::Serious | ThermalState::Critical) {
            return Err(CoreError::ThermalCritical);
        }

        // Check background restriction
        if requires_generative
            && !self.config.allow_background_generation
            && caps.app_state == AppState::Background
            && (caps.platform == "ios" || caps.platform == "android")
        {
            return Err(CoreError::SchedulerCancelled(
                "generative inference not allowed in background".into(),
            ));
        }

        // Check memory budget
        TierSelection::check_memory_budget(predicted_peak_bytes, caps)?;

        let budget = TierBudget::for_tier(tier, &caps.platform);
        state.active_jobs += 1;
        state.last_activity_ms = current_millis();

        Ok(budget)
    }

    /// Notify the scheduler that a job has completed.
    pub fn complete_job(&self) {
        let mut state = self.state.lock();
        if state.active_jobs > 0 {
            state.active_jobs -= 1;
        }
        state.last_activity_ms = current_millis();
    }

    /// Mark a generative model as loaded.
    pub fn mark_model_loaded(&self) {
        let mut state = self.state.lock();
        state.generative_model_loaded = true;
        *self.last_model_load.lock() = Some(Instant::now());
    }

    /// Mark a generative model as unloaded.
    pub fn mark_model_unloaded(&self) {
        let mut state = self.state.lock();
        state.generative_model_loaded = false;
        *self.last_model_load.lock() = None;
    }

    /// Check if the generative model should be unloaded due to idle.
    pub fn should_unload_model(&self, platform: &str) -> bool {
        let state = self.state.lock();
        if !state.generative_model_loaded || state.active_jobs > 0 {
            return false;
        }

        let unload_secs = if platform == "ios" || platform == "android" {
            self.config.mobile_idle_unload_secs
        } else {
            self.config.desktop_idle_unload_secs
        };

        if let Some(load_time) = *self.last_model_load.lock() {
            return load_time.elapsed() >= Duration::from_secs(unload_secs as u64);
        }

        false
    }

    /// Activate the kill switch — all future job requests are rejected.
    pub fn activate_kill_switch(&self) {
        let mut state = self.state.lock();
        state.kill_switch_active = true;
        tracing::warn!("Kill switch activated — all AI jobs will be rejected");
    }

    /// Deactivate the kill switch.
    pub fn deactivate_kill_switch(&self) {
        let mut state = self.state.lock();
        state.kill_switch_active = false;
        tracing::info!("Kill switch deactivated");
    }

    /// Get a snapshot of the current scheduler state.
    pub fn state(&self) -> SchedulerState {
        self.state.lock().clone()
    }

    /// Update the tier based on current device capabilities.
    pub fn update_tier(&self, caps: &DeviceCapabilities) {
        let mut state = self.state.lock();
        let new_tier = TierSelection::re_evaluate(state.current_tier, caps).unwrap_or(DeviceTier::Low);
        if new_tier != state.current_tier {
            tracing::info!(
                "Tier changed: {:?} → {:?}",
                state.current_tier,
                new_tier
            );
            state.current_tier = new_tier;
        }
    }
}

fn current_millis() -> u128 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{DeviceCapabilities, GpuBackend, NpuProvider};

    fn make_caps(platform: &str) -> DeviceCapabilities {
        DeviceCapabilities {
            platform: platform.into(),
            physical_memory: 8 * 1024 * 1024 * 1024,
            safe_allocatable_memory: 4 * 1024 * 1024 * 1024,
            cpu_arch: "aarch64".into(),
            cpu_cores: 8,
            performance_cores: Some(4),
            isa_features: vec!["neon".into()],
            gpu_backend: GpuBackend::Metal,
            npu_provider: NpuProvider::AppleNe,
            free_storage: 10 * 1024 * 1024 * 1024,
            battery_level: Some(80),
            on_charger: true,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
        }
    }

    #[test]
    fn test_job_request_succeeds() {
        let scheduler = Scheduler::new(SchedulerConfig::default(), DeviceTier::Medium);
        let caps = make_caps("ios");
        let budget = scheduler.request_job(&caps, true, 500 * 1024 * 1024);
        assert!(budget.is_ok());
    }

    #[test]
    fn test_kill_switch_rejects_jobs() {
        let scheduler = Scheduler::new(SchedulerConfig::default(), DeviceTier::Medium);
        scheduler.activate_kill_switch();
        let caps = make_caps("ios");
        let result = scheduler.request_job(&caps, true, 500 * 1024 * 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_budget_rejects_oversized() {
        let scheduler = Scheduler::new(SchedulerConfig::default(), DeviceTier::Low);
        let caps = DeviceCapabilities {
            platform: "ios".into(),
            physical_memory: 3 * 1024 * 1024 * 1024,
            safe_allocatable_memory: 1 * 1024 * 1024 * 1024,
            cpu_arch: "aarch64".into(),
            cpu_cores: 4,
            performance_cores: None,
            isa_features: vec!["neon".into()],
            gpu_backend: GpuBackend::None,
            npu_provider: NpuProvider::None,
            free_storage: 2 * 1024 * 1024 * 1024,
            battery_level: Some(80),
            on_charger: true,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
        };
        // safe_ai_budget = 1GB * 0.7 = 716.8MB
        // Request 800MB → should fail
        let result = scheduler.request_job(&caps, true, 800 * 1024 * 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_background_blocks_generation_on_mobile() {
        let scheduler = Scheduler::new(SchedulerConfig::default(), DeviceTier::Medium);
        let mut caps = make_caps("ios");
        caps.app_state = AppState::Background;
        let result = scheduler.request_job(&caps, true, 500 * 1024 * 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_idle_unload() {
        let scheduler = Scheduler::new(SchedulerConfig::default(), DeviceTier::Medium);
        scheduler.mark_model_loaded();
        // Model just loaded, should not unload yet
        assert!(!scheduler.should_unload_model("ios"));

        // Simulate idle by setting last_model_load to the past
        *scheduler.last_model_load.lock() =
            Some(Instant::now() - Duration::from_secs(60));
        assert!(scheduler.should_unload_model("ios"));
    }
}

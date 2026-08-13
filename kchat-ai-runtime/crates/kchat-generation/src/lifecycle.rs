//! Model lifecycle — load, idle unload, reload, and memory management.
//!
//! On mobile, the model is unloaded after 30-60 seconds of idle. On desktop,
//! the timeout is configurable (default 5 minutes). The lifecycle manager
//! coordinates with the scheduler to respect memory and thermal constraints.

use crate::backend::{BackendAdapter, BackendConfig, BackendError};
use kchat_core::tier::DeviceTier;
use parking_lot::Mutex;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

/// Current state of the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    /// Model is not loaded
    Unloaded,
    /// Model is being loaded
    Loading,
    /// Model is loaded and ready
    Ready,
    /// Model is being unloaded
    Unloading,
    /// Model failed to load
    Failed,
}

/// Internal lifecycle state protected by a single mutex.
struct LifecycleInner {
    state: ModelState,
    last_used: Option<Instant>,
    config: Option<BackendConfig>,
}

/// Model lifecycle manager — coordinates load/unload with the scheduler.
pub struct ModelLifecycle {
    inner: Mutex<LifecycleInner>,
    idle_timeout: Duration,
    tier: DeviceTier,
    platform: String,
}

impl ModelLifecycle {
    pub fn new(tier: DeviceTier, platform: impl Into<String>) -> Self {
        let platform_str: String = platform.into();
        let idle_timeout = match platform_str.as_str() {
            "ios" | "android" => Duration::from_secs(45),
            _ => Duration::from_secs(300),
        };

        Self {
            inner: Mutex::new(LifecycleInner {
                state: ModelState::Unloaded,
                last_used: None,
                config: None,
            }),
            idle_timeout,
            tier,
            platform: platform_str,
        }
    }

    /// Load the model with the given backend.
    /// Uses atomic state transition to prevent concurrent double-load.
    pub fn load(
        &self,
        backend: &dyn BackendAdapter,
        config: BackendConfig,
    ) -> Result<(), BackendError> {
        {
            let mut inner = self.inner.lock();
            match inner.state {
                ModelState::Ready => return Ok(()), // Already loaded
                ModelState::Loading => {
                    return Err(BackendError::LoadFailed(
                        "model load already in progress".into(),
                    ));
                }
                _ => inner.state = ModelState::Loading,
            }
        }

        match backend.load(&config) {
            Ok(()) => {
                let mut inner = self.inner.lock();
                inner.state = ModelState::Ready;
                inner.config = Some(config);
                inner.last_used = Some(Instant::now());
                tracing::info!("Model loaded successfully");
                Ok(())
            }
            Err(e) => {
                let mut inner = self.inner.lock();
                inner.state = ModelState::Failed;
                tracing::error!("Model load failed: {}", e);
                Err(e)
            }
        }
    }

    /// Unload the model.
    pub fn unload(&self, backend: &dyn BackendAdapter) -> Result<(), BackendError> {
        {
            let mut inner = self.inner.lock();
            if inner.state == ModelState::Unloaded {
                return Ok(());
            }
            inner.state = ModelState::Unloading;
        }

        match backend.unload() {
            Ok(()) => {
                let mut inner = self.inner.lock();
                inner.state = ModelState::Unloaded;
                inner.config = None;
                inner.last_used = None;
                tracing::info!("Model unloaded");
                Ok(())
            }
            Err(e) => {
                let mut inner = self.inner.lock();
                inner.state = ModelState::Failed;
                Err(e)
            }
        }
    }

    /// Mark the model as used (resets idle timer).
    pub fn touch(&self) {
        let mut inner = self.inner.lock();
        inner.last_used = Some(Instant::now());
    }

    /// Check if the model should be unloaded due to idle.
    pub fn should_unload(&self) -> bool {
        let inner = self.inner.lock();
        if inner.state != ModelState::Ready {
            return false;
        }
        if let Some(last) = inner.last_used {
            return last.elapsed() >= self.idle_timeout;
        }
        false
    }

    /// Get the current model state.
    pub fn state(&self) -> ModelState {
        self.inner.lock().state
    }

    /// Set the last-used time (for testing idle timeout behavior).
    #[cfg(test)]
    pub(crate) fn set_last_used_for_test(&self, instant: Instant) {
        self.inner.lock().last_used = Some(instant);
    }

    /// Get the idle timeout duration.
    pub fn idle_timeout(&self) -> Duration {
        self.idle_timeout
    }

    /// Get the current tier.
    pub fn tier(&self) -> DeviceTier {
        self.tier
    }

    /// Get the platform.
    pub fn platform(&self) -> &str {
        &self.platform
    }

    /// Check if generation is allowed on this tier.
    ///
    /// All tiers now have tier-appropriate generative models:
    /// - Low: 1.7B Q2_0 (~442MB)
    /// - Medium: 4B Q2_0 (~1.0GB)
    /// - High: 8B Q2_0 (~2.1GB)
    pub fn can_generate(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendType, GenerationConfig, GenerationResult};

    struct MockBackend {
        loaded: Mutex<bool>,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                loaded: Mutex::new(false),
            }
        }
    }

    impl BackendAdapter for MockBackend {
        fn load(&self, _config: &BackendConfig) -> Result<(), BackendError> {
            *self.loaded.lock() = true;
            Ok(())
        }

        fn unload(&self) -> Result<(), BackendError> {
            *self.loaded.lock() = false;
            Ok(())
        }

        fn is_loaded(&self) -> bool {
            *self.loaded.lock()
        }

        fn generate(
            &self,
            _prompt: &str,
            _config: &GenerationConfig,
        ) -> Result<GenerationResult, BackendError> {
            Ok(GenerationResult {
                text: "mock output".into(),
                prompt_tokens: 10,
                completion_tokens: 5,
                ttft_ms: 50,
                total_ms: 100,
                tokens_per_second: 50.0,
                backend: "mock".into(),
                grammar_valid: true,
            })
        }

        fn generate_stream(
            &self,
            _prompt: &str,
            _config: &GenerationConfig,
            stream: &crate::stream::StreamHandle,
        ) -> Result<GenerationResult, BackendError> {
            stream.push_token("mock");
            stream.complete(5, 100);
            Ok(GenerationResult {
                text: "mock output".into(),
                prompt_tokens: 10,
                completion_tokens: 5,
                ttft_ms: 50,
                total_ms: 100,
                tokens_per_second: 50.0,
                backend: "mock".into(),
                grammar_valid: true,
            })
        }

        fn backend_type(&self) -> BackendType {
            BackendType::LlamaCppMetal
        }
    }

    #[test]
    fn test_load_and_unload() {
        let lifecycle = ModelLifecycle::new(DeviceTier::Medium, "ios");
        let backend = MockBackend::new();
        let config = BackendConfig::for_tier(
            BackendType::LlamaCppMetal,
            "test-model",
            "/path/to/model.gguf",
            DeviceTier::Medium,
            "ios",
        );

        assert_eq!(lifecycle.state(), ModelState::Unloaded);

        lifecycle.load(&backend, config).unwrap();
        assert_eq!(lifecycle.state(), ModelState::Ready);

        lifecycle.unload(&backend).unwrap();
        assert_eq!(lifecycle.state(), ModelState::Unloaded);
    }

    #[test]
    fn test_idle_timeout_mobile() {
        let lifecycle = ModelLifecycle::new(DeviceTier::Medium, "ios");
        assert_eq!(lifecycle.idle_timeout(), Duration::from_secs(45));
    }

    #[test]
    fn test_idle_timeout_desktop() {
        let lifecycle = ModelLifecycle::new(DeviceTier::High, "macos");
        assert_eq!(lifecycle.idle_timeout(), Duration::from_secs(300));
    }

    #[test]
    fn test_should_unload_after_idle() {
        let lifecycle = ModelLifecycle::new(DeviceTier::Medium, "ios");
        let backend = MockBackend::new();
        let config = BackendConfig::for_tier(
            BackendType::LlamaCppMetal,
            "test",
            "/path",
            DeviceTier::Medium,
            "ios",
        );

        lifecycle.load(&backend, config).unwrap();
        assert!(!lifecycle.should_unload());

        // Simulate idle by setting last_used to the past
        lifecycle.set_last_used_for_test(Instant::now() - Duration::from_secs(60));
        assert!(lifecycle.should_unload());
    }

    #[test]
    fn test_touch_resets_idle() {
        let lifecycle = ModelLifecycle::new(DeviceTier::Medium, "ios");
        let backend = MockBackend::new();
        let config = BackendConfig::for_tier(
            BackendType::LlamaCppMetal,
            "test",
            "/path",
            DeviceTier::Medium,
            "ios",
        );

        lifecycle.load(&backend, config).unwrap();
        lifecycle.set_last_used_for_test(Instant::now() - Duration::from_secs(60));
        assert!(lifecycle.should_unload());

        lifecycle.touch();
        assert!(!lifecycle.should_unload());
    }

    #[test]
    fn test_low_tier_can_generate() {
        let lifecycle = ModelLifecycle::new(DeviceTier::Low, "ios");
        assert!(lifecycle.can_generate());
    }

    #[test]
    fn test_medium_tier_can_generate() {
        let lifecycle = ModelLifecycle::new(DeviceTier::Medium, "ios");
        assert!(lifecycle.can_generate());
    }
}

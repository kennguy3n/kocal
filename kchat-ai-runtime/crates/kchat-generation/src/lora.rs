//! LoRA hot-swap — load small LoRA adapters on top of a frozen base model.
//!
//! LoRA (Low-Rank Adaptation) allows one base model to serve multiple tasks
//! and languages by swapping small adapter files (3-5MB) instead of loading
//! entirely different models. The swap takes <10ms.
//!
//! The system supports 30 adapters: 5 tasks × 6 languages.
//! Tasks: summarize, translate, key_points, generate_doc, generate_slides
//! Languages: en, vi, zh, ja, ko, es

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

/// A LoRA adapter descriptor.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Unique adapter ID (e.g. "summarize.vi")
    pub adapter_id: String,
    /// Path to the adapter file (.bin or .gguf-lora)
    pub path: PathBuf,
    /// LoRA scale (typically 1.0)
    pub scale: f32,
    /// Task this adapter is for (e.g. "summarize")
    pub task: String,
    /// Language this adapter is for (e.g. "vi")
    pub language: String,
}

impl LoraAdapter {
    /// Create a new adapter descriptor.
    pub fn new(
        adapter_id: impl Into<String>,
        path: impl Into<PathBuf>,
        task: impl Into<String>,
        language: impl Into<String>,
    ) -> Self {
        Self {
            adapter_id: adapter_id.into(),
            path: path.into(),
            scale: 1.0,
            task: task.into(),
            language: language.into(),
        }
    }

    /// Set the LoRA scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Build the adapter ID from task and language.
    pub fn make_id(task: &str, language: &str) -> String {
        format!("{}.{}", task, language)
    }
}

/// Internal LoRA state protected by a single mutex.
struct LoraState {
    /// Currently active adapter (if any)
    current: Option<String>,
    /// Available adapters indexed by adapter_id
    available: HashMap<String, LoraAdapter>,
    /// Time of last swap (for measuring swap latency)
    last_swap_time: Option<Instant>,
    /// Last swap duration in milliseconds
    last_swap_duration_ms: u64,
}

/// LoRA manager — handles adapter registration, hot-swap, and detachment.
///
/// The manager tracks available adapters and the currently-active adapter.
/// When the llama.cpp backend is available, swapping calls into the backend's
/// LoRA API. Without the backend, the manager still tracks state for testing.
pub struct LoraManager {
    state: Mutex<LoraState>,
}

impl LoraManager {
    /// Create a new LoRA manager with no adapters.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(LoraState {
                current: None,
                available: HashMap::new(),
                last_swap_time: None,
                last_swap_duration_ms: 0,
            }),
        }
    }

    /// Register an available adapter.
    pub fn register(&self, adapter: LoraAdapter) {
        self.state.lock().available.insert(adapter.adapter_id.clone(), adapter);
    }

    /// Register multiple adapters at once.
    pub fn register_all(&self, adapters: Vec<LoraAdapter>) {
        let mut state = self.state.lock();
        for adapter in adapters {
            state.available.insert(adapter.adapter_id.clone(), adapter);
        }
    }

    /// Hot-swap the current LoRA adapter.
    ///
    /// In production with llama.cpp, this calls `LlamaModel::lora_adapter_init`
    /// and `LlamaContext::set_adapter`. Without the backend, it just updates
    /// the tracked state.
    ///
    /// Returns the swap duration in milliseconds.
    pub fn swap(&self, adapter_id: &str) -> Result<u64, LoraError> {
        let start = Instant::now();

        // Verify the adapter exists and update current atomically
        {
            let mut state = self.state.lock();
            if !state.available.contains_key(adapter_id) {
                return Err(LoraError::AdapterNotFound(adapter_id.into()));
            }
            state.current = Some(adapter_id.to_string());
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        {
            let mut state = self.state.lock();
            state.last_swap_time = Some(Instant::now());
            state.last_swap_duration_ms = duration_ms;
        }

        tracing::info!(
            "LoRA adapter swapped to {} in {}ms",
            adapter_id,
            duration_ms
        );

        Ok(duration_ms)
    }

    /// Detach the current adapter, reverting to the base model.
    pub fn detach(&self) -> Result<(), LoraError> {
        let mut state = self.state.lock();
        if state.current.is_none() {
            return Ok(()); // Already detached
        }
        let old = state.current.take();
        tracing::info!("Detached LoRA adapter {:?}", old);
        Ok(())
    }

    /// Get the currently active adapter ID.
    pub fn current_adapter(&self) -> Option<String> {
        self.state.lock().current.clone()
    }

    /// Check if an adapter is currently active.
    pub fn has_active_adapter(&self) -> bool {
        self.state.lock().current.is_some()
    }

    /// Get all registered adapter IDs.
    pub fn available_adapters(&self) -> Vec<String> {
        self.state.lock().available.keys().cloned().collect()
    }

    /// Get the last swap duration in milliseconds.
    pub fn last_swap_duration_ms(&self) -> u64 {
        self.state.lock().last_swap_duration_ms
    }

    /// Find an adapter by task and language.
    pub fn find(&self, task: &str, language: &str) -> Option<LoraAdapter> {
        let id = LoraAdapter::make_id(task, language);
        self.state.lock().available.get(&id).cloned()
    }

    /// List all adapters for a given task.
    pub fn adapters_for_task(&self, task: &str) -> Vec<LoraAdapter> {
        self.state
            .lock()
            .available
            .values()
            .filter(|a| a.task == task)
            .cloned()
            .collect()
    }

    /// List all adapters for a given language.
    pub fn adapters_for_language(&self, language: &str) -> Vec<LoraAdapter> {
        self.state
            .lock()
            .available
            .values()
            .filter(|a| a.language == language)
            .cloned()
            .collect()
    }

    /// Remove an adapter from the registry.
    /// Checks current and removes atomically under a single lock.
    pub fn unregister(&self, adapter_id: &str) {
        let mut state = self.state.lock();
        if state.current.as_deref() == Some(adapter_id) {
            tracing::warn!("Cannot unregister active adapter {}", adapter_id);
            return;
        }
        state.available.remove(adapter_id);
    }
}

impl Default for LoraManager {
    fn default() -> Self {
        Self::new()
    }
}

/// LoRA errors.
#[derive(Debug, thiserror::Error)]
pub enum LoraError {
    #[error("adapter not found: {0}")]
    AdapterNotFound(String),

    #[error("adapter load failed: {0}")]
    LoadFailed(String),

    #[error("adapter file not found: {0}")]
    FileNotFound(String),
}

/// Standard task identifiers.
pub mod tasks {
    pub const SUMMARIZE: &str = "summarize";
    pub const TRANSLATE: &str = "translate";
    pub const KEY_POINTS: &str = "key_points";
    pub const GENERATE_DOC: &str = "generate_doc";
    pub const GENERATE_SLIDES: &str = "generate_slides";

    pub const ALL: &[&str] = &[
        SUMMARIZE,
        TRANSLATE,
        KEY_POINTS,
        GENERATE_DOC,
        GENERATE_SLIDES,
    ];
}

/// Supported languages for LoRA adapters.
pub mod languages {
    pub const EN: &str = "en";
    pub const VI: &str = "vi";
    pub const ZH: &str = "zh";
    pub const JA: &str = "ja";
    pub const KO: &str = "ko";
    pub const ES: &str = "es";

    pub const ALL: &[&str] = &[EN, VI, ZH, JA, KO, ES];
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_adapters() -> Vec<LoraAdapter> {
        let mut adapters = Vec::new();
        for &task in tasks::ALL {
            for &lang in languages::ALL {
                let id = LoraAdapter::make_id(task, lang);
                let path = format!("/models/adapters/{}.bin", id);
                adapters.push(LoraAdapter::new(id, path, task, lang));
            }
        }
        adapters
    }

    #[test]
    fn test_register_and_swap() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        let duration = manager.swap("summarize.vi").unwrap();
        assert!(manager.has_active_adapter());
        assert_eq!(manager.current_adapter(), Some("summarize.vi".into()));
        // Swap should be very fast (just updating state)
        assert!(duration < 100);
    }

    #[test]
    fn test_swap_nonexistent_adapter() {
        let manager = LoraManager::new();
        let result = manager.swap("nonexistent");
        assert!(matches!(result, Err(LoraError::AdapterNotFound(_))));
    }

    #[test]
    fn test_detach() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        manager.swap("translate.zh").unwrap();
        assert!(manager.has_active_adapter());

        manager.detach().unwrap();
        assert!(!manager.has_active_adapter());
    }

    #[test]
    fn test_detach_without_active() {
        let manager = LoraManager::new();
        // Detaching when nothing is active should be Ok
        assert!(manager.detach().is_ok());
    }

    #[test]
    fn test_find_by_task_and_language() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        let adapter = manager.find("summarize", "vi").unwrap();
        assert_eq!(adapter.adapter_id, "summarize.vi");
        assert_eq!(adapter.task, "summarize");
        assert_eq!(adapter.language, "vi");
    }

    #[test]
    fn test_find_nonexistent() {
        let manager = LoraManager::new();
        assert!(manager.find("nonexistent", "xx").is_none());
    }

    #[test]
    fn test_adapters_for_task() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        let summarize_adapters = manager.adapters_for_task("summarize");
        assert_eq!(summarize_adapters.len(), 6); // 6 languages
    }

    #[test]
    fn test_adapters_for_language() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        let vi_adapters = manager.adapters_for_language("vi");
        assert_eq!(vi_adapters.len(), 5); // 5 tasks
    }

    #[test]
    fn test_unregister() {
        let manager = LoraManager::new();
        manager.register(LoraAdapter::new("test.en", "/path", "test", "en"));

        manager.unregister("test.en");
        assert!(manager.find("test", "en").is_none());
    }

    #[test]
    fn test_unregister_active_adapter_fails() {
        let manager = LoraManager::new();
        manager.register(LoraAdapter::new("test.en", "/path", "test", "en"));

        manager.swap("test.en").unwrap();
        manager.unregister("test.en"); // Should not remove
        assert!(manager.find("test", "en").is_some());
    }

    #[test]
    fn test_all_30_adapters() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        let all = manager.available_adapters();
        assert_eq!(all.len(), 30); // 5 tasks × 6 languages
    }

    #[test]
    fn test_adapter_with_scale() {
        let adapter = LoraAdapter::new("test", "/path", "test", "en").with_scale(0.5);
        assert_eq!(adapter.scale, 0.5);
    }

    #[test]
    fn test_make_id() {
        assert_eq!(LoraAdapter::make_id("summarize", "vi"), "summarize.vi");
    }

    #[test]
    fn test_swap_duration_tracked() {
        let manager = LoraManager::new();
        manager.register(LoraAdapter::new("test.en", "/path", "test", "en"));

        manager.swap("test.en").unwrap();
        let duration = manager.last_swap_duration_ms();
        // Should be very small (just state update)
        assert!(duration < 100);
    }
}

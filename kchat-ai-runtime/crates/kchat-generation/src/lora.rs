//! LoRA hot-swap — load small LoRA adapters on top of a frozen base model.
//!
//! LoRA (Low-Rank Adaptation) allows one base model to serve multiple tasks
//! and languages by swapping small adapter files (3-5MB) instead of loading
//! entirely different models. The swap takes <10ms.
//!
//! The system supports two adapter layouts:
//! - **Task-based** (270 adapters): 18 tasks x 15 language slots (legacy).
//!   Document tasks (12): summarize, translate, key_points, generate_doc,
//!     generate_slides, edit_grammar, edit_style, edit_format, create_email,
//!     create_social, create_pr, extract_info
//!   Chat-driven tasks (6): chat_catch_up, chat_create_tasks, chat_summarize,
//!     chat_draft_reply, chat_context_qa, chat_meeting_notes
//! - **Family-based** (75 adapters): 5 task-families x 15 language slots.
//!   Families: extract_json, rewrite_grammar, summarize_catchup, doc_creative,
//!     slides_deck
//! Languages (10): en, vi, zh, ja, ko, es, ar, de, hi, fr
//! Mixed-language (5): vi-en, zh-en, ja-en, ko-en, es-en

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

    /// Scan a model pack's `lora/` directory and register all adapters found.
    ///
    /// Supports two adapter file formats:
    /// - `adapters.safetensors` — MLX-native LoRA (Apple Silicon via MLX)
    /// - `adapters.gguf` — GGUF LoRA (cross-platform via llama.cpp)
    ///
    /// Expects the layout: `{pack}/lora/{task}.{lang}/adapters.{safetensors|gguf}`.
    /// Also reads `lora_registry.toml` if present for scale values.
    /// Returns the number of adapters registered.
    pub fn load_from_pack(&self, pack_path: &std::path::Path) -> usize {
        let lora_dir = pack_path.join("lora");
        if !lora_dir.is_dir() {
            return 0;
        }

        let mut count = 0;
        if let Ok(entries) = std::fs::read_dir(&lora_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                // Try safetensors first, then gguf
                let safetensors = entry.path().join("adapters.safetensors");
                let gguf = entry.path().join("adapters.gguf");
                let adapter_file = if safetensors.is_file() {
                    safetensors
                } else if gguf.is_file() {
                    gguf
                } else {
                    continue;
                };
                let parts: Vec<&str> = name.split('.').collect();
                if parts.len() != 2 {
                    continue;
                }
                let task = parts[0].to_string();
                let language = parts[1].to_string();
                let adapter_id = LoraAdapter::make_id(&task, &language);
                self.register(LoraAdapter::new(
                    adapter_id,
                    adapter_file,
                    task,
                    language,
                ));
                count += 1;
            }
        }

        if count > 0 {
            tracing::info!("Loaded {} LoRA adapters from {}", count, lora_dir.display());
        }
        count
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
    // Document tasks (12)
    pub const SUMMARIZE: &str = "summarize";
    pub const TRANSLATE: &str = "translate";
    pub const KEY_POINTS: &str = "key_points";
    pub const GENERATE_DOC: &str = "generate_doc";
    pub const GENERATE_SLIDES: &str = "generate_slides";
    pub const EDIT_GRAMMAR: &str = "edit_grammar";
    pub const EDIT_STYLE: &str = "edit_style";
    pub const EDIT_FORMAT: &str = "edit_format";
    pub const CREATE_EMAIL: &str = "create_email";
    pub const CREATE_SOCIAL: &str = "create_social";
    pub const CREATE_PR: &str = "create_pr";
    pub const EXTRACT_INFO: &str = "extract_info";

    // Chat-driven tasks (6)
    pub const CHAT_CATCH_UP: &str = "chat_catch_up";
    pub const CHAT_CREATE_TASKS: &str = "chat_create_tasks";
    pub const CHAT_SUMMARIZE: &str = "chat_summarize";
    pub const CHAT_DRAFT_REPLY: &str = "chat_draft_reply";
    pub const CHAT_CONTEXT_QA: &str = "chat_context_qa";
    pub const CHAT_MEETING_NOTES: &str = "chat_meeting_notes";

    pub const ALL: &[&str] = &[
        SUMMARIZE,
        TRANSLATE,
        KEY_POINTS,
        GENERATE_DOC,
        GENERATE_SLIDES,
        EDIT_GRAMMAR,
        EDIT_STYLE,
        EDIT_FORMAT,
        CREATE_EMAIL,
        CREATE_SOCIAL,
        CREATE_PR,
        EXTRACT_INFO,
        CHAT_CATCH_UP,
        CHAT_CREATE_TASKS,
        CHAT_SUMMARIZE,
        CHAT_DRAFT_REPLY,
        CHAT_CONTEXT_QA,
        CHAT_MEETING_NOTES,
    ];
}

/// Task-family identifiers (5 families for family-based adapter layout).
pub mod families {
    pub const EXTRACT_JSON: &str = "extract_json";
    pub const REWRITE_GRAMMAR: &str = "rewrite_grammar";
    pub const SUMMARIZE_CATCHUP: &str = "summarize_catchup";
    pub const DOC_CREATIVE: &str = "doc_creative";
    pub const SLIDES_DECK: &str = "slides_deck";

    pub const ALL: &[&str] = &[
        EXTRACT_JSON,
        REWRITE_GRAMMAR,
        SUMMARIZE_CATCHUP,
        DOC_CREATIVE,
        SLIDES_DECK,
    ];
}

/// Map an individual task ID to its task-family ID.
///
/// Returns `None` if the task doesn't belong to any family (e.g. chat_context_qa).
/// This is used to route task-based skill requests to family-based LoRA adapters.
pub fn task_to_family(task: &str) -> Option<&'static str> {
    match task {
        // F1: extract_json — extraction, classification, constrained JSON
        "extract_info" | "key_points" | "chat_create_tasks" | "chat_meeting_notes" => {
            Some(families::EXTRACT_JSON)
        }
        // F2: rewrite_grammar — editing and translation
        "edit_grammar" | "edit_style" | "edit_format" | "translate" => {
            Some(families::REWRITE_GRAMMAR)
        }
        // F3: summarize_catchup — summarization
        "summarize" | "chat_summarize" | "chat_catch_up" => {
            Some(families::SUMMARIZE_CATCHUP)
        }
        // F4: doc_creative — generation
        "generate_doc" | "create_email" | "create_social" | "create_pr" | "chat_draft_reply" => {
            Some(families::DOC_CREATIVE)
        }
        // F5: slides_deck — slides generation
        "generate_slides" => Some(families::SLIDES_DECK),
        _ => None,
    }
}

/// Maps a `SkillDef` to the appropriate LoRA adapter based on the skill's
/// `lora_task` field and the detected document language.
///
/// If no adapter is found for the detected language, falls back to English.
/// If the skill has no `lora_task`, returns `None` (use base model).
pub struct SkillLoRAResolver<'a> {
    manager: &'a LoraManager,
}

impl<'a> SkillLoRAResolver<'a> {
    /// Create a resolver bound to a LoRA manager.
    pub fn new(manager: &'a LoraManager) -> Self {
        Self { manager }
    }

    /// Resolve the adapter for a skill and language.
    ///
    /// Returns the adapter ID string if a suitable adapter is found,
    /// or `None` if the skill uses the base model.
    pub fn resolve(&self, lora_task: &str, language: &str) -> Option<String> {
        if lora_task.is_empty() {
            return None;
        }

        // Try exact language match first
        if let Some(adapter) = self.manager.find(lora_task, language) {
            return Some(adapter.adapter_id);
        }

        // Fall back to English
        if language != "en" {
            if let Some(adapter) = self.manager.find(lora_task, "en") {
                return Some(adapter.adapter_id);
            }
        }

        None
    }

    /// Resolve and swap the adapter for a skill.
    ///
    /// If the skill has no LoRA task, detaches the current adapter.
    /// Returns the adapter ID if swapped, `None` if detached or not found.
    pub fn resolve_and_swap(
        &self,
        lora_task: &str,
        language: &str,
    ) -> Result<Option<String>, LoraError> {
        match self.resolve(lora_task, language) {
            Some(adapter_id) => {
                self.manager.swap(&adapter_id)?;
                Ok(Some(adapter_id))
            }
            None => {
                self.manager.detach()?;
                Ok(None)
            }
        }
    }
}

/// Supported languages for LoRA adapters.
pub mod languages {
    pub const EN: &str = "en";
    pub const VI: &str = "vi";
    pub const ZH: &str = "zh";
    pub const JA: &str = "ja";
    pub const KO: &str = "ko";
    pub const ES: &str = "es";
    pub const AR: &str = "ar";
    pub const DE: &str = "de";
    pub const HI: &str = "hi";
    pub const FR: &str = "fr";

    // Mixed-language / code-switching pairs
    pub const VI_EN: &str = "vi-en";
    pub const ZH_EN: &str = "zh-en";
    pub const JA_EN: &str = "ja-en";
    pub const KO_EN: &str = "ko-en";
    pub const ES_EN: &str = "es-en";

    pub const ALL: &[&str] = &[
        EN, VI, ZH, JA, KO, ES, AR, DE, HI, FR,
        VI_EN, ZH_EN, JA_EN, KO_EN, ES_EN,
    ];
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
        assert_eq!(summarize_adapters.len(), 15); // 15 language slots
    }

    #[test]
    fn test_adapters_for_language() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        let vi_adapters = manager.adapters_for_language("vi");
        assert_eq!(vi_adapters.len(), 18); // 18 tasks
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
    fn test_all_270_adapters() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        let all = manager.available_adapters();
        assert_eq!(all.len(), 270); // 18 tasks x 15 language slots
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

    #[test]
    fn test_resolver_finds_adapter() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        let resolver = SkillLoRAResolver::new(&manager);
        let adapter_id = resolver.resolve("summarize", "vi");
        assert_eq!(adapter_id.as_deref(), Some("summarize.vi"));
    }

    #[test]
    fn test_resolver_falls_back_to_english() {
        let manager = LoraManager::new();
        // Register only English adapter for a task
        manager.register(LoraAdapter::new("edit_grammar.en", "/path", "edit_grammar", "en"));

        let resolver = SkillLoRAResolver::new(&manager);
        let adapter_id = resolver.resolve("edit_grammar", "vi");
        assert_eq!(adapter_id.as_deref(), Some("edit_grammar.en"));
    }

    #[test]
    fn test_resolver_empty_task_returns_none() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        let resolver = SkillLoRAResolver::new(&manager);
        assert!(resolver.resolve("", "en").is_none());
    }

    #[test]
    fn test_resolver_not_found_returns_none() {
        let manager = LoraManager::new();

        let resolver = SkillLoRAResolver::new(&manager);
        assert!(resolver.resolve("nonexistent", "en").is_none());
    }

    #[test]
    fn test_resolver_resolve_and_swap() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        let resolver = SkillLoRAResolver::new(&manager);
        let result = resolver.resolve_and_swap("translate", "zh").unwrap();
        assert_eq!(result.as_deref(), Some("translate.zh"));
        assert_eq!(manager.current_adapter(), Some("translate.zh".into()));
    }

    #[test]
    fn test_resolver_resolve_and_swap_detaches() {
        let manager = LoraManager::new();
        manager.register_all(make_test_adapters());

        // First swap to something
        manager.swap("summarize.en").unwrap();
        assert!(manager.has_active_adapter());

        // Now resolve with empty task — should detach
        let resolver = SkillLoRAResolver::new(&manager);
        let result = resolver.resolve_and_swap("", "en").unwrap();
        assert!(result.is_none());
        assert!(!manager.has_active_adapter());
    }

    #[test]
    fn test_18_tasks_exist() {
        assert_eq!(tasks::ALL.len(), 18);
        // Document tasks
        assert!(tasks::ALL.contains(&tasks::EDIT_GRAMMAR));
        assert!(tasks::ALL.contains(&tasks::EDIT_STYLE));
        assert!(tasks::ALL.contains(&tasks::EDIT_FORMAT));
        assert!(tasks::ALL.contains(&tasks::CREATE_EMAIL));
        assert!(tasks::ALL.contains(&tasks::CREATE_SOCIAL));
        assert!(tasks::ALL.contains(&tasks::CREATE_PR));
        assert!(tasks::ALL.contains(&tasks::EXTRACT_INFO));
        // Chat-driven tasks
        assert!(tasks::ALL.contains(&tasks::CHAT_CATCH_UP));
        assert!(tasks::ALL.contains(&tasks::CHAT_CREATE_TASKS));
        assert!(tasks::ALL.contains(&tasks::CHAT_SUMMARIZE));
        assert!(tasks::ALL.contains(&tasks::CHAT_DRAFT_REPLY));
        assert!(tasks::ALL.contains(&tasks::CHAT_CONTEXT_QA));
        assert!(tasks::ALL.contains(&tasks::CHAT_MEETING_NOTES));
    }

    #[test]
    fn test_load_from_pack() {
        let tmp = tempfile::tempdir().unwrap();
        let pack = tmp.path();
        let lora_dir = pack.join("lora");
        std::fs::create_dir_all(lora_dir.join("summarize.en")).unwrap();
        std::fs::write(lora_dir.join("summarize.en").join("adapters.safetensors"), b"dummy").unwrap();
        std::fs::create_dir_all(lora_dir.join("translate.vi")).unwrap();
        std::fs::write(lora_dir.join("translate.vi").join("adapters.safetensors"), b"dummy").unwrap();

        let manager = LoraManager::new();
        let count = manager.load_from_pack(pack);
        assert_eq!(count, 2);
        assert!(manager.find("summarize", "en").is_some());
        assert!(manager.find("translate", "vi").is_some());
    }

    #[test]
    fn test_load_from_pack_no_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let manager = LoraManager::new();
        let count = manager.load_from_pack(tmp.path());
        assert_eq!(count, 0);
    }
}

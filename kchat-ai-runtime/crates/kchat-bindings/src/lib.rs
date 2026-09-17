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
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(not(feature = "desktop"), derive(Clone, Copy))]
#[cfg_attr(feature = "mobile", derive(uniffi::Enum))]
#[cfg_attr(feature = "desktop", napi_derive::napi)]
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
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(not(feature = "desktop"), derive(Clone, Copy))]
#[cfg_attr(feature = "mobile", derive(uniffi::Enum))]
#[cfg_attr(feature = "desktop", napi_derive::napi)]
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
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiSafetyResult {
    pub action: FfiSafetyAction,
    pub severity: u8,
    pub category: u32,
    pub confidence: f64,
    pub reason_codes: Vec<String>,
    pub used_encoder: bool,
    pub used_slm: bool,
    pub duration_us: i64,
    pub rationale_id: String,
    pub resource_link_id: Option<String>,
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
            duration_us: result.duration_us as i64,
            rationale_id: v.rationale_id.clone(),
            resource_link_id: v.resource_link_id.clone(),
        }
    }
}

/// Retrieval result — the main FFI return type for context retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiRetrievalResult {
    pub evidence_id: String,
    pub score: f64,
    pub fts_score: f64,
    pub recency_score: f64,
    /// Dense-vector cosine score (0 when no embedding provider).
    pub vector_score: f64,
    /// Cross-encoder rerank score (0 when no reranker ran — Low/Medium tier
    /// or no encoder attached).
    pub rerank_score: f64,
}

/// Generation result — the main FFI return type for generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiGenerationResult {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub ttft_ms: i64,
    pub total_ms: i64,
    pub tokens_per_second: f64,
    pub grammar_valid: bool,
}

impl From<kchat_generation::backend::GenerationResult> for FfiGenerationResult {
    fn from(r: kchat_generation::backend::GenerationResult) -> Self {
        Self {
            text: r.text,
            prompt_tokens: r.prompt_tokens,
            completion_tokens: r.completion_tokens,
            ttft_ms: r.ttft_ms as i64,
            total_ms: r.total_ms as i64,
            tokens_per_second: r.tokens_per_second,
            grammar_valid: r.grammar_valid,
        }
    }
}

/// Tool plan validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiValidationResult {
    pub valid: bool,
    pub step_count: u32,
    pub error: Option<String>,
}

/// Skill descriptor — one entry of the skill registry, for UI listing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiSkillInfo {
    pub id: String,
    pub label: String,
    pub description: String,
    pub icon: String,
    /// "one_click" | "prompt_input" | "form_input" | "multi_step"
    pub mode: String,
    /// "free_text" | "json_schema" | "regex"
    pub grammar: String,
    pub min_tier: FfiDeviceTier,
    /// LoRA task family this skill resolves to (empty when none).
    pub lora_task: String,
}

/// Skill execution request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiSkillRequest {
    /// User instruction / selection / topic.
    pub input: String,
    /// Document text the skill operates on.
    pub context: String,
    /// Optional keyword hint.
    pub keywords: String,
    /// Sub-variant context.
    pub variant_context: String,
    /// Response-language hint for LoRA slot resolution ("en", "vi-en", …).
    pub language: Option<String>,
    /// Scope UUIDs authorizing retrieval; empty = no retrieval.
    pub scope_ids: Vec<String>,
    /// User UUID for retrieval ACL checks (required when scope_ids non-empty).
    pub user_id: Option<String>,
    /// Role grants for retrieval ACL checks (role-authorized scopes).
    pub roles: Vec<String>,
    /// Best-of-n candidates (High tier only; 0/1 = single stream).
    pub best_of_n: u32,
}

/// A generation stream event pushed to listeners.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiStreamEvent {
    /// "token" | "complete" | "cancelled" | "error"
    pub kind: String,
    /// Token text, cancel reason, or error message.
    pub text: String,
    pub total_tokens: u32,
    pub duration_ms: i64,
}

impl From<kchat_generation::stream::StreamEvent> for FfiStreamEvent {
    fn from(ev: kchat_generation::stream::StreamEvent) -> Self {
        use kchat_generation::stream::StreamEvent as E;
        match ev {
            E::Token { text } => Self {
                kind: "token".into(),
                text,
                total_tokens: 0,
                duration_ms: 0,
            },
            E::Complete {
                total_tokens,
                duration_ms,
            } => Self {
                kind: "complete".into(),
                text: String::new(),
                total_tokens,
                duration_ms: duration_ms as i64,
            },
            E::Cancelled { reason } => Self {
                kind: "cancelled".into(),
                text: reason,
                total_tokens: 0,
                duration_ms: 0,
            },
            E::Error { message } => Self {
                kind: "error".into(),
                text: message,
                total_tokens: 0,
                duration_ms: 0,
            },
        }
    }
}

/// Enriched skill outcome — generation result plus post-check metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiSkillOutcome {
    pub generation: FfiGenerationResult,
    pub output_action: FfiSafetyAction,
    pub output_blocked: bool,
    pub retrieved_passages: u32,
    pub lora_adapter: Option<String>,
}

/// Router suggestion — natural-language request → skill match.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiRouteSuggestion {
    pub skill_id: String,
    pub score: f64,
    /// "embedding" | "keyword" — truthful match-method reporting.
    pub method: String,
}

/// Image search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiImageResult {
    pub id: String,
    pub provider: String,
    pub url: String,
    pub thumb_url: String,
    pub width: u32,
    pub height: u32,
    pub alt_text: String,
    /// "landscape" | "portrait" | "square"
    pub orientation: String,
    pub photographer: String,
    pub license: String,
}

/// Errors surfaced across the FFI boundary.
#[derive(Debug, thiserror::Error)]
#[cfg_attr(feature = "mobile", derive(uniffi::Error))]
pub enum FfiError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for FfiError {
    fn from(s: String) -> Self {
        FfiError::Message(s)
    }
}

impl From<FfiError> for String {
    fn from(e: FfiError) -> String {
        e.to_string()
    }
}

/// Parse a scope UUID string.
fn parse_scope_id(s: &str) -> Result<kchat_context::ScopeId, FfiError> {
    uuid::Uuid::parse_str(s.trim())
        .map(kchat_context::ScopeId::from_uuid)
        .map_err(|e| FfiError::Message(format!("invalid scope id '{s}': {e}")))
}

/// Build a scope filter for retrieval ACL checks.
fn build_scope_filter(
    scope_ids: &[String],
    user_id: &str,
    roles: &[String],
) -> Result<kchat_context::ScopeFilter, FfiError> {
    let user = uuid::Uuid::parse_str(user_id.trim())
        .map_err(|e| FfiError::Message(format!("invalid user_id: {e}")))?;
    let allowed_scopes = scope_ids
        .iter()
        .map(|s| parse_scope_id(s))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(kchat_context::ScopeFilter {
        allowed_scopes,
        denied_scopes: Vec::new(),
        user_id: user,
        roles: roles.to_vec(),
    })
}

// ============================================================================
// High-level FFI facade — the main entry point for platform code
// ============================================================================

/// Sink for generation stream events — implemented by UniFFI callback
/// objects and N-API threadsafe functions.
pub trait StreamEventSink: Send + Sync {
    fn emit(&self, event: FfiStreamEvent);
}

/// The high-level KChat AI Runtime facade.
///
/// This struct is exposed via UniFFI (Swift/Kotlin) and N-API (Node.js).
/// Platform code creates an instance, configures it, and calls the methods.
#[cfg_attr(feature = "mobile", derive(uniffi::Object))]
pub struct KChatAiRuntime {
    /// Shared with the orchestrator — encoder/SLM attachments apply to both
    /// direct `classify_safety` calls and in-pipeline safety checks.
    safety: std::sync::Arc<kchat_safety::classify::SafetyClassifier>,
    tier: kchat_core::tier::DeviceTier,
    platform: String,
    /// Cached device capabilities from the last probe
    caps: Option<kchat_core::capability::DeviceCapabilities>,

    backend:
        parking_lot::Mutex<Option<std::sync::Arc<dyn kchat_generation::backend::BackendAdapter>>>,
    store: parking_lot::Mutex<Option<std::sync::Arc<kchat_context::ContextStore>>>,
    embeddings: parking_lot::Mutex<Option<std::sync::Arc<kchat_context::EmbeddingManager>>>,
    reranker: parking_lot::Mutex<Option<std::sync::Arc<dyn kchat_context::Reranker>>>,
    memory_scope: parking_lot::Mutex<Option<kchat_context::ScopeId>>,
    /// Rebuilt whenever a component is attached. Always `Some` — built
    /// eagerly so routing/listing work before any component attaches.
    orchestrator: parking_lot::RwLock<Option<std::sync::Arc<kchat_runtime::Orchestrator>>>,
    model_manager: parking_lot::Mutex<Option<kchat_core::model_manager::ModelManager>>,
    images: kchat_image::ImageSearchRegistry,
    tool_validator: parking_lot::Mutex<kchat_action::toolplan::ToolPlanValidator>,
    /// Running skill streams by run id — supports cancellation. Shared with
    /// worker threads so finished runs deregister themselves.
    runs: std::sync::Arc<
        parking_lot::Mutex<
            std::collections::HashMap<
                String,
                std::sync::Arc<kchat_generation::stream::StreamHandle>,
            >,
        >,
    >,
}

impl KChatAiRuntime {
    fn with_caps(
        platform: &str,
        tier: kchat_core::tier::DeviceTier,
        caps: Option<kchat_core::capability::DeviceCapabilities>,
    ) -> Self {
        let rt = Self {
            safety: std::sync::Arc::new(kchat_safety::classify::SafetyClassifier::new()),
            tier,
            platform: platform.to_string(),
            caps,
            backend: parking_lot::Mutex::new(None),
            store: parking_lot::Mutex::new(None),
            embeddings: parking_lot::Mutex::new(None),
            reranker: parking_lot::Mutex::new(None),
            memory_scope: parking_lot::Mutex::new(None),
            orchestrator: parking_lot::RwLock::new(None),
            model_manager: parking_lot::Mutex::new(None),
            images: kchat_image::ImageSearchRegistry::from_env(),
            tool_validator: parking_lot::Mutex::new(
                kchat_action::toolplan::ToolPlanValidator::new(),
            ),
            runs: std::sync::Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
        };
        // Build an empty orchestrator eagerly so keyword routing, skill
        // listing, and safety classification work before any attach.
        rt.rebuild_orchestrator();
        rt
    }

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
                Self::with_caps(platform, tier, Some(caps))
            }
            Err(e) => {
                // Fail-safe: default to Low tier when hardware detection fails
                tracing::warn!("Capability probe failed, defaulting to Low tier: {}", e);
                Self::with_caps(platform, kchat_core::tier::DeviceTier::Low, None)
            }
        }
    }

    /// Create a runtime with an explicit tier (for testing).
    pub fn with_tier(platform: &str, tier: kchat_core::tier::DeviceTier) -> Self {
        Self::with_caps(platform, tier, None)
    }

    /// Rebuild the orchestrator from the currently attached components.
    /// Called internally after every attach — the orchestrator composes
    /// backend + store + embeddings + reranker + shared safety classifier.
    fn rebuild_orchestrator(&self) {
        let mut builder = kchat_runtime::Orchestrator::builder()
            .tier(self.tier)
            .safety(self.safety.clone());
        if let Some(b) = self.backend.lock().clone() {
            builder = builder.backend(b);
        }
        if let Some(s) = self.store.lock().clone() {
            builder = builder.context_store(s);
        }
        if let Some(e) = self.embeddings.lock().clone() {
            builder = builder.embeddings(e);
        }
        if let Some(r) = self.reranker.lock().clone() {
            builder = builder.reranker(r);
        }
        if let Some(scope) = *self.memory_scope.lock() {
            builder = builder.memory_scope(scope);
        }
        let new_orc = std::sync::Arc::new(builder.build());
        // Carry over buffered memory turns — a rebuild on attach must not
        // silently drop unflushed conversation history.
        if let Some(old) = self.orchestrator.write().replace(new_orc.clone()) {
            new_orc.memory().adopt_from(old.memory());
        }
    }

    fn require_orchestrator(
        &self,
    ) -> Result<std::sync::Arc<kchat_runtime::Orchestrator>, FfiError> {
        self.orchestrator
            .read()
            .clone()
            .ok_or_else(|| FfiError::Message("no components attached to runtime".into()))
    }

    /// Attach a generation backend trait object (test/dynamic path).
    pub fn attach_backend_dyn(
        &self,
        backend: std::sync::Arc<dyn kchat_generation::backend::BackendAdapter>,
    ) {
        *self.backend.lock() = Some(backend);
        self.rebuild_orchestrator();
    }

    /// Load an in-process llama.cpp backend from a GGUF model file.
    ///
    /// Runs entirely in-process — no subprocess, no HTTP — so it works on
    /// iOS/Android. Returns `false` if the model fails to load; the runtime
    /// continues without a backend.
    #[cfg(feature = "llamacpp")]
    pub fn attach_llamacpp_backend(&self, model_path: &str) -> bool {
        use kchat_generation::backend::BackendAdapter;
        let backend = kchat_generation::backends::llamacpp::LlamaCppBackend::new();
        let backend_type = match self.platform.as_str() {
            "ios" | "macos" => kchat_generation::backend::BackendType::LlamaCppMetal,
            "android" | "windows" => kchat_generation::backend::BackendType::LlamaCppVulkan,
            _ => kchat_generation::backend::BackendType::LlamaCppCpu,
        };
        let config = kchat_generation::backend::BackendConfig::for_tier(
            backend_type,
            "ffi-gguf",
            model_path,
            self.tier,
            &self.platform,
        );
        match backend.load(&config) {
            Ok(()) => {
                self.attach_backend_dyn(std::sync::Arc::new(backend));
                true
            }
            Err(e) => {
                tracing::warn!("llama.cpp backend load failed: {}", e);
                false
            }
        }
    }

    /// Whether a generation backend is attached and loaded.
    pub fn backend_attached(&self) -> bool {
        self.backend
            .lock()
            .as_ref()
            .map(|b| b.is_loaded())
            .unwrap_or(false)
    }

    /// Open (or create) the encrypted context store and wire it into the
    /// pipeline — enables retrieval and memory write-back.
    pub fn open_context_store(
        &self,
        path: &str,
        db_password: &str,
        master_key_hex: &str,
    ) -> Result<(), FfiError> {
        let key_bytes = hex::decode(master_key_hex.trim())
            .map_err(|e| FfiError::Message(format!("master key hex: {e}")))?;
        let master_key: [u8; 32] = key_bytes
            .try_into()
            .map_err(|_| FfiError::Message("master key must be 32 bytes".into()))?;
        let config = match self.tier {
            kchat_core::tier::DeviceTier::Low => {
                kchat_context::ContextStoreConfig::for_low_tier(db_password.into(), master_key)
            }
            kchat_core::tier::DeviceTier::Medium => {
                kchat_context::ContextStoreConfig::for_medium_tier(db_password.into(), master_key)
            }
            kchat_core::tier::DeviceTier::High => {
                kchat_context::ContextStoreConfig::for_high_tier(db_password.into(), master_key)
            }
        };
        let store = kchat_context::ContextStore::open(std::path::Path::new(path), &config)
            .map_err(|e| FfiError::Message(format!("open store: {e}")))?;
        *self.store.lock() = Some(std::sync::Arc::new(store));
        self.rebuild_orchestrator();
        Ok(())
    }

    /// Insert evidence into the context store (embed-on-write when an
    /// embedding provider is attached). Returns the new evidence id.
    pub fn insert_evidence(
        &self,
        scope_id: &str,
        content: &str,
        source_ref: Option<String>,
        importance: u8,
        language_tag: Option<String>,
    ) -> Result<String, FfiError> {
        let scope = parse_scope_id(scope_id)?;
        let store = self
            .store
            .lock()
            .clone()
            .ok_or_else(|| FfiError::Message("context store not open".into()))?;
        let nonce = kchat_context::encryption::AeadNonce::random()
            .map_err(|e| FfiError::Message(format!("nonce: {e}")))?;
        let evidence = kchat_context::Evidence {
            id: kchat_context::EvidenceId::new(),
            scope_id: scope,
            content_hash: blake3::hash(content.as_bytes()).to_hex().to_string(),
            encrypted_body: Vec::new(),
            nonce: nonce.0.to_vec(),
            source_ref,
            importance,
            language_tag,
            created_at: chrono::Utc::now().timestamp(),
            fts_content: content.into(),
        };
        let id = evidence.id.0.to_string();
        match self.embeddings.lock().clone() {
            Some(embs) => store.insert_indexed(&evidence, &embs),
            None => store.insert(&evidence),
        }
        .map_err(|e| FfiError::Message(format!("insert evidence: {e}")))?;
        Ok(id)
    }

    /// Forget a scope — deletes evidence, FTS rows, and vectors (right to
    /// be forgotten).
    pub fn forget_scope(&self, scope_id: &str) -> Result<(), FfiError> {
        let scope = parse_scope_id(scope_id)?;
        let store = self
            .store
            .lock()
            .clone()
            .ok_or_else(|| FfiError::Message("context store not open".into()))?;
        store
            .forget_scope(scope)
            .map_err(|e| FfiError::Message(format!("forget scope: {e}")))
    }

    /// Set the scope used for conversation-memory write-back.
    pub fn set_memory_scope(&self, scope_id: &str) -> Result<(), FfiError> {
        *self.memory_scope.lock() = Some(parse_scope_id(scope_id)?);
        self.rebuild_orchestrator();
        Ok(())
    }

    /// Hybrid retrieval over the context store (FTS + dense + RRF, reranked
    /// on High tier).
    pub fn retrieve(
        &self,
        query: &str,
        scope_ids: Vec<String>,
        user_id: &str,
        roles: Vec<String>,
        limit: u32,
    ) -> Result<Vec<FfiRetrievalResult>, FfiError> {
        let store = self
            .store
            .lock()
            .clone()
            .ok_or_else(|| FfiError::Message("context store not open".into()))?;
        let filter = build_scope_filter(&scope_ids, user_id, &roles)?;
        let tier = match self.tier {
            kchat_core::tier::DeviceTier::Low => kchat_context::RetrievalTier::Low,
            kchat_core::tier::DeviceTier::Medium => kchat_context::RetrievalTier::Medium,
            kchat_core::tier::DeviceTier::High => kchat_context::RetrievalTier::High,
        };
        let embeddings_guard = self.embeddings.lock();
        let reranker_guard = self.reranker.lock();
        let mut retriever = kchat_context::Retriever::new(&store, tier);
        if let Some(embs) = embeddings_guard.as_ref() {
            retriever = retriever.with_embeddings(embs);
        }
        if let Some(r) = reranker_guard.as_ref() {
            retriever = retriever.with_reranker(r.as_ref());
        }
        let results = retriever
            .retrieve(query, &filter, limit.max(1) as usize)
            .map_err(|e| FfiError::Message(format!("retrieve: {e}")))?;
        Ok(results
            .into_iter()
            .map(|r| FfiRetrievalResult {
                evidence_id: r.evidence_id.0.to_string(),
                score: r.score,
                fts_score: r.fts_score,
                recency_score: r.recency_score,
                vector_score: r.vector_score,
                rerank_score: r.rerank_score,
            })
            .collect())
    }

    /// List all registered skills for UI display.
    pub fn list_skills(&self) -> Vec<FfiSkillInfo> {
        let registry = kchat_generation::skills::SkillRegistry::new();
        registry
            .all()
            .iter()
            .map(|s| FfiSkillInfo {
                id: s.id.clone(),
                label: s.label.clone(),
                description: s.description.clone(),
                icon: s.icon.clone(),
                mode: format!("{:?}", s.mode).to_lowercase(),
                grammar: format!("{:?}", s.grammar_type).to_lowercase(),
                min_tier: match s.min_tier {
                    kchat_generation::skills::SkillTier::Low => FfiDeviceTier::Low,
                    kchat_generation::skills::SkillTier::Medium => FfiDeviceTier::Medium,
                    kchat_generation::skills::SkillTier::High => FfiDeviceTier::High,
                },
                lora_task: s.lora_task.clone(),
            })
            .collect()
    }

    /// Route a natural-language request to skills (embedding match when an
    /// embedder is attached, keyword fallback otherwise).
    pub fn route_request(
        &self,
        text: &str,
        top_k: u32,
    ) -> Result<Vec<FfiRouteSuggestion>, FfiError> {
        let orc = self.require_orchestrator()?;
        Ok(orc
            .router()
            .route(text, top_k.max(1) as usize)
            .into_iter()
            .map(|s| FfiRouteSuggestion {
                skill_id: s.skill_id,
                score: s.score,
                method: s.method.to_string(),
            })
            .collect())
    }

    /// Run a skill end-to-end, pushing stream events to `sink`. Blocks the
    /// calling thread until completion — run it on a background thread from
    /// platform code. Returns the enriched outcome.
    pub fn run_skill(
        &self,
        skill_id: &str,
        req: FfiSkillRequest,
        sink: &dyn StreamEventSink,
    ) -> Result<FfiSkillOutcome, FfiError> {
        let orc = self.require_orchestrator()?;
        let mut run = orc
            .run_skill(skill_id, self.build_skill_request(req)?)
            .map_err(|e| FfiError::Message(e.to_string()))?;
        // Register so `cancel_run` can reach this run (id from
        // `active_run_ids`) even though the calling thread is blocked.
        let run_key = run.handle.id.0.to_string();
        self.runs.lock().insert(run_key.clone(), run.handle.clone());
        Self::drain_events(&mut run.events, sink);
        self.runs.lock().remove(&run_key);
        let outcome = run
            .outcome
            .blocking_recv()
            .map_err(|_| FfiError::Message("skill outcome channel closed".into()))?
            .map_err(|e| FfiError::Message(e.to_string()))?;
        Ok(Self::map_outcome(outcome))
    }

    /// Run a skill on a background thread. Events flow to `sink`; the
    /// returned run id can be passed to [`Self::cancel_run`].
    pub fn run_skill_async(
        &self,
        skill_id: &str,
        req: FfiSkillRequest,
        sink: std::sync::Arc<dyn StreamEventSink>,
    ) -> Result<String, FfiError> {
        let orc = self.require_orchestrator()?;
        let run = orc
            .run_skill(skill_id, self.build_skill_request(req)?)
            .map_err(|e| FfiError::Message(e.to_string()))?;
        let run_id = uuid::Uuid::new_v4().to_string();
        self.runs.lock().insert(run_id.clone(), run.handle.clone());
        let runs = self.runs.clone();
        let worker_run_id = run_id.clone();
        std::thread::spawn(move || {
            let mut run = run;
            Self::drain_events(&mut run.events, sink.as_ref());
            if run.outcome.blocking_recv().is_err() {
                sink.emit(FfiStreamEvent {
                    kind: "error".into(),
                    text: "skill outcome channel closed".into(),
                    total_tokens: 0,
                    duration_ms: 0,
                });
            }
            runs.lock().remove(&worker_run_id);
        });
        Ok(run_id)
    }

    /// Run ids of in-flight skills — pass one to [`Self::cancel_run`] to
    /// cancel a run started on another thread (including blocking
    /// [`Self::run_skill`] calls).
    pub fn active_run_ids(&self) -> Vec<String> {
        self.runs.lock().keys().cloned().collect()
    }

    /// Cancel a running skill by run id (from [`Self::run_skill_async`] or
    /// [`Self::active_run_ids`]).
    pub fn cancel_run(&self, run_id: &str, reason: &str) -> bool {
        if let Some(handle) = self.runs.lock().get(run_id) {
            handle.cancel(reason.to_string());
            true
        } else {
            false
        }
    }

    /// Validate a ToolPlan JSON against registered tool manifests.
    pub fn validate_tool_plan(&self, plan_json: &str) -> FfiValidationResult {
        let plan: kchat_action::toolplan::ToolPlan = match serde_json::from_str(plan_json) {
            Ok(p) => p,
            Err(e) => {
                return FfiValidationResult {
                    valid: false,
                    step_count: 0,
                    error: Some(format!("invalid plan JSON: {e}")),
                }
            }
        };
        let step_count = plan.steps.len() as u32;
        match self.tool_validator.lock().validate(&plan) {
            Ok(results) => FfiValidationResult {
                valid: results.iter().all(|r| r.valid),
                step_count,
                error: None,
            },
            Err(e) => FfiValidationResult {
                valid: false,
                step_count,
                error: Some(e.to_string()),
            },
        }
    }

    /// Register a signed tool manifest for plan validation.
    pub fn register_tool_manifest(&self, manifest_json: &str) -> Result<(), FfiError> {
        let manifest: kchat_action::toolplan::ToolManifest = serde_json::from_str(manifest_json)
            .map_err(|e| FfiError::Message(format!("invalid manifest JSON: {e}")))?;
        self.tool_validator
            .lock()
            .register_manifest(manifest)
            .map_err(|e| FfiError::Message(e.to_string()))
    }

    /// Set the model cache directory used for pack downloads.
    pub fn set_model_cache_dir(&self, path: &str) {
        *self.model_manager.lock() = Some(kchat_core::model_manager::ModelManager::new(
            path, self.tier,
        ));
    }

    /// Download a model pack (manifest JSON), reporting byte progress to
    /// `progress(downloaded, total)`. Streams to disk in bounded memory,
    /// resumes partial downloads, verifies SHA-256.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn download_model(
        &self,
        manifest_json: &str,
        progress: &(dyn Fn(u64, u64) + Send + Sync),
    ) -> Result<String, FfiError> {
        let manifest: kchat_core::manifest::ModelPackManifest = serde_json::from_str(manifest_json)
            .map_err(|e| FfiError::Message(format!("invalid manifest JSON: {e}")))?;
        let guard = self.model_manager.lock();
        let manager = guard.as_ref().ok_or_else(|| {
            FfiError::Message("model cache dir not set — call set_model_cache_dir".into())
        })?;
        manager
            .ensure_pack_with_progress(&manifest, progress)
            .map(|p| p.to_string_lossy().into_owned())
            .map_err(|e| FfiError::Message(format!("download: {e}")))
    }

    /// Search stock-image providers. Requires API keys configured via
    /// environment (PEXELS_API_KEY, PIXABAY_API_KEY, UNSPLASH_ACCESS_KEY,
    /// SHUTTERSTOCK_API_TOKEN) — returns an error when none are set.
    pub fn search_images(&self, query: &str, limit: u32) -> Result<Vec<FfiImageResult>, FfiError> {
        let req = kchat_image::types::ImageSearchRequest::new(query)
            .with_per_page(limit.clamp(1, 80))
            .with_safesearch(true);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| FfiError::Message(format!("runtime: {e}")))?;
        let resp = rt
            .block_on(self.images.search(&req))
            .map_err(|e| FfiError::Message(format!("image search: {e}")))?;
        Ok(resp
            .results
            .into_iter()
            .map(|r| FfiImageResult {
                id: r.id,
                provider: r.provider,
                url: r.url,
                thumb_url: r.thumb_url,
                width: r.width,
                height: r.height,
                alt_text: r.alt_text,
                orientation: format!("{:?}", r.orientation).to_lowercase(),
                photographer: r.attribution.photographer,
                license: format!("{:?}", r.license).to_lowercase(),
            })
            .collect())
    }

    /// Flush pending conversation-memory turns to the store (summarized
    /// write-back when a backend is attached, raw otherwise).
    pub fn flush_memory(&self) -> Result<(), FfiError> {
        let orc = self.require_orchestrator()?;
        if let Some(backend) = self.backend.lock().clone() {
            orc.memory().flush_summarized(&backend);
        } else {
            orc.memory().flush_pending();
        }
        Ok(())
    }

    fn build_skill_request(
        &self,
        req: FfiSkillRequest,
    ) -> Result<kchat_runtime::SkillRequest, FfiError> {
        let scope_filter = if req.scope_ids.is_empty() {
            None
        } else {
            let user_id = req
                .user_id
                .as_deref()
                .ok_or_else(|| FfiError::Message("user_id required with scope_ids".into()))?;
            Some(build_scope_filter(&req.scope_ids, user_id, &req.roles)?)
        };
        Ok(kchat_runtime::SkillRequest {
            input: req.input,
            context: req.context,
            keywords: req.keywords,
            variant_context: req.variant_context,
            language: req.language,
            scope_filter,
            best_of_n: req.best_of_n as usize,
        })
    }

    fn drain_events(
        events: &mut tokio::sync::mpsc::UnboundedReceiver<kchat_generation::stream::StreamEvent>,
        sink: &dyn StreamEventSink,
    ) {
        while let Some(ev) = events.blocking_recv() {
            let terminal = !matches!(ev, kchat_generation::stream::StreamEvent::Token { .. });
            sink.emit(ev.into());
            if terminal {
                break;
            }
        }
    }

    fn map_outcome(outcome: kchat_runtime::SkillOutcome) -> FfiSkillOutcome {
        FfiSkillOutcome {
            generation: outcome.generation.into(),
            output_action: outcome.output_verdict.action.into(),
            output_blocked: outcome.output_blocked,
            retrieved_passages: outcome.retrieved_passages as u32,
            lora_adapter: outcome.lora_adapter,
        }
    }

    /// Attach the in-process GGUF encoder (mmBERT via llama.cpp).
    ///
    /// One shared session serves all three encoder roles: safety escalation,
    /// text embeddings (retrieval + routing), and cross-encoder reranking.
    /// Fully in-process — no subprocess, no HTTP — works on iOS/Android.
    /// Returns `false` if the encoder fails to load; classification continues
    /// on the deterministic path.
    #[cfg(feature = "gguf-encoder")]
    pub fn attach_gguf_encoder(
        &self,
        model_path: &str,
        heads_path: &str,
        intra_threads: u32,
    ) -> bool {
        let session = match kchat_encoder::GgufEncoderSession::new(
            model_path,
            heads_path,
            intra_threads.max(1) as usize,
        ) {
            Ok(s) => std::sync::Arc::new(s),
            Err(e) => {
                tracing::warn!("GGUF encoder load failed, deterministic only: {}", e);
                return false;
            }
        };
        self.safety
            .attach_encoder(Box::new(kchat_safety::GgufSafetyEncoder::from_shared(
                session.clone(),
            )));
        *self.embeddings.lock() = Some(std::sync::Arc::new(
            kchat_context::EmbeddingManager::new()
                .with_primary(Box::new(kchat_context::GgufEmbedder::new(session.clone()))),
        ));
        *self.reranker.lock() = Some(std::sync::Arc::new(kchat_context::GgufReranker::new(
            session,
        )));
        self.rebuild_orchestrator();
        true
    }

    /// Whether an encoder is attached and available for escalation.
    pub fn encoder_available(&self) -> bool {
        self.safety.has_encoder()
    }

    /// Classify a message for safety.
    pub fn classify_safety(&self, text: &str, is_group: bool) -> FfiSafetyResult {
        // Report truthful availability — the pipeline only escalates when a
        // real encoder/SLM is actually attached, not merely tier-eligible.
        let request = kchat_safety::classify::ClassifyRequest {
            text: text.to_string(),
            is_group,
            age_mode: None,
            relationship: None,
            encoder_available: self.safety.has_encoder(),
            slm_available: self.safety.has_slm(),
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

    /// Whether a generation backend is currently attached — the runtime can
    /// actually produce tokens right now. Distinct from `can_generate`,
    /// which reports tier capability: all tiers have a model assignment,
    /// but generation still requires `attach_backend`.
    pub fn generation_ready(&self) -> bool {
        self.backend.lock().is_some()
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
        self.caps
            .as_ref()
            .map(|c| c.allows_generative())
            .unwrap_or(false)
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
#[cfg_attr(feature = "mobile", derive(uniffi::Record))]
#[cfg_attr(feature = "desktop", napi_derive::napi(object))]
pub struct FfiDeviceCapabilities {
    pub platform: String,
    pub physical_memory: i64,
    pub safe_allocatable_memory: i64,
    pub cpu_arch: String,
    pub cpu_cores: u32,
    pub performance_cores: Option<u32>,
    pub isa_features: Vec<String>,
    pub gpu_backend: String,
    pub npu_provider: String,
    pub free_storage: i64,
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
            physical_memory: c.physical_memory as i64,
            safe_allocatable_memory: c.safe_allocatable_memory as i64,
            cpu_arch: c.cpu_arch.clone(),
            cpu_cores: c.cpu_cores,
            performance_cores: c.performance_cores,
            isa_features: c.isa_features.clone(),
            gpu_backend: format!("{:?}", c.gpu_backend).to_lowercase(),
            npu_provider: format!("{:?}", c.npu_provider).to_lowercase(),
            free_storage: c.free_storage as i64,
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
    use std::sync::Arc;

    // UniFFI uses the `#[uniffi::export]` macro to generate bindings.
    // The callback interface allows platform code to provide implementations.

    /// Callback for platform-specific device capability probing.
    #[uniffi::export(with_foreign)]
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

    /// Streaming listener — receives generation events as they happen.
    /// Called from a worker thread; implementations must be thread-safe.
    #[uniffi::export(with_foreign)]
    pub trait SkillStreamListener: Send + Sync {
        fn on_event(&self, event: FfiStreamEvent);
    }

    /// Download progress listener — receives (downloaded, total) bytes.
    #[uniffi::export(with_foreign)]
    pub trait DownloadProgressListener: Send + Sync {
        fn on_progress(&self, downloaded: u64, total: u64);
    }

    /// Adapter: UniFFI listener → internal event sink.
    struct UniStreamSink(Arc<dyn SkillStreamListener>);
    impl StreamEventSink for UniStreamSink {
        fn emit(&self, event: FfiStreamEvent) {
            self.0.on_event(event);
        }
    }

    /// Adapter: UniFFI listener → progress fn.
    struct UniProgressSink(Arc<dyn DownloadProgressListener>);
    impl UniProgressSink {
        fn call(&self, downloaded: u64, total: u64) {
            self.0.on_progress(downloaded, total);
        }
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

        /// Whether a generation backend is attached right now.
        pub fn uniffi_generation_ready(&self) -> bool {
            self.generation_ready()
        }

        /// Probe device capabilities.
        pub fn uniffi_probe_capabilities(&self) -> FfiDeviceCapabilities {
            self.probe_capabilities()
        }

        /// Whether a generation backend is attached and loaded.
        pub fn uniffi_backend_attached(&self) -> bool {
            self.backend_attached()
        }

        /// Whether an encoder is attached for safety escalation.
        pub fn uniffi_encoder_available(&self) -> bool {
            self.encoder_available()
        }

        /// Open the encrypted context store.
        pub fn uniffi_open_context_store(
            &self,
            path: String,
            db_password: String,
            master_key_hex: String,
        ) -> Result<(), FfiError> {
            self.open_context_store(&path, &db_password, &master_key_hex)
        }

        /// Insert evidence; returns the new evidence id.
        pub fn uniffi_insert_evidence(
            &self,
            scope_id: String,
            content: String,
            source_ref: Option<String>,
            importance: u8,
            language_tag: Option<String>,
        ) -> Result<String, FfiError> {
            self.insert_evidence(&scope_id, &content, source_ref, importance, language_tag)
        }

        /// Forget a scope and all its data.
        pub fn uniffi_forget_scope(&self, scope_id: String) -> Result<(), FfiError> {
            self.forget_scope(&scope_id)
        }

        /// Set the conversation-memory write-back scope.
        pub fn uniffi_set_memory_scope(&self, scope_id: String) -> Result<(), FfiError> {
            self.set_memory_scope(&scope_id)
        }

        /// Hybrid retrieval over the context store.
        pub fn uniffi_retrieve(
            &self,
            query: String,
            scope_ids: Vec<String>,
            user_id: String,
            roles: Vec<String>,
            limit: u32,
        ) -> Result<Vec<FfiRetrievalResult>, FfiError> {
            self.retrieve(&query, scope_ids, &user_id, roles, limit)
        }

        /// List all registered skills.
        pub fn uniffi_list_skills(&self) -> Vec<FfiSkillInfo> {
            self.list_skills()
        }

        /// Route a natural-language request to skills.
        pub fn uniffi_route_request(
            &self,
            text: String,
            top_k: u32,
        ) -> Result<Vec<FfiRouteSuggestion>, FfiError> {
            self.route_request(&text, top_k)
        }

        /// Run a skill, streaming events to `listener`. Blocks the calling
        /// thread — invoke from a background queue on the platform side.
        pub fn uniffi_run_skill(
            &self,
            skill_id: String,
            req: FfiSkillRequest,
            listener: Arc<dyn SkillStreamListener>,
        ) -> Result<FfiSkillOutcome, FfiError> {
            self.run_skill(&skill_id, req, &UniStreamSink(listener))
        }

        /// Run a skill on a background thread; returns a run id for
        /// `uniffi_cancel_run`.
        pub fn uniffi_run_skill_async(
            &self,
            skill_id: String,
            req: FfiSkillRequest,
            listener: Arc<dyn SkillStreamListener>,
        ) -> Result<String, FfiError> {
            self.run_skill_async(&skill_id, req, Arc::new(UniStreamSink(listener)))
        }

        /// Run ids of in-flight skills (for `uniffi_cancel_run`).
        pub fn uniffi_active_run_ids(&self) -> Vec<String> {
            self.active_run_ids()
        }

        /// Cancel a running skill.
        pub fn uniffi_cancel_run(&self, run_id: String, reason: String) -> bool {
            self.cancel_run(&run_id, &reason)
        }

        /// Validate a ToolPlan JSON against registered manifests.
        pub fn uniffi_validate_tool_plan(&self, plan_json: String) -> FfiValidationResult {
            self.validate_tool_plan(&plan_json)
        }

        /// Register a signed tool manifest.
        pub fn uniffi_register_tool_manifest(&self, manifest_json: String) -> Result<(), FfiError> {
            self.register_tool_manifest(&manifest_json)
        }

        /// Set the model cache directory for downloads.
        pub fn uniffi_set_model_cache_dir(&self, path: String) {
            self.set_model_cache_dir(&path)
        }

        /// Download a model pack with progress callbacks.
        pub fn uniffi_download_model(
            &self,
            manifest_json: String,
            listener: Arc<dyn DownloadProgressListener>,
        ) -> Result<String, FfiError> {
            let sink = UniProgressSink(listener);
            self.download_model(&manifest_json, &|d, t| sink.call(d, t))
        }

        /// Search stock-image providers (requires API keys in env).
        pub fn uniffi_search_images(
            &self,
            query: String,
            limit: u32,
        ) -> Result<Vec<FfiImageResult>, FfiError> {
            self.search_images(&query, limit)
        }

        /// Flush pending conversation-memory turns.
        pub fn uniffi_flush_memory(&self) -> Result<(), FfiError> {
            self.flush_memory()
        }
    }

    #[cfg(feature = "llamacpp")]
    #[uniffi::export]
    impl KChatAiRuntime {
        /// Attach an in-process llama.cpp backend from a GGUF file.
        pub fn uniffi_attach_llamacpp_backend(&self, model_path: String) -> bool {
            self.attach_llamacpp_backend(&model_path)
        }
    }

    #[cfg(feature = "gguf-encoder")]
    #[uniffi::export]
    impl KChatAiRuntime {
        /// Attach the shared in-process GGUF encoder (safety + embeddings +
        /// reranking).
        pub fn uniffi_attach_gguf_encoder(
            &self,
            model_path: String,
            heads_path: String,
            intra_threads: u32,
        ) -> bool {
            self.attach_gguf_encoder(&model_path, &heads_path, intra_threads)
        }
    }
}

// ============================================================================
// N-API bindings (desktop: macOS/Windows)
// ============================================================================

#[cfg(feature = "desktop")]
mod napi_bindings {
    use napi::threadsafe_function::{
        ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode,
    };
    use napi_derive::napi;

    use super::*;
    use std::sync::Arc;

    type EventFn = ThreadsafeFunction<FfiStreamEvent, ErrorStrategy::CalleeHandled>;
    type DoneFn = ThreadsafeFunction<FfiSkillOutcome, ErrorStrategy::CalleeHandled>;
    type ProgressFn = ThreadsafeFunction<(i64, i64), ErrorStrategy::CalleeHandled>;

    /// Adapter: TSFN → internal event sink. Blocking call mode gives the
    /// JS consumer natural backpressure; safe because the emitter runs on a
    /// dedicated OS thread, never the libuv thread.
    struct NapiEventSink(EventFn);
    impl StreamEventSink for NapiEventSink {
        fn emit(&self, event: FfiStreamEvent) {
            self.0.call(Ok(event), ThreadsafeFunctionCallMode::Blocking);
        }
    }

    struct NapiDoneSink(DoneFn);
    impl NapiDoneSink {
        fn send(&self, outcome: Result<FfiSkillOutcome, FfiError>) {
            self.0.call(
                outcome.map_err(|e| napi::Error::from_reason(e.to_string())),
                ThreadsafeFunctionCallMode::Blocking,
            );
        }
    }

    fn napi_err(e: FfiError) -> napi::Error {
        napi::Error::from_reason(e.to_string())
    }

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

        /// Whether a generation backend is attached right now.
        #[napi]
        pub fn generation_ready(&self) -> bool {
            self.inner.generation_ready()
        }

        #[napi]
        pub fn platform(&self) -> String {
            self.inner.platform().to_string()
        }

        #[napi]
        pub fn probe_capabilities(&self) -> FfiDeviceCapabilities {
            self.inner.probe_capabilities()
        }

        #[napi]
        pub fn backend_attached(&self) -> bool {
            self.inner.backend_attached()
        }

        #[napi]
        pub fn encoder_available(&self) -> bool {
            self.inner.encoder_available()
        }

        #[napi]
        pub fn attach_llamacpp_backend(&self, model_path: String) -> bool {
            #[cfg(feature = "llamacpp")]
            {
                self.inner.attach_llamacpp_backend(&model_path)
            }
            #[cfg(not(feature = "llamacpp"))]
            {
                let _ = model_path;
                false
            }
        }

        #[napi]
        pub fn attach_gguf_encoder(
            &self,
            model_path: String,
            heads_path: String,
            intra_threads: u32,
        ) -> bool {
            #[cfg(feature = "gguf-encoder")]
            {
                self.inner
                    .attach_gguf_encoder(&model_path, &heads_path, intra_threads)
            }
            #[cfg(not(feature = "gguf-encoder"))]
            {
                let _ = (model_path, heads_path, intra_threads);
                false
            }
        }

        #[napi]
        pub fn open_context_store(
            &self,
            path: String,
            db_password: String,
            master_key_hex: String,
        ) -> napi::Result<()> {
            self.inner
                .open_context_store(&path, &db_password, &master_key_hex)
                .map_err(napi_err)
        }

        #[napi]
        pub fn insert_evidence(
            &self,
            scope_id: String,
            content: String,
            source_ref: Option<String>,
            importance: u8,
            language_tag: Option<String>,
        ) -> napi::Result<String> {
            self.inner
                .insert_evidence(&scope_id, &content, source_ref, importance, language_tag)
                .map_err(napi_err)
        }

        #[napi]
        pub fn forget_scope(&self, scope_id: String) -> napi::Result<()> {
            self.inner.forget_scope(&scope_id).map_err(napi_err)
        }

        #[napi]
        pub fn set_memory_scope(&self, scope_id: String) -> napi::Result<()> {
            self.inner.set_memory_scope(&scope_id).map_err(napi_err)
        }

        #[napi]
        pub fn retrieve(
            &self,
            query: String,
            scope_ids: Vec<String>,
            user_id: String,
            roles: Vec<String>,
            limit: u32,
        ) -> napi::Result<Vec<FfiRetrievalResult>> {
            self.inner
                .retrieve(&query, scope_ids, &user_id, roles, limit)
                .map_err(napi_err)
        }

        #[napi]
        pub fn list_skills(&self) -> Vec<FfiSkillInfo> {
            self.inner.list_skills()
        }

        #[napi]
        pub fn route_request(
            &self,
            text: String,
            top_k: u32,
        ) -> napi::Result<Vec<FfiRouteSuggestion>> {
            self.inner.route_request(&text, top_k).map_err(napi_err)
        }

        /// Run a skill on a background OS thread. `on_event` receives stream
        /// events; `on_done` receives exactly one terminal outcome (or throws
        /// the error). Returns a run id for `cancel_run`.
        #[napi]
        pub fn run_skill_async(
            &self,
            skill_id: String,
            req: FfiSkillRequest,
            on_event: EventFn,
            on_done: DoneFn,
        ) -> napi::Result<String> {
            let sink = Arc::new(NapiEventSink(on_event));
            // Wrap: drain events + forward the outcome to on_done.
            let orc = self.inner.require_orchestrator().map_err(napi_err)?;
            let skill_req = self.inner.build_skill_request(req).map_err(napi_err)?;
            let run = orc
                .run_skill(&skill_id, skill_req)
                .map_err(|e| napi::Error::from_reason(e.to_string()))?;
            let run_id = uuid::Uuid::new_v4().to_string();
            self.inner
                .runs
                .lock()
                .insert(run_id.clone(), run.handle.clone());
            let runs = self.inner.runs.clone();
            let worker_run_id = run_id.clone();
            std::thread::spawn(move || {
                let mut run = run;
                KChatAiRuntime::drain_events(&mut run.events, sink.as_ref());
                match run.outcome.blocking_recv() {
                    Ok(outcome) => NapiDoneSink(on_done).send(
                        outcome
                            .map(KChatAiRuntime::map_outcome)
                            .map_err(|e| FfiError::Message(e.to_string())),
                    ),
                    Err(_) => NapiDoneSink(on_done).send(Err(FfiError::Message(
                        "skill outcome channel closed".into(),
                    ))),
                }
                runs.lock().remove(&worker_run_id);
            });
            Ok(run_id)
        }

        /// Run ids of in-flight skills (for `cancel_run`).
        #[napi]
        pub fn active_run_ids(&self) -> Vec<String> {
            self.inner.active_run_ids()
        }

        #[napi]
        pub fn cancel_run(&self, run_id: String, reason: String) -> bool {
            self.inner.cancel_run(&run_id, &reason)
        }

        #[napi]
        pub fn validate_tool_plan(&self, plan_json: String) -> FfiValidationResult {
            self.inner.validate_tool_plan(&plan_json)
        }

        #[napi]
        pub fn register_tool_manifest(&self, manifest_json: String) -> napi::Result<()> {
            self.inner
                .register_tool_manifest(&manifest_json)
                .map_err(napi_err)
        }

        #[napi]
        pub fn set_model_cache_dir(&self, path: String) {
            self.inner.set_model_cache_dir(&path)
        }

        #[napi]
        pub fn download_model(
            &self,
            manifest_json: String,
            on_progress: ProgressFn,
        ) -> napi::Result<String> {
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.inner
                    .download_model(&manifest_json, &|d, t| {
                        on_progress.call(
                            Ok((d as i64, t as i64)),
                            ThreadsafeFunctionCallMode::Blocking,
                        );
                    })
                    .map_err(napi_err)
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = (manifest_json, on_progress);
                Err(napi::Error::from_reason("downloads unsupported on wasm"))
            }
        }

        #[napi]
        pub fn search_images(
            &self,
            query: String,
            limit: u32,
        ) -> napi::Result<Vec<FfiImageResult>> {
            self.inner.search_images(&query, limit).map_err(napi_err)
        }

        #[napi]
        pub fn flush_memory(&self) -> napi::Result<()> {
            self.inner.flush_memory().map_err(napi_err)
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

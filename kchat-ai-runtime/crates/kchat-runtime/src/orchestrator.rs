//! Skill orchestrator — the single entry point that composes the four planes:
//!
//! ```text
//! run_skill(id, request)
//!   → safety pre-check (input + context)
//!   → scoped retrieval (evidence context, if scopes provided)
//!   → prompt template (SkillDef::build_prompt → ChatML)
//!   → LoRA resolve + swap (task × language)
//!   → grammar-constrained streaming generation
//!   → mid-stream safety monitor (cancel on Block)
//!   → post-check on final output
//!   → memory write-back (rolling log into context store)
//! ```
//!
//! Everything is local and in-process: no subprocesses, no network calls.

use kchat_action::toolplan::ToolPlanValidator;
use kchat_context::embeddings::EmbeddingManager;
use kchat_context::reranker::Reranker;
use kchat_context::retrieval::{RetrievalTier, Retriever};
use kchat_context::scope::ScopeFilter;
use kchat_context::store::ContextStore;
use kchat_core::tier::DeviceTier;
use kchat_generation::backend::{BackendAdapter, BackendError, GenerationConfig, GenerationResult};
use kchat_generation::grammar::Grammar;
use kchat_generation::lora::{BackendLoraHook, LoraManager};
use kchat_generation::skills::{SkillDef, SkillPromptInput, SkillRegistry, SkillTier};
use kchat_generation::stream::{spawn_stream, StreamEvent, StreamHandle, StreamingGeneration};
use kchat_safety::classify::{ClassifyRequest, SafetyClassifier};
use kchat_safety::verdict::{Action, Verdict};
use std::sync::Arc;
use tokio::sync::{mpsc::UnboundedReceiver, oneshot};

/// How often the mid-stream safety monitor re-classifies the accumulated
/// output (in emitted tokens). Deterministic classification is sub-ms, so
/// this trades a little overhead for fast time-to-cancel.
const SAFETY_CHECK_EVERY_TOKENS: usize = 24;

/// Maximum retrieved passages injected into the prompt context.
const RETRIEVAL_TOP_K: usize = 4;

/// Maximum characters of retrieved context injected (bounds prompt size).
const RETRIEVAL_CONTEXT_BUDGET: usize = 4000;

/// Hard cap on best-of-n candidates — keeps worst-case latency bounded.
const MAX_BEST_OF_N: usize = 4;

/// Options for a single skill invocation.
#[derive(Debug, Clone, Default)]
pub struct SkillRequest {
    /// User instruction, selection text, or topic (per skill's input mode).
    pub input: String,
    /// Document/context text the skill operates on.
    pub context: String,
    /// Optional keywords (for skills that support them).
    pub keywords: String,
    /// Sub-variant context when the user picked a sub-variant.
    pub variant_context: String,
    /// Response-language hint for LoRA slot resolution (e.g. "en", "vi-en").
    pub language: Option<String>,
    /// Scope filter enabling retrieval-augmented context. When `None`,
    /// no retrieval runs (pure document skills don't need it).
    pub scope_filter: Option<ScopeFilter>,
    /// Best-of-n candidate count — High tier only, scored by reranker.
    /// 0 or 1 disables. Values >1 on lower tiers are clamped to 1.
    pub best_of_n: usize,
}

/// A running skill invocation.
pub struct SkillRun {
    /// Skill that was executed.
    pub skill_id: String,
    /// Cancel via [`StreamHandle::cancel`].
    pub handle: Arc<StreamHandle>,
    /// Real-time events (tokens, complete, cancelled, error).
    pub events: UnboundedReceiver<StreamEvent>,
    /// Final outcome — resolves after post-check + memory write-back.
    pub outcome: oneshot::Receiver<Result<SkillOutcome, SkillError>>,
}

/// The finished result of a skill run.
#[derive(Debug)]
pub struct SkillOutcome {
    /// Raw generation result (text, tokens, TTFT, tok/s).
    pub generation: GenerationResult,
    /// Post-generation safety verdict on the model's own output.
    pub output_verdict: Verdict,
    /// Whether the output was blocked by the post-check.
    pub output_blocked: bool,
    /// Number of evidence passages injected as retrieval context.
    pub retrieved_passages: usize,
    /// LoRA adapter applied, if any.
    pub lora_adapter: Option<String>,
}

/// Parameters for a best-of-n generation job.
struct BestOfNJob {
    backend: Arc<dyn BackendAdapter>,
    skill: SkillDef,
    prompt: String,
    config: GenerationConfig,
    n: usize,
    retrieved: usize,
    lora_applied: Option<String>,
    user_input: String,
}

/// Orchestrator errors.
#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("unknown skill: {0}")]
    UnknownSkill(String),

    #[error("skill requires {required:?} tier, device is {actual:?}")]
    TierTooLow {
        required: SkillTier,
        actual: DeviceTier,
    },

    #[error("input blocked by safety policy")]
    InputBlocked(Box<Verdict>),

    #[error("no generation backend attached")]
    NoBackend,

    #[error("retrieval failed: {0}")]
    Retrieval(String),

    #[error("backend error: {0}")]
    Backend(#[from] BackendError),
}

/// Builder for [`Orchestrator`].
pub struct OrchestratorBuilder {
    backend: Option<Arc<dyn BackendAdapter>>,
    store: Option<Arc<ContextStore>>,
    embeddings: Option<Arc<EmbeddingManager>>,
    reranker: Option<Arc<dyn Reranker>>,
    memory_scope: Option<kchat_context::scope::ScopeId>,
    safety: Option<Arc<SafetyClassifier>>,
    tier: DeviceTier,
    language: String,
}

impl OrchestratorBuilder {
    pub fn new() -> Self {
        Self {
            backend: None,
            store: None,
            embeddings: None,
            reranker: None,
            memory_scope: None,
            safety: None,
            tier: DeviceTier::Low,
            language: "en".into(),
        }
    }

    pub fn backend(mut self, backend: Arc<dyn BackendAdapter>) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn context_store(mut self, store: Arc<ContextStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn embeddings(mut self, embeddings: Arc<EmbeddingManager>) -> Self {
        self.embeddings = Some(embeddings);
        self
    }

    pub fn reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Scope used for conversation-memory write-back evidence.
    pub fn memory_scope(mut self, scope: kchat_context::scope::ScopeId) -> Self {
        self.memory_scope = Some(scope);
        self
    }

    /// Share a safety classifier with the caller (e.g. the FFI facade) so
    /// encoder/SLM attachments apply to both direct classification and the
    /// in-pipeline pre/mid/post checks.
    pub fn safety(mut self, safety: Arc<SafetyClassifier>) -> Self {
        self.safety = Some(safety);
        self
    }

    pub fn tier(mut self, tier: DeviceTier) -> Self {
        self.tier = tier;
        self
    }

    /// Default response language used for LoRA slot resolution.
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    pub fn build(self) -> Orchestrator {
        let lora = LoraManager::new();
        if let Some(backend) = &self.backend {
            lora.set_backend_hook(Box::new(BackendLoraHook::new(backend.clone())));
        }
        let mut memory = crate::memory::ConversationMemory::new();
        if let (Some(store), Some(scope)) = (&self.store, self.memory_scope) {
            memory = memory.with_store(store.clone(), self.embeddings.clone(), scope);
        }
        let skills = Arc::new(SkillRegistry::new());
        Orchestrator {
            backend: self.backend,
            safety: self
                .safety
                .unwrap_or_else(|| Arc::new(SafetyClassifier::new())),
            lora,
            store: self.store,
            embeddings: self.embeddings.clone(),
            reranker: self.reranker,
            tool_validator: ToolPlanValidator::new(),
            memory: Arc::new(memory),
            router: crate::router::SkillRouter::new(skills.clone(), self.embeddings),
            skills,
            tier: self.tier,
            language: self.language,
        }
    }
}

impl Default for OrchestratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The runtime orchestrator — one instance per device/session.
pub struct Orchestrator {
    backend: Option<Arc<dyn BackendAdapter>>,
    skills: Arc<SkillRegistry>,
    safety: Arc<SafetyClassifier>,
    lora: LoraManager,
    store: Option<Arc<ContextStore>>,
    embeddings: Option<Arc<EmbeddingManager>>,
    reranker: Option<Arc<dyn Reranker>>,
    tool_validator: ToolPlanValidator,
    memory: Arc<crate::memory::ConversationMemory>,
    router: crate::router::SkillRouter,
    tier: DeviceTier,
    language: String,
}

impl Orchestrator {
    pub fn builder() -> OrchestratorBuilder {
        OrchestratorBuilder::new()
    }

    /// The skill registry (for listing available skills over FFI).
    pub fn skills(&self) -> &SkillRegistry {
        &self.skills
    }

    /// The intent router — maps natural-language requests to skills.
    pub fn router(&self) -> &crate::router::SkillRouter {
        &self.router
    }

    /// Access the safety classifier (e.g. to attach an encoder/SLM or load
    /// policy packs).
    pub fn safety(&self) -> &Arc<SafetyClassifier> {
        &self.safety
    }

    /// Access the LoRA manager (to register adapters).
    pub fn lora(&self) -> &LoraManager {
        &self.lora
    }

    /// Access the tool-plan validator (to register tool manifests).
    pub fn tool_validator(&self) -> &ToolPlanValidator {
        &self.tool_validator
    }

    /// Access the conversation memory (shared across runs so the flush
    /// threshold accumulates over turns).
    pub fn memory(&self) -> &Arc<crate::memory::ConversationMemory> {
        &self.memory
    }

    /// Device tier this orchestrator was built for.
    pub fn tier(&self) -> DeviceTier {
        self.tier
    }

    /// Run a skill end-to-end. Returns a [`SkillRun`] with a live event
    /// stream and a completion channel for the enriched outcome.
    pub fn run_skill(&self, skill_id: &str, req: SkillRequest) -> Result<SkillRun, SkillError> {
        let backend = self.backend.clone().ok_or(SkillError::NoBackend)?;
        let skill = self
            .skills
            .get(skill_id)
            .ok_or_else(|| SkillError::UnknownSkill(skill_id.to_string()))?
            .clone();

        if !tier_satisfies(self.tier, skill.min_tier) {
            return Err(SkillError::TierTooLow {
                required: skill.min_tier,
                actual: self.tier,
            });
        }

        // 1. Safety pre-check on user-controlled text. Encoder/SLM
        // availability is reported truthfully so escalation runs when a
        // real classifier is attached.
        let precheck = self
            .safety
            .classify(&self.classify_request(format!("{}\n{}", req.input, req.context)));
        if precheck.verdict.action == Action::Block {
            return Err(SkillError::InputBlocked(Box::new(precheck.verdict)));
        }

        // 2. Scoped retrieval — inject top passages into context.
        let (context, retrieved) = match (&req.scope_filter, &self.store) {
            (Some(filter), Some(store)) => {
                let passages = self.retrieve_context(&req.input, filter, store)?;
                (merge_context(&req.context, &passages), passages.len())
            }
            _ => (req.context.clone(), 0),
        };

        // 3. Prompt template → ChatML.
        let device_skill_tier = match self.tier {
            DeviceTier::Low => SkillTier::Low,
            DeviceTier::Medium => SkillTier::Medium,
            DeviceTier::High => SkillTier::High,
        };
        let prompt = skill
            .build_prompt(SkillPromptInput {
                input: &req.input,
                context: &context,
                keywords: &req.keywords,
                variant_context: &req.variant_context,
                tier: Some(device_skill_tier),
            })
            .to_chatml(skill.response_prefix.as_deref());

        // 4. LoRA resolve + swap — task family × language slot.
        let lora_applied = self.resolve_lora(&skill, req.language.as_deref());

        // 5. Grammar-constrained generation config, capped by skill tier.
        let grammar = Grammar::for_skill(&skill);
        let (min_out, max_out) = skill.min_tier.output_cap();
        let config = GenerationConfig {
            max_tokens: skill.max_tokens.clamp(min_out, max_out.max(min_out)),
            temperature: skill.temperature,
            grammar,
            stop: skill.stop.clone(),
            ..GenerationConfig::default()
        };

        // 6. Best-of-n on High tier (reranker-scored), else single stream.
        // Keep `req.best_of_n > 1` as a direct operand of the branch — do not
        // route it through a match-computed local (miscompiled at opt-level=1
        // on aarch64 LLVM: the `n > 1` guard was folded away and every
        // High-tier request entered best-of-n with n=0).
        if self.tier == DeviceTier::High && req.best_of_n > 1 {
            return Ok(self.run_best_of_n(BestOfNJob {
                backend,
                skill,
                prompt,
                config,
                n: req.best_of_n.min(MAX_BEST_OF_N),
                retrieved,
                lora_applied,
                user_input: req.input,
            }));
        }

        let generation = spawn_stream(backend, prompt, config);
        Ok(self.wrap_run(skill, generation, retrieved, lora_applied, req.input))
    }

    /// Attach the safety monitor + post-check + memory write-back around a
    /// live streaming generation.
    fn wrap_run(
        &self,
        skill: SkillDef,
        generation: StreamingGeneration,
        retrieved: usize,
        lora_applied: Option<String>,
        user_input: String,
    ) -> SkillRun {
        let StreamingGeneration {
            handle,
            events: worker_events,
            result,
        } = generation;

        // Caller-facing event channel — written ONLY by the safety monitor.
        // Tokens are withheld until the classify covering them has passed, so
        // blocked text (including the post-check tail) never reaches the
        // caller mid-stream.
        let (caller_tx, caller_events) = tokio::sync::mpsc::unbounded_channel();
        let mut mon_events = worker_events;

        let (outcome_tx, outcome_rx) = oneshot::channel();
        let safety = self.safety.clone();
        let memory = self.memory.clone();
        let monitor_handle = handle.clone();

        std::thread::spawn(move || {
            // --- Mid-stream monitor: accumulate output, forward tokens only
            // after the classify covering them passes. `pending` holds the
            // unclassified tail; it flushes after each checkpoint and — on a
            // terminal event — only after a final full classify passes. ---
            let mut emitted = String::new();
            let mut pending: Vec<StreamEvent> = Vec::new();
            let mut since_check = 0usize;
            'monitor: while let Some(ev) = mon_events.blocking_recv() {
                match ev {
                    StreamEvent::Token { text } => {
                        emitted.push_str(&text);
                        pending.push(StreamEvent::Token { text });
                        since_check += 1;
                        if since_check >= SAFETY_CHECK_EVERY_TOKENS {
                            since_check = 0;
                            let v = safety.classify(&monitor_req(&safety, emitted.clone()));
                            if v.verdict.action == Action::Block {
                                monitor_handle.cancel("output blocked by safety policy");
                                let _ = caller_tx.send(StreamEvent::Cancelled {
                                    reason: "output blocked by safety policy".into(),
                                });
                                break 'monitor;
                            }
                            for e in pending.drain(..) {
                                let _ = caller_tx.send(e);
                            }
                        }
                    }
                    StreamEvent::Cancelled { .. } | StreamEvent::Error { .. } => {
                        // Pass through — cancellation/error carry no output.
                        let _ = caller_tx.send(ev);
                        break 'monitor;
                    }
                    StreamEvent::Complete { .. } => {
                        // Final check covers the withheld tail before the
                        // caller sees it or the completion.
                        let v = safety.classify(&monitor_req(&safety, emitted.clone()));
                        if v.verdict.action == Action::Block {
                            let _ = caller_tx.send(StreamEvent::Cancelled {
                                reason: "output blocked by safety policy".into(),
                            });
                        } else {
                            for e in pending.drain(..) {
                                let _ = caller_tx.send(e);
                            }
                            let _ = caller_tx.send(ev);
                        }
                        break 'monitor;
                    }
                }
            }
            // Drop caller_tx — receiver observes end-of-stream.

            // --- Final result → post-check → memory write-back ---
            let outcome = match result.blocking_recv() {
                Ok(Ok(gen)) => {
                    let post = safety.classify(&monitor_req(&safety, gen.text.clone()));
                    let output_blocked = post.verdict.action == Action::Block;

                    // Rolling memory: record the turn (persisted verbatim at
                    // threshold — no model needed on the critical path).
                    if !output_blocked {
                        memory.record_turn(&user_input, &gen.text);
                    }

                    // Blocked text must never reach the caller — scrub it,
                    // keeping token/latency metadata for telemetry.
                    let generation = if output_blocked {
                        kchat_generation::backend::GenerationResult {
                            text: String::new(),
                            ..gen
                        }
                    } else {
                        gen
                    };

                    Ok(SkillOutcome {
                        generation,
                        output_verdict: post.verdict,
                        output_blocked,
                        retrieved_passages: retrieved,
                        lora_adapter: lora_applied,
                    })
                }
                Ok(Err(e)) => Err(SkillError::Backend(e)),
                Err(_) => Err(SkillError::Backend(BackendError::GenerationFailed(
                    "generation worker dropped".into(),
                ))),
            };
            let _ = outcome_tx.send(outcome);
        });

        SkillRun {
            skill_id: skill.id,
            handle,
            events: caller_events,
            outcome: outcome_rx,
        }
    }

    /// Best-of-n: generate n candidates (sequentially — the backend's KV
    /// session is single-tenant), score with the reranker, emit the winner
    /// through the normal stream surface.
    fn run_best_of_n(&self, job: BestOfNJob) -> SkillRun {
        let BestOfNJob {
            backend,
            skill,
            prompt,
            config,
            n,
            retrieved,
            lora_applied,
            user_input,
        } = job;
        let reranker = self.reranker.clone();
        let safety = self.safety.clone();
        let memory = self.memory.clone();

        let handle = Arc::new(StreamHandle::new());
        let (event_tx, caller_events) = tokio::sync::mpsc::unbounded_channel();
        handle.forward_to(event_tx);
        let (outcome_tx, outcome_rx) = oneshot::channel();
        let worker_handle = handle.clone();

        std::thread::spawn(move || {
            let mut candidates: Vec<GenerationResult> = Vec::with_capacity(n);
            for i in 0..n {
                if worker_handle.is_cancelled() {
                    break;
                }
                let mut cfg = config.clone();
                cfg.seed = i as u64 + 1;
                // Candidate tokens must not reach the caller — a dedicated
                // handle shares the run's cancellation flag but has no
                // subscribers, so only the winner is emitted below.
                let cand_handle =
                    StreamHandle::with_cancellation(worker_handle.cancellation_flag());
                match backend.generate_stream(&prompt, &cfg, &cand_handle) {
                    Ok(res) => candidates.push(res),
                    Err(e) if i == 0 && !worker_handle.is_cancelled() => {
                        worker_handle.error(e.to_string());
                        let _ = outcome_tx.send(Err(SkillError::Backend(e)));
                        return;
                    }
                    Err(_) => {} // later candidates may fail; keep going
                }
            }

            if worker_handle.is_cancelled() {
                // The shared flag may have been set by a candidate handle —
                // ensure the caller observes a terminal event. Exactly-once
                // semantics make this a no-op if one was already emitted.
                worker_handle.cancel("cancelled");
                let _ = outcome_tx.send(Err(SkillError::Backend(BackendError::GenerationFailed(
                    "cancelled".into(),
                ))));
                return;
            }

            let Some(winner) = select_winner(candidates, &user_input, reranker.as_deref()) else {
                worker_handle.error("no candidates generated");
                let _ = outcome_tx.send(Err(SkillError::Backend(BackendError::GenerationFailed(
                    "no candidates generated".into(),
                ))));
                return;
            };

            // Post-check BEFORE emitting the winner — a blocked candidate's
            // text must never stream to the caller.
            let post = safety.classify(&monitor_req(&safety, winner.text.clone()));
            let output_blocked = post.verdict.action == Action::Block;
            if output_blocked {
                worker_handle.cancel("output blocked by safety policy");
            } else {
                worker_handle.push_token(winner.text.clone());
                worker_handle.complete(winner.completion_tokens, winner.total_ms);
                memory.record_turn(&user_input, &winner.text);
            }

            // Scrub blocked text before handing the outcome to the caller.
            let generation = if output_blocked {
                kchat_generation::backend::GenerationResult {
                    text: String::new(),
                    ..winner
                }
            } else {
                winner
            };

            let _ = outcome_tx.send(Ok(SkillOutcome {
                generation,
                output_verdict: post.verdict,
                output_blocked,
                retrieved_passages: retrieved,
                lora_adapter: lora_applied,
            }));
        });

        SkillRun {
            skill_id: skill.id,
            handle,
            events: caller_events,
            outcome: outcome_rx,
        }
    }

    /// Build a safety request with truthful escalation availability —
    /// the attached encoder/SLM must be visible or the pipeline silently
    /// runs deterministic-only.
    fn classify_request(&self, text: String) -> ClassifyRequest {
        monitor_req(&self.safety, text)
    }

    /// Retrieve top-k evidence passages for prompt injection.
    fn retrieve_context(
        &self,
        query: &str,
        filter: &ScopeFilter,
        store: &Arc<ContextStore>,
    ) -> Result<Vec<String>, SkillError> {
        let tier = match self.tier {
            DeviceTier::Low => RetrievalTier::Low,
            DeviceTier::Medium => RetrievalTier::Medium,
            DeviceTier::High => RetrievalTier::High,
        };
        let mut retriever = Retriever::new(store, tier);
        if let Some(embs) = &self.embeddings {
            retriever = retriever.with_embeddings(embs);
        }
        if let Some(r) = &self.reranker {
            retriever = retriever.with_reranker(r.as_ref());
        }
        let results = retriever
            .retrieve(query, filter, RETRIEVAL_TOP_K)
            .map_err(|e| SkillError::Retrieval(e.to_string()))?;

        let mut passages = Vec::new();
        let mut budget = RETRIEVAL_CONTEXT_BUDGET;
        for res in results {
            if budget == 0 {
                break;
            }
            if let Ok(Some(ev)) = store.get_evidence(res.evidence_id) {
                let text: String = ev.fts_content.chars().take(budget).collect();
                budget = budget.saturating_sub(text.len());
                passages.push(text);
            }
        }
        Ok(passages)
    }

    /// Resolve the skill's LoRA adapter (task × language) and hot-swap it.
    /// Detaches any active adapter when the skill has no LoRA task or no
    /// matching adapter is registered — a stale adapter must not leak into
    /// an unrelated skill's generation.
    fn resolve_lora(&self, skill: &SkillDef, language: Option<&str>) -> Option<String> {
        let adapter = if skill.lora_task.is_empty() {
            None
        } else {
            let lang = language.unwrap_or(&self.language);
            self.lora.find(&skill.lora_task, lang)
        };
        let Some(adapter) = adapter else {
            if self.lora.has_active_adapter() {
                if let Err(e) = self.lora.detach() {
                    tracing::warn!("LoRA detach failed: {}", e);
                }
            }
            return None;
        };
        match self.lora.swap(&adapter.adapter_id) {
            Ok(_) => Some(adapter.adapter_id.clone()),
            Err(e) => {
                tracing::warn!("LoRA swap to {} failed: {}", adapter.adapter_id, e);
                None
            }
        }
    }
}

/// Cap on text handed to the safety classifier — bounds classification
/// latency on very large inputs (~1µs/byte deterministic). Head+tail
/// sampling keeps both ends covered: injected instructions and the harmful
/// payload almost always live near the start or end.
const SAFETY_INPUT_CAP_CHARS: usize = 32_000;

fn bounded_safety_input(text: String) -> String {
    let total = text.chars().count();
    if total <= SAFETY_INPUT_CAP_CHARS {
        return text;
    }
    let head = SAFETY_INPUT_CAP_CHARS * 3 / 4;
    let tail = SAFETY_INPUT_CAP_CHARS - head;
    let mut s: String = text.chars().take(head).collect();
    s.push('\n');
    s.extend(text.chars().skip(total - tail));
    s
}

/// Build a classify request that truthfully reports escalation
/// availability (usable from worker threads that only hold the classifier)
/// and bounds the input length.
fn monitor_req(safety: &Arc<SafetyClassifier>, text: String) -> ClassifyRequest {
    ClassifyRequest {
        encoder_available: safety.has_encoder(),
        slm_available: safety.has_slm(),
        ..ClassifyRequest::from_text(bounded_safety_input(text))
    }
}

/// Pick the best candidate — cross-encoder reranker scored against the
/// user's actual request when available, otherwise the longest output
/// (a weak but deterministic signal; no quality claim is made).
fn select_winner(
    mut candidates: Vec<GenerationResult>,
    user_input: &str,
    reranker: Option<&dyn Reranker>,
) -> Option<GenerationResult> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return candidates.pop();
    }
    let idx = if let Some(r) = reranker {
        let docs: Vec<String> = candidates.iter().map(|c| c.text.clone()).collect();
        r.rerank(user_input, &docs, 1)
            .ok()
            .and_then(|ranked| ranked.first().map(|(i, _)| *i))
            .unwrap_or(0)
    } else {
        candidates
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| c.completion_tokens)
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    Some(candidates.swap_remove(idx))
}

/// Merge retrieved passages into the prompt context (bounded).
fn merge_context(base: &str, passages: &[String]) -> String {
    if passages.is_empty() {
        return base.to_string();
    }
    let joined = passages.join("\n---\n");
    if base.is_empty() {
        format!("Relevant context:\n{joined}")
    } else {
        format!("{base}\n\nRelevant context:\n{joined}")
    }
}

/// Whether a device tier satisfies a skill's minimum tier.
fn tier_satisfies(device: DeviceTier, required: SkillTier) -> bool {
    let d = match device {
        DeviceTier::Low => 0,
        DeviceTier::Medium => 1,
        DeviceTier::High => 2,
    };
    let r = match required {
        SkillTier::Low => 0,
        SkillTier::Medium => 1,
        SkillTier::High => 2,
    };
    d >= r
}

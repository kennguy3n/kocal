//! kchat-runtime — the orchestration layer that composes the four planes
//! (safety, context, generation, action) into a single skill-execution API.
//!
//! - [`orchestrator::Orchestrator`] — `run_skill` end-to-end pipeline:
//!   safety pre-check → scoped retrieval → prompt template → LoRA swap →
//!   grammar-constrained streaming → mid-stream safety monitor →
//!   post-check → memory write-back.
//! - [`router::SkillRouter`] — intent routing: natural-language request →
//!   skill + LoRA family + grammar (embedding-based, keyword fallback).
//! - [`memory::ConversationMemory`] — rolling memory: turns are persisted
//!   verbatim into the encrypted store at threshold, with optional
//!   LLM-summarized write-back.
//!
//! Fully local and in-process: no subprocesses, no network calls.

pub mod memory;
pub mod orchestrator;
pub mod router;

pub use memory::ConversationMemory;
pub use orchestrator::{
    Orchestrator, OrchestratorBuilder, SkillError, SkillOutcome, SkillRequest, SkillRun,
};
pub use router::{RouteDecision, RouteSuggestion, SkillRouter};

//! kchat-generation: Grammar-constrained generative plane.
//!
//! The generative plane runs on medium and high tier devices only. It uses
//! llama.cpp as the runtime with Metal/CoreML (iOS/macOS), Vulkan (Android/
//! Windows), and CPU fallback.
//!
//! Key principles:
//! - Prompt templates are versioned and hashed for provenance
//! - Output is grammar-constrained (JSON schema, regex, or Lark grammar)
//! - The model emits a ToolPlan, not an executable request
//! - Token streaming with early termination on safety violation
//! - Idle unload after 30-60 seconds on mobile, configurable on desktop

pub mod backend;
pub mod backends;
pub mod budget;
pub mod grammar;
pub mod lifecycle;
pub mod lora;
pub mod pipeline;
pub mod prompt;
pub mod skills;
pub mod slides_templates;
pub mod stream;
pub mod swarm;

pub use backend::{BackendAdapter, BackendConfig, BackendType, GenerationConfig, GenerationResult};
pub use backends::{select_backend, MockBackend};
#[cfg(feature = "llamacpp")]
pub use backends::LlamaCppBackend;
#[cfg(feature = "mlx")]
pub use backends::MlxBackend;
pub use budget::{
    adaptive_max_output, budget_for_context, chunk_document, estimate_tokens_text,
    extract_outline_context, get_local_context, truncate_context, truncate_head,
    truncate_tail, DocChunk,
};
pub use grammar::{Grammar, GrammarType, GrammarValidator};
pub use lifecycle::{ModelLifecycle, ModelState};
pub use lora::{LoraAdapter, LoraManager, LoraError, SkillLoRAResolver};
pub use pipeline::{GenerationPipeline, PipelineProgress, PipelineResult};
pub use prompt::{PromptTemplate, PromptTemplateRegistry, TemplateId};
pub use skills::{
    SkillDef, SkillGrammarType, SkillGroup, SkillMode, SkillPromptInput, SkillPromptOutput,
    SkillRegistry, SkillScope, SkillSubVariant, SkillSurface, SkillTier,
};
pub use slides_templates::{
    SlotDef, SlotType, SlidesTemplate, SlidesTemplateFamily, SlidesTemplateRegistry,
    TEMPLATE_CATALOG, TEMPLATE_REGISTRY,
};
pub use stream::{StreamEvent, StreamHandle, StreamId};
pub use swarm::{Peer, Swarm, SwarmConfig, SwarmError, SwarmResult};

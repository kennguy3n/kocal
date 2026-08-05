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
pub mod grammar;
pub mod lifecycle;
pub mod prompt;
pub mod stream;

pub use backend::{BackendAdapter, BackendConfig, BackendType, GenerationConfig, GenerationResult};
pub use grammar::{Grammar, GrammarType, GrammarValidator};
pub use lifecycle::{ModelLifecycle, ModelState};
pub use prompt::{PromptTemplate, PromptTemplateRegistry, TemplateId};
pub use stream::{StreamEvent, StreamHandle, StreamId};

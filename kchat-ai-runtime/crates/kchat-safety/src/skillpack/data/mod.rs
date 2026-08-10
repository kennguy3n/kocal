//! Embedded skill-pack data — global baselines, community overlays, jurisdiction
//! overlays, and prompt templates.
//!
//! All YAML, JSON, and text files are embedded here as `include_str!` /
//! `include_bytes!` constants from the `files/` subdirectory, making the crate
//! fully self-contained without external file paths.
//!
//! ## Sub-modules
//!
//! * [`global`] — taxonomy, baseline, severity rubric, privacy contract, vision
//!   baselines, output schema, local signal schema, transliteration map.
//! * [`communities`] — 38 community overlay YAML files keyed by community ID.
//! * [`jurisdictions`] — 62 jurisdiction overlay YAML files keyed by jurisdiction code.
//! * [`prompts`] — runtime instruction text, compiled prompt format, and 73 compiled
//!   prompt examples (baseline, community, jurisdiction, archetype combos).

pub mod communities;
pub mod global;
pub mod jurisdictions;
pub mod loaders;
pub mod prompts;

#[cfg(test)]
#[path = "adversarial_tests.rs"]
mod adversarial_tests;

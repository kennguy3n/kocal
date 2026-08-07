//! kchat-safety: Deterministic safety plane for KChat.
//!
//! The safety plane implements a layered guardrail:
//!
//! 1. Normalize text locally after decryption (NFKC + case fold + defang)
//! 2. Apply signed deterministic policy, allowlists, blocklists, rate signals
//! 3. If confidence is insufficient, run the compact encoder
//! 4. On eligible medium/high devices, invoke the SLM for ambiguous cases
//! 5. Apply deterministic policy to the structured result
//! 6. Return allow, warn, block, redact, or require-consent with reason codes
//!
//! The generative model must not be called on every message. The normal hot
//! path is deterministic evaluation, followed by the encoder only when rules
//! or uncertainty require it. Safety must remain operational on low-tier
//! devices with no generative pack installed.

pub mod classify;
pub mod crypto;
pub mod detectors;
pub mod encoder;
pub mod media;
pub mod normalize;
pub mod policy;
pub mod verdict;

#[cfg(feature = "text-pipeline")]
pub mod policy_interpreter;

#[cfg(feature = "skill-pack")]
pub mod skillpack;

#[cfg(feature = "onnx-runtime-vision")]
pub mod vision;
#[cfg(feature = "onnx-runtime-vision")]
pub mod vision_bridge;

pub use classify::{ClassifyRequest, ClassifyResult, SafetyClassifier};
pub use encoder::{MockEncoder, MockSlmAdjudicator, SlmDecision};
pub use media::{MediaDescriptor, MediaDescriptorInput, extract_media_descriptors};
#[cfg(feature = "onnx-runtime")]
pub use encoder::OnnxEncoder;
pub use policy::{PolicyPack, PolicyPackManifest, PolicyRule, RiskCategory};
pub use verdict::{Action, Verdict, VerdictBuilder};

#[cfg(feature = "onnx-runtime-vision")]
pub use vision::{
    VisionEncoderAdapter, VisionEncoderAdapterBuilder, VisionEncoderError,
    VisionEncoderVerdict, VisionImageClassifier,
};
#[cfg(feature = "onnx-runtime-vision")]
pub use vision_bridge::{VisionBridge, classify_image as classify_image_with_vision};

/// Re-export core types for convenience.
pub use kchat_core::ids::PolicyPackId;

//! On-device vision encoder — MobileCLIP-S2 image tower.
//!
//! Ported from slm-guardrail's `encoder/vision/` module. Provides:
//!
//! - [`image_preprocess`]: Raw image bytes → CHW f32 tensor (256×256×3)
//! - [`mobileclip_session`]: ONNX session wrapper for MobileCLIP-S2
//! - [`vision_adapter`]: End-to-end image bytes → MediaDescriptor scores
//! - [`frame_aggregation`]: Temporal aggregation of per-frame verdicts for video
//!
//! # Feature gating
//!
//! Gated behind `onnx-runtime-vision` (a strict superset of `onnx-runtime`).
//! Hosts that only need text classification omit this feature and pay zero
//! binary-size cost for the `image` crate.

pub mod frame_aggregation;
pub mod image_preprocess;
pub mod mobileclip_session;
pub mod vision_adapter;

pub use frame_aggregation::{
    aggregate_frame_verdicts, aggregate_frame_verdicts_smoothed, FrameAggregationError,
    FrameVerdict, TemporalSmoothingConfig,
};
pub use image_preprocess::{preprocess_image, VisionImagePreprocessError};
pub use mobileclip_session::{MobileClipSession, MobileClipSessionError};
pub use vision_adapter::{
    VisionEncoderAdapter, VisionEncoderAdapterBuilder, VisionEncoderError, VisionEncoderVerdict,
    VisionImageClassifier,
};

/// MobileCLIP-S2 embedding dimensionality.
pub const MOBILECLIP_EMBED_DIM: usize = 512;

/// MobileCLIP-S2 input image size (square).
pub const MOBILECLIP_IMAGE_SIZE: usize = 256;

/// MobileCLIP-S2 per-channel pixel mean (all zeros — trained on [0,1] pixels).
pub const MOBILECLIP_PIXEL_MEAN: [f32; 3] = [0.0, 0.0, 0.0];

/// MobileCLIP-S2 per-channel pixel std (all ones — no normalization).
pub const MOBILECLIP_PIXEL_STD: [f32; 3] = [1.0, 1.0, 1.0];

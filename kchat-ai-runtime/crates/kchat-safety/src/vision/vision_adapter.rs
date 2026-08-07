//! End-to-end MobileCLIP-S2 vision adapter — image bytes → verdict.
//!
//! Ported from slm-guardrail's `encoder/vision/vision_adapter.rs`.
//! Adapted to kocal's architecture: produces `MediaDescriptor` scores
//! instead of prototype-bank classifications.
//!
//! The adapter runs: preprocess → ONNX forward → L2-normalise →
//! score mapping → MediaDescriptor. The host then feeds the
//! MediaDescriptor into the deterministic priority chain via
//! `ClassifyRequest::with_media()`.

use std::path::PathBuf;

use super::image_preprocess::{preprocess_image, VisionImagePreprocessError};
use super::mobileclip_session::{MobileClipSession, MobileClipSessionError};
use super::MOBILECLIP_EMBED_DIM;
use crate::media::MediaDescriptor;

/// Top-level error type for the vision pipeline.
#[derive(Debug)]
pub enum VisionEncoderError {
    Preprocess(VisionImagePreprocessError),
    Session(MobileClipSessionError),
    InvalidConfiguration { reason: String },
}

impl std::fmt::Display for VisionEncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Preprocess(e) => write!(f, "vision preprocess: {e}"),
            Self::Session(e) => write!(f, "vision session: {e}"),
            Self::InvalidConfiguration { reason } => write!(f, "vision config: {reason}"),
        }
    }
}

impl std::error::Error for VisionEncoderError {}

impl From<VisionImagePreprocessError> for VisionEncoderError {
    fn from(e: VisionImagePreprocessError) -> Self {
        Self::Preprocess(e)
    }
}

impl From<MobileClipSessionError> for VisionEncoderError {
    fn from(e: MobileClipSessionError) -> Self {
        Self::Session(e)
    }
}

/// Vision encoder verdict — a MediaDescriptor with safety scores.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionEncoderVerdict {
    pub descriptor: MediaDescriptor,
    /// The raw 512-dim embedding (for telemetry / downstream use).
    pub embedding: Vec<f32>,
}

impl VisionEncoderVerdict {
    pub fn new(descriptor: MediaDescriptor, embedding: Vec<f32>) -> Self {
        Self { descriptor, embedding }
    }
}

/// Object-safe trait the FFI consumes.
pub trait VisionImageClassifier: Send + Sync {
    /// Encode image bytes to a 512-dim L2-normalised embedding.
    fn encode_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, VisionEncoderError>;

    /// Encode + classify in one call → MediaDescriptor.
    fn classify_image(
        &self,
        image_bytes: &[u8],
    ) -> Result<VisionEncoderVerdict, VisionEncoderError>;
}

/// MobileCLIP-S2-backed vision adapter.
///
/// The `score_mapper` closure maps the 512-dim embedding to 10 safety scores.
/// In production, this is a trained linear head or prototype-bank cosine
/// similarity. For v1, a placeholder mapper returns all-None scores
/// (the host provides its own mapper via the builder).
///
/// Type alias for the score mapper closure.
type ScoreMapper = Box<dyn Fn(&[f32]) -> MediaDescriptor + Send + Sync>;

pub struct VisionEncoderAdapter {
    session: MobileClipSession,
    score_mapper: ScoreMapper,
}

impl VisionEncoderAdapter {
    pub fn builder() -> VisionEncoderAdapterBuilder {
        VisionEncoderAdapterBuilder::default()
    }

    /// Run preprocess → ONNX forward → L2-normalise and return the raw embedding.
    pub fn encode_image_to_vec(&self, image_bytes: &[u8]) -> Result<Vec<f32>, VisionEncoderError> {
        let preprocessed = preprocess_image(image_bytes)?;
        let embedding = self.session.embed_image(&preprocessed)?;
        if embedding.len() != MOBILECLIP_EMBED_DIM {
            return Err(VisionEncoderError::InvalidConfiguration {
                reason: format!(
                    "embedding dim {} != expected {MOBILECLIP_EMBED_DIM}",
                    embedding.len()
                ),
            });
        }
        Ok(embedding)
    }
}

impl VisionImageClassifier for VisionEncoderAdapter {
    fn encode_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, VisionEncoderError> {
        self.encode_image_to_vec(image_bytes)
    }

    fn classify_image(&self, image_bytes: &[u8]) -> Result<VisionEncoderVerdict, VisionEncoderError> {
        let embedding = self.encode_image_to_vec(image_bytes)?;
        let descriptor = (self.score_mapper)(&embedding);
        Ok(VisionEncoderVerdict::new(descriptor, embedding))
    }
}

impl std::fmt::Debug for VisionEncoderAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionEncoderAdapter")
            .field("session", &self.session)
            .finish_non_exhaustive()
    }
}

/// Builder for [`VisionEncoderAdapter`].
#[derive(Default)]
pub struct VisionEncoderAdapterBuilder {
    model_source: Option<ModelSource>,
    intra_threads: usize,
    score_mapper: Option<ScoreMapper>,
}

enum ModelSource {
    File(PathBuf),
    Bytes(Vec<u8>),
}

impl VisionEncoderAdapterBuilder {
    /// Load the MobileCLIP-S2 ONNX graph from a filesystem path.
    pub fn with_onnx_model_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_source = Some(ModelSource::File(path.into()));
        self
    }

    /// Load the MobileCLIP-S2 ONNX graph from in-memory bytes.
    pub fn with_onnx_model_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.model_source = Some(ModelSource::Bytes(bytes));
        self
    }

    /// Set the ORT intra-op thread count. `0` means ORT picks.
    pub fn with_intra_threads(mut self, n: usize) -> Self {
        self.intra_threads = n;
        self
    }

    /// Set the score mapper that converts a 512-dim embedding to MediaDescriptor scores.
    /// If not set, a placeholder mapper returns all-None scores.
    pub fn with_score_mapper(mut self, mapper: impl Fn(&[f32]) -> MediaDescriptor + Send + Sync + 'static) -> Self {
        self.score_mapper = Some(Box::new(mapper));
        self
    }

    /// Validate and build the adapter.
    pub fn build(self) -> Result<VisionEncoderAdapter, VisionEncoderError> {
        let model_source = self.model_source.ok_or(VisionEncoderError::InvalidConfiguration {
            reason: "missing mobileclip onnx model".into(),
        })?;

        let session = match model_source {
            ModelSource::File(path) => MobileClipSession::from_file(path, self.intra_threads),
            ModelSource::Bytes(bytes) => MobileClipSession::from_bytes(&bytes, self.intra_threads),
        }?;

        let score_mapper = self.score_mapper.unwrap_or_else(|| {
            Box::new(|_embedding: &[f32]| MediaDescriptor {
                kind: "image".into(),
                nsfw_score: None,
                violence_score: None,
                self_harm_score: None,
                hate_score: None,
                harassment_score: None,
                drugs_weapons_score: None,
                extremism_score: None,
                child_safety_score: None,
                deepfake_score: None,
                malware_score: None,
                face_count: None,
            })
        });

        Ok(VisionEncoderAdapter { session, score_mapper })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_rejects_missing_model() {
        let err = VisionEncoderAdapter::builder()
            .build()
            .expect_err("missing model must be rejected");
        assert!(matches!(err, VisionEncoderError::InvalidConfiguration { .. }));
    }

    #[test]
    #[ignore = "requires libonnxruntime.dylib installed"]
    fn build_propagates_session_load_failure() {
        let err = VisionEncoderAdapter::builder()
            .with_onnx_model_bytes(b"not an onnx model".to_vec())
            .build()
            .expect_err("garbage bytes must fail");
        assert!(matches!(err, VisionEncoderError::Session(MobileClipSessionError::LoadFailed { .. })));
    }
}

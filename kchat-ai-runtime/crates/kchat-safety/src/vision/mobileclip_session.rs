//! ONNX session wrapper for the MobileCLIP-S2 image tower.
//!
//! Ported from slm-guardrail's `encoder/vision/mobileclip_session.rs`.
//! Adapted to kocal's `ort` 2.0.0-rc.10 setup with `load-dynamic`.
//!
//! Input: `[1, 3, 256, 256]` f32 CHW tensor (from `image_preprocess`)
//! Output: `[1, 512]` f32 L2-normalised image embedding

use std::path::Path;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use parking_lot::Mutex;

use super::{MOBILECLIP_EMBED_DIM, MOBILECLIP_IMAGE_SIZE};

/// Errors raised by [`MobileClipSession`].
#[derive(Debug)]
pub enum MobileClipSessionError {
    LoadFailed { reason: String },
    InvalidGraph { reason: String },
    TensorBuildFailed { reason: String },
    InferenceFailed { reason: String },
    UnexpectedOutputShape { got: Vec<i64> },
}

impl std::fmt::Display for MobileClipSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoadFailed { reason } => write!(f, "mobileclip session load failed: {reason}"),
            Self::InvalidGraph { reason } => {
                write!(f, "mobileclip graph contract violated: {reason}")
            }
            Self::TensorBuildFailed { reason } => {
                write!(f, "mobileclip input tensor build failed: {reason}")
            }
            Self::InferenceFailed { reason } => write!(f, "mobileclip inference failed: {reason}"),
            Self::UnexpectedOutputShape { got } => write!(
                f,
                "mobileclip output shape mismatch: got {got:?}, expected [1, {MOBILECLIP_EMBED_DIM}]"
            ),
        }
    }
}

impl std::error::Error for MobileClipSessionError {}

/// Wrapper around `ort::Session` for the MobileCLIP-S2 image tower.
pub struct MobileClipSession {
    inner: Mutex<Session>,
    image_features_output_index: usize,
    image_input_name: String,
}

impl MobileClipSession {
    /// Load from a filesystem path.
    pub fn from_file(path: impl AsRef<Path>, intra_threads: usize) -> Result<Self, MobileClipSessionError> {
        let builder = Session::builder()
            .map_err(|e| MobileClipSessionError::LoadFailed { reason: format!("session builder: {e}") })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| MobileClipSessionError::LoadFailed { reason: format!("set optimization level: {e}") })?
            .with_intra_threads(intra_threads)
            .map_err(|e| MobileClipSessionError::LoadFailed { reason: format!("set intra threads: {e}") })?;
        let session = builder
            .commit_from_file(path.as_ref())
            .map_err(|e| MobileClipSessionError::LoadFailed { reason: format!("commit from file: {e}") })?;
        Self::wrap(session)
    }

    /// Load from in-memory bytes.
    pub fn from_bytes(bytes: &[u8], intra_threads: usize) -> Result<Self, MobileClipSessionError> {
        let builder = Session::builder()
            .map_err(|e| MobileClipSessionError::LoadFailed { reason: format!("session builder: {e}") })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| MobileClipSessionError::LoadFailed { reason: format!("set optimization level: {e}") })?
            .with_intra_threads(intra_threads)
            .map_err(|e| MobileClipSessionError::LoadFailed { reason: format!("set intra threads: {e}") })?;
        let session = builder
            .commit_from_memory(bytes)
            .map_err(|e| MobileClipSessionError::LoadFailed { reason: format!("commit from memory: {e}") })?;
        Self::wrap(session)
    }

    /// Run the image tower forward pass and return a 512-dim L2-normalised embedding.
    pub fn embed_image(&self, preprocessed_chw: &[f32]) -> Result<Vec<f32>, MobileClipSessionError> {
        let expected_len = 3 * MOBILECLIP_IMAGE_SIZE * MOBILECLIP_IMAGE_SIZE;
        if preprocessed_chw.len() != expected_len {
            return Err(MobileClipSessionError::InvalidGraph {
                reason: format!(
                    "preprocessed tensor length {} != expected {expected_len}",
                    preprocessed_chw.len(),
                ),
            });
        }

        let h = MOBILECLIP_IMAGE_SIZE as i64;
        let w = MOBILECLIP_IMAGE_SIZE as i64;
        let input_tensor = Tensor::from_array(([1_i64, 3, h, w], preprocessed_chw.to_vec()))
            .map_err(|e| MobileClipSessionError::TensorBuildFailed {
                reason: format!("image tensor: {e}"),
            })?;

        let raw_embedding: Vec<f32> = {
            let mut session = self.inner.lock();
            let outputs = session
                .run(ort::inputs! {
                    self.image_input_name.as_str() => input_tensor,
                })
                .map_err(|e| MobileClipSessionError::InferenceFailed {
                    reason: format!("session.run: {e}"),
                })?;
            let (shape, data) = outputs[self.image_features_output_index]
                .try_extract_tensor::<f32>()
                .map_err(|e| MobileClipSessionError::InferenceFailed {
                    reason: format!("extract image_features: {e}"),
                })?;
            let dims: &[i64] = shape;
            validate_image_features_shape(dims)?;
            if data.len() != MOBILECLIP_EMBED_DIM {
                return Err(MobileClipSessionError::InvalidGraph {
                    reason: format!(
                        "image_features slice has {} elements but expected {MOBILECLIP_EMBED_DIM}",
                        data.len(),
                    ),
                });
            }
            data.to_vec()
        };
        Ok(l2_normalise(raw_embedding))
    }

    fn wrap(session: Session) -> Result<Self, MobileClipSessionError> {
        let image_input_name = session
            .inputs
            .iter()
            .find(|inp| inp.name == "image" || inp.name == "pixel_values" || inp.name == "input")
            .map(|inp| inp.name.clone())
            .or_else(|| session.inputs.first().map(|inp| inp.name.clone()))
            .ok_or_else(|| MobileClipSessionError::InvalidGraph {
                reason: "model declares zero inputs".to_string(),
            })?;

        let image_features_output_index = session
            .outputs
            .iter()
            .enumerate()
            .find_map(|(idx, out)| {
                if out.name == "image_features" || out.name == "image_embeds" || out.name == "output" {
                    Some(idx)
                } else {
                    None
                }
            })
            .or((!session.outputs.is_empty()).then_some(0))
            .ok_or_else(|| MobileClipSessionError::InvalidGraph {
                reason: "model declares zero outputs".to_string(),
            })?;

        Ok(Self {
            inner: Mutex::new(session),
            image_features_output_index,
            image_input_name,
        })
    }
}

impl std::fmt::Debug for MobileClipSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MobileClipSession")
            .field("image_input_name", &self.image_input_name)
            .field("image_features_output_index", &self.image_features_output_index)
            .finish_non_exhaustive()
    }
}

fn validate_image_features_shape(dims: &[i64]) -> Result<(), MobileClipSessionError> {
    let expected_dim = MOBILECLIP_EMBED_DIM as i64;
    let shape_ok = match dims {
        [d] => *d == expected_dim,
        [1, d] => *d == expected_dim,
        _ => false,
    };
    if shape_ok {
        Ok(())
    } else {
        Err(MobileClipSessionError::UnexpectedOutputShape { got: dims.to_vec() })
    }
}

/// L2-normalise a vector in place. NaN inputs collapse to 0.0.
pub(crate) fn l2_normalise(mut vec: Vec<f32>) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
        vec.fill(0.0);
    } else {
        for v in &mut vec {
            *v /= norm;
        }
    }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_normalise_unit_vector_returns_unit_vector() {
        let v = vec![1.0_f32 / (3.0_f32).sqrt(); 3];
        let n = l2_normalise(v);
        let norm: f32 = n.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalise_zero_vector_returns_zero_vector() {
        let n = l2_normalise(vec![0.0_f32; MOBILECLIP_EMBED_DIM]);
        for v in &n {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn l2_normalise_nan_input_returns_zero_vector() {
        let n = l2_normalise(vec![f32::NAN; MOBILECLIP_EMBED_DIM]);
        for v in &n {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn validate_shape_accepts_canonical() {
        validate_image_features_shape(&[1, MOBILECLIP_EMBED_DIM as i64]).unwrap();
    }

    #[test]
    fn validate_shape_accepts_squeezed() {
        validate_image_features_shape(&[MOBILECLIP_EMBED_DIM as i64]).unwrap();
    }

    #[test]
    fn validate_shape_rejects_wrong_batch() {
        let err = validate_image_features_shape(&[2, 256]).expect_err("must reject");
        assert!(matches!(err, MobileClipSessionError::UnexpectedOutputShape { .. }));
    }

    #[test]
    fn validate_shape_rejects_wrong_dim() {
        let err = validate_image_features_shape(&[1, 384]).expect_err("must reject");
        assert!(matches!(err, MobileClipSessionError::UnexpectedOutputShape { .. }));
    }

    #[test]
    #[ignore = "requires libonnxruntime.dylib installed"]
    fn from_bytes_garbage_returns_load_failed() {
        let err = MobileClipSession::from_bytes(b"not an onnx model", 0).expect_err("must fail");
        assert!(matches!(err, MobileClipSessionError::LoadFailed { .. }));
    }

    #[test]
    #[ignore = "requires libonnxruntime.dylib installed"]
    fn from_file_missing_path_returns_load_failed() {
        let err = MobileClipSession::from_file("/nonexistent/path/to/model.onnx", 0).expect_err("must fail");
        assert!(matches!(err, MobileClipSessionError::LoadFailed { .. }));
    }
}

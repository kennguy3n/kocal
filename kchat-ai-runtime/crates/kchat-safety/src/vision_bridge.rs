//! Vision bridge — connects the vision encoder's `MediaDescriptor` output
//! to the safety classification pipeline.
//!
//! Ported from slm-guardrail's `pipeline/vision_encoder_bridge.rs`, adapted
//! to kocal's architecture where the vision adapter produces `MediaDescriptor`
//! scores (not prototype-bank verdicts) and the safety pipeline consumes
//! them via `ClassifyRequest::with_media()`.
//!
//! Flow:
//! ```text
//! image bytes (PNG / JPEG / WebP)
//!   ↓ VisionImageClassifier::classify_image
//! VisionEncoderVerdict (MediaDescriptor + embedding)
//!   ↓ vision_bridge::classify_image
//! ClassifyResult (verdict + reason codes)
//! ```
//!
//! On encoder error, the bridge returns a safe-degraded `ClassifyResult`
//! with no media descriptors (the deterministic pipeline runs without
//! media branch detectors).

use std::sync::Arc;

use crate::classify::{ClassifyRequest, ClassifyResult, SafetyClassifier};
use crate::vision::{VisionEncoderError, VisionImageClassifier};

/// Classify an image through the supplied vision encoder and feed the
/// resulting `MediaDescriptor` into the safety pipeline.
///
/// The `text` parameter provides any accompanying text context (e.g. a
/// message caption). The vision-derived media descriptors are attached
/// to the `ClassifyRequest` and evaluated by the media branch detectors
/// in the priority chain before any text-based detectors run.
///
/// On encoder error, returns a `ClassifyResult` with no media descriptors
/// — the deterministic text pipeline still runs on the provided text.
pub fn classify_image(
    image_bytes: &[u8],
    text: &str,
    classifier: &dyn VisionImageClassifier,
    safety: &SafetyClassifier,
) -> ClassifyResult {
    let media_descriptors = match classifier.classify_image(image_bytes) {
        Ok(vision_verdict) => vec![vision_verdict.descriptor],
        Err(_err) => {
            tracing::warn!(
                target = "kchat-safety.vision_bridge",
                "vision encoder inference failed, falling back to text-only pipeline"
            );
            Vec::new()
        }
    };

    let request = ClassifyRequest::from_text(text).with_media(media_descriptors);
    safety.classify(&request)
}

/// Owning wrapper around an `Arc<dyn VisionImageClassifier>`.
///
/// The FFI / bindings runtimes store one of these in their internal state.
/// `Clone` is cheap (single `Arc::clone`).
#[derive(Clone)]
pub struct VisionBridge {
    inner: Arc<dyn VisionImageClassifier>,
}

impl VisionBridge {
    /// Construct a wrapper around `inner`.
    pub fn new(inner: Arc<dyn VisionImageClassifier>) -> Self {
        Self { inner }
    }

    /// Borrow the wrapped classifier.
    pub fn classifier(&self) -> &Arc<dyn VisionImageClassifier> {
        &self.inner
    }

    /// Classify `image_bytes` with accompanying `text` through the vision
    /// encoder + safety pipeline.
    pub fn classify(
        &self,
        image_bytes: &[u8],
        text: &str,
        safety: &SafetyClassifier,
    ) -> ClassifyResult {
        classify_image(image_bytes, text, self.inner.as_ref(), safety)
    }

    /// Encode `image_bytes` to a 512-dim L2-normalised embedding.
    pub fn encode_image(&self, image_bytes: &[u8]) -> Result<Vec<f32>, VisionEncoderError> {
        self.inner.encode_image(image_bytes)
    }
}

impl std::fmt::Debug for VisionBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionBridge")
            .field("inner", &"<dyn VisionImageClassifier>")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::MediaDescriptor;
    use crate::vision::VisionEncoderVerdict;

    /// Stub classifier that returns a fixed MediaDescriptor.
    struct FakeVisionClassifier {
        descriptor: Option<MediaDescriptor>,
    }

    impl VisionImageClassifier for FakeVisionClassifier {
        fn encode_image(&self, _image_bytes: &[u8]) -> Result<Vec<f32>, VisionEncoderError> {
            Ok(vec![0.0; 512])
        }

        fn classify_image(&self, _image_bytes: &[u8]) -> Result<VisionEncoderVerdict, VisionEncoderError> {
            self.descriptor
                .clone()
                .map(|d| VisionEncoderVerdict::new(d, vec![0.0; 512]))
                .ok_or(VisionEncoderError::InvalidConfiguration {
                    reason: "synthetic failure".into(),
                })
        }
    }

    #[test]
    fn wrapper_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VisionBridge>();
    }

    #[test]
    fn failure_path_returns_empty_descriptors() {
        // When the vision encoder fails, the bridge should still
        // produce a ClassifyResult via the text-only pipeline.
        let classifier = FakeVisionClassifier { descriptor: None };
        let safety = SafetyClassifier::new();
        let result = classify_image(b"placeholder", "hello world", &classifier, &safety);
        // The text "hello world" is safe, so the result should be Allow.
        assert_eq!(result.verdict.action, crate::verdict::Action::Allow);
    }

    #[test]
    fn happy_path_attaches_media_descriptor() {
        let descriptor = MediaDescriptor {
            kind: "image".into(),
            nsfw_score: Some(0.95),
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
        };
        let classifier = FakeVisionClassifier {
            descriptor: Some(descriptor),
        };
        let safety = SafetyClassifier::new();
        let result = classify_image(b"placeholder", "hello", &classifier, &safety);
        // NSFW score 0.95 > MEDIA_TRIGGER_THRESHOLD (0.7) → should block.
        assert_eq!(result.verdict.action, crate::verdict::Action::Block);
    }
}

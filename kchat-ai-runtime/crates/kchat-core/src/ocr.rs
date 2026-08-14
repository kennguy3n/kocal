//! Platform OCR bridge.
//!
//! On-device OCR is routed through the platform-native vision
//! stacks (Apple `VNRecognizeText`, ML Kit on Android,
//! `Windows.Media.Ocr` / Tesseract on Windows). Those backends
//! live outside the Rust core, so this module defines the
//! [`OcrBridge`] trait — an object-safe `Send + Sync` seam that
//! the platform glue (Swift / Kotlin / C++) implements.
//!
//! The trait surface is deliberately thin: one input (image bytes
//! plus MIME type) and one output (a vector of [`OcrResult`]).

use crate::error::{CoreError, Result};

/// One recognized-text region produced by an [`OcrBridge`] call.
///
/// `text` is the recognized string. `language` is a BCP-47 tag
/// when the platform reports it (e.g. `"en"`, `"zh-Hans"`,
/// `"ja"`); `None` means the platform did not detect a language
/// or the bridge does not surface that capability.
/// `confidence` is in `[0.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct OcrResult {
    /// Recognized text for this region.
    pub text: String,
    /// Detected language as a BCP-47 tag, when the platform
    /// surfaces one.
    pub language: Option<String>,
    /// Per-region confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Optional bounding box in image-pixel space.
    pub bounding_box: Option<BoundingBox>,
}

/// Image-pixel-space bounding box for an [`OcrResult`].
///
/// Coordinates are top-left-origin in the same coordinate space
/// as the input image (i.e. `(x, y)` is the top-left corner and
/// `(width, height)` extends right / down). Stored as `f32` so
/// platforms that report sub-pixel anchors (e.g. iOS Vision)
/// don't lose precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// Left edge, in image-pixel units.
    pub x: f32,
    /// Top edge, in image-pixel units.
    pub y: f32,
    /// Width of the box, in image-pixel units.
    pub width: f32,
    /// Height of the box, in image-pixel units.
    pub height: f32,
}

/// Object-safe seam for the platform OCR backend.
///
/// `recognize_text` runs the platform's text-recognition stack
/// over `image_data` and returns the recognized regions. The
/// trait is `Send + Sync` so the orchestration layer can stash
/// it behind a `Mutex<Option<Arc<dyn OcrBridge>>>` slot and fan
/// out from background workers.
pub trait OcrBridge: std::fmt::Debug + Send + Sync {
    /// Run OCR over `image_data`. `mime_type` carries the
    /// platform's MIME hint (`"image/jpeg"`, `"image/png"`, …)
    /// so the bridge can dispatch to the right decoder. Returns
    /// `Ok(vec![])` when the image contained no recognizable
    /// text — that is a successful run with zero hits, not an
    /// error.
    fn recognize_text(&self, image_data: &[u8], mime_type: &str) -> Result<Vec<OcrResult>>;
}

/// Graceful-skip `OcrBridge` for builds without a platform glue
/// layer.
///
/// On platforms where no native OCR backend is wired in, this
/// bridge degrades gracefully: `recognize_text` returns
/// `Ok(vec![])` (zero recognized regions) rather than an error,
/// so the media-indexing pipeline treats the image as "no text
/// found" and keeps processing the rest of the batch.
#[derive(Debug, Default, Clone, Copy)]
pub struct SkipOcrBridge;

impl OcrBridge for SkipOcrBridge {
    fn recognize_text(&self, image_data: &[u8], mime_type: &str) -> Result<Vec<OcrResult>> {
        tracing::debug!(
            image_bytes = image_data.len(),
            mime_type,
            "OCR skipped: no platform OCR bridge wired in; returning zero regions"
        );
        Ok(Vec::new())
    }
}

/// Deterministic mock [`OcrBridge`] keyed by a BLAKE3 hash of
/// `(mime_type, image_data)`. Used by integration tests to
/// stand in for a real platform OCR backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct MockOcrBridge;

impl OcrBridge for MockOcrBridge {
    fn recognize_text(&self, image_data: &[u8], mime_type: &str) -> Result<Vec<OcrResult>> {
        if !mime_type.starts_with("image/") {
            return Err(CoreError::Storage(format!(
                "MockOcrBridge rejects non-image mime_type: {mime_type}"
            )));
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(mime_type.as_bytes());
        hasher.update(&[0]);
        hasher.update(image_data);
        let hash = hasher.finalize();
        let hex = hash.to_hex();
        let prefix: String = hex.as_str().chars().take(16).collect();
        Ok(vec![
            OcrResult {
                text: format!("mock ocr {prefix} line 1"),
                language: Some("en".to_string()),
                confidence: 0.95,
                bounding_box: Some(BoundingBox {
                    x: 0.0,
                    y: 0.0,
                    width: 200.0,
                    height: 30.0,
                }),
            },
            OcrResult {
                text: format!("mock ocr {prefix} line 2"),
                language: Some("en".to_string()),
                confidence: 0.85,
                bounding_box: Some(BoundingBox {
                    x: 0.0,
                    y: 35.0,
                    width: 200.0,
                    height: 30.0,
                }),
            },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test bridge that always returns the same canned hits.
    #[derive(Debug)]
    struct FakeBridge {
        hits: Vec<OcrResult>,
    }

    impl OcrBridge for FakeBridge {
        fn recognize_text(&self, _image_data: &[u8], _mime_type: &str) -> Result<Vec<OcrResult>> {
            Ok(self.hits.clone())
        }
    }

    #[test]
    fn skip_bridge_returns_empty_regions() {
        let bridge = SkipOcrBridge;
        let hits = bridge
            .recognize_text(b"unused", "image/png")
            .expect("skip bridge degrades gracefully");
        assert!(hits.is_empty());
    }

    #[test]
    fn fake_bridge_round_trips_hits_through_dyn_dispatch() {
        let bridge = FakeBridge {
            hits: vec![
                OcrResult {
                    text: "Hello, world".into(),
                    language: Some("en".into()),
                    confidence: 0.95,
                    bounding_box: Some(BoundingBox {
                        x: 10.0,
                        y: 20.0,
                        width: 100.0,
                        height: 30.0,
                    }),
                },
                OcrResult {
                    text: "你好世界".into(),
                    language: Some("zh-Hans".into()),
                    confidence: 0.87,
                    bounding_box: None,
                },
            ],
        };
        let dynref: &dyn OcrBridge = &bridge;
        let hits = dynref.recognize_text(b"unused", "image/jpeg").unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].text, "Hello, world");
        assert_eq!(hits[0].language.as_deref(), Some("en"));
        assert!((hits[0].confidence - 0.95).abs() < 1e-6);
        assert_eq!(hits[1].language.as_deref(), Some("zh-Hans"));
    }

    #[test]
    fn bounding_box_is_copy_and_eq() {
        let a = BoundingBox {
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
        };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn mock_bridge_rejects_non_image_mime() {
        let bridge = MockOcrBridge;
        let err = bridge.recognize_text(b"bytes", "text/plain").unwrap_err();
        assert!(err.to_string().contains("non-image"));
    }

    #[test]
    fn mock_bridge_returns_deterministic_results() {
        let bridge = MockOcrBridge;
        let a = bridge.recognize_text(b"image-data", "image/png").unwrap();
        let b = bridge.recognize_text(b"image-data", "image/png").unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
        assert!(a[0].text.starts_with("mock ocr"));
    }
}

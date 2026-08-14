//! Whisper backend selection — Apple MLX (preferred on Apple
//! Silicon) with ONNX Runtime as the cross-platform fallback.
//!
//! On Apple Silicon (`macOS` / `iOS` `aarch64`), `Whisper-base`
//! runs through Apple MLX, which routes inference to the Neural
//! Engine. On every other target, the pipeline falls back to
//! the multilingual `whisper-base.int8.onnx` artifact through
//! ONNX Runtime.

// ---------------------------------------------------------------------------
// Canonical model identifiers
// ---------------------------------------------------------------------------

/// Canonical `model_version` tag for the MLX-flavored
/// `Whisper-base` artifact shipped to Apple Silicon devices.
pub const WHISPER_BASE_MLX_MODEL_VERSION: &str = "whisper_base_mlx@v1";

/// Canonical `model_version` tag for the ONNX-flavored
/// `Whisper-base` artifact shipped to non-Apple-Silicon devices.
pub const WHISPER_BASE_ONNX_MODEL_VERSION: &str = "whisper_base_onnx_int8@v1";

/// Hugging Face repo identifier for the MLX-quantized
/// `Whisper-base` artifact.
pub const WHISPER_BASE_MLX_MODEL_REPO: &str = "mlx-community/whisper-base-mlx";

/// Filename of the ONNX-quantized `Whisper-base` artifact
/// downloaded on every non-Apple-Silicon target.
pub const WHISPER_BASE_ONNX_ARTIFACT: &str = "whisper-base.int8.onnx";

// ---------------------------------------------------------------------------
// Backend-selection state machine — always compiled, no `mlx-rs` /
// no `ort` dependency, exhaustively unit-tested on any host.
// ---------------------------------------------------------------------------

/// Identifier for the Whisper backend that was actually
/// selected for a given device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WhisperBackend {
    /// Apple MLX — preferred on Apple Silicon.
    Mlx,
    /// ONNX Runtime — cross-platform fallback.
    Onnx,
}

impl WhisperBackend {
    /// Stable, telemetry-friendly name for the selected backend.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mlx => "MLX",
            Self::Onnx => "ONNX",
        }
    }

    /// Canonical `model_version` tag.
    pub const fn model_version(self) -> &'static str {
        match self {
            Self::Mlx => WHISPER_BASE_MLX_MODEL_VERSION,
            Self::Onnx => WHISPER_BASE_ONNX_MODEL_VERSION,
        }
    }
}

/// Result of [`select_whisper_backend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WhisperBackendReport {
    /// The backend that should be used.
    pub backend: WhisperBackend,
    /// `true` if the [`AppleSiliconProbe`] was consulted.
    pub mlx_attempted: bool,
}

/// Cheap probe of MLX availability on Apple Silicon.
pub trait AppleSiliconProbe {
    /// Return `true` if MLX inference is available.
    fn mlx_available(&self) -> bool;
}

/// Pure backend-selection function.
pub fn select_whisper_backend<P: AppleSiliconProbe + ?Sized>(
    probe: &P,
) -> WhisperBackendReport {
    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
    {
        if probe.mlx_available() {
            return WhisperBackendReport {
                backend: WhisperBackend::Mlx,
                mlx_attempted: true,
            };
        }
        WhisperBackendReport {
            backend: WhisperBackend::Onnx,
            mlx_attempted: true,
        }
    }
    #[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))))]
    {
        let _ = probe;
        WhisperBackendReport {
            backend: WhisperBackend::Onnx,
            mlx_attempted: false,
        }
    }
}

/// Hugging Face repo / artifact identifier the model manager
/// downloads for a given backend.
pub fn whisper_base_artifact_for(backend: WhisperBackend) -> &'static str {
    match backend {
        WhisperBackend::Mlx => WHISPER_BASE_MLX_MODEL_REPO,
        WhisperBackend::Onnx => WHISPER_BASE_ONNX_ARTIFACT,
    }
}

// ---------------------------------------------------------------------------
// Production probe — Apple Silicon only.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
mod apple_silicon {
    use super::AppleSiliconProbe;

    #[derive(Debug, Default, Clone, Copy)]
    pub struct MlxAppleSiliconProbe;

    impl AppleSiliconProbe for MlxAppleSiliconProbe {
        fn mlx_available(&self) -> bool {
            true
        }
    }
}

#[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
pub use apple_silicon::MlxAppleSiliconProbe;

#[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))))]
mod non_apple_silicon {
    use super::AppleSiliconProbe;

    #[derive(Debug, Default, Clone, Copy)]
    pub struct MlxAppleSiliconProbe;

    impl AppleSiliconProbe for MlxAppleSiliconProbe {
        fn mlx_available(&self) -> bool {
            false
        }
    }
}

#[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))))]
pub use non_apple_silicon::MlxAppleSiliconProbe;

// ---------------------------------------------------------------------------
// WhisperTranscriber trait
// ---------------------------------------------------------------------------

use crate::AsrError;

/// One contiguous timed segment of a Whisper transcription result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionSegment {
    /// Inclusive lower-bound timestamp in milliseconds.
    pub start_ms: u64,
    /// Inclusive upper-bound timestamp in milliseconds.
    pub end_ms: u64,
    /// Plaintext for this segment.
    pub text: String,
}

/// Result of [`WhisperTranscriber::transcribe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionResult {
    /// Concatenated transcript text.
    pub text: String,
    /// Detected language tag (`None` when not reported).
    pub language: Option<String>,
    /// Per-segment timeline.
    pub segments: Vec<TranscriptionSegment>,
}

/// On-device Whisper transcription seam.
pub trait WhisperTranscriber: std::fmt::Debug + Send + Sync {
    /// Run Whisper inference over `audio_data` and return the
    /// transcription.
    fn transcribe(&self, audio_data: &[u8], mime_type: &str)
        -> Result<TranscriptionResult, AsrError>;
}

/// Alias matching the task-spec name.
pub use WhisperTranscriber as AudioTranscriber;

/// Graceful-skip [`WhisperTranscriber`] for builds without a real
/// Whisper backend.
#[derive(Debug, Default, Clone, Copy)]
pub struct SkipWhisperTranscriber;

impl WhisperTranscriber for SkipWhisperTranscriber {
    fn transcribe(
        &self,
        audio_data: &[u8],
        mime_type: &str,
    ) -> Result<TranscriptionResult, AsrError> {
        tracing::debug!(
            audio_bytes = audio_data.len(),
            mime_type,
            "whisper transcription skipped: no transcriber wired in; returning empty transcript"
        );
        Ok(TranscriptionResult {
            text: String::new(),
            language: None,
            segments: Vec::new(),
        })
    }
}

/// Deterministic test [`WhisperTranscriber`] that derives a
/// reproducible transcription from a BLAKE3 hash.
#[derive(Debug, Default, Clone, Copy)]
pub struct MockWhisperTranscriber;

impl WhisperTranscriber for MockWhisperTranscriber {
    fn transcribe(
        &self,
        audio_data: &[u8],
        mime_type: &str,
    ) -> Result<TranscriptionResult, AsrError> {
        if !mime_type.starts_with("audio/") {
            return Err(AsrError::AudioDecode {
                op: "transcribe",
                detail: format!(
                    "MockWhisperTranscriber rejects non-audio mime_type: {mime_type}"
                ),
            });
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(mime_type.as_bytes());
        hasher.update(&[0]);
        hasher.update(audio_data);
        let hash = hasher.finalize();
        let hex = hash.to_hex();
        let prefix: String = hex.as_str().chars().take(16).collect();
        let span_ms = (audio_data.len() as u64).saturating_mul(10).max(20);
        let mid = span_ms / 2;
        let text = format!("mock transcription [{prefix}]");
        let segments = vec![
            TranscriptionSegment {
                start_ms: 0,
                end_ms: mid,
                text: format!("mock segment 1 [{prefix}]"),
            },
            TranscriptionSegment {
                start_ms: mid,
                end_ms: span_ms,
                text: format!("mock segment 2 [{prefix}]"),
            },
        ];
        Ok(TranscriptionResult {
            text,
            language: Some("en".to_string()),
            segments,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct StubProbe(bool);
    impl AppleSiliconProbe for StubProbe {
        fn mlx_available(&self) -> bool {
            self.0
        }
    }

    #[test]
    fn backend_name_is_stable() {
        assert_eq!(WhisperBackend::Mlx.name(), "MLX");
        assert_eq!(WhisperBackend::Onnx.name(), "ONNX");
    }

    #[test]
    fn model_version_tags_distinguish_backends() {
        assert_ne!(
            WhisperBackend::Mlx.model_version(),
            WhisperBackend::Onnx.model_version()
        );
        assert!(WhisperBackend::Mlx.model_version().contains('@'));
        assert!(WhisperBackend::Onnx.model_version().contains('@'));
    }

    #[test]
    fn artifact_repo_split_per_backend() {
        assert_eq!(
            whisper_base_artifact_for(WhisperBackend::Mlx),
            "mlx-community/whisper-base-mlx"
        );
        assert_eq!(
            whisper_base_artifact_for(WhisperBackend::Onnx),
            "whisper-base.int8.onnx"
        );
    }

    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn mlx_preferred_when_available_on_apple_silicon() {
        let report = select_whisper_backend(&StubProbe(true));
        assert_eq!(report.backend, WhisperBackend::Mlx);
        assert!(report.mlx_attempted);
    }

    #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn onnx_fallback_when_mlx_unavailable_on_apple_silicon() {
        let report = select_whisper_backend(&StubProbe(false));
        assert_eq!(report.backend, WhisperBackend::Onnx);
        assert!(report.mlx_attempted);
    }

    #[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))))]
    #[test]
    fn mlx_never_attempted_off_apple_silicon_even_if_probe_lies() {
        let report = select_whisper_backend(&StubProbe(true));
        assert_eq!(report.backend, WhisperBackend::Onnx);
        assert!(!report.mlx_attempted);
    }

    #[cfg(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))))]
    #[test]
    fn onnx_selected_off_apple_silicon_with_unavailable_probe() {
        let report = select_whisper_backend(&StubProbe(false));
        assert_eq!(report.backend, WhisperBackend::Onnx);
        assert!(!report.mlx_attempted);
    }

    #[test]
    fn select_backend_accepts_dyn_probe() {
        let probe: &dyn AppleSiliconProbe = &StubProbe(false);
        let report = select_whisper_backend(probe);
        assert_eq!(report.backend, WhisperBackend::Onnx);
    }

    #[test]
    fn skip_whisper_transcriber_returns_empty_transcript() {
        let t = SkipWhisperTranscriber;
        let result = t
            .transcribe(b"audio-bytes", "audio/wav")
            .expect("skip transcriber degrades gracefully");
        assert!(result.text.is_empty());
        assert!(result.language.is_none());
        assert!(result.segments.is_empty());
    }

    #[test]
    fn mock_whisper_transcriber_is_deterministic() {
        let t = MockWhisperTranscriber;
        let a = t.transcribe(b"hello-audio", "audio/wav").expect("a");
        let b = t.transcribe(b"hello-audio", "audio/wav").expect("b");
        assert_eq!(a, b);
        let c = t.transcribe(b"different-audio", "audio/wav").expect("c");
        assert_ne!(a.text, c.text);
        assert_eq!(a.segments.len(), 2);
        assert_eq!(a.language.as_deref(), Some("en"));
    }

    #[test]
    fn mock_whisper_transcriber_rejects_non_audio_mime() {
        let t = MockWhisperTranscriber;
        let err = t.transcribe(b"bytes", "text/plain").unwrap_err();
        assert!(matches!(err, AsrError::AudioDecode { .. }));
    }

    #[test]
    fn whisper_transcriber_trait_is_object_safe() {
        let mock = MockWhisperTranscriber;
        let dynref: &dyn WhisperTranscriber = &mock;
        let result = dynref.transcribe(b"X", "audio/mpeg").unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn audio_transcriber_skip_returns_empty_transcript() {
        let t: &dyn AudioTranscriber = &SkipWhisperTranscriber;
        let result = t
            .transcribe(b"audio", "audio/mpeg")
            .expect("skip transcriber degrades gracefully");
        assert!(result.text.is_empty());
        assert!(result.segments.is_empty());
    }

    #[test]
    fn mock_audio_transcriber_returns_configured_text() {
        let t: &dyn AudioTranscriber = &MockWhisperTranscriber;
        let result = t.transcribe(b"hello-audio", "audio/wav").unwrap();
        assert!(result.text.starts_with("mock transcription"));
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.language.as_deref(), Some("en"));
    }
}

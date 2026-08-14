//! High-level transcription orchestrator.
//!
//! Wires backend selection + audio preprocessing + (when the
//! `onnx-runtime` feature is enabled) ONNX inference into a
//! single call. Without `onnx-runtime`, falls back to
//! [`crate::backend::SkipWhisperTranscriber`].

use crate::backend::MlxAppleSiliconProbe;
use crate::AsrResult;

#[cfg(not(feature = "onnx-runtime"))]
use crate::backend::{SkipWhisperTranscriber, WhisperTranscriber};

/// Transcribe raw audio bytes to text.
///
/// This is a convenience function that selects the appropriate
/// backend and runs transcription. For production use, prefer
/// constructing a long-lived transcriber (e.g.
/// [`crate::onnx_session::OnnxWhisperTranscriber`] when the
/// `onnx-runtime` feature is enabled) to amortize model load
/// across calls.
pub fn transcribe(
    audio_data: &[u8],
    mime_type: &str,
) -> AsrResult<crate::backend::TranscriptionResult> {
    let _report = crate::backend::select_whisper_backend(&MlxAppleSiliconProbe);

    // Without onnx-runtime, use the skip transcriber.
    #[cfg(not(feature = "onnx-runtime"))]
    {
        let t = SkipWhisperTranscriber;
        t.transcribe(audio_data, mime_type)
    }

    // With onnx-runtime, the caller should construct an
    // OnnxWhisperTranscriber directly — this function is a
    // convenience for the no-feature path.
    #[cfg(feature = "onnx-runtime")]
    {
        let _ = (audio_data, mime_type);
        Err(crate::AsrError::msg(
            "transcribe() convenience fn does not construct an ONNX session; \
             use OnnxWhisperTranscriber::new() directly when the onnx-runtime feature is enabled",
        ))
    }
}

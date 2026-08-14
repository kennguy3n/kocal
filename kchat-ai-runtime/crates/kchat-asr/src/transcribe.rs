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
/// **Without `onnx-runtime`:** uses [`crate::backend::SkipWhisperTranscriber`]
/// which returns an empty transcript (audio ingestion continues but
/// no transcription is produced).
///
/// **With `onnx-runtime`:** returns [`AsrError::Custom`] because this
/// convenience function cannot construct an [`OnnxWhisperTranscriber`]
/// without knowing where the model artifacts live on disk. Callers
/// that need real transcription should construct an
/// [`crate::onnx_session::OnnxWhisperTranscriber`] with the
/// encoder/decoder/tokenizer paths and call `transcribe` on it
/// directly. This amortizes the session-load cost across calls.
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
    // OnnxWhisperTranscriber directly — this function cannot
    // load model artifacts from an unknown location.
    #[cfg(feature = "onnx-runtime")]
    {
        let _ = (audio_data, mime_type);
        Err(crate::AsrError::msg(
            "transcribe() convenience fn cannot construct an OnnxWhisperTranscriber \
             without model paths; use OnnxWhisperTranscriber::new(encoder_dir, intra_threads) \
             directly when the onnx-runtime feature is enabled",
        ))
    }
}

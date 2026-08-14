//! ONNX Runtime Whisper encoder/decoder inference loop.
//!
//! Gated behind the `onnx-runtime` feature. Provides
//! [`OnnxWhisperTranscriber`] which implements
//! [`crate::backend::WhisperTranscriber`].
//!
//! **Note:** This is a placeholder. The full inference loop is
//! being ported from `chat-storage-search` and will replace this
//! file.

#![cfg(feature = "onnx-runtime")]

use crate::backend::{TranscriptionResult, TranscriptionSegment, WhisperTranscriber};
use crate::{AsrError, AsrResult};

/// ONNX Runtime Whisper transcriber.
///
/// Wraps encoder + decoder ONNX sessions and a tokenizer to
/// perform greedy-decode Whisper transcription.
#[derive(Debug)]
pub struct OnnxWhisperTranscriber {
    _intra_threads: usize,
}

impl OnnxWhisperTranscriber {
    /// Create a new transcriber from ONNX model files.
    ///
    /// `encoder_path` / `decoder_path` are the Whisper encoder/decoder
    /// ONNX model files; `tokenizer_path` is the tokenizer.json.
    pub fn new(
        encoder_path: &str,
        decoder_path: &str,
        tokenizer_path: &str,
        intra_threads: usize,
    ) -> AsrResult<Self> {
        let _ = (encoder_path, decoder_path, tokenizer_path);
        // TODO: port full inference loop from chat-storage-search
        // For now, return a stub that will fail at transcribe time.
        Ok(Self {
            _intra_threads: intra_threads,
        })
    }
}

impl WhisperTranscriber for OnnxWhisperTranscriber {
    fn transcribe(
        &self,
        audio_data: &[u8],
        mime_type: &str,
    ) -> AsrResult<TranscriptionResult> {
        let _ = (audio_data, mime_type);
        Err(AsrError::msg(
            "OnnxWhisperTranscriber inference loop not yet ported; \
             use SkipWhisperTranscriber or MockWhisperTranscriber for testing",
        ))
    }
}

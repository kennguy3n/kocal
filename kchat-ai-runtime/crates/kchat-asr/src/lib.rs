//! kchat-asr — On-device Whisper audio transcription.
//!
//! Provides the Whisper ASR pipeline:
//!
//! - [`backend`] — backend selection state machine (Apple MLX on
//!   Apple Silicon, ONNX Runtime elsewhere) + the
//!   [`WhisperTranscriber`] trait.
//! - [`audio`] — pure-CPU audio preprocessing: WAV decode →
//!   16 kHz mono PCM → log-mel spectrogram `[80 × 3000]`.
//! - [`onnx_session`] — ONNX Runtime Whisper encoder/decoder
//!   inference loop (gated behind the `onnx-runtime` feature).
//! - [`transcribe`] — high-level orchestrator that wires
//!   backend selection + preprocessing + inference into a
//!   single `transcribe(audio, sample_rate) -> String` call.
//!
//! # Feature gating
//!
//! Without the `onnx-runtime` feature, the crate compiles with
//! backend selection, audio preprocessing, and a
//! [`backend::SkipWhisperTranscriber`] that returns empty
//! transcripts. Enable `onnx-runtime` for real Whisper inference.

pub mod audio;
pub mod backend;

pub mod onnx_session;

pub mod transcribe;

pub use backend::{
    select_whisper_backend, whisper_base_artifact_for, AudioTranscriber, MlxAppleSiliconProbe,
    SkipWhisperTranscriber, TranscriptionResult, TranscriptionSegment, WhisperBackend,
    WhisperBackendReport, WhisperTranscriber,
};

/// ASR error type.
#[derive(Debug, thiserror::Error)]
pub enum AsrError {
    /// Audio decoding failed (invalid WAV, unsupported codec, …).
    #[error("audio decode ({op}): {detail}")]
    AudioDecode { op: &'static str, detail: String },

    /// ONNX Runtime call failed (session create, inference).
    #[error("ort ({op}): {detail}")]
    Ort { op: &'static str, detail: String },

    /// Tokenizer call failed.
    #[error("tokenizer ({op}): {detail}")]
    Tokenizer { op: &'static str, detail: String },

    /// The requested model artifact is not present in the on-device
    /// cache.
    #[error("model `{0}` not cached")]
    NotCached(&'static str),

    /// A `Mutex` guarding an ASR resource was poisoned.
    #[error("`{0}` lock poisoned")]
    LockPoisoned(&'static str),

    /// Free-form fallback.
    #[error("{0}")]
    Custom(String),
}

impl AsrError {
    /// Construct an [`AsrError::Custom`] from anything convertible
    /// to [`String`].
    pub fn msg(msg: impl Into<String>) -> Self {
        AsrError::Custom(msg.into())
    }
}

pub type AsrResult<T> = std::result::Result<T, AsrError>;

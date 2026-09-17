//! whisper.cpp in-process ASR backend.
//!
//! Runs GGML/GGUF Whisper models (e.g. `ggml-base.en.bin`, `ggml-tiny.bin`)
//! fully in-process via `whisper-rs` — no subprocess, no ONNX Runtime, and
//! hardware-accelerated (Metal on Apple Silicon, CoreML/ANE when built with
//! the CoreML path, CUDA/Vulkan elsewhere via whisper.cpp's ggml backends).
//! This is the preferred ASR path on mobile, where the ONNX encoder/decoder
//! pipeline cannot ship.
//!
//! # Feature
//!
//! Gated behind `whispercpp` (`dep:whisper-rs`). Model artifacts are the
//! `whisper-*-gguf` registry packs (ggml format), distinct from the ONNX
//! `whisper-tiny`/`whisper-base` packs used by [`crate::onnx_session`].

use crate::audio::{whisper_decode_wav, whisper_to_mono_16k};
use crate::backend::{TranscriptionResult, TranscriptionSegment, WhisperTranscriber};
use crate::AsrError;
use std::path::Path;
use std::sync::Mutex;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// In-process whisper.cpp transcriber.
///
/// `WhisperContext` is cheap to keep resident (model stays mmap'd); each
/// call creates a short-lived `WhisperState`, so a single transcriber is
/// safe to share — calls serialize on the internal mutex.
pub struct WhisperCppTranscriber {
    ctx: WhisperContext,
    /// Inner serialization for `create_state` + `full` calls.
    lock: Mutex<()>,
    threads: i32,
    /// Forced language (`None` = auto-detect per utterance).
    language: Option<String>,
}

// `WhisperContext` has no `Debug` impl — provide one manually.
impl std::fmt::Debug for WhisperCppTranscriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WhisperCppTranscriber")
            .field("threads", &self.threads)
            .field("language", &self.language)
            .finish_non_exhaustive()
    }
}

impl WhisperCppTranscriber {
    /// Load a GGML/GGUF whisper model from `model_path`.
    ///
    /// * `threads`: decode thread count (`<= 0` → whisper.cpp default).
    /// * `language`: BCP-47-ish whisper language code (`"en"`, `"vi"`, …) or
    ///   `None` for per-call auto-detection.
    pub fn new(
        model_path: impl AsRef<Path>,
        threads: i32,
        language: Option<String>,
    ) -> Result<Self, AsrError> {
        let path_str = model_path
            .as_ref()
            .to_str()
            .ok_or_else(|| AsrError::msg("whisper.cpp model path is not valid UTF-8"))?;
        let ctx = WhisperContext::new_with_params(path_str, WhisperContextParameters::default())
            .map_err(|e| AsrError::msg(format!("whisper.cpp load: {e}")))?;
        Ok(Self {
            ctx,
            lock: Mutex::new(()),
            threads,
            language,
        })
    }

    /// Transcribe already-decoded 16 kHz mono f32 PCM.
    pub fn transcribe_pcm(&self, pcm: &[f32]) -> Result<TranscriptionResult, AsrError> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| AsrError::LockPoisoned("whispercpp"))?;

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| AsrError::msg(format!("whisper.cpp state: {e}")))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        if self.threads > 0 {
            params.set_n_threads(self.threads);
        }
        if let Some(lang) = &self.language {
            params.set_language(Some(lang));
        }
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_single_segment(false);

        state
            .full(params, pcm)
            .map_err(|e| AsrError::msg(format!("whisper.cpp decode: {e}")))?;

        let mut text = String::new();
        let mut segments = Vec::new();
        for seg in state.as_iter() {
            let seg_text = seg.to_str().unwrap_or_default().to_string();
            text.push_str(&seg_text);
            segments.push(TranscriptionSegment {
                // whisper timestamps are in 10 ms ticks.
                start_ms: (seg.start_timestamp() * 10).max(0) as u64,
                end_ms: (seg.end_timestamp() * 10).max(0) as u64,
                text: seg_text,
            });
        }

        Ok(TranscriptionResult {
            text,
            language: self.language.clone(),
            segments,
        })
    }
}

impl WhisperTranscriber for WhisperCppTranscriber {
    fn transcribe(
        &self,
        audio_data: &[u8],
        _mime_type: &str,
    ) -> Result<TranscriptionResult, AsrError> {
        let decoded = whisper_decode_wav(audio_data)?;
        let pcm = whisper_to_mono_16k(&decoded);
        self.transcribe_pcm(&pcm)
    }
}

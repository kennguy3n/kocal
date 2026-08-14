//! `OnnxWhisperTranscriber` — real Whisper encoder/decoder
//! inference loop behind the `onnx-runtime` cargo feature.
//!
//! The Whisper pipeline runs in three stages:
//!
//! 1. **Preprocessing** — bytes → 16 kHz mono PCM → log-mel
//!    `[80 × 3000]`. Implemented in [`crate::audio`] (pure CPU,
//!    no ORT, always compiled).
//! 2. **Encoder** — `encoder_model.onnx` consumes the log-mel
//!    grid as `input_features [1, 80, 3000]` and emits
//!    `last_hidden_state [1, 1500, d_model]` (audio time-axis
//!    is halved by Whisper's two stride-2 conv layers). The
//!    encoder runs once per audio buffer.
//! 3. **Decoder** — `decoder_model.onnx` consumes
//!    `(input_ids [1, prefix_len], encoder_hidden_states)` and
//!    emits `logits [1, prefix_len, vocab_size]`. We greedy-
//!    decode (argmax over the last position) one token at a
//!    time, appending each token to `input_ids` and re-running
//!    the decoder, until we hit `<|endoftext|>` or the
//!    `max_decode_tokens` ceiling.
//!
//! ## What lives in this module
//!
//! * Pure helpers — special-token resolver, decoder-prefix
//!   builder, argmax greedy-step, timestamp-token parser,
//!   token-stream → segment splitter, vocabulary-size sniffer.
//!   These are unit-tested on every host (no ORT required).
//! * `OnnxWhisperTranscriber` — the long-lived wrapper holding
//!   the encoder session, decoder session, [`tokenizers::Tokenizer`],
//!   and [`crate::audio::WhisperMelKernel`].
//!   Gated behind `feature = "onnx-runtime"`.
//! * Always-compiled stub `OnnxWhisperTranscriber` for builds
//!   without the feature so consumers can name the type
//!   unconditionally.
//!
//! ## Why no KV-cache
//!
//! Whisper's HuggingFace ONNX export ships both `decoder_model.onnx`
//! (full re-run per step) and `decoder_with_past_model.onnx`
//! (KV-cache). The KV-cache variant is faster (O(n) instead of
//! O(n²) for n decoded tokens) but the cache-tensor naming
//! convention (`past_key_values.0.decoder.key`, …) is fragile
//! across exports and Whisper transcripts are short (≤ 224
//! tokens per 30 s window). We use the no-KV-cache form for
//! correctness and forward-compatibility; the KV-cache path is
//! a future performance follow-up.

use crate::audio::{WHISPER_N_FRAMES, WHISPER_SAMPLE_RATE};
use crate::backend::TranscriptionSegment;

// Stub-only imports — only needed when the `onnx-runtime` feature
// is off, for the stub `OnnxWhisperTranscriber` at the bottom.
#[cfg(not(feature = "onnx-runtime"))]
use crate::backend::{TranscriptionResult, WhisperTranscriber};
#[cfg(not(feature = "onnx-runtime"))]
use crate::{AsrError, AsrResult};

// ---------------------------------------------------------------------------
// Whisper constants
// ---------------------------------------------------------------------------

/// Number of audio frames the encoder emits — half the
/// preprocessing frame count because Whisper's encoder front-end
/// has two stride-2 convolutions. The decoder consumes these as
/// `encoder_hidden_states[:, 1500, d_model]`.
pub const WHISPER_ENCODER_FRAMES: usize = WHISPER_N_FRAMES / 2;

/// Whisper's per-window decoder body-token cap — the maximum
/// number of *content* tokens the greedy loop will emit before
/// it is forced to stop. Matches OpenAI Whisper's reference
/// `whisper.decoding.DecodingTask.run` which clamps `sample_len`
/// to `n_ctx // 2 = 224` for transcription tasks.
pub const WHISPER_MAX_DECODE_TOKENS: usize = 224;

/// Whisper decoder's full positional-embedding pool — 448
/// positions total. Shared between the decoder prefix and the
/// greedy-loop body tokens. Used as the hard ceiling in the
/// greedy loop: `prefix.len() + emitted_body_tokens` MUST never
/// exceed this, regardless of the per-instance
/// `max_decode_tokens` override.
pub const WHISPER_DECODER_CONTEXT_TOKENS: usize = 448;

/// Token spacing (in seconds) of Whisper's timestamp tokens.
pub const WHISPER_TIMESTAMP_STEP_SECONDS: f32 = 0.02;

/// Token spacing in milliseconds (== 20 ms).
pub const WHISPER_TIMESTAMP_STEP_MS: u64 = 20;

/// Number of distinct Whisper timestamp tokens, inclusive on
/// both ends: `<|0.00|>` through `<|30.00|>` is `30 / 0.02 + 1
/// = 1501` ids.
pub const WHISPER_TIMESTAMP_TOKEN_COUNT: u32 = 1_501;

// ---------------------------------------------------------------------------
// Pure: WhisperTask
// ---------------------------------------------------------------------------

/// Whisper decoder task — either transcribe (source language)
/// or translate (always to English).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WhisperTask {
    /// Transcribe in source language. Default behaviour.
    #[default]
    Transcribe,
    /// Translate the source audio to English.
    Translate,
}

// ---------------------------------------------------------------------------
// Pure: WhisperSpecialTokens
// ---------------------------------------------------------------------------

/// Resolved special-token ids for one Whisper tokenizer.
#[derive(Debug, Clone)]
pub struct WhisperSpecialTokens {
    /// `<|endoftext|>` (the EOS token; same id as in GPT-2).
    pub end_of_text: u32,
    /// `<|startoftranscript|>` — first prefix token.
    pub start_of_transcript: u32,
    /// `<|transcribe|>`.
    pub transcribe: u32,
    /// `<|translate|>`.
    pub translate: u32,
    /// `<|notimestamps|>` — suppresses timestamp emission.
    pub no_timestamps: u32,
    /// `<|nospeech|>` (a.k.a. `<|nocaptions|>` on some
    /// exports) — emitted by the decoder for silence windows.
    /// `None` if the vocabulary doesn't expose it.
    pub no_speech: Option<u32>,
    /// `<|0.00|>` — first timestamp token. Subsequent timestamp
    /// tokens are at `timestamp_begin + i` for the `i`-th
    /// 20 ms step.
    pub timestamp_begin: u32,
    /// Mapping from BCP-47 / ISO-639-1 language code (`"en"`,
    /// `"zh"`, `"es"`, …) to the per-language token id.
    pub languages: std::collections::BTreeMap<String, u32>,
}

/// Whisper's full canonical language list — the 99 languages
/// `tokenize.py` declares plus `<|nospeech|>`.
pub const WHISPER_LANGUAGE_CODES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su",
];

impl WhisperSpecialTokens {
    /// Resolve every special token from the
    /// `(name, id)` map a tokenizer surfaces as its
    /// "added_tokens".
    pub fn resolve_from_added_tokens(
        added_tokens: &[(String, u32)],
    ) -> std::result::Result<Self, String> {
        let lookup = |needle: &str| {
            added_tokens
                .iter()
                .find(|(name, _)| name == needle)
                .map(|(_, id)| *id)
        };

        let end_of_text = lookup("<|endoftext|>")
            .ok_or_else(|| "Whisper vocab missing `<|endoftext|>`".to_string())?;
        let start_of_transcript = lookup("<|startoftranscript|>")
            .ok_or_else(|| "Whisper vocab missing `<|startoftranscript|>`".to_string())?;
        let transcribe = lookup("<|transcribe|>")
            .ok_or_else(|| "Whisper vocab missing `<|transcribe|>`".to_string())?;
        let translate = lookup("<|translate|>")
            .ok_or_else(|| "Whisper vocab missing `<|translate|>`".to_string())?;
        let no_timestamps = lookup("<|notimestamps|>")
            .ok_or_else(|| "Whisper vocab missing `<|notimestamps|>`".to_string())?;
        let timestamp_begin = lookup("<|0.00|>")
            .ok_or_else(|| "Whisper vocab missing timestamp anchor `<|0.00|>`".to_string())?;
        let no_speech = lookup("<|nospeech|>").or_else(|| lookup("<|nocaptions|>"));

        let mut languages = std::collections::BTreeMap::new();
        for code in WHISPER_LANGUAGE_CODES {
            let needle = format!("<|{code}|>");
            if let Some(id) = lookup(&needle) {
                languages.insert((*code).to_string(), id);
            }
        }
        if languages.is_empty() {
            return Err("Whisper vocab exposes no `<|lang|>` tokens".to_string());
        }

        Ok(Self {
            end_of_text,
            start_of_transcript,
            transcribe,
            translate,
            no_timestamps,
            no_speech,
            timestamp_begin,
            languages,
        })
    }

    /// Look up a language token id by ISO code. Returns `None`
    /// for codes the loaded vocab does not expose.
    pub fn language_token(&self, code: &str) -> Option<u32> {
        self.languages.get(code).copied()
    }
}

// ---------------------------------------------------------------------------
// Pure: decoder prefix builder
// ---------------------------------------------------------------------------

/// Build the initial decoder input-ids prefix for one
/// transcription pass.
///
/// Whisper was trained on prefixes shaped
/// `[SOT, <|lang|>, <|task|>, <|notimestamps|>?]`.
pub fn build_decoder_prefix(
    special: &WhisperSpecialTokens,
    language: Option<u32>,
    task: WhisperTask,
    with_timestamps: bool,
) -> Vec<u32> {
    let mut prefix = Vec::with_capacity(4);
    prefix.push(special.start_of_transcript);
    if let Some(lang) = language {
        prefix.push(lang);
    }
    let task_token = match task {
        WhisperTask::Transcribe => special.transcribe,
        WhisperTask::Translate => special.translate,
    };
    prefix.push(task_token);
    if !with_timestamps {
        prefix.push(special.no_timestamps);
    }
    prefix
}

// ---------------------------------------------------------------------------
// Pure: argmax greedy step
// ---------------------------------------------------------------------------

/// Argmax of the last position of a Whisper decoder logits
/// tensor, with optional suppression masks applied first.
///
/// `suppress` **MUST be sorted in ascending order and
/// deduplicated** — the membership test uses `binary_search`.
pub fn argmax_next_token(
    logits: &[f32],
    seq_len: usize,
    vocab_size: usize,
    suppress: &[u32],
) -> Option<u32> {
    if seq_len == 0 || vocab_size == 0 {
        return None;
    }
    let expected = seq_len.checked_mul(vocab_size)?;
    if logits.len() < expected {
        return None;
    }
    let start = (seq_len - 1).checked_mul(vocab_size)?;
    let row = &logits[start..start + vocab_size];

    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &v) in row.iter().enumerate() {
        if suppress.binary_search(&(idx as u32)).is_ok() {
            continue;
        }
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    if best_val == f32::NEG_INFINITY {
        return None;
    }
    Some(best_idx as u32)
}

/// Pure helper for Whisper-style language detection.
///
/// Runs a masked argmax over only the language token ids.
pub fn argmax_language_token(logits_row: &[f32], language_token_ids: &[u32]) -> Option<u32> {
    if logits_row.is_empty() || language_token_ids.is_empty() {
        return None;
    }
    let mut best_id: Option<u32> = None;
    let mut best_val = f32::NEG_INFINITY;
    for &id in language_token_ids {
        let idx = id as usize;
        if idx >= logits_row.len() {
            continue;
        }
        let v = logits_row[idx];
        if v > best_val {
            best_val = v;
            best_id = Some(id);
        }
    }
    best_id
}

// ---------------------------------------------------------------------------
// Pure: timestamp helpers + segment builder
// ---------------------------------------------------------------------------

/// Convert a timestamp token id to a millisecond offset.
pub fn timestamp_token_to_ms(token: u32, timestamp_begin: u32) -> Option<u64> {
    if token < timestamp_begin {
        return None;
    }
    let offset = token - timestamp_begin;
    if offset > 1_500 {
        return None;
    }
    Some(u64::from(offset) * WHISPER_TIMESTAMP_STEP_MS)
}

/// Split a decoded token stream into
/// [`TranscriptionSegment`]s using Whisper's paired-timestamp
/// convention.
pub fn segments_from_tokens<F>(
    tokens: &[u32],
    timestamp_begin: u32,
    end_of_text: u32,
    mut decode: F,
) -> Vec<TranscriptionSegment>
where
    F: FnMut(&[u32]) -> String,
{
    let mut segments = Vec::new();
    let mut current_start: Option<u64> = None;
    let mut body: Vec<u32> = Vec::new();

    for &tok in tokens {
        if tok == end_of_text {
            break;
        }
        if let Some(ms) = timestamp_token_to_ms(tok, timestamp_begin) {
            match current_start {
                None => {
                    if !body.is_empty() {
                        let text = decode(&body).trim().to_string();
                        if !text.is_empty() {
                            segments.push(TranscriptionSegment {
                                start_ms: 0,
                                end_ms: 0,
                                text,
                            });
                        }
                        body.clear();
                    }
                    current_start = Some(ms);
                }
                Some(start_ms) => {
                    let text = decode(&body).trim().to_string();
                    if !text.is_empty() {
                        segments.push(TranscriptionSegment {
                            start_ms,
                            end_ms: ms,
                            text,
                        });
                    }
                    body.clear();
                    current_start = None;
                }
            }
        } else {
            body.push(tok);
        }
    }

    if !body.is_empty() {
        let text = decode(&body).trim().to_string();
        if !text.is_empty() {
            let start_ms = current_start.unwrap_or(0);
            segments.push(TranscriptionSegment {
                start_ms,
                end_ms: start_ms,
                text,
            });
        }
    }

    segments
}

// ---------------------------------------------------------------------------
// Default artifact filenames
// ---------------------------------------------------------------------------

/// Default tokenizer artifact filename (HuggingFace convention).
pub const WHISPER_DEFAULT_TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Default decoder artifact filename.
pub const WHISPER_DEFAULT_DECODER_FILENAME: &str = "decoder_model.onnx";

/// Default encoder artifact filename.
pub const WHISPER_DEFAULT_ENCODER_FILENAME: &str = "encoder_model.onnx";

// ---------------------------------------------------------------------------
// `OnnxWhisperTranscriber` — feature-gated real wrapper
// ---------------------------------------------------------------------------

#[cfg(feature = "onnx-runtime")]
mod with_ort {
    use super::{
        build_decoder_prefix, segments_from_tokens, argmax_language_token, argmax_next_token,
        WhisperSpecialTokens, WhisperTask, WHISPER_DECODER_CONTEXT_TOKENS,
        WHISPER_DEFAULT_DECODER_FILENAME, WHISPER_DEFAULT_ENCODER_FILENAME,
        WHISPER_DEFAULT_TOKENIZER_FILENAME, WHISPER_ENCODER_FRAMES, WHISPER_MAX_DECODE_TOKENS,
        WHISPER_TIMESTAMP_TOKEN_COUNT,
    };
    use crate::audio::{
        whisper_log_mel_from_wav, WhisperMelKernel, WHISPER_N_FRAMES, WHISPER_N_MELS,
    };
    use crate::backend::{TranscriptionResult, TranscriptionSegment, WhisperTranscriber};
    use crate::{AsrError, AsrResult};
    use ort::session::Session;
    use ort::value::Tensor;
    use parking_lot::Mutex;
    use std::path::Path;

    // -----------------------------------------------------------------------
    // EP selection — mirrors kchat-encoder/src/session.rs and
    // kchat-safety/src/vision/ep_helpers.rs, using kchat_core::ep to pick
    // the platform-appropriate execution provider.
    // -----------------------------------------------------------------------

    /// Build the ort execution-provider dispatch list for the
    /// current host using [`kchat_core::ep`] selection.
    fn build_ort_eps_for_host() -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
        use kchat_core::ep::{EpDeviceCapabilities, EpFallbackChain, Platform};

        let (os, arch) = {
            #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
            {
                (Platform::MacOs, kchat_core::ep::Arch::Aarch64)
            }
            #[cfg(all(
                not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))),
                target_os = "macos"
            ))]
            {
                (Platform::MacOs, kchat_core::ep::Arch::X86_64)
            }
            #[cfg(target_os = "ios")]
            {
                (Platform::Ios, kchat_core::ep::Arch::Aarch64)
            }
            #[cfg(target_os = "android")]
            {
                (Platform::Android, kchat_core::ep::Arch::Aarch64)
            }
            #[cfg(target_os = "windows")]
            {
                (
                    Platform::Windows,
                    if cfg!(target_arch = "aarch64") {
                        kchat_core::ep::Arch::Aarch64
                    } else {
                        kchat_core::ep::Arch::X86_64
                    },
                )
            }
            #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "openbsd"))]
            {
                (
                    Platform::Linux,
                    if cfg!(target_arch = "aarch64") {
                        kchat_core::ep::Arch::Aarch64
                    } else {
                        kchat_core::ep::Arch::X86_64
                    },
                )
            }
            #[cfg(not(any(
                target_os = "macos",
                target_os = "ios",
                target_os = "android",
                target_os = "windows",
                target_os = "linux",
                target_os = "freebsd",
                target_os = "openbsd",
            )))]
            {
                (Platform::Unknown, kchat_core::ep::Arch::Other)
            }
        };

        let caps = match os {
            Platform::MacOs | Platform::Ios => EpDeviceCapabilities::apple_silicon_mac(),
            Platform::Android => EpDeviceCapabilities::android_with_npu(),
            Platform::Windows => EpDeviceCapabilities::windows_with_gpu("auto"),
            _ => EpDeviceCapabilities::cpu_only(os, arch),
        };

        let chain = EpFallbackChain::for_platform(os, &caps);
        chain
            .as_slice()
            .iter()
            .filter_map(|ep| ep_to_ort_dispatch(*ep))
            .collect()
    }

    fn ep_to_ort_dispatch(
        ep: kchat_core::ep::ExecutionProvider,
    ) -> Option<ort::execution_providers::ExecutionProviderDispatch> {
        use ort::execution_providers::{
            CPUExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
            NNAPIExecutionProvider,
        };

        match ep {
            kchat_core::ep::ExecutionProvider::CoreMl => {
                Some(CoreMLExecutionProvider::default().build())
            }
            kchat_core::ep::ExecutionProvider::Nnapi => {
                Some(NNAPIExecutionProvider::default().build())
            }
            kchat_core::ep::ExecutionProvider::DirectMl => {
                Some(DirectMLExecutionProvider::default().build())
            }
            kchat_core::ep::ExecutionProvider::MetalPerformanceShaders => None,
            kchat_core::ep::ExecutionProvider::Cpu => {
                Some(CPUExecutionProvider::default().build())
            }
        }
    }

    /// Create one ONNX session with EP selection and graph
    /// optimisation, routing any ORT error through
    /// [`AsrError::Ort`] with the call-site `op` label.
    fn create_whisper_session(model_path: &Path, op: &'static str) -> AsrResult<Session> {
        let mut builder = ort::session::Session::builder()
            .map_err(|e| AsrError::Ort { op, detail: format!("builder: {e}") })?;
        builder = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| AsrError::Ort { op, detail: format!("optimization: {e}") })?;
        builder = builder
            .with_intra_threads(2)
            .map_err(|e| AsrError::Ort { op, detail: format!("threads: {e}") })?;
        let ep_eps = build_ort_eps_for_host();
        if !ep_eps.is_empty() {
            builder = builder
                .with_execution_providers(&ep_eps)
                .map_err(|e| AsrError::Ort { op, detail: format!("ep selection: {e}") })?;
        }
        builder
            .commit_from_file(model_path)
            .map_err(|e| AsrError::Ort { op, detail: format!("load model: {e}") })
    }

    // -----------------------------------------------------------------------
    // OnnxWhisperTranscriber
    // -----------------------------------------------------------------------

    /// Long-lived ONNX Runtime wrapper for Whisper transcription.
    ///
    /// Holds the encoder and decoder sessions plus a single
    /// [`tokenizers::Tokenizer`] and [`WhisperMelKernel`].
    pub struct OnnxWhisperTranscriber {
        encoder: Mutex<Session>,
        decoder: Mutex<Session>,
        tokenizer: tokenizers::Tokenizer,
        mel_kernel: WhisperMelKernel,
        special: WhisperSpecialTokens,
        task: WhisperTask,
        language: Option<String>,
        with_timestamps: bool,
        max_decode_tokens: usize,
        vocab_size: usize,
        suppress: Vec<u32>,
        suppress_no_timestamps: Vec<u32>,
    }

    impl std::fmt::Debug for OnnxWhisperTranscriber {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("OnnxWhisperTranscriber")
                .field("task", &self.task)
                .field("language", &self.language)
                .field("with_timestamps", &self.with_timestamps)
                .field("max_decode_tokens", &self.max_decode_tokens)
                .field("vocab_size", &self.vocab_size)
                .finish_non_exhaustive()
        }
    }

    impl OnnxWhisperTranscriber {
        /// Build a Whisper transcriber from the canonical
        /// HuggingFace layout: `encoder_model.onnx`,
        /// `decoder_model.onnx`, `tokenizer.json` colocated in
        /// the same directory as `encoder_dir`.
        pub fn new(encoder_dir: &Path) -> AsrResult<Self> {
            let encoder_path = encoder_dir.join(WHISPER_DEFAULT_ENCODER_FILENAME);
            let decoder_path = encoder_dir.join(WHISPER_DEFAULT_DECODER_FILENAME);
            let tokenizer_path = encoder_dir.join(WHISPER_DEFAULT_TOKENIZER_FILENAME);
            Self::new_with_paths(&encoder_path, &decoder_path, &tokenizer_path)
        }

        /// Build a Whisper transcriber from explicit paths to
        /// the three artifacts.
        pub fn new_with_paths(
            encoder_path: &Path,
            decoder_path: &Path,
            tokenizer_path: &Path,
        ) -> AsrResult<Self> {
            let encoder = create_whisper_session(encoder_path, "whisper_encoder_session_create")?;
            let decoder = create_whisper_session(decoder_path, "whisper_decoder_session_create")?;

            let tokenizer = load_whisper_tokenizer(tokenizer_path)?;
            let special = resolve_special_tokens(&tokenizer)?;
            let vocab_size = tokenizer.get_vocab_size(true);
            let suppress = Self::compute_suppression_set(&special);
            let suppress_no_timestamps = Self::compute_suppression_set_no_timestamps(&special);

            Ok(Self {
                encoder: Mutex::new(encoder),
                decoder: Mutex::new(decoder),
                tokenizer,
                mel_kernel: WhisperMelKernel::new(),
                special,
                task: WhisperTask::Transcribe,
                language: None,
                with_timestamps: true,
                max_decode_tokens: WHISPER_MAX_DECODE_TOKENS,
                vocab_size,
                suppress,
                suppress_no_timestamps,
            })
        }

        /// Override the default decoder task.
        pub fn with_task(mut self, task: WhisperTask) -> Self {
            self.task = task;
            self
        }

        /// Pin the decoder prefix to a specific source language.
        pub fn with_language(mut self, language: Option<&str>) -> AsrResult<Self> {
            if let Some(code) = language {
                if self.special.language_token(code).is_none() {
                    return Err(AsrError::Tokenizer {
                        op: "whisper_with_language",
                        detail: format!("language `{code}` not exposed by this Whisper vocabulary"),
                    });
                }
                self.language = Some(code.to_string());
            } else {
                self.language = None;
            }
            Ok(self)
        }

        /// Enable / disable timestamp-token emission.
        pub fn with_timestamps(mut self, enabled: bool) -> Self {
            self.with_timestamps = enabled;
            self
        }

        /// Override the body-token cap for the greedy decode loop.
        pub fn with_max_decode_tokens(mut self, max: usize) -> Self {
            self.max_decode_tokens = max.max(1);
            self
        }

        /// Resolved special-token table for the loaded tokenizer.
        pub fn special_tokens(&self) -> &WhisperSpecialTokens {
            &self.special
        }

        /// Vocabulary cardinality.
        pub fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        /// Run the encoder over the log-mel grid.
        fn run_encoder(&self, mel: Vec<f32>) -> AsrResult<Vec<f32>> {
            debug_assert_eq!(mel.len(), WHISPER_N_MELS * WHISPER_N_FRAMES);
            let mel_tensor = Tensor::from_array((
                vec![1_i64, WHISPER_N_MELS as i64, WHISPER_N_FRAMES as i64],
                mel,
            ))
            .map_err(|e| AsrError::Ort {
                op: "whisper_encoder_input_tensor",
                detail: e.to_string(),
            })?;

            let mut encoder = self.encoder.lock();
            let outputs = encoder
                .run(ort::inputs!["input_features" => mel_tensor])
                .map_err(|e| AsrError::Ort {
                    op: "whisper_encoder_infer",
                    detail: e.to_string(),
                })?;
            let out = outputs
                .iter()
                .next()
                .ok_or_else(|| AsrError::Ort {
                    op: "whisper_encoder_no_output",
                    detail: "encoder run returned zero outputs".into(),
                })?
                .1;
            let (shape, data) = out
                .try_extract_tensor::<f32>()
                .map_err(|e| AsrError::Ort {
                    op: "whisper_encoder_output_extract",
                    detail: e.to_string(),
                })?;
            // `shape: &Shape` derefs to `&[i64]`; `data: &[f32]`.
            if shape.len() != 3
                || shape[0] != 1
                || shape[1] != WHISPER_ENCODER_FRAMES as i64
            {
                return Err(AsrError::Ort {
                    op: "whisper_encoder_output_shape",
                    detail: format!(
                        "encoder output expected `[1, {}, d_model]`, got `{:?}`",
                        WHISPER_ENCODER_FRAMES, shape
                    ),
                });
            }
            let d_model_dim = shape_inner_dim(shape)?;
            let expected_len = WHISPER_ENCODER_FRAMES
                .checked_mul(d_model_dim)
                .ok_or_else(|| AsrError::Ort {
                    op: "whisper_encoder_output_shape",
                    detail: format!("encoder output shape overflow: {shape:?}"),
                })?;
            if data.len() != expected_len {
                return Err(AsrError::Ort {
                    op: "whisper_encoder_output_shape",
                    detail: format!(
                        "encoder output length {} does not match expected {} from shape {:?}",
                        data.len(),
                        expected_len,
                        shape
                    ),
                });
            }
            Ok(data.to_vec())
        }

        /// Wrap the encoder hidden-state buffer into an ORT
        /// `Tensor<f32>` once per transcription.
        fn build_hidden_tensor(
            &self,
            encoder_hidden: Vec<f32>,
            encoder_d_model: usize,
        ) -> AsrResult<Tensor<f32>> {
            Tensor::from_array((
                vec![1_i64, WHISPER_ENCODER_FRAMES as i64, encoder_d_model as i64],
                encoder_hidden,
            ))
            .map_err(|e| AsrError::Ort {
                op: "whisper_hidden_tensor_build",
                detail: e.to_string(),
            })
        }

        /// Run the decoder once over the current prefix and a
        /// pre-built encoder hidden-state tensor; return ONLY
        /// the last-position row of the logits as a `Vec<f32>`
        /// of length `vocab_size`.
        fn run_decoder(&self, prefix: &[u32], hidden_tensor: &Tensor<f32>) -> AsrResult<Vec<f32>> {
            let input_ids: Vec<i64> = prefix.iter().map(|&t| i64::from(t)).collect();
            let prefix_len = input_ids.len();
            let ids_tensor = Tensor::from_array((vec![1_i64, prefix_len as i64], input_ids))
                .map_err(|e| AsrError::Ort {
                    op: "whisper_decoder_input_tensor",
                    detail: e.to_string(),
                })?;

            let mut decoder = self.decoder.lock();
            let outputs = decoder
                .run(ort::inputs![
                    "input_ids" => ids_tensor,
                    "encoder_hidden_states" => hidden_tensor,
                ])
                .map_err(|e| AsrError::Ort {
                    op: "whisper_decoder_infer",
                    detail: e.to_string(),
                })?;
            let out = outputs
                .iter()
                .next()
                .ok_or_else(|| AsrError::Ort {
                    op: "whisper_decoder_no_output",
                    detail: "decoder run returned zero outputs".into(),
                })?
                .1;
            let (shape, data) = out
                .try_extract_tensor::<f32>()
                .map_err(|e| AsrError::Ort {
                    op: "whisper_decoder_output_extract",
                    detail: e.to_string(),
                })?;
            if shape.len() != 3
                || shape[0] != 1
                || shape[1] != prefix_len as i64
                || shape[2] != self.vocab_size as i64
            {
                return Err(AsrError::Ort {
                    op: "whisper_decoder_output_shape",
                    detail: format!(
                        "decoder output expected `[1, {prefix_len}, {}]`, got `{:?}`",
                        self.vocab_size, shape
                    ),
                });
            }
            let total = prefix_len.checked_mul(self.vocab_size).ok_or_else(|| {
                AsrError::Ort {
                    op: "whisper_decoder_output_overflow",
                    detail: "prefix_len * vocab_size overflowed usize".into(),
                }
            })?;
            if data.len() != total {
                return Err(AsrError::Ort {
                    op: "whisper_decoder_output_shape",
                    detail: format!(
                        "decoder output length {} does not match prefix_len * vocab_size = {total} (shape {shape:?})",
                        data.len()
                    ),
                });
            }
            let last_row_start = (prefix_len - 1) * self.vocab_size;
            Ok(data[last_row_start..last_row_start + self.vocab_size].to_vec())
        }

        /// Whisper-style language detection.
        fn detect_language(&self, hidden_tensor: &Tensor<f32>) -> AsrResult<u32> {
            let probe_prefix = [self.special.start_of_transcript];
            let logits = self.run_decoder(&probe_prefix, hidden_tensor)?;
            let row: &[f32] = &logits;
            let language_ids: Vec<u32> = self.special.languages.values().copied().collect();
            argmax_language_token(row, &language_ids).ok_or_else(|| AsrError::Ort {
                op: "whisper_detect_language",
                detail: "decoder emitted no language-token logits during detection".into(),
            })
        }

        /// Build the token-id suppression list for greedy
        /// decoding: every special token EXCEPT timestamps and
        /// `<|endoftext|>`.
        pub(crate) fn compute_suppression_set(special: &WhisperSpecialTokens) -> Vec<u32> {
            let mut suppress = vec![
                special.start_of_transcript,
                special.transcribe,
                special.translate,
                special.no_timestamps,
            ];
            if let Some(ns) = special.no_speech {
                suppress.push(ns);
            }
            suppress.extend(special.languages.values().copied());
            suppress.sort_unstable();
            suppress.dedup();
            suppress
        }

        /// Variant of [`Self::compute_suppression_set`] used
        /// when `with_timestamps = false`.
        pub(crate) fn compute_suppression_set_no_timestamps(
            special: &WhisperSpecialTokens,
        ) -> Vec<u32> {
            let mut suppress = Self::compute_suppression_set(special);
            suppress.reserve(WHISPER_TIMESTAMP_TOKEN_COUNT as usize);
            for offset in 0..WHISPER_TIMESTAMP_TOKEN_COUNT {
                suppress.push(special.timestamp_begin + offset);
            }
            suppress.sort_unstable();
            suppress.dedup();
            suppress
        }

        /// Pick the suppression set the greedy decoder should
        /// use for the currently-configured `with_timestamps`
        /// flag.
        pub(crate) fn effective_suppression_set(&self) -> &[u32] {
            if self.with_timestamps {
                &self.suppress
            } else {
                &self.suppress_no_timestamps
            }
        }

        /// Expose the active suppression set for debugging.
        pub fn suppression_set(&self) -> &[u32] {
            self.effective_suppression_set()
        }
    }

    impl WhisperTranscriber for OnnxWhisperTranscriber {
        fn transcribe(&self, audio_data: &[u8], mime_type: &str) -> AsrResult<TranscriptionResult> {
            if !mime_type.starts_with("audio/") {
                return Err(AsrError::AudioDecode {
                    op: "whisper_transcribe",
                    detail: format!(
                        "OnnxWhisperTranscriber rejects non-audio mime_type: {mime_type}"
                    ),
                });
            }

            // 1. Preprocessing: bytes → [80, 3000] log-mel.
            let mel = whisper_log_mel_from_wav(audio_data, &self.mel_kernel)?;

            // 2. Encoder: [1, 80, 3000] → [1, 1500, d_model].
            let encoder_hidden = self.run_encoder(mel)?;
            debug_assert!(
                !encoder_hidden.is_empty()
                    && encoder_hidden.len() % WHISPER_ENCODER_FRAMES == 0,
                "run_encoder must guarantee `encoder_hidden.len() == WHISPER_ENCODER_FRAMES * d_model_dim`; got len = {}",
                encoder_hidden.len()
            );
            let encoder_d_model = encoder_hidden.len() / WHISPER_ENCODER_FRAMES;

            // 3. Wrap the encoder hidden state in an ORT tensor ONCE.
            let hidden_tensor = self.build_hidden_tensor(encoder_hidden, encoder_d_model)?;

            // 4. Resolve the language token.
            let (language_token, detected_language) = if let Some(code) = self.language.as_deref() {
                let id = self.special.language_token(code).ok_or_else(|| {
                    AsrError::Tokenizer {
                        op: "whisper_transcribe_language",
                        detail: format!(
                            "pinned language `{code}` not exposed by the loaded Whisper vocab"
                        ),
                    }
                })?;
                (id, Some(code.to_string()))
            } else {
                let id = self.detect_language(&hidden_tensor)?;
                let code = self
                    .special
                    .languages
                    .iter()
                    .find(|(_, &v)| v == id)
                    .map(|(code, _)| code.clone());
                (id, code)
            };

            // 5. Build the real decode prefix.
            let mut prefix = build_decoder_prefix(
                &self.special,
                Some(language_token),
                self.task,
                self.with_timestamps,
            );
            let prefix_initial_len = prefix.len();

            // 6. Greedy decode loop.
            let context_budget =
                WHISPER_DECODER_CONTEXT_TOKENS.saturating_sub(prefix_initial_len);
            let body_budget = self.max_decode_tokens.min(context_budget);
            let mut emitted: Vec<u32> = Vec::new();
            for _ in 0..body_budget {
                let logits = self.run_decoder(&prefix, &hidden_tensor)?;
                let next = argmax_next_token(
                    &logits,
                    1,
                    self.vocab_size,
                    self.effective_suppression_set(),
                )
                .ok_or_else(|| AsrError::Ort {
                    op: "whisper_decoder_argmax",
                    detail: "every vocabulary position was suppressed; refusing to advance".into(),
                })?;
                if next == self.special.end_of_text {
                    break;
                }
                emitted.push(next);
                prefix.push(next);
            }
            debug_assert!(
                prefix.len() <= WHISPER_DECODER_CONTEXT_TOKENS,
                "greedy loop overran decoder context: prefix.len() = {}, ceiling = {WHISPER_DECODER_CONTEXT_TOKENS}",
                prefix.len()
            );

            // 7. Decode token stream → text + segments.
            let tokenizer = &self.tokenizer;
            let decode = |body: &[u32]| -> String {
                tokenizer
                    .decode(body, true)
                    .inspect_err(|e| tracing::warn!("tokenizer decode failed: {e}"))
                    .unwrap_or_default()
            };
            let mut segments = segments_from_tokens(
                &emitted,
                self.special.timestamp_begin,
                self.special.end_of_text,
                decode,
            );
            if segments.is_empty() && !emitted.is_empty() {
                let text = tokenizer
                    .decode(&emitted, true)
                    .inspect_err(|e| tracing::warn!("tokenizer decode failed: {e}"))
                    .unwrap_or_default();
                let text = text.trim().to_string();
                if !text.is_empty() {
                    segments.push(TranscriptionSegment {
                        start_ms: 0,
                        end_ms: 0,
                        text,
                    });
                }
            }
            let text = segments
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let text = text.trim().to_string();

            let _ = prefix_initial_len;

            Ok(TranscriptionResult {
                text,
                language: detected_language,
                segments,
            })
        }
    }

    /// Load a HuggingFace tokenizer from disk.
    fn load_whisper_tokenizer(path: &Path) -> AsrResult<tokenizers::Tokenizer> {
        tokenizers::Tokenizer::from_file(path).map_err(|e| AsrError::Tokenizer {
            op: "whisper_tokenizer_load",
            detail: e.to_string(),
        })
    }

    /// Resolve [`WhisperSpecialTokens`] from a loaded tokenizer's
    /// added-token table.
    fn resolve_special_tokens(tokenizer: &tokenizers::Tokenizer) -> AsrResult<WhisperSpecialTokens> {
        let added: Vec<(String, u32)> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .map(|(id, tok)| (tok.content, id))
            .collect();
        WhisperSpecialTokens::resolve_from_added_tokens(&added).map_err(|detail| {
            AsrError::Tokenizer {
                op: "whisper_special_tokens",
                detail,
            }
        })
    }

    /// Pluck the inner-most dimension out of an ORT shape, used
    /// to extract `d_model` from the encoder's
    /// `[1, 1500, d_model]` output.
    fn shape_inner_dim(shape: &[i64]) -> AsrResult<usize> {
        let last = shape.last().copied().ok_or_else(|| AsrError::Ort {
            op: "whisper_encoder_output_shape",
            detail: "encoder output tensor has no dimensions".into(),
        })?;
        if last <= 0 {
            return Err(AsrError::Ort {
                op: "whisper_encoder_output_shape",
                detail: format!(
                    "encoder output last dim is non-positive ({last}); dynamic dims unsupported"
                ),
            });
        }
        Ok(last as usize)
    }

    // Compile-time sanity that WHISPER_SAMPLE_RATE is reachable.
    use crate::audio::WHISPER_SAMPLE_RATE as _WHISPER_SAMPLE_RATE;
    const _: u32 = _WHISPER_SAMPLE_RATE;
}

#[cfg(feature = "onnx-runtime")]
pub use with_ort::OnnxWhisperTranscriber;

// ---------------------------------------------------------------------------
// Stub for builds without the `onnx-runtime` feature
// ---------------------------------------------------------------------------

/// Always-`Err` `OnnxWhisperTranscriber` stub for builds without
/// the `onnx-runtime` cargo feature, so consumer crates can name
/// the type unconditionally.
#[cfg(not(feature = "onnx-runtime"))]
#[derive(Debug, Default, Clone, Copy)]
pub struct OnnxWhisperTranscriber;

#[cfg(not(feature = "onnx-runtime"))]
impl OnnxWhisperTranscriber {
    /// Always returns [`AsrError::Custom`].
    pub fn new(_encoder_dir: &std::path::Path) -> AsrResult<Self> {
        Err(AsrError::msg(
            "OnnxWhisperTranscriber::new (onnx-runtime feature disabled)",
        ))
    }

    /// Always returns [`AsrError::Custom`].
    pub fn new_with_paths(
        _encoder: &std::path::Path,
        _decoder: &std::path::Path,
        _tokenizer: &std::path::Path,
    ) -> AsrResult<Self> {
        Err(AsrError::msg(
            "OnnxWhisperTranscriber::new_with_paths (onnx-runtime feature disabled)",
        ))
    }
}

#[cfg(not(feature = "onnx-runtime"))]
impl WhisperTranscriber for OnnxWhisperTranscriber {
    fn transcribe(&self, _audio_data: &[u8], _mime_type: &str) -> AsrResult<TranscriptionResult> {
        Err(AsrError::msg(
            "OnnxWhisperTranscriber::transcribe (onnx-runtime feature disabled)",
        ))
    }
}

// Compile-time sanity that WHISPER_SAMPLE_RATE is reachable.
const _: u32 = WHISPER_SAMPLE_RATE;

// ---------------------------------------------------------------------------
// Tests — all pure-helper logic exercised on every host.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_added_tokens(timestamp_begin: u32) -> Vec<(String, u32)> {
        let mut t = vec![
            ("<|endoftext|>".to_string(), 50_256),
            ("<|startoftranscript|>".to_string(), 50_257),
            ("<|en|>".to_string(), 50_258),
            ("<|zh|>".to_string(), 50_259),
            ("<|de|>".to_string(), 50_260),
            ("<|es|>".to_string(), 50_261),
            ("<|translate|>".to_string(), 50_357),
            ("<|transcribe|>".to_string(), 50_358),
            ("<|notimestamps|>".to_string(), 50_362),
            ("<|nospeech|>".to_string(), 50_361),
        ];
        for i in 0..=1_500_u32 {
            t.push((format!("<|{:.2}|>", i as f32 * 0.02), timestamp_begin + i));
        }
        t
    }

    #[test]
    fn special_tokens_resolve_round_trip() {
        let timestamp_begin = 50_363;
        let added = synthetic_added_tokens(timestamp_begin);
        let resolved = WhisperSpecialTokens::resolve_from_added_tokens(&added)
            .expect("resolve must succeed for a complete vocab");
        assert_eq!(resolved.end_of_text, 50_256);
        assert_eq!(resolved.start_of_transcript, 50_257);
        assert_eq!(resolved.transcribe, 50_358);
        assert_eq!(resolved.translate, 50_357);
        assert_eq!(resolved.no_timestamps, 50_362);
        assert_eq!(resolved.no_speech, Some(50_361));
        assert_eq!(resolved.timestamp_begin, timestamp_begin);
        assert_eq!(resolved.languages.len(), 4);
        assert_eq!(resolved.language_token("en"), Some(50_258));
        assert_eq!(resolved.language_token("zh"), Some(50_259));
        assert_eq!(resolved.language_token("zz"), None);
    }

    #[test]
    fn special_tokens_resolve_accepts_nocaptions_alias() {
        let timestamp_begin = 50_363;
        let mut added = synthetic_added_tokens(timestamp_begin);
        added.retain(|(name, _)| name != "<|nospeech|>");
        added.push(("<|nocaptions|>".to_string(), 50_361));
        let resolved = WhisperSpecialTokens::resolve_from_added_tokens(&added).unwrap();
        assert_eq!(resolved.no_speech, Some(50_361));
    }

    #[test]
    fn special_tokens_resolve_tolerates_missing_nospeech() {
        let timestamp_begin = 50_363;
        let mut added = synthetic_added_tokens(timestamp_begin);
        added.retain(|(name, _)| name != "<|nospeech|>");
        let resolved = WhisperSpecialTokens::resolve_from_added_tokens(&added).unwrap();
        assert_eq!(resolved.no_speech, None);
    }

    #[test]
    fn special_tokens_resolve_rejects_missing_required_token() {
        let timestamp_begin = 50_363;
        let mut added = synthetic_added_tokens(timestamp_begin);
        added.retain(|(name, _)| name != "<|startoftranscript|>");
        let err = WhisperSpecialTokens::resolve_from_added_tokens(&added).unwrap_err();
        assert!(err.contains("startoftranscript"), "unexpected error: {err}");
    }

    #[test]
    fn special_tokens_resolve_rejects_empty_language_set() {
        let added = vec![
            ("<|endoftext|>".to_string(), 50_256),
            ("<|startoftranscript|>".to_string(), 50_257),
            ("<|transcribe|>".to_string(), 50_358),
            ("<|translate|>".to_string(), 50_357),
            ("<|notimestamps|>".to_string(), 50_362),
            ("<|0.00|>".to_string(), 50_363),
        ];
        let err = WhisperSpecialTokens::resolve_from_added_tokens(&added).unwrap_err();
        assert!(err.contains("`<|lang|>`"), "unexpected error: {err}");
    }

    #[test]
    fn decoder_prefix_with_language_transcribe_no_timestamps() {
        let timestamp_begin = 50_363;
        let added = synthetic_added_tokens(timestamp_begin);
        let s = WhisperSpecialTokens::resolve_from_added_tokens(&added).unwrap();
        let prefix =
            build_decoder_prefix(&s, s.language_token("en"), WhisperTask::Transcribe, false);
        assert_eq!(
            prefix,
            vec![s.start_of_transcript, 50_258, s.transcribe, s.no_timestamps]
        );
    }

    #[test]
    fn decoder_prefix_translate_with_timestamps() {
        let timestamp_begin = 50_363;
        let added = synthetic_added_tokens(timestamp_begin);
        let s = WhisperSpecialTokens::resolve_from_added_tokens(&added).unwrap();
        let prefix = build_decoder_prefix(&s, s.language_token("zh"), WhisperTask::Translate, true);
        assert_eq!(prefix, vec![s.start_of_transcript, 50_259, s.translate]);
    }

    #[test]
    fn decoder_prefix_omits_language_slot_when_none() {
        let timestamp_begin = 50_363;
        let added = synthetic_added_tokens(timestamp_begin);
        let s = WhisperSpecialTokens::resolve_from_added_tokens(&added).unwrap();
        let prefix = build_decoder_prefix(&s, None, WhisperTask::Transcribe, false);
        assert_eq!(
            prefix,
            vec![s.start_of_transcript, s.transcribe, s.no_timestamps]
        );
    }

    #[test]
    fn argmax_picks_max_logit_at_last_position() {
        let mut logits = vec![0.0_f32; 2 * 5];
        logits[5 + 3] = 7.0;
        let pick = argmax_next_token(&logits, 2, 5, &[]).unwrap();
        assert_eq!(pick, 3);
    }

    #[test]
    fn argmax_ignores_suppressed_positions() {
        let mut logits = vec![0.0_f32; 5];
        logits[1] = 3.0;
        logits[4] = 9.9;
        let pick = argmax_next_token(&logits, 1, 5, &[4]).unwrap();
        assert_eq!(pick, 1);
    }

    #[test]
    fn argmax_returns_none_when_all_suppressed() {
        let logits = vec![0.0_f32; 5];
        let pick = argmax_next_token(&logits, 1, 5, &[0, 1, 2, 3, 4]);
        assert_eq!(pick, None);
    }

    #[test]
    fn argmax_rejects_short_logits_buffer() {
        let logits = vec![0.0_f32; 8];
        let pick = argmax_next_token(&logits, 3, 4, &[]);
        assert_eq!(pick, None);
    }

    #[test]
    fn max_decode_tokens_matches_whisper_reference_sample_len() {
        assert_eq!(WHISPER_MAX_DECODE_TOKENS, 224);
        assert_eq!(WHISPER_DECODER_CONTEXT_TOKENS, 448);
        let headroom = WHISPER_DECODER_CONTEXT_TOKENS.saturating_sub(WHISPER_MAX_DECODE_TOKENS);
        assert!(
            headroom >= 4,
            "body cap {} + longest prefix 4 must fit decoder context {} (headroom = {})",
            WHISPER_MAX_DECODE_TOKENS,
            WHISPER_DECODER_CONTEXT_TOKENS,
            headroom,
        );
    }

    #[test]
    fn argmax_binary_search_pins_sorted_multi_id_suppression() {
        let mut logits = vec![0.0_f32; 12];
        logits[4] = 12.0;
        logits[7] = 9.5;
        logits[10] = 11.0;
        logits[3] = 1.5;
        let suppress: Vec<u32> = vec![0, 2, 4, 6, 8, 10];
        let pick = argmax_next_token(&logits, 1, 12, &suppress).unwrap();
        assert_eq!(pick, 7);
    }

    #[test]
    fn argmax_language_token_picks_highest_scoring_language() {
        let mut row = vec![0.0_f32; 12];
        row[0] = 9.9;
        row[3] = 2.0;
        row[5] = 8.5;
        row[7] = 5.0;
        row[10] = 1.0;
        let pick = argmax_language_token(&row, &[3, 7, 10]).unwrap();
        assert_eq!(pick, 7);
    }

    #[test]
    fn argmax_language_token_returns_none_for_empty_inputs() {
        assert_eq!(argmax_language_token(&[], &[3, 7]), None);
        assert_eq!(argmax_language_token(&[1.0, 2.0, 3.0], &[]), None);
    }

    #[test]
    fn argmax_language_token_skips_ids_past_logits_row() {
        let row = vec![0.0_f32, 1.0, 2.0, 7.0, 3.0];
        let pick = argmax_language_token(&row, &[3, 100]).unwrap();
        assert_eq!(pick, 3);
    }

    #[test]
    fn argmax_language_token_returns_none_when_every_id_out_of_range() {
        let row = vec![0.0_f32; 5];
        assert_eq!(argmax_language_token(&row, &[10, 20, 30]), None);
    }

    #[test]
    fn timestamp_token_to_ms_rejects_below_anchor() {
        assert_eq!(timestamp_token_to_ms(50_360, 50_363), None);
    }

    #[test]
    fn timestamp_token_to_ms_rejects_above_max_window() {
        assert_eq!(timestamp_token_to_ms(50_363 + 1_501, 50_363), None);
    }

    #[test]
    fn timestamp_token_to_ms_returns_milliseconds() {
        assert_eq!(timestamp_token_to_ms(50_363, 50_363), Some(0));
        assert_eq!(timestamp_token_to_ms(50_363 + 1, 50_363), Some(20));
        assert_eq!(timestamp_token_to_ms(50_363 + 50, 50_363), Some(1_000));
        assert_eq!(timestamp_token_to_ms(50_363 + 1_500, 50_363), Some(30_000));
    }

    #[test]
    fn segments_from_tokens_pairs_timestamps_into_segments() {
        let timestamp_begin: u32 = 50_363;
        let end_of_text: u32 = 50_256;
        let stream = vec![
            timestamp_begin,
            1_001,
            1_002,
            timestamp_begin + 50,
            timestamp_begin + 50,
            2_001,
            2_002,
            timestamp_begin + 100,
            end_of_text,
        ];
        let segments = segments_from_tokens(&stream, timestamp_begin, end_of_text, |body| {
            body.iter()
                .map(|t| format!("t{t}"))
                .collect::<Vec<_>>()
                .join(" ")
        });
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 1_000);
        assert_eq!(segments[0].text, "t1001 t1002");
        assert_eq!(segments[1].start_ms, 1_000);
        assert_eq!(segments[1].end_ms, 2_000);
        assert_eq!(segments[1].text, "t2001 t2002");
    }

    #[test]
    fn segments_from_tokens_flushes_unclosed_tail() {
        let timestamp_begin: u32 = 50_363;
        let end_of_text: u32 = 50_256;
        let stream = vec![timestamp_begin, 1_001, 1_002, end_of_text];
        let segments = segments_from_tokens(&stream, timestamp_begin, end_of_text, |body| {
            body.iter()
                .map(|t| format!("t{t}"))
                .collect::<Vec<_>>()
                .join(" ")
        });
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start_ms, 0);
        assert_eq!(segments[0].end_ms, 0);
        assert_eq!(segments[0].text, "t1001 t1002");
    }

    #[test]
    fn segments_from_tokens_returns_empty_for_empty_stream() {
        let segments = segments_from_tokens(&[], 50_363, 50_256, |_| String::new());
        assert!(segments.is_empty());
    }

    #[test]
    fn segments_from_tokens_skips_empty_body_segments() {
        let timestamp_begin: u32 = 50_363;
        let end_of_text: u32 = 50_256;
        let stream = vec![timestamp_begin, timestamp_begin + 10, end_of_text];
        let segments = segments_from_tokens(&stream, timestamp_begin, end_of_text, |_| {
            String::new()
        });
        assert!(segments.is_empty());
    }

    #[test]
    fn whisper_constants_match_audio_module() {
        assert_eq!(WHISPER_ENCODER_FRAMES, WHISPER_N_FRAMES / 2);
        assert_eq!(WHISPER_TIMESTAMP_STEP_MS, 20);
        assert_eq!(WHISPER_LANGUAGE_CODES[0], "en");
        let mut sorted: Vec<&str> = WHISPER_LANGUAGE_CODES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            WHISPER_LANGUAGE_CODES.len(),
            "WHISPER_LANGUAGE_CODES contains duplicates"
        );
        assert_eq!(
            WHISPER_LANGUAGE_CODES.len(),
            99,
            "Whisper supports 99 languages"
        );
    }

    // ---- ONNX-runtime-only suppression-set tests ----

    #[cfg(feature = "onnx-runtime")]
    #[test]
    fn compute_suppression_set_no_timestamps_adds_full_timestamp_range() {
        let timestamp_begin = 50_363;
        let added = synthetic_added_tokens(timestamp_begin);
        let s = WhisperSpecialTokens::resolve_from_added_tokens(&added).unwrap();
        let base = OnnxWhisperTranscriber::compute_suppression_set(&s);
        let extended = OnnxWhisperTranscriber::compute_suppression_set_no_timestamps(&s);

        let mut sorted_check = extended.clone();
        sorted_check.sort_unstable();
        sorted_check.dedup();
        assert_eq!(
            extended, sorted_check,
            "extended suppression set must be sorted and deduped"
        );

        for id in &base {
            assert!(
                extended.binary_search(id).is_ok(),
                "extended set missing base id {id}"
            );
        }

        for offset in 0..WHISPER_TIMESTAMP_TOKEN_COUNT {
            let id = timestamp_begin + offset;
            assert!(
                extended.binary_search(&id).is_ok(),
                "extended set missing timestamp id {id} (offset {offset})"
            );
        }

        let beyond = timestamp_begin + WHISPER_TIMESTAMP_TOKEN_COUNT;
        assert!(
            extended.binary_search(&beyond).is_err(),
            "extended set should not include {beyond} (one past the inclusive range)"
        );

        assert!(
            extended.binary_search(&s.end_of_text).is_err(),
            "extended set must NOT suppress end_of_text"
        );
    }

    #[cfg(feature = "onnx-runtime")]
    #[test]
    fn compute_suppression_set_excludes_timestamp_tokens() {
        let timestamp_begin = 50_363;
        let added = synthetic_added_tokens(timestamp_begin);
        let s = WhisperSpecialTokens::resolve_from_added_tokens(&added).unwrap();
        let base = OnnxWhisperTranscriber::compute_suppression_set(&s);
        for offset in 0..WHISPER_TIMESTAMP_TOKEN_COUNT {
            let id = timestamp_begin + offset;
            assert!(
                base.binary_search(&id).is_err(),
                "base set must not suppress timestamp id {id}"
            );
        }
    }

    // ---- Stub-only tests (feature off) ----

    #[cfg(not(feature = "onnx-runtime"))]
    #[test]
    fn stub_new_reports_feature_gate() {
        let err =
            OnnxWhisperTranscriber::new(&std::path::PathBuf::from("/nonexistent")).unwrap_err();
        assert!(matches!(err, AsrError::Custom(_)));
    }

    #[cfg(not(feature = "onnx-runtime"))]
    #[test]
    fn stub_transcribe_reports_feature_gate() {
        let stub = OnnxWhisperTranscriber;
        let err = stub.transcribe(b"audio", "audio/wav").unwrap_err();
        assert!(matches!(err, AsrError::Custom(_)));
    }
}

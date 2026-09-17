//! llama.cpp backend — real in-process inference via `llama-cpp-2`.
//!
//! This backend loads GGUF models and runs grammar-constrained generation
//! with Metal (macOS/iOS), Vulkan (Android/Windows), CUDA (Linux), or CPU
//! acceleration. It supports:
//! - Streaming generation with cancellation
//! - JSON Schema grammar constraints (via `json_schema_to_grammar`)
//! - LoRA adapter hot-swap (see [`crate::lora`])
//! - Embeddings (for the context plane's fallback embedder)
//!
//! The backend is gated behind the `llamacpp` feature flag. Platform-specific
//! GPU acceleration is selected by the `llamacpp-metal`, `llamacpp-vulkan`,
//! or `llamacpp-cuda` feature flags.

use crate::backend::{
    BackendAdapter, BackendConfig, BackendError, BackendType, GenerationConfig, GenerationResult,
};
#[cfg(any(
    feature = "llamacpp-metal",
    feature = "llamacpp-vulkan",
    feature = "llamacpp-cuda"
))]
use crate::gbnf::regex_to_gbnf;
use crate::grammar::{GrammarType, GrammarValidator};
use crate::stream::StreamHandle;
#[cfg(any(
    feature = "llamacpp-metal",
    feature = "llamacpp-vulkan",
    feature = "llamacpp-cuda"
))]
use llama_cpp_2::context::params::LlamaContextType;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::{BatchAddError, LlamaBatch};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::{LlamaLoraAdapter, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
#[cfg(any(
    feature = "llamacpp-metal",
    feature = "llamacpp-vulkan",
    feature = "llamacpp-cuda"
))]
use llama_cpp_2::speculative::{MtpSpeculative, MtpSpeculativeError, MtpSpeculativeParams};
use llama_cpp_2::token::LlamaToken;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::num::NonZeroU32;
use std::path::Path;
use std::pin::Pin;
use std::time::Instant;

/// Global singleton for the llama.cpp backend (can only be initialized once).
/// Stored as a `Result` so an init failure surfaces as an error instead of
/// panicking across the FFI boundary.
static LLAMA_BACKEND: OnceCell<Result<LlamaBackend, String>> = OnceCell::new();

/// Get the global llama.cpp backend instance, initializing it on first call.
fn llama_backend() -> Result<&'static LlamaBackend, BackendError> {
    LLAMA_BACKEND
        .get_or_init(|| {
            LlamaBackend::init().map_err(|e| format!("failed to init llama.cpp backend: {e}"))
        })
        .as_ref()
        .map_err(|e| BackendError::LoadFailed(e.clone()))
}

/// A loaded LoRA adapter plus its scale.
///
/// `LlamaLoraAdapter` wraps a `NonNull` pointer and is therefore not
/// `Send`/`Sync` by default. The underlying `llama_adapter_lora` object is
/// immutable after `llama_adapter_lora_init` — adapters are only *read* when
/// attached to a context via `llama_set_adapters_lora` — so it is safe to
/// share across threads here.
struct LoraSlot {
    adapter: LlamaLoraAdapter,
    scale: f32,
}

unsafe impl Send for LoraSlot {}
unsafe impl Sync for LoraSlot {}

/// A persistent llama.cpp context whose KV cache is reused across calls.
///
/// `cached_tokens` tracks the token sequence currently resident in seq 0's
/// KV cache (prompt + generated). On each call, the shared prefix between
/// `cached_tokens` and the new prompt is kept and the tail is removed via
/// `kv_cache_seq_rm`, so consecutive prompts that share a prefix (system
/// prompt, chat history, skill templates) skip re-prefill entirely.
struct PersistentSession {
    /// The long-lived decode context. The `'static` lifetime is obtained via
    /// `transmute` — see the safety note on [`LlamaCppBackend::session`].
    ctx: LlamaContext<'static>,
    /// Context parameters the session was created with — recreate on change.
    ctx_key: CtxKey,
    /// Tokens currently resident in seq 0's KV cache.
    cached_tokens: Vec<llama_cpp_2::token::LlamaToken>,
}

/// Identity of a persistent context — if any of these change on `load()`,
/// the session must be recreated.
#[derive(Clone, Copy, PartialEq, Eq)]
struct CtxKey {
    context_size: usize,
    batch_size: u32,
    threads: u32,
}

// SAFETY: `PersistentSession.ctx` internally holds `&'static LlamaModel`
// pointing into `LlamaCppBackend::model`, which stores the model behind a
// `Box` (stable heap address — moving the backend does not move the model).
// The session is always dropped BEFORE the model in `load()`/`unload()`, so
// the reference can never dangle. The context is only accessed while holding
// the `session` Mutex, so it is never used concurrently — `Send`/`Sync` are
// sound here.
unsafe impl Send for PersistentSession {}
unsafe impl Sync for PersistentSession {}

/// Real llama.cpp backend using the `llama-cpp-2` crate.
///
/// The backend holds a single loaded model plus a persistent decode context
/// with KV-cache prefix reuse across generation calls.
pub struct LlamaCppBackend {
    /// The loaded model, if any. Boxed so the model's address stays stable
    /// while a `PersistentSession` holds a reference into it.
    model: Mutex<Option<Box<LlamaModel>>>,
    /// The current backend configuration (set on load).
    config: Mutex<Option<BackendConfig>>,
    /// Currently active LoRA adapter, applied to each new context.
    lora: Mutex<Option<LoraSlot>>,
    /// Persistent decode context + KV prefix state. MUST be dropped before
    /// `model` (it references the boxed model).
    session: Mutex<Option<PersistentSession>>,
}

impl LlamaCppBackend {
    /// Create a new llama.cpp backend. The llama.cpp global backend is
    /// initialized lazily on first use via a static `OnceCell`.
    pub fn new() -> Self {
        // Touch the singleton to initialize it eagerly; failures surface
        // later at load/context-creation time as `BackendError`s.
        let _ = llama_backend();
        Self {
            model: Mutex::new(None),
            config: Mutex::new(None),
            lora: Mutex::new(None),
            session: Mutex::new(None),
        }
    }

    /// Load a LoRA adapter file and apply it to subsequent generations.
    ///
    /// The adapter is initialised against the currently-loaded model and
    /// attached to every new context via `llama_set_adapters_lora`. Passing a
    /// `.gguf` LoRA file (llama.cpp format) is required — safetensors adapters
    /// are MLX-only.
    pub fn set_lora(&self, adapter_path: &str, scale: f32) -> Result<(), BackendError> {
        if !Path::new(adapter_path).is_file() {
            return Err(BackendError::LoadFailed(format!(
                "lora adapter not found: {adapter_path}"
            )));
        }
        let mut adapter = self
            .with_model(|model| model.lora_adapter_init(adapter_path))?
            .map_err(|e| BackendError::LoadFailed(format!("lora init: {e}")))?;

        // Apply to the live session if one exists, and invalidate its KV —
        // cached tokens were produced under different weights.
        if let Some(sess) = self.session.lock().as_mut() {
            sess.ctx
                .lora_adapter_set(&mut adapter, scale)
                .map_err(|e| BackendError::LoadFailed(format!("lora attach to session: {e}")))?;
            sess.ctx.clear_kv_cache();
            sess.cached_tokens.clear();
        }

        *self.lora.lock() = Some(LoraSlot { adapter, scale });
        tracing::info!("LoRA adapter loaded: {} (scale={})", adapter_path, scale);
        Ok(())
    }

    /// Detach the current LoRA adapter — subsequent generations run the base model.
    pub fn clear_lora(&self) {
        // Lock order is session→lora everywhere (see ensure_session) — taking
        // them in the opposite order here would risk deadlock.
        let mut session_guard = self.session.lock();
        let mut slot = self.lora.lock().take();
        // Remove from the live session and invalidate its KV.
        if let (Some(sess), Some(slot)) = (session_guard.as_mut(), slot.as_mut()) {
            if let Err(e) = sess.ctx.lora_adapter_remove(&mut slot.adapter) {
                tracing::warn!("lora detach on session failed: {e}");
            }
            sess.ctx.clear_kv_cache();
            sess.cached_tokens.clear();
        }
    }

    /// Build a grammar sampler for the given grammar type.
    /// Only available when the `common` feature of llama-cpp-2 is enabled
    /// (via llamacpp-metal, llamacpp-vulkan, or llamacpp-cuda).
    #[cfg(any(
        feature = "llamacpp-metal",
        feature = "llamacpp-vulkan",
        feature = "llamacpp-cuda"
    ))]
    fn build_grammar_sampler(
        &self,
        model: &LlamaModel,
        grammar_type: &GrammarType,
    ) -> Result<Vec<LlamaSampler>, BackendError> {
        match grammar_type {
            GrammarType::JsonSchema { schema } => {
                let schema_str = serde_json::to_string(schema).map_err(|e| {
                    BackendError::GenerationFailed(format!("schema serialize: {e}"))
                })?;
                let gbnf = llama_cpp_2::json_schema_to_grammar(&schema_str)
                    .map_err(|e| BackendError::GenerationFailed(format!("schema→grammar: {e}")))?;
                let sampler = LlamaSampler::grammar(model, &gbnf, "root")
                    .map_err(|e| BackendError::GenerationFailed(format!("grammar init: {e}")))?;
                Ok(vec![sampler])
            }
            GrammarType::Regex { pattern } => {
                // llama.cpp GBNF has no `/regex/` literal syntax — translate the
                // regex to a real GBNF rule set instead of embedding it raw.
                let gbnf = regex_to_gbnf(pattern)
                    .map_err(|e| BackendError::GenerationFailed(format!("regex→gbnf: {e}")))?;
                let sampler = LlamaSampler::grammar(model, &gbnf, "root")
                    .map_err(|e| BackendError::GenerationFailed(format!("regex grammar: {e}")))?;
                Ok(vec![sampler])
            }
            GrammarType::Lark { .. } => Err(BackendError::GrammarValidationFailed(
                "Lark grammars are not supported by the llama.cpp backend; \
                 use JSON Schema or regex constraints (Lark EBNF ≠ GBNF)"
                    .into(),
            )),
            GrammarType::None => Ok(vec![]),
        }
    }

    /// Get a reference to the loaded model, if any.
    fn with_model<R>(&self, f: impl FnOnce(&LlamaModel) -> R) -> Result<R, BackendError> {
        let guard = self.model.lock();
        let model = guard.as_ref().ok_or(BackendError::NotLoaded)?;
        Ok(f(model))
    }

    /// Get or create the persistent decode session for `key`.
    ///
    /// The session is recreated when the context parameters change. The
    /// returned guard keeps the session locked for the duration of the call.
    /// Must be called while holding the model lock (via `with_model`) so the
    /// model cannot be unloaded mid-generation.
    fn ensure_session<'a>(
        &'a self,
        model: &LlamaModel,
        key: CtxKey,
    ) -> Result<parking_lot::MutexGuard<'a, Option<PersistentSession>>, BackendError> {
        let mut guard = self.session.lock();
        let stale = match guard.as_ref() {
            Some(sess) => sess.ctx_key != key,
            None => true,
        };
        if stale {
            *guard = None; // drop old context before creating a new one
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(key.context_size as u32))
                .with_n_batch(key.batch_size)
                .with_n_ubatch(key.batch_size)
                .with_n_threads(key.threads as i32)
                .with_n_threads_batch(key.threads as i32)
                .with_type_k(KvCacheType::Q8_0)
                .with_type_v(KvCacheType::Q8_0);
            let ctx = model
                .new_context(llama_backend()?, ctx_params)
                .map_err(|e| BackendError::GenerationFailed(format!("context init: {e}")))?;
            // SAFETY: the model outlives this context — `model` is boxed in
            // `self.model` and the session is always dropped first (see the
            // `unsafe impl Send` note on `PersistentSession`).
            let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
            let sess = PersistentSession {
                ctx,
                ctx_key: key,
                cached_tokens: Vec::new(),
            };
            // Attach the active LoRA adapter (if any) to the fresh context.
            if let Some(slot) = self.lora.lock().as_mut() {
                sess.ctx
                    .lora_adapter_set(&mut slot.adapter, slot.scale)
                    .map_err(|e| BackendError::GenerationFailed(format!("lora attach: {e}")))?;
            }
            *guard = Some(sess);
        }
        Ok(guard)
    }

    /// Build a sampler chain from generation config + optional grammar.
    fn build_sampler(
        &self,
        model: &LlamaModel,
        config: &GenerationConfig,
    ) -> Result<LlamaSampler, BackendError> {
        let mut samplers: Vec<LlamaSampler> = Vec::with_capacity(6);

        // 1. Grammar constraint (if any) — applied first to mask invalid tokens
        // Note: Grammar sampler requires the `common` feature of llama-cpp-2.
        // When not enabled, grammar constraints are not enforced at the sampler
        // level (but output is still validated post-generation by GrammarValidator).
        #[cfg(any(
            feature = "llamacpp-metal",
            feature = "llamacpp-vulkan",
            feature = "llamacpp-cuda"
        ))]
        if let Some(grammar) = &config.grammar {
            samplers.extend(self.build_grammar_sampler(model, &grammar.grammar_type)?);
        }
        #[cfg(not(any(
            feature = "llamacpp-metal",
            feature = "llamacpp-vulkan",
            feature = "llamacpp-cuda"
        )))]
        let _ = model;

        // 2. Repetition penalty (penalty_last_n=-1 means apply to all tokens)
        if config.repeat_penalty != 1.0 {
            samplers.push(LlamaSampler::penalties(
                -1, // last_n = -1 → apply to all tokens
                config.repeat_penalty,
                0.0, // freq penalty
                0.0, // present penalty
            ));
        }

        // 3. Temperature
        if config.temperature > 0.0 {
            samplers.push(LlamaSampler::temp(config.temperature));
        }

        // 4. Top-P (nucleus sampling)
        samplers.push(LlamaSampler::top_p(config.top_p, 1));

        // 5. Top-K
        if config.top_k > 0 {
            samplers.push(LlamaSampler::top_k(config.top_k as i32));
        }

        // 6. Final selection: greedy if temperature is 0, otherwise distribution
        if config.temperature <= 0.0 {
            samplers.push(LlamaSampler::greedy());
        } else {
            let seed = if config.seed == 0 {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as u32
            } else {
                config.seed as u32
            };
            samplers.push(LlamaSampler::dist(seed));
        }

        Ok(LlamaSampler::chain(samplers, false))
    }

    /// Merge stop sequences from config.stop + grammar (previously only
    /// grammar stop sequences were honored — config.stop was ignored).
    fn merged_stops(config: &GenerationConfig) -> Vec<String> {
        let mut stops: Vec<String> = config.stop.clone();
        if let Some(g) = &config.grammar {
            stops.extend(g.stop_sequences.iter().cloned());
            if g.stop_on_newline {
                stops.push("\n".into());
            }
        }
        stops.retain(|s| !s.is_empty());
        stops.dedup();
        stops
    }

    /// Whether a sampler-level grammar constraint was compiled in.
    fn grammar_is_enforced(config: &GenerationConfig) -> bool {
        cfg!(any(
            feature = "llamacpp-metal",
            feature = "llamacpp-vulkan",
            feature = "llamacpp-cuda"
        )) && config
            .grammar
            .as_ref()
            .is_some_and(|g| !matches!(g.grammar_type, GrammarType::None))
    }

    /// Emit one sampled token: detokenize, append, stream, and check stop
    /// sequences. Returns `true` when generation should stop (EOG token or a
    /// stop-sequence hit — the text is already truncated).
    fn emit_token(
        model: &LlamaModel,
        st: &mut EmitState,
        token: LlamaToken,
        stream: Option<&StreamHandle>,
        stops: &[String],
        max_stop_len: usize,
        start: Instant,
    ) -> Result<bool, BackendError> {
        if model.is_eog_token(token) {
            return Ok(true);
        }
        if st.completion_tokens == 0 {
            st.ttft_ms = start.elapsed().as_millis() as u64;
        }
        // Detokenize incrementally — a multi-byte UTF-8 character may span
        // several tokens; buffer incomplete sequences instead of emitting
        // replacement characters.
        let piece_bytes = model
            .token_to_piece_bytes(token, 128, true, None)
            .map_err(|e| BackendError::GenerationFailed(format!("token→bytes: {e}")))?;
        let piece = st.detok.push(&piece_bytes);
        st.generated_text.push_str(&piece);
        st.completion_tokens += 1;

        if let Some(s) = stream {
            if !piece.is_empty() {
                s.push_token(piece.clone());
            }
        }

        // Check stop sequences in the tail window only — a stop sequence
        // can only complete at the current tail (checked every token).
        if max_stop_len > 0 && !st.generated_text.is_empty() {
            let window = st
                .generated_text
                .len()
                .saturating_sub(max_stop_len + piece.len());
            let window = floor_char_boundary(&st.generated_text, window);
            let hit = stops
                .iter()
                .filter_map(|s| st.generated_text[window..].find(s).map(|p| window + p))
                .min();
            if let Some(pos) = hit {
                st.generated_text.truncate(pos);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Shared generation tail: flush detokenizer, complete the stream,
    /// post-validate grammar, and build the result.
    fn finish_generation(
        mut st: EmitState,
        config: &GenerationConfig,
        stream: Option<&StreamHandle>,
        backend_config: &BackendConfig,
        prompt_tokens: u32,
        start: Instant,
    ) -> GenerationResult {
        // Flush any remaining buffered bytes (e.g. truncated multibyte tail)
        let tail = st.detok.flush();
        st.generated_text.push_str(&tail);

        if let Some(s) = stream {
            if !s.is_cancelled() {
                s.complete(st.completion_tokens, start.elapsed().as_millis() as u64);
            }
        }

        // Truthful grammar_valid: always post-validate when a grammar is
        // present — sampler constraints guarantee structure but not schema
        // semantics; when the sampler path was unavailable (no `common`
        // feature) validation is the only enforcement.
        let grammar_valid = match &config.grammar {
            Some(g) if !matches!(g.grammar_type, GrammarType::None) => {
                let ok = GrammarValidator::validate(&st.generated_text, g).is_ok();
                if !ok && !Self::grammar_is_enforced(config) {
                    tracing::warn!("grammar output failed validation and was not sampler-enforced");
                }
                ok
            }
            _ => true,
        };

        let total_ms = start.elapsed().as_millis() as u64;
        let tps = if total_ms > 0 {
            st.completion_tokens as f64 * 1000.0 / total_ms as f64
        } else {
            0.0
        };

        GenerationResult {
            text: st.generated_text,
            prompt_tokens,
            completion_tokens: st.completion_tokens,
            ttft_ms: st.ttft_ms,
            total_ms,
            tokens_per_second: tps,
            backend: backend_config.backend_type.as_str().into(),
            grammar_valid,
        }
    }

    /// Run the generation loop, pushing tokens to the stream (if provided).
    #[allow(clippy::too_many_lines)]
    fn run_generation(
        &self,
        model: &LlamaModel,
        prompt: &str,
        config: &GenerationConfig,
        stream: Option<&StreamHandle>,
        backend_config: &BackendConfig,
    ) -> Result<GenerationResult, BackendError> {
        let start = Instant::now();

        // 1. Tokenize the prompt
        let tokens = model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| BackendError::GenerationFailed(format!("tokenize: {e}")))?;

        let prompt_tokens = tokens.len() as u32;

        let ctx_key = CtxKey {
            context_size: backend_config.context_size,
            batch_size: backend_config.batch_size.max(1),
            threads: backend_config.threads,
        };

        // 2. Optional same-model MTP speculative decoding. Uses dedicated
        // target + draft contexts owned by `MtpSpeculative` — the persistent
        // session is bypassed for this request. Falls back to the persistent
        // path when the model has no MTP draft heads.
        #[cfg(any(
            feature = "llamacpp-metal",
            feature = "llamacpp-vulkan",
            feature = "llamacpp-cuda"
        ))]
        if backend_config.speculative {
            if let Some(result) =
                self.run_generation_mtp(model, &tokens, config, stream, backend_config, ctx_key)?
            {
                return Ok(result);
            }
        }

        // 3. Get or create the persistent decode context (KV prefix reuse).
        let mut session_guard = self.ensure_session(model, ctx_key)?;
        let sess = session_guard
            .as_mut()
            .ok_or_else(|| BackendError::GenerationFailed("session init".into()))?;
        let ctx = &mut sess.ctx;
        let batch_size = ctx_key.batch_size;

        // 4. Build sampler (grammar constraints enforced here when compiled in)
        let mut sampler = self.build_sampler(model, config)?;

        let n_ctx = ctx.n_ctx() as i32;
        let n_prompt = tokens.len() as i32;
        if n_prompt > n_ctx {
            return Err(BackendError::GenerationFailed(format!(
                "prompt too long: {n_prompt} > ctx {n_ctx}"
            )));
        }

        // Verify prompt + max_tokens fits in context window
        let max_tokens = config.max_tokens as i32;
        if n_prompt + max_tokens > n_ctx {
            return Err(BackendError::GenerationFailed(format!(
                "prompt + max_tokens exceeds context: {} + {} > {}",
                n_prompt, max_tokens, n_ctx
            )));
        }

        // 5. KV prefix reuse — keep the longest shared prefix between the
        // tokens already resident in seq 0 and the new prompt, evict the
        // tail, then decode only the new suffix in `batch_size` chunks.
        // Logits are requested only for the final prompt token.
        let common = sess
            .cached_tokens
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| a == b)
            .count();
        // Always re-decode the final prompt token: logits are per-decode, not
        // cached — if the whole prompt were reused, sampling at -1 would read
        // stale logits from the previous request's last generated token.
        let common = if common == tokens.len() {
            common.saturating_sub(1)
        } else {
            common
        };
        if common < sess.cached_tokens.len() {
            ctx.kv_cache_seq_rm(0, Some(common as u32), None)
                .map_err(|e| BackendError::GenerationFailed(format!("kv trim: {e}")))?;
            sess.cached_tokens.truncate(common);
        }
        if common < tokens.len() {
            tracing::debug!(
                "KV prefix reuse: {} cached / {} new prompt tokens",
                common,
                tokens.len()
            );
        }

        let chunk = batch_size as usize;
        let mut batch = LlamaBatch::new(chunk, 1);
        let mut offset = common;
        while offset < tokens.len() {
            batch.clear();
            let end = (offset + chunk).min(tokens.len());
            for (i, &token) in tokens[offset..end].iter().enumerate() {
                let pos = (offset + i) as i32;
                let last = offset + i == tokens.len() - 1;
                batch
                    .add(token, pos, &[0], last)
                    .map_err(batch_add_error_to_backend_error)?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| BackendError::GenerationFailed(format!("prompt decode: {e}")))?;
            offset = end;
        }
        // cached_tokens already holds the `common` prefix — append only the
        // newly-decoded suffix so positions stay aligned with the KV cache.
        sess.cached_tokens.extend_from_slice(&tokens[common..]);

        // 6. Stop sequences: config.stop merged with grammar stop sequences.
        let stops = Self::merged_stops(config);
        let max_stop_len = stops.iter().map(|s| s.len()).max().unwrap_or(0);

        // 7. Generation loop with incremental UTF-8-safe detokenization.
        let mut st = EmitState::default();

        for _ in 0..max_tokens {
            // Check cancellation
            if let Some(s) = stream {
                if s.is_cancelled() {
                    break;
                }
            }

            // Sample next token
            let new_token = sampler.sample(ctx, -1);

            if Self::emit_token(
                model,
                &mut st,
                new_token,
                stream,
                &stops,
                max_stop_len,
                start,
            )? {
                break;
            }

            // Prepare next batch with the new token
            batch.clear();
            let pos = sess.cached_tokens.len() as i32;
            batch
                .add(new_token, pos, &[0], true)
                .map_err(batch_add_error_to_backend_error)?;
            ctx.decode(&mut batch)
                .map_err(|e| BackendError::GenerationFailed(format!("decode: {e}")))?;
            // Track the decoded token so its KV is reusable as prefix.
            sess.cached_tokens.push(new_token);
        }

        Ok(Self::finish_generation(
            st,
            config,
            stream,
            backend_config,
            prompt_tokens,
            start,
        ))
    }

    /// Same-model MTP speculative decoding path (requires the `common`
    /// feature of llama-cpp-2, enabled by the GPU backend features).
    ///
    /// Creates a dedicated target context plus an MTP draft context, prefills
    /// the prompt, then alternates between draft proposal and batched
    /// verification on the target model. Returns `Ok(None)` when the model
    /// has no MTP heads — the caller then falls back to the persistent path.
    #[cfg(any(
        feature = "llamacpp-metal",
        feature = "llamacpp-vulkan",
        feature = "llamacpp-cuda"
    ))]
    #[allow(clippy::too_many_lines)]
    fn run_generation_mtp(
        &self,
        model: &LlamaModel,
        tokens: &[LlamaToken],
        config: &GenerationConfig,
        stream: Option<&StreamHandle>,
        backend_config: &BackendConfig,
        ctx_key: CtxKey,
    ) -> Result<Option<GenerationResult>, BackendError> {
        let start = Instant::now();
        let ctx_params = || {
            LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(ctx_key.context_size as u32))
                .with_n_batch(ctx_key.batch_size)
                .with_n_ubatch(ctx_key.batch_size)
                .with_n_threads(ctx_key.threads as i32)
                .with_n_threads_batch(ctx_key.threads as i32)
        };
        let target_ctx = model
            .new_context(llama_backend()?, ctx_params())
            .map_err(|e| BackendError::GenerationFailed(format!("mtp target ctx: {e}")))?;
        // The draft context shares the model but runs the MTP draft head.
        let draft_ctx = model
            .new_context_with_ctx_other(
                llama_backend()?,
                ctx_params().with_context_type(LlamaContextType::Mtp),
                &target_ctx,
            )
            .map_err(|e| BackendError::GenerationFailed(format!("mtp draft ctx: {e}")))?;

        let mut spec =
            match MtpSpeculative::new(target_ctx, draft_ctx, MtpSpeculativeParams::default()) {
                Ok(spec) => spec,
                // Model lacks MTP draft heads — caller falls back.
                Err(MtpSpeculativeError::InitFailed) => return Ok(None),
                Err(e) => {
                    return Err(BackendError::GenerationFailed(format!("mtp init: {e}")));
                }
            };
        let mtp_err = |e: MtpSpeculativeError| BackendError::GenerationFailed(format!("mtp: {e}"));

        // Attach the active LoRA adapter (if any) to the target context.
        if let Some(slot) = self.lora.lock().as_mut() {
            spec.target_context_mut()
                .lora_adapter_set(&mut slot.adapter, slot.scale)
                .map_err(|e| BackendError::GenerationFailed(format!("lora attach: {e}")))?;
        }

        spec.begin(tokens).map_err(mtp_err)?;

        let mut sampler = self.build_sampler(model, config)?;
        let n_ctx = spec.target_context().n_ctx() as i32;
        let mut n_past = tokens.len() as i32;
        let max_tokens = config.max_tokens as i32;
        if n_past + max_tokens > n_ctx {
            return Err(BackendError::GenerationFailed(format!(
                "prompt + max_tokens exceeds context: {n_past} + {max_tokens} > {n_ctx}"
            )));
        }

        // Prefill the prompt on the target context (chunked; logits on last).
        let chunk = ctx_key.batch_size as usize;
        let mut batch = LlamaBatch::new(chunk.max(4), 1);
        let mut offset = 0usize;
        while offset < tokens.len() {
            batch.clear();
            let end = (offset + chunk).min(tokens.len());
            for (i, &token) in tokens[offset..end].iter().enumerate() {
                let last = offset + i == tokens.len() - 1;
                batch
                    .add(token, (offset + i) as i32, &[0], last)
                    .map_err(batch_add_error_to_backend_error)?;
            }
            spec.target_context_mut()
                .decode(&mut batch)
                .map_err(|e| BackendError::GenerationFailed(format!("mtp prefill: {e}")))?;
            offset = end;
        }

        let stops = Self::merged_stops(config);
        let max_stop_len = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut st = EmitState::default();

        'outer: while (st.completion_tokens as i32) < max_tokens {
            if let Some(s) = stream {
                if s.is_cancelled() {
                    break;
                }
            }

            let token = sampler.sample(spec.target_context(), -1);
            if Self::emit_token(model, &mut st, token, stream, &stops, max_stop_len, start)? {
                break;
            }

            // Feed the emitted token to both contexts.
            batch.clear();
            batch
                .add(token, n_past, &[0], true)
                .map_err(batch_add_error_to_backend_error)?;
            spec.target_context_mut()
                .decode(&mut batch)
                .map_err(|e| BackendError::GenerationFailed(format!("mtp decode: {e}")))?;
            spec.process(&batch).map_err(mtp_err)?;
            n_past += 1;

            // Draft only when there is context room for the maximum draft.
            if n_past + 4 >= n_ctx {
                continue;
            }
            let drafts = spec.draft(n_past, token, tokens).map_err(mtp_err)?;
            if drafts.is_empty() {
                continue;
            }

            // Verify: decode all draft tokens in one batch with logits at
            // every position, then accept the longest matching prefix.
            batch.clear();
            for (i, &d) in drafts.iter().enumerate() {
                batch
                    .add(d, n_past + i as i32, &[0], true)
                    .map_err(batch_add_error_to_backend_error)?;
            }
            spec.target_context_mut()
                .decode(&mut batch)
                .map_err(|e| BackendError::GenerationFailed(format!("mtp verify: {e}")))?;

            let mut accepted = 0usize;
            let mut replacement = None;
            for (i, &d) in drafts.iter().enumerate() {
                if (st.completion_tokens as i32) >= max_tokens {
                    break;
                }
                let s = sampler.sample(spec.target_context(), i as i32);
                if Self::emit_token(model, &mut st, s, stream, &stops, max_stop_len, start)? {
                    break 'outer;
                }
                if s == d {
                    accepted += 1;
                } else {
                    replacement = Some(s);
                    break;
                }
            }

            // Drop KV for rejected draft positions. When a replacement token
            // was emitted, the position at `n_past + accepted` holds KV for
            // the wrong (rejected) draft — remove it and re-decode below.
            let keep = n_past + accepted as i32;
            spec.target_context_mut()
                .kv_cache_seq_rm(0, Some(keep as u32), None)
                .map_err(|e| BackendError::GenerationFailed(format!("mtp rollback: {e}")))?;
            spec.accept(accepted as u16).map_err(mtp_err)?;
            n_past = keep;

            if let Some(tok) = replacement {
                batch.clear();
                batch
                    .add(tok, n_past, &[0], true)
                    .map_err(batch_add_error_to_backend_error)?;
                spec.target_context_mut()
                    .decode(&mut batch)
                    .map_err(|e| BackendError::GenerationFailed(format!("mtp decode: {e}")))?;
                spec.process(&batch).map_err(mtp_err)?;
                n_past += 1;
            }
        }

        Ok(Some(Self::finish_generation(
            st,
            config,
            stream,
            backend_config,
            tokens.len() as u32,
            start,
        )))
    }
}

/// Per-request emission state shared by the persistent and MTP decode loops.
#[derive(Default)]
struct EmitState {
    generated_text: String,
    detok: Detokenizer,
    completion_tokens: u32,
    ttft_ms: u64,
}

/// Incremental detokenizer — buffers token piece bytes until they form
/// complete UTF-8 characters.
///
/// Token pieces may split a multi-byte character across tokens (common for
/// CJK and emoji). Converting each piece with `from_utf8_lossy` would emit
/// `` for the split halves; this buffers until a valid boundary.
#[derive(Default)]
struct Detokenizer {
    pending: Vec<u8>,
}

impl Detokenizer {
    /// Push a token piece; returns the newly-decodable text (may be empty).
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        // The prefix is provably valid UTF-8.
                        out.push_str(
                            std::str::from_utf8(&self.pending[..valid])
                                .expect("valid_up_to prefix is valid UTF-8"),
                        );
                        self.pending.drain(..valid);
                        continue;
                    }
                    match e.error_len() {
                        // Genuinely invalid bytes — emit replacement char.
                        Some(bad_len) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..bad_len);
                        }
                        // Incomplete trailing sequence — wait for more bytes.
                        None => break,
                    }
                }
            }
        }
        out
    }

    /// Flush remaining buffered bytes, lossy if incomplete.
    fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// Floor `idx` to the nearest char boundary at or before it.
fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

/// Convert a [`BatchAddError`] to a [`BackendError`].
fn batch_add_error_to_backend_error(e: BatchAddError) -> BackendError {
    BackendError::GenerationFailed(format!("batch add: {e}"))
}

impl BackendAdapter for LlamaCppBackend {
    fn load(&self, config: &BackendConfig) -> Result<(), BackendError> {
        // Drop the session BEFORE the model — the session's context holds a
        // reference into the boxed model and must not outlive it.
        *self.session.lock() = None;
        *self.lora.lock() = None;
        {
            let mut model_guard = self.model.lock();
            if model_guard.is_some() {
                *model_guard = None; // Drop the old model
            }
        }

        // Build model params — offload GPU layers based on config
        // gpu_layers: -1 = all layers, 0 = CPU only, N = N layers on GPU
        let n_gpu_layers: u32 = if config.gpu_layers < 0 {
            // -1 means "all layers" — use a large number
            9999
        } else {
            config.gpu_layers as u32
        };

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model_params = Pin::new(Box::new(model_params));

        // Load the model from file
        let model = LlamaModel::load_from_file(
            llama_backend()?,
            Path::new(&config.model_path),
            &model_params,
        )
        .map_err(|e| BackendError::LoadFailed(format!("llama.cpp load: {e}")))?;

        *self.model.lock() = Some(Box::new(model));
        *self.config.lock() = Some(config.clone());

        tracing::info!(
            "Loaded model {} from {} (gpu_layers={})",
            config.model_pack_id,
            config.model_path,
            config.gpu_layers
        );

        Ok(())
    }

    fn unload(&self) -> Result<(), BackendError> {
        // Drop session (which references the boxed model) before the model.
        *self.session.lock() = None;
        *self.lora.lock() = None;
        *self.model.lock() = None;
        *self.config.lock() = None;
        tracing::info!("Unloaded llama.cpp model");
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.model.lock().is_some()
    }

    fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationResult, BackendError> {
        let backend_config = self.config.lock().clone().ok_or(BackendError::NotLoaded)?;

        self.with_model(|model| self.run_generation(model, prompt, config, None, &backend_config))?
    }

    fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        stream: &StreamHandle,
    ) -> Result<GenerationResult, BackendError> {
        let backend_config = self.config.lock().clone().ok_or(BackendError::NotLoaded)?;

        self.with_model(|model| {
            self.run_generation(model, prompt, config, Some(stream), &backend_config)
        })?
    }

    fn backend_type(&self) -> BackendType {
        // Determine from the loaded config, or default based on platform
        if let Some(cfg) = self.config.lock().as_ref() {
            return cfg.backend_type;
        }
        // Default: Metal on macOS/iOS, Vulkan on Android/Windows, CPU otherwise
        #[cfg(all(target_os = "macos", feature = "llamacpp-metal"))]
        {
            BackendType::LlamaCppMetal
        }
        #[cfg(all(target_os = "ios", feature = "llamacpp-metal"))]
        {
            BackendType::LlamaCppMetal
        }
        #[cfg(all(
            any(target_os = "android", target_os = "windows"),
            feature = "llamacpp-vulkan"
        ))]
        {
            BackendType::LlamaCppVulkan
        }
        #[cfg(not(any(
            all(target_os = "macos", feature = "llamacpp-metal"),
            all(target_os = "ios", feature = "llamacpp-metal"),
            all(
                any(target_os = "android", target_os = "windows"),
                feature = "llamacpp-vulkan"
            ),
        )))]
        {
            BackendType::LlamaCppCpu
        }
    }

    fn apply_lora(&self, adapter_path: &str, scale: f32) -> Result<(), BackendError> {
        self.set_lora(adapter_path, scale)
    }

    fn detach_lora(&self) -> Result<(), BackendError> {
        self.clear_lora();
        Ok(())
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        // Just verify the backend can be created (initializes llama.cpp)
        let _backend = LlamaCppBackend::new();
    }

    #[test]
    fn test_unload_without_load() {
        let backend = LlamaCppBackend::new();
        assert!(!backend.is_loaded());
        // Unloading when not loaded should succeed
        assert!(backend.unload().is_ok());
    }

    #[test]
    fn test_generate_not_loaded() {
        let backend = LlamaCppBackend::new();
        let config = GenerationConfig::default();
        let result = backend.generate("hello", &config);
        assert!(matches!(result, Err(BackendError::NotLoaded)));
    }
}

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
use crate::grammar::GrammarType;
use crate::stream::StreamHandle;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::{BatchAddError, LlamaBatch};
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::num::NonZeroU32;
use std::path::Path;
use std::pin::Pin;
use std::time::Instant;

/// Global singleton for the llama.cpp backend (can only be initialized once).
static LLAMA_BACKEND: OnceCell<LlamaBackend> = OnceCell::new();

/// Get the global llama.cpp backend instance, initializing it on first call.
fn llama_backend() -> &'static LlamaBackend {
    LLAMA_BACKEND.get_or_init(|| LlamaBackend::init().expect("failed to init llama.cpp backend"))
}

/// Real llama.cpp backend using the `llama-cpp-2` crate.
///
/// The backend holds a single loaded model and creates a fresh context per
/// generation call. The model is loaded once and reused across calls; the
/// context (KV cache) is recreated each time to avoid state leakage.
pub struct LlamaCppBackend {
    /// The loaded model, if any. Stored behind a Mutex for thread safety.
    model: Mutex<Option<LlamaModel>>,
    /// The current backend configuration (set on load).
    config: Mutex<Option<BackendConfig>>,
}

impl LlamaCppBackend {
    /// Create a new llama.cpp backend. The llama.cpp global backend is
    /// initialized lazily on first use via a static `OnceCell`.
    pub fn new() -> Self {
        // Touch the singleton to initialize it eagerly (panics on init failure)
        let _ = llama_backend();
        Self {
            model: Mutex::new(None),
            config: Mutex::new(None),
        }
    }

    /// Build a grammar sampler for the given grammar type.
    /// Only available when the `common` feature of llama-cpp-2 is enabled
    /// (via llamacpp-metal, llamacpp-vulkan, or llamacpp-cuda).
    #[cfg(any(feature = "llamacpp-metal", feature = "llamacpp-vulkan", feature = "llamacpp-cuda"))]
    fn build_grammar_sampler(
        &self,
        model: &LlamaModel,
        grammar_type: &GrammarType,
    ) -> Result<Vec<LlamaSampler>, BackendError> {
        match grammar_type {
            GrammarType::JsonSchema { schema } => {
                let schema_str = serde_json::to_string(schema)
                    .map_err(|e| BackendError::GenerationFailed(format!("schema serialize: {e}")))?;
                let gbnf = llama_cpp_2::json_schema_to_grammar(&schema_str)
                    .map_err(|e| BackendError::GenerationFailed(format!("schema→grammar: {e}")))?;
                let sampler = LlamaSampler::grammar(model, &gbnf, "root")
                    .map_err(|e| BackendError::GenerationFailed(format!("grammar init: {e}")))?;
                Ok(vec![sampler])
            }
            GrammarType::Regex { pattern } => {
                let gbnf = format!("root ::= /{}/\n", pattern.replace('/', "\\/"));
                let sampler = LlamaSampler::grammar(model, &gbnf, "root")
                    .map_err(|e| BackendError::GenerationFailed(format!("regex grammar: {e}")))?;
                Ok(vec![sampler])
            }
            GrammarType::Lark { grammar } => {
                let sampler = LlamaSampler::grammar(model, grammar, "root")
                    .map_err(|e| BackendError::GenerationFailed(format!("lark grammar: {e}")))?;
                Ok(vec![sampler])
            }
            GrammarType::None => Ok(vec![]),
        }
    }

    /// Get a reference to the loaded model, if any.
    fn with_model<R>(&self, f: impl FnOnce(&LlamaModel) -> R) -> Result<R, BackendError> {
        let guard = self.model.lock();
        let model = guard.as_ref().ok_or(BackendError::NotLoaded)?;
        Ok(f(model))
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
        #[cfg(feature = "llamacpp-metal")]
        if let Some(grammar) = &config.grammar {
            samplers.extend(self.build_grammar_sampler(model, &grammar.grammar_type)?);
        }
        #[cfg(feature = "llamacpp-vulkan")]
        if let Some(grammar) = &config.grammar {
            samplers.extend(self.build_grammar_sampler(model, &grammar.grammar_type)?);
        }
        #[cfg(feature = "llamacpp-cuda")]
        if let Some(grammar) = &config.grammar {
            samplers.extend(self.build_grammar_sampler(model, &grammar.grammar_type)?);
        }

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

        // 2. Create context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(backend_config.context_size as u32))
            .with_n_threads(backend_config.threads as i32)
            .with_n_threads_batch(backend_config.threads as i32)
            .with_type_k(KvCacheType::Q8_0)
            .with_type_v(KvCacheType::Q8_0);

        let mut ctx = model
            .new_context(llama_backend(), ctx_params)
            .map_err(|e| BackendError::GenerationFailed(format!("context init: {e}")))?;

        // 3. Build sampler
        let mut sampler = self.build_sampler(model, config)?;

        // 4. Feed prompt tokens via batch
        let batch_size = backend_config.batch_size as usize;
        let mut batch = LlamaBatch::new(batch_size, 1);

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

        // Add prompt tokens to batch
        for (i, &token) in tokens.iter().enumerate() {
            let i = i as i32;
            batch
                .add(token, i, &[0], true)
                .map_err(batch_add_error_to_backend_error)?;
        }

        // Decode the prompt batch
        ctx.decode(&mut batch)
            .map_err(|e| BackendError::GenerationFailed(format!("prompt decode: {e}")))?;

        // 5. Generation loop
        let mut generated_text = String::new();
        let mut completion_tokens = 0u32;
        let mut ttft_ms: u64 = 0;

        for _ in 0..max_tokens {
            // Check cancellation
            if let Some(s) = stream {
                if s.is_cancelled() {
                    break;
                }
            }

            // Sample next token
            let new_token = sampler.sample(&ctx, -1);

            // Check for end-of-generation token
            if model.is_eog_token(new_token) {
                break;
            }

            // Record TTFT on first token
            if completion_tokens == 0 {
                ttft_ms = start.elapsed().as_millis() as u64;
            }

            // Convert token to text — use token_to_piece_bytes for safety
            let piece_bytes = model
                .token_to_piece_bytes(new_token, 128, true, None)
                .map_err(|e| BackendError::GenerationFailed(format!("token→bytes: {e}")))?;
            let piece = String::from_utf8_lossy(&piece_bytes);
            generated_text.push_str(&piece);
            completion_tokens += 1;

            // Push to stream
            if let Some(s) = stream {
                s.push_token(piece.into_owned());
            }

            // Check stop sequences
            if let Some(g) = &config.grammar {
                for stop in &g.stop_sequences {
                    if let Some(pos) = generated_text.find(stop) {
                        // Truncate at the first occurrence of the stop sequence
                        generated_text.truncate(pos);
                        // Mark as complete and exit
                        if let Some(s) = stream {
                            s.complete(completion_tokens, start.elapsed().as_millis() as u64);
                        }
                        let total_ms = start.elapsed().as_millis() as u64;
                        let tps = if total_ms > 0 {
                            completion_tokens as f64 * 1000.0 / total_ms as f64
                        } else {
                            0.0
                        };
                        return Ok(GenerationResult {
                            text: generated_text,
                            prompt_tokens,
                            completion_tokens,
                            ttft_ms,
                            total_ms,
                            tokens_per_second: tps,
                            backend: backend_config.backend_type.as_str().into(),
                            grammar_valid: true,
                        });
                    }
                }
            }

            // Prepare next batch with the new token
            batch.clear();
            let pos = n_prompt + completion_tokens as i32 - 1;
            batch
                .add(new_token, pos, &[0], true)
                .map_err(batch_add_error_to_backend_error)?;
            ctx.decode(&mut batch)
                .map_err(|e| BackendError::GenerationFailed(format!("decode: {e}")))?;
        }

        // 6. Mark stream complete
        if let Some(s) = stream {
            if !s.is_cancelled() {
                s.complete(completion_tokens, start.elapsed().as_millis() as u64);
            }
        }

        let total_ms = start.elapsed().as_millis() as u64;
        let tps = if total_ms > 0 {
            completion_tokens as f64 * 1000.0 / total_ms as f64
        } else {
            0.0
        };

        Ok(GenerationResult {
            text: generated_text,
            prompt_tokens,
            completion_tokens,
            ttft_ms,
            total_ms,
            tokens_per_second: tps,
            backend: backend_config.backend_type.as_str().into(),
            grammar_valid: true,
        })
    }
}

/// Convert a [`BatchAddError`] to a [`BackendError`].
fn batch_add_error_to_backend_error(e: BatchAddError) -> BackendError {
    BackendError::GenerationFailed(format!("batch add: {e}"))
}

impl BackendAdapter for LlamaCppBackend {
    fn load(&self, config: &BackendConfig) -> Result<(), BackendError> {
        // Unload any existing model first
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
            llama_backend(),
            Path::new(&config.model_path),
            &model_params,
        )
        .map_err(|e| BackendError::LoadFailed(format!("llama.cpp load: {e}")))?;

        *self.model.lock() = Some(model);
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

        self.with_model(|model| {
            self.run_generation(model, prompt, config, None, &backend_config)
        })?
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

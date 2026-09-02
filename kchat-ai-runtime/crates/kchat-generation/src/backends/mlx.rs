//! MLX backend — subprocess-based inference via the Swift `kchat-mlx-server`.
//!
//! This backend spawns the Swift `kchat-mlx-server` binary (built from
//! `swift/kchat-mlx-server/`) which uses the PrismML `mlx-swift` fork with
//! 1-bit quantization Metal kernels for Bonsai models. The server exposes a
//! llama-server-compatible HTTP API (`/completion`, `/health`).
//!
//! Unlike the in-process `LlamaCppBackend`, this backend communicates with
//! the model server over HTTP. The Swift server does not support SSE
//! streaming — it always returns the full completion in a single JSON
//! response. `generate_stream` therefore fetches the full response and pushes
//! word-sized chunks to the `StreamHandle` for simulated streaming.
//!
//! The backend is gated behind the `mlx` feature flag.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use tokio::process::{Child, Command};

use crate::backend::{BackendAdapter, BackendConfig, BackendError, BackendType, GenerationConfig, GenerationResult};
use crate::stream::StreamHandle;

/// Default port for the MLX server subprocess.
const DEFAULT_MLX_PORT: u16 = 9943;

/// Timeout for waiting for the MLX server to become ready (30 seconds).
const READY_TIMEOUT_SECS: u64 = 30;

/// Timeout for a single completion request (5 minutes — allows for long
/// generations on slow hardware).
const COMPLETION_TIMEOUT_SECS: u64 = 300;

/// Global lazy-initialized blocking HTTP client.
///
/// `reqwest::blocking::Client` creates its own internal Tokio runtime, which
/// panics if initialized inside an existing Tokio runtime context. Using a
/// `OnceCell` ensures the client is created on first use — which always
/// happens inside `spawn_blocking` (from `load()` or `generate()`), not in
/// the async runtime context.
static HTTP_CLIENT: OnceCell<Client> = OnceCell::new();

/// Get the global blocking HTTP client, initializing it on first call.
/// Must be called from a non-async context (e.g., inside `spawn_blocking`).
fn http_client() -> &'static Client {
    HTTP_CLIENT.get_or_init(|| {
        Client::builder()
            .timeout(Duration::from_secs(COMPLETION_TIMEOUT_SECS))
            .build()
            .expect("failed to build reqwest blocking client")
    })
}

/// MLX backend that spawns the Swift `kchat-mlx-server` subprocess.
///
/// Holds a single loaded model (one subprocess at a time). The subprocess is
/// killed on `unload()` or when the backend is dropped.
///
/// Supports LoRA adapters via the Swift server's `/lora/load` and `/lora/detach`
/// HTTP endpoints. An initial adapter can be passed at startup via
/// `set_lora_path()`, and hot-swap is available at runtime via `load_lora()`
/// and `detach_lora()`.
pub struct MlxBackend {
    child: Mutex<Option<Child>>,
    port: u16,
    model_path: Mutex<Option<String>>,
    /// Optional LoRA adapter path to load at startup.
    lora_path: Mutex<Option<String>>,
}

impl MlxBackend {
    /// Create a new MLX backend. No subprocess is spawned until `load()`.
    /// The HTTP client is lazily initialized on first use (inside
    /// `spawn_blocking`) to avoid Tokio runtime conflicts.
    pub fn new() -> Self {
        Self {
            child: Mutex::new(None),
            port: DEFAULT_MLX_PORT,
            model_path: Mutex::new(None),
            lora_path: Mutex::new(None),
        }
    }

    /// Set a LoRA adapter path to load at startup (before the server becomes
    /// ready). Must be called before `load()`.
    pub fn set_lora_path(&self, path: impl Into<String>) {
        *self.lora_path.lock() = Some(path.into());
    }

    /// The base URL of the running MLX server subprocess.
    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Hot-swap the LoRA adapter at runtime via `POST /lora/load`.
    /// The Swift server unloads the current adapter (if any) and loads the
    /// new one. This blocks until the swap completes.
    pub fn load_lora(&self, adapter_path: &str) -> Result<(), BackendError> {
        let url = format!("{}/lora/load", self.base_url());
        let body = serde_json::json!({ "adapter_path": adapter_path });
        let resp = http_client()
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| BackendError::GenerationFailed(format!("/lora/load failed: {}", e)))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().unwrap_or_default();
            return Err(BackendError::GenerationFailed(format!(
                "/lora/load returned {}: {}",
                status,
                text
            )));
        }
        tracing::info!("LoRA adapter loaded: {}", adapter_path);
        Ok(())
    }

    /// Detach the current LoRA adapter via `POST /lora/detach`.
    /// The model reverts to its base form.
    pub fn detach_lora(&self) -> Result<(), BackendError> {
        let url = format!("{}/lora/detach", self.base_url());
        let resp = http_client()
            .post(&url)
            .send()
            .map_err(|e| BackendError::GenerationFailed(format!("/lora/detach failed: {}", e)))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().unwrap_or_default();
            return Err(BackendError::GenerationFailed(format!(
                "/lora/detach returned {}: {}",
                status,
                text
            )));
        }
        tracing::info!("LoRA adapter detached");
        Ok(())
    }
}

impl Default for MlxBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendAdapter for MlxBackend {
    fn load(&self, config: &BackendConfig) -> Result<(), BackendError> {
        // Kill any existing subprocess before starting a new one.
        self.unload()?;

        let model_path = Path::new(&config.model_path);
        if !model_path.exists() {
            return Err(BackendError::LoadFailed(format!(
                "model path does not exist: {}",
                config.model_path
            )));
        }

        // MLX packs are directories containing config.json + safetensors.
        if !model_path.is_dir() {
            return Err(BackendError::LoadFailed(format!(
                "MLX backend requires a model directory, not a file: {}",
                config.model_path
            )));
        }

        let server_bin = find_mlx_server().map_err(BackendError::LoadFailed)?;
        let port = self.port;

        tracing::info!(
            "Spawning kchat-mlx-server: {} --model {} --port {}",
            server_bin.display(),
            model_path.display(),
            port
        );

        let mut cmd = Command::new(&server_bin);
        cmd.arg("--model").arg(&config.model_path)
            .arg("--port").arg(port.to_string())
            .arg("--host").arg("127.0.0.1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        // Pass --lora if an adapter path was set before load()
        let lora_path = self.lora_path.lock().clone();
        if let Some(ref lora) = lora_path {
            if Path::new(lora).exists() {
                tracing::info!("Passing --lora {} to kchat-mlx-server", lora);
                cmd.arg("--lora").arg(lora);
            } else {
                tracing::warn!("LoRA path does not exist, skipping: {}", lora);
            }
        }

        let mut child = cmd.spawn().map_err(|e| {
            BackendError::LoadFailed(format!("failed to start kchat-mlx-server: {}", e))
        })?;

        // Drain stderr to tracing for debugging.
        if let Some(stderr) = child.stderr.take() {
            let reader = tokio::io::BufReader::new(stderr);
            use tokio::io::AsyncBufReadExt;
            let mut lines = reader.lines();
            tokio::spawn(async move {
                loop {
                    match lines.next_line().await {
                        Ok(Some(line)) => tracing::info!("[kchat-mlx-server] {}", line),
                        Ok(None) => break,
                        Err(e) => {
                            tracing::warn!("[kchat-mlx-server] stderr read error: {}", e);
                            break;
                        }
                    }
                }
            });
        }
        // Drain stdout so it doesn't block.
        if let Some(stdout) = child.stdout.take() {
            let reader = tokio::io::BufReader::new(stdout);
            use tokio::io::AsyncBufReadExt;
            let mut lines = reader.lines();
            tokio::spawn(async move {
                while let Ok(Some(_line)) = lines.next_line().await {}
            });
        }

        *self.child.lock() = Some(child);

        // Wait for the server to be ready.
        if let Err(e) = wait_for_ready(&self.base_url()) {
            self.unload()?;
            return Err(BackendError::LoadFailed(format!(
                "kchat-mlx-server did not become ready: {}",
                e
            )));
        }

        *self.model_path.lock() = Some(config.model_path.clone());

        tracing::info!(
            "kchat-mlx-server is ready on port {} (model: {})",
            port,
            config.model_path
        );
        Ok(())
    }

    fn unload(&self) -> Result<(), BackendError> {
        if let Some(mut child) = self.child.lock().take() {
            tracing::info!("Killing kchat-mlx-server subprocess");
            // unload() is sync (per BackendAdapter trait), so we use the
            // non-async start_kill() + try_wait() instead of kill().await.
            let _ = child.start_kill();
            let _ = child.try_wait();
        }
        *self.model_path.lock() = None;
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.child.lock().is_some()
    }

    fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationResult, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::NotLoaded);
        }

        let start = Instant::now();
        let resp = self.post_completion(prompt, config, None)?;
        let total_ms = start.elapsed().as_millis() as u64;

        Ok(GenerationResult {
            text: resp.content,
            prompt_tokens: resp.tokens_evaluated,
            completion_tokens: resp.tokens_predicted,
            ttft_ms: resp.prompt_ms as u64,
            total_ms,
            tokens_per_second: if resp.predicted_ms > 0.0 {
                (resp.tokens_predicted as f64) / (resp.predicted_ms / 1000.0)
            } else {
                0.0
            },
            backend: BackendType::Mlx.as_str().to_string(),
            grammar_valid: true,
        })
    }

    fn generate_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        stream: &StreamHandle,
    ) -> Result<GenerationResult, BackendError> {
        if !self.is_loaded() {
            return Err(BackendError::NotLoaded);
        }

        let start = Instant::now();
        let sse_result = self.post_completion_stream(prompt, config, stream)?;
        let total_ms = start.elapsed().as_millis() as u64;

        let result = GenerationResult {
            text: sse_result.content,
            prompt_tokens: sse_result.tokens_evaluated,
            completion_tokens: sse_result.tokens_predicted,
            ttft_ms: sse_result.prompt_ms as u64,
            total_ms,
            tokens_per_second: if sse_result.predicted_ms > 0.0 {
                (sse_result.tokens_predicted as f64) / (sse_result.predicted_ms / 1000.0)
            } else {
                0.0
            },
            backend: BackendType::Mlx.as_str().to_string(),
            grammar_valid: true,
        };

        if !stream.is_cancelled() {
            stream.complete(sse_result.tokens_predicted as u32, total_ms);
        }

        Ok(result)
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Mlx
    }
}

impl Drop for MlxBackend {
    fn drop(&mut self) {
        // Kill the subprocess synchronously on drop. We can't call .await in
        // Drop, so we use std::process::Child::kill via try_wait.
        if let Some(mut child) = self.child.lock().take() {
            // tokio::process::Child::start_kill is non-async and sends SIGKILL.
            let _ = child.start_kill();
            // Best-effort wait — non-blocking.
            let _ = child.try_wait();
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// JSON body sent to `/completion`.
#[derive(Serialize)]
struct CompletionBody<'a> {
    prompt: &'a str,
    n_predict: usize,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    seed: u64,
    stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: &'a Vec<String>,
}

/// JSON response from `/completion`.
#[derive(Deserialize)]
struct CompletionResponse {
    content: String,
    tokens_predicted: u32,
    tokens_evaluated: u32,
    #[serde(default)]
    prompt_ms: f64,
    #[serde(default)]
    predicted_ms: f64,
}

impl MlxBackend {
    /// POST to `/completion` and return the parsed response.
    fn post_completion(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        _grammar: Option<&str>,
    ) -> Result<CompletionResponse, BackendError> {
        let body = CompletionBody {
            prompt,
            n_predict: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            repeat_penalty: config.repeat_penalty,
            seed: config.seed,
            // The Swift server doesn't support SSE streaming; stream:false
            // returns the full content in one JSON response.
            stream: false,
            stop: &config.stop,
        };

        let url = format!("{}/completion", self.base_url());
        let resp = http_client()
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| BackendError::GenerationFailed(format!("completion request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(BackendError::GenerationFailed(format!(
                "kchat-mlx-server returned {}: {}",
                status, text
            )));
        }

        resp.json::<CompletionResponse>()
            .map_err(|e| BackendError::GenerationFailed(format!("failed to parse completion response: {}", e)))
    }

    /// POST to `/completion/stream` and consume the SSE stream token-by-token.
    /// Each `data: "token"` event is pushed to the `StreamHandle` immediately.
    /// The final `data: {json}` event contains the full result metadata.
    fn post_completion_stream(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        stream: &StreamHandle,
    ) -> Result<CompletionResponse, BackendError> {
        let body = CompletionBody {
            prompt,
            n_predict: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            repeat_penalty: config.repeat_penalty,
            seed: config.seed,
            stream: true,
            stop: &config.stop,
        };

        let url = format!("{}/completion/stream", self.base_url());
        let resp = http_client()
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| BackendError::GenerationFailed(format!("stream request failed: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(BackendError::GenerationFailed(format!(
                "kchat-mlx-server stream returned {}: {}",
                status, text
            )));
        }

        // Read the SSE stream as bytes, accumulating into a buffer and
        // processing events on "\n\n" boundaries. Using read_until instead of
        // lines() avoids UTF-8 validation issues when multi-byte characters
        // span chunk boundaries in the HTTP response.
        //
        // SSE format:
        //   data: "token"\n\n      → individual token (JSON-encoded string)
        //   data: {json}\n\n        → final result metadata
        //   data: [DONE]\n\n        → end marker
        let mut full_text = String::new();
        let mut final_result: Option<CompletionResponse> = None;
        let mut data_buf = String::new();

        use std::io::BufRead;
        let mut reader = std::io::BufReader::new(resp);
        let mut byte_buf: Vec<u8> = Vec::with_capacity(4096);

        loop {
            if stream.is_cancelled() {
                stream.cancel("cancelled by caller");
                break;
            }

            byte_buf.clear();
            let n = reader.read_until(b'\n', &mut byte_buf).map_err(|e| {
                BackendError::GenerationFailed(format!("SSE read error: {}", e))
            })?;
            if n == 0 {
                break; // EOF
            }

            // Convert bytes to string, handling potential UTF-8 issues
            // at chunk boundaries gracefully.
            let line = String::from_utf8_lossy(&byte_buf);
            let line = line.trim_end_matches('\n');

            if line.is_empty() {
                // Event boundary — flush data buffer
                if !data_buf.is_empty() {
                    process_sse_data(&data_buf, stream, &mut full_text, &mut final_result);
                    data_buf.clear();
                }
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                data_buf = data.to_string();
            }
        }

        // Flush any remaining buffered event
        if !data_buf.is_empty() {
            process_sse_data(&data_buf, stream, &mut full_text, &mut final_result);
        }

        // Return the final result from the server, or construct one from
        // accumulated text if the final JSON event was missing.
        if let Some(result) = final_result {
            Ok(result)
        } else {
            let text_len = full_text.len();
            Ok(CompletionResponse {
                content: full_text,
                tokens_predicted: (text_len / 4) as u32,
                tokens_evaluated: 0,
                prompt_ms: 0.0,
                predicted_ms: 0.0,
            })
        }
    }
}

/// Process a single SSE `data:` payload.
/// - If it's a JSON string (token), push it to the stream.
/// - If it's a JSON object (final result), parse it.
/// - If it's `[DONE]`, ignore (stream end marker).
fn process_sse_data(
    data: &str,
    stream: &StreamHandle,
    full_text: &mut String,
    final_result: &mut Option<CompletionResponse>,
) {
    if data == "[DONE]" {
        return;
    }

    // Try to parse as a JSON string (individual token)
    if let Ok(token) = serde_json::from_str::<String>(data) {
        full_text.push_str(&token);
        stream.push_token(token);
        return;
    }

    // Try to parse as a JSON object (final result metadata)
    if data.starts_with('{') {
        if let Ok(result) = serde_json::from_str::<CompletionResponse>(data) {
            *final_result = Some(result);
        }
    }
}

// ---------------------------------------------------------------------------
// Server discovery + readiness
// ---------------------------------------------------------------------------

/// Find the `kchat-mlx-server` binary.
///
/// Resolution order:
/// 1. `KCHAT_MLX_SERVER` env var (absolute path)
/// 2. `<CARGO_MANIFEST_DIR>/../../swift/kchat-mlx-server/.build/release/kchat-mlx-server`
/// 3. `which("kchat-mlx-server")` (PATH lookup)
fn find_mlx_server() -> Result<PathBuf, String> {
    if let Ok(p) = std::env::var("KCHAT_MLX_SERVER") {
        let path = PathBuf::from(&p);
        if path.exists() {
            tracing::info!("Found kchat-mlx-server at: {} (from KCHAT_MLX_SERVER)", path.display());
            return Ok(path);
        }
        return Err(format!(
            "KCHAT_MLX_SERVER points to non-existent path: {}",
            p
        ));
    }

    // Default: relative to this crate's CARGO_MANIFEST_DIR.
    // crates/kchat-generation -> ../../swift/kchat-mlx-server/.build/
    // SwiftPM 6.0+ uses .build/<arch>-<os>/<mode>/ layout, older versions
    // used .build/<mode>/. Check both.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let build_dir = manifest_dir.join("../../swift/kchat-mlx-server/.build");

    // Try new SwiftPM layout first: .build/arm64-apple-macosx/release/
    let new_path = build_dir.join("arm64-apple-macosx/release/kchat-mlx-server");
    if new_path.exists() {
        tracing::info!("Found kchat-mlx-server at: {} (new SwiftPM layout)", new_path.display());
        return Ok(new_path);
    }

    // Fall back to old SwiftPM layout: .build/release/
    let old_path = build_dir.join("release/kchat-mlx-server");
    if old_path.exists() {
        tracing::info!("Found kchat-mlx-server at: {} (old SwiftPM layout)", old_path.display());
        return Ok(old_path);
    }

    // Try PATH lookup (the `which` crate is available via the mlx feature).
    #[cfg(feature = "mlx")]
    {
        if let Some(p) = which::which("kchat-mlx-server").ok() {
            tracing::info!("Found kchat-mlx-server on PATH: {}", p.display());
            return Ok(p);
        }
    }

    Err(
        "kchat-mlx-server binary not found. Set KCHAT_MLX_SERVER env var, \
         or build it via: cd swift/kchat-mlx-server && swift build -c release"
            .to_string(),
    )
}

/// Wait for the MLX server to respond to `/health` with a 200 status.
fn wait_for_ready(base_url: &str) -> Result<(), String> {
    let url = format!("{}/health", base_url);
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| format!("failed to build health-check client: {}", e))?;

    let deadline = Instant::now() + Duration::from_secs(READY_TIMEOUT_SECS);
    let mut attempt = 0u32;

    while Instant::now() < deadline {
        match client.get(&url).send() {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => {
                attempt += 1;
                std::thread::sleep(Duration::from_millis(500));
                if attempt % 4 == 0 {
                    tracing::info!("Waiting for kchat-mlx-server to start... (attempt {})", attempt);
                }
            }
        }
    }

    Err(format!(
        "kchat-mlx-server did not become ready within {} seconds",
        READY_TIMEOUT_SECS
    ))
}

// ---------------------------------------------------------------------------
// Text chunking for simulated streaming
// ---------------------------------------------------------------------------

/// Split text into word-sized chunks (preserving whitespace) for simulated
/// streaming. Each chunk is one word plus any trailing whitespace, or a
/// standalone whitespace run.
fn split_into_word_chunks(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        // Emit a chunk at each word boundary (whitespace after non-whitespace).
        if ch.is_whitespace() && current.len() > 1 {
            chunks.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    chunks
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_backend_is_empty() {
        let backend = MlxBackend::new();
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_unload_when_not_loaded_is_noop() {
        let backend = MlxBackend::new();
        assert!(backend.unload().is_ok());
        assert!(!backend.is_loaded());
    }

    #[test]
    fn test_backend_type_is_mlx() {
        let backend = MlxBackend::new();
        assert_eq!(backend.backend_type(), BackendType::Mlx);
    }

    #[test]
    fn test_generate_without_load_returns_not_loaded() {
        let backend = MlxBackend::new();
        let config = GenerationConfig::default();
        let result = backend.generate("test", &config);
        assert!(matches!(result, Err(BackendError::NotLoaded)));
    }

    #[test]
    fn test_split_into_word_chunks() {
        let chunks = split_into_word_chunks("Hello world! How are you?");
        assert!(!chunks.is_empty());
        // Each chunk should end with whitespace (except possibly the last).
        let combined: String = chunks.concat();
        assert_eq!(combined, "Hello world! How are you?");
    }

    #[test]
    fn test_split_empty_text() {
        let chunks = split_into_word_chunks("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_split_single_word() {
        let chunks = split_into_word_chunks("Hello");
        assert_eq!(chunks, vec!["Hello"]);
    }

    /// Integration test: spawns the Swift kchat-mlx-server against the
    /// bonsai-1.7b-mlx-1bit pack and runs a small completion. Ignored by
    /// default since it requires the Swift binary + model pack to be present.
    #[test]
    #[ignore]
    fn integration_mlx_backend_completion() {
        let model_path = std::env::var("KCHAT_MLX_TEST_MODEL").unwrap_or_else(|_| {
            "/Users/Ken/workspaces/kocal/kchat-ai-runtime/manifest/packs/bonsai-1.7b-mlx-1bit"
                .to_string()
        });

        let backend = MlxBackend::new();
        let config = BackendConfig {
            backend_type: BackendType::Mlx,
            model_pack_id: "bonsai-1.7b-mlx-1bit".to_string(),
            model_path: model_path.clone(),
            gpu_layers: -1,
            context_size: 2048,
            threads: 4,
            batch_size: 512,
            model_quality: crate::backend::ModelQuality::Fast,
        };

        backend.load(&config).expect("failed to load MLX model");

        let gen_config = GenerationConfig {
            max_tokens: 16,
            temperature: 0.3,
            ..Default::default()
        };

        let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n";
        let result = backend.generate(prompt, &gen_config).expect("generation failed");
        assert!(!result.text.is_empty());
        assert!(result.completion_tokens > 0);
        assert_eq!(result.backend, "mlx");

        backend.unload().expect("failed to unload");
    }
}

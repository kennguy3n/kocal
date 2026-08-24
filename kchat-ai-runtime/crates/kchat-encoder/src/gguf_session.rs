//! GGUF encoder backend — uses llama.cpp for embedding extraction.
//!
//! This backend loads a GGUF model (e.g., mmBERT-small-Q4_K_M.gguf) via
//! llama-server's embedding endpoint and applies task-specific heads
//! (safety, embedding, rerank) that are loaded separately from the GGUF model.
//!
//! The GGUF model provides the encoder backbone (hidden states), while the
//! task heads are stored as safetensors weights and applied in Rust.
//!
//! # Architecture
//!
//! ```text
//! llama-server (GGUF)          Rust runtime
//! ┌─────────────────┐          ┌──────────────────────┐
//! │ mmBERT-small     │          │ classifier_weights    │
//! │ Q4_K_M.gguf      │          │ (safetensors)         │
//! │                  │          │                       │
//! │ /embedding       │◄────────►│ SafetyHead            │
//! │ endpoint         │  HTTP    │ EmbedHead             │
//! │ → 384-dim vector │          │ RerankHead            │
//! └─────────────────┘          └──────────────────────┘
//! ```
//!
//! # Loading
//!
//! The backend starts a llama-server subprocess with `--embedding` flag,
//! then sends tokenized text to the `/embedding` endpoint and receives
//! 384-dim hidden state vectors. These vectors are then processed by
//! the task heads (loaded from safetensors) to produce safety logits,
//! embeddings, or rerank scores.

use crate::{EncoderError, EncoderResult, EMBEDDING_DIM, MAX_SEQ_LENGTH, NUM_SAFETY_CATEGORIES};
use parking_lot::Mutex;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

/// Default port for the embedding server.
const DEFAULT_PORT: u16 = 18889;

/// Default timeout for server startup (seconds).
const SERVER_STARTUP_TIMEOUT: u64 = 30;

/// GGUF encoder backend using llama-server's embedding endpoint.
pub struct GgufEncoderSession {
    /// Server child process
    server: Mutex<Option<Child>>,
    /// HTTP port
    port: u16,
    /// Model path
    model_path: String,
    /// Classifier head weights (safety, embedding, rerank)
    heads: Arc<ClassifierHeads>,
    /// Model name
    model_name: String,
    /// Quantization level
    quantization: String,
    /// Number of intra-op threads for llama-server
    intra_threads: usize,
}

/// Classifier head weights loaded from safetensors.
///
/// Contains the safety, embedding, and rerank head weights that are
/// applied to the GGUF model's hidden state output.
#[derive(Debug, Clone)]
pub struct ClassifierHeads {
    /// Safety head: Linear(hidden, hidden) -> GELU -> Linear(hidden, 17)
    pub safety_layer1_weight: Vec<f32>,
    pub safety_layer1_bias: Vec<f32>,
    pub safety_layer2_weight: Vec<f32>,
    pub safety_layer2_bias: Vec<f32>,
    /// Embedding head: Linear(hidden, 384) -> GELU -> L2Norm
    pub embed_layer1_weight: Vec<f32>,
    pub embed_layer1_bias: Vec<f32>,
    /// Rerank head: Linear(hidden, hidden/2) -> GELU -> Linear(hidden/2, 1) -> Sigmoid
    pub rerank_layer1_weight: Vec<f32>,
    pub rerank_layer1_bias: Vec<f32>,
    pub rerank_layer2_weight: Vec<f32>,
    pub rerank_layer2_bias: Vec<f32>,
    /// Hidden size (384 for mmBERT-small)
    pub hidden_size: usize,
    /// Embedding dimension (384)
    pub embedding_dim: usize,
    /// Number of safety classes (17)
    pub num_safety_classes: usize,
}

impl ClassifierHeads {
    /// Load classifier heads from a safetensors file.
    ///
    /// The file should contain the following keys:
    /// - `safety_head.0.weight`, `safety_head.0.bias`
    /// - `safety_head.2.weight`, `safety_head.2.bias`
    /// - `embedding_head.0.weight`, `embedding_head.0.bias`
    /// - `rerank_head.0.weight`, `rerank_head.0.bias`
    /// - `rerank_head.2.weight`, `rerank_head.2.bias`
    pub fn from_safetensors(path: &str) -> EncoderResult<Self> {
        // Safetensors loading without the safetensors crate:
        // We read the file manually. Safetensors format:
        // 8 bytes: header length (u64 LE)
        // N bytes: JSON header with tensor names and metadata
        // rest: tensor data
        let data = std::fs::read(path)
            .map_err(|e| EncoderError::SessionError(format!("read safetensors: {e}")))?;

        if data.len() < 8 {
            return Err(EncoderError::SessionError("safetensors too short".into()));
        }

        let header_len = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]) as usize;

        if data.len() < 8 + header_len {
            return Err(EncoderError::SessionError("safetensors header truncated".into()));
        }

        let header_json = &data[8..8 + header_len];
        let header: serde_json::Value = serde_json::from_slice(header_json)
            .map_err(|e| EncoderError::SessionError(format!("parse safetensors header: {e}")))?;

        // Extract tensors
        let mut tensors: std::collections::HashMap<String, (Vec<usize>, &[u8])> = std::collections::HashMap::new();

        if let Some(obj) = header.as_object() {
            for (name, info) in obj {
                if name == "__metadata__" {
                    continue;
                }
                if let Some(dtype) = info.get("dtype").and_then(|v| v.as_str()) {
                    if dtype != "F32" {
                        return Err(EncoderError::SessionError(format!(
                            "unsupported dtype for {name}: {dtype} (expected F32)"
                        )));
                    }
                }
                let data_offsets = info.get("data_offsets").and_then(|v| v.as_array());
                let shape = info.get("shape").and_then(|v| v.as_array());
                if let (Some(offsets), Some(shape)) = (data_offsets, shape) {
                    let start = offsets[0].as_u64().unwrap_or(0) as usize + 8 + header_len;
                    let end = offsets[1].as_u64().unwrap_or(0) as usize + 8 + header_len;
                    let shape_vec: Vec<usize> = shape
                        .iter()
                        .filter_map(|v| v.as_u64().map(|x| x as usize))
                        .collect();
                    let tensor_data = &data[start..end];
                    tensors.insert(name.clone(), (shape_vec, tensor_data));
                }
            }
        }

        // Helper to extract a tensor as Vec<f32>
        let extract = |name: &str| -> EncoderResult<(Vec<usize>, Vec<f32>)> {
            let (shape, raw) = tensors.get(name).ok_or_else(|| {
                EncoderError::SessionError(format!("tensor not found: {name}"))
            })?;
            let count = raw.len() / 4;
            let mut values = Vec::with_capacity(count);
            for i in 0..count {
                let bytes = [raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]];
                values.push(f32::from_le_bytes(bytes));
            }
            Ok((shape.clone(), values))
        };

        let (_, safety_l1_w) = extract("safety_head.0.weight")?;
        let (_, safety_l1_b) = extract("safety_head.0.bias")?;
        let (_, safety_l2_w) = extract("safety_head.2.weight")?;
        let (_, safety_l2_b) = extract("safety_head.2.bias")?;
        let (_, embed_l1_w) = extract("embedding_head.0.weight")?;
        let (_, embed_l1_b) = extract("embedding_head.0.bias")?;
        let (_, rerank_l1_w) = extract("rerank_head.0.weight")?;
        let (_, rerank_l1_b) = extract("rerank_head.0.bias")?;
        let (_, rerank_l2_w) = extract("rerank_head.2.weight")?;
        let (_, rerank_l2_b) = extract("rerank_head.2.bias")?;

        // Infer hidden size from safety_layer1_bias length
        let hidden_size = safety_l1_b.len();
        let embedding_dim = embed_l1_b.len();
        let num_safety_classes = safety_l2_b.len();

        Ok(Self {
            safety_layer1_weight: safety_l1_w,
            safety_layer1_bias: safety_l1_b,
            safety_layer2_weight: safety_l2_w,
            safety_layer2_bias: safety_l2_b,
            embed_layer1_weight: embed_l1_w,
            embed_layer1_bias: embed_l1_b,
            rerank_layer1_weight: rerank_l1_w,
            rerank_layer1_bias: rerank_l1_b,
            rerank_layer2_weight: rerank_l2_w,
            rerank_layer2_bias: rerank_l2_b,
            hidden_size,
            embedding_dim,
            num_safety_classes,
        })
    }

    /// Create mock heads for testing.
    pub fn mock() -> Self {
        let hidden = EMBEDDING_DIM;
        let half = hidden / 2;
        Self {
            safety_layer1_weight: vec![0.0; hidden * hidden],
            safety_layer1_bias: vec![0.0; hidden],
            safety_layer2_weight: vec![0.0; hidden * NUM_SAFETY_CATEGORIES],
            safety_layer2_bias: vec![0.0; NUM_SAFETY_CATEGORIES],
            embed_layer1_weight: vec![0.0; hidden * EMBEDDING_DIM],
            embed_layer1_bias: vec![0.0; EMBEDDING_DIM],
            rerank_layer1_weight: vec![0.0; hidden * half],
            rerank_layer1_bias: vec![0.0; half],
            rerank_layer2_weight: vec![0.0; half],
            rerank_layer2_bias: vec![0.0; 1],
            hidden_size: hidden,
            embedding_dim: EMBEDDING_DIM,
            num_safety_classes: NUM_SAFETY_CATEGORIES,
        }
    }

    /// Apply the safety head to a hidden state vector.
    ///
    /// Returns logits for each of the 17 safety categories.
    pub fn apply_safety(&self, hidden: &[f32]) -> Vec<f32> {
        // Layer 1: Linear(hidden, hidden) + GELU
        let mut layer1 = vec![0.0f32; self.hidden_size];
        for i in 0..self.hidden_size {
            let mut sum = self.safety_layer1_bias[i];
            for j in 0..self.hidden_size.min(hidden.len()) {
                // Weight is stored as [out, in], so index = i * hidden + j
                sum += hidden[j] * self.safety_layer1_weight[i * self.hidden_size + j];
            }
            layer1[i] = gelu(sum);
        }

        // Layer 2: Linear(hidden, num_classes)
        let mut logits = vec![0.0f32; self.num_safety_classes];
        for i in 0..self.num_safety_classes {
            let mut sum = self.safety_layer2_bias[i];
            for j in 0..self.hidden_size {
                sum += layer1[j] * self.safety_layer2_weight[i * self.hidden_size + j];
            }
            logits[i] = sum;
        }

        logits
    }

    /// Apply the embedding head to a hidden state vector.
    ///
    /// Returns an L2-normalized embedding vector.
    pub fn apply_embedding(&self, hidden: &[f32]) -> Vec<f32> {
        // Layer 1: Linear(hidden, embedding_dim) + GELU
        let mut embedding = vec![0.0f32; self.embedding_dim];
        for i in 0..self.embedding_dim {
            let mut sum = self.embed_layer1_bias[i];
            for j in 0..self.hidden_size.min(hidden.len()) {
                sum += hidden[j] * self.embed_layer1_weight[i * self.hidden_size + j];
            }
            embedding[i] = gelu(sum);
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut embedding {
                *x /= norm;
            }
        }

        embedding
    }

    /// Apply the rerank head to a hidden state vector.
    ///
    /// Returns a relevance score (0.0 to 1.0 after sigmoid).
    pub fn apply_rerank(&self, hidden: &[f32]) -> f32 {
        let half = self.hidden_size / 2;

        // Layer 1: Linear(hidden, hidden/2) + GELU
        let mut layer1 = vec![0.0f32; half];
        for i in 0..half {
            let mut sum = self.rerank_layer1_bias[i];
            for j in 0..self.hidden_size.min(hidden.len()) {
                sum += hidden[j] * self.rerank_layer1_weight[i * self.hidden_size + j];
            }
            layer1[i] = gelu(sum);
        }

        // Layer 2: Linear(hidden/2, 1) + Sigmoid
        let mut logit = self.rerank_layer2_bias[0];
        for j in 0..half {
            logit += layer1[j] * self.rerank_layer2_weight[j];
        }

        // Sigmoid
        1.0 / (1.0 + (-logit).exp())
    }
}

/// GELU activation function.
fn gelu(x: f32) -> f32 {
    // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    // Approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + 0.044715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Softmax over a slice of logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exp.iter().map(|e| e / sum).collect()
}

impl GgufEncoderSession {
    /// Create a new GGUF encoder session.
    ///
    /// This starts a llama-server subprocess with `--embedding` flag.
    ///
    /// # Arguments
    /// * `model_path` - Path to the GGUF model file
    /// * `heads_path` - Path to the classifier heads safetensors file
    /// * `port` - HTTP port for llama-server (0 = auto-select)
    /// * `intra_threads` - Number of threads for llama-server
    pub fn new(
        model_path: &str,
        heads_path: &str,
        port: u16,
        intra_threads: usize,
    ) -> EncoderResult<Self> {
        let heads = ClassifierHeads::from_safetensors(heads_path)?;

        let port = if port == 0 { DEFAULT_PORT } else { port };
        let model_name = std::path::Path::new(model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("mmbert-safety")
            .to_string();

        let session = Self {
            server: Mutex::new(None),
            port,
            model_path: model_path.to_string(),
            heads: Arc::new(heads),
            model_name,
            quantization: "Q4_K_M".into(),
            intra_threads,
        };

        session.start_server()?;
        Ok(session)
    }

    /// Start the llama-server subprocess.
    fn start_server(&self) -> EncoderResult<()> {
        let mut cmd = Command::new("llama-server");
        cmd.arg("-m").arg(&self.model_path)
            .arg("--embedding")
            .arg("--port").arg(self.port.to_string())
            .arg("-t").arg(self.intra_threads.to_string())
            .arg("-c").arg("512")  // context size for encoder
            .arg("--log-disable")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn()
            .map_err(|e| EncoderError::SessionError(format!("start llama-server: {e}")))?;

        // Wait for server to be ready by checking the health endpoint
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(SERVER_STARTUP_TIMEOUT);

        loop {
            if start.elapsed() > timeout {
                // Kill the child if it's still running
                child.kill().ok();
                return Err(EncoderError::SessionError(
                    "llama-server startup timeout".into()
                ));
            }

            // Check if process exited
            match child.try_wait() {
                Ok(Some(status)) => {
                    return Err(EncoderError::SessionError(format!(
                        "llama-server exited early with status {status}"
                    )));
                }
                Ok(None) => {} // still running
                Err(e) => {
                    return Err(EncoderError::SessionError(format!(
                        "wait for llama-server: {e}"
                    )));
                }
            }

            // Try to connect to the health endpoint
            if self.check_health() {
                *self.server.lock() = Some(child);
                tracing::info!(
                    "GGUF encoder server started on port {} (model: {})",
                    self.port, self.model_path
                );
                return Ok(());
            }

            std::thread::sleep(Duration::from_millis(500));
        }
    }

    /// Check if the server is healthy.
    fn check_health(&self) -> bool {
        let url = format!("http://127.0.0.1:{}/health", self.port);
        reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .ok()
            .and_then(|c| c.get(&url).send().ok())
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Get the embedding (hidden state) for a text via llama-server.
    ///
    /// This calls the `/embedding` endpoint and returns the raw hidden state
    /// vector (384-dim for mmBERT-small).
    fn get_embedding(&self, text: &str) -> EncoderResult<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(EncoderError::InferenceFailed(
                "empty input text — cannot encode".into(),
            ));
        }

        let url = format!("http://127.0.0.1:{}/embedding", self.port);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| EncoderError::InferenceFailed(format!("HTTP client: {e}")))?;

        let body = serde_json::json!({
            "content": text,
        });

        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| EncoderError::InferenceFailed(format!("embedding request: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            return Err(EncoderError::InferenceFailed(format!(
                "embedding request failed: {status} - {text}"
            )));
        }

        let resp_json: serde_json::Value = resp
            .json()
            .map_err(|e| EncoderError::InferenceFailed(format!("parse response: {e}")))?;

        // llama-server embedding response format:
        // {"embedding": [0.1, 0.2, ...]}
        let embedding = resp_json
            .get("embedding")
            .and_then(|v| v.as_array())
            .ok_or_else(|| EncoderError::InferenceFailed("missing embedding in response".into()))?;

        let vec: Vec<f32> = embedding
            .iter()
            .filter_map(|v| v.as_f64().map(|x| x as f32))
            .collect();

        if vec.len() != self.heads.hidden_size {
            return Err(EncoderError::DimensionMismatch {
                expected: self.heads.hidden_size,
                actual: vec.len(),
            });
        }

        Ok(vec)
    }

    /// Classify text into one of 17 safety categories.
    pub fn classify(&self, text: &str) -> EncoderResult<crate::SafetyVerdict> {
        let hidden = self.get_embedding(text)?;
        let logits = self.heads.apply_safety(&hidden);
        let probs = softmax(&logits);

        let (best_idx, best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, p)| (i as u32, *p as f64))
            .unwrap_or((0, 0.0));

        Ok(crate::SafetyVerdict {
            category: best_idx,
            confidence: best_prob,
        })
    }

    /// Get a 384-dim L2-normalized embedding for a text.
    pub fn embed(&self, text: &str) -> EncoderResult<Vec<f32>> {
        let hidden = self.get_embedding(text)?;
        Ok(self.heads.apply_embedding(&hidden))
    }

    /// Score a query-document pair for reranking.
    ///
    /// Uses pair encoding: "query [SEP] document"
    pub fn rerank(&self, query: &str, document: &str) -> EncoderResult<f64> {
        let text = format!("{query} [SEP] {document}");
        let hidden = self.get_embedding(&text)?;
        Ok(self.heads.apply_rerank(&hidden) as f64)
    }

    /// Batch rerank: score multiple documents against a query.
    pub fn rerank_batch(&self, query: &str, documents: &[String]) -> EncoderResult<Vec<f64>> {
        documents
            .iter()
            .map(|doc| self.rerank(query, doc))
            .collect()
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.heads.embedding_dim
    }

    /// Get the quantization level.
    pub fn quantization(&self) -> &str {
        &self.quantization
    }

    /// Get the max sequence length.
    pub fn max_length(&self) -> usize {
        MAX_SEQ_LENGTH
    }

    /// Get the configured intra-op thread count.
    pub fn intra_threads(&self) -> usize {
        self.intra_threads
    }
}

impl Drop for GgufEncoderSession {
    fn drop(&mut self) {
        if let Some(mut child) = self.server.lock().take() {
            tracing::info!("Stopping GGUF encoder server (port {})", self.port);
            child.kill().ok();
            child.wait().ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gelu() {
        assert!((gelu(0.0) - 0.0).abs() < 0.01);
        assert!(gelu(1.0) > 0.7);
        assert!(gelu(-1.0) < -0.1);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 0.01);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_classifier_heads_mock() {
        let heads = ClassifierHeads::mock();
        let hidden = vec![0.5; heads.hidden_size];

        // Safety head should produce num_safety_classes logits
        let logits = heads.apply_safety(&hidden);
        assert_eq!(logits.len(), NUM_SAFETY_CATEGORIES);

        // Embedding head should produce embedding_dim values
        let embedding = heads.apply_embedding(&hidden);
        assert_eq!(embedding.len(), EMBEDDING_DIM);

        // Rerank head should produce a single score
        let score = heads.apply_rerank(&hidden);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_classifier_heads_safety_dimensions() {
        let heads = ClassifierHeads::mock();
        assert_eq!(heads.hidden_size, EMBEDDING_DIM);
        assert_eq!(heads.embedding_dim, EMBEDDING_DIM);
        assert_eq!(heads.num_safety_classes, NUM_SAFETY_CATEGORIES);
    }
}

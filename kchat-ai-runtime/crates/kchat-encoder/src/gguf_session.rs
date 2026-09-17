//! GGUF encoder backend — in-process llama.cpp embedding extraction.
//!
//! This backend loads a GGUF encoder model (e.g., mmBERT-small-Q4_K_M.gguf)
//! directly via `llama-cpp-2` and extracts pooled hidden states in-process —
//! no subprocess, no HTTP, no port binding. This is the only path that can
//! ship on iOS/Android, where spawning a `llama-server` child process is not
//! permitted.
//!
//! The GGUF model provides the encoder backbone (hidden states), while the
//! task heads are stored as safetensors weights and applied in Rust.
//!
//! # Architecture
//!
//! ```text
//! llama.cpp (in-process)       Rust runtime
//! ┌─────────────────┐          ┌──────────────────────┐
//! │ mmBERT-small     │          │ classifier_weights    │
//! │ Q4_K_M.gguf      │          │ (safetensors)         │
//! │                  │          │                       │
//! │ decode → pooled  │─────────►│ SafetyHead            │
//! │ hidden state     │  memcpy  │ EmbedHead             │
//! │ (384-dim)        │          │ RerankHead            │
//! └─────────────────┘          └──────────────────────┘
//! ```

use crate::{EncoderError, EncoderResult, EMBEDDING_DIM, MAX_SEQ_LENGTH, NUM_SAFETY_CATEGORIES};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use parking_lot::Mutex;
use std::num::NonZeroU32;
use std::sync::{Arc, OnceLock};

/// Global llama.cpp backend (process-wide; init is idempotent).
static LLAMA_BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();

fn llama_backend() -> EncoderResult<&'static LlamaBackend> {
    LLAMA_BACKEND
        .get_or_init(|| {
            LlamaBackend::init().map_err(|e| format!("failed to init llama.cpp backend: {e}"))
        })
        .as_ref()
        .map_err(|e| EncoderError::SessionError(e.clone()))
}

/// An embedding context whose `'static` lifetime is extended from the boxed
/// model stored in [`GgufEncoderSession`]. The session's `Drop` impl takes
/// this out of the `Mutex` before the `model` field drops, so the reference
/// can never dangle. All access is serialized by the mutex.
struct EncoderCtx {
    ctx: LlamaContext<'static>,
}

// SAFETY: `ctx` internally holds `&'static LlamaModel` pointing into the
// session's boxed model, which outlives it (dropped second). Access is
// serialized through `Mutex<Option<EncoderCtx>>`.
unsafe impl Send for EncoderCtx {}
unsafe impl Sync for EncoderCtx {}

/// GGUF encoder session — in-process llama.cpp embedding.
pub struct GgufEncoderSession {
    /// Embedding context. `Option` so `Drop` can take it before `model`.
    /// Field order matters: declared before `model` and explicitly dropped
    /// in `Drop::drop` — the context borrows the model internally.
    ctx: Mutex<Option<EncoderCtx>>,
    /// The loaded encoder model (boxed for a stable address).
    model: Box<LlamaModel>,
    /// Classifier head weights (safety, embedding, rerank)
    heads: Arc<ClassifierHeads>,
    /// Model name
    model_name: String,
    /// Quantization level
    quantization: String,
    /// Number of intra-op threads for llama.cpp
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

        let header_len_raw = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let header_len = usize::try_from(header_len_raw).map_err(|_| {
            EncoderError::SessionError("safetensors header length overflows usize".into())
        })?;

        if header_len > data.len().saturating_sub(8) {
            return Err(EncoderError::SessionError(
                "safetensors header truncated".into(),
            ));
        }

        let header_json = &data[8..8 + header_len];
        let header: serde_json::Value = serde_json::from_slice(header_json)
            .map_err(|e| EncoderError::SessionError(format!("parse safetensors header: {e}")))?;

        // Extract tensors
        let mut tensors: std::collections::HashMap<String, (Vec<usize>, &[u8])> =
            std::collections::HashMap::new();

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
                    if offsets.len() != 2 {
                        return Err(EncoderError::SessionError(format!(
                            "malformed data_offsets for {name}"
                        )));
                    }
                    let tensor_base = 8usize.checked_add(header_len).ok_or_else(|| {
                        EncoderError::SessionError("safetensors header overflow".into())
                    })?;
                    let Some(start) = offsets[0]
                        .as_u64()
                        .and_then(|o| usize::try_from(o).ok())
                        .and_then(|o| o.checked_add(tensor_base))
                    else {
                        return Err(EncoderError::SessionError(format!(
                            "malformed data_offsets for {name}"
                        )));
                    };
                    let Some(end) = offsets[1]
                        .as_u64()
                        .and_then(|o| usize::try_from(o).ok())
                        .and_then(|o| o.checked_add(tensor_base))
                    else {
                        return Err(EncoderError::SessionError(format!(
                            "malformed data_offsets for {name}"
                        )));
                    };
                    if start > end || end > data.len() {
                        return Err(EncoderError::SessionError(format!(
                            "data_offsets out of range for {name}"
                        )));
                    }
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
            let (shape, raw) = tensors
                .get(name)
                .ok_or_else(|| EncoderError::SessionError(format!("tensor not found: {name}")))?;
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
    /// Create a new GGUF encoder session — loads the model in-process.
    ///
    /// # Arguments
    /// * `model_path` - Path to the GGUF model file
    /// * `heads_path` - Path to the classifier heads safetensors file
    /// * `intra_threads` - Number of threads for llama.cpp
    pub fn new(model_path: &str, heads_path: &str, intra_threads: usize) -> EncoderResult<Self> {
        let heads = ClassifierHeads::from_safetensors(heads_path)?;

        let model_name = std::path::Path::new(model_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("mmbert-safety")
            .to_string();

        // Encoder models are small (~90-150MB) — CPU inference is a few ms
        // and GPU offload buys nothing at this size. n_ctx/n_batch are sized
        // to MAX_SEQ_LENGTH so a full-length input decodes in one shot.
        let model = LlamaModel::load_from_file(
            llama_backend()?,
            std::path::Path::new(model_path),
            &LlamaModelParams::default(),
        )
        .map_err(|e| EncoderError::SessionError(format!("load gguf encoder: {e}")))?;
        let model = Box::new(model);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(MAX_SEQ_LENGTH as u32))
            .with_n_batch(MAX_SEQ_LENGTH as u32)
            .with_n_ubatch(MAX_SEQ_LENGTH as u32)
            .with_n_threads(intra_threads.max(1) as i32)
            .with_n_threads_batch(intra_threads.max(1) as i32)
            .with_embeddings(true);
        let ctx = model
            .new_context(llama_backend()?, ctx_params)
            .map_err(|e| EncoderError::SessionError(format!("encoder ctx: {e}")))?;
        // SAFETY: the boxed model outlives this context — `Drop` takes the
        // context out of the mutex before the `model` field is dropped.
        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };

        tracing::info!("GGUF encoder loaded in-process (model: {})", model_name);

        Ok(Self {
            ctx: Mutex::new(Some(EncoderCtx { ctx })),
            model,
            heads: Arc::new(heads),
            model_name,
            quantization: "Q4_K_M".into(),
            intra_threads,
        })
    }

    /// Get the pooled hidden state for a text — in-process llama.cpp decode.
    fn get_embedding(&self, text: &str) -> EncoderResult<Vec<f32>> {
        if text.trim().is_empty() {
            return Err(EncoderError::InferenceFailed(
                "empty input text — cannot encode".into(),
            ));
        }

        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| EncoderError::InferenceFailed(format!("tokenize: {e}")))?;
        if tokens.is_empty() {
            return Err(EncoderError::InferenceFailed("empty token stream".into()));
        }
        // Truncate to the context window — encoder input is bounded.
        let tokens = &tokens[..tokens.len().min(MAX_SEQ_LENGTH)];

        let mut guard = self.ctx.lock();
        let enc = guard
            .as_mut()
            .ok_or_else(|| EncoderError::SessionError("encoder dropped".into()))?;
        let ctx = &mut enc.ctx;

        ctx.clear_kv_cache();
        let mut batch = LlamaBatch::new(tokens.len(), 1);
        for (i, &token) in tokens.iter().enumerate() {
            // Output flag on every position so the pooled embedding is
            // available regardless of the model's pooling strategy.
            batch
                .add(token, i as i32, &[0], true)
                .map_err(|e| EncoderError::InferenceFailed(format!("batch add: {e}")))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| EncoderError::InferenceFailed(format!("encode: {e}")))?;

        // Encoder models expose a pooled sequence embedding; decoder-family
        // GGUFs declare pooling=NONE — fall back to the last token's state
        // (standard last-token pooling for causal backbones).
        let vec: Vec<f32> = match ctx.embeddings_seq_ith(0) {
            Ok(hidden) => hidden.to_vec(),
            Err(_) => ctx
                .embeddings_ith(tokens.len() as i32 - 1)
                .map_err(|e| EncoderError::InferenceFailed(format!("read embedding: {e}")))?
                .to_vec(),
        };

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
        // Drop the context before the model field — it borrows the model.
        self.ctx.lock().take();
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

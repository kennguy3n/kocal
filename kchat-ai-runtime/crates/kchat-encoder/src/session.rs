//! Shared encoder session — loads ONNX model + tokenizer once.
//!
//! The session wraps an ONNX Runtime session for XLM-RoBERTa-base and
//! provides raw forward-pass methods used by the task heads.
//!
//! The ONNX model may export either:
//! - Task-specific outputs (`safety_logits`, `embedding`, `rerank_score`), or
//! - Raw `last_hidden_state` (the heads then apply their own projections).
//!
//! The session tries named outputs first and falls back to hidden states.

use parking_lot::Mutex;

use crate::{EncoderError, EncoderResult, Quantization};

/// Result of a forward pass through the encoder.
///
/// Contains the raw hidden states and attention mask, plus any task-specific
/// outputs that the ONNX model may have produced directly.
pub struct ForwardOutput {
    /// Hidden states of shape [1, seq_len, 768] (always present).
    pub hidden: ndarray::Array3<f32>,
    /// Attention mask used for this forward pass (1 = real token, 0 = padding).
    pub attention_mask: Vec<i64>,
    /// Safety logits from the model's classification head, if exported.
    pub safety_logits: Option<Vec<f32>>,
    /// L2-normalized embedding from the model's embedding head, if exported.
    pub embedding: Option<Vec<f32>>,
    /// Rerank relevance score from the model's rerank head, if exported.
    pub rerank_score: Option<f32>,
}

/// Shared ONNX encoder session for XLM-RoBERTa-base.
///
/// Load once, share across SafetyHead, EmbedHead, and RerankHead via `Arc`.
#[cfg(feature = "onnx-runtime")]
pub struct EncoderSession {
    session: Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    max_length: usize,
    model_name: String,
    quantization: Quantization,
    intra_threads: usize,
}

#[cfg(feature = "onnx-runtime")]
impl EncoderSession {
    /// Create a new encoder session from an ONNX model file and tokenizer.
    ///
    /// `intra_threads` controls ONNX Runtime intra-op parallelism. Use 2 for
    /// low-tier devices, 3 for medium, 4+ for high-tier.
    ///
    /// EP selection is driven by [`kchat_core::ep::ExecutionProviderSelector`]:
    /// the host platform's preferred accelerator EP (CoreML on Apple,
    /// DirectML on Windows, NNAPI on Android) is attempted first, with
    /// automatic CPU fallback if the accelerator fails to register.
    pub fn new(
        model_path: &str,
        tokenizer_path: &str,
        quantization: Quantization,
        intra_threads: usize,
    ) -> EncoderResult<Self> {
        let mut builder = ort::session::Session::builder()
            .map_err(|e| EncoderError::SessionError(format!("builder: {e}")))?;
        builder = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| EncoderError::SessionError(format!("optimization: {e}")))?;
        builder = builder
            .with_intra_threads(intra_threads)
            .map_err(|e| EncoderError::SessionError(format!("threads: {e}")))?;

        // Apply platform-aware EP selection from kchat-core.
        let ep_eps = build_ort_eps_for_host();
        if !ep_eps.is_empty() {
            builder = builder
                .with_execution_providers(&ep_eps)
                .map_err(|e| EncoderError::SessionError(format!("ep selection: {e}")))?;
        }

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| EncoderError::SessionError(format!("load model: {e}")))?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EncoderError::TokenizerError(format!("tokenizer: {e}")))?;
        tokenizer.with_truncation(Some(tokenizers::TruncationParams {
                max_length: crate::MAX_SEQ_LENGTH as usize,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                stride: 0,
                direction: tokenizers::TruncationDirection::Right,
            }))
            .map_err(|e| EncoderError::TokenizerError(format!("truncation config: {e}")))?;

        let model_name = match quantization {
            Quantization::Int8 => "kchat-encoder-int8",
            Quantization::Int4 => "kchat-encoder-int4",
        };

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_length: crate::MAX_SEQ_LENGTH,
            model_name: model_name.into(),
            quantization,
            intra_threads,
        })
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get the quantization level.
    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        crate::EMBEDDING_DIM
    }

    /// Get the max sequence length.
    pub fn max_length(&self) -> usize {
        self.max_length
    }

    /// Get the configured intra-op thread count.
    pub fn intra_threads(&self) -> usize {
        self.intra_threads
    }

    /// Run a forward pass on a single text.
    ///
    /// Returns a [`ForwardOutput`] containing hidden states, attention mask,
    /// and any task-specific outputs the ONNX model produces directly.
    pub fn forward(&self, text: &str) -> EncoderResult<ForwardOutput> {
        if text.trim().is_empty() {
            return Err(EncoderError::InferenceFailed(
                "empty input text — cannot encode".into(),
            ));
        }
        self.run_forward(text, None)
    }

    /// Run a forward pass on a query-document pair for reranking.
    ///
    /// Uses the tokenizer's native pair encoding (segment IDs) rather than
    /// string concatenation, ensuring correct special-token insertion for
    /// XLM-RoBERTa (which uses `</s>` as separator, not `[SEP]`).
    pub fn forward_pair(&self, query: &str, document: &str) -> EncoderResult<ForwardOutput> {
        if query.trim().is_empty() || document.trim().is_empty() {
            return Err(EncoderError::InferenceFailed(
                "empty query or document — cannot encode pair".into(),
            ));
        }
        self.run_forward(query, Some(document))
    }

    /// Run a batched forward pass on multiple query-document pairs for reranking.
    ///
    /// This is more efficient than calling `forward_pair` in a loop because it
    /// runs a single ONNX session call with all pairs padded to the same length.
    /// Returns a vector of rerank scores, one per pair.
    pub fn forward_pair_batch(
        &self,
        query: &str,
        documents: &[String],
    ) -> EncoderResult<Vec<f64>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        use ndarray::Array2;

        // Tokenize all pairs
        let mut all_input_ids: Vec<Vec<i64>> = Vec::with_capacity(documents.len());
        let mut all_attention_masks: Vec<Vec<i64>> = Vec::with_capacity(documents.len());
        let mut max_seq = 0usize;

        for doc in documents {
            if query.trim().is_empty() || doc.trim().is_empty() {
                // Use a minimal valid sequence for empty inputs (will produce 0.0 score)
                all_input_ids.push(vec![0; 2]);
                all_attention_masks.push(vec![0; 2]);
                max_seq = max_seq.max(2);
                continue;
            }
            let encoding = self
                .tokenizer
                .encode((query.to_string(), doc.to_string()), true)
                .map_err(|e| EncoderError::TokenizerError(format!("encode pair batch: {e}")))?;

            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let seq_len = ids.len();
            let ids_vec: Vec<i64> = ids.iter().map(|&v| v as i64).collect();
            let mask_vec: Vec<i64> = mask.iter().map(|&v| v as i64).collect();
            max_seq = max_seq.max(seq_len);
            all_input_ids.push(ids_vec);
            all_attention_masks.push(mask_vec);
        }

        // Pad all sequences to max_seq
        for (ids, mask) in all_input_ids.iter_mut().zip(all_attention_masks.iter_mut()) {
            while ids.len() < max_seq {
                ids.push(0); // pad token
                mask.push(0); // pad mask
            }
        }

        // Create batched tensors [batch, seq_len]
        let batch_size = documents.len();
        let flat_ids: Vec<i64> = all_input_ids.into_iter().flatten().collect();
        let flat_masks: Vec<i64> = all_attention_masks.into_iter().flatten().collect();

        let ids_arr = Array2::from_shape_vec((batch_size, max_seq), flat_ids)
            .map_err(|e| EncoderError::InferenceFailed(format!("batch input_ids array: {e}")))?;
        let mask_arr = Array2::from_shape_vec((batch_size, max_seq), flat_masks)
            .map_err(|e| EncoderError::InferenceFailed(format!("batch attention array: {e}")))?;

        let ids_tensor = ort::value::Tensor::from_array(ids_arr)
            .map_err(|e| EncoderError::InferenceFailed(format!("batch input_ids tensor: {e}")))?;
        let mask_tensor = ort::value::Tensor::from_array(mask_arr)
            .map_err(|e| EncoderError::InferenceFailed(format!("batch attention tensor: {e}")))?;

        let inputs = ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        };

        let mut session = self.session.lock();
        let outputs = session
            .run(inputs)
            .map_err(|e| EncoderError::InferenceFailed(format!("batch run: {e}")))?;

        // Extract rerank scores — try named output first
        if let Some(value) = outputs.get("rerank_score") {
            if let Ok((_shape, data)) = value.try_extract_tensor::<f32>() {
                return Ok(data.iter().map(|&v| v as f64).collect());
            }
        }

        // Fall back: extract hidden states and use CLS token first element as logit
        let hidden = self.extract_hidden_states(&outputs)?;
        let scores: Vec<f64> = (0..batch_size)
            .map(|i| hidden[[i, 0, 0]] as f64)
            .collect();
        Ok(scores)
    }

    /// Internal: tokenize (single or pair), run ONNX, extract outputs.
    fn run_forward(&self, text: &str, pair: Option<&str>) -> EncoderResult<ForwardOutput> {
        use ndarray::Array2;

        let encoding = if let Some(doc) = pair {
            self.tokenizer
                .encode((text.to_string(), doc.to_string()), true)
                .map_err(|e| EncoderError::TokenizerError(format!("encode pair: {e}")))?
        } else {
            self.tokenizer
                .encode(text, true)
                .map_err(|e| EncoderError::TokenizerError(format!("encode: {e}")))?
        };

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Truncation is now handled by the tokenizer itself (configured in `new`),
        // so the encoding is already <= max_length with proper special tokens.
        let seq_len = input_ids.len();
        let input_ids = &input_ids[..seq_len];
        let attention_mask = &attention_mask[..seq_len];

        let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids.iter().map(|&v| v as i64).collect())
            .map_err(|e| EncoderError::InferenceFailed(format!("input_ids array: {e}")))?;
        let attention_arr = Array2::from_shape_vec((1, seq_len), attention_mask.iter().map(|&v| v as i64).collect())
            .map_err(|e| EncoderError::InferenceFailed(format!("attention array: {e}")))?;

        let input_ids_tensor = ort::value::Tensor::from_array(input_ids_arr)
            .map_err(|e| EncoderError::InferenceFailed(format!("input_ids tensor: {e}")))?;
        let attention_tensor = ort::value::Tensor::from_array(attention_arr)
            .map_err(|e| EncoderError::InferenceFailed(format!("attention tensor: {e}")))?;

        let inputs = ort::inputs! {
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_tensor,
        };

        let mut session = self.session.lock();
        let outputs = session
            .run(inputs)
            .map_err(|e| EncoderError::InferenceFailed(format!("run: {e}")))?;

        let attention_mask_vec: Vec<i64> = attention_mask.iter().map(|&v| v as i64).collect();

        // Try to extract task-specific named outputs from the ONNX model.
        // The model may export any subset of: safety_logits, embedding, rerank_score,
        // last_hidden_state. We try each by name and fall back gracefully.
        let safety_logits = self.try_extract_named_floats(&outputs, "safety_logits");
        let embedding = self.try_extract_named_floats(&outputs, "embedding");
        let rerank_score = self
            .try_extract_named_floats(&outputs, "rerank_score")
            .map(|v| v.first().copied().unwrap_or(0.0));

        // Only extract hidden states if at least one head needs the fallback path.
        // When the ONNX model exports all task-specific outputs directly, we skip
        // the ~600KB hidden state copy.
        let need_hidden = safety_logits.is_none() || embedding.is_none() || rerank_score.is_none();
        let hidden = if need_hidden {
            self.extract_hidden_states(&outputs)?
        } else {
            // Dummy empty hidden state — won't be used by any head.
            ndarray::Array3::from_shape_vec((1, 1, crate::EMBEDDING_DIM), vec![0.0; crate::EMBEDDING_DIM])
                .map_err(|e| EncoderError::InferenceFailed(format!("dummy hidden: {e}")))?
        };

        Ok(ForwardOutput {
            hidden,
            attention_mask: attention_mask_vec,
            safety_logits,
            embedding,
            rerank_score,
        })
    }

    /// Try to extract a named output as a flat Vec<f32>.
    fn try_extract_named_floats(
        &self,
        outputs: &ort::session::SessionOutputs,
        name: &str,
    ) -> Option<Vec<f32>> {
        let value = outputs.get(name)?;
        let (_shape, data) = value.try_extract_tensor::<f32>().ok()?;
        Some(data.to_vec())
    }

    /// Extract hidden states from model outputs.
    ///
    /// Tries "last_hidden_state" named output first, then falls back to
    /// the first output (index 0).
    fn extract_hidden_states(
        &self,
        outputs: &ort::session::SessionOutputs,
    ) -> EncoderResult<ndarray::Array3<f32>> {
        // Try named "last_hidden_state" output first.
        if let Some(value) = outputs.get("last_hidden_state") {
            if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
                return self.reshape_hidden(&shape, &data);
            }
        }

        // Fall back to first output (index 0).
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| EncoderError::InferenceFailed(format!("extract hidden: {e}")))?;
        self.reshape_hidden(&shape, &data)
    }

    fn reshape_hidden(&self, dims: &[i64], data: &[f32]) -> EncoderResult<ndarray::Array3<f32>> {
        if dims.len() != 3 {
            return Err(EncoderError::InferenceFailed(format!(
                "expected 3D output [batch, seq, hidden], got {dims:?}"
            )));
        }

        let d0 = dims[0] as usize;
        let d1 = dims[1] as usize;
        let d2 = dims[2] as usize;

        if d2 != crate::EMBEDDING_DIM {
            return Err(EncoderError::DimensionMismatch {
                expected: crate::EMBEDDING_DIM,
                actual: d2,
            });
        }

        ndarray::Array3::from_shape_vec((d0, d1, d2), data.to_vec())
            .map_err(|e| EncoderError::InferenceFailed(format!("reshape: {e}")))
    }

    /// CLS token pooling — extracts the hidden state at position 0 (the `<s>` token).
    ///
    /// This matches the ONNX export's pooling strategy (`last_hidden_state[:, 0, :]`),
    /// ensuring the fallback path produces results consistent with the ONNX heads.
    pub fn cls_pool(&self, hidden: &ndarray::Array3<f32>) -> Vec<f32> {
        let dim = hidden.shape()[2];
        (0..dim).map(|j| hidden[[0, 0, j]]).collect()
    }

    /// Attention-mask-weighted mean pooling over hidden states.
    ///
    /// Only averages over tokens where `attention_mask[i] == 1`, ignoring
    /// padding tokens. Returns a 1D vector of length 768.
    ///
    /// Note: This is kept for compatibility but should not be used for
    /// heads that expect CLS pooling (which is what the ONNX export uses).
    pub fn mean_pool(&self, hidden: &ndarray::Array3<f32>, attention_mask: &[i64]) -> Vec<f32> {
        let shape = hidden.shape();
        let seq_len = shape[1];
        let dim = shape[2];

        let mut pooled = vec![0.0f32; dim];
        let mut count = 0u32;

        for i in 0..seq_len {
            let mask_val = attention_mask.get(i).copied().unwrap_or(1);
            if mask_val == 0 {
                continue;
            }
            count += 1;
            for j in 0..dim {
                pooled[j] += hidden[[0, i, j]];
            }
        }

        if count == 0 {
            return pooled;
        }
        let inv_count = 1.0 / count as f32;
        for x in &mut pooled {
            *x *= inv_count;
        }
        pooled
    }

    /// L2 normalize a vector in-place.
    pub fn l2_normalize(vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
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
}

// ---------------------------------------------------------------------------
// EP selection — maps kchat-core's platform-aware EP state machine to
// ort execution-provider dispatch objects.
// ---------------------------------------------------------------------------

/// Build the ort execution-provider dispatch list for the current host
/// using [`kchat_core::ep`] selection.
///
/// Returns a `Vec<ExecutionProviderDispatch>` suitable for
/// [`ort::session::SessionBuilder::with_execution_providers`]. The
/// list is ordered most-preferred-first; ort's runtime handles
/// silent fallback to CPU if an accelerator EP fails to register.
#[cfg(feature = "onnx-runtime")]
fn build_ort_eps_for_host() -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
    use kchat_core::ep::{EpDeviceCapabilities, EpFallbackChain, Platform};

    // Detect host platform from cfg.
    let (os, arch) = {
        #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
        {
            (Platform::MacOs, kchat_core::ep::Arch::Aarch64)
        }
        #[cfg(all(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))), target_os = "macos"))]
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

    // Build capabilities — assume accelerator is present on Apple Silicon
    // and Windows with GPU. Production code should probe via
    // `kchat_core::capability::CapabilityProbe::probe()` and bridge through
    // `EpDeviceCapabilities::from_full_caps`.
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

/// Map a [`kchat_core::ep::ExecutionProvider`] to the corresponding
/// ort execution-provider dispatch object. Returns `None` for EPs
/// that have no ort equivalent (should not happen with the current
/// enum, but keeps the match exhaustive).
#[cfg(feature = "onnx-runtime")]
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
        kchat_core::ep::ExecutionProvider::MetalPerformanceShaders => {
            // MPS is not a standalone ort EP; CoreML subsumes it on Apple.
            None
        }
        kchat_core::ep::ExecutionProvider::Cpu => {
            Some(CPUExecutionProvider::default().build())
        }
    }
}

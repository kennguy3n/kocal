//! Multi-step generation pipelines for document-level AI skills.
//!
//! Some skills require multiple LLM calls to process large documents:
//! - **Translate document**: chunk → translate each → stitch
//! - **Improve document**: chunk → improve each with outline context → stitch
//! - **Find contradictions**: extract claims per chunk → cross-reference
//!
//! Each step uses the existing `BackendAdapter::generate` API. Progress
//! callbacks allow the UI to show streaming progress.

use crate::backend::{BackendAdapter, BackendError, GenerationConfig};
use crate::budget::{chunk_document, continuation_prompt, extract_outline_context, DocChunk};
use crate::skills::{SkillDef, SkillPromptInput};
use serde::{Deserialize, Serialize};

/// Maximum chunks a pipeline will process before stopping.
const MAX_CHUNKS: usize = 50;

/// Maximum continuation iterations per chunk.
const MAX_CONTINUATIONS: usize = 5;

/// Progress update from a multi-step pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineProgress {
    /// Current chunk index (0-based).
    pub chunk_index: usize,
    /// Total number of chunks.
    pub total_chunks: usize,
    /// Text produced so far for the current chunk.
    pub current_chunk_output: String,
    /// Cumulative result text (stitched so far).
    pub accumulated_output: String,
    /// Whether this is the final progress update.
    pub done: bool,
}

/// Result of a multi-step pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// Final stitched output text.
    pub text: String,
    /// Number of chunks processed.
    pub chunks_processed: usize,
    /// Total tokens generated across all chunks.
    pub total_tokens: u32,
    /// Total time in milliseconds.
    pub total_ms: u64,
    /// Whether any chunk required continuation.
    pub used_continuation: bool,
}

/// Callback type for pipeline progress updates.
pub type ProgressCallback = Box<dyn Fn(PipelineProgress) + Send + Sync>;

/// A multi-step generation pipeline for document-level skills.
pub struct GenerationPipeline<'a> {
    backend: &'a dyn BackendAdapter,
    skill: &'a SkillDef,
    max_chunk_chars: usize,
    max_continuation_chars: usize,
}

impl<'a> GenerationPipeline<'a> {
    /// Create a new pipeline bound to a backend and skill.
    pub fn new(backend: &'a dyn BackendAdapter, skill: &'a SkillDef) -> Self {
        Self {
            backend,
            skill,
            max_chunk_chars: 6000,
            max_continuation_chars: 500,
        }
    }

    /// Set the maximum characters per chunk.
    pub fn with_max_chunk_chars(mut self, max: usize) -> Self {
        self.max_chunk_chars = max;
        self
    }

    /// Run the pipeline on a document.
    ///
    /// For each chunk:
    /// 1. Build the prompt (using outline context for coherence)
    /// 2. Generate with the skill's config
    /// 3. If output hits max_tokens, continue with a continuation prompt
    /// 4. Stitch results together
    pub fn run(
        &self,
        document: &str,
        variant_context: &str,
        on_progress: Option<&ProgressCallback>,
    ) -> Result<PipelineResult, BackendError> {
        let chunks = chunk_document(document, self.max_chunk_chars);
        let total_chunks = chunks.len().min(MAX_CHUNKS);
        let outline = extract_outline_context(document, 2000);

        let mut accumulated = String::new();
        let mut total_tokens: u32 = 0;
        let mut used_continuation = false;
        let start = std::time::Instant::now();

        for (i, chunk) in chunks.iter().enumerate().take(total_chunks) {
            let chunk_result = self.process_chunk(chunk, &outline, variant_context)?;

            if chunk_result.used_continuation {
                used_continuation = true;
            }
            total_tokens += chunk_result.total_tokens;

            if !accumulated.is_empty() && !chunk_result.text.starts_with('#') {
                accumulated.push_str("\n\n");
            }
            accumulated.push_str(&chunk_result.text);

            if let Some(cb) = on_progress {
                cb(PipelineProgress {
                    chunk_index: i,
                    total_chunks,
                    current_chunk_output: chunk_result.text.clone(),
                    accumulated_output: accumulated.clone(),
                    done: i == total_chunks - 1,
                });
            }
        }

        Ok(PipelineResult {
            text: accumulated,
            chunks_processed: total_chunks,
            total_tokens,
            total_ms: start.elapsed().as_millis() as u64,
            used_continuation,
        })
    }

    /// Process a single chunk, including continuation if needed.
    fn process_chunk(
        &self,
        chunk: &DocChunk,
        _outline: &str,
        variant_context: &str,
    ) -> Result<ChunkResult, BackendError> {
        let prompt_input = SkillPromptInput {
            input: "",
            context: &chunk.text,
            keywords: "",
            variant_context,
            tier: None,
        };
        let prompt_output = self.skill.build_prompt(prompt_input);

        let full_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt_output.system, prompt_output.user
        );

        let config = GenerationConfig {
            max_tokens: self.skill.max_tokens,
            temperature: self.skill.temperature,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            grammar: None,
            seed: 0,
        };

        let mut result = self.backend.generate(&full_prompt, &config)?;
        let mut output = result.text.clone();
        let mut total_tokens = result.completion_tokens;
        let mut used_continuation = false;

        // Check if output was cut off (hit max_tokens)
        for _ in 0..MAX_CONTINUATIONS {
            if result.completion_tokens < self.skill.max_tokens as u32 {
                break; // Output ended naturally
            }

            used_continuation = true;
            let cont_prompt = continuation_prompt(&output, self.max_continuation_chars);
            let full_cont = format!(
                "{}{}<|im_end|>\n<|im_start|>assistant\n{}",
                full_prompt, output, cont_prompt
            );

            result = self.backend.generate(&full_cont, &config)?;
            output.push_str(&result.text);
            total_tokens += result.completion_tokens;
        }

        Ok(ChunkResult {
            text: output,
            total_tokens,
            used_continuation,
        })
    }

    /// Run the contradiction-detection pipeline (two-pass).
    ///
    /// Pass 1: Extract claims from each chunk.
    /// Pass 2: Cross-reference all claims to find contradictions.
    pub fn run_contradiction_check(
        &self,
        document: &str,
        on_progress: Option<&ProgressCallback>,
    ) -> Result<PipelineResult, BackendError> {
        let chunks = chunk_document(document, self.max_chunk_chars);
        let total_chunks = chunks.len().min(MAX_CHUNKS);

        // Pass 1: Extract claims from each chunk
        let mut all_claims = String::new();
        let mut total_tokens: u32 = 0;
        let start = std::time::Instant::now();

        for (i, chunk) in chunks.iter().enumerate().take(total_chunks) {
            let claim_prompt = format!(
                "<|im_start|>system\nExtract all factual claims and statements from the text. Output as a numbered list (1. 2. 3.).<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                chunk.text
            );

            let config = GenerationConfig {
                max_tokens: 300,
                temperature: 0.2,
                ..Default::default()
            };

            let result = self.backend.generate(&claim_prompt, &config)?;
            all_claims.push_str(&format!("--- Section {} ---\n{}\n\n", i + 1, result.text));
            total_tokens += result.completion_tokens;

            if let Some(cb) = on_progress {
                cb(PipelineProgress {
                    chunk_index: i,
                    total_chunks,
                    current_chunk_output: result.text.clone(),
                    accumulated_output: all_claims.clone(),
                    done: false,
                });
            }
        }

        // Pass 2: Cross-reference claims
        let cross_ref_prompt = format!(
            "<|im_start|>system\nFind contradictions in the following claims. Output JSON: {{\"contradictions\": [{{\"quote1\": \"string\", \"quote2\": \"string\", \"explanation\": \"string\"}}]}}. If none found, return empty array.<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n",
            all_claims
        );

        let config = GenerationConfig {
            max_tokens: 400,
            temperature: 0.2,
            ..Default::default()
        };

        let result = self.backend.generate(&cross_ref_prompt, &config)?;
        total_tokens += result.completion_tokens;

        if let Some(cb) = on_progress {
            cb(PipelineProgress {
                chunk_index: total_chunks,
                total_chunks,
                current_chunk_output: result.text.clone(),
                accumulated_output: result.text.clone(),
                done: true,
            });
        }

        Ok(PipelineResult {
            text: result.text,
            chunks_processed: total_chunks,
            total_tokens,
            total_ms: start.elapsed().as_millis() as u64,
            used_continuation: false,
        })
    }
}

/// Internal result for processing a single chunk.
struct ChunkResult {
    text: String,
    total_tokens: u32,
    used_continuation: bool,
}

/// Convenience function to run a document translation pipeline.
pub fn translate_document(
    backend: &dyn BackendAdapter,
    skill: &SkillDef,
    document: &str,
    target_language: &str,
    on_progress: Option<&ProgressCallback>,
) -> Result<PipelineResult, BackendError> {
    let pipeline = GenerationPipeline::new(backend, skill);
    pipeline.run(document, target_language, on_progress)
}

/// Convenience function to run a document improvement pipeline.
pub fn improve_document(
    backend: &dyn BackendAdapter,
    skill: &SkillDef,
    document: &str,
    on_progress: Option<&ProgressCallback>,
) -> Result<PipelineResult, BackendError> {
    let pipeline = GenerationPipeline::new(backend, skill);
    pipeline.run(document, "", on_progress)
}

/// Convenience function to run a contradiction check pipeline.
pub fn find_contradictions(
    backend: &dyn BackendAdapter,
    skill: &SkillDef,
    document: &str,
    on_progress: Option<&ProgressCallback>,
) -> Result<PipelineResult, BackendError> {
    let pipeline = GenerationPipeline::new(backend, skill);
    pipeline.run_contradiction_check(document, on_progress)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::MockBackend;
    use crate::backend::{BackendConfig, BackendType};
    use crate::skills::SkillRegistry;
    use kchat_core::tier::DeviceTier;

    fn get_skill(id: &str) -> SkillDef {
        let registry = SkillRegistry::new();
        registry.get(id).unwrap().clone()
    }

    fn make_loaded_backend() -> MockBackend {
        let backend = MockBackend::new();
        let config = BackendConfig::for_tier(
            BackendType::LlamaCppCpu,
            "mock",
            "/dev/null",
            DeviceTier::Medium,
            "macos",
        );
        backend.load(&config).unwrap();
        backend
    }

    #[test]
    fn test_pipeline_translate_small_doc() {
        let backend = make_loaded_backend();
        let skill = get_skill("edit_translate_document");
        let pipeline = GenerationPipeline::new(&backend, &skill);

        let result = pipeline.run("Hello world.", "Spanish", None).unwrap();
        assert!(!result.text.is_empty());
        assert!(result.chunks_processed >= 1);
    }

    #[test]
    fn test_pipeline_improve_small_doc() {
        let backend = make_loaded_backend();
        let skill = get_skill("edit_improve_document");
        let pipeline = GenerationPipeline::new(&backend, &skill);

        let result = pipeline.run("This is a test.", "", None).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn test_pipeline_contradiction_check() {
        let backend = make_loaded_backend();
        let skill = get_skill("doc_find_contradictions");
        let pipeline = GenerationPipeline::new(&backend, &skill);

        let result = pipeline.run_contradiction_check("The sky is blue. The sky is green.", None).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn test_pipeline_large_doc_multiple_chunks() {
        let backend = make_loaded_backend();
        let skill = get_skill("edit_translate_document");
        let pipeline = GenerationPipeline::new(&backend, &skill)
            .with_max_chunk_chars(50); // Force multiple chunks

        let doc = "# Section 1\nThis is the first section with some content.\n\n# Section 2\nThis is the second section with more content.";
        let result = pipeline.run(doc, "French", None).unwrap();
        assert!(result.chunks_processed > 1);
    }

    #[test]
    fn test_pipeline_progress_callback() {
        let backend = make_loaded_backend();
        let skill = get_skill("edit_translate_document");
        let pipeline = GenerationPipeline::new(&backend, &skill)
            .with_max_chunk_chars(30);

        let doc = "# A\nContent A.\n\n# B\nContent B.\n\n# C\nContent C.";
        let progress_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let pc_clone = progress_count.clone();

        let callback: ProgressCallback = Box::new(move |_| {
            pc_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });

        pipeline.run(doc, "German", Some(&callback)).unwrap();
        assert!(progress_count.load(std::sync::atomic::Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_translate_document_convenience() {
        let backend = make_loaded_backend();
        let skill = get_skill("edit_translate_document");

        let result = translate_document(&backend, &skill, "Hello", "Spanish", None).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn test_improve_document_convenience() {
        let backend = make_loaded_backend();
        let skill = get_skill("edit_improve_document");

        let result = improve_document(&backend, &skill, "Hello world", None).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn test_find_contradictions_convenience() {
        let backend = make_loaded_backend();
        let skill = get_skill("doc_find_contradictions");

        let result = find_contradictions(&backend, &skill, "Test text", None).unwrap();
        assert!(!result.text.is_empty());
    }
}

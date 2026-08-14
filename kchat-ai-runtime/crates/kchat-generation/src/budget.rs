//! Token budget management and document chunking.
//!
//! Provides utilities for estimating token counts, calculating context budgets,
//! adaptively truncating output to fit within the model's context window, and
//! chunking large documents into semantically coherent pieces.
//!
//! Conservative estimate: ~3 chars/token for English text.
//! Context windows: Low 1K–2K, Medium 2K–4K, High 4K–16K (platform-dependent).

use crate::skills::estimate_tokens;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Conservative chars-per-token ratio for English text.
pub const CHARS_PER_TOKEN: usize = 3;

/// Overhead tokens for system prompt + chat template formatting.
pub const TEMPLATE_OVERHEAD: usize = 120;

/// Safety margin tokens to avoid edge-of-window truncation.
pub const SAFETY_MARGIN: usize = 80;

/// Maximum characters per chunk when splitting documents.
pub const DEFAULT_MAX_CHUNK_CHARS: usize = 6000;

/// Maximum characters for outline context (headings + first sentence per section).
pub const DEFAULT_OUTLINE_MAX_CHARS: usize = 2000;

/// Maximum characters for local context around a selection.
pub const DEFAULT_LOCAL_CONTEXT_CHARS: usize = 500;

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Estimate the number of tokens in a text string.
///
/// Uses a conservative ratio of ~3 chars/token. This is intentionally
/// conservative to avoid underestimating and running out of context.
pub fn estimate_tokens_text(text: &str) -> usize {
    estimate_tokens(text)
}

/// Estimate the total token cost of a system + user prompt pair.
pub fn estimate_prompt_tokens(system: &str, user: &str) -> usize {
    estimate_tokens(system) + estimate_tokens(user) + TEMPLATE_OVERHEAD
}

// ---------------------------------------------------------------------------
// Context budget
// ---------------------------------------------------------------------------

/// Calculate the available character budget for context text.
///
/// Returns the maximum number of characters that can be used for context
/// given the total context window, the system prompt, and the desired
/// output token allocation.
pub fn budget_for_context(
    total_context_tokens: usize,
    system_prompt: &str,
    max_output_tokens: usize,
) -> usize {
    let system_tokens = estimate_tokens(system_prompt);
    let overhead = TEMPLATE_OVERHEAD + SAFETY_MARGIN;
    let used = system_tokens + max_output_tokens + overhead;
    let remaining_tokens = total_context_tokens.saturating_sub(used);
    remaining_tokens * CHARS_PER_TOKEN
}

/// Adaptively reduce max_output_tokens to ensure the input context fits
/// within the model's context window.
///
/// Returns the adjusted max_output_tokens value. If the context alone
/// exceeds the window, returns a minimum of 64 tokens.
pub fn adaptive_max_output(
    total_context_tokens: usize,
    system_prompt: &str,
    context_text: &str,
    desired_max_tokens: usize,
) -> usize {
    let system_tokens = estimate_tokens(system_prompt);
    let context_tokens = estimate_tokens(context_text);
    let overhead = TEMPLATE_OVERHEAD + SAFETY_MARGIN;
    let available_for_output = total_context_tokens
        .saturating_sub(system_tokens + context_tokens + overhead);
    available_for_output.min(desired_max_tokens).max(64)
}

// ---------------------------------------------------------------------------
// Truncation
// ---------------------------------------------------------------------------

/// Truncate text to fit within a maximum character count, adding an ellipsis
/// if truncation occurs.
pub fn truncate_context(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let truncated = &text[..max_chars.saturating_sub(3)];
    // Try to cut at a word boundary
    let cut_point = truncated.rfind(' ').unwrap_or(truncated.len());
    format!("{}...", &truncated[..cut_point])
}

/// Truncate text from the end, keeping the first `max_chars` characters.
/// Useful for preserving the beginning of a document.
pub fn truncate_head(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    text[..max_chars].to_string()
}

/// Truncate text from the beginning, keeping the last `max_chars` characters.
/// Useful for preserving the most recent context.
pub fn truncate_tail(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let start = text.len() - max_chars;
    text[start..].to_string()
}

// ---------------------------------------------------------------------------
// Document chunking
// ---------------------------------------------------------------------------

/// A document chunk with positional metadata.
#[derive(Debug, Clone)]
pub struct DocChunk {
    /// The chunk text content.
    pub text: String,
    /// 0-based chunk index.
    pub index: usize,
    /// Total number of chunks.
    pub total: usize,
    /// Character offset of this chunk within the original document.
    pub char_offset: usize,
}

/// Split a document into semantically coherent chunks.
///
/// Chunking strategy (in priority order):
/// 1. Split at markdown headings (`#`, `##`, etc.)
/// 2. If a section is too large, split at paragraph boundaries (double newline)
/// 3. If a paragraph is too large, split at sentence boundaries (`. `, `! `, `? `)
/// 4. If a sentence is too large, hard-split at `max_chars`
pub fn chunk_document(text: &str, max_chars: usize) -> Vec<DocChunk> {
    if text.len() <= max_chars {
        return vec![DocChunk {
            text: text.to_string(),
            index: 0,
            total: 1,
            char_offset: 0,
        }];
    }

    // Strategy 1: Split at headings
    let sections = split_at_headings(text);
    let mut chunks = Vec::new();
    let mut current_offset = 0;

    for section in sections {
        if section.len() <= max_chars {
            chunks.push((section.clone(), current_offset));
        } else {
            // Strategy 2: Split at paragraphs
            let paras = split_at_paragraphs(&section, max_chars);
            for para in paras {
                if para.len() <= max_chars {
                    chunks.push((para.clone(), current_offset));
                } else {
                    // Strategy 3: Split at sentences
                    let sentences = split_at_sentences(&para, max_chars);
                    for sentence in sentences {
                        if sentence.len() <= max_chars {
                            chunks.push((sentence.clone(), current_offset));
                        } else {
                            // Strategy 4: Hard split
                            let hard = hard_split(&sentence, max_chars);
                            for piece in hard {
                                chunks.push((piece, current_offset));
                            }
                        }
                        current_offset += sentence.len();
                    }
                    continue; // Already advanced offset
                }
                current_offset += para.len();
            }
            continue;
        }
        current_offset += section.len();
    }

    let total = chunks.len();
    chunks
        .into_iter()
        .enumerate()
        .map(|(i, (text, offset))| DocChunk {
            text,
            index: i,
            total,
            char_offset: offset,
        })
        .collect()
}

/// Extract a compact outline of the document: headings + first sentence of
/// each section. This fits any document size in a small token budget.
pub fn extract_outline_context(text: &str, max_chars: usize) -> String {
    let mut result = String::new();
    let sections = split_at_headings(text);

    for section in &sections {
        // Check if this section starts with a heading
        let heading_end = section.find('\n').unwrap_or(section.len());
        let heading = &section[..heading_end];

        if heading.starts_with('#') {
            if result.len() + heading.len() + 1 > max_chars {
                break;
            }
            result.push_str(heading);
            result.push('\n');
        }

        // Get first sentence of the section body (after heading)
        let body = if heading_end < section.len() {
            &section[heading_end + 1..]
        } else {
            ""
        };
        let body = body.trim();
        if !body.is_empty() {
            let first_sentence = extract_first_sentence(body);
            if !first_sentence.is_empty() {
                let line = format!("  {}\n", first_sentence);
                if result.len() + line.len() > max_chars {
                    break;
                }
                result.push_str(&line);
            }
        }
        result.push('\n');
    }

    result.trim_end().to_string()
}

/// Extract local context around a position in the text.
///
/// Returns the nearest preceding heading and a window of characters
/// before and after the position.
pub fn get_local_context(
    text: &str,
    pos: usize,
    max_chars: usize,
) -> String {
    let half = max_chars / 2;
    let start = pos.saturating_sub(half);
    let end = (pos + half).min(text.len());

    let window = &text[start..end];

    // Find nearest preceding heading
    let heading = text[..pos]
        .lines()
        .rev()
        .find(|line| line.starts_with('#'))
        .map(|h| h.to_string());

    let mut result = String::new();
    if let Some(h) = heading {
        result.push_str(&h);
        result.push('\n');
    }
    result.push_str(window);
    result
}

// ---------------------------------------------------------------------------
// Continuation prompt
// ---------------------------------------------------------------------------

/// Build a continuation prompt for multi-step generation when output
/// hits the max_tokens limit.
pub fn continuation_prompt(last_output: &str, max_context_chars: usize) -> String {
    let tail = truncate_tail(last_output, max_context_chars);
    format!(
        "Continue from exactly where you stopped. Last output:\n{}\n\nContinue:",
        tail
    )
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Split text at markdown headings (#, ##, ###, etc.)
fn split_at_headings(text: &str) -> Vec<String> {
    let mut sections = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        if line.starts_with('#') && !current.is_empty() {
            sections.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
    }

    if !current.is_empty() {
        sections.push(current);
    }

    if sections.is_empty() {
        vec![text.to_string()]
    } else {
        sections
    }
}

/// Split text at paragraph boundaries (double newline).
fn split_at_paragraphs(text: &str, max_chars: usize) -> Vec<String> {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut result = Vec::new();
    let mut current = String::new();

    for para in paragraphs {
        if current.len() + para.len() + 2 > max_chars && !current.is_empty() {
            result.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }

    if !current.is_empty() {
        result.push(current);
    }

    if result.is_empty() {
        vec![text.to_string()]
    } else {
        result
    }
}

/// Split text at sentence boundaries (. ! ?)
fn split_at_sentences(text: &str, max_chars: usize) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for sentence_end in text.split_inclusive(['.', '!', '?']) {
        if current.len() + sentence_end.len() > max_chars && !current.is_empty() {
            sentences.push(std::mem::take(&mut current));
        }
        current.push_str(sentence_end);
    }

    if !current.is_empty() {
        sentences.push(current);
    }

    if sentences.is_empty() {
        vec![text.to_string()]
    } else {
        sentences
    }
}

/// Hard-split text at character boundaries.
fn hard_split(text: &str, max_chars: usize) -> Vec<String> {
    let mut result = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = (start + max_chars).min(text.len());
        result.push(text[start..end].to_string());
        start = end;
    }

    if result.is_empty() {
        vec![text.to_string()]
    } else {
        result
    }
}

/// Extract the first sentence from a text body.
fn extract_first_sentence(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    // Find the first sentence-ending punctuation
    for (i, c) in trimmed.char_indices() {
        if c == '.' || c == '!' || c == '?' {
            return trimmed[..=i].to_string();
        }
    }

    // No sentence ending found — return up to first newline or whole text
    let nl = trimmed.find('\n').unwrap_or(trimmed.len());
    if nl <= 200 {
        trimmed[..nl].to_string()
    } else {
        // Truncate at word boundary near 200 chars
        let truncated = &trimmed[..200];
        let cut = truncated.rfind(' ').unwrap_or(200);
        format!("{}...", &truncated[..cut])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_text() {
        assert_eq!(estimate_tokens_text(""), 0);
        assert_eq!(estimate_tokens_text("abc"), 1);
        assert_eq!(estimate_tokens_text("abcdef"), 2);
    }

    #[test]
    fn test_estimate_prompt_tokens() {
        let tokens = estimate_prompt_tokens("System prompt", "User prompt here");
        // system ~5 + user ~6 + overhead 120 = ~131
        assert!(tokens >= 130);
        assert!(tokens <= 135);
    }

    #[test]
    fn test_budget_for_context() {
        let budget = budget_for_context(4096, "Short system", 200);
        // 4096 - (4 + 200 + 200) * 3 = 4096 - 204 = 3892 * 3 = 11676
        // Actually: system_tokens=4, overhead=120, safety=80, max_output=200
        // remaining = 4096 - 4 - 200 - 200 = 3692, * 3 = 11076
        assert!(budget > 10000);
        assert!(budget < 12000);
    }

    #[test]
    fn test_budget_for_context_zero_output() {
        let budget = budget_for_context(1024, "System", 0);
        // 1024 - (1 + 0 + 200) = 823 * 3 = 2469
        assert!(budget > 2000);
        assert!(budget < 3000);
    }

    #[test]
    fn test_adaptive_max_output() {
        let max = adaptive_max_output(4096, "System prompt", "Short context", 500);
        // Plenty of room — should return desired 500
        assert_eq!(max, 500);
    }

    #[test]
    fn test_adaptive_max_output_tight() {
        // Very tight context — system + context nearly fills window
        let large_context = "a".repeat(10000);
        let max = adaptive_max_output(1024, "System", &large_context, 500);
        // Should be reduced to minimum 64
        assert_eq!(max, 64);
    }

    #[test]
    fn test_truncate_context_short() {
        let result = truncate_context("short text", 100);
        assert_eq!(result, "short text");
    }

    #[test]
    fn test_truncate_context_long() {
        let text = "This is a long sentence that needs to be truncated at a word boundary.";
        let result = truncate_context(text, 30);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 30);
    }

    #[test]
    fn test_truncate_head() {
        let result = truncate_head("Hello World", 5);
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_truncate_tail() {
        let result = truncate_tail("Hello World", 5);
        assert_eq!(result, "World");
    }

    #[test]
    fn test_chunk_document_small() {
        let text = "This is a small document.";
        let chunks = chunk_document(text, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
        assert_eq!(chunks[0].index, 0);
        assert_eq!(chunks[0].total, 1);
    }

    #[test]
    fn test_chunk_document_at_headings() {
        let text = "# Section 1\nContent one.\n\n# Section 2\nContent two.";
        let chunks = chunk_document(text, 25);
        assert!(chunks.len() >= 2);
        // First chunk should contain "Section 1"
        assert!(chunks[0].text.contains("Section 1"));
    }

    #[test]
    fn test_chunk_document_large() {
        let text = "a".repeat(10000);
        let chunks = chunk_document(&text, 1000);
        assert!(chunks.len() > 1);
        // All chunks should have correct indices
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.index, i);
            assert_eq!(chunk.total, chunks.len());
        }
    }

    #[test]
    fn test_extract_outline_context() {
        let text = "# Introduction\nThis is the first section. It has content.\n\n# Methods\nWe used various methods. They were effective.";
        let outline = extract_outline_context(text, 2000);
        assert!(outline.contains("# Introduction"));
        assert!(outline.contains("# Methods"));
        assert!(outline.contains("This is the first section."));
        assert!(outline.contains("We used various methods."));
    }

    #[test]
    fn test_extract_outline_context_truncated() {
        let mut text = String::new();
        for i in 0..50 {
            text.push_str(&format!("# Section {}\nThis is content for section {}.\n\n", i, i));
        }
        let outline = extract_outline_context(&text, 200);
        assert!(outline.len() <= 200);
    }

    #[test]
    fn test_get_local_context() {
        let text = "# Heading\nSome text before the cursor position and some text after.";
        let ctx = get_local_context(text, 20, 50);
        assert!(ctx.contains("# Heading"));
    }

    #[test]
    fn test_continuation_prompt() {
        let prompt = continuation_prompt("This is some generated text that was cut off", 100);
        assert!(prompt.contains("Continue from exactly"));
        assert!(prompt.contains("cut off"));
    }

    #[test]
    fn test_split_at_headings() {
        let text = "# H1\nContent 1\n## H2\nContent 2";
        let sections = split_at_headings(text);
        assert!(sections.len() >= 2);
    }

    #[test]
    fn test_split_at_paragraphs() {
        let text = "Para one.\n\nPara two.\n\nPara three.";
        let paras = split_at_paragraphs(text, 100);
        assert_eq!(paras.len(), 1); // All fit in one chunk
    }

    #[test]
    fn test_split_at_sentences() {
        let text = "First sentence. Second sentence. Third one!";
        let sentences = split_at_sentences(text, 20);
        assert!(sentences.len() >= 2);
    }

    #[test]
    fn test_hard_split() {
        let text = "abcdefghij";
        let parts = hard_split(text, 3);
        assert_eq!(parts.len(), 4);
        assert_eq!(parts[0], "abc");
        assert_eq!(parts[3], "j");
    }

    #[test]
    fn test_extract_first_sentence() {
        assert_eq!(extract_first_sentence("Hello world. Next."), "Hello world.");
        assert_eq!(extract_first_sentence("No ending here"), "No ending here");
        assert_eq!(extract_first_sentence(""), "");
    }

    #[test]
    fn test_chunk_offsets() {
        let text = "# A\nContent A.\n\n# B\nContent B.";
        let chunks = chunk_document(text, 20);
        // Verify offsets are within bounds
        for chunk in &chunks {
            assert!(chunk.char_offset <= text.len());
        }
    }
}

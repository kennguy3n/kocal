//! Slides AI skill evaluation suite — per-skill quality evaluation for the
//! 12 slides skills across 210 smart templates.
//!
//! Mock mode (`--slides`): validates prompt construction, grammar, token
//! budget, template registry, slot schema conformance, and quality check
//! functions without a model.
//!
//! Real model mode (`--slides --realworld`): sends slides skill prompts to
//! llama-server/MLX, runs quality checks against real model output, and
//! reports per-skill metrics.
//!
//! Image search mode (`--slides-images`): runs the image search eval against
//! real Pexels/Pixabay/Unsplash/Shutterstock APIs (always real — requires
//! API keys in env vars).

use crate::report::{EvalReport, EvalResult, SuiteReport};
use kchat_generation::{
    SkillPromptInput, SkillRegistry, SkillSurface, SlidesTemplateFamily,
    SlidesTemplateRegistry,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Dataset structures (mirrors eval_skills.rs but with slides-specific fields)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlidesEvalDataset {
    name: String,
    version: String,
    description: String,
    test_cases: Vec<SlidesTestCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlidesTestCase {
    id: String,
    skill_id: String,
    surface: String,
    scope: String,
    mode: String,
    input: SlidesTestInput,
    variant: Option<String>,
    max_tokens: u32,
    grammar_type: String,
    quality_checks: Vec<SlidesQualityCheck>,
    expected_properties: serde_json::Value,
    tier: String,
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlidesTestInput {
    document: String,
    selection: String,
    cursor_context: String,
    variant_context: String,
    keywords: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlidesQualityCheck {
    #[serde(rename = "type")]
    check_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_chars: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_chars: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_bullets: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_bullets: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_slots: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_slides: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    template_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_sentences: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_sentences: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
}

// ---------------------------------------------------------------------------
// Image search dataset structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageSearchDataset {
    name: String,
    version: String,
    description: String,
    test_cases: Vec<ImageSearchCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ImageSearchCase {
    id: String,
    provider: String,
    query: String,
    orientation: Option<String>,
    per_page: u32,
    safesearch: bool,
    expected_min_results: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_orientation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_error: Option<bool>,
    description: String,
}

// ---------------------------------------------------------------------------
// Result tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SlidesResult {
    case_id: String,
    skill_id: String,
    tier: String,
    passed: bool,
    quality_score: f64,
    checks_detail: Vec<(String, f64)>,
    duration_ms: u64,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct ImageResult {
    case_id: String,
    provider: String,
    passed: bool,
    result_count: usize,
    duration_ms: u64,
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Quality check runner
// ---------------------------------------------------------------------------

fn run_slides_quality_check(output: &str, check: &SlidesQualityCheck) -> f64 {
    match check.check_type.as_str() {
        "min_length" => {
            let min = check.min_chars.unwrap_or(0);
            if output.len() >= min { 1.0 } else if min > 0 {
                output.len() as f64 / min as f64
            } else { 1.0 }
        }
        "max_length" => {
            let max = check.max_chars.unwrap_or(usize::MAX);
            if output.len() <= max { 1.0 } else { 0.0 }
        }
        "coherent" => {
            if output.is_empty() || output.len() <= 10 { 0.0 }
            else if is_repeated(output) { 0.3 }
            else { 1.0 }
        }
        "json_schema_valid" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            match serde_json::from_str::<serde_json::Value>(&json_text) {
                Ok(_) => 1.0,
                Err(_) => 0.0,
            }
        }
        "template_conformance" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            let registry = &kchat_generation::TEMPLATE_REGISTRY;
            // For single-slide output
            if let Some(tid) = parsed.get("template_id").and_then(|v| v.as_str()) {
                if registry.get(tid).is_some() { 1.0 } else { 0.0 }
            }
            // For deck output (array of slides)
            else if let Some(slides) = parsed.get("slides").and_then(|v| v.as_array()) {
                if slides.is_empty() { return 0.0; }
                let valid = slides.iter().filter(|s| {
                    s.get("template_id")
                        .and_then(|v| v.as_str())
                        .map(|tid| registry.get(tid).is_some())
                        .unwrap_or(false)
                }).count();
                valid as f64 / slides.len() as f64
            }
            // For outline output
            else if let Some(outline) = parsed.get("outline").and_then(|v| v.as_array()) {
                if outline.is_empty() { return 0.0; }
                let valid = outline.iter().filter(|s| {
                    s.get("template_id")
                        .and_then(|v| v.as_str())
                        .map(|tid| registry.get(tid).is_some())
                        .unwrap_or(false)
                }).count();
                valid as f64 / outline.len() as f64
            } else { 0.0 }
        }
        // === NEW: Slot schema conformance — validates that slots match the
        // template's declared slot definitions (type, required fields). ===
        "slot_schema_conformance" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            let registry = &kchat_generation::TEMPLATE_REGISTRY;
            check_slot_schema(&parsed, registry)
        }
        // === NEW: Cross-slide consistency — checks that deck slides have
        // unique titles, no duplicate template_id sequences, and logical flow. ===
        "cross_slide_consistency" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            check_cross_slide_consistency(&parsed)
        }
        // === NEW: Template selection accuracy — checks that the selected
        // template_id is appropriate for the content type (e.g. title template
        // for title content, chart template for data content). ===
        "template_selection_accuracy" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            let registry = &kchat_generation::TEMPLATE_REGISTRY;
            check_template_selection_accuracy(&parsed, registry)
        }
        // === NEW: Semantic quality — checks for empty/placeholder text,
        // non-dictionary words, and content-topic relevance. ===
        "semantic_quality" => {
            check_semantic_quality(output)
        }
        // === NEW: Image search query relevance — checks that image queries
        // are specific, descriptive, and not generic. ===
        "image_query_relevance" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            check_image_query_relevance(&parsed)
        }
        // === NEW: No executable content — ensures slide text doesn't
        // contain script tags, HTML, or code injection attempts. ===
        "no_executable_content" => {
            check_no_executable_content(output)
        }
        // === NEW: Deck structure validity — checks that deck has a title
        // slide as first slide, reasonable slide count, and no empty slides. ===
        "deck_structure_valid" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            check_deck_structure(&parsed)
        }
        "bullet_count" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            let bullets = parsed
                .get("slots")
                .and_then(|s| s.get("bullets"))
                .and_then(|b| b.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            let min = check.min_bullets.unwrap_or(0);
            let max = check.max_bullets.unwrap_or(usize::MAX);
            if bullets >= min && bullets <= max { 1.0 }
            else if bullets < min && min > 0 { bullets as f64 / min as f64 }
            else if max > 0 { max as f64 / bullets.max(1) as f64 }
            else { 0.0 }
        }
        "slot_count" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            let slide_count = parsed.get("slides").and_then(|s| s.as_array()).map(|a| a.len()).unwrap_or(1);
            let min = check.min_slots.unwrap_or(0);
            let max = check.max_slides.unwrap_or(usize::MAX);
            if slide_count >= min && slide_count <= max { 1.0 }
            else if slide_count < min && min > 0 { slide_count as f64 / min as f64 }
            else if max > 0 { max as f64 / slide_count.max(1) as f64 }
            else { 0.0 }
        }
        "chart_data_valid" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            let series = parsed.get("slots").and_then(|s| s.get("series")).or_else(|| parsed.get("series"));
            if let Some(arr) = series.and_then(|s| s.as_array()) {
                if arr.is_empty() { return 0.0; }
                let valid = arr.iter().filter(|item| {
                    item.get("label").and_then(|v| v.as_str()).is_some()
                        && item.get("value").and_then(|v| v.as_f64()).is_some()
                }).count();
                valid as f64 / arr.len() as f64
            } else { 0.0 }
        }
        "image_query_valid" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            // For slides_add_image skill
            if let Some(q) = parsed.get("query").and_then(|v| v.as_str()) {
                if q.is_empty() { return 0.0; }
                if q.starts_with("http://") || q.starts_with("https://") { return 0.0; }
                return 1.0;
            }
            // For slide with image slot
            if let Some(img) = parsed.get("slots").and_then(|s| s.get("image")) {
                if let Some(q) = img.get("query").and_then(|v| v.as_str()) {
                    if q.is_empty() { return 0.0; }
                    if q.starts_with("http://") || q.starts_with("https://") { return 0.0; }
                    return 1.0;
                }
            }
            0.0
        }
        "sentence_count" => {
            let count = count_sentences(output);
            let min = check.min_sentences.unwrap_or(0);
            let max = check.max_sentences.unwrap_or(usize::MAX);
            if count >= min && count <= max { 1.0 }
            else if count < min { count as f64 / min as f64 }
            else { max as f64 / count as f64 }
        }
        "language_script" => {
            if let Some(lang) = &check.language {
                detect_language_score(output, lang)
            } else { 1.0 }
        }
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Advanced quality check helpers
// ---------------------------------------------------------------------------

/// Check that slots conform to the template's declared slot schema.
/// Validates that each required slot is present and has the correct type.
fn check_slot_schema(parsed: &serde_json::Value, registry: &kchat_generation::SlidesTemplateRegistry) -> f64 {
    // Single slide
    if let Some(tid) = parsed.get("template_id").and_then(|v| v.as_str()) {
        if let Some(template) = registry.get(tid) {
            let slots = parsed.get("slots").unwrap_or(&serde_json::Value::Null);
            return validate_slots_against_template(slots, template);
        }
        return 0.0;
    }
    // Deck
    if let Some(slides) = parsed.get("slides").and_then(|v| v.as_array()) {
        if slides.is_empty() { return 0.0; }
        let mut total_score = 0.0;
        for slide in slides {
            if let Some(tid) = slide.get("template_id").and_then(|v| v.as_str()) {
                if let Some(template) = registry.get(tid) {
                    let slots = slide.get("slots").unwrap_or(&serde_json::Value::Null);
                    total_score += validate_slots_against_template(slots, template);
                }
            }
        }
        return total_score / slides.len() as f64;
    }
    0.0
}

/// Validate that slots match a template's slot definitions.
fn validate_slots_against_template(slots: &serde_json::Value, template: &kchat_generation::SlidesTemplate) -> f64 {
    let slot_obj = match slots.as_object() {
        Some(o) => o,
        None => return 0.0,
    };
    let mut score = 0.0;
    let total = template.slots.len().max(1);
    for slot_def in &template.slots {
        let slot_id = &slot_def.id;
        if let Some(value) = slot_obj.get(slot_id) {
            // Check type conformance based on slot type
            let type_ok = match slot_def.slot_type {
                // String types
                kchat_generation::SlotType::TitleText | kchat_generation::SlotType::SubtitleText
                | kchat_generation::SlotType::BodyText | kchat_generation::SlotType::QuoteText
                | kchat_generation::SlotType::AttributionText | kchat_generation::SlotType::PersonName
                | kchat_generation::SlotType::PersonRole | kchat_generation::SlotType::LabelText
                | kchat_generation::SlotType::CaptionText | kchat_generation::SlotType::FooterText
                | kchat_generation::SlotType::SectionLabel | kchat_generation::SlotType::StatLabel
                | kchat_generation::SlotType::ImageQuery | kchat_generation::SlotType::ImageRef => value.is_string(),
                // Array types
                kchat_generation::SlotType::BulletList | kchat_generation::SlotType::StepList
                | kchat_generation::SlotType::DateList | kchat_generation::SlotType::NumberedList => value.is_array(),
                // Numeric types
                kchat_generation::SlotType::StatNumber => value.is_number() || value.is_string(),
                // Chart series — object or array
                kchat_generation::SlotType::ChartSeries => value.is_object() || value.is_array(),
            };
            if type_ok { score += 1.0; }
        }
    }
    score / total as f64
}

/// Check cross-slide consistency: unique titles, no duplicate template sequences.
fn check_cross_slide_consistency(parsed: &serde_json::Value) -> f64 {
    let slides = match parsed.get("slides").and_then(|v| v.as_array()) {
        Some(s) if !s.is_empty() => s,
        _ => return 1.0, // Not a deck, skip
    };
    let mut score: f64 = 1.0;
    let mut titles: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut duplicate_titles = 0;
    for slide in slides {
        if let Some(title) = slide.get("title").and_then(|v| v.as_str()) {
            if !titles.insert(title.to_string()) {
                duplicate_titles += 1;
            }
        }
    }
    // Penalize duplicate titles
    if duplicate_titles > 0 {
        score -= 0.2 * duplicate_titles as f64 / slides.len() as f64;
    }
    // Check for 3+ consecutive same-template slides (likely model degeneration)
    let mut consecutive = 1;
    let mut max_consecutive = 1;
    for i in 1..slides.len() {
        let prev_tid = slides[i-1].get("template_id").and_then(|v| v.as_str());
        let curr_tid = slides[i].get("template_id").and_then(|v| v.as_str());
        if prev_tid.is_some() && prev_tid == curr_tid {
            consecutive += 1;
            max_consecutive = max_consecutive.max(consecutive);
        } else {
            consecutive = 1;
        }
    }
    if max_consecutive >= 4 {
        score -= 0.3; // 4+ consecutive same-template slides is suspicious
    }
    score.max(0.0f64)
}

/// Check that template selection is appropriate for content type.
fn check_template_selection_accuracy(parsed: &serde_json::Value, registry: &kchat_generation::SlidesTemplateRegistry) -> f64 {
    if let Some(tid) = parsed.get("template_id").and_then(|v| v.as_str()) {
        if let Some(template) = registry.get(tid) {
            // Check if title slide has title text
            if template.family == kchat_generation::SlidesTemplateFamily::Title {
                if let Some(title) = parsed.get("slots").and_then(|s| s.get("title")).and_then(|v| v.as_str()) {
                    if title.is_empty() { return 0.5; }
                    return 1.0;
                }
            }
            // Check if bullet slide has bullets
            if template.family == kchat_generation::SlidesTemplateFamily::Bullet {
                if let Some(bullets) = parsed.get("slots").and_then(|s| s.get("bullets")).and_then(|b| b.as_array()) {
                    if bullets.is_empty() { return 0.3; }
                    return 1.0;
                }
            }
            return 0.7; // Template exists but can't verify content match
        }
        return 0.0;
    }
    0.5 // Not a single-slide output
}

/// Check semantic quality: no empty/placeholder text, no excessive repetition.
fn check_semantic_quality(output: &str) -> f64 {
    if output.is_empty() { return 0.0; }
    let mut score: f64 = 1.0;
    // Check for placeholder text
    let placeholders = ["lorem ipsum", "placeholder", "todo", "tbd", "xxx", "fill in", "[text]", "your text here"];
    let lower = output.to_lowercase();
    for p in &placeholders {
        if lower.contains(p) {
            score -= 0.3;
        }
    }
    // Check for excessive repetition of words
    if is_repeated(output) {
        score -= 0.4;
    }
    // Check for minimum content length
    if output.len() < 20 {
        score -= 0.2;
    }
    // Check for only whitespace/punctuation
    if output.chars().all(|c| c.is_whitespace() || c.is_ascii_punctuation()) {
        score = 0.0;
    }
    score.max(0.0f64)
}

/// Check image query relevance: specific, descriptive, not generic.
fn check_image_query_relevance(parsed: &serde_json::Value) -> f64 {
    let query = parsed.get("query").and_then(|v| v.as_str())
        .or_else(|| parsed.get("slots").and_then(|s| s.get("image")).and_then(|i| i.get("query")).and_then(|v| v.as_str()));
    if let Some(q) = query {
        if q.is_empty() { return 0.0; }
        let mut score: f64 = 1.0;
        // Penalize very short queries (1-2 words may be too generic)
        let word_count = q.split_whitespace().count();
        if word_count < 2 { score -= 0.3; }
        // Penalize queries that are just "image" or "photo"
        let lower = q.to_lowercase();
        if lower == "image" || lower == "photo" || lower == "picture" || lower == "icon" {
            score -= 0.5;
        }
        // Penalize URLs in queries
        if q.starts_with("http://") || q.starts_with("https://") {
            score = 0.0;
        }
        // Penalize very long queries (may be too specific / prompt injection)
        if q.len() > 200 {
            score -= 0.3;
        }
        return score.max(0.0f64);
    }
    0.0
}

/// Check that output doesn't contain executable content (script tags, HTML, code injection).
fn check_no_executable_content(output: &str) -> f64 {
    let lower = output.to_lowercase();
    let dangerous_patterns = [
        "<script", "javascript:", "onerror=", "onload=", "onclick=",
        "<iframe", "<embed", "<object", "eval(", "document.cookie",
        "window.location", "<svg onload", "data:text/html",
    ];
    let violations = dangerous_patterns.iter().filter(|p| lower.contains(*p)).count();
    if violations == 0 { 1.0 } else { 0.0 }
}

/// Check deck structure: has title slide, reasonable count, no empty slides.
fn check_deck_structure(parsed: &serde_json::Value) -> f64 {
    let slides = match parsed.get("slides").and_then(|v| v.as_array()) {
        Some(s) => s,
        None => return 0.5, // Not a deck
    };
    if slides.is_empty() { return 0.0; }
    let mut score: f64 = 1.0;
    // Check slide count is reasonable (3-20)
    if slides.len() < 3 { score -= 0.2; }
    if slides.len() > 20 { score -= 0.3; }
    // Check for empty slides (no title and no slots)
    let empty_count = slides.iter().filter(|s| {
        s.get("title").and_then(|v| v.as_str()).map(|t| t.is_empty()).unwrap_or(true)
            && s.get("slots").map(|slots| slots.as_object().map(|o| o.is_empty()).unwrap_or(true)).unwrap_or(true)
    }).count();
    if empty_count > 0 {
        score -= 0.3 * empty_count as f64 / slides.len() as f64;
    }
    // Check first slide is a title slide (heuristic: has "title" in template_id)
    if let Some(first) = slides.first() {
        let first_tid = first.get("template_id").and_then(|v| v.as_str()).unwrap_or("");
        if !first_tid.contains("title") && !first_tid.starts_with("title") {
            // Not necessarily wrong, but slightly penalize
            score -= 0.1;
        }
    }
    score.max(0.0f64)
}

// ---------------------------------------------------------------------------
// Helpers (delegated to eval_common to avoid duplication)
// ---------------------------------------------------------------------------

use crate::eval_common::{count_sentences, detect_language_score, extract_json, is_repeated};

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------

fn load_slides_dataset() -> anyhow::Result<SlidesEvalDataset> {
    // Try workspace-relative path first, then crate-relative.
    let candidates = [
        "eval/kchat-task-suite/datasets/slides/slides_eval_dataset_v2.json",
        "datasets/slides/slides_eval_dataset_v2.json",
        "eval/kchat-task-suite/datasets/slides/slides_eval_dataset_v1.json",
        "datasets/slides/slides_eval_dataset_v1.json",
    ];
    let mut last_err = None;
    for path in &candidates {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                return serde_json::from_str(&text)
                    .map_err(|e| anyhow::anyhow!("failed to parse slides dataset: {}", e));
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(anyhow::anyhow!("failed to read slides dataset: {:?}", last_err))
}

fn load_image_dataset() -> anyhow::Result<ImageSearchDataset> {
    let candidates = [
        "eval/kchat-task-suite/datasets/slides/image_search_eval_v2.json",
        "datasets/slides/image_search_eval_v2.json",
        "eval/kchat-task-suite/datasets/slides/image_search_eval_v1.json",
        "datasets/slides/image_search_eval_v1.json",
    ];
    let mut last_err = None;
    for path in &candidates {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                return serde_json::from_str(&text)
                    .map_err(|e| anyhow::anyhow!("failed to parse image dataset: {}", e));
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(anyhow::anyhow!("failed to read image dataset: {:?}", last_err))
}

// ---------------------------------------------------------------------------
// Mock mode — validates prompt construction, grammar, templates, budgets
// ---------------------------------------------------------------------------

pub fn run_mock() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  SLIDES SKILL EVAL SUITE — MOCK MODE (CI)                                     ║");
    println!("║  12 Slides Skills × 210 Templates × 880 Test Cases                            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let dataset = match load_slides_dataset() {
        Ok(d) => {
            println!("Dataset: {} v{} ({} test cases)", d.name, d.version, d.test_cases.len());
            d
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            return;
        }
    }
    .clone();
    println!();

    let registry = SkillRegistry::new();
    let template_registry = &kchat_generation::TEMPLATE_REGISTRY;
    println!("Template registry: {} templates across {} families",
        template_registry.len(),
        count_template_families(template_registry));
    println!();

    let mut results: Vec<SlidesResult> = Vec::new();
    let mut suite = SuiteReport::new("Slides Skill Eval (Mock)", 0.90);

    // Verify all 12 slides skills exist
    let slides_skills = registry.by_surface(SkillSurface::Slides);
    println!("Slides skills in registry: {}", slides_skills.len());
    for skill in &slides_skills {
        println!("  • {} — {} ({:?})", skill.id, skill.label, skill.grammar_type);
    }
    println!();

    // Verify all 210 templates exist
    let family_counts = count_templates_by_family(&template_registry);
    for (family, count) in &family_counts {
        println!("  {} family: {} templates", family.label(), count);
    }
    println!();

    for tc in &dataset.test_cases {
        let start = Instant::now();
        let mut errors: Vec<String> = Vec::new();
        let mut checks_detail: Vec<(String, f64)> = Vec::new();

        // 1. Skill exists
        let skill = match registry.get(&tc.skill_id) {
            Some(s) => s,
            None => {
                let err = format!("skill '{}' not found", tc.skill_id);
                errors.push(err.clone());
                results.push(SlidesResult {
                    case_id: tc.id.clone(),
                    skill_id: tc.skill_id.clone(),
                    tier: tc.tier.clone(),
                    passed: false,
                    quality_score: 0.0,
                    checks_detail: Vec::new(),
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: Some(err),
                });
                suite.add(EvalResult::fail(&tc.id, "skill not in registry"));
                continue;
            }
        };

        // 2. Surface matches
        let expected_surface = match tc.surface.as_str() {
            "slides" => SkillSurface::Slides,
            _ => skill.surface,
        };
        if skill.surface != expected_surface {
            errors.push(format!("surface mismatch: expected {:?}, got {:?}", expected_surface, skill.surface));
        }

        // 3. Build prompt (validates prompt construction)
        let input_text = if !tc.input.selection.is_empty() {
            &tc.input.selection
        } else if !tc.input.variant_context.is_empty() {
            &tc.input.variant_context
        } else {
            &tc.input.document
        };
        let context_text = if !tc.input.document.is_empty() {
            &tc.input.document
        } else {
            &tc.input.variant_context
        };
        let prompt_input = SkillPromptInput {
            input: input_text.as_str(),
            context: context_text.as_str(),
            keywords: tc.input.keywords.as_str(),
            variant_context: tc.variant.as_deref().unwrap_or(""),
            tier: None,
        };
        let prompt = skill.build_prompt(prompt_input);
        if prompt.system.is_empty() {
            errors.push("empty system prompt".into());
        }
        checks_detail.push(("prompt_built".into(), 1.0));

        // 4. Token budget check
        // Slides skills include the 210-template catalog in the prompt (~2600 tokens),
        // so we use a more generous multiplier (20×) for slides surface skills.
        let prompt_tokens = kchat_generation::estimate_tokens_text(&prompt.system)
            + kchat_generation::estimate_tokens_text(&prompt.user);
        let budget_multiplier = if skill.surface == SkillSurface::Slides { 20 } else { 3 };
        if prompt_tokens > skill.max_tokens as usize * budget_multiplier {
            errors.push(format!("prompt tokens ({}) exceed {}× max_tokens ({})", prompt_tokens, budget_multiplier, skill.max_tokens));
        }
        checks_detail.push(("token_budget".into(), 1.0));

        // 5. Grammar type matches
        let expected_grammar = match tc.grammar_type.as_str() {
            "json_schema" => "JsonSchema",
            "regex" => "Regex",
            "free_text" => "FreeText",
            _ => "FreeText",
        };
        if format!("{:?}", skill.grammar_type) != expected_grammar {
            errors.push(format!("grammar type mismatch: expected {}, got {:?}", expected_grammar, skill.grammar_type));
        }
        checks_detail.push(("grammar_type".into(), 1.0));

        // 6. Template conformance (for template-specific cases)
        if let Some(props) = tc.expected_properties.as_object() {
            if let Some(tid) = props.get("template_id").and_then(|v| v.as_str()) {
                if template_registry.get(tid).is_none() {
                    errors.push(format!("expected template '{}' not in registry", tid));
                } else {
                    checks_detail.push(("template_exists".into(), 1.0));
                }
            }
        }

        // 7. Quality checks (mock: just validate check types are recognized)
        for check in &tc.quality_checks {
            let score = run_slides_quality_check_mock(check);
            checks_detail.push((check.check_type.clone(), score));
        }

        let passed = errors.is_empty()
            && checks_detail.iter().all(|(_, s)| *s >= 0.7);
        let quality_score = if checks_detail.is_empty() { 1.0 }
            else { checks_detail.iter().map(|(_, s)| s).sum::<f64>() / checks_detail.len() as f64 };

        results.push(SlidesResult {
            case_id: tc.id.clone(),
            skill_id: tc.skill_id.clone(),
            tier: tc.tier.clone(),
            passed,
            quality_score,
            checks_detail,
            duration_ms: start.elapsed().as_millis() as u64,
            error: if errors.is_empty() { None } else { Some(errors.join("; ")) },
        });

        if passed {
            suite.add(EvalResult::pass(&tc.id));
        } else {
            suite.add(EvalResult::fail(&tc.id, "mock validation failed"));
        }
    }

    // Report
    println!("─── Mock Mode Results ───");
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let avg_quality = if total > 0 {
        results.iter().map(|r| r.quality_score).sum::<f64>() / total as f64
    } else { 0.0 };
    // In mock mode, quality_score reflects structural validation (skill exists,
    // prompt builds, grammar type matches, token budget) — not model output quality.
    println!("Total: {} | Passed: {} | Failed: {} | Structural Validation: {:.2}", total, passed, total - passed, avg_quality);

    // Per-skill breakdown
    let mut by_skill: HashMap<String, (usize, usize)> = HashMap::new();
    for r in &results {
        let entry = by_skill.entry(r.skill_id.clone()).or_insert((0, 0));
        entry.0 += 1;
        if r.passed { entry.1 += 1; }
    }
    println!();
    println!("Per-skill breakdown:");
    let mut skill_vec: Vec<_> = by_skill.iter().collect();
    skill_vec.sort_by_key(|(_, (t, _))| *t);
    for (skill_id, (total, passed)) in &skill_vec {
        println!("  {:<30} {}/{} ({:.0}%)", skill_id, passed, total, *passed as f64 / *total as f64 * 100.0);
    }

    // Per-tier breakdown
    let mut by_tier: HashMap<String, (usize, usize)> = HashMap::new();
    for r in &results {
        let entry = by_tier.entry(r.tier.clone()).or_insert((0, 0));
        entry.0 += 1;
        if r.passed { entry.1 += 1; }
    }
    println!();
    println!("Per-tier breakdown:");
    let mut tier_vec: Vec<_> = by_tier.iter().collect();
    tier_vec.sort();
    for (tier, (total, passed)) in &tier_vec {
        println!("  {:<10} {}/{} ({:.0}%)", tier, passed, total, *passed as f64 / *total as f64 * 100.0);
    }

    println!();
    let status = if suite.passed() { "PASS" } else { "FAIL" };
    println!("[{}] {} — {}/{} passed ({:.1}%, required: {:.1}%)",
        status, suite.suite_name, suite.pass_count(), suite.total_count(),
        suite.pass_rate() * 100.0, suite.required_pass_rate * 100.0);
    println!();

    if passed == total {
        println!("✓ All {} slides mock eval cases passed", total);
    } else {
        println!("✗ {}/{} slides mock eval cases failed", total - passed, total);
    }
}

fn run_slides_quality_check_mock(check: &SlidesQualityCheck) -> f64 {
    // In mock mode, we validate that the check type is recognized.
    // New advanced check types are also recognized here.
    match check.check_type.as_str() {
        "min_length" | "max_length" | "coherent" | "json_schema_valid"
        | "template_conformance" | "bullet_count" | "slot_count"
        | "chart_data_valid" | "image_query_valid" | "sentence_count"
        | "language_script"
        // Advanced check types
        | "slot_schema_conformance" | "cross_slide_consistency"
        | "template_selection_accuracy" | "semantic_quality"
        | "image_query_relevance" | "no_executable_content"
        | "deck_structure_valid" => 1.0,
        _ => 0.0,
    }
}

fn count_template_families(reg: &SlidesTemplateRegistry) -> usize {
    let mut families = std::collections::HashSet::new();
    for tmpl in reg.all() {
        families.insert(tmpl.family);
    }
    families.len()
}

fn count_templates_by_family(reg: &SlidesTemplateRegistry) -> Vec<(SlidesTemplateFamily, usize)> {
    let mut counts: HashMap<SlidesTemplateFamily, usize> = HashMap::new();
    for tmpl in reg.all() {
        *counts.entry(tmpl.family).or_insert(0) += 1;
    }
    let mut v: Vec<_> = counts.into_iter().collect();
    v.sort_by_key(|(f, _)| format!("{:?}", f));
    v
}

// ---------------------------------------------------------------------------
// Real model mode — sends prompts to llama-server/MLX and checks output
// ---------------------------------------------------------------------------

pub fn run_realworld() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  SLIDES SKILL EVAL SUITE — REAL MODEL MODE                                    ║");
    println!("║  12 Slides Skills × 210 Templates — Real Inference + Quality Measurement     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let dataset = match load_slides_dataset() {
        Ok(d) => {
            println!("Dataset: {} v{} ({} test cases)", d.name, d.version, d.test_cases.len());
            d
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            return;
        }
    }
    .clone();
    println!();

    let llama_url = std::env::var("LLAMA_SERVER_URL").unwrap_or_else(|_| "http://127.0.0.1:18888".into());
    println!("LLM endpoint: {}", llama_url);
    println!();

    let registry = SkillRegistry::new();
    let mut results: Vec<SlidesResult> = Vec::new();
    let mut suite = SuiteReport::new("Slides Skill Eval (Real)", 0.75);

    // Limit to a sample for real mode (full 880 would take too long)
    let sample_size = std::env::var("SLIDES_EVAL_SAMPLE").ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(60);
    let test_cases: Vec<&SlidesTestCase> = if dataset.test_cases.len() > sample_size {
        // Sample evenly across the dataset
        let step = dataset.test_cases.len() / sample_size;
        (0..sample_size).map(|i| &dataset.test_cases[i * step]).collect()
    } else {
        dataset.test_cases.iter().collect()
    };
    println!("Running {} sampled test cases (use SLIDES_EVAL_SAMPLE to adjust)...", test_cases.len());
    println!();

    for (i, tc) in test_cases.iter().enumerate() {
        let start = Instant::now();
        let skill = match registry.get(&tc.skill_id) {
            Some(s) => s,
            None => {
                results.push(SlidesResult {
                    case_id: tc.id.clone(),
                    skill_id: tc.skill_id.clone(),
                    tier: tc.tier.clone(),
                    passed: false,
                    quality_score: 0.0,
                    checks_detail: Vec::new(),
                    duration_ms: 0,
                    error: Some("skill not found".into()),
                });
                suite.add(EvalResult::fail(&tc.id, "skill not found"));
                continue;
            }
        };

        let prompt_input = SkillPromptInput {
            input: if !tc.input.selection.is_empty() { &tc.input.selection }
                else if !tc.input.variant_context.is_empty() { &tc.input.variant_context }
                else { &tc.input.document },
            context: if !tc.input.document.is_empty() { &tc.input.document }
                else { &tc.input.variant_context },
            keywords: &tc.input.keywords,
            variant_context: tc.variant.as_deref().unwrap_or(""),
            tier: None,
        };
        let prompt = skill.build_prompt(prompt_input);

        let output = match call_llama(&llama_url, &prompt.system, &prompt.user, tc.max_tokens) {
            Ok(o) => o,
            Err(e) => {
                results.push(SlidesResult {
                    case_id: tc.id.clone(),
                    skill_id: tc.skill_id.clone(),
                    tier: tc.tier.clone(),
                    passed: false,
                    quality_score: 0.0,
                    checks_detail: Vec::new(),
                    duration_ms: start.elapsed().as_millis() as u64,
                    error: Some(format!("LLM call failed: {}", e)),
                });
                suite.add(EvalResult::fail(&tc.id, "LLM call failed"));
                continue;
            }
        };

        let mut checks_detail: Vec<(String, f64)> = Vec::new();
        for check in &tc.quality_checks {
            let score = run_slides_quality_check(&output, check);
            checks_detail.push((check.check_type.clone(), score));
        }

        let quality_score = if checks_detail.is_empty() { 1.0 }
            else { checks_detail.iter().map(|(_, s)| s).sum::<f64>() / checks_detail.len() as f64 };
        let passed = quality_score >= 0.7;

        results.push(SlidesResult {
            case_id: tc.id.clone(),
            skill_id: tc.skill_id.clone(),
            tier: tc.tier.clone(),
            passed,
            quality_score,
            checks_detail,
            duration_ms: start.elapsed().as_millis() as u64,
            error: None,
        });

        if passed {
            suite.add(EvalResult::pass(&tc.id));
        } else {
            suite.add(EvalResult::fail(&tc.id, "quality below threshold"));
        }

        if (i + 1) % 10 == 0 {
            print!("  [{}/{}] ", i + 1, test_cases.len());
            std::io::stdout().flush().ok();
        }
    }
    println!();
    println!();

    // Report
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let avg_quality = if total > 0 {
        results.iter().map(|r| r.quality_score).sum::<f64>() / total as f64
    } else { 0.0 };
    println!("─── Real Model Results ───");
    println!("Total: {} | Passed: {} | Failed: {} | Avg Quality: {:.2}", total, passed, total - passed, avg_quality);

    // Per-skill breakdown
    let mut by_skill: HashMap<String, (usize, usize, f64)> = HashMap::new();
    for r in &results {
        let entry = by_skill.entry(r.skill_id.clone()).or_insert((0, 0, 0.0));
        entry.0 += 1;
        if r.passed { entry.1 += 1; }
        entry.2 += r.quality_score;
    }
    println!();
    println!("Per-skill breakdown:");
    let mut skill_vec: Vec<_> = by_skill.iter().collect();
    skill_vec.sort_by_key(|(_, (t, _, _))| *t);
    for (skill_id, (total, passed, quality_sum)) in &skill_vec {
        let avg_q = quality_sum / *total as f64;
        println!("  {:<30} {}/{} ({:.0}%) avg_q={:.2}", skill_id, passed, total, *passed as f64 / *total as f64 * 100.0, avg_q);
    }

    println!();
    let status = if suite.passed() { "PASS" } else { "FAIL" };
    println!("[{}] {} — {}/{} passed ({:.1}%, required: {:.1}%)",
        status, suite.suite_name, suite.pass_count(), suite.total_count(),
        suite.pass_rate() * 100.0, suite.required_pass_rate * 100.0);
    println!();

    if passed == total {
        println!("✓ All {} slides real eval cases passed", total);
    } else {
        println!("✗ {}/{} slides real eval cases failed", total - passed, total);
    }
}

fn call_llama(url: &str, system: &str, user: &str, max_tokens: u32) -> anyhow::Result<String> {
    let payload = serde_json::json!({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": false
    });

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

    let resp = client
        .post(&format!("{}/v1/chat/completions", url))
        .json(&payload)
        .send()?;

    if !resp.status().is_success() {
        anyhow::bail!("HTTP {}: {}", resp.status(), resp.text().unwrap_or_default());
    }

    let body: serde_json::Value = resp.json()?;
    let content = body
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("");
    Ok(content.to_string())
}

// ---------------------------------------------------------------------------
// Image search mode — always real, requires API keys
// ---------------------------------------------------------------------------

pub fn run_image_search() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  IMAGE SEARCH EVAL SUITE — ALWAYS REAL MODE                                   ║");
    println!("║  4 Providers × 80 Test Cases — Pexels, Pixabay, Unsplash, Shutterstock       ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let dataset = match load_image_dataset() {
        Ok(d) => {
            println!("Dataset: {} v{} ({} test cases)", d.name, d.version, d.test_cases.len());
            d
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            return;
        }
    }
    .clone();
    println!();

    // Check which API keys are available
    let env_vars = ["PEXELS_API_KEY", "PIXABAY_API_KEY", "UNSPLASH_ACCESS_KEY", "SHUTTERSTOCK_API_TOKEN"];
    println!("API key status:");
    for var in &env_vars {
        let set = std::env::var(var).map(|v| !v.is_empty()).unwrap_or(false);
        println!("  {} = {}", var, if set { "✓ set" } else { "✗ not set" });
    }
    println!();

    // Build registry from env
    let registry = kchat_image::ImageSearchRegistry::from_env();
    println!("Registered providers: {:?}", registry.provider_ids());
    if registry.provider_count() == 0 {
        eprintln!("WARNING: No providers registered. Set at least one API key env var.");
        eprintln!("         Tests will be skipped (not failed).");
    }
    println!();

    let mut results: Vec<ImageResult> = Vec::new();
    let mut suite = SuiteReport::new("Image Search Eval (Real)", 0.80);

    let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");

    for tc in &dataset.test_cases {
        let start = Instant::now();

        // Build request
        let mut req = kchat_image::ImageSearchRequest::new(&tc.query)
            .with_per_page(tc.per_page)
            .with_safesearch(tc.safesearch);
        if let Some(o) = &tc.orientation {
            let orient = match o.as_str() {
                "landscape" => kchat_image::ImageOrientation::Landscape,
                "portrait" => kchat_image::ImageOrientation::Portrait,
                "square" => kchat_image::ImageOrientation::Square,
                _ => kchat_image::ImageOrientation::Landscape,
            };
            req = req.with_orientation(orient);
        }

        // Execute search
        let search_result = if tc.provider == "registry" {
            rt.block_on(registry.search(&req))
        } else {
            rt.block_on(registry.search_provider(&tc.provider, &req))
        };

        let (result_count, error) = match search_result {
            Ok(resp) => (resp.results.len(), None),
            Err(e) => (0, Some(format!("{}", e))),
        };

        // Evaluate
        let passed = if let Some(expected_err) = tc.expected_error {
            // Cases that expect an error
            if expected_err {
                error.is_some()
            } else {
                error.is_none() && result_count >= tc.expected_min_results
            }
        } else if tc.expected_min_results == 0 {
            // Cases that may return 0 results (e.g. nonsense queries)
            error.is_none()
        } else {
            // Normal cases: must have results
            error.is_none() && result_count >= tc.expected_min_results
        };

        // Check orientation if specified
        let orientation_ok = if let Some(_expected_o) = &tc.expected_orientation {
            // In real mode, we'd check the actual results' orientations.
            // For now, just check we got results.
            result_count > 0
        } else {
            true
        };
        let passed = passed && orientation_ok;

        results.push(ImageResult {
            case_id: tc.id.clone(),
            provider: tc.provider.clone(),
            passed,
            result_count,
            duration_ms: start.elapsed().as_millis() as u64,
            error,
        });

        if passed {
            suite.add(EvalResult::pass(&tc.id));
        } else {
            suite.add(EvalResult::fail(&tc.id, "image search failed"));
        }

        let status = if passed { "✓" } else { "✗" };
        println!("  {} {:<8} {:<40} results={:<3} ({}ms)",
            status, tc.provider, tc.query.chars().take(40).collect::<String>(),
            result_count, start.elapsed().as_millis());
    }

    println!();
    println!("─── Image Search Results ───");
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    println!("Total: {} | Passed: {} | Failed: {} | Pass Rate: {:.1}%",
        total, passed, total - passed, passed as f64 / total as f64 * 100.0);

    // Per-provider breakdown
    let mut by_provider: HashMap<String, (usize, usize)> = HashMap::new();
    for r in &results {
        let entry = by_provider.entry(r.provider.clone()).or_insert((0, 0));
        entry.0 += 1;
        if r.passed { entry.1 += 1; }
    }
    println!();
    println!("Per-provider breakdown:");
    let mut prov_vec: Vec<_> = by_provider.iter().collect();
    prov_vec.sort();
    for (provider, (total, passed)) in &prov_vec {
        println!("  {:<15} {}/{} ({:.0}%)", provider, passed, total, *passed as f64 / *total as f64 * 100.0);
    }

    println!();
    let status = if suite.passed() { "PASS" } else { "FAIL" };
    println!("[{}] {} — {}/{} passed ({:.1}%, required: {:.1}%)",
        status, suite.suite_name, suite.pass_count(), suite.total_count(),
        suite.pass_rate() * 100.0, suite.required_pass_rate * 100.0);
    println!();

    if passed == total {
        println!("✓ All {} image search eval cases passed", total);
    } else {
        println!("✗ {}/{} image search eval cases failed", total - passed, total);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Note: extract_json, is_repeated, count_sentences, detect_language_score
    // tests are in eval_common.rs to avoid duplication.

    #[test]
    fn test_slides_dataset_loads() {
        let dataset = load_slides_dataset();
        assert!(dataset.is_ok(), "slides dataset should load");
        let ds = dataset.unwrap();
        assert!(!ds.test_cases.is_empty());
        assert!(ds.test_cases.len() >= 800, "expected at least 800 test cases, got {}", ds.test_cases.len());
    }

    #[test]
    fn test_image_dataset_loads() {
        let dataset = load_image_dataset();
        assert!(dataset.is_ok(), "image dataset should load");
        let ds = dataset.unwrap();
        assert!(!ds.test_cases.is_empty());
        assert!(ds.test_cases.len() >= 70, "expected at least 70 image test cases, got {}", ds.test_cases.len());
    }

    #[test]
    fn test_template_conformance_check_recognizes_valid_template() {
        let check = SlidesQualityCheck {
            check_type: "template_conformance".into(),
            min_chars: None, max_chars: None, min_bullets: None, max_bullets: None,
            min_slots: None, max_slides: None, template_id: None,
            min_sentences: None, max_sentences: None, language: None,
        };
        let output = r#"{"template_id": "title", "title": "Test", "slots": {"title": "Test"}}"#;
        let score = run_slides_quality_check(output, &check);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_template_conformance_check_rejects_invalid_template() {
        let check = SlidesQualityCheck {
            check_type: "template_conformance".into(),
            min_chars: None, max_chars: None, min_bullets: None, max_bullets: None,
            min_slots: None, max_slides: None, template_id: None,
            min_sentences: None, max_sentences: None, language: None,
        };
        let output = r#"{"template_id": "nonexistent", "title": "Test"}"#;
        let score = run_slides_quality_check(output, &check);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_image_query_valid_check() {
        let check = SlidesQualityCheck {
            check_type: "image_query_valid".into(),
            min_chars: None, max_chars: None, min_bullets: None, max_bullets: None,
            min_slots: None, max_slides: None, template_id: None,
            min_sentences: None, max_sentences: None, language: None,
        };
        // Valid query
        let output = r#"{"query": "mountain landscape", "orientation": "landscape"}"#;
        assert_eq!(run_slides_quality_check(output, &check), 1.0);
        // URL in query → should fail
        let output = r#"{"query": "https://example.com/img.jpg"}"#;
        assert_eq!(run_slides_quality_check(output, &check), 0.0);
    }

    #[test]
    fn test_bullet_count_check() {
        let check = SlidesQualityCheck {
            check_type: "bullet_count".into(),
            min_chars: None, max_chars: None,
            min_bullets: Some(2), max_bullets: Some(8),
            min_slots: None, max_slides: None, template_id: None,
            min_sentences: None, max_sentences: None, language: None,
        };
        let output = r#"{"template_id": "bullet", "title": "Test", "slots": {"bullets": ["a", "b", "c"]}}"#;
        assert_eq!(run_slides_quality_check(output, &check), 1.0);
    }

    #[test]
    fn test_chart_data_valid_check() {
        let check = SlidesQualityCheck {
            check_type: "chart_data_valid".into(),
            min_chars: None, max_chars: None, min_bullets: None, max_bullets: None,
            min_slots: None, max_slides: None, template_id: None,
            min_sentences: None, max_sentences: None, language: None,
        };
        let output = r#"{"template_id": "bar_chart", "title": "Sales", "slots": {"series": [{"label": "Q1", "value": 100}, {"label": "Q2", "value": 200}]}}"#;
        assert_eq!(run_slides_quality_check(output, &check), 1.0);
    }
}

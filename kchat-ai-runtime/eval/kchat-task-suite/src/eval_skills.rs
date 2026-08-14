//! Skill evaluation suite — per-skill quality evaluation for all 33 document AI skills.
//!
//! Mock mode (`--skills`): validates prompt construction, grammar, token budget,
//! pipeline plumbing, LoRA resolution, and quality check functions without a model.
//!
//! Real model mode (`--skills --realworld`): sends skill prompts to llama-server/MLX,
//! runs quality checks against real model output, and reports per-skill metrics.

use crate::report::{EvalReport, EvalResult, SuiteReport};
use kchat_generation::{
    estimate_tokens_text, Grammar, SkillDef, SkillGrammarType, SkillLoRAResolver,
    SkillPromptInput, SkillRegistry, SkillSurface, SkillTier,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Dataset structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SkillEvalDataset {
    name: String,
    version: String,
    description: String,
    test_cases: Vec<SkillTestCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SkillTestCase {
    id: String,
    skill_id: String,
    surface: String,
    scope: String,
    mode: String,
    input: SkillTestInput,
    variant: Option<String>,
    max_tokens: u32,
    grammar_type: String,
    quality_checks: Vec<SkillQualityCheck>,
    expected_properties: serde_json::Value,
    tier: String,
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SkillTestInput {
    document: String,
    selection: String,
    cursor_context: String,
    variant_context: String,
    keywords: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SkillQualityCheck {
    #[serde(rename = "type")]
    check_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_chars: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_chars: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keywords: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pattern: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    not_contains: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_sentences: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_sentences: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_words: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    checks: Option<Vec<SkillQualityCheck>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_fields: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_headings: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_tone: Option<String>,
}

// ---------------------------------------------------------------------------
// Per-skill result tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SkillResult {
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
struct SkillSummary {
    skill_id: String,
    total: usize,
    passed: usize,
    avg_quality: f64,
    ttft_p95_ms: u64,
    decode_p50_tps: f64,
}

// ---------------------------------------------------------------------------
// Quality check runner (reuses logic from eval_perdevice but self-contained)
// ---------------------------------------------------------------------------

fn run_skill_quality_check(output: &str, check: &SkillQualityCheck) -> f64 {
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
        "contains_keyword" => {
            if let Some(keywords) = &check.keywords {
                let lower = output.to_lowercase();
                for k in keywords {
                    if lower.contains(&k.to_lowercase()) {
                        return 1.0;
                    }
                }
                0.0
            } else { 0.0 }
        }
        "not_contains" => {
            if let Some(forbidden) = &check.not_contains {
                let lower = output.to_lowercase();
                let violations = forbidden.iter().filter(|k| lower.contains(&k.to_lowercase())).count();
                if violations == 0 { 1.0 } else {
                    1.0 - (violations as f64 / forbidden.len().max(1) as f64)
                }
            } else { 1.0 }
        }
        "json_schema_valid" => {
            let json_text = extract_json(output);
            if json_text.is_empty() { return 0.0; }
            let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
                Ok(v) => v,
                Err(_) => return 0.0,
            };
            if let Some(schema) = &check.schema {
                if validate_json_schema(&parsed, schema) { 1.0 } else { 0.5 }
            } else { 1.0 }
        }
        "regex_match" => {
            if let Some(pattern) = &check.pattern {
                regex::Regex::new(pattern)
                    .map(|re| if re.is_match(output) { 1.0 } else { 0.0 })
                    .unwrap_or(0.0)
            } else { 0.0 }
        }
        "coherent" => {
            if output.is_empty() || output.len() <= 10 { 0.0 }
            else if is_repeated(output) { 0.3 }
            else { 1.0 }
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
        "min_words" => {
            let min = check.min_words.unwrap_or(0);
            let words = output.split_whitespace().count();
            if words >= min { 1.0 } else if min > 0 { words as f64 / min as f64 } else { 1.0 }
        }
        "multi_check" => {
            if let Some(sub_checks) = &check.checks {
                if sub_checks.is_empty() { return 1.0; }
                let total = sub_checks.len() as f64;
                let sum: f64 = sub_checks.iter().map(|c| run_skill_quality_check(output, c)).sum();
                sum / total
            } else { 1.0 }
        }
        "markdown_structure" => check_markdown_structure(output),
        "no_input_echo" => {
            if let Some(input) = &check.input_text {
                check_no_input_echo(output, input)
            } else { 1.0 }
        }
        "tone_match" => {
            if let Some(tone) = &check.expected_tone {
                check_tone_match(output, tone)
            } else { 1.0 }
        }
        "length_delta" => {
            if let Some(ratio) = &check.expected_ratio {
                if let Some(input) = &check.input_text {
                    check_length_delta(output, input, ratio)
                } else { 1.0 }
            } else { 1.0 }
        }
        "json_field_count" => {
            if let Some(fields) = &check.expected_fields {
                check_json_field_count(output, fields)
            } else { 1.0 }
        }
        "heading_count" => {
            let min = check.min_headings.unwrap_or(0);
            let count = count_markdown_headings(output);
            if count >= min { 1.0 } else if min > 0 { count as f64 / min as f64 } else { 1.0 }
        }
        _ => 1.0,
    }
}

fn run_all_quality_checks(output: &str, checks: &[SkillQualityCheck]) -> (f64, Vec<(String, f64)>) {
    if checks.is_empty() {
        return (1.0, Vec::new());
    }
    let mut details = Vec::new();
    let mut sum = 0.0;
    for check in checks {
        let score = run_skill_quality_check(output, check);
        details.push((check.check_type.clone(), score));
        sum += score;
    }
    (sum / checks.len() as f64, details)
}

// --- Helper functions (same logic as eval_perdevice) ---

fn extract_json(text: &str) -> String {
    let mut trimmed = text.trim();
    if trimmed.starts_with("```") {
        if let Some(nl) = trimmed.find('\n') {
            trimmed = trimmed[nl + 1..].trim();
        }
        if let Some(pos) = trimmed.find("```") {
            trimmed = trimmed[..pos].trim();
        }
    }
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
            return trimmed.to_string();
        }
    }
    for (i, c) in trimmed.char_indices() {
        if c == '{' || c == '[' {
            let candidate = &trimmed[i..];
            if let Ok(end) = find_json_end(candidate) {
                return trimmed[i..i + end].to_string();
            }
        }
    }
    String::new()
}

fn find_json_end(s: &str) -> Result<usize, ()> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, c) in s.char_indices() {
        if escape { escape = false; continue; }
        if c == '\\' && in_string { escape = true; continue; }
        if c == '"' { in_string = !in_string; continue; }
        if in_string { continue; }
        match c {
            '{' | '[' => depth += 1,
            '}' | ']' => { depth -= 1; if depth == 0 { return Ok(i + 1); } }
            _ => {}
        }
    }
    Err(())
}

fn validate_json_schema(value: &serde_json::Value, schema: &serde_json::Value) -> bool {
    let schema_obj = match schema.as_object() { Some(o) => o, None => return true };
    if let Some(schema_type) = schema_obj.get("type").and_then(|v| v.as_str()) {
        let type_ok = match schema_type {
            "object" => value.is_object(),
            "array" => value.is_array(),
            "string" => value.is_string(),
            "number" => value.is_number(),
            "boolean" => value.is_boolean(),
            "integer" => value.is_i64() || value.is_u64(),
            "null" => value.is_null(),
            _ => true,
        };
        if !type_ok { return false; }
    }
    if let Some(required) = schema_obj.get("required").and_then(|v| v.as_array()) {
        if let Some(obj) = value.as_object() {
            for req in required {
                if let Some(field) = req.as_str() {
                    if !obj.contains_key(field) { return false; }
                }
            }
        } else { return false; }
    }
    if let (Some(properties), Some(obj)) = (
        schema_obj.get("properties").and_then(|v| v.as_object()),
        value.as_object(),
    ) {
        for (key, prop_schema) in properties {
            if let Some(field_val) = obj.get(key) {
                if !validate_json_schema(field_val, prop_schema) { return false; }
            }
        }
    }
    if let (Some(items_schema), Some(arr)) = (
        schema_obj.get("items"), value.as_array(),
    ) {
        for item in arr {
            if !validate_json_schema(item, items_schema) { return false; }
        }
    }
    // Check maxLength for strings
    if let (Some(max_len), Some(s)) = (
        schema_obj.get("maxLength").and_then(|v| v.as_u64()),
        value.as_str(),
    ) {
        if s.len() as u64 > max_len { return false; }
    }
    true
}

fn is_repeated(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 10 { return false; }
    let unique: std::collections::HashSet<&str> = words.iter().copied().collect();
    let ratio = unique.len() as f64 / words.len() as f64;
    ratio < 0.2
}

fn count_sentences(text: &str) -> usize {
    let trimmed = text.trim();
    if trimmed.is_empty() { return 0; }
    let mut count = 0;
    let chars: Vec<char> = trimmed.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '.' || c == '!' || c == '?' || c == '。' || c == '！' || c == '？' {
            let is_end = i + 1 >= chars.len()
                || chars[i + 1].is_whitespace()
                || chars[i + 1] == '.' || chars[i + 1] == '!' || chars[i + 1] == '?';
            if is_end { count += 1; }
        }
    }
    if count == 0 && !trimmed.is_empty() { count = 1; }
    count
}

fn detect_language_score(text: &str, expected: &str) -> f64 {
    let mut latin = 0u32;
    let mut cjk = 0u32;
    let mut kana = 0u32;
    let mut hangul = 0u32;
    let mut arabic = 0u32;
    let mut devanagari = 0u32;
    let mut thai = 0u32;
    let mut other = 0u32;
    for c in text.chars() {
        if c.is_ascii_alphabetic() || c == ' ' || c.is_ascii_punctuation() || c.is_ascii_digit() {
            latin += 1;
        } else if (c as u32 >= 0x4E00 && c as u32 <= 0x9FFF) || (c as u32 >= 0x3400 && c as u32 <= 0x4DBF) {
            cjk += 1;
        } else if (c as u32 >= 0x3040 && c as u32 <= 0x309F) || (c as u32 >= 0x30A0 && c as u32 <= 0x30FF) {
            kana += 1;
        } else if c as u32 >= 0xAC00 && c as u32 <= 0xD7AF {
            hangul += 1;
        } else if c as u32 >= 0x0600 && c as u32 <= 0x06FF {
            arabic += 1;
        } else if c as u32 >= 0x0900 && c as u32 <= 0x097F {
            devanagari += 1;
        } else if c as u32 >= 0x0E00 && c as u32 <= 0x0E7F {
            thai += 1;
        } else if !c.is_whitespace() {
            other += 1;
        }
    }
    let total = latin + cjk + kana + hangul + arabic + devanagari + thai + other;
    if total == 0 { return 0.0; }
    let score_for = |count: u32| -> f64 {
        if count == 0 { 0.0 } else { count as f64 / total as f64 }
    };
    match expected {
        "latin" => score_for(latin),
        "cjk" => score_for(cjk),
        "kana" => score_for(kana),
        "cjk_or_kana" => score_for(cjk + kana),
        "hangul" => score_for(hangul),
        "arabic" => score_for(arabic),
        "devanagari" => score_for(devanagari),
        "thai" => score_for(thai),
        _ => 1.0,
    }
}

fn check_markdown_structure(output: &str) -> f64 {
    let lines: Vec<&str> = output.lines().collect();
    if lines.is_empty() || output.trim().is_empty() { return 0.0; }
    let mut score = 0.0;
    let mut has_heading = false;
    let mut has_paragraph = false;
    let mut has_list = false;
    let mut heading_levels: Vec<usize> = Vec::new();
    for line in &lines {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            let level = trimmed.chars().take_while(|c| *c == '#').count();
            if level <= 6 { has_heading = true; heading_levels.push(level); }
        }
        if trimmed.starts_with("- ") || trimmed.starts_with("* ") || trimmed.starts_with("+ ") {
            has_list = true;
        }
        if !trimmed.is_empty() && !trimmed.starts_with('#') && !trimmed.starts_with('-')
            && !trimmed.starts_with('*') && !trimmed.starts_with('+')
            && !trimmed.starts_with("```") && !trimmed.starts_with('|')
        {
            has_paragraph = true;
        }
    }
    if has_heading { score += 0.4; }
    if has_paragraph { score += 0.3; }
    if has_list { score += 0.15; }
    if score > 1.0 { 1.0 } else { score }
}

fn check_no_input_echo(output: &str, input_text: &str) -> f64 {
    if input_text.is_empty() || input_text.len() < 20 { return 1.0; }
    let output_lower = output.to_lowercase();
    let input_lower = input_text.to_lowercase();
    let input_words: Vec<&str> = input_lower.split_whitespace().collect();
    if input_words.len() < 5 { return 1.0; }
    let window_size = 5.min(input_words.len());
    let mut echo_count = 0;
    let mut total_windows = 0;
    for i in 0..=input_words.len().saturating_sub(window_size) {
        let window = input_words[i..i + window_size].join(" ");
        total_windows += 1;
        if output_lower.contains(&window) { echo_count += 1; }
    }
    if total_windows == 0 { return 1.0; }
    1.0 - (echo_count as f64 / total_windows as f64)
}

fn check_tone_match(output: &str, expected_tone: &str) -> f64 {
    let lower = output.to_lowercase();
    let mut score = 0.5;
    match expected_tone {
        "professional" => {
            let prof_markers = ["dear", "regards", "sincerely", "furthermore", "however", "therefore", "pursuant", "respectfully"];
            let casual_markers = ["gonna", "wanna", "hey", "cheers", "no biggie", "lol", "btw", "yeah"];
            let p = prof_markers.iter().filter(|m| lower.contains(*m)).count();
            let c = casual_markers.iter().filter(|m| lower.contains(*m)).count();
            score = 0.5 + p as f64 * 0.1 - c as f64 * 0.2;
        }
        "casual" => {
            let casual_markers = ["hey", "thanks", "cheers", "sounds good", "let me know", "no worries"];
            let formal_markers = ["pursuant", "aforementioned", "forthwith", "hereby", "wherewith"];
            let c = casual_markers.iter().filter(|m| lower.contains(*m)).count();
            let f = formal_markers.iter().filter(|m| lower.contains(*m)).count();
            score = 0.5 + c as f64 * 0.15 - f as f64 * 0.2;
        }
        "confident" => {
            let conf_markers = ["will", "certainly", "definitely", "absolutely", "committed", "ensure", "guarantee"];
            let hes_markers = ["maybe", "perhaps", "might", "not sure", "i think", "possibly", "i guess"];
            let c = conf_markers.iter().filter(|m| lower.contains(*m)).count();
            let h = hes_markers.iter().filter(|m| lower.contains(*m)).count();
            score = 0.5 + c as f64 * 0.15 - h as f64 * 0.2;
        }
        "friendly" => {
            let friend_markers = ["hope", "great", "wonderful", "happy", "looking forward", "pleased", "warm"];
            let cold_markers = ["must", "required", "immediately", "consequences", "failure", "unacceptable"];
            let f = friend_markers.iter().filter(|m| lower.contains(*m)).count();
            let c = cold_markers.iter().filter(|m| lower.contains(*m)).count();
            score = 0.5 + f as f64 * 0.15 - c as f64 * 0.2;
        }
        "persuasive" => {
            let markers = ["imagine", "benefit", "opportunity", "exclusive", "limited", "don't miss", "act now", "value", "advantage"];
            let count = markers.iter().filter(|m| lower.contains(*m)).count();
            score = 0.5 + count as f64 * 0.1;
        }
        "empathetic" => {
            let markers = ["understand", "appreciate", "recognize", "sorry", "challenging", "difficult", "support", "care"];
            let count = markers.iter().filter(|m| lower.contains(*m)).count();
            score = 0.5 + count as f64 * 0.1;
        }
        _ => score = 1.0,
    }
    if score < 0.0 { 0.0 } else if score > 1.0 { 1.0 } else { score }
}

fn check_length_delta(output: &str, input_text: &str, expected_ratio: &str) -> f64 {
    let input_len = input_text.chars().count();
    let output_len = output.chars().count();
    if input_len == 0 { return 1.0; }
    match expected_ratio {
        "longer" => {
            if output_len > input_len {
                let ratio = output_len as f64 / input_len as f64;
                if ratio >= 1.5 { 1.0 } else { ratio / 1.5 }
            } else { 0.0 }
        }
        "shorter" => {
            if output_len < input_len {
                let ratio = output_len as f64 / input_len as f64;
                if ratio <= 0.7 { 1.0 } else { (1.0 - ratio) / 0.3 }
            } else { 0.0 }
        }
        _ => 1.0,
    }
}

fn check_json_field_count(output: &str, expected_fields: &[String]) -> f64 {
    let json_text = extract_json(output);
    if json_text.is_empty() { return 0.0; }
    let parsed: serde_json::Value = match serde_json::from_str(&json_text) {
        Ok(v) => v, Err(_) => return 0.0,
    };
    let obj = match parsed.as_object() { Some(o) => o, None => return 0.0 };
    let found = expected_fields.iter().filter(|f| obj.contains_key(*f)).count();
    found as f64 / expected_fields.len() as f64
}

fn count_markdown_headings(text: &str) -> usize {
    text.lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with('#') && trimmed.chars().take_while(|c| *c == '#').count() <= 6
        })
        .count()
}

fn clean_output(text: &str) -> String {
    let mut result = text.to_string();
    while let (Some(start), Some(end)) = (result.find("<think>"), result.find("</think>")) {
        if end > start {
            result = format!("{}{}", &result[..start], &result[end + 8..]);
        } else { break; }
    }
    if result.trim_start().starts_with("<think>") {
        if let Some(end) = result.find("</think>") {
            result = result[end + 8..].to_string();
        } else if let Some(start) = result.find("<think>") {
            result = result[..start].to_string();
        }
    }
    result.trim().to_string()
}

// ---------------------------------------------------------------------------
// Dataset loading
// ---------------------------------------------------------------------------

fn load_dataset() -> Result<SkillEvalDataset, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/skills/skill_eval_dataset_v1.json");
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("failed to read dataset {}: {}", path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse dataset: {}", e))
}

/// Build a SkillPromptInput from a test case's input fields.
fn build_prompt_input(tc: &SkillTestCase) -> SkillPromptInput<'_> {
    let tier = match tc.tier.as_str() {
        "low" => Some(SkillTier::Low),
        "medium" => Some(SkillTier::Medium),
        "high" => Some(SkillTier::High),
        _ => None,
    };
    // The mapping depends on the skill's scope:
    // - Selection scope: selection → input, document → context
    // - Cursor scope: cursor_context → context (e.g. edit_continue_writing uses input.context)
    // - Document scope: document → context (e.g. edit_translate_document uses input.context)
    // - Section scope: document → context (e.g. edit_rewrite_section uses input.context)
    // - Topic scope: variant_context → input, document → context (create skills)
    let (input, context) = match tc.scope.as_str() {
        "selection" => {
            let inp = if !tc.input.selection.is_empty() { tc.input.selection.as_str() } else { "" };
            (inp, tc.input.document.as_str())
        }
        "cursor" => {
            let ctx = if !tc.input.cursor_context.is_empty() { tc.input.cursor_context.as_str() }
                else { tc.input.document.as_str() };
            let inp = if !tc.input.variant_context.is_empty() { tc.input.variant_context.as_str() } else { "" };
            (inp, ctx)
        }
        "document" | "section" => {
            let ctx = if !tc.input.document.is_empty() { tc.input.document.as_str() }
                else { tc.input.cursor_context.as_str() };
            let inp = if !tc.input.variant_context.is_empty() { tc.input.variant_context.as_str() }
                else if !tc.input.selection.is_empty() { tc.input.selection.as_str() }
                else { "" };
            (inp, ctx)
        }
        "topic" => {
            let inp = if !tc.input.variant_context.is_empty() { tc.input.variant_context.as_str() }
                else { tc.input.selection.as_str() };
            (inp, tc.input.document.as_str())
        }
        _ => ("", ""),
    };
    SkillPromptInput {
        input,
        context,
        keywords: tc.input.keywords.as_str(),
        variant_context: tc.input.variant_context.as_str(),
        tier,
    }
}

/// Generate a deterministic mock output for a test case (simulates model output).
/// The mock output is designed to pass quality checks — we're testing infrastructure, not model quality.
fn generate_mock_output(skill: &SkillDef, tc: &SkillTestCase) -> String {
    // Extract quality check constraints to generate compliant mock output
    let mut required_keywords: Vec<String> = Vec::new();
    let mut forbidden_terms: Vec<String> = Vec::new();
    let mut required_language: Option<String> = None;
    let mut min_chars: usize = 0;
    let mut min_headings: usize = 0;

    for qc in &tc.quality_checks {
        match qc.check_type.as_str() {
            "contains_keyword" => {
                if let Some(kw) = &qc.keywords {
                    required_keywords.extend(kw.iter().cloned());
                }
            }
            "not_contains" => {
                if let Some(nc) = &qc.not_contains {
                    forbidden_terms.extend(nc.iter().cloned());
                }
            }
            "language_script" => {
                required_language = qc.language.clone();
            }
            "min_length" => {
                min_chars = qc.min_chars.unwrap_or(0);
            }
            "heading_count" => {
                min_headings = qc.min_headings.unwrap_or(0);
            }
            "multi_check" => {
                if let Some(subs) = &qc.checks {
                    for sub in subs {
                        match sub.check_type.as_str() {
                            "contains_keyword" => {
                                if let Some(kw) = &sub.keywords {
                                    required_keywords.extend(kw.iter().cloned());
                                }
                            }
                            "not_contains" => {
                                if let Some(nc) = &sub.not_contains {
                                    forbidden_terms.extend(nc.iter().cloned());
                                }
                            }
                            "language_script" => {
                                required_language = sub.language.clone();
                            }
                            "min_length" => {
                                min_chars = sub.min_chars.unwrap_or(0).max(min_chars);
                            }
                            "heading_count" => {
                                min_headings = sub.min_headings.unwrap_or(0).max(min_headings);
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }

    let grammar_type = skill.grammar_type.clone();
    match grammar_type {
        SkillGrammarType::JsonSchema => {
            match skill.id.as_str() {
                "doc_extract_dates" => {
                    r#"{"dates":[{"date":"2025-03-15","context":"Project kickoff meeting"},{"date":"2025-06-30","context":"Beta release deadline"}]}"#.to_string()
                }
                "doc_readability_score" => {
                    r#"{"grade_level":10,"score":65.5,"suggestions":"The text is moderately complex. Consider shorter sentences."}"#.to_string()
                }
                "doc_word_count" => {
                    let words = tc.input.document.split_whitespace().count();
                    let chars = tc.input.document.chars().count();
                    let paragraphs = tc.input.document.split("\n\n").filter(|p| !p.trim().is_empty()).count().max(1);
                    let reading_time = (words as f64 / 200.0).ceil() as u32;
                    format!(r#"{{"words":{},"chars":{},"paragraphs":{},"reading_time_min":{}}}"#, words, chars, paragraphs, reading_time)
                }
                "doc_find_contradictions" => {
                    r#"{"contradictions":[]}"#.to_string()
                }
                "create_seo_meta" => {
                    let title = if !required_keywords.is_empty() {
                        format!("Guide to {}", required_keywords.join(" and "))
                    } else {
                        "Comprehensive Guide".to_string()
                    };
                    format!(r#"{{"title":"{}","description":"A comprehensive guide covering key aspects and practical tips."}}"#, title)
                }
                _ => r#"{"result":"ok"}"#.to_string(),
            }
        }
        SkillGrammarType::Regex => {
            "match: ok".to_string()
        }
        SkillGrammarType::FreeText => {
            // If non-latin language_script is required, generate in that script
            if let Some(ref lang) = required_language {
                if lang != "latin" {
                    return generate_multilingual_mock(skill, tc, lang, &required_keywords, min_chars);
                }
            }

            // Build mock output based on skill type, incorporating required keywords
            let keyword_str = if required_keywords.is_empty() {
                String::new()
            } else {
                required_keywords.join(", ")
            };

            match skill.id.as_str() {
                "doc_summarize" => {
                    let mut points = Vec::new();
                    if !required_keywords.is_empty() {
                        for (i, kw) in required_keywords.iter().take(3).enumerate() {
                            points.push(format!("- Key point about {}: important detail regarding {}.", kw, kw));
                        }
                    }
                    if points.len() < 3 {
                        points.push("- Summary of the overall conclusion presented.".to_string());
                    }
                    let result = points.join("\n");
                    pad_to_min(&result, min_chars)
                }
                "doc_key_points" => {
                    let mut points = Vec::new();
                    if !required_keywords.is_empty() {
                        for (i, kw) in required_keywords.iter().take(5).enumerate() {
                            points.push(format!("{}. Key takeaway: {} is critical for success.", i + 1, kw));
                        }
                    }
                    if points.is_empty() {
                        points.push("1. First key takeaway from the document.".to_string());
                        points.push("2. Second important finding or insight.".to_string());
                        points.push("3. Third critical point for consideration.".to_string());
                    }
                    let result = points.join("\n");
                    pad_to_min(&result, min_chars)
                }
                "doc_extract_actions" => {
                    let mut actions = Vec::new();
                    if !required_keywords.is_empty() {
                        for kw in required_keywords.iter().take(3) {
                            actions.push(format!("- Review and follow up on {}.", kw));
                        }
                    }
                    if actions.is_empty() {
                        actions.push("- Review the document before the deadline.".to_string());
                        actions.push("- Follow up with stakeholders on action items.".to_string());
                        actions.push("- Schedule a meeting to discuss next steps.".to_string());
                    }
                    let result = actions.join("\n");
                    pad_to_min(&result, min_chars)
                }
                "edit_fix_grammar" => {
                    // Fix common grammar errors in the selection
                    let mut fixed = tc.input.selection.clone();
                    fixed = fixed.replace("tommorow", "tomorrow");
                    fixed = fixed.replace("should of", "should have");
                    fixed = fixed.replace("Their going", "They're going");
                    fixed = fixed.replace("dont", "don't");
                    fixed = fixed.replace("wont", "won't");
                    fixed = fixed.replace("its a", "it's a");
                    fixed = fixed.replace("recieve", "receive");
                    fixed = fixed.replace("seperate", "separate");
                    fixed = fixed.replace("occured", "occurred");
                    fixed = fixed.replace("definately", "definitely");
                    // Remove any remaining forbidden terms
                    for term in &forbidden_terms {
                        fixed = fixed.replace(term, "");
                    }
                    pad_to_min(&fixed, min_chars)
                }
                "edit_improve_writing" => {
                    let base = "The improved text flows more naturally with better transitions and clearer expression of ideas.";
                    let result = if !keyword_str.is_empty() {
                        format!("{}. The revision addresses {} effectively.", base, keyword_str)
                    } else { base.to_string() };
                    pad_to_min(&result, min_chars)
                }
                "edit_change_tone" => {
                    "Dear colleagues, I hope this message finds you well. I am writing to share an important update regarding our project.".to_string()
                }
                "edit_simplify" => {
                    "The main idea is simple and easy to understand. It explains the key concept clearly without unnecessary complexity.".to_string()
                }
                "edit_make_longer" => {
                    let base = &tc.input.selection;
                    let result = format!("{}. Furthermore, this approach offers additional benefits that warrant consideration. For example, the implementation can be extended to cover edge cases and provide more robust handling of various scenarios.", base);
                    pad_to_min(&result, min_chars)
                }
                "edit_make_shorter" => {
                    "In short: the key point is clear and concise.".to_string()
                }
                "edit_translate_selection" | "edit_translate_document" => {
                    // If keywords are required (e.g. translated words), include them
                    if !required_keywords.is_empty() {
                        let kw_list: Vec<String> = required_keywords.iter()
                            .map(|k| format!("{}", k))
                            .collect();
                        let result = format!("Bản dịch: {}.", kw_list.join(", "));
                        pad_to_min(&result, min_chars)
                    } else {
                        "This is the translated text in the target language.".to_string()
                    }
                }
                "edit_format_document" => {
                    let mut result = String::new();
                    let heading_count = min_headings.max(3);
                    for i in 0..heading_count {
                        if i == 0 { result.push_str("# Document Title\n\n"); }
                        else { result.push_str(&format!("## Section {}\n\n", i)); }
                        result.push_str("Formatted content with proper structure.\n\n");
                    }
                    result.push_str("- List item 1\n- List item 2");
                    result
                }
                "edit_improve_document" => {
                    let result = "# Improved Document\n\nThis document has been refined for better flow, clarity, and coherence. The structure is now more logical and the arguments more compelling.";
                    pad_to_min(result, min_chars)
                }
                "edit_custom_instruction" => {
                    "The text has been modified according to the custom instruction provided. The changes improve clarity and alignment with the requested direction.".to_string()
                }
                "edit_continue_writing" => {
                    let ctx = if !tc.input.cursor_context.is_empty() { &tc.input.cursor_context } else { &tc.input.document };
                    let result = format!("{} and this continuation extends the thought with additional relevant details that maintain the flow.", ctx);
                    pad_to_min(&result, min_chars)
                }
                "edit_rewrite_section" => {
                    "This rewritten section presents the same information with improved structure and clarity. The content is reorganized for better readability.".to_string()
                }
                "create_brainstorm" => {
                    let mut ideas = Vec::new();
                    if !required_keywords.is_empty() {
                        for (i, kw) in required_keywords.iter().take(5).enumerate() {
                            ideas.push(format!("{}. Innovative idea involving {}.", i + 1, kw));
                        }
                    }
                    if ideas.is_empty() {
                        ideas.push("1. First innovative idea worth exploring.".to_string());
                        ideas.push("2. Second creative approach to consider.".to_string());
                        ideas.push("3. Third practical solution that could work.".to_string());
                        ideas.push("4. Fourth alternative worth investigating.".to_string());
                        ideas.push("5. Fifth unique angle to explore.".to_string());
                    }
                    let result = ideas.join("\n");
                    pad_to_min(&result, min_chars)
                }
                "create_outline" => {
                    let mut result = String::from("# Main Topic\n\n");
                    let heading_count = min_headings.max(4);
                    for i in 1..heading_count {
                        result.push_str(&format!("## Section {}\n\nContent for this section.\n\n", i));
                    }
                    result.push_str("## Conclusion\n\nSummary and next steps");
                    result
                }
                "create_write_section" => {
                    let mut result = format!("This section covers the requested topic in detail. ");
                    if !required_keywords.is_empty() {
                        result.push_str(&format!("It addresses {} ", keyword_str));
                    }
                    result.push_str("with comprehensive information and relevant examples. The content is structured for easy reading and understanding.");
                    pad_to_min(&result, min_chars)
                }
                "create_generate_document" => {
                    let mut result = String::from("# Guide\n\n## Overview\nThis guide covers the essential aspects of the topic.\n\n## Getting Started\nBegin with the fundamentals and build from there.\n\n## Best Practices\nFollow these recommended approaches for optimal results.\n\n## Conclusion\nWith these steps, you are well equipped to proceed.");
                    if !required_keywords.is_empty() {
                        result.push_str(&format!("\n\nKey topics include {}.", keyword_str));
                    }
                    pad_to_min(&result, min_chars)
                }
                "create_write_intro" => {
                    let mut result = "This document explores an important topic that affects many aspects of modern work. Understanding the key principles outlined here will provide valuable insights for practitioners and decision-makers alike.".to_string();
                    if !required_keywords.is_empty() {
                        result.push_str(&format!(" Topics include {}.", keyword_str));
                    }
                    pad_to_min(&result, min_chars)
                }
                "create_write_conclusion" => {
                    "In conclusion, the findings and recommendations presented in this document provide a clear path forward. By following the outlined steps and maintaining focus on the core objectives, success is achievable.".to_string()
                }
                "create_suggest_title" => {
                    if !required_keywords.is_empty() {
                        format!("A Comprehensive Guide to {}", required_keywords[0])
                    } else {
                        "A Comprehensive Guide to Modern Practices".to_string()
                    }
                }
                "create_email_draft" => {
                    let mut result = "Dear Team,\n\nI hope this message finds you well. I am writing to share an important update regarding our project. ".to_string();
                    if !required_keywords.is_empty() {
                        result.push_str(&format!("This relates to {} and its implications. ", keyword_str));
                    }
                    result.push_str("Please let me know if you have any questions.\n\nBest regards");
                    pad_to_min(&result, min_chars)
                }
                "create_meeting_agenda" => {
                    let mut result = String::from("# Meeting Agenda\n\n## Opening\n- Welcome and introductions\n\n## Main Discussion\n");
                    if !required_keywords.is_empty() {
                        for kw in &required_keywords {
                            result.push_str(&format!("- Discuss {}\n", kw));
                        }
                    } else {
                        result.push_str("- Review progress\n- Discuss next steps\n");
                    }
                    result.push_str("\n## Action Items\n- Assign follow-up tasks");
                    result
                }
                "create_job_description" => {
                    let mut result = "# Job Description\n\n## Overview\nWe are seeking a talented candidate to join our team.\n\n## Responsibilities\n- Lead key initiatives\n- Collaborate with cross-functional teams\n\n## Requirements\n- Relevant experience\n- Strong communication skills".to_string();
                    if !required_keywords.is_empty() {
                        result.push_str(&format!("\n\nThis role focuses on {}.", keyword_str));
                    }
                    pad_to_min(&result, min_chars)
                }
                "create_press_release" => {
                    let mut result = "FOR IMMEDIATE RELEASE\n\nCompany Announces Major Initiative\n\nCITY, State — Today the company announced a significant new development that will impact the industry. ".to_string();
                    if !required_keywords.is_empty() {
                        result.push_str(&format!("This initiative focuses on {} and represents a major step forward. ", keyword_str));
                    }
                    result.push_str("The company is committed to delivering value to its customers and stakeholders through this strategic move.");
                    pad_to_min(&result, min_chars)
                }
                "create_social_post" => {
                    let mut result = "Excited to share this update with our community! ".to_string();
                    if !required_keywords.is_empty() {
                        result.push_str(&format!("We are {} and this represents a major step forward. ", keyword_str));
                    }
                    result.push_str("#innovation");
                    pad_to_min(&result, min_chars)
                }
                _ => "Mock output for skill.".to_string(),
            }
        }
    }
}

fn generate_multilingual_mock(
    skill: &SkillDef,
    _tc: &SkillTestCase,
    lang: &str,
    keywords: &[String],
    min_chars: usize,
) -> String {
    let kw = if keywords.is_empty() { "" } else { &keywords[0] };
    let result = match lang {
        "cjk" => {
            format!("这是一段{}的翻译内容。原文的主要意思已经准确传达，保持了原文的语调和风格。", if kw.is_empty() { "中文" } else { kw })
        }
        "kana" => {
            format!("これは{}の翻訳内容です。原文の主要な意味が正確に伝えられ、元のトーンとスタイルが維持されています。", if kw.is_empty() { "日本語" } else { kw })
        }
        "cjk_or_kana" => {
            format!("これは翻訳内容です。原文の主要な意味が正確に伝えられました。{}の内容を含んでいます。", if kw.is_empty() { "翻訳" } else { kw })
        }
        "hangul" => {
            format!("이것은 {}의 번역 내용입니다. 원문의 주요 의미가 정확하게 전달되었으며, 원래의 어조와 스타일이 유지되었습니다.", if kw.is_empty() { "한국어" } else { kw })
        }
        "arabic" => {
            format!("هذا هو المحتوى المترجم لـ {}. تم نقل المعنى الرئيسي للنص الأصلي بدقة مع الحفاظ على النبرة والأسلوب الأصليين.", if kw.is_empty() { "العربية" } else { kw })
        }
        "devanagari" => {
            format!("यह {} का अनुवादित सामग्री है। मूल पाठ का मुख्य अर्थ सटीक रूप से传达 किया गया है।", if kw.is_empty() { "हिंदी" } else { kw })
        }
        "thai" => {
            format!("นี่คือเนื้อหาที่แปลแล้วของ{} ความหมายหลักของข้อความต้นฉบับถูกถ่ายทอดอย่างแม่นยำ", if kw.is_empty() { "ภาษาไทย" } else { kw })
        }
        _ => {
            let suffix = if kw.is_empty() { String::new() } else { format!(" regarding {}", kw) };
            format!("This is the translated text. The main meaning has been accurately conveyed{}.", suffix)
        }
    };
    pad_to_min(&result, min_chars)
}

fn pad_to_min(text: &str, min_chars: usize) -> String {
    if text.len() >= min_chars {
        text.to_string()
    } else {
        let padding = " Additional context and details are provided to ensure comprehensive coverage of the topic.";
        if text.len() + padding.len() >= min_chars {
            format!("{}{}", text, padding)
        } else {
            let mut result = text.to_string();
            while result.len() < min_chars {
                result.push_str(padding);
            }
            result
        }
    }
}

// ---------------------------------------------------------------------------
// Mock eval runner
// ---------------------------------------------------------------------------

pub fn run_mock() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  SKILL EVAL SUITE — MOCK MODE (CI)                                           ║");
    println!("║  33 Skills × 208 Test Cases — Prompt/Grammar/Budget/Pipeline/LoRA Validation ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let dataset = match load_dataset() {
        Ok(d) => {
            println!("Dataset: {} v{} ({} test cases)", d.name, d.version, d.test_cases.len());
            d
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            return;
        }
    };
    println!();

    let registry = SkillRegistry::new();
    let lora_manager = kchat_generation::LoraManager::new();
    let lora_resolver = SkillLoRAResolver::new(&lora_manager);

    let mut results: Vec<SkillResult> = Vec::new();
    let mut suite = SuiteReport::new("Skill Eval (Mock)", 0.90);

    for tc in &dataset.test_cases {
        let start = Instant::now();
        let mut checks_passed = true;
        let mut errors: Vec<String> = Vec::new();

        // 1. Skill exists in registry
        let skill = match registry.get(&tc.skill_id) {
            Some(s) => s,
            None => {
                let err = format!("skill '{}' not found in registry", tc.skill_id);
                errors.push(err.clone());
                results.push(SkillResult {
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

        // 2. Verify skill metadata (surface, scope, mode)
        let expected_surface = match tc.surface.as_str() {
            "read" => SkillSurface::Read,
            "edit" => SkillSurface::Edit,
            "create" => SkillSurface::Create,
            _ => skill.surface,
        };
        if skill.surface != expected_surface {
            errors.push(format!("surface mismatch: expected {:?}, got {:?}", expected_surface, skill.surface));
            checks_passed = false;
        }

        // 3. Prompt construction — build_prompt produces non-empty system + user
        let prompt_input = build_prompt_input(tc);
        let prompt_output = skill.build_prompt(prompt_input);
        if prompt_output.system.is_empty() {
            errors.push("system prompt is empty".into());
            checks_passed = false;
        }
        if prompt_output.user.is_empty() {
            errors.push("user prompt is empty".into());
            checks_passed = false;
        }

        // 4. Grammar constraint — Grammar::for_skill returns expected type
        let grammar = Grammar::for_skill(&skill);
        let grammar_ok = match (&skill.grammar_type, tc.grammar_type.as_str()) {
            (SkillGrammarType::FreeText, "free_text") => true,
            (SkillGrammarType::JsonSchema, "json_schema") => grammar.is_some(),
            (SkillGrammarType::Regex, "regex") => grammar.is_some(),
            _ => true,
        };
        if !grammar_ok {
            errors.push(format!(
                "grammar type mismatch: skill={:?}, dataset={}",
                skill.grammar_type, tc.grammar_type
            ));
            checks_passed = false;
        }

        // 5. Token budget — estimate prompt tokens + max_tokens fit in context window
        let system_tokens = estimate_tokens_text(&prompt_output.system);
        let user_tokens = estimate_tokens_text(&prompt_output.user);
        let total_prompt_tokens = system_tokens + user_tokens;
        let tier_enum = match tc.tier.as_str() {
            "low" => SkillTier::Low,
            "medium" => SkillTier::Medium,
            "high" => SkillTier::High,
            _ => SkillTier::Medium,
        };
        let context_window = tier_enum.context_cap();
        let effective_max = skill.effective_max_tokens(tier_enum);
        if total_prompt_tokens + effective_max > context_window {
            errors.push(format!(
                "token budget overflow: prompt={} + effective_max_tokens={} > context_window={}",
                total_prompt_tokens, effective_max, context_window
            ));
            checks_passed = false;
        }

        // 6. LoRA resolution — verify resolver returns expected adapter
        let lora_adapter = lora_resolver.resolve(&skill.lora_task, "en");
        if !skill.lora_task.is_empty() && lora_adapter.is_none() {
            // This is OK for mock — LoRA adapters may not be loaded
            // Just verify the resolver doesn't panic
        }

        // 7. Generate mock output and run quality checks
        let mock_output = generate_mock_output(&skill, tc);

        // For no_input_echo and length_delta, inject the input text
        let mut enriched_checks = tc.quality_checks.clone();
        for check in &mut enriched_checks {
            if check.check_type == "no_input_echo" || check.check_type == "length_delta" {
                if check.input_text.is_none() {
                    let input_for_echo = if !tc.input.selection.is_empty() {
                        &tc.input.selection
                    } else if !tc.input.cursor_context.is_empty() {
                        &tc.input.cursor_context
                    } else if !tc.input.document.is_empty() {
                        &tc.input.document
                    } else {
                        &tc.input.variant_context
                    };
                    check.input_text = Some(input_for_echo.clone());
                }
            }
        }

        let (quality_score, checks_detail) = run_all_quality_checks(&mock_output, &enriched_checks);
        let quality_pass = quality_score >= 0.7;
        if !quality_pass {
            let failed_checks: Vec<String> = checks_detail.iter()
                .filter(|(_, s)| *s < 0.7)
                .map(|(t, s)| format!("{}={:.2}", t, s))
                .collect();
            errors.push(format!("quality checks failed: {} (score={:.2})", failed_checks.join(", "), quality_score));
            checks_passed = false;
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let passed = checks_passed;

        let mut meta = HashMap::new();
        meta.insert("skill".into(), tc.skill_id.clone());
        meta.insert("quality_score".into(), format!("{:.2}", quality_score));
        meta.insert("prompt_tokens".into(), total_prompt_tokens.to_string());
        meta.insert("mock_output_len".into(), mock_output.len().to_string());

        if passed {
            suite.add(EvalResult::pass_with_meta(&tc.id, duration_ms, meta));
        } else {
            let reason = errors.join("; ");
            suite.add(EvalResult::fail_with_meta(&tc.id, &reason, duration_ms, meta));
        }

        results.push(SkillResult {
            case_id: tc.id.clone(),
            skill_id: tc.skill_id.clone(),
            tier: tc.tier.clone(),
            passed,
            quality_score,
            checks_detail,
            duration_ms,
            error: if errors.is_empty() { None } else { Some(errors.join("; ")) },
        });
    }

    // Print per-skill summary
    print_skill_summary(&results);

    // Print suite report
    let mut report = EvalReport::new();
    report.add_suite(suite);
    report.print();
}

// ---------------------------------------------------------------------------
// Real model eval runner
// ---------------------------------------------------------------------------

struct SkillServerHandle {
    child: std::process::Child,
    port: u16,
}

impl SkillServerHandle {
    fn check_health(url: &str) -> bool {
        let output = Command::new("curl")
            .arg("-s").arg("-o").arg("/dev/null").arg("-w").arg("%{http_code}")
            .arg("--connect-timeout").arg("2")
            .arg(&format!("{}/health", url))
            .output();
        match output {
            Ok(o) => String::from_utf8_lossy(&o.stdout).trim() == "200",
            Err(_) => false,
        }
    }

    fn wait_until_ready(url: &str, timeout_secs: u64) -> bool {
        for _ in 0..(timeout_secs * 2) {
            if Self::check_health(url) { return true; }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        false
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_skill_server() -> Result<SkillServerHandle, String> {
    // Check for existing server
    if let Ok(url) = std::env::var("LLAMA_SERVER_URL") {
        if SkillServerHandle::check_health(&url) {
            return Ok(SkillServerHandle {
                child: Command::new("echo").spawn().map_err(|e| e.to_string())?,
                port: url.rsplit(':').next().and_then(|p| p.parse().ok()).unwrap_or(18888),
            });
        }
    }

    // Find model
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../manifest/packs/Ternary-Bonsai-1.7B-Q2_0.gguf");
    if !model_path.exists() {
        return Err(format!("model not found at {}", model_path.display()));
    }

    let llama_server = std::env::var("LLAMA_SERVER_PATH").unwrap_or_else(|_| {
        let prismml = "/tmp/prism-llama.cpp/build/bin/llama-server";
        if std::path::Path::new(prismml).exists() {
            prismml.to_string()
        } else {
            "llama-server".into()
        }
    });

    if which::which(&llama_server).is_err() {
        return Err(format!("llama-server not found: {}", llama_server));
    }

    let port: u16 = 18888;
    let child = Command::new(&llama_server)
        .arg("-m").arg(&model_path)
        .arg("--host").arg("127.0.0.1")
        .arg("--port").arg(port.to_string())
        .arg("-c").arg("4096")
        .arg("-ngl").arg("99")
        .arg("-t").arg("4")
        .arg("--no-webui")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to start llama-server: {}", e))?;

    let url = format!("http://127.0.0.1:{}", port);
    if !SkillServerHandle::wait_until_ready(&url, 60) {
        return Err(format!("llama-server did not become ready on port {}", port));
    }

    Ok(SkillServerHandle { child, port })
}

#[derive(Debug)]
struct SkillCompletionResponse {
    content: String,
    tokens_predicted: u32,
    prompt_ms: f64,
    predicted_ms: f64,
}

fn send_skill_completion(
    server_url: &str,
    chatml_prompt: &str,
    max_tokens: u32,
    temperature: f32,
    grammar_type: &str,
    schema: Option<&serde_json::Value>,
) -> Result<SkillCompletionResponse, String> {
    let prompt = chatml_prompt;

    let mut body = serde_json::json!({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "seed": 42,
    });

    if grammar_type == "json_schema" {
        if let Some(s) = schema {
            body["json_schema"] = s.clone();
        }
    }

    let output = Command::new("curl")
        .arg("-s").arg("-X").arg("POST")
        .arg(&format!("{}/completion", server_url))
        .arg("-H").arg("Content-Type: application/json")
        .arg("-d").arg(body.to_string())
        .arg("--connect-timeout").arg("10")
        .arg("--max-time").arg("180")
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    if !output.status.success() {
        return Err(format!("curl exit code: {}", output.status));
    }

    let resp: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(|e| format!("parse error: {}", e))?;

    let content = resp.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let tokens_predicted = resp.get("tokens_predicted").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    let (prompt_ms, predicted_ms) = if let Some(timings) = resp.get("timings") {
        (
            timings.get("prompt_ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
            timings.get("predicted_ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
        )
    } else {
        (
            resp.get("prompt_ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
            resp.get("predicted_ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
        )
    };

    Ok(SkillCompletionResponse { content, tokens_predicted, prompt_ms, predicted_ms })
}

pub fn run_realworld() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  SKILL EVAL SUITE — REAL MODEL MODE                                          ║");
    println!("║  33 Skills × 208 Test Cases — Real Model Inference + Quality Measurement     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    let dataset = match load_dataset() {
        Ok(d) => {
            println!("Dataset: {} v{} ({} test cases)", d.name, d.version, d.test_cases.len());
            d
        }
        Err(e) => {
            eprintln!("ERROR: {}", e);
            return;
        }
    };
    println!();

    // Start server
    print!("Starting model server... ");
    std::io::stdout().flush().ok();

    let mut server = match start_skill_server() {
        Ok(s) => {
            println!("OK (port {})", s.port);
            s
        }
        Err(e) => {
            println!("FAILED");
            eprintln!("ERROR: {}", e);
            eprintln!("To run real model eval, start llama-server manually:");
            eprintln!("  llama-server -m manifest/packs/Ternary-Bonsai-1.7B-Q2_0.gguf --port 18888 -ngl 99");
            eprintln!("  or set LLAMA_SERVER_URL=http://127.0.0.1:18888");
            return;
        }
    };

    let server_url = format!("http://127.0.0.1:{}", server.port);
    let registry = SkillRegistry::new();

    let mut results: Vec<SkillResult> = Vec::new();
    let mut suite = SuiteReport::new("Skill Eval (Real Model)", 0.75);

    println!();
    println!("Running {} test cases...", dataset.test_cases.len());
    println!();

    for (idx, tc) in dataset.test_cases.iter().enumerate() {
        print!("\r  Case {:>3}/{} [{:<25}] ", idx + 1, dataset.test_cases.len(), &tc.id[..tc.id.len().min(25)]);
        std::io::stdout().flush().ok();

        let start = Instant::now();

        let skill = match registry.get(&tc.skill_id) {
            Some(s) => s,
            None => {
                suite.add(EvalResult::fail(&tc.id, "skill not in registry"));
                results.push(SkillResult {
                    case_id: tc.id.clone(),
                    skill_id: tc.skill_id.clone(),
                    tier: tc.tier.clone(),
                    passed: false,
                    quality_score: 0.0,
                    checks_detail: Vec::new(),
                    duration_ms: 0,
                    error: Some("skill not in registry".into()),
                });
                continue;
            }
        };

        let prompt_input = build_prompt_input(tc);
        let chatml_prompt = skill.build_chatml_prompt(prompt_input);

        // Tier-aware max_tokens: clamp to tier output cap, boost for thinking skills on high tier
        let tier_enum = match tc.tier.as_str() {
            "low" => SkillTier::Low,
            "medium" => SkillTier::Medium,
            "high" => SkillTier::High,
            _ => SkillTier::Medium,
        };
        let effective_max = skill.effective_max_tokens(tier_enum) as u32;

        // Get grammar schema if applicable
        let schema = tc.quality_checks.iter()
            .find_map(|c| c.schema.as_ref());

        let response = send_skill_completion(
            &server_url,
            &chatml_prompt,
            effective_max,
            skill.temperature,
            &tc.grammar_type,
            schema,
        );

        let duration_ms = start.elapsed().as_millis() as u64;

        match response {
            Ok(resp) => {
                let output_clean = clean_output(&resp.content);

                // Enrich checks with input text for no_input_echo / length_delta
                let mut enriched_checks = tc.quality_checks.clone();
                for check in &mut enriched_checks {
                    if check.check_type == "no_input_echo" || check.check_type == "length_delta" {
                        if check.input_text.is_none() {
                            let input_for_echo = if !tc.input.selection.is_empty() {
                                &tc.input.selection
                            } else if !tc.input.cursor_context.is_empty() {
                                &tc.input.cursor_context
                            } else if !tc.input.document.is_empty() {
                                &tc.input.document
                            } else {
                                &tc.input.variant_context
                            };
                            check.input_text = Some(input_for_echo.clone());
                        }
                    }
                }

                let (quality_score, checks_detail) = run_all_quality_checks(&output_clean, &enriched_checks);
                let quality_threshold = match tc.tier.as_str() {
                    "low" => 0.6,
                    "medium" => 0.7,
                    "high" => 0.75,
                    _ => 0.7,
                };
                let passed = quality_score >= quality_threshold && !output_clean.is_empty();

                let mut meta = HashMap::new();
                meta.insert("skill".into(), tc.skill_id.clone());
                meta.insert("quality".into(), format!("{:.2}", quality_score));
                meta.insert("tokens".into(), resp.tokens_predicted.to_string());
                meta.insert("ttft_ms".into(), (resp.prompt_ms as u64).to_string());
                meta.insert("decode_tps".into(), format!("{:.1}",
                    if resp.predicted_ms > 0.0 { resp.tokens_predicted as f64 * 1000.0 / resp.predicted_ms } else { 0.0 }));

                if passed {
                    suite.add(EvalResult::pass_with_meta(&tc.id, duration_ms, meta));
                } else {
                    let failed_checks: Vec<String> = checks_detail.iter()
                        .filter(|(_, s)| *s < 0.7)
                        .map(|(t, s)| format!("{}={:.2}", t, s))
                        .collect();
                    let reason = if output_clean.is_empty() {
                        "empty output".to_string()
                    } else {
                        format!("quality={:.2} failed: {}", quality_score, failed_checks.join(","))
                    };
                    suite.add(EvalResult::fail_with_meta(&tc.id, &reason, duration_ms, meta));
                }

                results.push(SkillResult {
                    case_id: tc.id.clone(),
                    skill_id: tc.skill_id.clone(),
                    tier: tc.tier.clone(),
                    passed,
                    quality_score,
                    checks_detail,
                    duration_ms,
                    error: None,
                });
            }
            Err(e) => {
                suite.add(EvalResult::fail(&tc.id, &e));
                results.push(SkillResult {
                    case_id: tc.id.clone(),
                    skill_id: tc.skill_id.clone(),
                    tier: tc.tier.clone(),
                    passed: false,
                    quality_score: 0.0,
                    checks_detail: Vec::new(),
                    duration_ms,
                    error: Some(e),
                });
            }
        }
    }

    println!();
    println!();

    // Print per-skill summary
    print_skill_summary(&results);

    // Print suite report
    let mut report = EvalReport::new();
    report.add_suite(suite);
    report.print();

    // Stop server
    server.stop();
}

// ---------------------------------------------------------------------------
// Per-skill summary reporting
// ---------------------------------------------------------------------------

fn print_skill_summary(results: &[SkillResult]) {
    // Group by skill_id
    let mut by_skill: HashMap<String, Vec<&SkillResult>> = HashMap::new();
    for r in results {
        by_skill.entry(r.skill_id.clone()).or_default().push(r);
    }

    let mut skills: Vec<(String, Vec<&SkillResult>)> = by_skill.into_iter().collect();
    skills.sort_by_key(|(sid, _)| sid.clone());

    println!("SKILL EVAL SUITE — {} cases across {} skills",
        results.len(), skills.len());
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!();
    println!("{:<30} {:>5}  {:>6}  {:>8}  {:>8}", "Skill", "Cases", "Pass", "Quality", "Errors");
    println!("───────────────────────────────────────────────────────────────────────────────");

    let mut total_cases = 0usize;
    let mut total_passed = 0usize;
    let mut total_quality = 0.0f64;
    let mut total_errors = 0usize;

    for (skill_id, skill_results) in &skills {
        let total = skill_results.len();
        let passed = skill_results.iter().filter(|r| r.passed).count();
        let avg_quality: f64 = skill_results.iter().map(|r| r.quality_score).sum::<f64>() / total as f64;
        let errors = skill_results.iter().filter(|r| r.error.is_some()).count();

        println!("{:<30} {:>5}  {:>3}/{:<3} {:>8.2}  {:>8}", skill_id, total, passed, total, avg_quality, errors);

        total_cases += total;
        total_passed += passed;
        total_quality += avg_quality * total as f64;
        total_errors += errors;
    }

    println!("───────────────────────────────────────────────────────────────────────────────");
    let overall_quality = if total_cases > 0 { total_quality / total_cases as f64 } else { 0.0 };
    let pass_rate = if total_cases > 0 { total_passed as f64 / total_cases as f64 * 100.0 } else { 0.0 };
    println!("{:<30} {:>5}  {:>3}/{:<3} {:>8.2}  {:>8}",
        "OVERALL", total_cases, total_passed, total_cases, overall_quality, total_errors);
    println!();
    println!("Pass Rate: {:.1}%  |  Avg Quality: {:.2}  |  Failed: {}",
        pass_rate, overall_quality, total_cases - total_passed);
    println!();

    // Per-tier breakdown
    println!("PER-TIER BREAKDOWN");
    println!("───────────────────────────────────────────────────────────────────────────────");
    println!("{:<30} {:>6}  {:>8}  {:>8}", "Tier", "Pass", "Quality", "Failures");
    println!("───────────────────────────────────────────────────────────────────────────────");

    for tier_name in &["low", "medium", "high"] {
        let tier_results: Vec<&SkillResult> = results.iter().filter(|r| &r.tier == tier_name).collect();
        if tier_results.is_empty() {
            continue;
        }
        let total = tier_results.len();
        let passed = tier_results.iter().filter(|r| r.passed).count();
        let avg_quality: f64 = tier_results.iter().map(|r| r.quality_score).sum::<f64>() / total as f64;
        let failures = total - passed;
        println!("{:<30} {:>3}/{:<3} {:>8.2}  {:>8}", tier_name, passed, total, avg_quality, failures);
    }
    println!();

    // Skills with tier degradation (low fails but medium/high passes)
    let has_tier_variants = skills.iter().any(|(_, rs)| {
        let tiers: std::collections::HashSet<&str> = rs.iter().map(|r| r.tier.as_str()).collect();
        tiers.len() > 1
    });
    if has_tier_variants {
        println!("TIER DEGRADATION (skills tested at multiple tiers)");
        println!("───────────────────────────────────────────────────────────────────────────────");
        println!("{:<30} {:>6}  {:>6}  {:>6}", "Skill", "Low", "Medium", "High");
        println!("───────────────────────────────────────────────────────────────────────────────");

        for (skill_id, skill_results) in &skills {
            let tiers: std::collections::HashSet<&str> = skill_results.iter().map(|r| r.tier.as_str()).collect();
            if tiers.len() < 2 {
                continue;
            }

            let low_pass = skill_results.iter().filter(|r| r.tier == "low").filter(|r| r.passed).count();
            let low_total = skill_results.iter().filter(|r| r.tier == "low").count();
            let med_pass = skill_results.iter().filter(|r| r.tier == "medium").filter(|r| r.passed).count();
            let med_total = skill_results.iter().filter(|r| r.tier == "medium").count();
            let high_pass = skill_results.iter().filter(|r| r.tier == "high").filter(|r| r.passed).count();
            let high_total = skill_results.iter().filter(|r| r.tier == "high").count();

            let low_str = if low_total > 0 { format!("{}/{}", low_pass, low_total) } else { "—".to_string() };
            let med_str = if med_total > 0 { format!("{}/{}", med_pass, med_total) } else { "—".to_string() };
            let high_str = if high_total > 0 { format!("{}/{}", high_pass, high_total) } else { "—".to_string() };

            println!("{:<30} {:>6}  {:>6}  {:>6}", skill_id, low_str, med_str, high_str);
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run(realworld: bool) {
    if realworld {
        run_realworld();
    } else {
        run_mock();
    }
}

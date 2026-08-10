//! Per-device real-world eval runner.
//!
//! Tests each of the 12 device profiles against its assigned real model,
//! measuring performance + quality, and producing a judgment report.

use crate::report::{EvalReport, EvalResult, EvalStatus, SuiteReport};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MultitaskDataset {
    name: String,
    version: String,
    tasks: Vec<TaskSpec>,
    performance_targets: PerformanceTargets,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskSpec {
    id: String,
    category: String,
    prompt: String,
    max_tokens: u32,
    expected_min_tokens: u32,
    grammar: Option<GrammarSpec>,
    quality_check: Option<QualityCheck>,
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GrammarSpec {
    #[serde(rename = "type")]
    grammar_type: String,
    schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QualityCheck {
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
    expected: Option<String>,
    // --- New quality assessment fields ---
    /// Keywords that must ALL be present (not just any)
    #[serde(skip_serializing_if = "Option::is_none")]
    contains_all: Option<Vec<String>>,
    /// Keywords that must NOT appear in the output
    #[serde(skip_serializing_if = "Option::is_none")]
    not_contains: Option<Vec<String>>,
    /// Minimum number of sentences (split by . ! ? 。！？)
    #[serde(skip_serializing_if = "Option::is_none")]
    min_sentences: Option<usize>,
    /// Maximum number of sentences
    #[serde(skip_serializing_if = "Option::is_none")]
    max_sentences: Option<usize>,
    /// Minimum word count (whitespace-split)
    #[serde(skip_serializing_if = "Option::is_none")]
    min_words: Option<usize>,
    /// Expected writing system: latin, cjk, arabic, devanagari, thai, hangul, kana
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    /// Sub-checks for multi_check type — all must pass, score is average
    #[serde(skip_serializing_if = "Option::is_none")]
    checks: Option<Vec<QualityCheck>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceTargets {
    low_tier: TierTarget,
    medium_tier: TierTarget,
    high_tier: TierTarget,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TierTarget {
    ttft_p95_ms: u64,
    decode_p50_tps: f64,
    max_memory_gb: f64,
}

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum ServerType {
    LlamaServer,
    MlxServer,
}

#[derive(Debug, Clone)]
struct ModelConfig {
    pack_id: String,
    model_path: PathBuf,
    server_type: ServerType,
    server_port: u16,
    context_size: usize,
    display_name: String,
}

#[derive(Debug, Clone)]
struct DeviceProfileInfo {
    name: String,
    tier: String,
    platform: String,
    model_pack_id: String,
    /// Vision model pack ID (None = no vision model on this tier)
    vision_pack_id: Option<String>,
    /// Safety encoder pack ID (always present — INT8 for Medium+, INT4 for Low)
    safety_pack_id: String,
    /// ASR model pack ID (None = no ASR model on this tier)
    asr_pack_id: Option<String>,
    /// Video model pack ID (None = no video model on Low tier)
    video_pack_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TaskResult {
    task_id: String,
    category: String,
    success: bool,
    quality_pass: bool,
    quality_score: f64,
    ttft_ms: u64,
    decode_rate_tps: f64,
    output_tokens: u32,
    output_text: String,
    error: Option<String>,
}

#[derive(Debug, Clone)]
enum Judgment {
    Pass,
    Marginal,
    Fail,
}

impl std::fmt::Display for Judgment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Judgment::Pass => write!(f, "PASS"),
            Judgment::Marginal => write!(f, "MARG"),
            Judgment::Fail => write!(f, "FAIL"),
        }
    }
}

#[derive(Debug, Clone)]
struct DeviceJudgment {
    profile_name: String,
    tier: String,
    model: String,
    tasks_total: usize,
    tasks_passed: usize,
    quality_passed: usize,
    quality_score_avg: f64,
    ttft_p50_ms: u64,
    ttft_p95_ms: u64,
    decode_p50_tps: f64,
    decode_p95_tps: f64,
    perf_target_met: bool,
    quality_target_met: bool,
    overall_score: f64,
    judgment: Judgment,
    judgment_reason: String,
    per_category: HashMap<String, (usize, usize)>,
}

// ---------------------------------------------------------------------------
// Server management
// ---------------------------------------------------------------------------

struct ServerHandle {
    child: Child,
    port: u16,
}

impl ServerHandle {
    fn check_health(url: &str) -> bool {
        let output = Command::new("curl")
            .arg("-s")
            .arg("-o")
            .arg("/dev/null")
            .arg("-w")
            .arg("%{http_code}")
            .arg("--connect-timeout")
            .arg("2")
            .arg(&format!("{}/health", url))
            .output();
        match output {
            Ok(o) => String::from_utf8_lossy(&o.stdout).trim() == "200",
            Err(_) => false,
        }
    }

    fn wait_until_ready(url: &str, timeout_secs: u64) -> bool {
        for _ in 0..(timeout_secs * 2) {
            if Self::check_health(url) {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        false
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start_llama_server(config: &ModelConfig) -> Result<ServerHandle, String> {
    // Ternary-Bonsai GGUF models use Q2_0 g128 ternary format requiring the PrismML fork
    let is_ternary = config
        .model_path
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.contains("Ternary-Bonsai"))
        .unwrap_or(false);

    let llama_server = if is_ternary {
        // Try PrismML fork first, then env override, then PATH
        let prismml_path = "/tmp/prism-llama.cpp/build/bin/llama-server";
        if std::path::Path::new(prismml_path).exists() {
            prismml_path.to_string()
        } else {
            std::env::var("PRISM_LLAMA_SERVER_PATH")
                .unwrap_or_else(|_| {
                    eprintln!("│  WARNING: Ternary-Bonsai model requires PrismML llama-server fork.");
                    eprintln!("│  Build it: git clone --branch prism https://github.com/PrismML-Eng/llama.cpp /tmp/prism-llama.cpp && cd /tmp/prism-llama.cpp && cmake -B build -DGGML_METAL=ON && cmake --build build -j");
                    "llama-server".into()
                })
        }
    } else {
        std::env::var("LLAMA_SERVER_PATH")
            .unwrap_or_else(|_| "llama-server".into())
    };

    if which::which(&llama_server).is_err() {
        return Err(format!("llama-server not found: {}", llama_server));
    }

    let port_str = config.server_port.to_string();
    let ctx_str = config.context_size.to_string();

    let child = Command::new(&llama_server)
        .arg("-m")
        .arg(&config.model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(&port_str)
        .arg("-c")
        .arg(&ctx_str)
        .arg("-ngl")
        .arg("99")
        .arg("-t")
        .arg("4")
        .arg("--no-webui")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("failed to start llama-server: {}", e))?;

    let url = format!("http://127.0.0.1:{}", config.server_port);
    if !ServerHandle::wait_until_ready(&url, 60) {
        return Err(format!(
            "llama-server did not become ready on port {}",
            config.server_port
        ));
    }

    Ok(ServerHandle {
        child,
        port: config.server_port,
    })
}

fn start_mlx_server(config: &ModelConfig) -> Result<ServerHandle, String> {
    // Find Swift binary — check env override, then xcodebuild DerivedData, then SwiftPM .build
    let (swift_binary, framework_path) = if let Ok(path) = std::env::var("MLX_SERVER_PATH") {
        (path, None)
    } else {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let swift_pkg_dir = manifest_dir.join("../../swift/kchat-mlx-server");

        // xcodebuild output: ~/Library/Developer/Xcode/DerivedData/kchat-mlx-server-*/Build/Products/Release
        let mut xcode_binary = None;
        let mut xcode_framework = None;
        if let Ok(home) = std::env::var("HOME") {
            let dd_root = PathBuf::from(&home)
                .join("Library/Developer/Xcode/DerivedData");
            if let Ok(entries) = std::fs::read_dir(&dd_root) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with("kchat-mlx-server-") {
                        let release_dir = entry.path()
                            .join("Build/Products/Release");
                        let bin = release_dir.join("kchat-mlx-server");
                        if bin.exists() {
                            xcode_binary = Some(bin.to_string_lossy().to_string());
                            xcode_framework = Some(release_dir.to_string_lossy().to_string());
                            break;
                        }
                    }
                }
            }
        }

        if let (Some(bin), Some(fw)) = (xcode_binary, xcode_framework) {
            (bin, Some(fw))
        } else {
            // SwiftPM .build paths
            let candidates = [
                swift_pkg_dir.join(".build/release/kchat-mlx-server"),
                swift_pkg_dir.join(".build/debug/kchat-mlx-server"),
            ];
            for c in &candidates {
                if c.exists() {
                    return Err(format!(
                        "Swift binary found at {} but was built with `swift build` — \
                         Metal shaders are missing. Build with `xcodebuild` instead:\n  \
                         cd swift/kchat-mlx-server && xcodebuild build -scheme kchat-mlx-server \
                         -destination 'platform=OS X' -configuration Release -skipPackagePluginValidation",
                        c.display()
                    ));
                }
            }
            return Err(
                "kchat-mlx-server binary not found. Build with:\n  \
                 cd swift/kchat-mlx-server && xcodebuild build -scheme kchat-mlx-server \
                 -destination 'platform=OS X' -configuration Release -skipPackagePluginValidation"
                    .to_string(),
            );
        }
    };

    let port_str = config.server_port.to_string();

    let mut cmd = Command::new(&swift_binary);
    cmd.arg("--model")
        .arg(&config.model_path)
        .arg("--port")
        .arg(&port_str)
        .arg("--host")
        .arg("127.0.0.1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // Set DYLD_FRAMEWORK_PATH so the metallib bundle is found
    if let Some(ref fw) = framework_path {
        cmd.env("DYLD_FRAMEWORK_PATH", fw);
    }

    let child = cmd
        .spawn()
        .map_err(|e| format!("failed to start kchat-mlx-server: {}", e))?;

    let url = format!("http://127.0.0.1:{}", config.server_port);
    if !ServerHandle::wait_until_ready(&url, 120) {
        return Err(format!(
            "kchat-mlx-server did not become ready on port {}",
            config.server_port
        ));
    }

    Ok(ServerHandle {
        child,
        port: config.server_port,
    })
}

// ---------------------------------------------------------------------------
// Completion API (works for both llama-server and kchat-mlx-server)
// ---------------------------------------------------------------------------

struct CompletionResponse {
    content: String,
    tokens_predicted: u32,
    tokens_evaluated: u32,
    prompt_ms: f64,
    predicted_ms: f64,
}

fn send_completion(
    server_url: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
    grammar: Option<&GrammarSpec>,
) -> Result<CompletionResponse, String> {
    let mut body = serde_json::json!({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "seed": 42,
    });

    // Pass grammar to llama-server for constrained generation
    if let Some(g) = grammar {
        if g.grammar_type == "json_schema" {
            body["json_schema"] = serde_json::json!(g.schema);
        }
    }

    let output = Command::new("curl")
        .arg("-s")
        .arg("-X")
        .arg("POST")
        .arg(&format!("{}/completion", server_url))
        .arg("-H")
        .arg("Content-Type: application/json")
        .arg("-d")
        .arg(body.to_string())
        .arg("--connect-timeout")
        .arg("10")
        .arg("--max-time")
        .arg("180")
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    if !output.status.success() {
        return Err(format!("curl exit code: {}", output.status));
    }

    let resp: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(|e| format!("parse error: {}", e))?;

    let content = resp
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let tokens_predicted = resp
        .get("tokens_predicted")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let tokens_evaluated = resp
        .get("tokens_evaluated")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    // llama-server puts timings under "timings" key; kchat-mlx-server puts them at top level
    let (prompt_ms, predicted_ms) = if let Some(timings) = resp.get("timings") {
        (
            timings
                .get("prompt_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            timings
                .get("predicted_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
        )
    } else {
        (
            resp.get("prompt_ms").and_then(|v| v.as_f64()).unwrap_or(0.0),
            resp.get("predicted_ms")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
        )
    };

    Ok(CompletionResponse {
        content,
        tokens_predicted,
        tokens_evaluated,
        prompt_ms,
        predicted_ms,
    })
}

// ---------------------------------------------------------------------------
// Quality checks
// ---------------------------------------------------------------------------

fn run_quality_check(output: &str, check: &QualityCheck) -> f64 {
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
            // OR check: any keyword match = 1.0. Keywords are alternative acceptable answers.
            if let Some(keywords) = &check.keywords {
                let lower = output.to_lowercase();
                for k in keywords {
                    let kl = k.to_lowercase();
                    if lower.contains(&kl) {
                        return 1.0;
                    }
                    // Numeric keyword: also match word form (e.g. "19" matches "nineteen")
                    if let Ok(n) = k.parse::<u64>() {
                        if number_to_words(n).map(|w| lower.contains(&w.to_lowercase())).unwrap_or(false) {
                            return 1.0;
                        }
                    }
                }
                0.0
            } else { 0.0 }
        }
        "contains_all" => {
            if let Some(keywords) = &check.contains_all {
                let lower = output.to_lowercase();
                let mut all_found = true;
                for k in keywords {
                    if !lower.contains(&k.to_lowercase()) {
                        all_found = false;
                        break;
                    }
                }
                if all_found { 1.0 } else { 0.0 }
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
        "exact_match" => {
            let expected = check.expected.as_deref().unwrap_or("");
            if output.trim() == expected.trim() { 1.0 } else { 0.0 }
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
            else if count >= min && max == usize::MAX { 1.0 }
            else if count < min {
                count as f64 / min as f64
            } else {
                max as f64 / count as f64
            }
        }
        "language_script" => {
            if let Some(lang) = &check.language {
                detect_language_score(output, lang)
            } else { 1.0 }
        }
        "min_words" => {
            let min = check.min_words.unwrap_or(0);
            let words = output.split_whitespace().count();
            if words >= min { 1.0 } else if min > 0 {
                words as f64 / min as f64
            } else { 1.0 }
        }
        "multi_check" => {
            if let Some(sub_checks) = &check.checks {
                if sub_checks.is_empty() { return 1.0; }
                let total = sub_checks.len() as f64;
                let sum: f64 = sub_checks.iter().map(|c| run_quality_check(output, c)).sum();
                sum / total
            } else { 1.0 }
        }
        _ => 1.0,
    }
}

fn number_to_words(n: u64) -> Option<String> {
    let ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen"];
    let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

    if n >= 1000 {
        return None; // Keep it simple for eval purposes
    }
    if n < 20 {
        return Some(ones[n as usize].to_string());
    }
    if n < 100 {
        let t = (n / 10) as usize;
        let o = (n % 10) as usize;
        if o == 0 {
            return Some(tens[t].to_string());
        }
        return Some(format!("{}-{}", tens[t], ones[o]));
    }
    let h = (n / 100) as usize;
    let rest = n % 100;
    if rest == 0 {
        return Some(format!("{} hundred", ones[h]));
    }
    if rest < 20 {
        return Some(format!("{} hundred and {}", ones[h], ones[rest as usize]));
    }
    let t = (rest / 10) as usize;
    let o = (rest % 10) as usize;
    if o == 0 {
        return Some(format!("{} hundred and {}", ones[h], tens[t]));
    }
    Some(format!("{} hundred and {}-{}", ones[h], tens[t], ones[o]))
}

fn validate_json_schema(value: &serde_json::Value, schema: &serde_json::Value) -> bool {
    // Basic JSON Schema validation — checks type, required fields, and nested objects
    let schema_obj = match schema.as_object() {
        Some(o) => o,
        None => return true, // No schema object = just check it parses
    };

    // Check type
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
        if !type_ok {
            return false;
        }
    }

    // Check required fields for objects
    if let Some(required) = schema_obj.get("required").and_then(|v| v.as_array()) {
        if let Some(obj) = value.as_object() {
            for req in required {
                if let Some(field) = req.as_str() {
                    if !obj.contains_key(field) {
                        return false;
                    }
                }
            }
        } else {
            return false;
        }
    }

    // Check properties (nested validation)
    if let (Some(properties), Some(obj)) = (
        schema_obj.get("properties").and_then(|v| v.as_object()),
        value.as_object(),
    ) {
        for (key, prop_schema) in properties {
            if let Some(field_val) = obj.get(key) {
                if !validate_json_schema(field_val, prop_schema) {
                    return false;
                }
            }
        }
    }

    // Check array items
    if let (Some(items_schema), Some(arr)) = (
        schema_obj.get("items"),
        value.as_array(),
    ) {
        for item in arr {
            if !validate_json_schema(item, items_schema) {
                return false;
            }
        }
    }

    true
}

fn extract_json(text: &str) -> String {
    // Strip markdown code fences: ```json ... ``` or ``` ... ```
    let mut trimmed = text.trim();
    if trimmed.starts_with("```") {
        // Remove opening fence line
        if let Some(nl) = trimmed.find('\n') {
            trimmed = trimmed[nl + 1..].trim();
        }
        // Remove closing fence
        if let Some(pos) = trimmed.find("```") {
            trimmed = trimmed[..pos].trim();
        }
    }

    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        // Try the whole string first
        if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
            return trimmed.to_string();
        }
    }
    // Search for JSON within the text
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
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' && in_string {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(i + 1);
                }
            }
            _ => {}
        }
    }
    Err(())
}

fn is_repeated(text: &str) -> bool {
    // Check if output is highly repetitive (a sign of model degradation)
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 10 {
        return false;
    }
    let unique: std::collections::HashSet<&str> = words.iter().copied().collect();
    let ratio = unique.len() as f64 / words.len() as f64;
    ratio < 0.2 // less than 20% unique words = repetitive
}

fn count_sentences(text: &str) -> usize {
    // Count sentences by looking for sentence-ending punctuation followed by
    // whitespace or end of string. Supports both Western (.!?) and CJK (。！？) punctuation.
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return 0;
    }
    let mut count = 0;
    let chars: Vec<char> = trimmed.chars().collect();
    for (i, &c) in chars.iter().enumerate() {
        if c == '.' || c == '!' || c == '?' || c == '。' || c == '！' || c == '？' {
            // Count this as a sentence ending if it's followed by space, newline, or end
            let is_end = i + 1 >= chars.len()
                || chars[i + 1].is_whitespace()
                || chars[i + 1] == '.'
                || chars[i + 1] == '!'
                || chars[i + 1] == '?';
            if is_end {
                count += 1;
            }
        }
    }
    // If no sentence-ending punctuation found but text is non-empty, count as 1
    if count == 0 && !trimmed.is_empty() {
        count = 1;
    }
    count
}

fn detect_language_score(text: &str, expected: &str) -> f64 {
    // Detect writing system from Unicode character ranges
    let mut latin = 0u32;
    let mut cjk = 0u32;      // CJK Unified Ideographs
    let mut kana = 0u32;     // Hiragana + Katakana
    let mut hangul = 0u32;   // Korean Hangul
    let mut arabic = 0u32;   // Arabic
    let mut devanagari = 0u32; // Devanagari (Hindi)
    let mut thai = 0u32;     // Thai
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
    if total == 0 {
        return 0.0;
    }

    let score_for = |count: u32| -> f64 {
        if count == 0 { 0.0 }
        else { count as f64 / total as f64 }
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

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

fn find_model_path(pack_id: &str) -> Option<PathBuf> {
    let pack_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../manifest/packs");

    match pack_id {
        "ternary-bonsai-1.7b-mlx-2bit" => {
            let dir = pack_dir.join("ternary-bonsai-1.7b-mlx-2bit");
            if dir.exists() {
                Some(dir)
            } else {
                None
            }
        }
        "ternary-bonsai-4b-mlx-2bit" => {
            let dir = pack_dir.join("ternary-bonsai-4b-mlx-2bit");
            if dir.exists() {
                Some(dir)
            } else {
                None
            }
        }
        "ternary-bonsai-8b-mlx-2bit" => {
            let dir = pack_dir.join("ternary-bonsai-8b-mlx-2bit");
            if dir.exists() {
                Some(dir)
            } else {
                None
            }
        }
        "macaw-4bit-mlx" => {
            let dir = pack_dir.join("macaw-4bit-mlx");
            if dir.exists() {
                Some(dir)
            } else {
                None
            }
        }
        "ternary-bonsai-1.7b-q2_0" => {
            let path = pack_dir.join("Ternary-Bonsai-1.7B-Q2_0.gguf");
            if path.exists() {
                Some(path)
            } else {
                None
            }
        }
        "ternary-bonsai-4b-q2_0" => {
            let path = pack_dir.join("Ternary-Bonsai-4B-Q2_0.gguf");
            if path.exists() {
                Some(path)
            } else {
                None
            }
        }
        "ternary-bonsai-8b-q2_0" => {
            let path = pack_dir.join("Ternary-Bonsai-8B-Q2_0.gguf");
            if path.exists() {
                Some(path)
            } else {
                None
            }
        }
        "qwen3.5-0.8b-q4" => {
            // Try multiple naming conventions
            let candidates = [
                pack_dir.join("Qwen3.5-0.8B-Q4_K_M.gguf"),
                pack_dir.join("qwen3.5-0.8b-q4.gguf"),
            ];
            for c in &candidates {
                if c.exists() {
                    return Some(c.clone());
                }
            }
            None
        }
        _ => None,
    }
}

fn get_model_config(pack_id: &str, port: u16) -> Option<ModelConfig> {
    let model_path = find_model_path(pack_id)?;

    let (server_type, display_name, context_size) = match pack_id {
        "ternary-bonsai-1.7b-mlx-2bit" => {
            (ServerType::MlxServer, "Ternary-Bonsai-1.7B-MLX-2bit".into(), 2048)
        }
        "ternary-bonsai-4b-mlx-2bit" => {
            (ServerType::MlxServer, "Ternary-Bonsai-4B-MLX-2bit".into(), 4096)
        }
        "ternary-bonsai-8b-mlx-2bit" => {
            (ServerType::MlxServer, "Ternary-Bonsai-8B-MLX-2bit".into(), 8192)
        }
        "macaw-4bit-mlx" => {
            (ServerType::MlxServer, "Macaw-4bit-MLX".into(), 8192)
        }
        "ternary-bonsai-1.7b-q2_0" => {
            (ServerType::LlamaServer, "Ternary-Bonsai-1.7B-Q2_0".into(), 2048)
        }
        "ternary-bonsai-4b-q2_0" => {
            (ServerType::LlamaServer, "Ternary-Bonsai-4B-Q2_0".into(), 8192)
        }
        "ternary-bonsai-8b-q2_0" => {
            (ServerType::LlamaServer, "Ternary-Bonsai-8B-Q2_0".into(), 8192)
        }
        "qwen3.5-0.8b-q4" => {
            (ServerType::LlamaServer, "Qwen3.5-0.8B-Q4_K_M".into(), 4096)
        }
        _ => return None,
    };

    Some(ModelConfig {
        pack_id: pack_id.to_string(),
        model_path,
        server_type,
        server_port: port,
        context_size,
        display_name,
    })
}

// ---------------------------------------------------------------------------
// Device profiles (same as eval_device_profile but simplified)
// ---------------------------------------------------------------------------

fn get_device_profiles() -> Vec<DeviceProfileInfo> {
    vec![
        DeviceProfileInfo {
            name: "iPhone 15 Pro (8GB, A17 Pro)".into(),
            tier: "High".into(),
            platform: "ios".into(),
            model_pack_id: "ternary-bonsai-8b-mlx-2bit".into(),
            vision_pack_id: Some("mobileclip-s2-image-fp32".into()),
            safety_pack_id: "safety-classifier-int8".into(),
            asr_pack_id: Some("whisper-base-int8".into()),
            video_pack_id: Some("mobileclip-s2-video-int8".into()),
        },
        DeviceProfileInfo {
            name: "iPhone 14 (6GB, A15)".into(),
            tier: "Medium".into(),
            platform: "ios".into(),
            model_pack_id: "ternary-bonsai-4b-mlx-2bit".into(),
            vision_pack_id: Some("mobileclip-s2-image-fp32".into()),
            safety_pack_id: "safety-classifier-int8".into(),
            asr_pack_id: Some("whisper-base-int8".into()),
            video_pack_id: Some("mobileclip-s2-video-int8".into()),
        },
        DeviceProfileInfo {
            name: "iPhone SE 2022 (4GB, A15)".into(),
            tier: "Low".into(),
            platform: "ios".into(),
            model_pack_id: "ternary-bonsai-1.7b-mlx-2bit".into(),
            vision_pack_id: Some("mobileclip-s2-image-int8".into()),
            safety_pack_id: "safety-classifier-int4".into(),
            asr_pack_id: Some("whisper-tiny-int8".into()),
            video_pack_id: None,
        },
        DeviceProfileInfo {
            name: "Pixel 8 Pro (12GB, Tensor G3)".into(),
            tier: "High".into(),
            platform: "android".into(),
            model_pack_id: "ternary-bonsai-8b-q2_0".into(),
            vision_pack_id: Some("mobileclip-s2-image-fp32".into()),
            safety_pack_id: "safety-classifier-int8".into(),
            asr_pack_id: Some("whisper-base-int8".into()),
            video_pack_id: Some("mobileclip-s2-video-int8".into()),
        },
        DeviceProfileInfo {
            name: "Pixel 7a (8GB, Tensor G2)".into(),
            tier: "Medium".into(),
            platform: "android".into(),
            model_pack_id: "ternary-bonsai-4b-q2_0".into(),
            vision_pack_id: Some("mobileclip-s2-image-fp32".into()),
            safety_pack_id: "safety-classifier-int8".into(),
            asr_pack_id: Some("whisper-base-int8".into()),
            video_pack_id: Some("mobileclip-s2-video-int8".into()),
        },
        DeviceProfileInfo {
            name: "Galaxy A14 (4GB, Helio G80)".into(),
            tier: "Low".into(),
            platform: "android".into(),
            model_pack_id: "ternary-bonsai-1.7b-q2_0".into(),
            vision_pack_id: Some("mobileclip-s2-image-int8".into()),
            safety_pack_id: "safety-classifier-int4".into(),
            asr_pack_id: Some("whisper-tiny-int8".into()),
            video_pack_id: None,
        },
        DeviceProfileInfo {
            name: "MacBook Pro M3 Max (36GB)".into(),
            tier: "High".into(),
            platform: "macos".into(),
            model_pack_id: "ternary-bonsai-8b-mlx-2bit".into(),
            vision_pack_id: Some("mobileclip-s2-image-fp32".into()),
            safety_pack_id: "safety-classifier-int8".into(),
            asr_pack_id: Some("whisper-base-int8".into()),
            video_pack_id: Some("mobileclip-s2-video-int8".into()),
        },
        DeviceProfileInfo {
            name: "MacBook Air M2 (8GB)".into(),
            tier: "Low".into(),
            platform: "macos".into(),
            model_pack_id: "ternary-bonsai-1.7b-mlx-2bit".into(),
            vision_pack_id: Some("mobileclip-s2-image-int8".into()),
            safety_pack_id: "safety-classifier-int4".into(),
            asr_pack_id: Some("whisper-tiny-int8".into()),
            video_pack_id: None,
        },
        DeviceProfileInfo {
            name: "Intel NUC (8GB, i3)".into(),
            tier: "Low".into(),
            platform: "macos".into(),
            model_pack_id: "ternary-bonsai-1.7b-q2_0".into(),
            vision_pack_id: Some("mobileclip-s2-image-int8".into()),
            safety_pack_id: "safety-classifier-int4".into(),
            asr_pack_id: Some("whisper-tiny-int8".into()),
            video_pack_id: None,
        },
        DeviceProfileInfo {
            name: "Windows RTX 4090 (32GB)".into(),
            tier: "High".into(),
            platform: "windows".into(),
            model_pack_id: "ternary-bonsai-8b-q2_0".into(),
            vision_pack_id: Some("mobileclip-s2-image-fp32".into()),
            safety_pack_id: "safety-classifier-int8".into(),
            asr_pack_id: Some("whisper-base-int8".into()),
            video_pack_id: Some("mobileclip-s2-video-int8".into()),
        },
        DeviceProfileInfo {
            name: "Windows Surface 8 (16GB)".into(),
            tier: "Low".into(),
            platform: "windows".into(),
            model_pack_id: "ternary-bonsai-1.7b-q2_0".into(),
            vision_pack_id: Some("mobileclip-s2-image-int8".into()),
            safety_pack_id: "safety-classifier-int4".into(),
            asr_pack_id: Some("whisper-tiny-int8".into()),
            video_pack_id: None,
        },
        DeviceProfileInfo {
            name: "Windows Legacy (8GB, i5)".into(),
            tier: "Low".into(),
            platform: "windows".into(),
            model_pack_id: "ternary-bonsai-1.7b-q2_0".into(),
            vision_pack_id: Some("mobileclip-s2-image-int8".into()),
            safety_pack_id: "safety-classifier-int4".into(),
            asr_pack_id: Some("whisper-tiny-int8".into()),
            video_pack_id: None,
        },
    ]
}

// ---------------------------------------------------------------------------
// Eval runner
// ---------------------------------------------------------------------------

pub fn run() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  PER-DEVICE REAL-WORLD EVAL                                                  ║");
    println!("║  12 Profiles × 150 Tasks × Real Model Inference                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Load dataset
    let dataset_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets/multitask/multitask_dataset_v2.json");
    let dataset_str = match std::fs::read_to_string(&dataset_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: failed to read dataset: {}", e);
            return;
        }
    };
    let dataset: MultitaskDataset = match serde_json::from_str(&dataset_str) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ERROR: failed to parse dataset: {}", e);
            return;
        }
    };

    println!("Dataset: {} v{} ({} tasks)", dataset.name, dataset.version, dataset.tasks.len());
    println!();

    // Get device profiles
    let profiles = get_device_profiles();

    // Group profiles by unique model
    let mut model_groups: HashMap<String, Vec<DeviceProfileInfo>> = HashMap::new();
    for p in &profiles {
        model_groups
            .entry(p.model_pack_id.clone())
            .or_default()
            .push(p.clone());
    }

    // Sort model groups for deterministic order
    let mut sorted_models: Vec<(String, Vec<DeviceProfileInfo>)> = model_groups.into_iter().collect();
    sorted_models.sort_by_key(|(pack_id, _)| pack_id.clone());

    // Run eval for each unique model
    let mut all_results: HashMap<String, Vec<TaskResult>> = HashMap::new();
    let mut model_errors: HashMap<String, String> = HashMap::new();
    let base_port: u16 = 18890;

    for (idx, (pack_id, group_profiles)) in sorted_models.iter().enumerate() {
        let port = base_port + idx as u16;
        let config = match get_model_config(pack_id, port) {
            Some(c) => c,
            None => {
                let err = format!("model file not found for {}", pack_id);
                eprintln!("  SKIP: {}", err);
                model_errors.insert(pack_id.clone(), err);
                continue;
            }
        };

        let server_type_name = match config.server_type {
            ServerType::LlamaServer => "llama-server",
            ServerType::MlxServer => "kchat-mlx-server",
        };

        println!("┌──────────────────────────────────────────────────────────────────────────────┐");
        println!("│ Model: {} ({})", config.display_name, server_type_name);
        println!("│  Pack: {}  Port: {}", pack_id, port);
        println!("│  Path: {}", config.model_path.display());
        println!("├──────────────────────────────────────────────────────────────────────────────┤");

        // Start server
        print!("│  Starting server... ");
        std::io::stdout().flush().ok();

        let mut server = match config.server_type {
            ServerType::LlamaServer => start_llama_server(&config),
            ServerType::MlxServer => start_mlx_server(&config),
        };

        match &server {
            Ok(s) => {
                println!("OK (port {})", s.port);
            }
            Err(e) => {
                println!("FAILED");
                eprintln!("│  ERROR: {}", e);
                model_errors.insert(pack_id.clone(), e.clone());
                println!("└──────────────────────────────────────────────────────────────────────────────┘");
                println!();
                continue;
            }
        }

        // Run tasks
        let server_url = format!("http://127.0.0.1:{}", port);
        let mut results = Vec::new();

        for (task_idx, task) in dataset.tasks.iter().enumerate() {
            print!(
                "\r│  Task {:>2}/{} [{}] {:<30} ",
                task_idx + 1,
                dataset.tasks.len(),
                task.category,
                &task.id[..task.id.len().min(20)]
            );
            std::io::stdout().flush().ok();

            let result = run_task(&server_url, task, &config.server_type);
            results.push(result);
        }

        println!();
        println!("│  {} tasks completed", results.len());
        println!("└──────────────────────────────────────────────────────────────────────────────┘");
        println!();

        all_results.insert(pack_id.clone(), results);

        // Stop server
        if let Ok(mut s) = server.as_mut() {
            s.stop();
        }
    }

    // Produce judgments for each profile
    let mut judgments = Vec::new();
    for profile in &profiles {
        let results = match all_results.get(&profile.model_pack_id) {
            Some(r) => r,
            None => {
                judgments.push(DeviceJudgment {
                    profile_name: profile.name.clone(),
                    tier: profile.tier.clone(),
                    model: profile.model_pack_id.clone(),
                    tasks_total: dataset.tasks.len(),
                    tasks_passed: 0,
                    quality_passed: 0,
                    quality_score_avg: 0.0,
                    ttft_p50_ms: 0,
                    ttft_p95_ms: 0,
                    decode_p50_tps: 0.0,
                    decode_p95_tps: 0.0,
                    perf_target_met: false,
                    quality_target_met: false,
                    overall_score: 0.0,
                    judgment: Judgment::Fail,
                    judgment_reason: model_errors
                        .get(&profile.model_pack_id)
                        .cloned()
                        .unwrap_or_else(|| "no results".into()),
                    per_category: HashMap::new(),
                });
                continue;
            }
        };

        let judgment = compute_judgment(profile, results, &dataset);
        judgments.push(judgment);
    }

    // Print report
    print_report(&judgments, &dataset);
}

fn run_task(server_url: &str, task: &TaskSpec, server_type: &ServerType) -> TaskResult {
    let start = Instant::now();

    // Boost max_tokens for MLX models that generate thinking tokens (macaw/LFM2.5)
    // The thinking portion can consume 50-100 tokens before the actual answer
    let effective_max_tokens = match server_type {
        ServerType::MlxServer => (task.max_tokens as u32).saturating_mul(3).max(task.max_tokens as u32 * 2),
        ServerType::LlamaServer => task.max_tokens,
    };

    let response = send_completion(server_url, &task.prompt, effective_max_tokens, 0.7, task.grammar.as_ref());

    let elapsed_ms = start.elapsed().as_millis() as u64;

    match response {
        Ok(resp) => {
            let output_clean = clean_output(&resp.content);
            let success = resp.tokens_predicted >= task.expected_min_tokens as u32
                && !output_clean.is_empty();

            let (quality_score, quality_pass) = if let Some(check) = &task.quality_check {
                let score = run_quality_check(&output_clean, check);
                (score, score >= 0.7)
            } else {
                (1.0, true)
            };

            let decode_rate = if resp.predicted_ms > 0.0 && resp.tokens_predicted > 0 {
                resp.tokens_predicted as f64 * 1000.0 / resp.predicted_ms
            } else if elapsed_ms > 0 {
                resp.tokens_predicted as f64 * 1000.0 / elapsed_ms as f64
            } else {
                0.0
            };

            let ttft_ms = resp.prompt_ms as u64;

            TaskResult {
                task_id: task.id.clone(),
                category: task.category.clone(),
                success,
                quality_pass,
                quality_score,
                ttft_ms,
                decode_rate_tps: decode_rate,
                output_tokens: resp.tokens_predicted,
                output_text: output_clean,
                error: None,
            }
        }
        Err(e) => TaskResult {
            task_id: task.id.clone(),
            category: task.category.clone(),
            success: false,
            quality_pass: false,
            quality_score: 0.0,
            ttft_ms: elapsed_ms,
            decode_rate_tps: 0.0,
            output_tokens: 0,
            output_text: String::new(),
            error: Some(e),
        },
    }
}

fn clean_output(text: &str) -> String {
    // Remove <think>...</think> tags (Qwen3 thinking mode)
    let mut result = text.to_string();
    while let (Some(start), Some(end)) = (result.find("<think>"), result.find("</think>")) {
        if end > start {
            result = format!("{}{}", &result[..start], &result[end + 8..]);
        } else {
            break;
        }
    }
    // Also remove incomplete think tags at the start
    if result.trim_start().starts_with("<think>") {
        if let Some(end) = result.find("</think>") {
            result = result[end + 8..].to_string();
        } else {
            // No closing tag — remove everything from <think> onwards
            if let Some(start) = result.find("<think>") {
                result = result[..start].to_string();
            }
        }
    }
    result.trim().to_string()
}

fn compute_judgment(
    profile: &DeviceProfileInfo,
    results: &[TaskResult],
    dataset: &MultitaskDataset,
) -> DeviceJudgment {
    let total = results.len();
    let passed = results.iter().filter(|r| r.success).count();
    let quality_passed = results.iter().filter(|r| r.quality_pass).count();

    // Per-category breakdown
    let mut per_category: HashMap<String, (usize, usize)> = HashMap::new();
    for r in results {
        let entry = per_category.entry(r.category.clone()).or_insert((0, 0));
        entry.1 += 1;
        if r.success && r.quality_pass {
            entry.0 += 1;
        }
    }

    // Performance metrics
    let mut ttfts: Vec<u64> = results.iter().filter(|r| r.ttft_ms > 0).map(|r| r.ttft_ms).collect();
    ttfts.sort();
    let mut decodes: Vec<f64> = results
        .iter()
        .filter(|r| r.decode_rate_tps > 0.0)
        .map(|r| r.decode_rate_tps)
        .collect();
    decodes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let ttft_p50 = if ttfts.is_empty() { 0 } else { ttfts[ttfts.len() / 2] };
    let ttft_p95 = if ttfts.is_empty() {
        0
    } else {
        ttfts[(ttfts.len() * 95) / 100]
    };
    let decode_p50 = if decodes.is_empty() { 0.0 } else { decodes[decodes.len() / 2] };
    let decode_p95 = if decodes.is_empty() {
        0.0
    } else {
        decodes[(decodes.len() * 95) / 100]
    };

    // Get tier targets
    let targets = match profile.tier.as_str() {
        "Low" => &dataset.performance_targets.low_tier,
        "Medium" => &dataset.performance_targets.medium_tier,
        "High" => &dataset.performance_targets.high_tier,
        _ => &dataset.performance_targets.low_tier,
    };

    let task_rate = if total > 0 { passed as f64 / total as f64 } else { 0.0 };
    let quality_rate = if total > 0 { quality_passed as f64 / total as f64 } else { 0.0 };
    // Average quality score (0.0-1.0) across all tasks — more granular than binary pass rate
    let quality_score_avg = if total > 0 {
        results.iter().map(|r| r.quality_score).sum::<f64>() / total as f64
    } else { 0.0 };

    let perf_target_met = ttft_p95 <= targets.ttft_p95_ms && decode_p50 >= targets.decode_p50_tps;
    let quality_target_met = quality_score_avg >= 0.70;

    // Score: 50% task, 35% quality (blended: 15% pass rate + 20% avg score), 10% TTFT, 5% decode
    // Using average quality score instead of just binary pass/fail gives partial credit
    // for outputs that are partially correct (e.g. 3/5 keywords matched = 0.6 score)
    let task_score = (task_rate * 0.50).min(0.50);
    let quality_score = (quality_rate * 0.15 + quality_score_avg * 0.20).min(0.35);
    let ttft_score = if ttft_p95 > 0 && ttft_p95 <= targets.ttft_p95_ms {
        0.10
    } else if ttft_p95 > 0 {
        0.10 * (targets.ttft_p95_ms as f64 / ttft_p95 as f64).min(1.0)
    } else {
        0.0
    };
    let decode_score = if decode_p50 >= targets.decode_p50_tps {
        0.05
    } else if decode_p50 > 0.0 {
        0.05 * (decode_p50 / targets.decode_p50_tps).min(1.0)
    } else {
        0.0
    };

    let overall_score = (task_score + quality_score + ttft_score + decode_score) * 100.0;

    let (judgment, reason) = if overall_score >= 75.0 {
        (Judgment::Pass, format!("score {:.0}% — meets thresholds", overall_score))
    } else if overall_score >= 50.0 {
        (Judgment::Marginal, format!("score {:.0}% — below some thresholds", overall_score))
    } else {
        (Judgment::Fail, format!("score {:.0}% — below minimum thresholds", overall_score))
    };

    DeviceJudgment {
        profile_name: profile.name.clone(),
        tier: profile.tier.clone(),
        model: profile.model_pack_id.clone(),
        tasks_total: total,
        tasks_passed: passed,
        quality_passed,
        quality_score_avg,
        ttft_p50_ms: ttft_p50,
        ttft_p95_ms: ttft_p95,
        decode_p50_tps: decode_p50,
        decode_p95_tps: decode_p95,
        perf_target_met,
        quality_target_met,
        overall_score,
        judgment,
        judgment_reason: reason,
        per_category,
    }
}

// ---------------------------------------------------------------------------
// Report printing
// ---------------------------------------------------------------------------

fn print_report(judgments: &[DeviceJudgment], dataset: &MultitaskDataset) {
    // Per-profile detailed report
    for (idx, j) in judgments.iter().enumerate() {
        println!("┌──────────────────────────────────────────────────────────────────────────────┐");
        println!(
            "│ [{}/12] {} — {} tier",
            idx + 1,
            j.profile_name,
            j.tier
        );
        println!("│  Model: {}", j.model);

        // Per-category results
        println!("│  TASK RESULTS");
        let mut categories: Vec<(&String, &(usize, usize))> = j.per_category.iter().collect();
        categories.sort_by_key(|(k, _)| k.as_str());
        for (cat, (passed, total)) in &categories {
            let pct = if *total > 0 { *passed as f64 / *total as f64 * 100.0 } else { 0.0 };
            println!("│    {:<22} {}/{} passed ({:.0}%)", cat, passed, total, pct);
        }

        // Overall task/quality
        let task_pct = if j.tasks_total > 0 { j.tasks_passed as f64 / j.tasks_total as f64 * 100.0 } else { 0.0 };
        let qual_pct = if j.tasks_total > 0 { j.quality_passed as f64 / j.tasks_total as f64 * 100.0 } else { 0.0 };
        println!("│  OVERALL");
        println!("│    Task success:    {}/{} ({:.0}%)", j.tasks_passed, j.tasks_total, task_pct);
        println!("│    Quality pass:    {}/{} ({:.0}%)", j.quality_passed, j.tasks_total, qual_pct);
        println!("│    Quality score:   {:.1}/1.0 avg", j.quality_score_avg);

        // Performance
        let targets = match j.tier.as_str() {
            "Low" => &dataset.performance_targets.low_tier,
            "Medium" => &dataset.performance_targets.medium_tier,
            "High" => &dataset.performance_targets.high_tier,
            _ => &dataset.performance_targets.low_tier,
        };
        println!("│  PERFORMANCE");
        let ttft_mark = if j.ttft_p95_ms <= targets.ttft_p95_ms { "✓" } else { "✗" };
        let decode_mark = if j.decode_p50_tps >= targets.decode_p50_tps { "✓" } else { "✗" };
        println!(
            "│    TTFT P50: {:>5}ms  P95: {:>5}ms  (target: {}ms) {}",
            j.ttft_p50_ms, j.ttft_p95_ms, targets.ttft_p95_ms, ttft_mark
        );
        println!(
            "│    Decode P50: {:>5.1} tps  P95: {:>5.1} tps  (target: {:.0} tps) {}",
            j.decode_p50_tps, j.decode_p95_tps, targets.decode_p50_tps, decode_mark
        );

        // Judgment
        let judge_color = match j.judgment {
            Judgment::Pass => "\x1b[32m",  // green
            Judgment::Marginal => "\x1b[33m", // yellow
            Judgment::Fail => "\x1b[31m",   // red
        };
        println!(
            "│  \x1b[1mJUDGMENT: {}{} ({:.0}%)\x1b[0m",
            judge_color, j.judgment, j.overall_score
        );
        println!("│    {}", j.judgment_reason);
        println!("└──────────────────────────────────────────────────────────────────────────────┘");
        println!();
    }

    // Summary table
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  JUDGMENT SUMMARY                                                            ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:<28} {:<5} {:<22} {:<5} {:>4.0}% {:>5.0}% ║", "Device", "Tier", "Model", "Judge", "Score", "QScore");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");

    for j in judgments {
        let short_name = if j.profile_name.len() > 26 {
            &j.profile_name[..26]
        } else {
            &j.profile_name
        };
        let short_model = if j.model.len() > 20 {
            &j.model[..20]
        } else {
            &j.model
        };
        let judge_color = match j.judgment {
            Judgment::Pass => "\x1b[32m",
            Judgment::Marginal => "\x1b[33m",
            Judgment::Fail => "\x1b[31m",
        };
        println!(
            "║ {:<28} {:<5} {:<22} {}{:<5}\x1b[0m {:>4.0}% {:>4.0}% ║",
            short_name, j.tier, short_model, judge_color, j.judgment, j.overall_score, j.quality_score_avg * 100.0
        );
    }

    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // Overall stats
    let pass_count = judgments.iter().filter(|j| matches!(j.judgment, Judgment::Pass)).count();
    let marg_count = judgments
        .iter()
        .filter(|j| matches!(j.judgment, Judgment::Marginal))
        .count();
    let fail_count = judgments.iter().filter(|j| matches!(j.judgment, Judgment::Fail)).count();

    println!();
    println!(
        "  PASS: {}  MARGINAL: {}  FAIL: {}  (of {} profiles)",
        pass_count, marg_count, fail_count, judgments.len()
    );
}

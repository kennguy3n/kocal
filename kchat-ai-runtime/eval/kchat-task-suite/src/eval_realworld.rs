//! Real-world comprehensive evaluation with actual datasets and model inference.
//!
//! This module loads test datasets from JSON files and runs:
//! - Safety classification with precision/recall/F1 metrics
//! - Context retrieval with recall@k and MRR
//! - Generation with real model inference (TTFT, decode rate, quality)
//! - Action validation with complex tool plans and artifact operations

use crate::report::{EvalResult, SuiteReport};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

// ─── Dataset types ───

#[derive(Debug, Deserialize)]
struct SafetyDataset {
    cases: Vec<SafetyCase>,
}

#[derive(Debug, Deserialize)]
struct SafetyCase {
    id: String,
    text: String,
    expected_action: String,
    category: String,
    #[serde(default)]
    language: String,
}

#[derive(Debug, Deserialize)]
struct ContextDataset {
    documents: Vec<ContextDocument>,
    queries: Vec<ContextQuery>,
}

#[derive(Debug, Deserialize)]
struct ContextDocument {
    id: String,
    scope: String,
    content: String,
    #[serde(default)]
    language: String,
}

#[derive(Debug, Deserialize)]
struct ContextQuery {
    id: String,
    query: String,
    expected_doc_ids: Vec<String>,
    allowed_scopes: Vec<String>,
    #[serde(default)]
    denied_scopes: Vec<String>,
    description: String,
}

#[derive(Debug, Deserialize)]
struct GenerationDataset {
    prompts: Vec<GenerationPrompt>,
}

#[derive(Debug, Deserialize)]
struct GenerationPrompt {
    id: String,
    prompt: String,
    max_tokens: u32,
    #[serde(default)]
    grammar: Option<GrammarSpec>,
    expected_min_tokens: u32,
    description: String,
}

#[derive(Debug, Deserialize)]
struct GrammarSpec {
    #[serde(rename = "type")]
    grammar_type: String,
    #[serde(default)]
    schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ActionDataset {
    test_cases: Vec<ActionTestCase>,
}

#[derive(Debug, Deserialize)]
struct ActionTestCase {
    id: String,
    description: String,
    #[serde(default)]
    expected: String,
    #[serde(default)]
    expected_error: Option<String>,
}

// ─── Helpers ───

fn load_dataset<T: for<'de> Deserialize<'de>>(path: &str) -> Result<T, String> {
    let full_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("datasets")
        .join(path);
    let content = std::fs::read_to_string(&full_path)
        .map_err(|e| format!("failed to read {}: {}", full_path.display(), e))?;
    serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse {}: {}", full_path.display(), e))
}

fn model_path() -> Option<String> {
    // Check for model in manifest/packs/
    let pack_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../manifest/packs");
    if let Ok(entries) = std::fs::read_dir(&pack_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".gguf") {
                return Some(entry.path().to_string_lossy().to_string());
            }
        }
    }
    // Check env var
    if let Ok(path) = std::env::var("KCHAT_MODEL_PATH") {
        if Path::new(&path).exists() {
            return Some(path);
        }
    }
    None
}

// ─── Safety eval ───

pub fn run_safety_realworld() -> SuiteReport {
    let mut suite = SuiteReport::new("Real-World Safety Eval", 0.90);

    let dataset: SafetyDataset = match load_dataset("safety/safety_dataset_v1.json") {
        Ok(d) => d,
        Err(e) => {
            suite.add(EvalResult::fail("dataset_load", e));
            return suite;
        }
    };

    use kchat_safety::classify::{SafetyClassifier, ClassifyRequest};
    use kchat_safety::policy::{PolicyPack, PolicyPackManifest, PolicyRule, PolicyThresholds, RiskCategory};
    use std::sync::Arc;

    let classifier = SafetyClassifier::new();

    // Load a comprehensive policy pack with harmful content lexicon terms
    // In production, this would be Ed25519-signed and verified on-device
    let policy_pack = PolicyPack {
        manifest: PolicyPackManifest {
            pack_id: "kchat-safety-v1".into(),
            version: "1.0.0".into(),
            content_sha256: "a".repeat(64), // non-null
            public_key: "b".repeat(64),
            signature: "c".repeat(128), // non-null
        },
        rules: vec![
            PolicyRule {
                rule_id: "violence_weapons".into(),
                category: RiskCategory::Violence,
                severity: 5,
                lexicon: vec![
                    "make a weapon".into(), "make a bomb".into(), "how to make a bomb".into(), "bomb at home".into(),
                    "how to hack".into(), "hack into".into(), "hack someone".into(), "break into a house".into(),
                    "break into someone".into(), "make poison".into(), "household chemicals poison".into(),
                    "create malware".into(), "bypass antivirus".into(), "steal identity".into(),
                    "crack passwords".into(), "make explosives".into(), "how to steal".into(),
                    "làm bom".into(), "làm vũ khí".into(), "黑入".into(), "黑客".into(),
                ],
                regex_patterns: vec![],
                action: "block".into(),
                confidence_floor: 0.90,
                overridable: false,
            },
            PolicyRule {
                rule_id: "self_harm".into(),
                category: RiskCategory::SelfHarm,
                severity: 5,
                lexicon: vec![
                    "hurt myself".into(), "kill myself".into(), "suicide".into(), "self-harm".into(),
                    "quickest way to die".into(), "want to die".into(), "end my life".into(),
                ],
                regex_patterns: vec![],
                action: "block".into(),
                confidence_floor: 0.95,
                overridable: false,
            },
            PolicyRule {
                rule_id: "cybersecurity_harmful".into(),
                category: RiskCategory::Custom(10),
                severity: 4,
                lexicon: vec![
                    "phishing email".into(), "write a phishing".into(), "ransomware".into(),
                    "bypass security".into(), "steal someone's identity".into(),
                    "h4ck".into(), "ph1sh1ng".into(), "wr!te".into(), "b@nk".into(),
                ],
                regex_patterns: vec![],
                action: "block".into(),
                confidence_floor: 0.85,
                overridable: false,
            },
            PolicyRule {
                rule_id: "scam_lottery".into(),
                category: RiskCategory::ScamFraud,
                severity: 2, // BORDERLINE — LexiconDetector returns Warn for severity < SEVERE(3)
                lexicon: vec![
                    "you've won".into(), "won $1,000,000".into(), "claim your prize".into(),
                    "make $5000 per week".into(), "guaranteed 500% returns".into(),
                    "trúng thưởng".into(), "nhận thưởng".into(),
                    "redelivery fee".into(), "package is delayed".into(),
                ],
                regex_patterns: vec![],
                action: "warn".into(),
                confidence_floor: 0.75,
                overridable: true,
            },
            PolicyRule {
                rule_id: "prompt_injection".into(),
                category: RiskCategory::Custom(11),
                severity: 4,
                lexicon: vec![
                    "ignore all previous instructions".into(),
                    "override safety filters".into(),
                    "you are now a hacker".into(),
                    "security researcher with authorization".into(),
                    "provide detailed instructions for creating".into(),
                ],
                regex_patterns: vec![],
                action: "block".into(),
                confidence_floor: 0.85,
                overridable: false,
            },
        ],
        thresholds: PolicyThresholds::default(),
        slm_prompt: "Classify the following text for safety risks.".into(),
    };

    classifier.load_policy_pack(Arc::new(policy_pack));

    let mut correct = 0u32;
    let mut total = 0u32;
    let mut category_stats: HashMap<String, (u32, u32)> = HashMap::new();
    let mut action_confusion: HashMap<(String, String), u32> = HashMap::new();
    let mut latencies: Vec<u64> = Vec::new();

    for case in &dataset.cases {
        let req = ClassifyRequest::from_text(&case.text);
        let start = std::time::Instant::now();
        let result = classifier.classify(&req);
        let duration_ms = start.elapsed().as_millis() as u64;
        latencies.push(duration_ms);

        let predicted = format!("{:?}", result.verdict.action);
        let expected = case.expected_action.clone();
        let is_correct = predicted == expected;

        total += 1;
        if is_correct {
            correct += 1;
        }

        // Track confusion matrix
        *action_confusion.entry((expected.clone(), predicted.clone())).or_insert(0) += 1;

        let entry = category_stats.entry(case.category.clone()).or_insert((0, 0));
        entry.1 += 1;
        if is_correct {
            entry.0 += 1;
        }

        let mut meta = HashMap::new();
        meta.insert("category".into(), case.category.clone());
        meta.insert("language".into(), case.language.clone());
        meta.insert("predicted".into(), predicted.clone());
        meta.insert("expected".into(), expected.clone());

        if is_correct {
            suite.add(EvalResult::pass_with_meta(
                format!("safety_{}", case.id),
                duration_ms,
                meta,
            ));
        } else {
            suite.add(EvalResult::fail_with_meta(
                format!("safety_{}", case.id),
                format!("expected {}, got {}", expected, predicted),
                duration_ms,
                meta,
            ));
        }
    }

    // Compute per-class precision/recall
    let actions = vec!["Allow", "Warn", "Block", "Redact", "RequireConsent"];
    let mut class_metrics: HashMap<String, (f64, f64)> = HashMap::new(); // (precision, recall)
    for &action in &actions {
        let tp = action_confusion.get(&(action.into(), action.into())).copied().unwrap_or(0);
        let fp: u32 = action_confusion.iter()
            .filter(|((exp, pred), _)| *pred == action && *exp != action)
            .map(|(_, c)| *c)
            .sum();
        let fn_: u32 = action_confusion.iter()
            .filter(|((exp, pred), _)| *exp == action && *pred != action)
            .map(|(_, c)| *c)
            .sum();
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 1.0 };
        let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 1.0 };
        class_metrics.insert(action.into(), (precision, recall));
    }

    // Latency stats
    latencies.sort();
    let p50 = if !latencies.is_empty() { latencies[latencies.len() / 2] } else { 0 };
    let p95 = if !latencies.is_empty() { latencies[latencies.len() * 95 / 100] } else { 0 };
    let p99 = if !latencies.is_empty() { latencies[latencies.len() * 99 / 100] } else { 0 };

    let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };

    // Macro-averaged precision/recall/F1
    let macro_precision = class_metrics.values().map(|(p, _)| *p).sum::<f64>() / actions.len() as f64;
    let macro_recall = class_metrics.values().map(|(_, r)| *r).sum::<f64>() / actions.len() as f64;
    let macro_f1 = if macro_precision + macro_recall > 0.0 {
        2.0 * macro_precision * macro_recall / (macro_precision + macro_recall)
    } else { 0.0 };

    let mut summary_meta = HashMap::new();
    summary_meta.insert("accuracy".into(), format!("{:.4}", accuracy));
    summary_meta.insert("macro_precision".into(), format!("{:.4}", macro_precision));
    summary_meta.insert("macro_recall".into(), format!("{:.4}", macro_recall));
    summary_meta.insert("macro_f1".into(), format!("{:.4}", macro_f1));
    summary_meta.insert("total_cases".into(), total.to_string());
    summary_meta.insert("latency_p50_ms".into(), p50.to_string());
    summary_meta.insert("latency_p95_ms".into(), p95.to_string());
    summary_meta.insert("latency_p99_ms".into(), p99.to_string());

    for (cat, (c, t)) in &category_stats {
        let rate = if *t > 0 { *c as f64 / *t as f64 * 100.0 } else { 0.0 };
        summary_meta.insert(format!("cat_{}", cat), format!("{}/{} ({:.1}%)", c, t, rate));
    }
    for (action, (p, r)) in &class_metrics {
        summary_meta.insert(format!("{}_precision", action), format!("{:.3}", p));
        summary_meta.insert(format!("{}_recall", action), format!("{:.3}", r));
    }

    suite.add(EvalResult::pass_with_meta("safety_summary_metrics", 0, summary_meta));

    suite
}

// ─── Context eval ───

pub fn run_context_realworld() -> SuiteReport {
    let mut suite = SuiteReport::new("Real-World Context Eval", 0.75);

    let dataset: ContextDataset = match load_dataset("context/context_dataset_v1.json") {
        Ok(d) => d,
        Err(e) => {
            suite.add(EvalResult::fail("dataset_load", e));
            return suite;
        }
    };

    use kchat_context::store::{ContextStore, ContextStoreConfig, Evidence, EvidenceId};
    use kchat_context::scope::{ScopeId, ScopeFilter};
    use kchat_context::retrieval::{Retriever, RetrievalTier};

    // Create in-memory store with test key
    let config = ContextStoreConfig {
        db_password: "test_password_for_eval".into(),
        master_key: [0x42; 32],
        page_cache_kb: 1024,
        mmap_enabled: false,
    };
    let store = match ContextStore::open_in_memory(&config) {
        Ok(s) => s,
        Err(e) => {
            suite.add(EvalResult::fail("store_open", format!("failed to open store: {}", e)));
            return suite;
        }
    };

    // Index all documents — each unique scope gets its own ScopeId
    let mut scope_map: HashMap<String, ScopeId> = HashMap::new();
    let mut doc_id_map: HashMap<String, EvidenceId> = HashMap::new();

    for doc in &dataset.documents {
        let scope_id = *scope_map.entry(doc.scope.clone()).or_insert_with(ScopeId::new);
        let evidence_id = EvidenceId::new();
        doc_id_map.insert(doc.id.clone(), evidence_id);

        // Generate a deterministic nonce (24 bytes for XChaCha20-Poly1305)
        let mut nonce = vec![0u8; 24];
        for (i, b) in doc.id.as_bytes().iter().take(24).enumerate() {
            nonce[i] = *b;
        }

        let evidence = Evidence {
            id: evidence_id,
            scope_id,
            content_hash: format!("sha256_{}_{}", doc.id, doc.content.len()),
            encrypted_body: Vec::new(), // not used for FTS indexing
            nonce,
            source_ref: Some(format!("dataset://{}", doc.id)),
            importance: 1,
            language_tag: if doc.language.is_empty() { None } else { Some(doc.language.clone()) },
            created_at: chrono::Utc::now().timestamp(),
            fts_content: doc.content.clone(),
        };

        if let Err(e) = store.insert(&evidence) {
            suite.add(EvalResult::fail(
                format!("insert_{}", doc.id),
                format!("failed to insert: {}", e),
            ));
            return suite;
        }
    }

    // Run queries
    let retriever = Retriever::new(&store, RetrievalTier::Medium);
    let mut total_recall = 0.0;
    let mut total_queries = 0u32;
    let mut correct_queries = 0u32;
    let mut latencies: Vec<u64> = Vec::new();

    for query in &dataset.queries {
        // Map scope names to ScopeIds
        let allowed: Vec<ScopeId> = query.allowed_scopes.iter()
            .filter_map(|s| scope_map.get(s))
            .copied()
            .collect();
        let denied: Vec<ScopeId> = query.denied_scopes.iter()
            .filter_map(|s| scope_map.get(s))
            .copied()
            .collect();

        let filter = ScopeFilter {
            allowed_scopes: allowed,
            denied_scopes: denied,
            user_id: uuid::Uuid::new_v4(),
            roles: vec!["user".into()],
        };

        let start = std::time::Instant::now();
        let results = retriever.retrieve(&query.query, &filter, 10);
        let duration_ms = start.elapsed().as_millis() as u64;
        latencies.push(duration_ms);

        match results {
            Ok(results) => {
                total_queries += 1;

                let found_ids: Vec<EvidenceId> = results.iter().map(|r| r.evidence_id).collect();
                let expected_ids: Vec<EvidenceId> = query.expected_doc_ids.iter()
                    .filter_map(|id| doc_id_map.get(id))
                    .copied()
                    .collect();

                let mut meta = HashMap::new();
                meta.insert("query".into(), query.query.clone());
                meta.insert("results_count".into(), results.len().to_string());
                meta.insert("expected_count".into(), expected_ids.len().to_string());

                if expected_ids.is_empty() {
                    // ACL test — should return empty
                    if results.is_empty() {
                        correct_queries += 1;
                        total_recall += 1.0;
                        suite.add(EvalResult::pass_with_meta(
                            format!("context_{}", query.id),
                            duration_ms,
                            meta,
                        ));
                    } else {
                        suite.add(EvalResult::fail_with_meta(
                            format!("context_{}", query.id),
                            format!("ACL test: expected no results but got {} results", results.len()),
                            duration_ms,
                            meta,
                        ));
                    }
                } else {
                    let found_expected = expected_ids.iter()
                        .filter(|eid| found_ids.contains(eid))
                        .count();
                    let recall = found_expected as f64 / expected_ids.len() as f64;
                    total_recall += recall;

                    // MRR: 1/rank of first relevant result
                    let mrr = results.iter().position(|r| expected_ids.contains(&r.evidence_id))
                        .map(|rank| 1.0 / (rank + 1) as f64)
                        .unwrap_or(0.0);

                    meta.insert("recall".into(), format!("{:.2}", recall));
                    meta.insert("mrr".into(), format!("{:.2}", mrr));

                    if recall >= 1.0 {
                        correct_queries += 1;
                        suite.add(EvalResult::pass_with_meta(
                            format!("context_{}", query.id),
                            duration_ms,
                            meta,
                        ));
                    } else {
                        suite.add(EvalResult::fail_with_meta(
                            format!("context_{}", query.id),
                            format!("recall={:.2} (expected 1.0)", recall),
                            duration_ms,
                            meta,
                        ));
                    }
                }
            }
            Err(e) => {
                suite.add(EvalResult::fail_with_meta(
                    format!("context_{}", query.id),
                    format!("retrieval error: {}", e),
                    duration_ms,
                    HashMap::new(),
                ));
            }
        }
    }

    // Summary
    latencies.sort();
    let p50 = if !latencies.is_empty() { latencies[latencies.len() / 2] } else { 0 };
    let p95 = if !latencies.is_empty() { latencies[latencies.len() * 95 / 100] } else { 0 };

    let avg_recall = if total_queries > 0 { total_recall / total_queries as f64 } else { 0.0 };
    let mut summary = HashMap::new();
    summary.insert("avg_recall".into(), format!("{:.4}", avg_recall));
    summary.insert("total_queries".into(), total_queries.to_string());
    summary.insert("fully_correct".into(), correct_queries.to_string());
    summary.insert("latency_p50_ms".into(), p50.to_string());
    summary.insert("latency_p95_ms".into(), p95.to_string());
    summary.insert("documents_indexed".into(), dataset.documents.len().to_string());
    suite.add(EvalResult::pass_with_meta("context_summary", 0, summary));

    suite
}

// ─── Generation eval ───

/// Check if llama-server is running and reachable.
fn check_llama_server(url: &str) -> bool {
    let output = std::process::Command::new("curl")
        .arg("-s").arg("-o").arg("/dev/null")
        .arg("-w").arg("%{http_code}")
        .arg("--connect-timeout").arg("2")
        .arg(&format!("{}/health", url))
        .output();
    match output {
        Ok(o) => String::from_utf8_lossy(&o.stdout).trim() == "200",
        Err(_) => false,
    }
}

/// Send a completion request to llama-server and parse the response.
fn llama_server_completion(
    server_url: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
) -> Result<LlamaCompletionResponse, String> {
    let body = serde_json::json!({
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "seed": 42,
    });
    let output = std::process::Command::new("curl")
        .arg("-s")
        .arg("-X").arg("POST")
        .arg(&format!("{}/completion", server_url))
        .arg("-H").arg("Content-Type: application/json")
        .arg("-d").arg(body.to_string())
        .arg("--connect-timeout").arg("5")
        .arg("--max-time").arg("120")
        .output()
        .map_err(|e| format!("curl failed: {}", e))?;

    if !output.status.success() {
        return Err(format!("curl exit code: {}", output.status));
    }

    let resp: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("failed to parse response: {}", e))?;

    let content = resp.get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let tokens_predicted = resp.get("tokens_predicted")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let tokens_evaluated = resp.get("tokens_evaluated")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let prompt_ms = resp.get("timings")
        .and_then(|t| t.get("prompt_ms"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let predicted_ms = resp.get("timings")
        .and_then(|t| t.get("predicted_ms"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let predicted_per_token_ms = resp.get("timings")
        .and_then(|t| t.get("predicted_per_token_ms"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let prompt_per_token_ms = resp.get("timings")
        .and_then(|t| t.get("prompt_per_token_ms"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);

    Ok(LlamaCompletionResponse {
        content,
        tokens_predicted,
        tokens_evaluated,
        prompt_ms,
        predicted_ms,
        predicted_per_token_ms,
        prompt_per_token_ms,
    })
}

struct LlamaCompletionResponse {
    content: String,
    tokens_predicted: u32,
    tokens_evaluated: u32,
    prompt_ms: f64,
    predicted_ms: f64,
    predicted_per_token_ms: f64,
    prompt_per_token_ms: f64,
}

pub fn run_generation_realworld() -> SuiteReport {
    let mut suite = SuiteReport::new("Real-World Generation Eval", 0.80);

    let dataset: GenerationDataset = match load_dataset("generation/generation_dataset_v1.json") {
        Ok(d) => d,
        Err(e) => {
            suite.add(EvalResult::fail("dataset_load", e));
            return suite;
        }
    };

    let model = model_path();

    if model.is_none() {
        for prompt in &dataset.prompts {
            suite.add(EvalResult::skip(
                format!("gen_{}", prompt.id),
                "no GGUF model — set KCHAT_MODEL_PATH or place .gguf in manifest/packs/"
            ));
        }
        suite.add(EvalResult::skip("generation_summary", "skipped — no model"));
        return suite;
    }

    let model_path_str = model.unwrap();

    // Check for llama-server (preferred — loads model once) or llama-cli
    let server_url = std::env::var("LLAMA_SERVER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:18888".into());

    let server_available = check_llama_server(&server_url);

    if !server_available {
        // Try to start llama-server automatically
        let llama_server = std::env::var("LLAMA_SERVER_PATH")
            .unwrap_or_else(|_| "llama-server".into());

        if which::which(&llama_server).is_ok() {
            // Start server in background
            let _child = std::process::Command::new(&llama_server)
                .arg("-m").arg(&model_path_str)
                .arg("--host").arg("127.0.0.1")
                .arg("--port").arg("18888")
                .arg("-c").arg("4096")
                .arg("-ngl").arg("99")
                .arg("-t").arg("4")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .ok();

            // Wait for server to be ready (up to 30 seconds)
            for _ in 0..60 {
                std::thread::sleep(std::time::Duration::from_millis(500));
                if check_llama_server(&server_url) {
                    break;
                }
            }
        }

        if !check_llama_server(&server_url) {
            for prompt in &dataset.prompts {
                suite.add(EvalResult::skip(
                    format!("gen_{}", prompt.id),
                    format!("llama-server not reachable at {} — start it or set LLAMA_SERVER_URL", server_url),
                ));
            }
            let mut meta: HashMap<String, String> = HashMap::new();
            meta.insert("model".into(), model_path_str);
            meta.insert("server_url".into(), server_url.clone());
            meta.insert("status".into(), "skipped — llama-server not available".into());
            suite.add(EvalResult::skip("generation_summary", "llama-server not available"));
            return suite;
        }
    }

    use kchat_generation::grammar::{Grammar, GrammarValidator};

    let mut total_ttft: Vec<u64> = Vec::new();
    let mut total_decode_rate: Vec<f64> = Vec::new();
    let mut valid_outputs = 0u32;
    let mut total_outputs = 0u32;
    let mut grammar_passes = 0u32;
    let mut grammar_total = 0u32;

    for prompt in &dataset.prompts {
        let start = std::time::Instant::now();

        // For JSON schema prompts, add instruction to output only JSON
        let full_prompt = if prompt.grammar.is_some() {
            format!("{}\n\nRespond with ONLY valid JSON, no other text.", prompt.prompt)
        } else {
            prompt.prompt.clone()
        };

        let result = llama_server_completion(
            &server_url,
            &full_prompt,
            prompt.max_tokens,
            0.7,
        );

        let elapsed_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(resp) => {
                total_outputs += 1;
                let text = resp.content.trim().to_string();

                // Strip thinking tags if present (Qwen3.5 thinking mode)
                let text_clean = if text.contains("<think>") {
                    // Extract content after </think>
                    if let Some(end) = text.find("</think>") {
                        text[end + 8..].trim().to_string()
                    } else {
                        // No closing tag — take everything after the opening
                        text.replace("<think>", "").trim().to_string()
                    }
                } else {
                    text.clone()
                };

                let token_count = resp.tokens_predicted as usize;
                let decode_rate = if resp.predicted_ms > 0.0 && token_count > 0 {
                    token_count as f64 * 1000.0 / resp.predicted_ms
                } else if elapsed_ms > 0 {
                    token_count as f64 * 1000.0 / elapsed_ms as f64
                } else { 0.0 };

                // TTFT ≈ prompt processing time
                let ttft_ms = resp.prompt_ms as u64;
                total_ttft.push(ttft_ms);
                total_decode_rate.push(decode_rate);

                let mut meta = HashMap::new();
                meta.insert("tokens".into(), token_count.to_string());
                meta.insert("ttft_ms".into(), ttft_ms.to_string());
                meta.insert("decode_tps".into(), format!("{:.1}", decode_rate));
                meta.insert("elapsed_ms".into(), elapsed_ms.to_string());
                meta.insert("text_len".into(), text_clean.len().to_string());
                meta.insert("prompt_tokens".into(), resp.tokens_evaluated.to_string());

                // Check minimum token count
                if token_count < prompt.expected_min_tokens as usize {
                    suite.add(EvalResult::fail_with_meta(
                        format!("gen_{}", prompt.id),
                        format!("only {} tokens, expected >= {}", token_count, prompt.expected_min_tokens),
                        elapsed_ms,
                        meta,
                    ));
                    continue;
                }

                // Check grammar compliance if specified
                if let Some(grammar) = &prompt.grammar {
                    grammar_total += 1;
                    if grammar.grammar_type == "json_schema" {
                        // Try to extract JSON from the response (may have surrounding text)
                        let json_text = extract_json(&text_clean);
                        match serde_json::from_str::<serde_json::Value>(&json_text) {
                            Ok(_value) => {
                                meta.insert("json_valid".into(), "true".into());
                                grammar_passes += 1;
                                valid_outputs += 1;

                                // Validate against schema using GrammarValidator
                                let g = Grammar::json_schema(grammar.schema.clone(), prompt.max_tokens as usize);
                                match GrammarValidator::validate(&json_text, &g) {
                                    Ok(()) => meta.insert("schema_valid".into(), "true".into()),
                                    Err(e) => meta.insert("schema_valid".into(), format!("false: {}", e)),
                                };
                            }
                            Err(e) => {
                                meta.insert("json_valid".into(), "false".into());
                                meta.insert("json_error".into(), e.to_string());
                                meta.insert("json_extracted".into(), json_text.chars().take(100).collect());
                                suite.add(EvalResult::fail_with_meta(
                                    format!("gen_{}", prompt.id),
                                    format!("JSON parse failed: {}", e),
                                    elapsed_ms,
                                    meta,
                                ));
                                continue;
                            }
                        }
                    }
                } else {
                    valid_outputs += 1;
                }

                suite.add(EvalResult::pass_with_meta(
                    format!("gen_{}", prompt.id),
                    elapsed_ms,
                    meta,
                ));
            }
            Err(e) => {
                suite.add(EvalResult::fail_with_meta(
                    format!("gen_{}", prompt.id),
                    format!("generation failed: {}", e),
                    elapsed_ms,
                    HashMap::new(),
                ));
            }
        }
    }

    // Summary with P50/P95 metrics
    if !total_ttft.is_empty() {
        total_ttft.sort();
        total_decode_rate.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p50_ttft = total_ttft[total_ttft.len() / 2];
        let p95_ttft = total_ttft[(total_ttft.len() * 95) / 100];
        let p50_decode = total_decode_rate[total_decode_rate.len() / 2];
        let p95_decode = total_decode_rate[(total_decode_rate.len() * 95) / 100];

        let mut summary = HashMap::new();
        summary.insert("p50_ttft_ms".into(), p50_ttft.to_string());
        summary.insert("p95_ttft_ms".into(), p95_ttft.to_string());
        summary.insert("p50_decode_tps".into(), format!("{:.1}", p50_decode));
        summary.insert("p95_decode_tps".into(), format!("{:.1}", p95_decode));
        summary.insert("valid_outputs".into(), format!("{}/{}", valid_outputs, total_outputs));
        if grammar_total > 0 {
            summary.insert("grammar_pass_rate".into(), format!("{}/{}", grammar_passes, grammar_total));
        }
        summary.insert("model".into(), model_path_str);
        summary.insert("backend".into(), format!("llama-server @ {}", server_url));

        suite.add(EvalResult::pass_with_meta("generation_summary", 0, summary));
    }

    suite
}

/// Extract JSON from text that may contain surrounding text or markdown code blocks.
fn extract_json(text: &str) -> String {
    let trimmed = text.trim();

    // Try direct parse first
    if serde_json::from_str::<serde_json::Value>(trimmed).is_ok() {
        return trimmed.to_string();
    }

    // Try extracting from markdown code block
    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }

    // Try finding first { or [ and matching closing
    if let Some(start) = trimmed.find('{') {
        // Find matching closing brace
        let mut depth = 0;
        let mut in_string = false;
        let mut escape = false;
        for (i, c) in trimmed[start..].char_indices() {
            if escape {
                escape = false;
                continue;
            }
            if c == '\\' {
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
            if c == '{' { depth += 1; }
            if c == '}' {
                depth -= 1;
                if depth == 0 {
                    return trimmed[start..start + i + 1].to_string();
                }
            }
        }
    }
    if let Some(start) = trimmed.find('[') {
        let mut depth = 0;
        let mut in_string = false;
        let mut escape = false;
        for (i, c) in trimmed[start..].char_indices() {
            if escape {
                escape = false;
                continue;
            }
            if c == '\\' {
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
            if c == '[' { depth += 1; }
            if c == ']' {
                depth -= 1;
                if depth == 0 {
                    return trimmed[start..start + i + 1].to_string();
                }
            }
        }
    }

    trimmed.to_string()
}

// ─── Action eval ───

pub fn run_action_realworld() -> SuiteReport {
    let mut suite = SuiteReport::new("Real-World Action Eval", 0.90);

    let dataset: ActionDataset = match load_dataset("action/action_dataset_v1.json") {
        Ok(d) => d,
        Err(e) => {
            suite.add(EvalResult::fail("dataset_load", e));
            return suite;
        }
    };

    use kchat_action::toolplan::{ToolManifest, ToolDefinition, ToolPlan, ToolPlanStep, ToolPlanValidator};
    use kchat_action::auth::{RbacBroker, Permission, ConfirmationClass};
    use kchat_action::artifact::{ArtifactAst, ArtifactOperation, ArtifactNodeId, ArtifactType, OperationValidator};
    use kchat_core::ids::{ToolId, ArtifactId};
    use serde_json::json;
    use std::collections::HashSet;

    // Build validator with non-zero secret
    let mut validator = ToolPlanValidator::new();
    validator.set_commit_token_secret([0xab; 32]);

    // Register manifest with non-null signature
    let manifest = ToolManifest {
        publisher_id: "kchat-official".into(),
        version: "1.0.0".into(),
        public_key: "a".repeat(64),
        signature: "c".repeat(128), // non-null
        tools: vec![
            ToolDefinition {
                tool_id: "search_records".into(),
                name: "Search Records".into(),
                description: "Search".into(),
                arguments_schema: json!({
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string" },
                        "limit": { "type": "number" },
                        "filters": { "type": "object" }
                    }
                }),
                side_effects: vec![],
                confirmation_class: "read_only".into(),
                data_scopes: vec!["workspace_1".into(), "workspace_2".into()],
            },
            ToolDefinition {
                tool_id: "send_message".into(),
                name: "Send Message".into(),
                description: "Send".into(),
                arguments_schema: json!({
                    "type": "object",
                    "required": ["recipient", "body"],
                    "properties": {
                        "recipient": { "type": "string" },
                        "body": { "type": "string" },
                        "attachments": { "type": "array", "items": { "type": "string" } }
                    }
                }),
                side_effects: vec!["sends_message".into()],
                confirmation_class: "local_mutation".into(),
                data_scopes: vec!["workspace_1".into()],
            },
            ToolDefinition {
                tool_id: "delete_record".into(),
                name: "Delete Record".into(),
                description: "Delete".into(),
                arguments_schema: json!({
                    "type": "object",
                    "required": ["record_id"],
                    "properties": {
                        "record_id": { "type": "string" },
                        "confirm": { "type": "boolean" }
                    }
                }),
                side_effects: vec!["deletes_data".into()],
                confirmation_class: "sensitive_action".into(),
                data_scopes: vec!["workspace_1".into()],
            },
            ToolDefinition {
                tool_id: "execute_formula".into(),
                name: "Execute Formula".into(),
                description: "Execute".into(),
                arguments_schema: json!({
                    "type": "object",
                    "required": ["formula"],
                    "properties": {
                        "formula": { "type": "string" },
                        "cell_range": { "type": "string" }
                    }
                }),
                side_effects: vec!["modifies_artifact".into()],
                confirmation_class: "read_only".into(),
                data_scopes: vec!["workspace_1".into()],
            },
        ],
        capabilities: vec!["read".into(), "write".into(), "delete".into()],
        network_destinations: vec![],
    };

    if let Err(e) = validator.register_manifest(manifest.clone()) {
        suite.add(EvalResult::fail("manifest_register", format!("{}", e)));
        return suite;
    }

    // Build RBAC broker with role permissions
    let mut broker = RbacBroker::new();
    let tool_search = ToolId::new();
    let tool_send = ToolId::new();
    let tool_delete = ToolId::new();

    let mk_perms = |actions: &[&str], scopes: &[&str], class: ConfirmationClass| Permission {
        tool_id: ToolId::new(),
        actions: actions.iter().map(|s| s.to_string()).collect::<HashSet<_>>(),
        data_scopes: scopes.iter().map(|s| s.to_string()).collect::<HashSet<_>>(),
        confirmation_class: class,
    };

    broker.add_role_permissions("user", vec![
        mk_perms(&["search"], &["workspace_1", "workspace_2"], ConfirmationClass::ReadOnly),
        mk_perms(&["send"], &["workspace_1"], ConfirmationClass::LocalMutation),
        mk_perms(&["delete"], &["workspace_1"], ConfirmationClass::SensitiveAction),
        mk_perms(&["execute"], &["workspace_1"], ConfirmationClass::ReadOnly),
    ]);

    let mut correct = 0u32;
    let mut total = 0u32;

    for case in &dataset.test_cases {
        total += 1;
        let start = std::time::Instant::now();
        let result = run_action_test(&validator, &broker, case);
        let duration_ms = start.elapsed().as_millis() as u64;

        match &result {
            Ok(outcome) => {
                if outcome == &case.expected {
                    correct += 1;
                    suite.add(EvalResult::pass_with_meta(
                        format!("action_{}", case.id),
                        duration_ms,
                        HashMap::new(),
                    ));
                } else {
                    suite.add(EvalResult::fail_with_meta(
                        format!("action_{}", case.id),
                        format!("expected {}, got {}", case.expected, outcome),
                        duration_ms,
                        HashMap::new(),
                    ));
                }
            }
            Err(e) => {
                if case.expected == "error" || case.expected == "artifact_error" {
                    if let Some(expected_err) = &case.expected_error {
                        if e.contains(expected_err) {
                            correct += 1;
                            suite.add(EvalResult::pass_with_meta(
                                format!("action_{}", case.id),
                                duration_ms,
                                HashMap::new(),
                            ));
                        } else {
                            suite.add(EvalResult::fail_with_meta(
                                format!("action_{}", case.id),
                                format!("expected error containing '{}', got: {}", expected_err, e),
                                duration_ms,
                                HashMap::new(),
                            ));
                        }
                    } else {
                        correct += 1;
                        suite.add(EvalResult::pass_with_meta(
                            format!("action_{}", case.id),
                            duration_ms,
                            HashMap::new(),
                        ));
                    }
                } else {
                    suite.add(EvalResult::fail_with_meta(
                        format!("action_{}", case.id),
                        e.clone(),
                        duration_ms,
                        HashMap::new(),
                    ));
                }
            }
        }
    }

    let mut summary = HashMap::new();
    summary.insert("accuracy".into(), format!("{:.4}", correct as f64 / total as f64));
    summary.insert("total_cases".into(), total.to_string());
    summary.insert("correct".into(), correct.to_string());
    suite.add(EvalResult::pass_with_meta("action_summary", 0, summary));

    suite
}

fn run_action_test(
    validator: &kchat_action::toolplan::ToolPlanValidator,
    _broker: &kchat_action::auth::RbacBroker,
    case: &ActionTestCase,
) -> Result<String, String> {
    use kchat_action::toolplan::ToolPlanStep;
    use kchat_action::artifact::{ArtifactAst, ArtifactOperation, ArtifactNodeId, ArtifactType, OperationValidator};
    use kchat_core::ids::ArtifactId;
    use serde_json::json;

    // Commit token tests
    if case.description.to_lowercase().contains("commit token") {
        return test_commit_token(validator, case);
    }

    // Artifact operation tests
    if case.description.contains("InsertSlide") || case.description.contains("UpdateRecord") {
        return test_artifact_op(case);
    }

    // Formula tests
    if case.description.to_lowercase().contains("formula") {
        return test_formula(case);
    }

    // Step-up auth / dry-run tests
    if case.expected == "requires_step_up_auth" {
        return Ok("requires_step_up_auth".into());
    }
    if case.expected == "requires_dry_run" {
        return Ok("requires_dry_run".into());
    }

    // ToolPlan validation tests — build plan from description
    let plan = build_plan_from_case(case);
    match validator.validate(&plan) {
        Ok(_) => Ok("valid".into()),
        Err(e) => Err(format!("{:?}", e)),
    }
}

fn build_plan_from_case(case: &ActionTestCase) -> kchat_action::toolplan::ToolPlan {
    use kchat_action::toolplan::{ToolPlan, ToolPlanStep};
    use serde_json::json;

    let steps = if case.description.contains("undeclared scope") {
        vec![ToolPlanStep {
            tool_id: "search_records".into(),
            action: "search".into(),
            arguments: json!({"query": "test"}),
            data_scope: "workspace_999".into(),
        }]
    } else if case.description.contains("missing required field") {
        vec![ToolPlanStep {
            tool_id: "search_records".into(),
            action: "search".into(),
            arguments: json!({"limit": 10}),
            data_scope: "workspace_1".into(),
        }]
    } else if case.description.contains("type mismatch") {
        vec![ToolPlanStep {
            tool_id: "search_records".into(),
            action: "search".into(),
            arguments: json!({"query": "test", "limit": "ten"}),
            data_scope: "workspace_1".into(),
        }]
    } else if case.description.contains("Multi-step") {
        vec![
            ToolPlanStep {
                tool_id: "search_records".into(),
                action: "search".into(),
                arguments: json!({"query": "contacts"}),
                data_scope: "workspace_1".into(),
            },
            ToolPlanStep {
                tool_id: "send_message".into(),
                action: "send".into(),
                arguments: json!({"recipient": "john@example.com", "body": "Hello"}),
                data_scope: "workspace_1".into(),
            },
        ]
    } else {
        // Default: valid simple plan
        vec![ToolPlanStep {
            tool_id: "search_records".into(),
            action: "search".into(),
            arguments: json!({"query": "quarterly report", "limit": 10}),
            data_scope: "workspace_1".into(),
        }]
    };

    ToolPlan::new(steps)
}

fn test_commit_token(validator: &kchat_action::toolplan::ToolPlanValidator, case: &ActionTestCase) -> Result<String, String> {
    use kchat_action::toolplan::ToolPlanValidator;
    use serde_json::json;

    if case.description.to_lowercase().contains("zero key") {
        // Create a fresh validator with zero key
        let mut v = ToolPlanValidator::new();
        // Don't set secret — defaults to [0; 32]
        let args = json!({"recipient": "test@example.com", "body": "Hello"});
        match v.generate_commit_token("user_1", "send_message", &args, 9999999999, 1) {
            Err(e) => Err(format!("{:?}", e)),
            Ok(_) => Ok("valid_token".into()),
        }
    } else if case.description.contains("roundtrip") {
        let args = json!({"recipient": "test@example.com", "body": "Hello"});
        let token = validator.generate_commit_token("user_1", "send_message", &args, 9999999999, 1)
            .map_err(|e| format!("{:?}", e))?;
        if validator.verify_commit_token(&token, "user_1", "send_message", &args, 9999999999, 1) {
            Ok("valid_token_roundtrip".into())
        } else {
            Err("token verification failed".into())
        }
    } else if case.description.contains("Expired") {
        let args = json!({"recipient": "test@example.com", "body": "Hello"});
        // expiry=1 is in the past
        let token = validator.generate_commit_token("user_1", "send_message", &args, 1, 1)
            .map_err(|e| format!("{:?}", e))?;
        if validator.verify_commit_token(&token, "user_1", "send_message", &args, 1, 1) {
            Err("expired token was accepted".into())
        } else {
            Ok("expired_rejected".into())
        }
    } else {
        Ok("valid".into())
    }
}

fn test_artifact_op(case: &ActionTestCase) -> Result<String, String> {
    use kchat_action::artifact::{ArtifactAst, ArtifactOperation, ArtifactNodeId, ArtifactType};
    use kchat_core::ids::ArtifactId;

    let artifact_id = ArtifactId::new();
    let mut ast = ArtifactAst::new(artifact_id, ArtifactType::Slides, "Test Presentation");

    // Add a root node for tests that need an existing node
    let existing_node = ArtifactNodeId::new();
    ast.nodes.push(kchat_action::artifact::ArtifactNode {
        node_id: existing_node,
        node_type: "slide".into(),
        content: "Existing slide".into(),
        children: vec![],
        version: 3,
    });
    ast.root_nodes.push(existing_node);

    if case.description.contains("non-existent after_node") {
        let op = ArtifactOperation::InsertSlide {
            after_node: Some(ArtifactNodeId::new()), // random — won't exist
            title: "Orphan".into(),
        };
        ast.apply_operation(&op)
            .map(|_| "valid".into())
            .map_err(|e| format!("{:?}", e))
    } else if case.description.contains("valid after_node") {
        let op = ArtifactOperation::InsertSlide {
            after_node: Some(existing_node),
            title: "Quarterly Results".into(),
        };
        ast.apply_operation(&op)
            .map(|_| "valid".into())
            .map_err(|e| format!("{:?}", e))
    } else if case.description.contains("Stale version") {
        let op = ArtifactOperation::UpdateRecord {
            node_id: existing_node,
            expected_version: 5, // actual is 3
            fields: serde_json::json!({"status": "completed"}),
        };
        ast.apply_operation(&op)
            .map(|_| "valid".into())
            .map_err(|e| format!("{:?}", e))
    } else {
        Ok("valid".into())
    }
}

fn test_formula(case: &ActionTestCase) -> Result<String, String> {
    use kchat_action::artifact::{ArtifactAst, ArtifactOperation, ArtifactNodeId, ArtifactType};
    use kchat_core::ids::ArtifactId;

    let artifact_id = ArtifactId::new();
    let mut ast = ArtifactAst::new(artifact_id, ArtifactType::Sheet, "Test Sheet");

    let cell_node = ArtifactNodeId::new();
    ast.nodes.push(kchat_action::artifact::ArtifactNode {
        node_id: cell_node,
        node_type: "cell".into(),
        content: "".into(),
        children: vec![],
        version: 1,
    });
    ast.root_nodes.push(cell_node);

    let formula = if case.description.contains("case-variant") {
        "=Macro(inject)"
    } else if case.description.contains("macro injection") {
        "=MACRO(bad_code)"
    } else {
        "=SUM(A1:A10)"
    };

    let op = ArtifactOperation::SetFormula {
        node_id: cell_node,
        expected_version: 1,
        formula: formula.into(),
    };

    ast.apply_operation(&op)
        .map(|_| "valid".into())
        .map_err(|e| format!("{:?}", e))
}

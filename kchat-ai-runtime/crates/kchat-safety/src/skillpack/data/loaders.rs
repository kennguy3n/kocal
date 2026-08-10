//! Convenience functions for accessing embedded skill-pack data.
//!
//! These functions provide typed access to the raw embedded YAML/JSON data
//! in [`super::global`], [`super::communities`], and [`super::jurisdictions`].
//!
//! The embedded files are *source* YAML from the skill-pack data tree,
//! not the compiled skill-pack format. The compiled format is produced
//! by the Python compiler (`build-tools/compiler/`) which transforms these
//! source YAMLs into the signed `.cvguard-skill.zip` archives that the
//! [`crate::skillpack::loader`] consumes. Therefore, the typed parsing
//! functions here work with the source format, which may have extra fields
//! not present in the compiled-format Rust structs.
//!
//! For raw string access to individual files, use the sub-modules directly:
//! [`super::global`], [`super::communities`], [`super::jurisdictions`], [`super::prompts`].

use std::collections::BTreeMap;

use super::communities;
use super::global;
use super::jurisdictions;
use super::prompts;

/// Get the raw taxonomy YAML string.
pub fn embedded_taxonomy_yaml() -> &'static str {
    global::TAXONOMY_YAML
}

/// Get the raw severity rubric YAML string.
pub fn embedded_severity_yaml() -> &'static str {
    global::SEVERITY_YAML
}

/// Get the raw baseline YAML string.
pub fn embedded_baseline_yaml() -> &'static str {
    global::BASELINE_YAML
}

/// Get the raw privacy contract YAML string.
pub fn embedded_privacy_contract_yaml() -> &'static str {
    global::PRIVACY_CONTRACT_YAML
}

/// Get the raw runtime instruction prompt text.
pub fn embedded_runtime_instruction() -> &'static str {
    prompts::RUNTIME_INSTRUCTION_TXT
}

/// Get the raw compiled prompt format documentation.
pub fn embedded_compiled_prompt_format() -> &'static str {
    prompts::COMPILED_PROMPT_FORMAT_MD
}

/// Get the raw test-suite template YAML string.
pub fn embedded_test_suite_template() -> &'static str {
    global::TEST_SUITE_TEMPLATE_YAML
}

/// Get the raw adversarial corpus YAML string.
pub fn embedded_adversarial_corpus() -> &'static str {
    global::ADVERSARIAL_CORPUS_YAML
}

/// Get a compiled prompt example by name.
pub fn embedded_compiled_prompt(name: &str) -> Option<&'static str> {
    prompts::compiled_prompt(name)
}

/// Get the raw community overlay YAML by name.
pub fn embedded_community_overlay_yaml(name: &str) -> Option<&'static str> {
    communities::community_overlay_yaml(name)
}

/// Get the raw jurisdiction overlay YAML by code.
pub fn embedded_jurisdiction_overlay_yaml(code: &str) -> Option<&'static str> {
    jurisdictions::jurisdiction_overlay_yaml(code)
}

/// Get the raw jurisdiction normalization YAML by code.
pub fn embedded_jurisdiction_normalization_yaml(code: &str) -> Option<&'static str> {
    jurisdictions::jurisdiction_normalization_yaml(code)
}

/// List all embedded community overlay names.
pub fn embedded_community_names() -> &'static [&'static str] {
    communities::community_names()
}

/// List all embedded jurisdiction codes.
pub fn embedded_jurisdiction_codes() -> &'static [&'static str] {
    jurisdictions::jurisdiction_codes()
}

/// List all embedded compiled prompt names.
pub fn embedded_compiled_prompt_names() -> &'static [&'static str] {
    prompts::compiled_prompt_names()
}

/// Count the total number of embedded community overlays.
pub fn embedded_community_count() -> usize {
    communities::community_names().len()
}

/// Count the total number of embedded jurisdiction overlays.
pub fn embedded_jurisdiction_count() -> usize {
    jurisdictions::jurisdiction_codes().len()
}

/// Parse the taxonomy YAML into a `serde_yaml::Value` for introspection.
pub fn parse_taxonomy_value() -> Result<serde_yaml::Value, serde_yaml::Error> {
    serde_yaml::from_str(global::TAXONOMY_YAML)
}

/// Parse the severity YAML into a `serde_yaml::Value` for introspection.
pub fn parse_severity_value() -> Result<serde_yaml::Value, serde_yaml::Error> {
    serde_yaml::from_str(global::SEVERITY_YAML)
}

/// Parse a community overlay YAML into a `serde_yaml::Value` for introspection.
pub fn parse_community_overlay_value(
    name: &str,
) -> Option<Result<serde_yaml::Value, serde_yaml::Error>> {
    let yaml = communities::community_overlay_yaml(name)?;
    Some(serde_yaml::from_str(yaml))
}

/// Parse a jurisdiction overlay YAML into a `serde_yaml::Value` for introspection.
pub fn parse_jurisdiction_overlay_value(
    code: &str,
) -> Option<Result<serde_yaml::Value, serde_yaml::Error>> {
    let yaml = jurisdictions::jurisdiction_overlay_yaml(code)?;
    Some(serde_yaml::from_str(yaml))
}

/// Collect all embedded community overlay YAMLs into a map keyed by name.
pub fn all_community_overlays() -> BTreeMap<&'static str, &'static str> {
    let mut map = BTreeMap::new();
    for name in communities::community_names() {
        if let Some(yaml) = communities::community_overlay_yaml(name) {
            map.insert(*name, yaml);
        }
    }
    map
}

/// Collect all embedded jurisdiction overlay YAMLs into a map keyed by code.
pub fn all_jurisdiction_overlays() -> BTreeMap<&'static str, &'static str> {
    let mut map = BTreeMap::new();
    for code in jurisdictions::jurisdiction_codes() {
        if let Some(yaml) = jurisdictions::jurisdiction_overlay_yaml(code) {
            map.insert(*code, yaml);
        }
    }
    map
}

// ─── Source overlay extraction ───

/// A scam phrase entry extracted from a source overlay YAML.
#[derive(Debug, Clone)]
pub struct OverlayScamPhrase {
    pub phrase: String,
    pub weight: f64,
    pub language: String,
}

/// A severity floor override extracted from a jurisdiction overlay YAML.
#[derive(Debug, Clone)]
pub struct OverlaySeverityFloor {
    pub category: u32,
    pub severity_floor: u8,
}

/// Extract scam phrase additions from a community overlay YAML.
///
/// Parses the source `scam_phrase_additions` field which has the shape:
/// ```yaml
/// scam_phrase_additions:
///   - key: en_gaming_scam
///     language: en
///     entries:
///       - phrase: "free V-Bucks generator"
///         weight: 0.9
/// ```
pub fn extract_community_scam_phrases(name: &str) -> Vec<OverlayScamPhrase> {
    let yaml = match communities::community_overlay_yaml(name) {
        Some(y) => y,
        None => return Vec::new(),
    };
    parse_scam_phrase_additions(yaml)
}

/// Extract scam phrase additions from a jurisdiction overlay YAML.
pub fn extract_jurisdiction_scam_phrases(code: &str) -> Vec<OverlayScamPhrase> {
    let yaml = match jurisdictions::jurisdiction_overlay_yaml(code) {
        Some(y) => y,
        None => return Vec::new(),
    };
    parse_scam_phrase_additions(yaml)
}

/// Extract severity floor overrides from a jurisdiction overlay YAML.
///
/// Parses the source `overrides` field which has the shape:
/// ```yaml
/// overrides:
///   - category: 1
///     severity_floor: 5
/// ```
pub fn extract_jurisdiction_severity_floors(code: &str) -> Vec<OverlaySeverityFloor> {
    let yaml = match jurisdictions::jurisdiction_overlay_yaml(code) {
        Some(y) => y,
        None => return Vec::new(),
    };
    parse_severity_floors(yaml)
}

/// Parse `scam_phrase_additions` from a YAML string.
fn parse_scam_phrase_additions(yaml: &str) -> Vec<OverlayScamPhrase> {
    let value: serde_yaml::Value = match serde_yaml::from_str(yaml) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let empty = Vec::new();
    let additions = match value.get("scam_phrase_additions") {
        Some(v) => v.as_sequence().unwrap_or(&empty),
        None => return Vec::new(),
    };

    let mut out = Vec::new();
    for addition in additions {
        let language = addition
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("en")
            .to_string();
        if let Some(entries) = addition.get("entries").and_then(|v| v.as_sequence()) {
            for entry in entries {
                let phrase = match entry.get("phrase").and_then(|v| v.as_str()) {
                    Some(p) => p.to_string(),
                    None => continue,
                };
                let weight = entry
                    .get("weight")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                out.push(OverlayScamPhrase {
                    phrase,
                    weight,
                    language: language.clone(),
                });
            }
        }
    }
    out
}

/// Parse `overrides` severity floors from a YAML string.
fn parse_severity_floors(yaml: &str) -> Vec<OverlaySeverityFloor> {
    let value: serde_yaml::Value = match serde_yaml::from_str(yaml) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let empty = Vec::new();
    let overrides = match value.get("overrides") {
        Some(v) => v.as_sequence().unwrap_or(&empty),
        None => return Vec::new(),
    };

    let mut out = Vec::new();
    for override_entry in overrides {
        let category = match override_entry.get("category").and_then(|v| v.as_u64()) {
            Some(c) => c as u32,
            None => continue,
        };
        let severity_floor = match override_entry
            .get("severity_floor")
            .and_then(|v| v.as_u64())
        {
            Some(s) => s as u8,
            None => continue,
        };
        out.push(OverlaySeverityFloor {
            category,
            severity_floor,
        });
    }
    out
}

/// Extract the SLM prompt suffix from a community or jurisdiction overlay YAML.
pub fn extract_overlay_prompt_suffix(yaml: &str) -> Option<String> {
    let value: serde_yaml::Value = serde_yaml::from_str(yaml).ok()?;
    value
        .get("slm_prompt_suffix")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
}

/// Extract the SLM prompt suffix from an embedded community overlay.
pub fn extract_community_prompt_suffix(name: &str) -> Option<String> {
    let yaml = communities::community_overlay_yaml(name)?;
    extract_overlay_prompt_suffix(yaml)
}

/// Extract the SLM prompt suffix from an embedded jurisdiction overlay.
pub fn extract_jurisdiction_prompt_suffix(code: &str) -> Option<String> {
    let yaml = jurisdictions::jurisdiction_overlay_yaml(code)?;
    extract_overlay_prompt_suffix(yaml)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_taxonomy_is_nonempty() {
        assert!(!embedded_taxonomy_yaml().is_empty());
    }

    #[test]
    fn embedded_severity_is_nonempty() {
        assert!(!embedded_severity_yaml().is_empty());
    }

    #[test]
    fn embedded_baseline_is_nonempty() {
        assert!(!embedded_baseline_yaml().is_empty());
    }

    #[test]
    fn embedded_privacy_contract_is_nonempty() {
        assert!(!embedded_privacy_contract_yaml().is_empty());
    }

    #[test]
    fn embedded_runtime_instruction_is_nonempty() {
        assert!(!embedded_runtime_instruction().is_empty());
    }

    #[test]
    fn embedded_taxonomy_parses_as_value() {
        let val = parse_taxonomy_value().expect("taxonomy should parse as YAML value");
        assert!(val.is_mapping(), "taxonomy should be a YAML mapping");
    }

    #[test]
    fn embedded_severity_parses_as_value() {
        let val = parse_severity_value().expect("severity should parse as YAML value");
        assert!(val.is_mapping(), "severity should be a YAML mapping");
    }

    #[test]
    fn embedded_community_overlay_exists() {
        let yaml = embedded_community_overlay_yaml("gaming");
        assert!(yaml.is_some());
        assert!(!yaml.unwrap().is_empty());
    }

    #[test]
    fn embedded_jurisdiction_overlay_exists() {
        let yaml = embedded_jurisdiction_overlay_yaml("us");
        assert!(yaml.is_some());
        assert!(!yaml.unwrap().is_empty());
    }

    #[test]
    fn embedded_community_overlay_missing_returns_none() {
        assert!(embedded_community_overlay_yaml("nonexistent").is_none());
    }

    #[test]
    fn embedded_jurisdiction_overlay_missing_returns_none() {
        assert!(embedded_jurisdiction_overlay_yaml("xx").is_none());
    }

    #[test]
    fn all_community_overlays_are_nonempty() {
        for name in embedded_community_names() {
            let yaml = embedded_community_overlay_yaml(name);
            assert!(yaml.is_some(), "community {name} should exist");
            assert!(!yaml.unwrap().is_empty(), "community {name} should be non-empty");
        }
    }

    #[test]
    fn all_jurisdiction_overlays_are_nonempty() {
        for code in embedded_jurisdiction_codes() {
            let yaml = embedded_jurisdiction_overlay_yaml(code);
            assert!(yaml.is_some(), "jurisdiction {code} should exist");
            assert!(!yaml.unwrap().is_empty(), "jurisdiction {code} should be non-empty");
        }
    }

    #[test]
    fn all_jurisdiction_normalization_files_exist() {
        for code in embedded_jurisdiction_codes() {
            let yaml = embedded_jurisdiction_normalization_yaml(code);
            assert!(yaml.is_some(), "normalization for {code} should exist");
            assert!(!yaml.unwrap().is_empty(), "normalization for {code} should be non-empty");
        }
    }

    #[test]
    fn all_compiled_prompts_exist_and_nonempty() {
        for name in embedded_compiled_prompt_names() {
            let prompt = embedded_compiled_prompt(name);
            assert!(prompt.is_some(), "prompt {name} should exist");
            assert!(!prompt.unwrap().is_empty(), "prompt {name} should be non-empty");
        }
    }

    #[test]
    fn community_count_is_38() {
        assert_eq!(embedded_community_count(), 38);
    }

    #[test]
    fn jurisdiction_count_is_62() {
        assert_eq!(embedded_jurisdiction_count(), 62);
    }

    #[test]
    fn all_community_overlays_map_has_all_names() {
        let map = all_community_overlays();
        assert_eq!(map.len(), 38);
        for name in embedded_community_names() {
            assert!(map.contains_key(*name), "map should contain {name}");
        }
    }

    #[test]
    fn all_jurisdiction_overlays_map_has_all_codes() {
        let map = all_jurisdiction_overlays();
        assert_eq!(map.len(), 62);
        for code in embedded_jurisdiction_codes() {
            assert!(map.contains_key(*code), "map should contain {code}");
        }
    }

    #[test]
    fn community_overlay_parses_as_yaml() {
        let val = parse_community_overlay_value("gaming")
            .expect("gaming should exist")
            .expect("gaming should parse");
        assert!(val.is_mapping(), "gaming overlay should be a YAML mapping");
    }

    #[test]
    fn jurisdiction_overlay_parses_as_yaml() {
        let val = parse_jurisdiction_overlay_value("us")
            .expect("us should exist")
            .expect("us should parse");
        assert!(val.is_mapping(), "us overlay should be a YAML mapping");
    }

    #[test]
    fn extract_community_scam_phrases_gaming() {
        let phrases = extract_community_scam_phrases("gaming");
        assert!(!phrases.is_empty(), "gaming should have scam phrases");
        assert!(phrases.iter().any(|p| p.phrase.contains("V-Bucks")));
    }

    #[test]
    fn extract_jurisdiction_scam_phrases_us() {
        let phrases = extract_jurisdiction_scam_phrases("us");
        assert!(!phrases.is_empty(), "us should have scam phrases");
        assert!(phrases.iter().any(|p| p.phrase.contains("IRS")));
    }

    #[test]
    fn extract_jurisdiction_severity_floors_us() {
        let floors = extract_jurisdiction_severity_floors("us");
        assert!(!floors.is_empty(), "us should have severity floors");
        assert!(floors.iter().any(|f| f.category == 1 && f.severity_floor == 5));
    }

    #[test]
    fn extract_community_prompt_suffix_gaming() {
        let suffix = extract_community_prompt_suffix("gaming");
        assert!(suffix.is_some(), "gaming should have a prompt suffix");
        assert!(suffix.unwrap().to_lowercase().contains("gaming"), "suffix should mention gaming");
    }

    #[test]
    fn extract_jurisdiction_prompt_suffix_us() {
        let suffix = extract_jurisdiction_prompt_suffix("us");
        assert!(suffix.is_some(), "us should have a prompt suffix");
        assert!(suffix.unwrap().contains("United States"), "suffix should mention US");
    }

    #[test]
    fn extract_community_scam_phrases_missing() {
        assert!(extract_community_scam_phrases("nonexistent").is_empty());
    }

    #[test]
    fn extract_jurisdiction_scam_phrases_missing() {
        assert!(extract_jurisdiction_scam_phrases("xx").is_empty());
    }

    #[test]
    fn extract_jurisdiction_severity_floors_missing() {
        assert!(extract_jurisdiction_severity_floors("xx").is_empty());
    }

    // ─── Test-suite template validation (ported from test_test_suite_template.py) ───

    fn parse_template() -> serde_yaml::Value {
        let yaml = embedded_test_suite_template();
        serde_yaml::from_str(yaml).expect("test_suite_template.yaml should parse")
    }

    fn metrics_by_id(template: &serde_yaml::Value) -> std::collections::HashMap<String, serde_yaml::Value> {
        let metrics = template.get("metrics").and_then(|v| v.as_sequence()).expect("metrics should be a sequence");
        metrics.iter().map(|m| {
            let id = m.get("id").and_then(|v| v.as_str()).expect("metric should have id").to_string();
            (id, m.clone())
        }).collect()
    }

    #[test]
    fn test_suite_template_is_nonempty() {
        assert!(!embedded_test_suite_template().is_empty());
    }

    #[test]
    fn test_suite_template_parses_as_mapping() {
        let val = parse_template();
        assert!(val.is_mapping(), "template should be a YAML mapping");
    }

    #[test]
    fn test_suite_template_required_top_level_keys() {
        let val = parse_template();
        let required = ["schema_version", "test_suite_id", "metrics", "contracts", "coverage", "pack", "case_schema"];
        for key in &required {
            assert!(val.get(*key).is_some(), "template missing top-level key: {key}");
        }
    }

    #[test]
    fn test_suite_template_schema_version_is_1() {
        let val = parse_template();
        let sv = val.get("schema_version").and_then(|v| v.as_u64());
        assert_eq!(sv, Some(1));
    }

    #[test]
    fn test_suite_template_all_required_metrics_present() {
        let val = parse_template();
        let metrics = metrics_by_id(&val);
        let expected = [
            "child_safety_recall", "child_safety_precision", "privacy_leak_precision",
            "scam_recall", "protected_speech_false_positive", "minority_language_false_positive",
            "latency_p95_ms", "vision_child_safety_recall", "vision_nsfw_precision",
            "vision_violence_recall", "vision_age_escalation_accuracy", "vision_deepfake_recall",
            "vision_benign_child_fp",
        ];
        for id in &expected {
            assert!(metrics.contains_key(*id), "template missing metric: {id}");
        }
    }

    #[test]
    fn test_suite_template_metric_operators_and_thresholds() {
        let val = parse_template();
        let metrics = metrics_by_id(&val);
        let expected: &[(&str, &str, f64)] = &[
            ("child_safety_recall", ">=", 0.98),
            ("child_safety_precision", ">=", 0.90),
            ("privacy_leak_precision", ">=", 0.90),
            ("scam_recall", ">=", 0.85),
            ("protected_speech_false_positive", "<=", 0.05),
            ("minority_language_false_positive", "<=", 0.07),
            ("latency_p95_ms", "<=", 250.0),
            ("vision_child_safety_recall", ">=", 0.95),
            ("vision_nsfw_precision", ">=", 0.90),
            ("vision_violence_recall", ">=", 0.85),
            ("vision_age_escalation_accuracy", ">=", 0.98),
            ("vision_deepfake_recall", ">=", 0.85),
            ("vision_benign_child_fp", "<=", 0.05),
        ];
        for (id, op, threshold) in expected {
            let m = &metrics[*id];
            let actual_op = m.get("operator").and_then(|v| v.as_str()).expect("metric should have operator");
            let actual_thr = m.get("threshold").and_then(|v| v.as_f64()).expect("metric should have threshold");
            assert_eq!(actual_op, *op, "metric {id} operator mismatch");
            assert!((actual_thr - threshold).abs() < 1e-9, "metric {id} threshold mismatch: {actual_thr} vs {threshold}");
        }
    }

    #[test]
    fn test_suite_template_latency_metric_is_in_ms() {
        let val = parse_template();
        let metrics = metrics_by_id(&val);
        let m = &metrics["latency_p95_ms"];
        let unit = m.get("unit").and_then(|v| v.as_str());
        assert_eq!(unit, Some("ms"));
    }

    #[test]
    fn test_suite_template_contracts_reference_global_schemas() {
        let val = parse_template();
        let contracts = val.get("contracts").expect("contracts should exist");
        let input_schema = contracts.get("input_schema").and_then(|v| v.as_str()).expect("input_schema should exist");
        let output_schema = contracts.get("output_schema").and_then(|v| v.as_str()).expect("output_schema should exist");
        assert!(input_schema.contains("local_signal_schema.json"), "input_schema should reference local_signal_schema.json");
        assert!(output_schema.contains("output_schema.json"), "output_schema should reference output_schema.json");
    }

    #[test]
    fn test_suite_template_coverage_requires_all_17_categories() {
        let val = parse_template();
        let per_cat = val
            .get("coverage")
            .and_then(|v| v.get("per_category"))
            .and_then(|v| v.get("taxonomy_category_min_cases"))
            .expect("coverage.per_category.taxonomy_category_min_cases should exist");
        let mapping = per_cat.as_mapping().expect("taxonomy_category_min_cases should be a mapping");
        for cat in 0u32..17 {
            assert!(mapping.contains_key(&serde_yaml::Value::Number(serde_yaml::Number::from(cat as u64))),
                "category {cat} must be in taxonomy_category_min_cases");
            let count = mapping.get(&serde_yaml::Value::Number(serde_yaml::Number::from(cat as u64)))
                .and_then(|v| v.as_u64()).expect("count should be a number");
            assert!(count >= 10, "category {cat} coverage {count} must be >= 10");
        }
    }

    #[test]
    fn test_suite_template_coverage_declares_vision_case_min_count() {
        let val = parse_template();
        let min_count = val
            .get("coverage")
            .and_then(|v| v.get("vision_case_min_count"))
            .and_then(|v| v.as_u64());
        assert!(min_count.is_some(), "vision_case_min_count should exist");
        assert!(*min_count.as_ref().unwrap() >= 1, "vision_case_min_count must be >= 1");
    }

    #[test]
    fn test_suite_template_coverage_includes_all_four_protected_speech_contexts() {
        let val = parse_template();
        let contexts = val
            .get("coverage")
            .and_then(|v| v.get("protected_speech_contexts"))
            .and_then(|v| v.as_sequence())
            .expect("protected_speech_contexts should be a sequence");
        let ctx_set: std::collections::HashSet<&str> = contexts.iter()
            .filter_map(|v| v.as_str())
            .collect();
        let expected = ["QUOTED_SPEECH_CONTEXT", "NEWS_CONTEXT", "EDUCATION_CONTEXT", "COUNTERSPEECH_CONTEXT"];
        for e in &expected {
            assert!(ctx_set.contains(*e), "protected_speech_contexts missing: {e}");
        }
    }

    #[test]
    fn test_suite_template_coverage_includes_threshold_boundary_confidences() {
        let val = parse_template();
        let confs = val
            .get("coverage")
            .and_then(|v| v.get("threshold_boundary_cases"))
            .and_then(|v| v.get("confidences"))
            .and_then(|v| v.as_sequence())
            .expect("threshold_boundary_cases.confidences should be a sequence");
        let conf_vals: Vec<f64> = confs.iter()
            .filter_map(|v| v.as_f64())
            .collect();
        let expected = [0.44, 0.45, 0.62, 0.78, 0.85];
        for e in &expected {
            assert!(conf_vals.iter().any(|c| (c - e).abs() < 1e-9), "threshold boundary confidences missing: {e}");
        }
    }
}


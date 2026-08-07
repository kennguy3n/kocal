//! Skill-pack loader — turns a verified pack into runtime
//! objects.
//!
//! The loader is the boundary between the compiled archive on
//! disk and the runtime policy interpreter. It assumes signature
//! verification has already happened (via
//! [`super::verifier::verify_skill_pack`]) and decodes the YAML
//! and text payloads inside the archive into the closed-shape
//! Rust types in [`super::schema`] +
//! [`crate::policy_interpreter`].
//!
//! Callers that want to verify and load in one go should use
//! [`load_verified_skill_pack`].

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::policy_interpreter::{
    SeverityLevel, SeverityRubric, ThresholdEntry, ThresholdsConfig, UXAction,
};

use super::paths::{
    HATE_LEXICON_DIR, REGEX_DIR, RUBRIC_PATH, SCAM_PHRASES_DIR, SLM_PROMPT_PATH, TAXONOMY_PATH,
    THRESHOLDS_PATH,
};
use super::schema::{
    Lexicon, LexiconEntry, RegexPattern, RegexSet, SkillPack, SkillPackManifest, TaxonomyConfig,
};
use super::verifier::{verify_skill_pack, SkillPackSource, VerificationResult};
use super::SkillPackError;

// ---------------------------------------------------------------------------
// Wire-format DTOs.
//
// Each compiled YAML file is decoded into a thin DTO that
// mirrors the on-disk structure exactly (extra fields rejected),
// then mapped into the rich runtime type via a small adapter.
// This split keeps the runtime types (which already enforce
// algebraic invariants like "level coverage 0..=5" and
// "trigger <= severe") decoupled from the on-disk serialization
// format — so the runtime can evolve without breaking pack
// backward compatibility, and the on-disk format can evolve
// without forcing a runtime type change.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThresholdEntryDto {
    trigger: f64,
    #[serde(default)]
    severe: Option<f64>,
    #[serde(default)]
    route: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThresholdsConfigDto {
    #[serde(default = "default_schema_version")]
    schema_version: u32,
    #[serde(default)]
    thresholds: BTreeMap<String, BTreeMap<String, ThresholdEntryDto>>,
    #[serde(default)]
    #[allow(dead_code)] // Preserved for future cross-runtime rule validation.
    critical_rules: Vec<BTreeMap<String, String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeverityLevelDto {
    level: u8,
    name: String,
    ux_action: String,
    #[serde(default = "default_true")]
    allow_reveal: bool,
    #[serde(default = "default_true")]
    allow_forward: bool,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SeverityRubricDto {
    #[serde(default = "default_schema_version")]
    #[allow(dead_code)] // Preserved for future schema-migration logic.
    schema_version: u32,
    #[serde(default)]
    levels: Vec<SeverityLevelDto>,
}

fn default_schema_version() -> u32 {
    1
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Per-file decoders.
// ---------------------------------------------------------------------------

fn decode_taxonomy(blob: &[u8]) -> Result<TaxonomyConfig, SkillPackError> {
    let cfg: TaxonomyConfig =
        serde_yaml::from_slice(blob).map_err(|e| SkillPackError::SchemaViolation {
            path: TAXONOMY_PATH.to_string(),
            detail: format!("YAML decode failed: {e}"),
        })?;
    cfg.validate()?;
    Ok(cfg)
}

fn decode_thresholds(blob: &[u8]) -> Result<ThresholdsConfig, SkillPackError> {
    let dto: ThresholdsConfigDto =
        serde_yaml::from_slice(blob).map_err(|e| SkillPackError::SchemaViolation {
            path: THRESHOLDS_PATH.to_string(),
            detail: format!("YAML decode failed: {e}"),
        })?;
    if dto.schema_version < 1 {
        return Err(SkillPackError::SchemaViolation {
            path: THRESHOLDS_PATH.to_string(),
            detail: "schema_version must be >= 1".to_string(),
        });
    }
    let mut norm: BTreeMap<String, BTreeMap<String, ThresholdEntry>> = BTreeMap::new();
    for (cat, labels) in dto.thresholds {
        let mut inner = BTreeMap::new();
        for (name, entry) in labels {
            let te = ThresholdEntry::new_with_route(entry.trigger, entry.severe, entry.route)
                .map_err(|e| SkillPackError::SchemaViolation {
                    path: THRESHOLDS_PATH.to_string(),
                    detail: format!("threshold {cat}.{name}: {e}"),
                })?;
            inner.insert(name, te);
        }
        norm.insert(cat, inner);
    }
    ThresholdsConfig::new(norm).map_err(|e| SkillPackError::SchemaViolation {
        path: THRESHOLDS_PATH.to_string(),
        detail: format!("ThresholdsConfig construction failed: {e}"),
    })
}

fn decode_rubric(blob: &[u8]) -> Result<SeverityRubric, SkillPackError> {
    let dto: SeverityRubricDto =
        serde_yaml::from_slice(blob).map_err(|e| SkillPackError::SchemaViolation {
            path: RUBRIC_PATH.to_string(),
            detail: format!("YAML decode failed: {e}"),
        })?;
    let mut levels = Vec::with_capacity(dto.levels.len());
    for lv in dto.levels {
        let action =
            parse_ux_action(&lv.ux_action).map_err(|e| SkillPackError::SchemaViolation {
                path: RUBRIC_PATH.to_string(),
                detail: format!("level {} ux_action: {e}", lv.level),
            })?;
        let level = SeverityLevel::new(lv.level, &lv.name, action)
            .map_err(|e| SkillPackError::SchemaViolation {
                path: RUBRIC_PATH.to_string(),
                detail: format!("level {}: {e}", lv.level),
            })?
            .with_allow_reveal(lv.allow_reveal)
            .with_allow_forward(lv.allow_forward)
            .with_description(&lv.description);
        levels.push(level);
    }
    SeverityRubric::new(levels).map_err(|e| SkillPackError::SchemaViolation {
        path: RUBRIC_PATH.to_string(),
        detail: format!("rubric construction failed: {e}"),
    })
}

fn parse_ux_action(s: &str) -> Result<UXAction, String> {
    match s {
        "clear" => Ok(UXAction::Clear),
        "blur_tap" => Ok(UXAction::BlurTap),
        "pixelate" => Ok(UXAction::Pixelate),
        "blocked_card" => Ok(UXAction::BlockedCard),
        other => Err(format!(
            "ux_action {other:?} not in {{clear, blur_tap, pixelate, blocked_card}}"
        )),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LexiconDto {
    /// Optional in the YAML; the loader fills it in from the
    /// filename when the file omits it explicitly.
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    entries: Vec<LexiconEntryDto>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum LexiconEntryDto {
    /// Bare-string entry, e.g. `- "free money"`.
    Bare(String),
    /// Full struct entry, e.g.
    /// `{phrase: "free money", weight: 2.0, tags: ["scam"]}`.
    Full(LexiconEntryStructDto),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LexiconEntryStructDto {
    phrase: String,
    #[serde(default = "default_weight")]
    weight: f64,
    #[serde(default)]
    tags: Vec<String>,
}

fn default_weight() -> f64 {
    1.0
}

fn decode_lexicon(blob: &[u8], path: &str, default_lang: &str) -> Result<Lexicon, SkillPackError> {
    let dto: LexiconDto =
        serde_yaml::from_slice(blob).map_err(|e| SkillPackError::SchemaViolation {
            path: path.to_string(),
            detail: format!("YAML decode failed: {e}"),
        })?;
    let language = dto.language.unwrap_or_else(|| default_lang.to_string());
    let entries: Vec<LexiconEntry> = dto
        .entries
        .into_iter()
        .map(|e| match e {
            LexiconEntryDto::Bare(phrase) => LexiconEntry {
                phrase,
                weight: 1.0,
                tags: Vec::new(),
            },
            LexiconEntryDto::Full(s) => LexiconEntry {
                phrase: s.phrase,
                weight: s.weight,
                tags: s.tags,
            },
        })
        .collect();
    let lex = Lexicon { language, entries };
    lex.validate(path)?;
    Ok(lex)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegexSetDto {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    patterns: Vec<RegexPatternDto>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegexPatternDto {
    name: String,
    pattern: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    flags: Vec<String>,
}

fn decode_regex_set(
    blob: &[u8],
    path: &str,
    default_name: &str,
) -> Result<RegexSet, SkillPackError> {
    let dto: RegexSetDto =
        serde_yaml::from_slice(blob).map_err(|e| SkillPackError::SchemaViolation {
            path: path.to_string(),
            detail: format!("YAML decode failed: {e}"),
        })?;
    let name = dto.name.unwrap_or_else(|| default_name.to_string());
    let patterns: Vec<RegexPattern> = dto
        .patterns
        .into_iter()
        .map(|p| RegexPattern {
            name: p.name,
            pattern: p.pattern,
            description: p.description,
            flags: p.flags,
        })
        .collect();
    let set = RegexSet { name, patterns };
    set.validate(path)?;
    Ok(set)
}

// ---------------------------------------------------------------------------
// Top-level loaders.
// ---------------------------------------------------------------------------

/// Deserialize an already-verified set of `(path, bytes)` entries
/// into a [`SkillPack`].
///
/// `file_bytes` is typically obtained from
/// [`super::verifier::verify_skill_pack`]'s
/// [`VerificationResult::file_bytes`]. The loader does **not**
/// re-verify the signature — callers that want a one-shot
/// verify-and-load should call [`load_verified_skill_pack`].
///
/// # Errors
///
/// * [`SkillPackError::MissingFile`] for any required file that
///   is absent from the map (`taxonomy.yaml`, `thresholds.yaml`,
///   `severity_rubric.yaml`, `slm_prompt.txt`).
/// * [`SkillPackError::SchemaViolation`] for any YAML / text
///   payload that fails to deserialize or fails its structural
///   validator.
pub fn load_skill_pack_from_files(
    file_bytes: &BTreeMap<String, Vec<u8>>,
    manifest: SkillPackManifest,
) -> Result<SkillPack, SkillPackError> {
    for required in &[TAXONOMY_PATH, THRESHOLDS_PATH, RUBRIC_PATH, SLM_PROMPT_PATH] {
        if !file_bytes.contains_key(*required) {
            return Err(SkillPackError::MissingFile((*required).to_string()));
        }
    }

    let taxonomy = decode_taxonomy(&file_bytes[TAXONOMY_PATH])?;
    let thresholds = decode_thresholds(&file_bytes[THRESHOLDS_PATH])?;
    let severity_rubric = decode_rubric(&file_bytes[RUBRIC_PATH])?;
    let slm_prompt = std::str::from_utf8(&file_bytes[SLM_PROMPT_PATH])
        .map_err(|e| SkillPackError::SchemaViolation {
            path: SLM_PROMPT_PATH.to_string(),
            detail: format!("slm_prompt.txt is not valid UTF-8: {e}"),
        })?
        .to_string();

    let mut scam_phrases: BTreeMap<String, Lexicon> = BTreeMap::new();
    let mut hate_lexicons: BTreeMap<String, Lexicon> = BTreeMap::new();
    let mut regex_sets: BTreeMap<String, RegexSet> = BTreeMap::new();

    let scam_prefix = format!("{SCAM_PHRASES_DIR}/");
    let hate_prefix = format!("{HATE_LEXICON_DIR}/");
    let regex_prefix = format!("{REGEX_DIR}/");

    for (path, blob) in file_bytes {
        if let Some(key) = strip_yaml_dir_key(path, &scam_prefix) {
            scam_phrases.insert(key.clone(), decode_lexicon(blob, path, &key)?);
        } else if let Some(key) = strip_yaml_dir_key(path, &hate_prefix) {
            hate_lexicons.insert(key.clone(), decode_lexicon(blob, path, &key)?);
        } else if let Some(key) = strip_yaml_dir_key(path, &regex_prefix) {
            regex_sets.insert(key.clone(), decode_regex_set(blob, path, &key)?);
        }
    }

    Ok(SkillPack {
        manifest,
        taxonomy,
        thresholds,
        severity_rubric,
        scam_phrases,
        hate_lexicons,
        regex_sets,
        slm_prompt,
    })
}

/// If `path` lives directly under `prefix` and ends in `.yaml`,
/// return the basename stripped of both. Returns `None` for
/// paths that are nested under a deeper sub-directory (the
/// compiler does not emit those; rejecting them keeps the
/// key shape `<dir>/<key>.yaml` rigid).
fn strip_yaml_dir_key(path: &str, prefix: &str) -> Option<String> {
    let rest = path.strip_prefix(prefix)?;
    if rest.contains('/') {
        return None;
    }
    let key = rest.strip_suffix(".yaml")?;
    if key.is_empty() {
        return None;
    }
    Some(key.to_string())
}

/// Verify and load a skill pack in a single call. The most
/// common entrypoint for production hosts.
pub fn load_verified_skill_pack<'a>(
    source: impl Into<SkillPackSource<'a>>,
    pinned_public_key_hex: &str,
) -> Result<SkillPack, SkillPackError> {
    let VerificationResult {
        manifest,
        file_bytes,
    } = verify_skill_pack(source, pinned_public_key_hex)?;
    load_skill_pack_from_files(&file_bytes, manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_yaml_dir_key_handles_canonical_layout() {
        assert_eq!(
            strip_yaml_dir_key("scam_phrases/en.yaml", "scam_phrases/"),
            Some("en".to_string())
        );
    }

    #[test]
    fn strip_yaml_dir_key_rejects_nested_paths() {
        assert_eq!(
            strip_yaml_dir_key("scam_phrases/nested/en.yaml", "scam_phrases/"),
            None
        );
    }

    #[test]
    fn strip_yaml_dir_key_rejects_non_yaml_extension() {
        assert_eq!(
            strip_yaml_dir_key("scam_phrases/en.json", "scam_phrases/"),
            None
        );
    }

    #[test]
    fn strip_yaml_dir_key_rejects_empty_basename() {
        assert_eq!(
            strip_yaml_dir_key("scam_phrases/.yaml", "scam_phrases/"),
            None
        );
    }

    #[test]
    fn strip_yaml_dir_key_rejects_wrong_prefix() {
        assert_eq!(strip_yaml_dir_key("regex/pii.yaml", "scam_phrases/"), None);
    }

    #[test]
    fn decode_taxonomy_parses_canonical_yaml() {
        let yaml = b"schema_version: 1\nlabels:\n  adult: [nudity]\n  scam: [money_request]\n";
        let t = decode_taxonomy(yaml).unwrap();
        assert_eq!(t.schema_version, 1);
        assert_eq!(t.labels.len(), 2);
        assert_eq!(t.labels["adult"], vec!["nudity".to_string()]);
    }

    #[test]
    fn decode_taxonomy_rejects_empty_labels() {
        let yaml = b"schema_version: 1\nlabels: {}\n";
        let err = decode_taxonomy(yaml).unwrap_err();
        assert!(
            matches!(err, SkillPackError::SchemaViolation { path, .. } if path == TAXONOMY_PATH)
        );
    }

    #[test]
    fn decode_taxonomy_rejects_unknown_top_level_field() {
        let yaml = b"schema_version: 1\nlabels:\n  adult: [nudity]\nrogue: 1\n";
        assert!(decode_taxonomy(yaml).is_err());
    }

    #[test]
    fn decode_thresholds_parses_canonical_yaml() {
        let yaml = b"schema_version: 1\nthresholds:\n  adult:\n    nudity:\n      trigger: 0.4\n      severe: 0.85\n";
        let t = decode_thresholds(yaml).unwrap();
        let entry = t.entry("adult", "nudity").expect("entry should exist");
        assert!((entry.trigger - 0.4).abs() < 1e-9);
        assert!((entry.severe.unwrap() - 0.85).abs() < 1e-9);
    }

    #[test]
    fn decode_thresholds_rejects_trigger_greater_than_severe() {
        let yaml = b"schema_version: 1\nthresholds:\n  adult:\n    nudity:\n      trigger: 0.9\n      severe: 0.5\n";
        assert!(decode_thresholds(yaml).is_err());
    }

    #[test]
    fn decode_rubric_parses_full_zero_to_five_coverage() {
        let yaml = b"
schema_version: 1
levels:
  - { level: 0, name: safe,     ux_action: clear }
  - { level: 1, name: low,      ux_action: clear }
  - { level: 2, name: medium,   ux_action: blur_tap }
  - { level: 3, name: high,     ux_action: blur_tap, allow_forward: false }
  - { level: 4, name: severe,   ux_action: pixelate, allow_forward: false }
  - { level: 5, name: critical, ux_action: blocked_card, allow_forward: false, allow_reveal: false }
";
        let r = decode_rubric(yaml).unwrap();
        assert_eq!(r.levels.len(), 6);
    }

    #[test]
    fn decode_rubric_rejects_missing_level() {
        let yaml = b"
schema_version: 1
levels:
  - { level: 0, name: safe,   ux_action: clear }
  - { level: 1, name: low,    ux_action: clear }
  - { level: 2, name: medium, ux_action: blur_tap }
  - { level: 3, name: high,   ux_action: blur_tap }
  - { level: 4, name: severe, ux_action: pixelate }
";
        assert!(decode_rubric(yaml).is_err());
    }

    #[test]
    fn decode_rubric_rejects_unknown_ux_action() {
        let yaml = b"
schema_version: 1
levels:
  - { level: 0, name: safe, ux_action: ignore_completely }
  - { level: 1, name: low, ux_action: clear }
  - { level: 2, name: medium, ux_action: blur_tap }
  - { level: 3, name: high, ux_action: blur_tap }
  - { level: 4, name: severe, ux_action: pixelate }
  - { level: 5, name: critical, ux_action: blocked_card }
";
        assert!(decode_rubric(yaml).is_err());
    }

    #[test]
    fn decode_lexicon_handles_bare_string_entries() {
        let yaml = b"language: en\nentries:\n  - free money\n  - urgent transfer\n";
        let lex = decode_lexicon(yaml, "scam_phrases/en.yaml", "en").unwrap();
        assert_eq!(lex.language, "en");
        assert_eq!(lex.entries.len(), 2);
        assert_eq!(lex.entries[0].phrase, "free money");
        assert!((lex.entries[0].weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn decode_lexicon_handles_struct_entries_with_weight_and_tags() {
        let yaml = b"
language: en
entries:
  - phrase: free money
    weight: 2.5
    tags: [scam, urgency]
";
        let lex = decode_lexicon(yaml, "scam_phrases/en.yaml", "en").unwrap();
        assert_eq!(lex.entries.len(), 1);
        assert!((lex.entries[0].weight - 2.5).abs() < 1e-9);
        assert_eq!(
            lex.entries[0].tags,
            vec!["scam".to_string(), "urgency".to_string()]
        );
    }

    #[test]
    fn decode_lexicon_fills_language_from_default_when_omitted() {
        let yaml = b"entries:\n  - bare entry\n";
        let lex = decode_lexicon(yaml, "scam_phrases/ja.yaml", "ja").unwrap();
        assert_eq!(lex.language, "ja");
    }

    #[test]
    fn decode_lexicon_rejects_blank_phrase() {
        let yaml = b"language: en\nentries:\n  - phrase: \"\"\n    weight: 1\n";
        assert!(decode_lexicon(yaml, "scam_phrases/en.yaml", "en").is_err());
    }

    #[test]
    fn decode_lexicon_rejects_invalid_language_code() {
        let yaml = b"language: english\nentries:\n  - ok\n";
        assert!(decode_lexicon(yaml, "scam_phrases/en.yaml", "english").is_err());
    }

    #[test]
    fn decode_regex_set_parses_canonical_yaml() {
        let yaml = b"
name: pii
patterns:
  - name: luhn_credit_card
    pattern: '\\b\\d{13,16}\\b'
    flags: [m]
  - name: email
    pattern: '[a-z]+@[a-z]+\\.[a-z]+'
";
        let set = decode_regex_set(yaml, "regex/pii.yaml", "pii").unwrap();
        assert_eq!(set.name, "pii");
        assert_eq!(set.patterns.len(), 2);
    }

    #[test]
    fn decode_regex_set_rejects_malformed_pattern() {
        let yaml = b"
name: pii
patterns:
  - name: broken
    pattern: '[unclosed'
";
        assert!(decode_regex_set(yaml, "regex/pii.yaml", "pii").is_err());
    }

    #[test]
    fn load_from_files_rejects_missing_required_file() {
        let mut file_bytes: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        // Only one of the four required files present.
        file_bytes.insert(
            TAXONOMY_PATH.to_string(),
            b"schema_version: 1\nlabels:\n  adult: [nudity]\n".to_vec(),
        );
        let m = SkillPackManifest {
            pack_id: "cvguard.skill.x.v1".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            schema_version: 1,
            author: "test".to_string(),
            description: String::new(),
            min_runtime_version: "0.1.0".to_string(),
            content_sha256: "0".repeat(64),
            signature: None,
            public_key: None,
        };
        let err = load_skill_pack_from_files(&file_bytes, m).unwrap_err();
        assert!(matches!(err, SkillPackError::MissingFile(_)));
    }
}

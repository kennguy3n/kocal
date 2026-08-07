//! Closed-shape Rust types for the compiled skill-pack format.
//!
//! Mirrors cv-guard's `shared/skillpack/schema.py` one-for-one:
//!
//! * [`SkillPackManifest`] — top-level metadata, the only file
//!   directly covered by the ed25519 signature.
//! * [`TaxonomyConfig`] — `taxonomy.yaml`, the closed-set
//!   per-category label vocabulary the trained classifier head
//!   must emit.
//! * [`LexiconEntry`] / [`Lexicon`] — per-language scam phrase
//!   and hate-speech lexicons stored under
//!   [`super::SCAM_PHRASES_DIR`] / [`super::HATE_LEXICON_DIR`].
//! * [`RegexPattern`] / [`RegexSet`] — named regex sets under
//!   [`super::REGEX_DIR`] (e.g. `pii.yaml` → Luhn / IBAN / phone
//!   patterns). The Rust side validates that every pattern is
//!   well-formed under the `regex` crate at deserialization time
//!   so a malformed pack fails to load instead of silently
//!   skipping the pattern at scan time.
//! * [`SkillPack`] — fully-validated in-memory pack, composed
//!   together by [`super::loader`] after [`super::verifier`]
//!   accepts the archive's signature.
//!
//! The [`ThresholdsConfig`] / [`SeverityRubric`] types are re-used
//! from [`crate::policy_interpreter`] rather than re-defined here,
//! so the runtime carries exactly one definition of each shape.
//! This means downstream consumers reading e.g. the thresholds
//! out of a loaded pack hold the same type the
//! [`crate::policy_interpreter::PolicyInterpreter`] reads — no
//! conversion step required.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::SkillPackError;

/// Fully-qualified skill-pack ID convention:
/// `<namespace>.skill.<name>.v<major>`. Matches Python's
/// `SKILL_PACK_ID_PATTERN` regex byte-for-byte. The compiler
/// refuses IDs that don't match this shape so pack filenames
/// and content-addressed distribution paths stay predictable.
///
/// Returns `true` iff `s` is a well-formed pack ID.
pub fn is_valid_pack_id(s: &str) -> bool {
    // Equivalent to `r"^[a-z][a-z0-9_]*\.skill\.[a-z][a-z0-9_]*\.v\d+$"`.
    // Hand-rolled to avoid a circular dep on the regex crate
    // (which only ships under `text-pipeline`).
    let mut iter = s.split('.');
    let Some(ns) = iter.next() else {
        return false;
    };
    if !is_valid_snake_word(ns) {
        return false;
    }
    let Some(kw) = iter.next() else {
        return false;
    };
    if kw != "skill" {
        return false;
    }
    let Some(name) = iter.next() else {
        return false;
    };
    if !is_valid_snake_word(name) {
        return false;
    }
    let Some(ver) = iter.next() else {
        return false;
    };
    if !is_valid_version_segment(ver) {
        return false;
    }
    // No trailing segments allowed.
    iter.next().is_none()
}

fn is_valid_snake_word(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return false,
    }
    for c in chars {
        if !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            return false;
        }
    }
    true
}

fn is_valid_version_segment(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some('v') => {}
        _ => return false,
    }
    let digits: String = chars.collect();
    if digits.is_empty() {
        return false;
    }
    digits.chars().all(|c| c.is_ascii_digit())
}

/// Top-level metadata for a compiled skill pack.
///
/// The manifest is the only file directly covered by the
/// ed25519 signature — the `content_sha256` field is the
/// SHA-256-of-sorted-paths digest of every other file in the
/// archive (see
/// [`crate::crypto::digest::compute_content_digest`]).
///
/// Schema parity: mirrors Python's
/// `shared.skillpack.schema.SkillPackManifest` with
/// `ConfigDict(extra="forbid")` — unknown keys at the manifest
/// level cause a [`SkillPackError::SchemaViolation`] at parse
/// time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SkillPackManifest {
    /// Fully-qualified pack ID, e.g.
    /// `cvguard.skill.global_baseline.v1`.
    pub pack_id: String,
    /// Semver-ish version string, e.g. `1.0.0`.
    pub version: String,
    /// ISO-8601 UTC timestamp emitted by the compiler.
    pub created_at: String,
    /// Manifest schema-version pin. Defaults to `1` if absent.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Human-readable author. Defaults to `"CV-Guard"`.
    #[serde(default = "default_author")]
    pub author: String,
    /// Free-form description.
    #[serde(default)]
    pub description: String,
    /// Minimum runtime semver that can load this pack.
    #[serde(default = "default_min_runtime_version")]
    pub min_runtime_version: String,
    /// SHA-256 of the canonical content digest. 64-char lowercase
    /// hex.
    pub content_sha256: String,
    /// Hex-encoded ed25519 signature over the signing preimage
    /// `{content_sha256}|{pack_id}|{version}`. 128-char lowercase
    /// hex. `None` for unsigned development builds.
    #[serde(default)]
    pub signature: Option<String>,
    /// Hex-encoded ed25519 public key (64 chars). Mandatory on
    /// signed builds; the verifier compares this against the
    /// caller-pinned key before trusting the signature.
    #[serde(default)]
    pub public_key: Option<String>,
}

fn default_schema_version() -> u32 {
    1
}

fn default_author() -> String {
    "CV-Guard".to_string()
}

fn default_min_runtime_version() -> String {
    "0.1.0".to_string()
}

impl SkillPackManifest {
    /// Validate the manifest's structural invariants. Called
    /// after `serde_json::from_slice` to surface
    /// `SkillPackError::SchemaViolation` instead of leaking a
    /// raw deserialize error.
    pub fn validate(&self) -> Result<(), SkillPackError> {
        if !is_valid_pack_id(&self.pack_id) {
            return Err(SkillPackError::SchemaViolation {
                path: "manifest.json".to_string(),
                detail: format!(
                    "invalid pack_id {:?}: expected <ns>.skill.<name>.v<n>",
                    self.pack_id
                ),
            });
        }
        if self.schema_version < 1 {
            return Err(SkillPackError::SchemaViolation {
                path: "manifest.json".to_string(),
                detail: "schema_version must be >= 1".to_string(),
            });
        }
        if !is_valid_sha256_hex(&self.content_sha256) {
            return Err(SkillPackError::SchemaViolation {
                path: "manifest.json".to_string(),
                detail: format!(
                    "content_sha256 must be 64 lowercase hex chars, got {:?}",
                    self.content_sha256
                ),
            });
        }
        if let Some(sig) = self.signature.as_ref() {
            if !is_valid_signature_hex(sig) {
                return Err(SkillPackError::SchemaViolation {
                    path: "manifest.json".to_string(),
                    detail: format!("signature must be 128 lowercase hex chars, got {:?}", sig),
                });
            }
        }
        if let Some(pk) = self.public_key.as_ref() {
            if !is_valid_pubkey_hex(pk) {
                return Err(SkillPackError::SchemaViolation {
                    path: "manifest.json".to_string(),
                    detail: format!("public_key must be 64 lowercase hex chars, got {:?}", pk),
                });
            }
        }
        Ok(())
    }
}

fn is_valid_sha256_hex(s: &str) -> bool {
    s.len() == 64
        && s.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

fn is_valid_signature_hex(s: &str) -> bool {
    s.len() == 128
        && s.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

fn is_valid_pubkey_hex(s: &str) -> bool {
    is_valid_sha256_hex(s)
}

/// Mirror of `taxonomy.yaml`. The compiler validates this against
/// the trained classifier head's emitted labels at build time;
/// the runtime side keeps the structure but does not re-validate
/// against the head (loading the head is a separate concern).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaxonomyConfig {
    /// Schema-version pin. Defaults to `1` if absent.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Per-category list of label names (e.g.
    /// `{"adult": ["nudity"], "scam": ["money_request"]}`). The
    /// category insertion order is preserved (we use a
    /// `BTreeMap` so iteration is deterministic; the original
    /// Python uses dict-insertion-order but the order is
    /// irrelevant to correctness — only the set of labels matters).
    pub labels: BTreeMap<String, Vec<String>>,
    /// Free-form output-mapping metadata. Carried verbatim
    /// (mostly used by the compiler for hint generation; the
    /// runtime treats it as opaque).
    #[serde(default)]
    pub output: BTreeMap<String, serde_yaml::Value>,
}

impl TaxonomyConfig {
    /// Flatten to `("category.name", ...)` in declaration order.
    /// Categories are iterated in sorted-key order (BTreeMap);
    /// labels within a category preserve their YAML order.
    pub fn all_labels(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (cat, names) in &self.labels {
            for n in names {
                out.push(format!("{cat}.{n}"));
            }
        }
        out
    }

    /// Structural validation. Called by the loader after YAML
    /// deserialization. Mirrors Python's `_non_empty`
    /// `model_validator`.
    pub fn validate(&self) -> Result<(), SkillPackError> {
        if self.labels.is_empty() {
            return Err(SkillPackError::SchemaViolation {
                path: super::TAXONOMY_PATH.to_string(),
                detail: "taxonomy.labels must not be empty".to_string(),
            });
        }
        for (cat, names) in &self.labels {
            if names.is_empty() {
                return Err(SkillPackError::SchemaViolation {
                    path: super::TAXONOMY_PATH.to_string(),
                    detail: format!("taxonomy category {cat:?} has no labels"),
                });
            }
        }
        Ok(())
    }
}

/// One phrase / token in a language-tagged lexicon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexiconEntry {
    /// The exact phrase / token text.
    pub phrase: String,
    /// Weight in `[0.0, 10.0]`. Defaults to `1.0`.
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Free-form tags (e.g. `["scam", "phishing"]`).
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_weight() -> f64 {
    1.0
}

impl LexiconEntry {
    /// Validate the structural invariants on a single entry.
    /// Mirrors Python's `_non_empty` field validator on `phrase`
    /// plus the `Field(..., ge=0.0, le=10.0)` constraint on
    /// `weight`.
    pub fn validate(&self) -> Result<(), SkillPackError> {
        if self.phrase.trim().is_empty() {
            return Err(SkillPackError::SchemaViolation {
                path: "<lexicon>".to_string(),
                detail: "LexiconEntry.phrase must be non-empty".to_string(),
            });
        }
        if !self.weight.is_finite() || !(0.0..=10.0).contains(&self.weight) {
            return Err(SkillPackError::SchemaViolation {
                path: "<lexicon>".to_string(),
                detail: format!("LexiconEntry.weight {} not in [0.0, 10.0]", self.weight),
            });
        }
        Ok(())
    }
}

/// Language-tagged lexicon (one YAML file under
/// [`super::SCAM_PHRASES_DIR`] or [`super::HATE_LEXICON_DIR`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lexicon {
    /// ISO-639-1 or ISO-639-3 language code (e.g. `"en"`,
    /// `"es"`, `"jpn"`). Hand-validated to mirror Python's
    /// `^[a-z]{2,3}$` regex.
    pub language: String,
    /// Entries in YAML declaration order.
    pub entries: Vec<LexiconEntry>,
}

impl Lexicon {
    /// Validate `language` is 2–3 lowercase ASCII characters
    /// and every entry passes [`LexiconEntry::validate`].
    pub fn validate(&self, path: &str) -> Result<(), SkillPackError> {
        if !(2..=3).contains(&self.language.len())
            || !self.language.chars().all(|c| c.is_ascii_lowercase())
        {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!(
                    "Lexicon.language {:?} must be 2-3 lowercase ASCII chars",
                    self.language
                ),
            });
        }
        for (idx, entry) in self.entries.iter().enumerate() {
            entry.validate().map_err(|e| match e {
                SkillPackError::SchemaViolation { detail, .. } => SkillPackError::SchemaViolation {
                    path: path.to_string(),
                    detail: format!("entry[{idx}]: {detail}"),
                },
                other => other,
            })?;
        }
        Ok(())
    }
}

/// A named regex pattern (one row of a [`RegexSet`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegexPattern {
    /// Snake-case ASCII name (e.g. `"luhn_credit_card"`).
    pub name: String,
    /// The raw regex source. Validated at load time by
    /// attempting to compile the pattern through the `regex`
    /// crate — a malformed regex fails the load immediately
    /// rather than at the first scan that uses it.
    pub pattern: String,
    /// Free-form description.
    #[serde(default)]
    pub description: String,
    /// Compile-time flags (e.g. `"i"` for case-insensitive).
    /// Validated to be a subset of `{"i", "m", "s", "x"}` at
    /// load time.
    #[serde(default)]
    pub flags: Vec<String>,
}

impl RegexPattern {
    /// Validate the pattern's structural invariants. Compiles
    /// the regex via [`regex::Regex`] so a malformed source
    /// fails loudly here.
    pub fn validate(&self, path: &str) -> Result<(), SkillPackError> {
        if !is_valid_snake_word(&self.name) {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!(
                    "regex name {:?} must be snake_case ascii (lowercase letters / digits / underscore)",
                    self.name
                ),
            });
        }
        for flag in &self.flags {
            if !matches!(flag.as_str(), "i" | "m" | "s" | "x") {
                return Err(SkillPackError::SchemaViolation {
                    path: path.to_string(),
                    detail: format!("regex flag {flag:?} not in {{\"i\", \"m\", \"s\", \"x\"}}"),
                });
            }
        }
        if let Err(e) = regex::Regex::new(&self.pattern) {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!("regex pattern fails to compile: {e}"),
            });
        }
        Ok(())
    }

    /// Compile the pattern into a runtime [`regex::Regex`] with
    /// the declared flags applied via [`regex::RegexBuilder`].
    /// This is the canonical consumer-side entry point — the
    /// validator at [`Self::validate`] only checks structural
    /// invariants + that the bare pattern parses, but does NOT
    /// apply the flags. Callers running the regex against
    /// content MUST go through `compile()` so that the closed
    /// flag set `{i, m, s, x}` is consistently honored.
    ///
    /// Flag mapping (matches Python `re.compile(pattern, flags)`
    /// semantics):
    /// - `"i"` -> [`RegexBuilder::case_insensitive`]
    /// - `"m"` -> [`RegexBuilder::multi_line`]
    /// - `"s"` -> [`RegexBuilder::dot_matches_new_line`]
    /// - `"x"` -> [`RegexBuilder::ignore_whitespace`]
    ///
    /// Any flag outside this closed set yields
    /// [`SkillPackError::SchemaViolation`] — same path as
    /// [`Self::validate`], so a consumer that did not pre-validate
    /// still gets a useful error rather than a silent semantic
    /// drift.
    ///
    /// # Errors
    /// * [`SkillPackError::SchemaViolation`] if the pattern fails
    ///   to compile or a flag outside `{"i", "m", "s", "x"}` is
    ///   present.
    pub fn compile(&self) -> Result<regex::Regex, SkillPackError> {
        let mut builder = regex::RegexBuilder::new(&self.pattern);
        for flag in &self.flags {
            match flag.as_str() {
                "i" => {
                    builder.case_insensitive(true);
                }
                "m" => {
                    builder.multi_line(true);
                }
                "s" => {
                    builder.dot_matches_new_line(true);
                }
                "x" => {
                    builder.ignore_whitespace(true);
                }
                other => {
                    return Err(SkillPackError::SchemaViolation {
                        path: self.name.clone(),
                        detail: format!(
                            "regex flag {other:?} not in {{\"i\", \"m\", \"s\", \"x\"}}"
                        ),
                    });
                }
            }
        }
        builder
            .build()
            .map_err(|e| SkillPackError::SchemaViolation {
                path: self.name.clone(),
                detail: format!("regex pattern fails to compile with flags: {e}"),
            })
    }
}

/// A named collection of regex patterns (one YAML file under
/// [`super::REGEX_DIR`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegexSet {
    /// Snake-case ASCII set name (e.g. `"pii"`).
    pub name: String,
    /// Patterns in declaration order.
    pub patterns: Vec<RegexPattern>,
}

impl RegexSet {
    /// Validate the set's name + every contained pattern.
    pub fn validate(&self, path: &str) -> Result<(), SkillPackError> {
        if !is_valid_snake_word(&self.name) {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!("RegexSet.name {:?} must be snake_case ascii", self.name),
            });
        }
        for (idx, pat) in self.patterns.iter().enumerate() {
            pat.validate(path).map_err(|e| match e {
                SkillPackError::SchemaViolation { detail, .. } => SkillPackError::SchemaViolation {
                    path: path.to_string(),
                    detail: format!("patterns[{idx}]: {detail}"),
                },
                other => other,
            })?;
        }
        Ok(())
    }
}

/// Complete, verified skill pack in memory. Returned by
/// [`super::loader::load_verified_skill_pack`] (or, for already-
/// extracted file bytes, by
/// [`super::loader::load_skill_pack_from_files`]).
#[derive(Debug, Clone)]
pub struct SkillPack {
    /// Verified manifest.
    pub manifest: SkillPackManifest,
    /// Decoded `taxonomy.yaml`.
    pub taxonomy: TaxonomyConfig,
    /// Decoded `thresholds.yaml` — re-uses the shared
    /// [`crate::policy_interpreter::ThresholdsConfig`].
    pub thresholds: crate::policy_interpreter::ThresholdsConfig,
    /// Decoded `severity_rubric.yaml` — re-uses the shared
    /// [`crate::policy_interpreter::SeverityRubric`].
    pub severity_rubric: crate::policy_interpreter::SeverityRubric,
    /// `scam_phrases/<key>.yaml` → [`Lexicon`], keyed by file
    /// basename without the `.yaml` suffix.
    pub scam_phrases: BTreeMap<String, Lexicon>,
    /// `hate_lexicon/<key>.yaml` → [`Lexicon`], same key shape.
    pub hate_lexicons: BTreeMap<String, Lexicon>,
    /// `regex/<key>.yaml` → [`RegexSet`], same key shape.
    pub regex_sets: BTreeMap<String, RegexSet>,
    /// Raw UTF-8 contents of `slm_prompt.txt`. Empty string if
    /// the pack ships no prompt.
    pub slm_prompt: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_valid_pack_id_accepts_canonical_shape() {
        assert!(is_valid_pack_id("cvguard.skill.global_baseline.v1"));
        assert!(is_valid_pack_id("cvguard.skill.global_baseline.v42"));
        assert!(is_valid_pack_id("ns.skill.a.v0"));
        assert!(is_valid_pack_id("a1.skill.b_c_d2.v3"));
    }

    #[test]
    fn is_valid_pack_id_rejects_drift() {
        assert!(!is_valid_pack_id(""));
        assert!(!is_valid_pack_id("cvguard.skill.global_baseline"));
        assert!(!is_valid_pack_id("cvguard.skill.global_baseline.v"));
        assert!(!is_valid_pack_id("cvguard.skill.global_baseline.v1.x"));
        assert!(!is_valid_pack_id("Cvguard.skill.global_baseline.v1"));
        assert!(!is_valid_pack_id("cvguard.SKILL.global_baseline.v1"));
        assert!(!is_valid_pack_id("cvguard.skill..v1"));
        assert!(!is_valid_pack_id("1cvguard.skill.global_baseline.v1"));
        assert!(!is_valid_pack_id("cvguard.skill.global_baseline.va"));
    }

    #[test]
    fn manifest_validate_rejects_short_digest() {
        let m = SkillPackManifest {
            pack_id: "cvguard.skill.x.v1".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            schema_version: 1,
            author: "test".to_string(),
            description: String::new(),
            min_runtime_version: "0.1.0".to_string(),
            content_sha256: "abc".to_string(),
            signature: None,
            public_key: None,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn manifest_validate_accepts_signed_canonical_shape() {
        let m = SkillPackManifest {
            pack_id: "cvguard.skill.x.v1".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            schema_version: 1,
            author: "test".to_string(),
            description: String::new(),
            min_runtime_version: "0.1.0".to_string(),
            content_sha256: "0".repeat(64),
            signature: Some("0".repeat(128)),
            public_key: Some("0".repeat(64)),
        };
        m.validate().unwrap();
    }

    #[test]
    fn manifest_validate_rejects_signature_wrong_length() {
        let m = SkillPackManifest {
            pack_id: "cvguard.skill.x.v1".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            schema_version: 1,
            author: "test".to_string(),
            description: String::new(),
            min_runtime_version: "0.1.0".to_string(),
            content_sha256: "0".repeat(64),
            signature: Some("00".to_string()),
            public_key: None,
        };
        assert!(m.validate().is_err());
    }

    #[test]
    fn manifest_unknown_field_rejected_by_serde() {
        let json = r#"{
            "pack_id": "cvguard.skill.x.v1",
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "content_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "rogue_field": "evil"
        }"#;
        let res: Result<SkillPackManifest, _> = serde_json::from_str(json);
        assert!(res.is_err(), "deny_unknown_fields should reject");
    }

    #[test]
    fn taxonomy_all_labels_flattens_in_sorted_category_order() {
        let mut labels = BTreeMap::new();
        labels.insert("scam".to_string(), vec!["money_request".to_string()]);
        labels.insert(
            "adult".to_string(),
            vec!["nudity".to_string(), "explicit".to_string()],
        );
        let t = TaxonomyConfig {
            schema_version: 1,
            labels,
            output: BTreeMap::new(),
        };
        // BTreeMap orders categories ascii-sorted: adult, scam.
        assert_eq!(
            t.all_labels(),
            vec!["adult.nudity", "adult.explicit", "scam.money_request"]
        );
    }

    #[test]
    fn taxonomy_validate_rejects_empty_labels() {
        let t = TaxonomyConfig {
            schema_version: 1,
            labels: BTreeMap::new(),
            output: BTreeMap::new(),
        };
        assert!(t.validate().is_err());
    }

    #[test]
    fn taxonomy_validate_rejects_empty_category() {
        let mut labels = BTreeMap::new();
        labels.insert("adult".to_string(), Vec::<String>::new());
        let t = TaxonomyConfig {
            schema_version: 1,
            labels,
            output: BTreeMap::new(),
        };
        assert!(t.validate().is_err());
    }

    #[test]
    fn lexicon_entry_validate_rejects_blank_phrase() {
        let e = LexiconEntry {
            phrase: "   ".to_string(),
            weight: 1.0,
            tags: vec![],
        };
        assert!(e.validate().is_err());
    }

    #[test]
    fn lexicon_entry_validate_rejects_negative_weight() {
        let e = LexiconEntry {
            phrase: "ok".to_string(),
            weight: -0.1,
            tags: vec![],
        };
        assert!(e.validate().is_err());
    }

    #[test]
    fn lexicon_entry_validate_rejects_weight_above_ten() {
        let e = LexiconEntry {
            phrase: "ok".to_string(),
            weight: 10.0001,
            tags: vec![],
        };
        assert!(e.validate().is_err());
    }

    #[test]
    fn lexicon_entry_validate_rejects_nan_weight() {
        let e = LexiconEntry {
            phrase: "ok".to_string(),
            weight: f64::NAN,
            tags: vec![],
        };
        assert!(e.validate().is_err());
    }

    #[test]
    fn lexicon_validate_rejects_bad_language_code() {
        let lex = Lexicon {
            language: "english".to_string(),
            entries: vec![LexiconEntry {
                phrase: "ok".to_string(),
                weight: 1.0,
                tags: vec![],
            }],
        };
        assert!(lex.validate("scam_phrases/x.yaml").is_err());
    }

    #[test]
    fn lexicon_validate_accepts_two_or_three_char_iso_codes() {
        for lang in &["en", "ja", "zh", "eng", "jpn", "zho"] {
            let lex = Lexicon {
                language: (*lang).to_string(),
                entries: vec![LexiconEntry {
                    phrase: "ok".to_string(),
                    weight: 1.0,
                    tags: vec![],
                }],
            };
            lex.validate("x").unwrap();
        }
    }

    #[test]
    fn regex_pattern_validate_rejects_malformed_pattern() {
        let p = RegexPattern {
            name: "broken".to_string(),
            pattern: "[unclosed".to_string(),
            description: String::new(),
            flags: vec![],
        };
        assert!(p.validate("regex/x.yaml").is_err());
    }

    #[test]
    fn regex_pattern_validate_rejects_non_snake_name() {
        let p = RegexPattern {
            name: "BrokenName".to_string(),
            pattern: "abc".to_string(),
            description: String::new(),
            flags: vec![],
        };
        assert!(p.validate("regex/x.yaml").is_err());
    }

    #[test]
    fn regex_pattern_validate_rejects_unknown_flag() {
        let p = RegexPattern {
            name: "ok".to_string(),
            pattern: "abc".to_string(),
            description: String::new(),
            flags: vec!["z".to_string()],
        };
        assert!(p.validate("regex/x.yaml").is_err());
    }

    #[test]
    fn regex_pattern_validate_accepts_canonical_flags() {
        let p = RegexPattern {
            name: "ok".to_string(),
            pattern: "abc".to_string(),
            description: String::new(),
            flags: vec![
                "i".to_string(),
                "m".to_string(),
                "s".to_string(),
                "x".to_string(),
            ],
        };
        p.validate("regex/x.yaml").unwrap();
    }

    #[test]
    fn regex_pattern_compile_applies_case_insensitive_flag() {
        let p = RegexPattern {
            name: "ci".to_string(),
            pattern: "abc".to_string(),
            description: String::new(),
            flags: vec!["i".to_string()],
        };
        let re = p.compile().unwrap();
        assert!(re.is_match("ABC"));
        assert!(re.is_match("aBc"));
        assert!(re.is_match("abc"));

        let no_flag = RegexPattern {
            name: "no_ci".to_string(),
            pattern: "abc".to_string(),
            description: String::new(),
            flags: vec![],
        };
        assert!(!no_flag.compile().unwrap().is_match("ABC"));
    }

    #[test]
    fn regex_pattern_compile_applies_multi_line_flag() {
        let p = RegexPattern {
            name: "ml".to_string(),
            pattern: "^foo$".to_string(),
            description: String::new(),
            flags: vec!["m".to_string()],
        };
        let re = p.compile().unwrap();
        assert!(re.is_match("foo\nbar"));
        assert!(re.is_match("bar\nfoo"));
    }

    #[test]
    fn regex_pattern_compile_applies_dot_matches_new_line_flag() {
        let p = RegexPattern {
            name: "dotall".to_string(),
            pattern: "a.b".to_string(),
            description: String::new(),
            flags: vec!["s".to_string()],
        };
        let re = p.compile().unwrap();
        assert!(re.is_match("a\nb"));
    }

    #[test]
    fn regex_pattern_compile_applies_ignore_whitespace_flag() {
        let p = RegexPattern {
            name: "extended".to_string(),
            pattern: "a b c # trailing".to_string(),
            description: String::new(),
            flags: vec!["x".to_string()],
        };
        let re = p.compile().unwrap();
        assert!(re.is_match("abc"));
    }

    #[test]
    fn regex_pattern_compile_combines_multiple_flags() {
        let p = RegexPattern {
            name: "combo".to_string(),
            pattern: "^a.b$".to_string(),
            description: String::new(),
            flags: vec!["i".to_string(), "m".to_string(), "s".to_string()],
        };
        let re = p.compile().unwrap();
        assert!(re.is_match("x\nA\nB\ny"));
        assert!(re.is_match("A\nB"));
    }

    #[test]
    fn regex_pattern_compile_rejects_unknown_flag() {
        let p = RegexPattern {
            name: "bad".to_string(),
            pattern: "abc".to_string(),
            description: String::new(),
            flags: vec!["z".to_string()],
        };
        let err = p.compile().unwrap_err();
        assert!(
            matches!(err, SkillPackError::SchemaViolation { ref detail, .. }
                          if detail.contains("not in {\"i\", \"m\", \"s\", \"x\"}"))
        );
    }

    #[test]
    fn regex_pattern_compile_rejects_malformed_pattern() {
        let p = RegexPattern {
            name: "broken".to_string(),
            pattern: "[unclosed".to_string(),
            description: String::new(),
            flags: vec![],
        };
        let err = p.compile().unwrap_err();
        assert!(
            matches!(err, SkillPackError::SchemaViolation { ref detail, .. }
                          if detail.contains("fails to compile with flags"))
        );
    }

    #[test]
    fn regex_set_validate_rejects_non_snake_set_name() {
        let s = RegexSet {
            name: "Broken".to_string(),
            patterns: vec![],
        };
        assert!(s.validate("regex/x.yaml").is_err());
    }

    #[test]
    fn regex_set_validate_propagates_pattern_error() {
        let s = RegexSet {
            name: "pii".to_string(),
            patterns: vec![RegexPattern {
                name: "broken".to_string(),
                pattern: "[unclosed".to_string(),
                description: String::new(),
                flags: vec![],
            }],
        };
        let err = s.validate("regex/pii.yaml").unwrap_err();
        match err {
            SkillPackError::SchemaViolation { path, detail } => {
                assert_eq!(path, "regex/pii.yaml");
                assert!(detail.contains("patterns[0]"), "detail: {detail}");
            }
            other => panic!("expected SchemaViolation, got {other:?}"),
        }
    }
}

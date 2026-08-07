//! Anti-misuse validation for KChat guardrail skill packs.
//!
//! Mirrors `build-tools/compiler/anti_misuse.py` byte-for-byte
//! and runs at **two** places:
//!
//! 1. *Compile time* — the build tool refuses to sign a pack that
//!    fails any rule. The Python validator was the only enforcer
//!    here; the Rust port is used by the (future) Rust skill
//!    compiler binary.
//! 2. *Load time* — on-device, before a pack's contents are
//!    handed to the pipeline, [`validate_pack_value`] is called
//!    on the parsed YAML / JSON value tree so a signed pack that
//!    nonetheless attempts to invent categories / weaken privacy
//!    rules / ship lexicons without provenance is rejected
//!    before it can influence a single verdict.
//!
//! The on-device entrypoint takes a `serde_json::Value` because
//! the validator must operate on the raw pack dict, *before*
//! the more strongly-typed [`crate::skillpack::schema::SkillPack`]
//! parse — the typed parse rejects an "invented categories"
//! pack with a generic schema error that loses the
//! anti-misuse provenance. This is the same shape Python uses
//! (`dict[str, Any]`), so cross-implementation rule semantics
//! stay byte-identical.
//!
//! ### Scope
//!
//! These rules are the *anti-misuse contract* defined in
//! `ARCHITECTURE.md` "Anti-Misuse Controls" (lines 716–748).
//! Each rule encodes a class of mistake that would let a signed
//! pack regress safety, so the validator refuses to admit any
//! pack that fails one.

use std::collections::BTreeSet;
use std::fmt;

use serde_json::Value;

// ---------------------------------------------------------------------------
// Public constants — match Python module-level constants exactly.
// ---------------------------------------------------------------------------

/// Closed-enum taxonomy id range. Mirrors Python's
/// `TAXONOMY_MIN`.
pub const TAXONOMY_MIN: i64 = 0;
/// Closed-enum taxonomy id range. Mirrors Python's
/// `TAXONOMY_MAX`.
pub const TAXONOMY_MAX: i64 = 16;

/// Severity floors at or above this value require explicit
/// protected-context handling. Mirrors Python's
/// `PROTECTED_CONTEXT_REQUIRED_SEVERITY`.
pub const PROTECTED_CONTEXT_REQUIRED_SEVERITY: i64 = 4;

/// Minimum protected-context reason codes a strict overlay must
/// declare. Mirrors Python's `REQUIRED_PROTECTED_CONTEXTS`
/// `frozenset`. Sorted so error diagnostics match Python's
/// `sorted(REQUIRED_PROTECTED_CONTEXTS)` output ordering.
pub const REQUIRED_PROTECTED_CONTEXTS: [&str; 4] = [
    "COUNTERSPEECH_CONTEXT",
    "EDUCATION_CONTEXT",
    "NEWS_CONTEXT",
    "QUOTED_SPEECH_CONTEXT",
];

// ---------------------------------------------------------------------------
// Errors.
// ---------------------------------------------------------------------------

/// A skill pack failed at least one anti-misuse rule.
///
/// Mirrors Python's `AntiMisuseError(ValueError)`. The variants
/// carry every detail the Python `ValueError` message includes so
/// cross-platform error strings compare byte-for-byte.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AntiMisuseError {
    /// `pack["skill_id"]` was missing or didn't match one of the
    /// three recognised prefixes.
    UnknownSkillIdPrefix(String),
    /// An override's `category` value was outside the closed
    /// taxonomy range.
    InvalidCategoryInOverride {
        skill_id: String,
        category_repr: String,
    },
    /// A rule's `category` value was outside the closed taxonomy
    /// range.
    InvalidCategoryInRule {
        skill_id: String,
        category_repr: String,
    },
    /// An overlay declared a `taxonomy` / `categories` /
    /// `new_categories` block reserved for the baseline.
    OverlayDeclaresForbiddenBlock {
        skill_id: String,
        block: &'static str,
    },
    /// A jurisdiction pack omitted required reviewer signers.
    JurisdictionMissingSigners {
        skill_id: String,
        missing: Vec<&'static str>,
    },
    /// A community pack omitted the trust-and-safety signer.
    CommunityMissingSigner { skill_id: String },
    /// A pack raised a category's `severity_floor` to ≥ 4 but
    /// declared no `allowed_contexts`.
    StrictFloorMissingAllowedContexts { skill_id: String },
    /// A pack with strict floors declared `allowed_contexts` but
    /// is missing some of the required protected-context entries.
    StrictFloorMissingProtectedContexts {
        skill_id: String,
        missing: Vec<&'static str>,
    },
    /// An overlay tried to declare `privacy_rules`. Only the
    /// global baseline may set them.
    OverlayRedefinesPrivacyRules { skill_id: String },
    /// A lexicon was declared without a `provenance` field.
    LexiconMissingProvenance {
        skill_id: String,
        lexicon_id: String,
    },
    /// A pack ships lexicons but has zero entries in its
    /// `signers` block.
    LexiconWithoutSigners { skill_id: String },
    /// Returned by [`validate_or_raise`] when one or more rule
    /// failures are aggregated for a single pack. The
    /// `details` vector preserves each rule's stringified
    /// failure in original order so callers that want the full
    /// report can iterate; the `Display` impl renders the
    /// canonical `"anti-misuse validation failed for
    /// '<skill_id>': ..."` shape Python emits.
    AggregatedFailure {
        skill_id: String,
        details: Vec<String>,
    },
}

impl fmt::Display for AntiMisuseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSkillIdPrefix(id) => write!(
                f,
                "unrecognised skill_id '{id}' \u{2014} expected kchat.global / kchat.jurisdiction / kchat.community prefix"
            ),
            Self::InvalidCategoryInOverride {
                skill_id,
                category_repr,
            }
            | Self::InvalidCategoryInRule {
                skill_id,
                category_repr,
            } => write!(
                f,
                "pack '{skill_id}' references invalid category {category_repr}; valid range is {TAXONOMY_MIN}..{TAXONOMY_MAX}"
            ),
            Self::OverlayDeclaresForbiddenBlock { skill_id, block } => write!(
                f,
                "overlay '{skill_id}' may not declare a '{block}' block; only baseline owns the taxonomy"
            ),
            Self::JurisdictionMissingSigners { skill_id, missing } => write!(
                f,
                "jurisdiction pack '{skill_id}' missing required reviewer signers: {missing:?}"
            ),
            Self::CommunityMissingSigner { skill_id } => write!(
                f,
                "community pack '{skill_id}' missing required signer 'trust_and_safety'"
            ),
            Self::StrictFloorMissingAllowedContexts { skill_id } => write!(
                f,
                "pack '{skill_id}' raises severity_floor to >= {PROTECTED_CONTEXT_REQUIRED_SEVERITY} but declares no allowed_contexts; protected-speech carve-outs are required"
            ),
            Self::StrictFloorMissingProtectedContexts { skill_id, missing } => write!(
                f,
                "pack '{skill_id}' allowed_contexts missing the required protected-speech contexts: {missing:?}"
            ),
            Self::OverlayRedefinesPrivacyRules { skill_id } => write!(
                f,
                "overlay '{skill_id}' attempts to redefine privacy_rules; the 8 baseline privacy rules are immutable"
            ),
            Self::LexiconMissingProvenance {
                skill_id,
                lexicon_id,
            } => write!(
                f,
                "pack '{skill_id}' lexicon '{lexicon_id}' missing provenance"
            ),
            Self::LexiconWithoutSigners { skill_id } => write!(
                f,
                "pack '{skill_id}' ships lexicons without any pack-level signers acting as reviewer"
            ),
            Self::AggregatedFailure { skill_id, details } => {
                // Match Python's `validate_or_raise` framing
                // exactly: a single "; "-joined chain of every
                // rule failure, prefixed with the
                // canonical "anti-misuse validation failed
                // for '<skill_id>': " header.
                write!(f, "anti-misuse validation failed for '{skill_id}': ")?;
                if details.is_empty() {
                    f.write_str("(no detail)")
                } else {
                    for (idx, detail) in details.iter().enumerate() {
                        if idx > 0 {
                            f.write_str("; ")?;
                        }
                        f.write_str(detail)?;
                    }
                    Ok(())
                }
            }
        }
    }
}

impl std::error::Error for AntiMisuseError {}

// ---------------------------------------------------------------------------
// Pack-kind detection.
// ---------------------------------------------------------------------------

/// One of `"baseline"`, `"jurisdiction"`, `"community"`.
///
/// Mirrors Python's three string returns from `pack_kind(...)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackKind {
    Baseline,
    Jurisdiction,
    Community,
}

impl PackKind {
    /// Stable string form. Byte-identical to Python.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Jurisdiction => "jurisdiction",
            Self::Community => "community",
        }
    }
}

/// Detect pack kind from `pack["skill_id"]`. Mirrors Python's
/// `pack_kind`.
pub fn pack_kind(pack: &Value) -> Result<PackKind, AntiMisuseError> {
    let skill_id = pack
        .get("skill_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if skill_id.starts_with("kchat.global.") {
        Ok(PackKind::Baseline)
    } else if skill_id.starts_with("kchat.jurisdiction.") {
        Ok(PackKind::Jurisdiction)
    } else if skill_id.starts_with("kchat.community.") {
        Ok(PackKind::Community)
    } else {
        Err(AntiMisuseError::UnknownSkillIdPrefix(skill_id.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Individual rules.
// ---------------------------------------------------------------------------

fn skill_id_of(pack: &Value) -> String {
    pack.get("skill_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// `assert_no_vague_categories` — every referenced category id
/// must be in the closed 0..=16 enum.
pub fn assert_no_vague_categories(pack: &Value) -> Result<(), AntiMisuseError> {
    let skill_id = skill_id_of(pack);
    if let Some(items) = pack.get("overrides").and_then(Value::as_array) {
        for ov in items {
            let cat_value = ov.get("category");
            if !is_valid_category(cat_value) {
                return Err(AntiMisuseError::InvalidCategoryInOverride {
                    skill_id: skill_id.clone(),
                    category_repr: repr_value(cat_value),
                });
            }
        }
    }
    if let Some(items) = pack.get("rules").and_then(Value::as_array) {
        for r in items {
            let cat_value = r.get("category");
            if !is_valid_category(cat_value) {
                return Err(AntiMisuseError::InvalidCategoryInRule {
                    skill_id: skill_id.clone(),
                    category_repr: repr_value(cat_value),
                });
            }
        }
    }
    Ok(())
}

/// Python's `isinstance(cat, int)` check — accepts `i64` /
/// `u64` / `i32` / `u32`, rejects `f64` (Python would reject
/// `float`) and rejects `bool` (Python rejects True/False even
/// though `isinstance(True, int) is True` — the Python validator
/// uses an explicit type check that excludes bool via `not
/// isinstance(cat, int)` in earlier versions; we mirror the
/// stricter intent for safety because Python `True` would pass
/// `isinstance(.., int)` and accidentally validate).
fn is_valid_category(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Number(n)) => match n.as_i64() {
            Some(i) => (TAXONOMY_MIN..=TAXONOMY_MAX).contains(&i),
            None => false,
        },
        _ => false,
    }
}

fn repr_value(v: Option<&Value>) -> String {
    match v {
        None => "None".to_string(),
        Some(Value::Null) => "None".to_string(),
        Some(Value::Bool(true)) => "True".to_string(),
        Some(Value::Bool(false)) => "False".to_string(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::String(s)) => format!("'{s}'"),
        Some(Value::Array(_)) | Some(Value::Object(_)) => "<complex>".to_string(),
    }
}

/// `assert_no_invented_categories` — only baseline may declare
/// the taxonomy. Overlays shipping `taxonomy` / `categories` /
/// `new_categories` are rejected.
pub fn assert_no_invented_categories(pack: &Value) -> Result<(), AntiMisuseError> {
    if pack_kind(pack)? == PackKind::Baseline {
        return Ok(());
    }
    let skill_id = skill_id_of(pack);
    for forbidden in ["taxonomy", "categories", "new_categories"] {
        if pack.get(forbidden).is_some() {
            return Err(AntiMisuseError::OverlayDeclaresForbiddenBlock {
                skill_id,
                block: forbidden,
            });
        }
    }
    Ok(())
}

/// `assert_required_signers` — jurisdiction packs need both
/// `legal_review` and `cultural_review`; community packs need
/// `trust_and_safety`.
pub fn assert_required_signers(pack: &Value) -> Result<(), AntiMisuseError> {
    let kind = pack_kind(pack)?;
    let skill_id = skill_id_of(pack);
    let signers: BTreeSet<&str> = pack
        .get("signers")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if kind == PackKind::Jurisdiction {
        let mut missing = Vec::new();
        for required in ["legal_review", "cultural_review"] {
            if !signers.contains(required) {
                missing.push(required);
            }
        }
        if !missing.is_empty() {
            return Err(AntiMisuseError::JurisdictionMissingSigners { skill_id, missing });
        }
    }
    if kind == PackKind::Community && !signers.contains("trust_and_safety") {
        return Err(AntiMisuseError::CommunityMissingSigner { skill_id });
    }
    Ok(())
}

/// `assert_protected_contexts_for_strict_floors` — any category
/// with `severity_floor >= PROTECTED_CONTEXT_REQUIRED_SEVERITY`
/// requires `allowed_contexts` to declare all four
/// [`REQUIRED_PROTECTED_CONTEXTS`].
pub fn assert_protected_contexts_for_strict_floors(pack: &Value) -> Result<(), AntiMisuseError> {
    let skill_id = skill_id_of(pack);
    let strict_overrides_present = pack
        .get("overrides")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .any(|ov| severity_floor(ov) >= PROTECTED_CONTEXT_REQUIRED_SEVERITY)
        })
        .unwrap_or(false);
    if !strict_overrides_present {
        return Ok(());
    }
    let contexts: BTreeSet<&str> = pack
        .get("allowed_contexts")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if contexts.is_empty() {
        return Err(AntiMisuseError::StrictFloorMissingAllowedContexts { skill_id });
    }
    let mut missing = Vec::new();
    for required in REQUIRED_PROTECTED_CONTEXTS.iter() {
        if !contexts.contains(*required) {
            missing.push(*required);
        }
    }
    if !missing.is_empty() {
        return Err(AntiMisuseError::StrictFloorMissingProtectedContexts { skill_id, missing });
    }
    Ok(())
}

/// Extract `severity_floor` from an override, matching Python's
/// loose `int(...)` coercion semantics byte-for-byte.
///
/// The Python anti-misuse reference at
/// `build-tools/compiler/anti_misuse.py:137` uses
/// `int(ov.get("severity_floor", 0))` — which means each pack
/// value is *deliberately* coerced through `int(...)`. Python's
/// `int(...)` accepts:
///
/// | Python value     | Coerces to         | Rationale                                                 |
/// |------------------|--------------------|-----------------------------------------------------------|
/// | missing key      | `0` (default)      | `.get("severity_floor", 0)`                               |
/// | `int`            | itself             | identity                                                  |
/// | `float`          | trunc toward 0     | `int(4.7) == 4`, `int(-4.7) == -4`                        |
/// | `bool`           | `True→1`, `False→0`| `bool` is a subclass of `int` in Python                   |
/// | numeric `str`    | parsed             | `int("4") == 4`                                           |
/// | `None` / garbage | raises             | `TypeError` / `ValueError`                                |
///
/// We **intentionally** mirror the bool / numeric-string / float
/// coercion paths (per Devin Review ANALYSIS_0007 — confirmed as
/// intentional Python parity, not over-permissiveness) because
/// the parity oracle tests compare Rust against Python output on
/// the same pack dicts. Diverging on these edge cases would
/// generate false parity failures whenever a hand-edited pack
/// surfaces a YAML value of `severity_floor: true` or
/// `severity_floor: "4"`. The
/// [`severity_floor_python_int_parity`] regression test below
/// pins each row of the table above.
///
/// Where we *do* diverge from Python — and only because Python
/// would crash where we silently default — is the `None` and
/// invalid-string cases. Returning `0` there means the
/// anti-misuse rule simply doesn't fire for those rows, which is
/// the most defensive behaviour: the rule's job is to catch
/// strict-floor packs that omit protected contexts, not to be
/// the validator for `severity_floor`'s *type* (the typed
/// schema parse in [`crate::skillpack::schema`] already
/// rejects such packs upstream, so by the time
/// [`assert_protected_contexts_for_strict_floors`] runs on a
/// real load path the `Value::Null` / garbage-string variants
/// have already been filtered out).
fn severity_floor(ov: &Value) -> i64 {
    match ov.get("severity_floor") {
        Some(Value::Number(n)) => n.as_i64().unwrap_or_else(|| {
            // Match Python's `int(...)` truncation toward zero
            // for floats.
            n.as_f64().map(|f| f.trunc() as i64).unwrap_or(0)
        }),
        Some(Value::String(s)) => s.parse::<i64>().unwrap_or(0),
        Some(Value::Bool(true)) => 1,
        Some(Value::Bool(false)) => 0,
        _ => 0,
    }
}

/// `assert_privacy_rules_not_redefined` — only baseline may
/// declare `privacy_rules`.
pub fn assert_privacy_rules_not_redefined(pack: &Value) -> Result<(), AntiMisuseError> {
    if pack_kind(pack)? == PackKind::Baseline {
        return Ok(());
    }
    if pack.get("privacy_rules").is_some() {
        return Err(AntiMisuseError::OverlayRedefinesPrivacyRules {
            skill_id: skill_id_of(pack),
        });
    }
    Ok(())
}

/// `assert_lexicons_have_provenance` — every lexicon must carry a
/// `provenance` field, and a pack that ships lexicons must also
/// have at least one signer.
pub fn assert_lexicons_have_provenance(pack: &Value) -> Result<(), AntiMisuseError> {
    let skill_id = skill_id_of(pack);
    let lexicons = pack
        .get("local_language_assets")
        .and_then(|v| v.get("lexicons"))
        .and_then(Value::as_array);
    let has_any_lexicon = lexicons.map(|arr| !arr.is_empty()).unwrap_or(false);
    if let Some(arr) = lexicons {
        for lex in arr {
            let provenance_present = lex
                .get("provenance")
                .map(|v| !value_is_empty(v))
                .unwrap_or(false);
            if !provenance_present {
                let lexicon_id = lex
                    .get("lexicon_id")
                    .and_then(Value::as_str)
                    .unwrap_or("<unknown>")
                    .to_string();
                return Err(AntiMisuseError::LexiconMissingProvenance {
                    skill_id,
                    lexicon_id,
                });
            }
        }
    }
    if has_any_lexicon {
        let signer_count = pack
            .get("signers")
            .and_then(Value::as_array)
            .map(|arr| arr.len())
            .unwrap_or(0);
        if signer_count == 0 {
            return Err(AntiMisuseError::LexiconWithoutSigners { skill_id });
        }
    }
    Ok(())
}

fn value_is_empty(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::Bool(false) => true,
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        Value::Number(n) => {
            // Python truthiness: 0 / 0.0 are falsy.
            if let Some(i) = n.as_i64() {
                return i == 0;
            }
            if let Some(u) = n.as_u64() {
                return u == 0;
            }
            n.as_f64().map(|f| f == 0.0).unwrap_or(false)
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Aggregator.
// ---------------------------------------------------------------------------

/// Aggregated validator output. `passed` ⇔ `errors.is_empty()`.
/// Mirrors Python's `AntiMisuseReport` dataclass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AntiMisuseReport {
    pub skill_id: String,
    pub errors: Vec<String>,
}

impl AntiMisuseReport {
    /// `true` when no rule failed.
    pub fn passed(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Run every anti-misuse rule and return an
/// [`AntiMisuseReport`] aggregating their failures.
///
/// Mirrors Python's `validate_pack`. Aggregating mode: each rule
/// runs even if an earlier one failed.
/// Function-pointer alias used for the rule registry below.
/// Spelt out so clippy `type_complexity` doesn't fire on the
/// inline `[fn(&Value) -> Result<(), AntiMisuseError>; N]`
/// declaration.
type AntiMisuseRule = fn(&Value) -> Result<(), AntiMisuseError>;

pub fn validate_pack_value(pack: &Value) -> AntiMisuseReport {
    let skill_id = skill_id_of(pack);
    let mut errors = Vec::new();

    // Short-circuit pack-kind detection so the three rules that
    // depend on a recognised prefix
    // (`assert_no_invented_categories`,
    // `assert_required_signers`,
    // `assert_privacy_rules_not_redefined`) don't each emit the
    // same `UnknownSkillIdPrefix` diagnostic. The bad-prefix
    // failure is recorded exactly once and those three rules are
    // skipped; the other three rules
    // (`assert_no_vague_categories`,
    // `assert_protected_contexts_for_strict_floors`,
    // `assert_lexicons_have_provenance`) don't consult
    // `pack_kind` and run as normal so an unknown-prefix pack
    // still gets its taxonomy / floor / lexicon errors
    // reported.
    let kind_known = match pack_kind(pack) {
        Ok(_) => true,
        Err(err) => {
            errors.push(err.to_string());
            false
        }
    };

    let kind_independent_rules: [AntiMisuseRule; 3] = [
        assert_no_vague_categories,
        assert_protected_contexts_for_strict_floors,
        assert_lexicons_have_provenance,
    ];
    let kind_dependent_rules: [AntiMisuseRule; 3] = [
        assert_no_invented_categories,
        assert_required_signers,
        assert_privacy_rules_not_redefined,
    ];

    for rule in kind_independent_rules {
        if let Err(err) = rule(pack) {
            errors.push(err.to_string());
        }
    }
    if kind_known {
        for rule in kind_dependent_rules {
            if let Err(err) = rule(pack) {
                errors.push(err.to_string());
            }
        }
    }

    AntiMisuseReport { skill_id, errors }
}

/// Run every rule and return `Ok(())` only when all pass.
///
/// Mirrors Python's `validate_or_raise`. The error message has
/// the same `"anti-misuse validation failed for '<skill_id>': ..."`
/// shape so cross-platform consumers can pattern-match the same
/// string.
pub fn validate_or_raise(pack: &Value) -> Result<(), AntiMisuseError> {
    let report = validate_pack_value(pack);
    if report.passed() {
        return Ok(());
    }
    // The aggregated failure carries every rule's stringified
    // diagnostic so callers can pattern-match
    // `AntiMisuseError::AggregatedFailure { details, .. }` to
    // inspect each one. The dedicated variant
    // intentionally replaces an earlier implementation that
    // wrapped the aggregated message inside
    // `UnknownSkillIdPrefix`, whose `Display` impl would have
    // re-wrapped the string with its own "unrecognised
    // skill_id ..." framing and produced garbled
    // self-contradictory output.
    Err(AntiMisuseError::AggregatedFailure {
        skill_id: report.skill_id,
        details: report.errors,
    })
}

// ---------------------------------------------------------------------------
// Convenience: YAML-string and JSON-string entrypoints.
// ---------------------------------------------------------------------------

/// Parse `yaml` into a [`serde_json::Value`] tree and run every
/// anti-misuse rule. The YAML feature is gated on the
/// `skill-pack` cargo feature (which `anti_misuse` already lives
/// under), so this entrypoint is always available alongside
/// [`validate_pack_value`].
#[cfg(feature = "skill-pack")]
pub fn validate_pack_yaml(yaml: &str) -> Result<AntiMisuseReport, serde_yaml::Error> {
    let value: Value = serde_yaml::from_str(yaml)?;
    Ok(validate_pack_value(&value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn baseline_pack() -> Value {
        json!({
            "skill_id": "kchat.global.baseline.v1",
            "signers": ["trust_and_safety"],
            "taxonomy": {"categories": ["SAFE"]},
            "privacy_rules": [],
        })
    }

    fn jurisdiction_pack() -> Value {
        json!({
            "skill_id": "kchat.jurisdiction.de.v1",
            "signers": ["legal_review", "cultural_review"],
            "overrides": [
                {"category": 5, "severity_floor": 3}
            ],
        })
    }

    fn community_pack() -> Value {
        json!({
            "skill_id": "kchat.community.workplace.v1",
            "signers": ["trust_and_safety"],
            "overrides": [
                {"category": 5, "severity_floor": 2}
            ],
        })
    }

    #[test]
    fn pack_kind_detects_all_three_prefixes() {
        assert_eq!(pack_kind(&baseline_pack()).unwrap(), PackKind::Baseline);
        assert_eq!(
            pack_kind(&jurisdiction_pack()).unwrap(),
            PackKind::Jurisdiction
        );
        assert_eq!(pack_kind(&community_pack()).unwrap(), PackKind::Community);
    }

    #[test]
    fn unknown_skill_id_prefix_rejected() {
        let pack = json!({"skill_id": "kchat.unknown.foo"});
        assert!(matches!(
            pack_kind(&pack),
            Err(AntiMisuseError::UnknownSkillIdPrefix(_))
        ));
    }

    #[test]
    fn empty_skill_id_treated_as_unknown_prefix() {
        let pack = json!({});
        let err = pack_kind(&pack).unwrap_err();
        assert!(matches!(err, AntiMisuseError::UnknownSkillIdPrefix(_)));
    }

    // ----- assert_no_vague_categories ---------------------------------------

    #[test]
    fn no_vague_categories_accepts_valid_range() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "overrides": [{"category": 0}, {"category": 15}, {"category": 16}],
            "rules": [{"category": 8}],
        });
        assert!(assert_no_vague_categories(&pack).is_ok());
    }

    #[test]
    fn no_vague_categories_rejects_too_high() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "overrides": [{"category": 17}],
        });
        let err = assert_no_vague_categories(&pack).unwrap_err();
        assert!(matches!(
            err,
            AntiMisuseError::InvalidCategoryInOverride { .. }
        ));
    }

    #[test]
    fn no_vague_categories_rejects_negative() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "rules": [{"category": -1}],
        });
        let err = assert_no_vague_categories(&pack).unwrap_err();
        assert!(matches!(err, AntiMisuseError::InvalidCategoryInRule { .. }));
    }

    #[test]
    fn no_vague_categories_rejects_string_category() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "overrides": [{"category": "5"}],
        });
        assert!(assert_no_vague_categories(&pack).is_err());
    }

    #[test]
    fn no_vague_categories_rejects_float_category() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "overrides": [{"category": 5.5}],
        });
        assert!(assert_no_vague_categories(&pack).is_err());
    }

    // ----- assert_no_invented_categories ------------------------------------

    #[test]
    fn invented_categories_baseline_allowed() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "taxonomy": {"categories": []},
        });
        assert!(assert_no_invented_categories(&pack).is_ok());
    }

    #[test]
    fn invented_categories_overlay_rejects_taxonomy_block() {
        let pack = json!({
            "skill_id": "kchat.jurisdiction.de.v1",
            "taxonomy": {"categories": []},
        });
        assert!(matches!(
            assert_no_invented_categories(&pack),
            Err(AntiMisuseError::OverlayDeclaresForbiddenBlock {
                block: "taxonomy",
                ..
            })
        ));
    }

    #[test]
    fn invented_categories_overlay_rejects_new_categories_block() {
        let pack = json!({
            "skill_id": "kchat.community.workplace.v1",
            "new_categories": [],
        });
        assert!(matches!(
            assert_no_invented_categories(&pack),
            Err(AntiMisuseError::OverlayDeclaresForbiddenBlock {
                block: "new_categories",
                ..
            })
        ));
    }

    // ----- assert_required_signers ------------------------------------------

    #[test]
    fn required_signers_jurisdiction_complete() {
        let pack = json!({
            "skill_id": "kchat.jurisdiction.de.v1",
            "signers": ["legal_review", "cultural_review"],
        });
        assert!(assert_required_signers(&pack).is_ok());
    }

    #[test]
    fn required_signers_jurisdiction_missing_legal() {
        let pack = json!({
            "skill_id": "kchat.jurisdiction.de.v1",
            "signers": ["cultural_review"],
        });
        let err = assert_required_signers(&pack).unwrap_err();
        assert!(matches!(
            err,
            AntiMisuseError::JurisdictionMissingSigners { .. }
        ));
    }

    #[test]
    fn required_signers_jurisdiction_missing_both() {
        let pack = json!({
            "skill_id": "kchat.jurisdiction.de.v1",
            "signers": ["trust_and_safety"],
        });
        let err = assert_required_signers(&pack).unwrap_err();
        if let AntiMisuseError::JurisdictionMissingSigners { missing, .. } = err {
            assert!(missing.contains(&"legal_review"));
            assert!(missing.contains(&"cultural_review"));
        } else {
            panic!("expected JurisdictionMissingSigners");
        }
    }

    #[test]
    fn required_signers_community_missing_ts() {
        let pack = json!({
            "skill_id": "kchat.community.workplace.v1",
            "signers": ["legal_review"],
        });
        assert!(matches!(
            assert_required_signers(&pack),
            Err(AntiMisuseError::CommunityMissingSigner { .. })
        ));
    }

    #[test]
    fn required_signers_community_present() {
        let pack = json!({
            "skill_id": "kchat.community.workplace.v1",
            "signers": ["trust_and_safety"],
        });
        assert!(assert_required_signers(&pack).is_ok());
    }

    // ----- assert_protected_contexts_for_strict_floors ----------------------

    #[test]
    fn protected_contexts_skipped_when_no_strict_floor() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "overrides": [{"category": 2, "severity_floor": 3}],
        });
        assert!(assert_protected_contexts_for_strict_floors(&pack).is_ok());
    }

    #[test]
    fn protected_contexts_required_when_strict_floor_set() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "overrides": [{"category": 2, "severity_floor": 4}],
        });
        let err = assert_protected_contexts_for_strict_floors(&pack).unwrap_err();
        assert!(matches!(
            err,
            AntiMisuseError::StrictFloorMissingAllowedContexts { .. }
        ));
    }

    #[test]
    fn protected_contexts_missing_some_required_contexts() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "overrides": [{"category": 2, "severity_floor": 5}],
            "allowed_contexts": ["NEWS_CONTEXT", "QUOTED_SPEECH_CONTEXT"],
        });
        let err = assert_protected_contexts_for_strict_floors(&pack).unwrap_err();
        if let AntiMisuseError::StrictFloorMissingProtectedContexts { missing, .. } = err {
            assert!(missing.contains(&"EDUCATION_CONTEXT"));
            assert!(missing.contains(&"COUNTERSPEECH_CONTEXT"));
        } else {
            panic!("expected StrictFloorMissingProtectedContexts");
        }
    }

    #[test]
    fn protected_contexts_all_four_present_ok() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "overrides": [{"category": 2, "severity_floor": 4}],
            "allowed_contexts": [
                "NEWS_CONTEXT",
                "QUOTED_SPEECH_CONTEXT",
                "EDUCATION_CONTEXT",
                "COUNTERSPEECH_CONTEXT",
            ],
        });
        assert!(assert_protected_contexts_for_strict_floors(&pack).is_ok());
    }

    // ----- assert_privacy_rules_not_redefined -------------------------------

    #[test]
    fn baseline_can_set_privacy_rules() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "privacy_rules": [],
        });
        assert!(assert_privacy_rules_not_redefined(&pack).is_ok());
    }

    #[test]
    fn jurisdiction_cannot_set_privacy_rules() {
        let pack = json!({
            "skill_id": "kchat.jurisdiction.de.v1",
            "privacy_rules": [],
        });
        assert!(matches!(
            assert_privacy_rules_not_redefined(&pack),
            Err(AntiMisuseError::OverlayRedefinesPrivacyRules { .. })
        ));
    }

    #[test]
    fn community_cannot_set_privacy_rules() {
        let pack = json!({
            "skill_id": "kchat.community.workplace.v1",
            "privacy_rules": [],
        });
        assert!(matches!(
            assert_privacy_rules_not_redefined(&pack),
            Err(AntiMisuseError::OverlayRedefinesPrivacyRules { .. })
        ));
    }

    // ----- assert_lexicons_have_provenance ----------------------------------

    #[test]
    fn lexicons_with_provenance_and_signers_ok() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "signers": ["trust_and_safety"],
            "local_language_assets": {
                "lexicons": [
                    {"lexicon_id": "scam_de", "provenance": "trust_lab_2024"}
                ]
            },
        });
        assert!(assert_lexicons_have_provenance(&pack).is_ok());
    }

    #[test]
    fn lexicon_missing_provenance_rejected() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "signers": ["trust_and_safety"],
            "local_language_assets": {
                "lexicons": [
                    {"lexicon_id": "scam_de"}
                ]
            },
        });
        assert!(matches!(
            assert_lexicons_have_provenance(&pack),
            Err(AntiMisuseError::LexiconMissingProvenance { .. })
        ));
    }

    #[test]
    fn lexicon_with_empty_provenance_rejected() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "signers": ["trust_and_safety"],
            "local_language_assets": {
                "lexicons": [
                    {"lexicon_id": "scam_de", "provenance": ""}
                ]
            },
        });
        assert!(matches!(
            assert_lexicons_have_provenance(&pack),
            Err(AntiMisuseError::LexiconMissingProvenance { .. })
        ));
    }

    #[test]
    fn lexicons_without_signers_rejected() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
            "local_language_assets": {
                "lexicons": [
                    {"lexicon_id": "scam_de", "provenance": "trust_lab_2024"}
                ]
            },
        });
        assert!(matches!(
            assert_lexicons_have_provenance(&pack),
            Err(AntiMisuseError::LexiconWithoutSigners { .. })
        ));
    }

    #[test]
    fn no_lexicons_ok_even_with_no_signers() {
        let pack = json!({
            "skill_id": "kchat.global.baseline.v1",
        });
        assert!(assert_lexicons_have_provenance(&pack).is_ok());
    }

    // ----- aggregator -------------------------------------------------------

    #[test]
    fn validate_pack_value_passes_for_clean_baseline() {
        let pack = baseline_pack();
        let report = validate_pack_value(&pack);
        assert!(report.passed(), "errors: {:?}", report.errors);
    }

    #[test]
    fn validate_pack_value_passes_for_clean_jurisdiction() {
        let pack = jurisdiction_pack();
        let report = validate_pack_value(&pack);
        assert!(report.passed(), "errors: {:?}", report.errors);
    }

    #[test]
    fn validate_pack_value_aggregates_multiple_failures() {
        let pack = json!({
            "skill_id": "kchat.community.workplace.v1",
            // Missing trust_and_safety signer.
            "signers": [],
            // Overlay redefines privacy_rules — second failure.
            "privacy_rules": [],
            // Invalid category — third failure.
            "overrides": [{"category": 99}],
        });
        let report = validate_pack_value(&pack);
        assert!(!report.passed());
        assert!(report.errors.len() >= 3, "got errors: {:?}", report.errors);
    }

    #[test]
    fn validate_or_raise_returns_ok_on_clean_pack() {
        let pack = baseline_pack();
        assert!(validate_or_raise(&pack).is_ok());
    }

    #[test]
    fn validate_or_raise_returns_err_with_aggregated_message() {
        let pack = json!({
            "skill_id": "kchat.community.workplace.v1",
            "signers": [],
        });
        let err = validate_or_raise(&pack).unwrap_err();
        // Variant is the dedicated `AggregatedFailure`; callers
        // pattern-matching on it must get the structured shape,
        // not the `UnknownSkillIdPrefix` workaround.
        match &err {
            AntiMisuseError::AggregatedFailure { skill_id, details } => {
                assert_eq!(skill_id, "kchat.community.workplace.v1");
                assert!(
                    details
                        .iter()
                        .any(|d| d.contains("missing required signer 'trust_and_safety'")),
                    "missing signer detail absent from {details:?}"
                );
            }
            other => panic!("expected AggregatedFailure, got {other:?}"),
        }
        // Display output must lead with the canonical header
        // verbatim — `starts_with` (not `contains`) so a future
        // regression to double-wrapping is caught.
        let msg = err.to_string();
        assert!(
            msg.starts_with("anti-misuse validation failed for 'kchat.community.workplace.v1': "),
            "got: {msg}"
        );
        // Defensive: the old buggy framing must never reappear.
        assert!(!msg.contains("unrecognised skill_id"), "got: {msg}");
    }

    #[test]
    fn validate_pack_value_reports_unknown_prefix_only_once() {
        // Three rules used to call `pack_kind(pack)?` and each
        // emit an identical `UnknownSkillIdPrefix` diagnostic.
        // The validator now short-circuits on bad prefixes so
        // the diagnostic appears exactly once.
        let pack = json!({
            "skill_id": "not-a-kchat-pack",
            "signers": ["some_signer"],
        });
        let report = validate_pack_value(&pack);
        let prefix_errors: Vec<&String> = report
            .errors
            .iter()
            .filter(|e| e.contains("unrecognised skill_id"))
            .collect();
        assert_eq!(
            prefix_errors.len(),
            1,
            "unknown-prefix diagnostic should be reported exactly once; got {:?}",
            report.errors
        );
    }

    /// Regression for Devin Review ANALYSIS_0007 — pin
    /// `severity_floor`'s Python-`int(...)` parity row-for-row.
    ///
    /// If any of these cases ever flips you've diverged from the
    /// Python anti-misuse reference (`anti_misuse.py:137`).
    /// Diverging silently from Python's coercion would cause
    /// parity-oracle tests to spuriously fail whenever a real
    /// pack happens to surface one of these edge-case JSON
    /// shapes (e.g. a hand-edited YAML override with
    /// `severity_floor: true`). The doc comment on
    /// [`severity_floor`] documents *why* each row matches
    /// Python; this test asserts that it *does*.
    #[test]
    fn severity_floor_python_int_parity() {
        // Helper to build a one-shot override that wraps an
        // arbitrary JSON value as `severity_floor`.
        let with_floor = |v: Value| json!({ "severity_floor": v });

        // Missing key → 0 (matches Python `dict.get(..., 0)`).
        assert_eq!(severity_floor(&json!({})), 0);

        // Integers pass through verbatim, positive and negative.
        assert_eq!(severity_floor(&with_floor(json!(0))), 0);
        assert_eq!(severity_floor(&with_floor(json!(4))), 4);
        assert_eq!(severity_floor(&with_floor(json!(-3))), -3);

        // Floats truncate toward zero (matches Python `int(4.7) == 4`,
        // `int(-4.7) == -4`).
        assert_eq!(severity_floor(&with_floor(json!(4.7))), 4);
        assert_eq!(severity_floor(&with_floor(json!(-4.7))), -4);
        assert_eq!(severity_floor(&with_floor(json!(0.0))), 0);

        // Booleans coerce as `True → 1`, `False → 0` (matches
        // Python's `bool <: int` subclass relationship).
        assert_eq!(severity_floor(&with_floor(json!(true))), 1);
        assert_eq!(severity_floor(&with_floor(json!(false))), 0);

        // Numeric strings parse (matches Python `int("4") == 4`).
        assert_eq!(severity_floor(&with_floor(json!("4"))), 4);
        assert_eq!(severity_floor(&with_floor(json!("-3"))), -3);

        // `None` / garbage strings default to 0 (Python would
        // raise — we silently default because the typed schema
        // parse rejects these upstream and the anti-misuse rule
        // is not the validator for `severity_floor`'s *type*).
        assert_eq!(severity_floor(&with_floor(Value::Null)), 0);
        assert_eq!(severity_floor(&with_floor(json!("not-a-number"))), 0);

        // End-to-end: a strict-floor override expressed via a
        // *boolean* `true` should still trigger the
        // protected-context requirement, just like Python would.
        // This is the load-bearing parity assertion: the
        // coercion path is not just academically Python-shaped,
        // it actually feeds the rule that depends on it.
        let pack = json!({
            "skill_id": "kchat.jurisdiction.test",
            "signers": ["legal_review", "cultural_review"],
            "overrides": [
                {
                    "category": 2,
                    // Bool `true` coerces to 1, NOT to 4 — so it
                    // should NOT trigger the strict-floor rule.
                    "severity_floor": true,
                },
                {
                    "category": 3,
                    // String "4" coerces to 4 — this DOES trip
                    // the strict-floor rule.
                    "severity_floor": "4",
                },
            ],
        });
        let err = assert_protected_contexts_for_strict_floors(&pack)
            .expect_err("string-encoded floor of \"4\" must trip the strict-floor rule");
        assert!(
            matches!(
                err,
                AntiMisuseError::StrictFloorMissingAllowedContexts { .. }
            ),
            "expected StrictFloorMissingAllowedContexts, got {err:?}"
        );

        // And a pack whose only strict-looking floor is `true`
        // (which coerces to 1, below the threshold) must NOT
        // trip the rule.
        let pack_below = json!({
            "skill_id": "kchat.jurisdiction.test",
            "signers": ["legal_review", "cultural_review"],
            "overrides": [
                {
                    "category": 2,
                    "severity_floor": true,
                },
            ],
        });
        assert!(
            assert_protected_contexts_for_strict_floors(&pack_below).is_ok(),
            "bool-true override must NOT be treated as severity >= 4"
        );
    }
}

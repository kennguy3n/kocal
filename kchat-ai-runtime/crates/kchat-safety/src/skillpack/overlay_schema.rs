//! Closed-shape types for the community + jurisdiction overlay
//! formats.
//!
//! An *overlay* is a partial, signed YAML/JSON document that
//! modifies an already-loaded [`super::SkillPack`] at decision
//! time. Two flavours:
//!
//! * [`CommunityOverlay`] (community preference — e.g. an
//!   organisation's house policy, a group's house rules).
//! * [`JurisdictionOverlay`] (legal requirement — e.g. a country's
//!   regulatory floor).
//!
//! Both flavours share the same machinery; the only difference is
//! their declared `overlay_kind` and the `.overlay.community.` /
//! `.overlay.jurisdiction.` token in their `overlay_id`. The
//! resolver ([`super::overlay_resolver::resolve_effective_pack`])
//! applies jurisdiction overlays first so a legal requirement
//! always wins over a community preference.
//!
//! Overlays are deliberately narrow:
//!
//! * `threshold_overrides` — tighten / loosen `(category, label)`
//!   triggers, with the [`super::overlay::PROTECTED_CATEGORIES`]
//!   floor enforced inside [`super::overlay::apply_community_overlay`]
//!   so a community / jurisdiction can never weaken child safety.
//! * `severity_overrides` — change the named fields of one row of
//!   the severity rubric.
//! * `scam_phrase_additions` / `hate_lexicon_additions` — append
//!   extra entries onto a base lexicon or introduce a brand-new
//!   one.
//! * `regex_additions` — replace / add a named regex set.
//! * `slm_prompt_suffix` — text appended to the base SLM prompt,
//!   bounded in length + character set so an overlay can't smuggle
//!   prompt-injection payloads.
//!
//! Mirrors cv-guard's
//! [`shared/skillpack/overlay_schema.py`](https://github.com/kennguy3n/cv-guard)
//! one-for-one. Pydantic's `extra="forbid"` is implemented here
//! via `#[serde(deny_unknown_fields)]`; the model_validator
//! cross-field checks live in the [`*::validate`] methods.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::schema::{Lexicon, LexiconEntry, RegexPattern, RegexSet};
use super::SkillPackError;

/// Snake-word convention for the namespace + name segments inside
/// an `overlay_id` or lexicon / regex `key`. Matches Python's
/// `^[a-z][a-z0-9_]*$` regex.
fn is_valid_snake_word(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().expect("len > 0");
    if !first.is_ascii_lowercase() {
        return false;
    }
    for c in chars {
        if !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            return false;
        }
    }
    true
}

/// Validate that `s` matches the overlay-id pattern
/// `^[a-z][a-z0-9_]*\.overlay\.(community|jurisdiction)\.[a-z][a-z0-9_]*\.v\d+$`.
/// Returns the matched kind (`"community"` or `"jurisdiction"`).
///
/// Hand-rolled to avoid a circular dep on the `regex` crate (which
/// only ships under `text-pipeline`).
fn parse_overlay_id(s: &str) -> Option<&'static str> {
    let mut iter = s.split('.');
    let ns = iter.next()?;
    if !is_valid_snake_word(ns) {
        return None;
    }
    if iter.next()? != "overlay" {
        return None;
    }
    let kind = match iter.next()? {
        "community" => "community",
        "jurisdiction" => "jurisdiction",
        _ => return None,
    };
    let name = iter.next()?;
    if !is_valid_snake_word(name) {
        return None;
    }
    let version = iter.next()?;
    let v_rest = version.strip_prefix('v')?;
    if v_rest.is_empty() || !v_rest.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if iter.next().is_some() {
        return None;
    }
    Some(kind)
}

/// Validate that `s` matches the language-code shape
/// `^[a-z]{2,3}$` (ISO-639-1 / ISO-639-3 lowercase).
fn is_valid_language_code(s: &str) -> bool {
    matches!(s.len(), 2 | 3) && s.chars().all(|c| c.is_ascii_lowercase())
}

/// Bound on the SLM-prompt-suffix length an overlay can append.
///
/// 2 KiB leaves room for legitimate locale-specific guidance
/// (e.g. "in this jurisdiction, lèse-majesté is severity 5")
/// without crowding out the base prompt + SIGNALS blob in a
/// 1536-token context. Mirrors `MAX_SLM_PROMPT_SUFFIX_CHARS` in
/// the Python reference.
pub const MAX_SLM_PROMPT_SUFFIX_CHARS: usize = 2048;

/// Cap on the number of entries in a single
/// [`OverlayLexiconAddition`]. Real overlays ship at most a few
/// dozen; this cap stops a malicious overlay from shipping a
/// 100-MB lexicon that bloats the runtime matcher state.
pub const MAX_LEXICON_ADDITION_ENTRIES: usize = 200;

/// Cap on the character length of a single
/// [`super::LexiconEntry::phrase`] inside an overlay. Legitimate
/// phrases are well under 200 chars.
pub const MAX_LEXICON_ENTRY_PHRASE_CHARS: usize = 200;

/// Cap on the number of lexicon additions a single overlay can
/// carry (separately for scam phrases and hate lexicons).
pub const MAX_LEXICON_ADDITIONS_PER_OVERLAY: usize = 32;

/// Cap on the number of regex additions a single overlay can
/// carry.
pub const MAX_REGEX_ADDITIONS_PER_OVERLAY: usize = 16;

/// Returns `Some(offset)` for the first byte index in `s` whose
/// `char` is forbidden inside an overlay's SLM prompt suffix.
///
/// The closed forbidden set mirrors
/// `_PROMPT_SUFFIX_FORBIDDEN_CHAR` in
/// `cv-guard/shared/skillpack/overlay_schema.py`:
///
/// * **C0 / C1 controls** (`U+0000-U+001F`, `U+007F-U+009F`) except
///   tab (`U+0009`) and line feed (`U+000A`). Carriage return is
///   forbidden — the old allowlist permitted only tab + LF +
///   printable ASCII, so CR was already rejected. Allowing CR
///   would let an overlay smuggle `\r\n` line splits past loggers
///   that strip only `\n`.
/// * **Bidi controls** (LRM/RLM/LRO/RLO/PDF/LRI/RLI/FSI/PDI) at
///   `U+200E-U+200F`, `U+202A-U+202E`, `U+2066-U+2069`. Prevents
///   visual-spoofing reorderings.
/// * **Zero-width / format characters**: `U+200B` ZWSP, `U+200C`
///   ZWNJ, `U+200D` ZWJ, `U+2060` WJ, `U+FEFF` ZWNBSP, `U+00AD`
///   SOFT HYPHEN, `U+180E` MONGOLIAN VOWEL SEPARATOR.
///
/// Everything else — including every non-ASCII printable letter,
/// combining mark, CJK ideograph, and emoji — is permitted, so
/// native-language jurisdiction overlays (Thai, French,
/// Vietnamese, …) remain representable.
fn first_forbidden_prompt_suffix_char(s: &str) -> Option<(usize, char)> {
    for (offset, c) in s.char_indices() {
        let code = c as u32;
        // C0 controls (U+0000-U+001F) except TAB (0x09) and LF (0x0A).
        if code <= 0x1F && code != 0x09 && code != 0x0A {
            return Some((offset, c));
        }
        // DEL + C1 controls (U+007F-U+009F).
        if (0x7F..=0x9F).contains(&code) {
            return Some((offset, c));
        }
        // SOFT HYPHEN.
        if code == 0x00AD {
            return Some((offset, c));
        }
        // MONGOLIAN VOWEL SEPARATOR.
        if code == 0x180E {
            return Some((offset, c));
        }
        // ZWSP / ZWNJ / ZWJ / LRM / RLM.
        if (0x200B..=0x200F).contains(&code) {
            return Some((offset, c));
        }
        // LRE / RLE / PDF / LRO / RLO.
        if (0x202A..=0x202E).contains(&code) {
            return Some((offset, c));
        }
        // WJ + invisibles + bidi isolates (U+2060-U+2069).
        if (0x2060..=0x2069).contains(&code) {
            return Some((offset, c));
        }
        // BOM / ZWNBSP.
        if code == 0xFEFF {
            return Some((offset, c));
        }
    }
    None
}

/// Partial override for a single
/// [`super::policy_interpreter::ThresholdEntry`].
///
/// Both `trigger` and `severe` are optional — an overlay can
/// override only `trigger`, only `severe`, or both. `clear_severe
/// = true` instructs the merge to *erase* the severe floor for
/// this label (set `severe = None`) — needed because Pydantic
/// collapses both "severe omitted" and "severe: null" to `None`,
/// so we need an explicit flag to distinguish "keep base severe"
/// from "drop severe".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayThresholdEntry {
    #[serde(default)]
    pub trigger: Option<f64>,
    #[serde(default)]
    pub severe: Option<f64>,
    #[serde(default)]
    pub clear_severe: bool,
}

impl OverlayThresholdEntry {
    /// Validate `(trigger, severe)` are in `[0.0, 1.0]` and that
    /// `trigger <= severe` when both are present. Mirrors
    /// Python's `Field(..., ge=0.0, le=1.0)` constraints plus the
    /// `_trigger_le_severe` model_validator.
    pub fn validate(&self, path: &str) -> Result<(), SkillPackError> {
        if let Some(t) = self.trigger {
            if !t.is_finite() || !(0.0..=1.0).contains(&t) {
                return Err(SkillPackError::SchemaViolation {
                    path: path.to_string(),
                    detail: format!("overlay trigger {t} not in [0.0, 1.0]"),
                });
            }
        }
        if let Some(s) = self.severe {
            if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                return Err(SkillPackError::SchemaViolation {
                    path: path.to_string(),
                    detail: format!("overlay severe {s} not in [0.0, 1.0]"),
                });
            }
        }
        if let (Some(t), Some(s)) = (self.trigger, self.severe) {
            if t > s {
                return Err(SkillPackError::SchemaViolation {
                    path: path.to_string(),
                    detail: format!("overlay trigger {t} > severe {s}"),
                });
            }
        }
        Ok(())
    }
}

/// Allowed UX-action strings for [`OverlaySeverityLevel::ux_action`].
const ALLOWED_UX_ACTIONS: &[&str] = &["clear", "blur_tap", "pixelate", "blocked_card"];

/// Partial override for one row of the severity rubric.
///
/// Every field except `level` is optional — fields left as
/// `None` fall through to the base value at merge time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlaySeverityLevel {
    /// Target severity rank `0..=5`.
    pub level: u8,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub ux_action: Option<String>,
    #[serde(default)]
    pub allow_reveal: Option<bool>,
    #[serde(default)]
    pub allow_forward: Option<bool>,
    #[serde(default)]
    pub description: Option<String>,
}

impl OverlaySeverityLevel {
    /// Validate `level` is in `0..=5` and `ux_action` (when
    /// present) is in [`ALLOWED_UX_ACTIONS`]. Mirrors Python's
    /// `Field(..., ge=0, le=5)` + `_valid_action` field
    /// validator.
    pub fn validate(&self, path: &str) -> Result<(), SkillPackError> {
        if self.level > 5 {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!("overlay severity level {} not in 0..=5", self.level),
            });
        }
        if let Some(action) = &self.ux_action {
            if !ALLOWED_UX_ACTIONS.iter().any(|a| a == action) {
                return Err(SkillPackError::SchemaViolation {
                    path: path.to_string(),
                    detail: format!("ux_action {action:?} not in {ALLOWED_UX_ACTIONS:?}"),
                });
            }
        }
        Ok(())
    }
}

/// Extra entries appended to a base lexicon (or a brand-new
/// lexicon if `key` doesn't exist in the base pack).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayLexiconAddition {
    /// Lexicon file stem, e.g. `"en"` or `"medical_en"`. Must be
    /// snake_case ASCII.
    pub key: String,
    /// ISO-639-1 / ISO-639-3 lowercase code, e.g. `"en"`,
    /// `"jpn"`.
    pub language: String,
    /// Entries in declaration order. Capped at
    /// [`MAX_LEXICON_ADDITION_ENTRIES`]; each entry's phrase is
    /// capped at [`MAX_LEXICON_ENTRY_PHRASE_CHARS`].
    pub entries: Vec<LexiconEntry>,
}

impl OverlayLexiconAddition {
    /// Validate the addition's key shape, language shape, and
    /// entry bounds. Mirrors Python's `_bound_entry_list`
    /// validator plus the field-level `pattern=` constraints.
    pub fn validate(&self, path: &str) -> Result<(), SkillPackError> {
        if !is_valid_snake_word(&self.key) {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!(
                    "OverlayLexiconAddition.key {:?} must be snake_case ASCII (lowercase letters / digits / underscore)",
                    self.key
                ),
            });
        }
        if !is_valid_language_code(&self.language) {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!(
                    "OverlayLexiconAddition.language {:?} must be 2-3 lowercase ASCII chars",
                    self.language
                ),
            });
        }
        if self.entries.len() > MAX_LEXICON_ADDITION_ENTRIES {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!(
                    "OverlayLexiconAddition has {} entries (limit {})",
                    self.entries.len(),
                    MAX_LEXICON_ADDITION_ENTRIES
                ),
            });
        }
        for entry in &self.entries {
            entry.validate().map_err(|e| match e {
                SkillPackError::SchemaViolation { detail, .. } => SkillPackError::SchemaViolation {
                    path: path.to_string(),
                    detail,
                },
                other => other,
            })?;
            // Phrase cap is in *characters* (Python `len(str)`
            // returns chars). Counting bytes would silently reject
            // valid non-ASCII lexicon phrases the Python reference
            // accepts.
            if entry.phrase.chars().count() > MAX_LEXICON_ENTRY_PHRASE_CHARS {
                let prefix: String = entry.phrase.chars().take(32).collect();
                return Err(SkillPackError::SchemaViolation {
                    path: path.to_string(),
                    detail: format!(
                        "OverlayLexiconAddition entry {prefix:?}... exceeds {MAX_LEXICON_ENTRY_PHRASE_CHARS}-char phrase limit"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Convert this addition into a fresh [`Lexicon`] (used when
    /// the base pack does not yet carry a lexicon under
    /// [`Self::key`]).
    pub fn into_lexicon(self) -> Lexicon {
        Lexicon {
            language: self.language,
            entries: self.entries,
        }
    }
}

/// Extra named regex set added by an overlay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverlayRegexAddition {
    /// `regex/<key>.yaml` file stem.
    pub key: String,
    /// `RegexSet.name` for the new set.
    pub name: String,
    /// Patterns in declaration order.
    pub patterns: Vec<RegexPattern>,
}

impl OverlayRegexAddition {
    /// Validate the addition's key shape + the inner
    /// [`RegexSet`]. Compiles every pattern (under the
    /// `text-pipeline` feature) so a malformed regex fails the
    /// load loudly rather than at the first scan.
    pub fn validate(&self, path: &str) -> Result<(), SkillPackError> {
        if !is_valid_snake_word(&self.key) {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!(
                    "OverlayRegexAddition.key {:?} must be snake_case ASCII",
                    self.key
                ),
            });
        }
        // Reuse the canonical RegexSet validator: it already
        // checks the set name + every pattern's name + flag set,
        // and compiles each pattern through the regex crate
        // (under the `text-pipeline` feature).
        self.to_regex_set().validate(path)
    }

    /// Convert this addition into a fresh [`RegexSet`].
    pub fn to_regex_set(&self) -> RegexSet {
        RegexSet {
            name: self.name.clone(),
            patterns: self.patterns.clone(),
        }
    }
}

/// Internal — shared validation entrypoint for both overlay
/// kinds. Exposed at module scope so tests can hit it directly.
fn validate_overlay_fields<O: OverlayCommon>(
    overlay: &O,
    expected_kind: &str,
    path: &str,
) -> Result<(), SkillPackError> {
    // overlay_id shape + kind match.
    let parsed_kind = parse_overlay_id(overlay.overlay_id()).ok_or_else(|| {
        SkillPackError::SchemaViolation {
            path: path.to_string(),
            detail: format!(
                "invalid overlay_id {:?}: expected <ns>.overlay.(community|jurisdiction).<name>.v<n>",
                overlay.overlay_id()
            ),
        }
    })?;
    if parsed_kind != expected_kind {
        return Err(SkillPackError::SchemaViolation {
            path: path.to_string(),
            detail: format!(
                "{expected_kind} overlay_id {:?} must contain '.overlay.{expected_kind}.'",
                overlay.overlay_id()
            ),
        });
    }
    if overlay.schema_version() < 1 {
        return Err(SkillPackError::SchemaViolation {
            path: path.to_string(),
            detail: format!(
                "schema_version must be >= 1 (got {})",
                overlay.schema_version()
            ),
        });
    }

    // slm_prompt_suffix length + character set. The cap is in
    // *characters* (Python `len(str)` returns chars), not bytes, so
    // a Thai / CJK / accented-Latin overlay can use the full 2 KiB
    // budget. Counting bytes here would silently reject overlays
    // the Python reference accepts and break cross-platform parity.
    let suffix = overlay.slm_prompt_suffix();
    let suffix_chars = suffix.chars().count();
    if suffix_chars > MAX_SLM_PROMPT_SUFFIX_CHARS {
        return Err(SkillPackError::SchemaViolation {
            path: path.to_string(),
            detail: format!(
                "slm_prompt_suffix is {suffix_chars} chars (limit {MAX_SLM_PROMPT_SUFFIX_CHARS})"
            ),
        });
    }
    if let Some((offset, c)) = first_forbidden_prompt_suffix_char(suffix) {
        return Err(SkillPackError::SchemaViolation {
            path: path.to_string(),
            detail: format!(
                "slm_prompt_suffix contains forbidden character U+{:04X} at byte offset {offset}: control characters, bidi overrides, and zero-width format characters are not allowed (Unicode printable characters in any script are accepted)",
                c as u32
            ),
        });
    }

    // No duplicate severity_overrides levels + per-entry validation.
    let mut seen_levels = std::collections::BTreeSet::new();
    for lv in overlay.severity_overrides() {
        lv.validate(path)?;
        if !seen_levels.insert(lv.level) {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!("duplicate severity_overrides entry for level {}", lv.level),
            });
        }
    }

    // threshold_overrides per-entry validation. The merge function
    // re-checks against the base pack; here we only validate the
    // overlay-local shape (range + trigger <= severe).
    for (cat, label_map) in overlay.threshold_overrides() {
        if cat.is_empty() {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: "threshold_overrides category name must not be empty".to_string(),
            });
        }
        for (label, entry) in label_map {
            if label.is_empty() {
                return Err(SkillPackError::SchemaViolation {
                    path: path.to_string(),
                    detail: format!(
                        "threshold_overrides label name in category {cat:?} must not be empty"
                    ),
                });
            }
            entry.validate(path)?;
        }
    }

    // Addition-count caps + per-entry validation.
    if overlay.scam_phrase_additions().len() > MAX_LEXICON_ADDITIONS_PER_OVERLAY {
        return Err(SkillPackError::SchemaViolation {
            path: path.to_string(),
            detail: format!(
                "scam_phrase_additions has {} entries (limit {MAX_LEXICON_ADDITIONS_PER_OVERLAY})",
                overlay.scam_phrase_additions().len()
            ),
        });
    }
    if overlay.hate_lexicon_additions().len() > MAX_LEXICON_ADDITIONS_PER_OVERLAY {
        return Err(SkillPackError::SchemaViolation {
            path: path.to_string(),
            detail: format!(
                "hate_lexicon_additions has {} entries (limit {MAX_LEXICON_ADDITIONS_PER_OVERLAY})",
                overlay.hate_lexicon_additions().len()
            ),
        });
    }
    if overlay.regex_additions().len() > MAX_REGEX_ADDITIONS_PER_OVERLAY {
        return Err(SkillPackError::SchemaViolation {
            path: path.to_string(),
            detail: format!(
                "regex_additions has {} entries (limit {MAX_REGEX_ADDITIONS_PER_OVERLAY})",
                overlay.regex_additions().len()
            ),
        });
    }
    for add in overlay.scam_phrase_additions() {
        add.validate(path)?;
    }
    for add in overlay.hate_lexicon_additions() {
        add.validate(path)?;
    }
    for add in overlay.regex_additions() {
        add.validate(path)?;
    }
    Ok(())
}

/// Internal trait abstracting the fields shared by
/// [`CommunityOverlay`] and [`JurisdictionOverlay`] so the
/// validator can run over either. Not part of the public API.
trait OverlayCommon {
    fn overlay_id(&self) -> &str;
    fn schema_version(&self) -> u32;
    fn slm_prompt_suffix(&self) -> &str;
    fn severity_overrides(&self) -> &[OverlaySeverityLevel];
    fn threshold_overrides(&self) -> &BTreeMap<String, BTreeMap<String, OverlayThresholdEntry>>;
    fn scam_phrase_additions(&self) -> &[OverlayLexiconAddition];
    fn hate_lexicon_additions(&self) -> &[OverlayLexiconAddition];
    fn regex_additions(&self) -> &[OverlayRegexAddition];
}

fn default_schema_version() -> u32 {
    1
}

/// Community-driven overlay.
///
/// Selected by `MediaSafetyRequest.context_hints['community_type']`
/// at decision time. The resolver always layers a community
/// overlay *on top of* any active jurisdiction overlay so a
/// jurisdiction's legal requirement cannot be relaxed by a
/// community preference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommunityOverlay {
    pub overlay_id: String,
    pub version: String,
    pub base_pack_id: String,
    pub base_pack_version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub threshold_overrides: BTreeMap<String, BTreeMap<String, OverlayThresholdEntry>>,
    #[serde(default)]
    pub severity_overrides: Vec<OverlaySeverityLevel>,
    #[serde(default)]
    pub scam_phrase_additions: Vec<OverlayLexiconAddition>,
    #[serde(default)]
    pub hate_lexicon_additions: Vec<OverlayLexiconAddition>,
    #[serde(default)]
    pub regex_additions: Vec<OverlayRegexAddition>,
    #[serde(default)]
    pub slm_prompt_suffix: String,
    /// Frozen to `"community"` for type-introspection. Defaults
    /// to the literal so YAML doesn't need to carry it.
    #[serde(default = "default_community_kind")]
    pub overlay_kind: String,
}

fn default_community_kind() -> String {
    "community".to_string()
}

fn default_jurisdiction_kind() -> String {
    "jurisdiction".to_string()
}

impl CommunityOverlay {
    /// Validate the overlay's structural invariants.
    ///
    /// `path` identifies the source (e.g. an overlay YAML's
    /// archive path or `<inline>`) and is woven into the error
    /// detail so a failure attributes back to the offending file.
    pub fn validate(&self, path: &str) -> Result<(), SkillPackError> {
        if self.overlay_kind != "community" {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!(
                    "CommunityOverlay.overlay_kind must be \"community\", got {:?}",
                    self.overlay_kind
                ),
            });
        }
        validate_overlay_fields(self, "community", path)
    }
}

impl OverlayCommon for CommunityOverlay {
    fn overlay_id(&self) -> &str {
        &self.overlay_id
    }
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
    fn slm_prompt_suffix(&self) -> &str {
        &self.slm_prompt_suffix
    }
    fn severity_overrides(&self) -> &[OverlaySeverityLevel] {
        &self.severity_overrides
    }
    fn threshold_overrides(&self) -> &BTreeMap<String, BTreeMap<String, OverlayThresholdEntry>> {
        &self.threshold_overrides
    }
    fn scam_phrase_additions(&self) -> &[OverlayLexiconAddition] {
        &self.scam_phrase_additions
    }
    fn hate_lexicon_additions(&self) -> &[OverlayLexiconAddition] {
        &self.hate_lexicon_additions
    }
    fn regex_additions(&self) -> &[OverlayRegexAddition] {
        &self.regex_additions
    }
}

/// Jurisdiction-driven overlay.
///
/// Selected by `MediaSafetyRequest.context_hints['jurisdiction']`
/// at decision time. Applied *before* any community overlay so a
/// jurisdiction's legal requirement always takes precedence over
/// a community preference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JurisdictionOverlay {
    pub overlay_id: String,
    pub version: String,
    pub base_pack_id: String,
    pub base_pack_version: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    #[serde(default)]
    pub threshold_overrides: BTreeMap<String, BTreeMap<String, OverlayThresholdEntry>>,
    #[serde(default)]
    pub severity_overrides: Vec<OverlaySeverityLevel>,
    #[serde(default)]
    pub scam_phrase_additions: Vec<OverlayLexiconAddition>,
    #[serde(default)]
    pub hate_lexicon_additions: Vec<OverlayLexiconAddition>,
    #[serde(default)]
    pub regex_additions: Vec<OverlayRegexAddition>,
    #[serde(default)]
    pub slm_prompt_suffix: String,
    #[serde(default = "default_jurisdiction_kind")]
    pub overlay_kind: String,
}

impl JurisdictionOverlay {
    /// Validate the overlay's structural invariants. See
    /// [`CommunityOverlay::validate`] for details.
    pub fn validate(&self, path: &str) -> Result<(), SkillPackError> {
        if self.overlay_kind != "jurisdiction" {
            return Err(SkillPackError::SchemaViolation {
                path: path.to_string(),
                detail: format!(
                    "JurisdictionOverlay.overlay_kind must be \"jurisdiction\", got {:?}",
                    self.overlay_kind
                ),
            });
        }
        validate_overlay_fields(self, "jurisdiction", path)
    }
}

impl OverlayCommon for JurisdictionOverlay {
    fn overlay_id(&self) -> &str {
        &self.overlay_id
    }
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
    fn slm_prompt_suffix(&self) -> &str {
        &self.slm_prompt_suffix
    }
    fn severity_overrides(&self) -> &[OverlaySeverityLevel] {
        &self.severity_overrides
    }
    fn threshold_overrides(&self) -> &BTreeMap<String, BTreeMap<String, OverlayThresholdEntry>> {
        &self.threshold_overrides
    }
    fn scam_phrase_additions(&self) -> &[OverlayLexiconAddition] {
        &self.scam_phrase_additions
    }
    fn hate_lexicon_additions(&self) -> &[OverlayLexiconAddition] {
        &self.hate_lexicon_additions
    }
    fn regex_additions(&self) -> &[OverlayRegexAddition] {
        &self.regex_additions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn minimal_community() -> CommunityOverlay {
        CommunityOverlay {
            overlay_id: "ns.overlay.community.test.v1".to_string(),
            version: "1.0.0".to_string(),
            base_pack_id: "ns.skill.base.v1".to_string(),
            base_pack_version: "1.0.0".to_string(),
            description: String::new(),
            schema_version: 1,
            threshold_overrides: BTreeMap::new(),
            severity_overrides: Vec::new(),
            scam_phrase_additions: Vec::new(),
            hate_lexicon_additions: Vec::new(),
            regex_additions: Vec::new(),
            slm_prompt_suffix: String::new(),
            overlay_kind: "community".to_string(),
        }
    }

    fn minimal_jurisdiction() -> JurisdictionOverlay {
        JurisdictionOverlay {
            overlay_id: "ns.overlay.jurisdiction.test.v1".to_string(),
            version: "1.0.0".to_string(),
            base_pack_id: "ns.skill.base.v1".to_string(),
            base_pack_version: "1.0.0".to_string(),
            description: String::new(),
            schema_version: 1,
            threshold_overrides: BTreeMap::new(),
            severity_overrides: Vec::new(),
            scam_phrase_additions: Vec::new(),
            hate_lexicon_additions: Vec::new(),
            regex_additions: Vec::new(),
            slm_prompt_suffix: String::new(),
            overlay_kind: "jurisdiction".to_string(),
        }
    }

    #[test]
    fn parse_overlay_id_accepts_canonical_shapes() {
        assert_eq!(
            parse_overlay_id("ns.overlay.community.x.v1"),
            Some("community")
        );
        assert_eq!(
            parse_overlay_id("ns.overlay.jurisdiction.x.v1"),
            Some("jurisdiction")
        );
        assert_eq!(
            parse_overlay_id("cvguard.overlay.community.workplace.v42"),
            Some("community")
        );
        assert_eq!(
            parse_overlay_id("a1.overlay.jurisdiction.fr.v0"),
            Some("jurisdiction")
        );
    }

    #[test]
    fn parse_overlay_id_rejects_drift() {
        // Wrong literal.
        assert!(parse_overlay_id("ns.skill.x.v1").is_none());
        // Missing kind.
        assert!(parse_overlay_id("ns.overlay.x.v1").is_none());
        // Unknown kind.
        assert!(parse_overlay_id("ns.overlay.random.x.v1").is_none());
        // Trailing junk.
        assert!(parse_overlay_id("ns.overlay.community.x.v1.extra").is_none());
        // Bad version.
        assert!(parse_overlay_id("ns.overlay.community.x.va").is_none());
        assert!(parse_overlay_id("ns.overlay.community.x.v").is_none());
        // Uppercase rejection.
        assert!(parse_overlay_id("NS.overlay.community.x.v1").is_none());
    }

    #[test]
    fn overlay_threshold_entry_validates_range_and_ordering() {
        OverlayThresholdEntry {
            trigger: Some(0.3),
            severe: Some(0.7),
            clear_severe: false,
        }
        .validate("test")
        .unwrap();

        OverlayThresholdEntry {
            trigger: Some(0.3),
            severe: None,
            clear_severe: true,
        }
        .validate("test")
        .unwrap();

        assert!(matches!(
            OverlayThresholdEntry {
                trigger: Some(0.8),
                severe: Some(0.5),
                clear_severe: false,
            }
            .validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));

        assert!(matches!(
            OverlayThresholdEntry {
                trigger: Some(1.5),
                severe: None,
                clear_severe: false,
            }
            .validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));

        assert!(matches!(
            OverlayThresholdEntry {
                trigger: Some(f64::NAN),
                severe: None,
                clear_severe: false,
            }
            .validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn overlay_severity_level_validates_action_set() {
        for action in ["clear", "blur_tap", "pixelate", "blocked_card"] {
            OverlaySeverityLevel {
                level: 3,
                name: None,
                ux_action: Some(action.to_string()),
                allow_reveal: None,
                allow_forward: None,
                description: None,
            }
            .validate("test")
            .unwrap();
        }

        assert!(matches!(
            OverlaySeverityLevel {
                level: 3,
                name: None,
                ux_action: Some("nuke_from_orbit".to_string()),
                allow_reveal: None,
                allow_forward: None,
                description: None,
            }
            .validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));

        assert!(matches!(
            OverlaySeverityLevel {
                level: 6,
                name: None,
                ux_action: None,
                allow_reveal: None,
                allow_forward: None,
                description: None,
            }
            .validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn community_overlay_validates_minimal_shape() {
        minimal_community().validate("test").unwrap();
    }

    #[test]
    fn community_overlay_rejects_jurisdiction_id() {
        let mut o = minimal_community();
        o.overlay_id = "ns.overlay.jurisdiction.test.v1".to_string();
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn jurisdiction_overlay_rejects_community_id() {
        let mut o = minimal_jurisdiction();
        o.overlay_id = "ns.overlay.community.test.v1".to_string();
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn overlay_rejects_oversize_prompt_suffix() {
        let mut o = minimal_community();
        o.slm_prompt_suffix = "x".repeat(MAX_SLM_PROMPT_SUFFIX_CHARS + 1);
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn overlay_accepts_at_limit_prompt_suffix() {
        let mut o = minimal_community();
        // Boundary: exactly MAX is accepted.
        o.slm_prompt_suffix = "x".repeat(MAX_SLM_PROMPT_SUFFIX_CHARS);
        o.validate("test").unwrap();
    }

    #[test]
    fn overlay_accepts_unicode_prompt_suffix() {
        // Native-language regulatory prose: Thai, French,
        // Vietnamese, CJK, emoji — all must be representable.
        let mut o = minimal_community();
        o.slm_prompt_suffix =
            "ในเขตอำนาจนี้ การหมิ่นพระบรมเดชานุภาพคือระดับ 5.\nLèse-majesté.\n冒涜 — 5⃣".to_string();
        o.validate("test").unwrap();
    }

    #[test]
    fn overlay_accepts_prompt_suffix_at_char_limit_with_multibyte_chars() {
        // Regression guard for the byte-vs-char bug:
        // `MAX_SLM_PROMPT_SUFFIX_CHARS` is a *character* cap to
        // match Python's `len(str)`. A Thai prose body that fits
        // the character limit but blows past the byte budget (each
        // Thai cluster is 3 UTF-8 bytes) must be accepted, not
        // rejected. Python `len("ก" * 2048) == 2048`; Rust must
        // agree.
        let mut o = minimal_community();
        // 2048 Thai 'ko kai' characters = 6144 UTF-8 bytes. Far
        // past the byte limit; right at the char limit.
        o.slm_prompt_suffix = "ก".repeat(MAX_SLM_PROMPT_SUFFIX_CHARS);
        assert!(o.slm_prompt_suffix.len() > MAX_SLM_PROMPT_SUFFIX_CHARS);
        assert_eq!(
            o.slm_prompt_suffix.chars().count(),
            MAX_SLM_PROMPT_SUFFIX_CHARS
        );
        o.validate("test").unwrap();
    }

    #[test]
    fn overlay_rejects_prompt_suffix_one_char_over_limit_in_multibyte() {
        // Boundary: one char past the limit in multi-byte
        // characters must still be rejected. Guards against a
        // future regression where someone tightens the check back
        // to bytes — that would *accept* this case (since the
        // byte length is well below 2 KiB once you switch to a
        // single-byte test string) and let an oversized overlay
        // through.
        let mut o = minimal_community();
        o.slm_prompt_suffix = "ก".repeat(MAX_SLM_PROMPT_SUFFIX_CHARS + 1);
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn overlay_rejects_c0_controls_except_tab_and_lf() {
        let mut o = minimal_community();
        // CR is forbidden (would let \r\n smuggling past loggers
        // that only strip \n).
        o.slm_prompt_suffix = "ok\rbad".to_string();
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
        // NUL byte forbidden.
        o.slm_prompt_suffix = "ok\0bad".to_string();
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
        // ESC forbidden.
        o.slm_prompt_suffix = "ok\x1bbad".to_string();
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn overlay_accepts_tab_and_lf() {
        let mut o = minimal_community();
        o.slm_prompt_suffix = "line1\tcol2\nline2".to_string();
        o.validate("test").unwrap();
    }

    #[test]
    fn overlay_rejects_c1_controls() {
        let mut o = minimal_community();
        // U+0080 — first C1 control.
        o.slm_prompt_suffix = "ok\u{0080}bad".to_string();
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
        // U+007F DEL also in the forbidden set.
        o.slm_prompt_suffix = "ok\u{007f}bad".to_string();
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn overlay_rejects_bidi_overrides() {
        let mut o = minimal_community();
        // LRO (U+202D) — used in Trojan Source attacks.
        o.slm_prompt_suffix = "ok\u{202d}bad".to_string();
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
        // RLI (U+2067).
        o.slm_prompt_suffix = "ok\u{2067}bad".to_string();
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn overlay_rejects_zero_width_format() {
        let mut o = minimal_community();
        for cp in [
            '\u{200B}', '\u{200C}', '\u{200D}', '\u{2060}', '\u{FEFF}', '\u{00AD}', '\u{180E}',
        ] {
            o.slm_prompt_suffix = format!("ok{cp}bad");
            assert!(
                matches!(
                    o.validate("test"),
                    Err(SkillPackError::SchemaViolation { .. })
                ),
                "should reject U+{:04X}",
                cp as u32
            );
        }
    }

    #[test]
    fn overlay_rejects_duplicate_severity_overrides() {
        let mut o = minimal_community();
        o.severity_overrides = vec![
            OverlaySeverityLevel {
                level: 3,
                name: None,
                ux_action: None,
                allow_reveal: None,
                allow_forward: None,
                description: None,
            },
            OverlaySeverityLevel {
                level: 3,
                name: None,
                ux_action: None,
                allow_reveal: None,
                allow_forward: None,
                description: None,
            },
        ];
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn overlay_caps_addition_counts() {
        let mut o = minimal_community();
        let add = OverlayLexiconAddition {
            key: "en".to_string(),
            language: "en".to_string(),
            entries: vec![LexiconEntry {
                phrase: "x".to_string(),
                weight: 1.0,
                tags: Vec::new(),
            }],
        };
        o.scam_phrase_additions = vec![add.clone(); MAX_LEXICON_ADDITIONS_PER_OVERLAY + 1];
        assert!(matches!(
            o.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn overlay_lexicon_addition_caps_entries_and_phrase_length() {
        // Entry-count cap.
        let too_many = OverlayLexiconAddition {
            key: "en".to_string(),
            language: "en".to_string(),
            entries: (0..=MAX_LEXICON_ADDITION_ENTRIES)
                .map(|i| LexiconEntry {
                    phrase: format!("phrase {i}"),
                    weight: 1.0,
                    tags: Vec::new(),
                })
                .collect(),
        };
        assert!(matches!(
            too_many.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));

        // Per-phrase length cap.
        let phrase_too_long = OverlayLexiconAddition {
            key: "en".to_string(),
            language: "en".to_string(),
            entries: vec![LexiconEntry {
                phrase: "x".repeat(MAX_LEXICON_ENTRY_PHRASE_CHARS + 1),
                weight: 1.0,
                tags: Vec::new(),
            }],
        };
        assert!(matches!(
            phrase_too_long.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn overlay_lexicon_addition_phrase_cap_counts_chars_not_bytes() {
        // Regression guard: `MAX_LEXICON_ENTRY_PHRASE_CHARS` is a
        // character cap (Python `len(str)`), not a byte cap. A CJK
        // phrase that fits the character limit but exceeds the
        // byte budget must be accepted; one char past the limit
        // must still be rejected.

        // Accepted: 200 CJK characters = 600 UTF-8 bytes (well
        // past the 200 byte budget, exactly at the 200 char
        // budget).
        let at_char_limit = OverlayLexiconAddition {
            key: "zh".to_string(),
            language: "zh".to_string(),
            entries: vec![LexiconEntry {
                phrase: "中".repeat(MAX_LEXICON_ENTRY_PHRASE_CHARS),
                weight: 1.0,
                tags: Vec::new(),
            }],
        };
        assert_eq!(
            at_char_limit.entries[0].phrase.chars().count(),
            MAX_LEXICON_ENTRY_PHRASE_CHARS
        );
        assert!(at_char_limit.entries[0].phrase.len() > MAX_LEXICON_ENTRY_PHRASE_CHARS);
        at_char_limit.validate("test").unwrap();

        // Rejected: one char past the limit in multi-byte chars.
        let one_over = OverlayLexiconAddition {
            key: "zh".to_string(),
            language: "zh".to_string(),
            entries: vec![LexiconEntry {
                phrase: "中".repeat(MAX_LEXICON_ENTRY_PHRASE_CHARS + 1),
                weight: 1.0,
                tags: Vec::new(),
            }],
        };
        assert!(matches!(
            one_over.validate("test"),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn community_overlay_round_trips_through_json() {
        let mut o = minimal_community();
        o.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "nudity".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.45),
                    severe: Some(0.80),
                    clear_severe: false,
                },
            )]),
        );
        o.severity_overrides = vec![OverlaySeverityLevel {
            level: 3,
            name: Some("medium".to_string()),
            ux_action: Some("blur_tap".to_string()),
            allow_reveal: Some(true),
            allow_forward: Some(false),
            description: None,
        }];
        let s = serde_json::to_string(&o).unwrap();
        let back: CommunityOverlay = serde_json::from_str(&s).unwrap();
        assert_eq!(o, back);
        back.validate("inline").unwrap();
    }

    #[test]
    fn overlay_deserialization_rejects_unknown_fields() {
        // deny_unknown_fields path on the overlay itself.
        let payload = json!({
            "overlay_id": "ns.overlay.community.x.v1",
            "version": "1.0.0",
            "base_pack_id": "ns.skill.base.v1",
            "base_pack_version": "1.0.0",
            "overlay_kind": "community",
            "unexpected": "field",
        });
        let res: Result<CommunityOverlay, _> = serde_json::from_value(payload);
        assert!(res.is_err());
    }
}

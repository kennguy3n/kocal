//! Severity rubric and severity → UX-action mapper.
//!
//! Mirrors `cv-guard/shared/policy/severity_mapper.py` and the
//! [`SeverityRubric`] / [`SeverityLevel`] subset of
//! `cv-guard/shared/skillpack/schema.py`. Pure function: given a
//! severity rank and the active skill pack's severity rubric,
//! returns the prescribed `(ux_action, allow_reveal,
//! allow_forward)` triple.
//!
//! Stateless on purpose so the scanner can invoke the mapper from
//! rule-based fast paths *and* from the SLM post-processing path
//! without any shared object lifetime.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::decision::UXAction;

/// Single row of the severity rubric.
///
/// The renderer reads `(ux_action, allow_reveal, allow_forward)`;
/// `name` and `description` are surfaced to UI / telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeverityLevel {
    /// Severity rank `0..=5`.
    pub level: u8,
    /// Short label: `safe`, `low`, `medium`, ...
    pub name: String,
    /// UX action prescribed at this severity.
    pub ux_action: UXAction,
    /// Whether the renderer may offer a "tap to reveal" affordance.
    #[serde(default = "default_true")]
    pub allow_reveal: bool,
    /// Whether the user may forward the media at this severity.
    #[serde(default = "default_true")]
    pub allow_forward: bool,
    /// Human-readable description; never embedded in the SLM
    /// prompt.
    #[serde(default)]
    pub description: String,
}

fn default_true() -> bool {
    true
}

/// Errors raised when constructing or validating a
/// [`SeverityRubric`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SeverityRubricError {
    /// `level` was outside `0..=5`.
    LevelOutOfRange(i32),
    /// Two [`SeverityLevel`] entries shared the same `level`.
    DuplicateLevel(u8),
    /// One or more of `0..=5` was missing.
    MissingLevels(Vec<u8>),
    /// `name` was empty.
    EmptyName(u8),
}

impl fmt::Display for SeverityRubricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LevelOutOfRange(n) => write!(f, "level must be in 0..=5 (got {n})"),
            Self::DuplicateLevel(n) => write!(f, "duplicate severity level {n}"),
            Self::MissingLevels(missing) => write!(
                f,
                "SeverityRubric missing level(s): {missing:?} (must cover 0..=5)"
            ),
            Self::EmptyName(level) => {
                write!(f, "name for severity level {level} must not be empty")
            }
        }
    }
}

impl std::error::Error for SeverityRubricError {}

impl SeverityLevel {
    /// Construct a [`SeverityLevel`] with full validation.
    pub fn new(
        level: u8,
        name: impl Into<String>,
        ux_action: UXAction,
    ) -> Result<Self, SeverityRubricError> {
        if level > 5 {
            return Err(SeverityRubricError::LevelOutOfRange(level as i32));
        }
        let name = name.into();
        if name.is_empty() {
            return Err(SeverityRubricError::EmptyName(level));
        }
        Ok(Self {
            level,
            name,
            ux_action,
            allow_reveal: true,
            allow_forward: true,
            description: String::new(),
        })
    }

    /// Fluent setter for `allow_reveal`.
    #[must_use]
    pub fn with_allow_reveal(mut self, allow: bool) -> Self {
        self.allow_reveal = allow;
        self
    }

    /// Fluent setter for `allow_forward`.
    #[must_use]
    pub fn with_allow_forward(mut self, allow: bool) -> Self {
        self.allow_forward = allow;
        self
    }

    /// Fluent setter for `description`.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }
}

/// Full `0..=5` severity → UX mapping. PROPOSAL.md §9.
///
/// Validation invariants (mirror the Python `model_validator`):
///
/// * every level in `0..=5` is covered exactly once;
/// * each entry's `name` is non-empty.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeverityRubric {
    /// Skill-pack schema version; surfaced for diagnostics but not
    /// used by the mapper itself.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Per-level UX disposition. Validated to cover `0..=5`.
    pub levels: Vec<SeverityLevel>,
}

fn default_schema_version() -> u32 {
    1
}

impl SeverityRubric {
    /// Construct from a list of levels, validating coverage of
    /// `0..=5` and uniqueness.
    pub fn new(levels: Vec<SeverityLevel>) -> Result<Self, SeverityRubricError> {
        Self::with_schema_version(levels, default_schema_version())
    }

    /// Construct with an explicit `schema_version`.
    pub fn with_schema_version(
        levels: Vec<SeverityLevel>,
        schema_version: u32,
    ) -> Result<Self, SeverityRubricError> {
        let rubric = Self {
            schema_version,
            levels,
        };
        rubric.validate()?;
        Ok(rubric)
    }

    /// Validate the rubric covers `0..=5` exactly once and every
    /// level has a non-empty name.
    pub fn validate(&self) -> Result<(), SeverityRubricError> {
        let mut seen: BTreeSet<u8> = BTreeSet::new();
        for level in &self.levels {
            if level.level > 5 {
                return Err(SeverityRubricError::LevelOutOfRange(level.level as i32));
            }
            if level.name.is_empty() {
                return Err(SeverityRubricError::EmptyName(level.level));
            }
            if !seen.insert(level.level) {
                return Err(SeverityRubricError::DuplicateLevel(level.level));
            }
        }
        let missing: Vec<u8> = (0u8..=5).filter(|n| !seen.contains(n)).collect();
        if !missing.is_empty() {
            return Err(SeverityRubricError::MissingLevels(missing));
        }
        Ok(())
    }

    /// Look up the [`SeverityLevel`] for `level`. Returns `None`
    /// if the level is out of the validated range (impossible on a
    /// constructed [`SeverityRubric`] but defensive for callers
    /// that bypassed [`SeverityRubric::new`]).
    pub fn by_level(&self, level: u8) -> Option<&SeverityLevel> {
        self.levels.iter().find(|lv| lv.level == level)
    }
}

/// Resolved UX disposition for a single severity rank.
///
/// Mirrors `cv-guard/shared/policy/severity_mapper.py::UXDisposition`
/// — a value type the renderer can consume directly without
/// holding a reference to the source rubric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UXDisposition {
    /// Action the renderer must apply.
    pub ux_action: UXAction,
    /// Whether the renderer may offer a "tap to reveal" affordance.
    pub allow_reveal: bool,
    /// Whether the user may forward the media at this severity.
    pub allow_forward: bool,
}

/// Stateless helper: rubric → per-severity UX disposition.
///
/// Constructed once per scan and discarded; holds a reference to
/// the active rubric so the lookup is `O(levels)` (small constant)
/// without any allocation per call.
#[derive(Debug, Clone, Copy)]
pub struct SeverityMapper<'a> {
    rubric: &'a SeverityRubric,
}

impl<'a> SeverityMapper<'a> {
    /// Borrow the rubric for the lifetime of this mapper.
    pub fn new(rubric: &'a SeverityRubric) -> Self {
        Self { rubric }
    }

    /// Resolve the [`UXDisposition`] for `severity`. Returns
    /// [`SeverityRubricError::LevelOutOfRange`] if the underlying
    /// rubric was constructed with a level outside `0..=5`
    /// (impossible on a constructed [`SeverityRubric`]) or if
    /// `severity > 5`.
    pub fn disposition(&self, severity: u8) -> Result<UXDisposition, SeverityRubricError> {
        if severity > 5 {
            return Err(SeverityRubricError::LevelOutOfRange(severity as i32));
        }
        // by_level is None only if the rubric bypassed validation
        // — defensive fallback returns the same error.
        let level = self
            .rubric
            .by_level(severity)
            .ok_or(SeverityRubricError::LevelOutOfRange(severity as i32))?;
        Ok(UXDisposition {
            ux_action: level.ux_action,
            allow_reveal: level.allow_reveal,
            allow_forward: level.allow_forward,
        })
    }
}

/// Build the canonical default rubric matching the PROPOSAL.md §9
/// table. Useful for tests / smoke checks where the caller does
/// not have a skill pack on hand.
pub fn default_rubric() -> SeverityRubric {
    SeverityRubric::new(vec![
        SeverityLevel::new(0, "safe", UXAction::Clear)
            .unwrap()
            .with_description("Benign content; no friction."),
        SeverityLevel::new(1, "low", UXAction::Clear)
            .unwrap()
            .with_description("Edge case; no friction but logged."),
        SeverityLevel::new(2, "medium", UXAction::BlurTap)
            .unwrap()
            .with_description("Possibly unsafe; blur with tap-to-reveal."),
        SeverityLevel::new(3, "high", UXAction::BlurTap)
            .unwrap()
            .with_allow_forward(false)
            .with_description("Probably unsafe; blur and disable forward."),
        SeverityLevel::new(4, "severe", UXAction::Pixelate)
            .unwrap()
            .with_allow_reveal(false)
            .with_allow_forward(false)
            .with_description("Highly unsafe; persistent pixelation."),
        SeverityLevel::new(5, "block", UXAction::BlockedCard)
            .unwrap()
            .with_allow_reveal(false)
            .with_allow_forward(false)
            .with_description("Blocked; rendered behind a card."),
    ])
    .expect("default rubric covers 0..=5")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_rubric_covers_all_levels() {
        let r = default_rubric();
        r.validate().unwrap();
        for level in 0u8..=5 {
            assert!(r.by_level(level).is_some(), "missing level {level}");
        }
    }

    #[test]
    fn rubric_rejects_duplicate_levels() {
        let levels = vec![
            SeverityLevel::new(0, "a", UXAction::Clear).unwrap(),
            SeverityLevel::new(0, "b", UXAction::Clear).unwrap(),
            SeverityLevel::new(1, "c", UXAction::Clear).unwrap(),
            SeverityLevel::new(2, "d", UXAction::Clear).unwrap(),
            SeverityLevel::new(3, "e", UXAction::Clear).unwrap(),
            SeverityLevel::new(4, "f", UXAction::Clear).unwrap(),
            SeverityLevel::new(5, "g", UXAction::Clear).unwrap(),
        ];
        let err = SeverityRubric::new(levels).unwrap_err();
        assert_eq!(err, SeverityRubricError::DuplicateLevel(0));
    }

    #[test]
    fn rubric_rejects_missing_levels() {
        let levels = vec![
            SeverityLevel::new(0, "a", UXAction::Clear).unwrap(),
            SeverityLevel::new(1, "b", UXAction::Clear).unwrap(),
            SeverityLevel::new(4, "c", UXAction::Clear).unwrap(),
            SeverityLevel::new(5, "d", UXAction::Clear).unwrap(),
        ];
        let err = SeverityRubric::new(levels).unwrap_err();
        match err {
            SeverityRubricError::MissingLevels(missing) => {
                assert_eq!(missing, vec![2u8, 3u8]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn severity_level_rejects_out_of_range() {
        let err = SeverityLevel::new(6, "x", UXAction::Clear).unwrap_err();
        assert!(matches!(err, SeverityRubricError::LevelOutOfRange(6)));
    }

    #[test]
    fn severity_level_rejects_empty_name() {
        let err = SeverityLevel::new(0, "", UXAction::Clear).unwrap_err();
        assert_eq!(err, SeverityRubricError::EmptyName(0));
    }

    #[test]
    fn severity_mapper_returns_default_rubric_dispositions() {
        let r = default_rubric();
        let m = SeverityMapper::new(&r);
        assert_eq!(
            m.disposition(0).unwrap(),
            UXDisposition {
                ux_action: UXAction::Clear,
                allow_reveal: true,
                allow_forward: true,
            }
        );
        assert_eq!(
            m.disposition(3).unwrap(),
            UXDisposition {
                ux_action: UXAction::BlurTap,
                allow_reveal: true,
                allow_forward: false,
            }
        );
        assert_eq!(
            m.disposition(4).unwrap(),
            UXDisposition {
                ux_action: UXAction::Pixelate,
                allow_reveal: false,
                allow_forward: false,
            }
        );
        assert_eq!(
            m.disposition(5).unwrap(),
            UXDisposition {
                ux_action: UXAction::BlockedCard,
                allow_reveal: false,
                allow_forward: false,
            }
        );
    }

    #[test]
    fn severity_mapper_rejects_out_of_range_query() {
        let r = default_rubric();
        let m = SeverityMapper::new(&r);
        assert!(matches!(
            m.disposition(6),
            Err(SeverityRubricError::LevelOutOfRange(6))
        ));
    }

    #[test]
    fn severity_rubric_serde_round_trips() {
        let r = default_rubric();
        let json = serde_json::to_string(&r).unwrap();
        let parsed: SeverityRubric = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, r);
        parsed.validate().unwrap();
    }

    #[test]
    fn severity_rubric_serde_rejects_extra_fields() {
        // mirrors pydantic ConfigDict(extra="forbid").
        let json = r#"{
            "schema_version": 1,
            "levels": [],
            "evil": true
        }"#;
        let result: Result<SeverityRubric, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn severity_level_serde_rejects_extra_fields() {
        let json = r#"{
            "level": 0,
            "name": "safe",
            "ux_action": "clear",
            "evil": "x"
        }"#;
        let result: Result<SeverityLevel, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn severity_level_serde_applies_defaults_on_missing_fields() {
        let json = r#"{
            "level": 0,
            "name": "safe",
            "ux_action": "clear"
        }"#;
        let parsed: SeverityLevel = serde_json::from_str(json).unwrap();
        assert!(parsed.allow_reveal);
        assert!(parsed.allow_forward);
        assert_eq!(parsed.description, "");
    }
}

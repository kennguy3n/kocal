//! Threshold configuration consumed by the [`super::interpreter`].
//!
//! Mirrors the `ThresholdEntry` + `ThresholdsConfig` pair from
//! cv-guard's `shared/skillpack/schema.py`. The skill-pack
//! runtime (loader + verifier + overlay applier) produces a
//! [`ThresholdsConfig`] from a signed skill-pack tarball; until
//! that lands, the interpreter consumes a [`ThresholdsConfig`]
//! directly so tests + the SLM dispatch path can be unit-tested
//! without dragging in the entire pack machinery.
//!
//! Schema (kept in lock-step with cv-guard):
//!
//! ```yaml
//! schema_version: 1
//! thresholds:
//!   child_safety:
//!     any_hit:
//!       trigger: 0.20
//!       severe: null        # child_safety only uses the floor path
//!   adult:
//!     nudity:
//!       trigger: 0.40
//!       severe: 0.85
//! critical_rules: []        # cross-category overrides
//! ```
//!
//! Invariants validated at construction:
//!
//! * `trigger ∈ [0.0, 1.0]`, finite.
//! * `severe`, when present, `∈ [0.0, 1.0]`, finite, and
//!   `>= trigger`.
//! * Outer / inner maps are non-empty (an empty taxonomy bucket is
//!   a configuration bug — the compiler should always strip empty
//!   buckets before signing).

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Category name of the deterministic child-safety floor. Must match
/// the category half of [`super::interpreter::CHILD_SAFETY_LABEL`]
/// (`"child_safety.any_hit"`). The floor (decision case 1) is an
/// absolute, deterministic priority that *never* consults the SLM, so
/// a cascade-router `route` band is forbidden on this category — see
/// [`ThresholdsConfig::validate`].
const CHILD_SAFETY_CATEGORY: &str = "child_safety";

/// Per-label trigger / severe pair, plus an optional `route` floor.
///
/// `severe = None` is legal and indicates the label only
/// participates in the trigger path (it can fire the SLM, but the
/// severe-floor ceiling never applies). The child-safety floor
/// uses this shape — `child_safety.any_hit` is a floor signal, not
/// a severe-band signal.
///
/// `route` is the cascade-router lever (Stream H). When `route`
/// is `Some(r)` with `r <= trigger`, scores in the half-open band
/// `[route, trigger)` produce a *route-only* hit: the encoder is
/// treated as a high-recall ROUTER that escalates the case to the
/// SLM arbiter (decision case 4) instead of silently demoting it
/// to SAFE. Crucially, a route-only hit that cannot reach the SLM
/// (SLM disabled, rate-limited, or invariant fallback) collapses
/// to BENIGN (severity 0) — never a rule-path verdict — so the
/// widened band introduces **zero** rule-path false positives.
/// `route = None` (the default) means "no routing band": the label
/// behaves byte-for-byte as a classic `trigger`/`severe` pair, so
/// every pre-existing pack and fixture is unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdEntry {
    pub trigger: f64,
    #[serde(default)]
    pub severe: Option<f64>,
    #[serde(default)]
    pub route: Option<f64>,
}

impl ThresholdEntry {
    /// Construct + validate a classic (non-routing) entry.
    pub fn new(trigger: f64, severe: Option<f64>) -> Result<Self, ThresholdsError> {
        Self::new_with_route(trigger, severe, None)
    }

    /// Construct + validate an entry with an explicit routing band.
    pub fn new_with_route(
        trigger: f64,
        severe: Option<f64>,
        route: Option<f64>,
    ) -> Result<Self, ThresholdsError> {
        let entry = Self {
            trigger,
            severe,
            route,
        };
        entry.validate()?;
        Ok(entry)
    }

    /// Lower score bound at which this label produces a hit: the
    /// `route` floor when a routing band is configured, else the
    /// `trigger`. This is the single value [`super::find_hits`]
    /// gates on, so a `None` route is exactly today's behaviour.
    #[inline]
    pub fn route_or_trigger(&self) -> f64 {
        match self.route {
            Some(r) => r,
            None => self.trigger,
        }
    }

    /// Re-validate after deserialisation or direct field mutation.
    pub fn validate(&self) -> Result<(), ThresholdsError> {
        if !self.trigger.is_finite() || self.trigger < 0.0 || self.trigger > 1.0 {
            return Err(ThresholdsError::TriggerOutOfRange {
                trigger: self.trigger,
            });
        }
        if let Some(s) = self.severe {
            if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                return Err(ThresholdsError::SevereOutOfRange { severe: s });
            }
            if self.trigger > s {
                return Err(ThresholdsError::TriggerGreaterThanSevere {
                    trigger: self.trigger,
                    severe: s,
                });
            }
        }
        if let Some(r) = self.route {
            if !r.is_finite() || !(0.0..=1.0).contains(&r) {
                return Err(ThresholdsError::RouteOutOfRange { route: r });
            }
            if r > self.trigger {
                return Err(ThresholdsError::RouteGreaterThanTrigger {
                    route: r,
                    trigger: self.trigger,
                });
            }
        }
        Ok(())
    }
}

/// Full thresholds configuration: a 2-level map indexed by
/// `(category, label_name)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThresholdsConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    pub thresholds: BTreeMap<String, BTreeMap<String, ThresholdEntry>>,
    #[serde(default)]
    pub critical_rules: Vec<BTreeMap<String, String>>,
}

fn default_schema_version() -> u32 {
    1
}

impl ThresholdsConfig {
    /// Construct + validate every threshold entry.
    pub fn new(
        thresholds: BTreeMap<String, BTreeMap<String, ThresholdEntry>>,
    ) -> Result<Self, ThresholdsError> {
        let config = Self {
            schema_version: default_schema_version(),
            thresholds,
            critical_rules: Vec::new(),
        };
        config.validate()?;
        Ok(config)
    }

    /// Re-validate after deserialisation or direct field mutation.
    pub fn validate(&self) -> Result<(), ThresholdsError> {
        if self.schema_version < 1 {
            return Err(ThresholdsError::InvalidSchemaVersion {
                version: self.schema_version,
            });
        }
        if self.thresholds.is_empty() {
            return Err(ThresholdsError::EmptyThresholds);
        }
        for (cat, bucket) in &self.thresholds {
            if cat.is_empty() {
                return Err(ThresholdsError::EmptyCategoryName);
            }
            if bucket.is_empty() {
                return Err(ThresholdsError::EmptyCategoryBucket {
                    category: cat.clone(),
                });
            }
            for (label, entry) in bucket {
                if label.is_empty() {
                    return Err(ThresholdsError::EmptyLabelName {
                        category: cat.clone(),
                    });
                }
                // Defense-in-depth (Stream H): the child-safety
                // category is a pure deterministic floor (decision
                // case 1) that must never enter the SLM route band. A
                // `route` on any child_safety label would let
                // borderline child-safety content be arbitrated by —
                // or cleared to benign by — the SLM, contradicting the
                // "floor never consults the SLM" invariant. Reject it
                // at config construction so no signed pack can ever
                // configure it.
                if cat == CHILD_SAFETY_CATEGORY && entry.route.is_some() {
                    return Err(ThresholdsError::ChildSafetyRouteForbidden {
                        label: label.clone(),
                    });
                }
                entry.validate().map_err(|inner| match inner {
                    ThresholdsError::TriggerOutOfRange { trigger } => {
                        ThresholdsError::EntryTriggerOutOfRange {
                            category: cat.clone(),
                            label: label.clone(),
                            trigger,
                        }
                    }
                    ThresholdsError::SevereOutOfRange { severe } => {
                        ThresholdsError::EntrySevereOutOfRange {
                            category: cat.clone(),
                            label: label.clone(),
                            severe,
                        }
                    }
                    ThresholdsError::TriggerGreaterThanSevere { trigger, severe } => {
                        ThresholdsError::EntryTriggerGreaterThanSevere {
                            category: cat.clone(),
                            label: label.clone(),
                            trigger,
                            severe,
                        }
                    }
                    ThresholdsError::RouteOutOfRange { route } => {
                        ThresholdsError::EntryRouteOutOfRange {
                            category: cat.clone(),
                            label: label.clone(),
                            route,
                        }
                    }
                    ThresholdsError::RouteGreaterThanTrigger { route, trigger } => {
                        ThresholdsError::EntryRouteGreaterThanTrigger {
                            category: cat.clone(),
                            label: label.clone(),
                            route,
                            trigger,
                        }
                    }
                    // The other variants are not produced by
                    // `ThresholdEntry::validate`; preserve them as-
                    // is so the error path stays exhaustive.
                    other => other,
                })?;
            }
        }
        Ok(())
    }

    /// Look up `(category, label_name)`. Returns `None` if either
    /// the category bucket or the label entry is absent.
    pub fn entry(&self, category: &str, label_name: &str) -> Option<&ThresholdEntry> {
        self.thresholds
            .get(category)
            .and_then(|bucket| bucket.get(label_name))
    }
}

/// Errors raised by [`ThresholdEntry::new`] / [`ThresholdsConfig::new`].
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdsError {
    InvalidSchemaVersion {
        version: u32,
    },
    EmptyThresholds,
    EmptyCategoryName,
    EmptyCategoryBucket {
        category: String,
    },
    EmptyLabelName {
        category: String,
    },
    TriggerOutOfRange {
        trigger: f64,
    },
    SevereOutOfRange {
        severe: f64,
    },
    TriggerGreaterThanSevere {
        trigger: f64,
        severe: f64,
    },
    RouteOutOfRange {
        route: f64,
    },
    RouteGreaterThanTrigger {
        route: f64,
        trigger: f64,
    },
    /// A `route` band was configured on a `child_safety` label. The
    /// child-safety floor is a deterministic priority that never
    /// consults the SLM, so routing it is forbidden.
    ChildSafetyRouteForbidden {
        label: String,
    },
    EntryTriggerOutOfRange {
        category: String,
        label: String,
        trigger: f64,
    },
    EntrySevereOutOfRange {
        category: String,
        label: String,
        severe: f64,
    },
    EntryTriggerGreaterThanSevere {
        category: String,
        label: String,
        trigger: f64,
        severe: f64,
    },
    EntryRouteOutOfRange {
        category: String,
        label: String,
        route: f64,
    },
    EntryRouteGreaterThanTrigger {
        category: String,
        label: String,
        route: f64,
        trigger: f64,
    },
}

impl fmt::Display for ThresholdsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSchemaVersion { version } => {
                write!(f, "thresholds.schema_version must be >= 1, got {version}")
            }
            Self::EmptyThresholds => f.write_str("thresholds map is empty"),
            Self::EmptyCategoryName => f.write_str("thresholds: category name is empty"),
            Self::EmptyCategoryBucket { category } => {
                write!(f, "thresholds[{category:?}] is empty")
            }
            Self::EmptyLabelName { category } => {
                write!(f, "thresholds[{category:?}]: label name is empty")
            }
            Self::TriggerOutOfRange { trigger } => {
                write!(f, "trigger {trigger} not finite or outside [0.0, 1.0]")
            }
            Self::SevereOutOfRange { severe } => {
                write!(f, "severe {severe} not finite or outside [0.0, 1.0]")
            }
            Self::TriggerGreaterThanSevere { trigger, severe } => {
                write!(f, "trigger {trigger} > severe {severe}")
            }
            Self::RouteOutOfRange { route } => {
                write!(f, "route {route} not finite or outside [0.0, 1.0]")
            }
            Self::RouteGreaterThanTrigger { route, trigger } => {
                write!(f, "route {route} > trigger {trigger}")
            }
            Self::EntryTriggerOutOfRange {
                category,
                label,
                trigger,
            } => {
                write!(
                    f,
                    "thresholds[{category:?}][{label:?}].trigger {trigger} not finite or outside [0.0, 1.0]"
                )
            }
            Self::EntrySevereOutOfRange {
                category,
                label,
                severe,
            } => {
                write!(
                    f,
                    "thresholds[{category:?}][{label:?}].severe {severe} not finite or outside [0.0, 1.0]"
                )
            }
            Self::EntryTriggerGreaterThanSevere {
                category,
                label,
                trigger,
                severe,
            } => {
                write!(
                    f,
                    "thresholds[{category:?}][{label:?}]: trigger {trigger} > severe {severe}"
                )
            }
            Self::EntryRouteOutOfRange {
                category,
                label,
                route,
            } => {
                write!(
                    f,
                    "thresholds[{category:?}][{label:?}].route {route} not finite or outside [0.0, 1.0]"
                )
            }
            Self::EntryRouteGreaterThanTrigger {
                category,
                label,
                route,
                trigger,
            } => {
                write!(
                    f,
                    "thresholds[{category:?}][{label:?}]: route {route} > trigger {trigger}"
                )
            }
            Self::ChildSafetyRouteForbidden { label } => {
                write!(
                    f,
                    "thresholds[\"child_safety\"][{label:?}]: route is forbidden on the \
                     child_safety floor category (the floor never consults the SLM)"
                )
            }
        }
    }
}

impl std::error::Error for ThresholdsError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(trigger: f64, severe: Option<f64>) -> ThresholdEntry {
        ThresholdEntry::new(trigger, severe).unwrap()
    }

    fn pack(items: &[(&str, &str, f64, Option<f64>)]) -> ThresholdsConfig {
        let mut thresholds = BTreeMap::new();
        for (cat, label, t, s) in items {
            thresholds
                .entry((*cat).to_string())
                .or_insert_with(BTreeMap::new)
                .insert((*label).to_string(), entry(*t, *s));
        }
        ThresholdsConfig::new(thresholds).unwrap()
    }

    #[test]
    fn entry_accepts_valid_pair_with_severe() {
        let e = entry(0.4, Some(0.85));
        assert_eq!(e.trigger, 0.4);
        assert_eq!(e.severe, Some(0.85));
    }

    #[test]
    fn entry_accepts_pair_without_severe() {
        let e = entry(0.2, None);
        assert_eq!(e.trigger, 0.2);
        assert_eq!(e.severe, None);
    }

    #[test]
    fn entry_rejects_trigger_below_zero() {
        let err = ThresholdEntry::new(-0.1, None).unwrap_err();
        assert!(matches!(err, ThresholdsError::TriggerOutOfRange { .. }));
    }

    #[test]
    fn entry_rejects_trigger_above_one() {
        let err = ThresholdEntry::new(1.5, None).unwrap_err();
        assert!(matches!(err, ThresholdsError::TriggerOutOfRange { .. }));
    }

    #[test]
    fn entry_rejects_nonfinite_trigger() {
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = ThresholdEntry::new(v, None).unwrap_err();
            assert!(matches!(err, ThresholdsError::TriggerOutOfRange { .. }));
        }
    }

    #[test]
    fn entry_rejects_severe_out_of_range() {
        let err = ThresholdEntry::new(0.5, Some(1.5)).unwrap_err();
        assert!(matches!(err, ThresholdsError::SevereOutOfRange { .. }));
    }

    #[test]
    fn entry_rejects_trigger_greater_than_severe() {
        let err = ThresholdEntry::new(0.9, Some(0.5)).unwrap_err();
        assert!(matches!(
            err,
            ThresholdsError::TriggerGreaterThanSevere { .. }
        ));
    }

    // ---- route (cascade-router band) validation ----

    #[test]
    fn entry_accepts_route_below_trigger() {
        let e = ThresholdEntry::new_with_route(0.55, Some(0.85), Some(0.40)).unwrap();
        assert_eq!(e.trigger, 0.55);
        assert_eq!(e.severe, Some(0.85));
        assert_eq!(e.route, Some(0.40));
    }

    #[test]
    fn entry_accepts_route_equal_to_trigger() {
        // route == trigger is the degenerate (zero-width) band and
        // must be legal: it means "no extra routing", identical to
        // route = None for find_hits purposes.
        let e = ThresholdEntry::new_with_route(0.55, None, Some(0.55)).unwrap();
        assert_eq!(e.route, Some(0.55));
        assert_eq!(e.route_or_trigger(), 0.55);
    }

    #[test]
    fn route_or_trigger_prefers_route_then_trigger() {
        assert_eq!(entry(0.55, None).route_or_trigger(), 0.55);
        let routed = ThresholdEntry::new_with_route(0.55, None, Some(0.30)).unwrap();
        assert_eq!(routed.route_or_trigger(), 0.30);
    }

    #[test]
    fn entry_rejects_route_below_zero() {
        let err = ThresholdEntry::new_with_route(0.55, None, Some(-0.1)).unwrap_err();
        assert!(matches!(err, ThresholdsError::RouteOutOfRange { .. }));
    }

    #[test]
    fn entry_rejects_route_above_one() {
        // route > 1 is out of range; it also exceeds trigger, but the
        // range check fires first and is the more specific signal.
        let err = ThresholdEntry::new_with_route(1.0, None, Some(1.5)).unwrap_err();
        assert!(matches!(err, ThresholdsError::RouteOutOfRange { .. }));
    }

    #[test]
    fn entry_rejects_nonfinite_route() {
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = ThresholdEntry::new_with_route(0.55, None, Some(v)).unwrap_err();
            assert!(matches!(err, ThresholdsError::RouteOutOfRange { .. }));
        }
    }

    #[test]
    fn entry_rejects_route_greater_than_trigger() {
        let err = ThresholdEntry::new_with_route(0.30, None, Some(0.50)).unwrap_err();
        match err {
            ThresholdsError::RouteGreaterThanTrigger { route, trigger } => {
                assert_eq!(route, 0.50);
                assert_eq!(trigger, 0.30);
            }
            other => panic!("expected RouteGreaterThanTrigger, got {other:?}"),
        }
    }

    #[test]
    fn config_surfaces_entry_route_out_of_range_with_location() {
        let mut thresholds = BTreeMap::new();
        thresholds.insert(
            "hate".to_string(),
            BTreeMap::from([(
                "slur".to_string(),
                ThresholdEntry {
                    trigger: 0.55,
                    severe: Some(0.85),
                    route: Some(1.5),
                },
            )]),
        );
        let err = ThresholdsConfig::new(thresholds).unwrap_err();
        match err {
            ThresholdsError::EntryRouteOutOfRange {
                category,
                label,
                route,
            } => {
                assert_eq!(category, "hate");
                assert_eq!(label, "slur");
                assert_eq!(route, 1.5);
            }
            other => panic!("expected EntryRouteOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn config_surfaces_entry_route_greater_than_trigger_with_location() {
        let mut thresholds = BTreeMap::new();
        thresholds.insert(
            "self_harm".to_string(),
            BTreeMap::from([(
                "ideation".to_string(),
                ThresholdEntry {
                    trigger: 0.40,
                    severe: None,
                    route: Some(0.60),
                },
            )]),
        );
        let err = ThresholdsConfig::new(thresholds).unwrap_err();
        match err {
            ThresholdsError::EntryRouteGreaterThanTrigger {
                category,
                label,
                route,
                trigger,
            } => {
                assert_eq!(category, "self_harm");
                assert_eq!(label, "ideation");
                assert_eq!(route, 0.60);
                assert_eq!(trigger, 0.40);
            }
            other => panic!("expected EntryRouteGreaterThanTrigger, got {other:?}"),
        }
    }

    #[test]
    fn config_rejects_route_on_child_safety_label() {
        // Defense-in-depth: child_safety is the deterministic floor and
        // must never enter the SLM route band, regardless of whether
        // the route value is otherwise valid.
        let mut thresholds = BTreeMap::new();
        thresholds.insert(
            "child_safety".to_string(),
            BTreeMap::from([(
                "any_hit".to_string(),
                ThresholdEntry {
                    trigger: 0.20,
                    severe: None,
                    route: Some(0.10),
                },
            )]),
        );
        let err = ThresholdsConfig::new(thresholds).unwrap_err();
        match err {
            ThresholdsError::ChildSafetyRouteForbidden { label } => {
                assert_eq!(label, "any_hit");
            }
            other => panic!("expected ChildSafetyRouteForbidden, got {other:?}"),
        }
    }

    #[test]
    fn config_accepts_child_safety_without_route() {
        // The classic floor-only shape (route = None) stays valid.
        let p = pack(&[("child_safety", "any_hit", 0.20, None)]);
        let e = p.entry("child_safety", "any_hit").unwrap();
        assert_eq!(e.route, None);
    }

    #[test]
    fn config_rejects_child_safety_route_from_yaml() {
        // The loader path (decode_thresholds -> ThresholdsConfig::new)
        // must reject a route configured on child_safety by a pack.
        let mut thresholds = BTreeMap::new();
        thresholds.insert(
            "child_safety".to_string(),
            BTreeMap::from([(
                "any_hit".to_string(),
                ThresholdEntry {
                    trigger: 0.20,
                    severe: None,
                    route: Some(0.05),
                },
            )]),
        );
        let err = ThresholdsConfig::new(thresholds).unwrap_err();
        assert!(err.to_string().contains("route is forbidden"), "got: {err}");
    }

    #[test]
    fn entry_defaults_route_to_none_when_absent_in_json() {
        let e: ThresholdEntry = serde_json::from_str(r#"{"trigger":0.55,"severe":0.85}"#).unwrap();
        assert_eq!(e.route, None);
    }

    #[test]
    fn entry_deserializes_explicit_route_from_json() {
        let e: ThresholdEntry =
            serde_json::from_str(r#"{"trigger":0.55,"severe":0.85,"route":0.40}"#).unwrap();
        assert_eq!(e.route, Some(0.40));
    }

    #[test]
    fn entry_serializes_route_field_even_when_none() {
        // Symmetric with `severe`: the field is always emitted (as
        // null) so the Rust serde shape matches the Python
        // pydantic `model_dump` shape the parity fixtures encode.
        let json = serde_json::to_string(&entry(0.55, Some(0.85))).unwrap();
        assert!(json.contains(r#""route":null"#), "got: {json}");
    }

    #[test]
    fn config_rejects_empty_thresholds() {
        let err = ThresholdsConfig::new(BTreeMap::new()).unwrap_err();
        assert!(matches!(err, ThresholdsError::EmptyThresholds));
    }

    #[test]
    fn config_rejects_empty_category_bucket() {
        let mut thresholds = BTreeMap::new();
        thresholds.insert("adult".to_string(), BTreeMap::new());
        let err = ThresholdsConfig::new(thresholds).unwrap_err();
        match err {
            ThresholdsError::EmptyCategoryBucket { category } => {
                assert_eq!(category, "adult");
            }
            _ => panic!("expected EmptyCategoryBucket"),
        }
    }

    #[test]
    fn config_entry_lookup_returns_some_for_known_label() {
        let p = pack(&[("adult", "nudity", 0.4, Some(0.85))]);
        let e = p.entry("adult", "nudity").unwrap();
        assert_eq!(e.trigger, 0.4);
        assert_eq!(e.severe, Some(0.85));
    }

    #[test]
    fn config_entry_lookup_returns_none_for_unknown_label() {
        let p = pack(&[("adult", "nudity", 0.4, Some(0.85))]);
        assert!(p.entry("adult", "missing").is_none());
        assert!(p.entry("missing_cat", "nudity").is_none());
    }

    #[test]
    fn config_roundtrips_through_json_with_sorted_keys() {
        let p = pack(&[
            ("child_safety", "any_hit", 0.2, None),
            ("adult", "nudity", 0.4, Some(0.85)),
        ]);
        let json = serde_json::to_string(&p).unwrap();
        // Outer-key sort: adult before child_safety.
        assert!(json.contains(r#""thresholds":{"adult":"#));
        // Roundtrip back into a ThresholdsConfig and re-validate.
        let p2: ThresholdsConfig = serde_json::from_str(&json).unwrap();
        p2.validate().unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn config_rejects_unknown_top_level_field() {
        let bad = r#"{"schema_version":1,"thresholds":{},"critical_rules":[],"unknown":"x"}"#;
        let err = serde_json::from_str::<ThresholdsConfig>(bad).unwrap_err();
        assert!(err.to_string().contains("unknown"));
    }

    #[test]
    fn entry_rejects_unknown_field_in_json() {
        let bad = r#"{"trigger":0.4,"severe":0.85,"injected":true}"#;
        let err = serde_json::from_str::<ThresholdEntry>(bad).unwrap_err();
        assert!(err.to_string().contains("injected"));
    }
}

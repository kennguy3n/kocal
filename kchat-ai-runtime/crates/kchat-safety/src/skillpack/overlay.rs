//! Overlay merge algorithm.
//!
//! Both [`apply_community_overlay`] and [`apply_jurisdiction_overlay`]
//! share the same internal merge primitives — the two public
//! functions exist as separate symbols so the resolver
//! ([`super::overlay_resolver::resolve_effective_pack`]) can apply
//! them in a fixed *jurisdiction → community* order. Mirrors the
//! Python pair `shared/skillpack/overlay.py` +
//! `shared/skillpack/jurisdiction_overlay.py` one-for-one.
//!
//! ### Determinism
//!
//! The merge is a pure function of `(base, overlay)`. The inputs
//! are never mutated; each call allocates fresh maps and returns a
//! freshly constructed [`SkillPack`]. Both Rust and Python keep
//! their threshold / lexicon / regex maps in sorted order
//! (`BTreeMap` / `dict` since Python 3.7), and the merge preserves
//! that ordering, so the canonical-JSON of the merged pack is
//! byte-identical across platforms.
//!
//! ### Child-safety floor
//!
//! For any `child_safety.*` threshold, the merge raises
//! [`SkillPackError::OverlayFloorViolation`] if the overlay would
//! *raise* the trigger or severe above the base value (which would
//! mean less sensitivity → weaker protection). Tightening — i.e.
//! lowering — is always allowed. When the overlay introduces a
//! brand-new `child_safety.*` label the hard floor of `0.20`
//! applies. See [`PROTECTED_CATEGORIES`].

use std::collections::BTreeMap;

use crate::policy_interpreter::{SeverityLevel, SeverityRubric, ThresholdEntry, ThresholdsConfig};

use super::overlay_schema::{
    CommunityOverlay, JurisdictionOverlay, OverlayLexiconAddition, OverlayRegexAddition,
    OverlaySeverityLevel, OverlayThresholdEntry,
};
use super::schema::{Lexicon, LexiconEntry, RegexSet, SkillPack};
use super::SkillPackError;

/// Categories whose thresholds can never be loosened by an
/// overlay. Mirrors `PROTECTED_CATEGORIES` in
/// `cv-guard/shared/skillpack/overlay.py`.
pub const PROTECTED_CATEGORIES: &[&str] = &["child_safety"];

/// PROPOSAL §10 hard floor for a *new* `child_safety.*` label the
/// overlay introduces. Any overlay-declared trigger / severe above
/// this value is a floor violation — even though there is no base
/// entry to compare against, the floor is the global baseline by
/// definition. Mirrors the literal `0.20` in
/// `_check_protected_floor`.
const CHILD_SAFETY_NEW_LABEL_FLOOR: f64 = 0.20;

fn is_protected(category: &str) -> bool {
    PROTECTED_CATEGORIES.contains(&category)
}

/// Reject overlays that would *loosen* a protected-category
/// threshold.
///
/// "Loosen" means raising `trigger` or `severe` above the base
/// value, since a higher threshold = less sensitive = weaker
/// protection. Lowering or matching is fine. Clearing the severe
/// floor on a protected label is rejected outright.
fn check_protected_floor(
    category: &str,
    name: &str,
    base: Option<&ThresholdEntry>,
    overlay: &OverlayThresholdEntry,
) -> Result<(), SkillPackError> {
    if !is_protected(category) {
        return Ok(());
    }
    match base {
        None => {
            // Overlay is introducing a brand-new child_safety
            // label. Anything above 0.20 would be weaker than the
            // hard floor by definition.
            if let Some(t) = overlay.trigger {
                if t > CHILD_SAFETY_NEW_LABEL_FLOOR {
                    return Err(SkillPackError::OverlayFloorViolation {
                        category: category.to_string(),
                        label: name.to_string(),
                        detail: format!(
                            "trigger {t} > new-label floor {CHILD_SAFETY_NEW_LABEL_FLOOR}"
                        ),
                    });
                }
            }
            if let Some(s) = overlay.severe {
                if s > CHILD_SAFETY_NEW_LABEL_FLOOR {
                    return Err(SkillPackError::OverlayFloorViolation {
                        category: category.to_string(),
                        label: name.to_string(),
                        detail: format!(
                            "severe {s} > new-label floor {CHILD_SAFETY_NEW_LABEL_FLOOR}"
                        ),
                    });
                }
            }
        }
        Some(base_entry) => {
            if let Some(t) = overlay.trigger {
                if t > base_entry.trigger {
                    return Err(SkillPackError::OverlayFloorViolation {
                        category: category.to_string(),
                        label: name.to_string(),
                        detail: format!("trigger {t} > base {}", base_entry.trigger),
                    });
                }
            }
            if let Some(s) = overlay.severe {
                let base_severe = base_entry.severe.unwrap_or(CHILD_SAFETY_NEW_LABEL_FLOOR);
                if s > base_severe {
                    return Err(SkillPackError::OverlayFloorViolation {
                        category: category.to_string(),
                        label: name.to_string(),
                        detail: format!("severe {s} > base {base_severe}"),
                    });
                }
            }
            if overlay.clear_severe && base_entry.severe.is_some() {
                return Err(SkillPackError::OverlayFloorViolation {
                    category: category.to_string(),
                    label: name.to_string(),
                    detail: "cannot clear severe floor on protected label".to_string(),
                });
            }
        }
    }
    Ok(())
}

/// Merge one (`base?`, `overlay`) entry into a fresh
/// [`ThresholdEntry`]. The base may be missing — the overlay can
/// introduce a brand-new label as long as it supplies a `trigger`.
fn merge_threshold_entry(
    base: Option<&ThresholdEntry>,
    overlay: &OverlayThresholdEntry,
) -> Result<ThresholdEntry, SkillPackError> {
    let base_trigger = base.map(|b| b.trigger);
    let base_severe = base.and_then(|b| b.severe);
    // The cascade-router `route` floor is a baseline-only property
    // today (`OverlayThresholdEntry` cannot set or clear it), so a
    // merge must carry the base's `route` forward verbatim. Without
    // this an overlay that merely re-tunes `trigger`/`severe` for a
    // weak category would silently drop its routing band, collapsing
    // the encoder=router behaviour back to today's demote-to-SAFE.
    let base_route = base.and_then(|b| b.route);

    let new_trigger =
        overlay
            .trigger
            .or(base_trigger)
            .ok_or_else(|| SkillPackError::SchemaViolation {
                path: "<overlay>".to_string(),
                detail: "overlay entry must define trigger when no base entry exists".to_string(),
            })?;

    let new_severe = if overlay.clear_severe {
        None
    } else if overlay.severe.is_some() {
        overlay.severe
    } else {
        base_severe
    };

    // Re-validate the merged shape via the canonical constructor —
    // this catches a corner case where the overlay clears the
    // severe field while leaving the trigger at the base value,
    // and the merged entry must still satisfy
    // `trigger <= severe` (vacuously true when severe is None) and
    // `route <= trigger` (an overlay may lower `trigger` below a
    // carried-forward `route`, which must be rejected).
    ThresholdEntry::new_with_route(new_trigger, new_severe, base_route).map_err(|e| {
        SkillPackError::SchemaViolation {
            path: "<overlay>".to_string(),
            detail: format!("merged threshold entry invalid: {e}"),
        }
    })
}

fn apply_threshold_overrides(
    base: &ThresholdsConfig,
    overrides: &BTreeMap<String, BTreeMap<String, OverlayThresholdEntry>>,
) -> Result<ThresholdsConfig, SkillPackError> {
    let mut new_thresholds: BTreeMap<String, BTreeMap<String, ThresholdEntry>> =
        base.thresholds.clone();
    for (category, label_map) in overrides {
        let bucket = new_thresholds.entry(category.clone()).or_default();
        for (name, overlay_entry) in label_map {
            // Pulls the *base* entry for this label. Safe because
            // each `(category, label)` pair is visited exactly
            // once per merge: the outer iteration is over
            // `overrides` (a BTreeMap, so unique categories) and
            // this inner iteration is over `label_map` (also a
            // BTreeMap, so unique labels within the category).
            // No prior iteration of *this* inner loop could have
            // written to `name`, so `bucket.get(name)` is the
            // pre-merge entry — never an overlay-on-overlay
            // intermediate. Defense-in-depth check below makes
            // the invariant a runtime assertion in debug builds.
            let base_entry = bucket.get(name).copied();
            debug_assert_eq!(
                base_entry,
                base.thresholds.get(category).and_then(|m| m.get(name)).copied(),
                "apply_threshold_overrides invariant: bucket.get must equal base.thresholds.get for (category={category}, label={name})"
            );
            check_protected_floor(category, name, base_entry.as_ref(), overlay_entry)?;
            let merged = merge_threshold_entry(base_entry.as_ref(), overlay_entry)?;
            bucket.insert(name.clone(), merged);
        }
    }
    // Re-construct through the canonical validator so the merged
    // config goes through the same closed-shape checks the loader
    // applies.
    let mut config =
        ThresholdsConfig::new(new_thresholds).map_err(|e| SkillPackError::SchemaViolation {
            path: "<overlay>".to_string(),
            detail: format!("merged ThresholdsConfig invalid: {e}"),
        })?;
    config.schema_version = base.schema_version;
    config.critical_rules = base.critical_rules.clone();
    Ok(config)
}

fn apply_severity_overrides(
    base: &SeverityRubric,
    overrides: &[OverlaySeverityLevel],
) -> Result<SeverityRubric, SkillPackError> {
    let mut by_level: BTreeMap<u8, SeverityLevel> = base
        .levels
        .iter()
        .map(|lv| (lv.level, lv.clone()))
        .collect();
    for ov in overrides {
        let Some(existing) = by_level.get(&ov.level).cloned() else {
            return Err(SkillPackError::SchemaViolation {
                path: "<overlay>".to_string(),
                detail: format!("severity override targets missing level {}", ov.level),
            });
        };
        let merged_ux_action = match &ov.ux_action {
            Some(s) => {
                // Translate the string back into the closed
                // UXAction enum so the merged rubric has the same
                // shape as a freshly loaded one.
                crate::policy_interpreter::UXAction::from_str_strict(s).map_err(|e| {
                    SkillPackError::SchemaViolation {
                        path: "<overlay>".to_string(),
                        detail: format!("severity override ux_action invalid: {e}"),
                    }
                })?
            }
            None => existing.ux_action,
        };
        let merged = SeverityLevel {
            level: ov.level,
            name: ov.name.clone().unwrap_or(existing.name),
            ux_action: merged_ux_action,
            allow_reveal: ov.allow_reveal.unwrap_or(existing.allow_reveal),
            allow_forward: ov.allow_forward.unwrap_or(existing.allow_forward),
            description: ov.description.clone().unwrap_or(existing.description),
        };
        by_level.insert(ov.level, merged);
    }
    let levels: Vec<SeverityLevel> = by_level.into_values().collect();
    SeverityRubric::with_schema_version(levels, base.schema_version).map_err(|e| {
        SkillPackError::SchemaViolation {
            path: "<overlay>".to_string(),
            detail: format!("merged SeverityRubric invalid: {e}"),
        }
    })
}

fn apply_lexicon_additions(
    base: &BTreeMap<String, Lexicon>,
    additions: &[OverlayLexiconAddition],
) -> Result<BTreeMap<String, Lexicon>, SkillPackError> {
    let mut out: BTreeMap<String, Lexicon> = base.clone();
    for add in additions {
        match out.get(&add.key).cloned() {
            None => {
                out.insert(add.key.clone(), add.clone().into_lexicon());
            }
            Some(existing) => {
                if existing.language != add.language {
                    return Err(SkillPackError::SchemaViolation {
                        path: "<overlay>".to_string(),
                        detail: format!(
                            "lexicon language mismatch for key {:?}: base={} overlay={}",
                            add.key, existing.language, add.language
                        ),
                    });
                }
                let mut merged_entries: Vec<LexiconEntry> = existing.entries;
                merged_entries.extend(add.entries.iter().cloned());
                out.insert(
                    add.key.clone(),
                    Lexicon {
                        language: existing.language,
                        entries: merged_entries,
                    },
                );
            }
        }
    }
    Ok(out)
}

/// Wholesale-replace regex additions into the base map.
///
/// Note: this is *replace*, not *append* — different semantics
/// from [`apply_lexicon_additions`] which appends new entries
/// onto an existing lexicon. The asymmetry is intentional:
///
/// * A [`RegexSet`] carries a `name` field used in audit logs
///   and decision rationales (e.g. `"matched
///   scam_phrases.urgency_v2"`). An overlay addition either *is*
///   that set or it isn't — augmenting with extra patterns under
///   the same key would silently change what the audit string
///   refers to.
/// * Lexicons are identified by `key` (= file stem) and their
///   entries are leaf items with no semantic identity, so adding
///   more phrases to an `"en"` lexicon is a meaningful
///   incremental edit.
///
/// Mirrors cv-guard reference (`shared/skillpack/overlay.py::
/// _apply_regex_additions` does the same wholesale-replace via
/// `dict.update`).
fn apply_regex_additions(
    base: &BTreeMap<String, RegexSet>,
    additions: &[OverlayRegexAddition],
) -> BTreeMap<String, RegexSet> {
    let mut out: BTreeMap<String, RegexSet> = base.clone();
    for add in additions {
        out.insert(add.key.clone(), add.to_regex_set());
    }
    out
}

/// Concatenate a non-empty suffix onto the base SLM prompt with
/// the canonical `\n---\n` separator. Empty suffix is a no-op.
fn join_prompt(base_prompt: &str, suffix: &str) -> String {
    if suffix.is_empty() {
        return base_prompt.to_string();
    }
    format!("{}\n---\n{}\n", base_prompt.trim_end(), suffix.trim())
}

/// Apply `overlay` on top of `base`, returning a freshly
/// constructed [`SkillPack`].
///
/// The function is a pure function of its inputs — neither
/// argument is mutated. Determinism guarantee: the same
/// `(base, overlay)` always yields a byte-identical
/// canonical-JSON encoding of the result, on both Rust and Python.
///
/// Raises [`SkillPackError::OverlayFloorViolation`] if `overlay`
/// would loosen any `child_safety.*` threshold (see
/// [`PROTECTED_CATEGORIES`]).
pub fn apply_community_overlay(
    base: &SkillPack,
    overlay: &CommunityOverlay,
) -> Result<SkillPack, SkillPackError> {
    Ok(SkillPack {
        manifest: base.manifest.clone(),
        taxonomy: base.taxonomy.clone(),
        thresholds: apply_threshold_overrides(&base.thresholds, &overlay.threshold_overrides)?,
        severity_rubric: apply_severity_overrides(
            &base.severity_rubric,
            &overlay.severity_overrides,
        )?,
        scam_phrases: apply_lexicon_additions(&base.scam_phrases, &overlay.scam_phrase_additions)?,
        hate_lexicons: apply_lexicon_additions(
            &base.hate_lexicons,
            &overlay.hate_lexicon_additions,
        )?,
        regex_sets: apply_regex_additions(&base.regex_sets, &overlay.regex_additions),
        slm_prompt: join_prompt(&base.slm_prompt, &overlay.slm_prompt_suffix),
    })
}

/// Apply `overlay` on top of `base`. Same semantics as
/// [`apply_community_overlay`]; the two functions exist as
/// separate symbols so the resolver can apply them in the correct
/// (jurisdiction → community) order.
pub fn apply_jurisdiction_overlay(
    base: &SkillPack,
    overlay: &JurisdictionOverlay,
) -> Result<SkillPack, SkillPackError> {
    Ok(SkillPack {
        manifest: base.manifest.clone(),
        taxonomy: base.taxonomy.clone(),
        thresholds: apply_threshold_overrides(&base.thresholds, &overlay.threshold_overrides)?,
        severity_rubric: apply_severity_overrides(
            &base.severity_rubric,
            &overlay.severity_overrides,
        )?,
        scam_phrases: apply_lexicon_additions(&base.scam_phrases, &overlay.scam_phrase_additions)?,
        hate_lexicons: apply_lexicon_additions(
            &base.hate_lexicons,
            &overlay.hate_lexicon_additions,
        )?,
        regex_sets: apply_regex_additions(&base.regex_sets, &overlay.regex_additions),
        slm_prompt: join_prompt(&base.slm_prompt, &overlay.slm_prompt_suffix),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::policy_interpreter::UXAction;
    use crate::skillpack::overlay_schema::OverlayLexiconAddition;
    use crate::skillpack::schema::{LexiconEntry, RegexPattern, SkillPackManifest, TaxonomyConfig};

    fn base_pack() -> SkillPack {
        let mut thresholds = BTreeMap::new();
        thresholds.insert(
            "child_safety".to_string(),
            BTreeMap::from([(
                "any_hit".to_string(),
                ThresholdEntry::new(0.20, None).unwrap(),
            )]),
        );
        thresholds.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "nudity".to_string(),
                ThresholdEntry::new(0.40, Some(0.80)).unwrap(),
            )]),
        );
        let thresholds_cfg = ThresholdsConfig::new(thresholds).unwrap();

        let severity_rubric = SeverityRubric::new(vec![
            SeverityLevel::new(0, "safe", UXAction::Clear).unwrap(),
            SeverityLevel::new(1, "low", UXAction::Clear).unwrap(),
            SeverityLevel::new(2, "low_blur", UXAction::BlurTap).unwrap(),
            SeverityLevel::new(3, "medium", UXAction::BlurTap).unwrap(),
            SeverityLevel::new(4, "high", UXAction::Pixelate)
                .unwrap()
                .with_allow_forward(false),
            SeverityLevel::new(5, "severe", UXAction::BlockedCard)
                .unwrap()
                .with_allow_reveal(false)
                .with_allow_forward(false),
        ])
        .unwrap();

        let mut scam_phrases = BTreeMap::new();
        scam_phrases.insert(
            "en".to_string(),
            Lexicon {
                language: "en".to_string(),
                entries: vec![LexiconEntry {
                    phrase: "send gift cards".to_string(),
                    weight: 1.0,
                    tags: vec!["scam".to_string()],
                }],
            },
        );
        let hate_lexicons = BTreeMap::new();
        let mut regex_sets = BTreeMap::new();
        regex_sets.insert(
            "pii".to_string(),
            RegexSet {
                name: "pii".to_string(),
                patterns: vec![RegexPattern {
                    name: "email".to_string(),
                    pattern: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b".to_string(),
                    description: "Email addresses".to_string(),
                    flags: vec!["i".to_string()],
                }],
            },
        );

        let manifest = SkillPackManifest {
            pack_id: "cvguard.skill.base.v1".to_string(),
            version: "1.0.0".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            schema_version: 1,
            author: "tests".to_string(),
            description: String::new(),
            min_runtime_version: "0.1.0".to_string(),
            content_sha256: "0".repeat(64),
            signature: Some("0".repeat(128)),
            public_key: Some("0".repeat(64)),
        };

        let mut taxonomy_labels: BTreeMap<String, Vec<String>> = BTreeMap::new();
        taxonomy_labels.insert("child_safety".to_string(), vec!["any_hit".to_string()]);
        taxonomy_labels.insert("adult".to_string(), vec!["nudity".to_string()]);
        let taxonomy = TaxonomyConfig {
            schema_version: 1,
            labels: taxonomy_labels,
            output: BTreeMap::new(),
        };

        SkillPack {
            manifest,
            taxonomy,
            thresholds: thresholds_cfg,
            severity_rubric,
            scam_phrases,
            hate_lexicons,
            regex_sets,
            slm_prompt: "BASE PROMPT".to_string(),
        }
    }

    fn empty_community() -> CommunityOverlay {
        CommunityOverlay {
            overlay_id: "cvguard.overlay.community.test.v1".to_string(),
            version: "1.0.0".to_string(),
            base_pack_id: "cvguard.skill.base.v1".to_string(),
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

    fn empty_jurisdiction() -> JurisdictionOverlay {
        JurisdictionOverlay {
            overlay_id: "cvguard.overlay.jurisdiction.test.v1".to_string(),
            version: "1.0.0".to_string(),
            base_pack_id: "cvguard.skill.base.v1".to_string(),
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
    fn empty_overlay_is_identity_on_thresholds() {
        let base = base_pack();
        let overlay = empty_community();
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        assert_eq!(merged.thresholds, base.thresholds);
        assert_eq!(merged.severity_rubric, base.severity_rubric);
        assert_eq!(merged.scam_phrases, base.scam_phrases);
        assert_eq!(merged.hate_lexicons, base.hate_lexicons);
        assert_eq!(merged.regex_sets, base.regex_sets);
        assert_eq!(merged.slm_prompt, base.slm_prompt);
    }

    #[test]
    fn threshold_override_can_lower_trigger() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "nudity".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.25),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        let entry = merged.thresholds.thresholds["adult"]["nudity"];
        assert_eq!(entry.trigger, 0.25);
        // severe unchanged
        assert_eq!(entry.severe, Some(0.80));
    }

    #[test]
    fn threshold_override_can_clear_severe() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "nudity".to_string(),
                OverlayThresholdEntry {
                    trigger: None,
                    severe: None,
                    clear_severe: true,
                },
            )]),
        );
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        let entry = merged.thresholds.thresholds["adult"]["nudity"];
        assert_eq!(entry.severe, None);
        assert_eq!(entry.trigger, 0.40); // base
    }

    #[test]
    fn threshold_override_can_introduce_new_label() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "graphic_violence".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.30),
                    severe: Some(0.70),
                    clear_severe: false,
                },
            )]),
        );
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        let entry = merged.thresholds.thresholds["adult"]["graphic_violence"];
        assert_eq!(entry.trigger, 0.30);
        assert_eq!(entry.severe, Some(0.70));
    }

    /// Build a base pack whose `hate.slur` weak category carries a
    /// cascade-router routing band (route=0.40 < trigger=0.55).
    fn base_pack_with_route() -> SkillPack {
        let mut base = base_pack();
        let mut thresholds = base.thresholds.thresholds.clone();
        thresholds.insert(
            "hate".to_string(),
            BTreeMap::from([(
                "slur".to_string(),
                ThresholdEntry::new_with_route(0.55, Some(0.85), Some(0.40)).unwrap(),
            )]),
        );
        base.thresholds = ThresholdsConfig::new(thresholds).unwrap();
        base
    }

    #[test]
    fn overlay_retuning_weak_category_preserves_route_band() {
        // An overlay that only re-tunes trigger/severe for a routed
        // weak category MUST NOT drop the baseline routing band — else
        // the encoder=router behaviour silently collapses back to
        // demote-to-SAFE for that category.
        let base = base_pack_with_route();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "hate".to_string(),
            BTreeMap::from([(
                "slur".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.60),
                    severe: Some(0.90),
                    clear_severe: false,
                },
            )]),
        );
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        let entry = merged.thresholds.thresholds["hate"]["slur"];
        assert_eq!(entry.trigger, 0.60);
        assert_eq!(entry.severe, Some(0.90));
        assert_eq!(entry.route, Some(0.40), "route band must survive the merge");
    }

    #[test]
    fn overlay_clearing_severe_preserves_route_band() {
        let base = base_pack_with_route();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "hate".to_string(),
            BTreeMap::from([(
                "slur".to_string(),
                OverlayThresholdEntry {
                    trigger: None,
                    severe: None,
                    clear_severe: true,
                },
            )]),
        );
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        let entry = merged.thresholds.thresholds["hate"]["slur"];
        assert_eq!(entry.severe, None);
        assert_eq!(entry.trigger, 0.55); // base
        assert_eq!(entry.route, Some(0.40), "route band must survive the merge");
    }

    #[test]
    fn overlay_lowering_trigger_below_carried_route_is_rejected() {
        // Lowering trigger under the carried-forward route would make
        // route > trigger, which the canonical constructor rejects —
        // surfacing as a SchemaViolation rather than silently clamping.
        let base = base_pack_with_route();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "hate".to_string(),
            BTreeMap::from([(
                "slur".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.30), // below carried route 0.40
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        let err = apply_community_overlay(&base, &overlay).unwrap_err();
        assert!(matches!(err, SkillPackError::SchemaViolation { .. }));
    }

    #[test]
    fn threshold_override_new_label_without_trigger_fails() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "graphic_violence".to_string(),
                OverlayThresholdEntry {
                    trigger: None,
                    severe: Some(0.7),
                    clear_severe: false,
                },
            )]),
        );
        assert!(matches!(
            apply_community_overlay(&base, &overlay),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn child_safety_can_be_tightened() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "child_safety".to_string(),
            BTreeMap::from([(
                "any_hit".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.15),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        let entry = merged.thresholds.thresholds["child_safety"]["any_hit"];
        assert_eq!(entry.trigger, 0.15);
    }

    #[test]
    fn child_safety_cannot_be_loosened_via_trigger() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "child_safety".to_string(),
            BTreeMap::from([(
                "any_hit".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.40),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        let err = apply_community_overlay(&base, &overlay).unwrap_err();
        assert!(
            matches!(err, SkillPackError::OverlayFloorViolation { ref category, ref label, .. } if category == "child_safety" && label == "any_hit"),
            "expected OverlayFloorViolation, got: {err:?}"
        );
    }

    #[test]
    fn child_safety_cannot_be_loosened_via_severe() {
        let base = base_pack();
        let mut overlay = empty_community();
        // Set severe above the base's severe-floor (which is None,
        // so it falls back to 0.20). Anything above 0.20 must
        // fail.
        overlay.threshold_overrides.insert(
            "child_safety".to_string(),
            BTreeMap::from([(
                "any_hit".to_string(),
                OverlayThresholdEntry {
                    trigger: None,
                    severe: Some(0.30),
                    clear_severe: false,
                },
            )]),
        );
        assert!(matches!(
            apply_community_overlay(&base, &overlay),
            Err(SkillPackError::OverlayFloorViolation { .. })
        ));
    }

    #[test]
    fn child_safety_new_label_above_floor_is_rejected() {
        let base = base_pack();
        let mut overlay = empty_community();
        // Introducing a *new* child_safety label with trigger >
        // 0.20 is a floor violation because the global baseline
        // says child_safety can't be looser than 0.20.
        overlay.threshold_overrides.insert(
            "child_safety".to_string(),
            BTreeMap::from([(
                "exposed_minor".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.30),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        assert!(matches!(
            apply_community_overlay(&base, &overlay),
            Err(SkillPackError::OverlayFloorViolation { .. })
        ));
    }

    #[test]
    fn child_safety_new_label_at_floor_is_allowed() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "child_safety".to_string(),
            BTreeMap::from([(
                "exposed_minor".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.20),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        apply_community_overlay(&base, &overlay).unwrap();
    }

    #[test]
    fn severity_override_can_change_ux_action() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.severity_overrides.push(OverlaySeverityLevel {
            level: 3,
            name: None,
            ux_action: Some("pixelate".to_string()),
            allow_reveal: None,
            allow_forward: None,
            description: None,
        });
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        let lv3 = merged
            .severity_rubric
            .levels
            .iter()
            .find(|lv| lv.level == 3)
            .unwrap();
        assert_eq!(lv3.ux_action, UXAction::Pixelate);
        // name unchanged
        assert_eq!(lv3.name, "medium");
    }

    #[test]
    fn severity_override_fields_fall_through_when_none() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.severity_overrides.push(OverlaySeverityLevel {
            level: 4,
            name: None,
            ux_action: None,
            allow_reveal: None,
            allow_forward: None,
            description: None,
        });
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        let lv4 = merged
            .severity_rubric
            .levels
            .iter()
            .find(|lv| lv.level == 4)
            .unwrap();
        let lv4_base = base
            .severity_rubric
            .levels
            .iter()
            .find(|lv| lv.level == 4)
            .unwrap();
        assert_eq!(lv4, lv4_base);
    }

    #[test]
    fn severity_override_targets_missing_level_fails() {
        let base = base_pack();
        let mut overlay = empty_community();
        // Try to override a level not in base. Note Pydantic-side
        // would let this through field-level (level <= 5) but the
        // merge function rejects via `_apply_severity_overrides`.
        let mut bad_base = base.clone();
        // Drop level 5 to force a missing-level merge.
        bad_base.severity_rubric.levels.retain(|lv| lv.level != 5);
        overlay.severity_overrides.push(OverlaySeverityLevel {
            level: 5,
            name: None,
            ux_action: None,
            allow_reveal: None,
            allow_forward: None,
            description: None,
        });
        assert!(matches!(
            apply_community_overlay(&bad_base, &overlay),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn lexicon_addition_appends_to_existing_key() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.scam_phrase_additions.push(OverlayLexiconAddition {
            key: "en".to_string(),
            language: "en".to_string(),
            entries: vec![LexiconEntry {
                phrase: "wire transfer".to_string(),
                weight: 1.0,
                tags: vec!["scam".to_string()],
            }],
        });
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        let lex = &merged.scam_phrases["en"];
        assert_eq!(lex.entries.len(), 2);
        // Order: base first, then overlay appended (preserves
        // declaration order).
        assert_eq!(lex.entries[0].phrase, "send gift cards");
        assert_eq!(lex.entries[1].phrase, "wire transfer");
    }

    #[test]
    fn lexicon_addition_introduces_new_key() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.hate_lexicon_additions.push(OverlayLexiconAddition {
            key: "fr".to_string(),
            language: "fr".to_string(),
            entries: vec![LexiconEntry {
                phrase: "phrase de test".to_string(),
                weight: 2.0,
                tags: Vec::new(),
            }],
        });
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        assert_eq!(merged.hate_lexicons["fr"].entries.len(), 1);
        assert_eq!(merged.hate_lexicons["fr"].language, "fr");
    }

    #[test]
    fn lexicon_addition_language_mismatch_is_rejected() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.scam_phrase_additions.push(OverlayLexiconAddition {
            key: "en".to_string(),
            language: "fr".to_string(),
            entries: vec![LexiconEntry {
                phrase: "x".to_string(),
                weight: 1.0,
                tags: Vec::new(),
            }],
        });
        assert!(matches!(
            apply_community_overlay(&base, &overlay),
            Err(SkillPackError::SchemaViolation { .. })
        ));
    }

    #[test]
    fn regex_addition_replaces_existing_key() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.regex_additions.push(OverlayRegexAddition {
            key: "pii".to_string(),
            name: "pii".to_string(),
            patterns: vec![RegexPattern {
                name: "ssn".to_string(),
                pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
                description: "SSN".to_string(),
                flags: Vec::new(),
            }],
        });
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        let rs = &merged.regex_sets["pii"];
        assert_eq!(rs.patterns.len(), 1);
        // Overlay replaces base set entirely
        assert_eq!(rs.patterns[0].name, "ssn");
    }

    #[test]
    fn regex_addition_introduces_new_key() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.regex_additions.push(OverlayRegexAddition {
            key: "urls".to_string(),
            name: "urls".to_string(),
            patterns: vec![RegexPattern {
                name: "http".to_string(),
                pattern: r"https?://\S+".to_string(),
                description: "URL".to_string(),
                flags: Vec::new(),
            }],
        });
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        assert_eq!(merged.regex_sets["urls"].patterns.len(), 1);
        // Existing key still there
        assert!(merged.regex_sets.contains_key("pii"));
    }

    #[test]
    fn prompt_suffix_is_joined_with_separator() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.slm_prompt_suffix = "Extra rules: lèse-majesté is severity 5.".to_string();
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        assert_eq!(
            merged.slm_prompt,
            "BASE PROMPT\n---\nExtra rules: lèse-majesté is severity 5.\n"
        );
    }

    #[test]
    fn prompt_suffix_empty_is_noop() {
        let base = base_pack();
        let overlay = empty_community();
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        assert_eq!(merged.slm_prompt, "BASE PROMPT");
    }

    #[test]
    fn prompt_suffix_strips_trailing_newline_on_base() {
        let mut base = base_pack();
        base.slm_prompt = "BASE PROMPT\n\n".to_string();
        let mut overlay = empty_community();
        overlay.slm_prompt_suffix = "S".to_string();
        let merged = apply_community_overlay(&base, &overlay).unwrap();
        assert_eq!(merged.slm_prompt, "BASE PROMPT\n---\nS\n");
    }

    #[test]
    fn jurisdiction_overlay_has_identical_semantics() {
        let base = base_pack();
        let mut com = empty_community();
        let mut jur = empty_jurisdiction();
        com.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "nudity".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.25),
                    severe: Some(0.60),
                    clear_severe: false,
                },
            )]),
        );
        jur.threshold_overrides = com.threshold_overrides.clone();
        let merged_com = apply_community_overlay(&base, &com).unwrap();
        let merged_jur = apply_jurisdiction_overlay(&base, &jur).unwrap();
        // Same inputs, same outputs — modulo overlay_id metadata
        // which the merge doesn't propagate into the SkillPack.
        assert_eq!(merged_com.thresholds, merged_jur.thresholds);
        assert_eq!(merged_com.severity_rubric, merged_jur.severity_rubric);
        assert_eq!(merged_com.scam_phrases, merged_jur.scam_phrases);
        assert_eq!(merged_com.hate_lexicons, merged_jur.hate_lexicons);
        assert_eq!(merged_com.regex_sets, merged_jur.regex_sets);
        assert_eq!(merged_com.slm_prompt, merged_jur.slm_prompt);
    }

    #[test]
    fn apply_is_deterministic_across_invocations() {
        let base = base_pack();
        let mut overlay = empty_community();
        overlay.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "nudity".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.25),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        overlay.scam_phrase_additions.push(OverlayLexiconAddition {
            key: "en".to_string(),
            language: "en".to_string(),
            entries: vec![LexiconEntry {
                phrase: "wire transfer".to_string(),
                weight: 1.0,
                tags: vec!["scam".to_string()],
            }],
        });
        overlay.slm_prompt_suffix = "Extra.".to_string();

        let a = apply_community_overlay(&base, &overlay).unwrap();
        let b = apply_community_overlay(&base, &overlay).unwrap();
        // Use the cloned types for equality. `SkillPack` itself is
        // not PartialEq because of the manifest, so compare the
        // observable fields the resolver consumers care about.
        assert_eq!(a.thresholds, b.thresholds);
        assert_eq!(a.severity_rubric, b.severity_rubric);
        assert_eq!(a.scam_phrases, b.scam_phrases);
        assert_eq!(a.hate_lexicons, b.hate_lexicons);
        assert_eq!(a.regex_sets, b.regex_sets);
        assert_eq!(a.slm_prompt, b.slm_prompt);
    }
}

//! Resolve a base pack + optional overlays into the *effective* pack.
//!
//! Merge order:
//!
//! ```text
//! base pack  ->  jurisdiction overlay  ->  community overlay
//! ```
//!
//! Jurisdiction overlays come first because they encode legal
//! requirements; community overlays come second because they
//! encode preferences. A community overlay can *tighten* further
//! past what a jurisdiction requires, but can never *loosen* a
//! `child_safety` threshold — that hard floor is enforced inside
//! both [`super::overlay::apply_community_overlay`] and
//! [`super::overlay::apply_jurisdiction_overlay`].
//!
//! Overlay selection is driven by
//! `MediaSafetyRequest.context_hints`:
//!
//! * `context_hints["jurisdiction"]`  → which jurisdiction overlay to use
//! * `context_hints["community_type"]` → which community overlay to use
//!
//! The resolver does not concern itself with how the overlays were
//! loaded — that's the loader's job. It just applies the merge
//! order.

use std::collections::BTreeMap;

use super::overlay::{apply_community_overlay, apply_jurisdiction_overlay};
use super::overlay_schema::{CommunityOverlay, JurisdictionOverlay};
use super::schema::SkillPack;
use super::SkillPackError;

/// Result of [`resolve_effective_pack`].
///
/// `effective` is the in-memory [`SkillPack`] the policy
/// interpreter should consult. The optional `*_overlay_id` /
/// `*_overlay_version` fields record which overlays (if any) were
/// applied so telemetry can attribute a decision to a specific
/// overlay version without re-running the resolver.
#[derive(Debug, Clone)]
pub struct ResolvedPack {
    /// The fully-merged pack ready to feed into the policy
    /// interpreter.
    pub effective: SkillPack,
    /// `manifest.pack_id` of the base pack the resolver started
    /// from.
    pub base_pack_id: String,
    /// `manifest.version` of the base pack.
    pub base_pack_version: String,
    /// `overlay_id` of the jurisdiction overlay if one was
    /// applied, else `None`.
    pub jurisdiction_overlay_id: Option<String>,
    /// `version` of the jurisdiction overlay.
    pub jurisdiction_overlay_version: Option<String>,
    /// `overlay_id` of the community overlay if one was applied,
    /// else `None`.
    pub community_overlay_id: Option<String>,
    /// `version` of the community overlay.
    pub community_overlay_version: Option<String>,
}

/// Apply overlays in *(jurisdiction, then community)* order.
///
/// Either or both overlays may be `None` — in which case the
/// matching merge step is skipped. The function is deterministic:
/// same inputs → byte-identical output. Returns
/// [`SkillPackError`] if either merge step fails (e.g. a community
/// overlay would loosen `child_safety` past the jurisdiction floor).
pub fn resolve_effective_pack(
    base: &SkillPack,
    jurisdiction_overlay: Option<&JurisdictionOverlay>,
    community_overlay: Option<&CommunityOverlay>,
) -> Result<ResolvedPack, SkillPackError> {
    let after_jurisdiction = match jurisdiction_overlay {
        Some(jur) => apply_jurisdiction_overlay(base, jur)?,
        None => base.clone(),
    };
    let effective = match community_overlay {
        Some(com) => apply_community_overlay(&after_jurisdiction, com)?,
        None => after_jurisdiction,
    };
    Ok(ResolvedPack {
        effective,
        base_pack_id: base.manifest.pack_id.clone(),
        base_pack_version: base.manifest.version.clone(),
        jurisdiction_overlay_id: jurisdiction_overlay.map(|o| o.overlay_id.clone()),
        jurisdiction_overlay_version: jurisdiction_overlay.map(|o| o.version.clone()),
        community_overlay_id: community_overlay.map(|o| o.overlay_id.clone()),
        community_overlay_version: community_overlay.map(|o| o.version.clone()),
    })
}

/// Pick the overlays referenced by `hints` from the caller's
/// registries.
///
/// `jurisdictions` and `communities` are name-keyed registries the
/// caller maintains (e.g. loaded once at scanner-init time). The
/// function never raises on missing keys — an unknown community
/// or jurisdiction simply yields `None` so the resolver falls
/// through to the base pack. Mirrors Python's
/// `select_overlays_from_hints` semantic for that fall-through.
pub fn select_overlays_from_hints<'a>(
    hints: &BTreeMap<String, String>,
    jurisdictions: &'a BTreeMap<String, JurisdictionOverlay>,
    communities: &'a BTreeMap<String, CommunityOverlay>,
) -> (
    Option<&'a JurisdictionOverlay>,
    Option<&'a CommunityOverlay>,
) {
    let jur = hints.get("jurisdiction").and_then(|k| jurisdictions.get(k));
    let com = hints.get("community_type").and_then(|k| communities.get(k));
    (jur, com)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::policy_interpreter::{
        SeverityLevel, SeverityRubric, ThresholdEntry, ThresholdsConfig, UXAction,
    };
    use crate::skillpack::overlay_schema::{
        OverlayLexiconAddition, OverlaySeverityLevel, OverlayThresholdEntry,
    };
    use crate::skillpack::schema::{
        Lexicon, LexiconEntry, RegexSet, SkillPackManifest, TaxonomyConfig,
    };

    fn base_pack() -> SkillPack {
        let mut thresholds: BTreeMap<String, BTreeMap<String, ThresholdEntry>> = BTreeMap::new();
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
                ThresholdEntry::new(0.50, Some(0.85)).unwrap(),
            )]),
        );
        let thresholds_cfg = ThresholdsConfig::new(thresholds).unwrap();
        let severity_rubric = SeverityRubric::new(vec![
            SeverityLevel::new(0, "safe", UXAction::Clear).unwrap(),
            SeverityLevel::new(1, "low", UXAction::Clear).unwrap(),
            SeverityLevel::new(2, "low_blur", UXAction::BlurTap).unwrap(),
            SeverityLevel::new(3, "medium", UXAction::BlurTap).unwrap(),
            SeverityLevel::new(4, "high", UXAction::Pixelate).unwrap(),
            SeverityLevel::new(5, "severe", UXAction::BlockedCard).unwrap(),
        ])
        .unwrap();

        let manifest = SkillPackManifest {
            pack_id: "cvguard.skill.base.v1".to_string(),
            version: "1.2.0".to_string(),
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
            scam_phrases: BTreeMap::<String, Lexicon>::new(),
            hate_lexicons: BTreeMap::new(),
            regex_sets: BTreeMap::<String, RegexSet>::new(),
            slm_prompt: "BASE".to_string(),
        }
    }

    fn community_lowers_adult() -> CommunityOverlay {
        let mut o = CommunityOverlay {
            overlay_id: "ns.overlay.community.workplace.v1".to_string(),
            version: "1.0.0".to_string(),
            base_pack_id: "cvguard.skill.base.v1".to_string(),
            base_pack_version: "1.2.0".to_string(),
            description: String::new(),
            schema_version: 1,
            threshold_overrides: BTreeMap::new(),
            severity_overrides: Vec::new(),
            scam_phrase_additions: Vec::new(),
            hate_lexicon_additions: Vec::new(),
            regex_additions: Vec::new(),
            slm_prompt_suffix: "WORKPLACE".to_string(),
            overlay_kind: "community".to_string(),
        };
        o.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "nudity".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.30),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        o
    }

    fn jurisdiction_lowers_adult_more() -> JurisdictionOverlay {
        let mut o = JurisdictionOverlay {
            overlay_id: "ns.overlay.jurisdiction.fr.v1".to_string(),
            version: "2.0.0".to_string(),
            base_pack_id: "cvguard.skill.base.v1".to_string(),
            base_pack_version: "1.2.0".to_string(),
            description: String::new(),
            schema_version: 1,
            threshold_overrides: BTreeMap::new(),
            severity_overrides: Vec::new(),
            scam_phrase_additions: Vec::new(),
            hate_lexicon_additions: Vec::new(),
            regex_additions: Vec::new(),
            slm_prompt_suffix: "JURISDICTION".to_string(),
            overlay_kind: "jurisdiction".to_string(),
        };
        // Jurisdiction lowers further to 0.20.
        o.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "nudity".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.20),
                    severe: Some(0.60),
                    clear_severe: false,
                },
            )]),
        );
        o
    }

    #[test]
    fn resolve_with_no_overlays_returns_base() {
        let base = base_pack();
        let resolved = resolve_effective_pack(&base, None, None).unwrap();
        assert_eq!(resolved.effective.thresholds, base.thresholds);
        assert!(resolved.jurisdiction_overlay_id.is_none());
        assert!(resolved.community_overlay_id.is_none());
        assert_eq!(resolved.base_pack_id, "cvguard.skill.base.v1");
        assert_eq!(resolved.base_pack_version, "1.2.0");
    }

    #[test]
    fn resolve_jurisdiction_only_applies_jurisdiction() {
        let base = base_pack();
        let jur = jurisdiction_lowers_adult_more();
        let resolved = resolve_effective_pack(&base, Some(&jur), None).unwrap();
        // Jurisdiction's override wins.
        let entry = resolved.effective.thresholds.thresholds["adult"]["nudity"];
        assert_eq!(entry.trigger, 0.20);
        assert_eq!(entry.severe, Some(0.60));
        assert_eq!(
            resolved.jurisdiction_overlay_id.as_deref(),
            Some("ns.overlay.jurisdiction.fr.v1")
        );
        assert_eq!(
            resolved.jurisdiction_overlay_version.as_deref(),
            Some("2.0.0")
        );
        assert!(resolved.community_overlay_id.is_none());
    }

    #[test]
    fn resolve_community_only_applies_community() {
        let base = base_pack();
        let com = community_lowers_adult();
        let resolved = resolve_effective_pack(&base, None, Some(&com)).unwrap();
        let entry = resolved.effective.thresholds.thresholds["adult"]["nudity"];
        assert_eq!(entry.trigger, 0.30);
        // severe falls through to base
        assert_eq!(entry.severe, Some(0.85));
        assert!(resolved.jurisdiction_overlay_id.is_none());
        assert_eq!(
            resolved.community_overlay_id.as_deref(),
            Some("ns.overlay.community.workplace.v1")
        );
    }

    #[test]
    fn resolve_applies_jurisdiction_then_community_in_order() {
        let base = base_pack();
        let jur = jurisdiction_lowers_adult_more();
        let com = community_lowers_adult(); // tighter trigger=0.30
        let resolved = resolve_effective_pack(&base, Some(&jur), Some(&com)).unwrap();
        // Community is applied *after* jurisdiction. Community
        // tightens trigger to 0.30, which is more permissive than
        // the jurisdiction's 0.20 — so community *raises* trigger
        // from 0.20 → 0.30 on adult.nudity. This is allowed
        // because `adult` is not in PROTECTED_CATEGORIES (only
        // `child_safety` is).
        let entry = resolved.effective.thresholds.thresholds["adult"]["nudity"];
        assert_eq!(entry.trigger, 0.30);
        // severe came from jurisdiction (0.60) and community
        // didn't touch it
        assert_eq!(entry.severe, Some(0.60));
    }

    #[test]
    fn resolve_community_cannot_loosen_jurisdiction_child_safety_floor() {
        let base = base_pack();
        // Jurisdiction tightens child_safety to 0.10.
        let mut jur = JurisdictionOverlay {
            overlay_id: "ns.overlay.jurisdiction.eu.v1".to_string(),
            version: "1.0.0".to_string(),
            base_pack_id: "cvguard.skill.base.v1".to_string(),
            base_pack_version: "1.2.0".to_string(),
            description: String::new(),
            schema_version: 1,
            threshold_overrides: BTreeMap::new(),
            severity_overrides: Vec::new(),
            scam_phrase_additions: Vec::new(),
            hate_lexicon_additions: Vec::new(),
            regex_additions: Vec::new(),
            slm_prompt_suffix: String::new(),
            overlay_kind: "jurisdiction".to_string(),
        };
        jur.threshold_overrides.insert(
            "child_safety".to_string(),
            BTreeMap::from([(
                "any_hit".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.10),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        // Community tries to raise child_safety back to 0.20 —
        // which is above the post-jurisdiction value of 0.10. The
        // floor enforcement compares against the *current* base
        // (after jurisdiction), so this must be rejected.
        let mut com = community_lowers_adult();
        com.threshold_overrides.insert(
            "child_safety".to_string(),
            BTreeMap::from([(
                "any_hit".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.20),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        assert!(matches!(
            resolve_effective_pack(&base, Some(&jur), Some(&com)),
            Err(SkillPackError::OverlayFloorViolation { .. })
        ));
    }

    #[test]
    fn select_overlays_from_hints_returns_none_on_empty_hints() {
        let hints = BTreeMap::new();
        let jurs = BTreeMap::new();
        let coms = BTreeMap::new();
        let (j, c) = select_overlays_from_hints(&hints, &jurs, &coms);
        assert!(j.is_none());
        assert!(c.is_none());
    }

    #[test]
    fn select_overlays_from_hints_returns_none_on_unknown_keys() {
        let hints: BTreeMap<String, String> = BTreeMap::from([
            ("jurisdiction".to_string(), "xx".to_string()),
            ("community_type".to_string(), "yy".to_string()),
        ]);
        let jurs: BTreeMap<String, JurisdictionOverlay> =
            BTreeMap::from([("us".to_string(), jurisdiction_lowers_adult_more())]);
        let coms: BTreeMap<String, CommunityOverlay> =
            BTreeMap::from([("workplace".to_string(), community_lowers_adult())]);
        let (j, c) = select_overlays_from_hints(&hints, &jurs, &coms);
        assert!(j.is_none());
        assert!(c.is_none());
    }

    #[test]
    fn select_overlays_from_hints_resolves_known_keys() {
        let hints: BTreeMap<String, String> = BTreeMap::from([
            ("jurisdiction".to_string(), "us".to_string()),
            ("community_type".to_string(), "workplace".to_string()),
        ]);
        let jurs: BTreeMap<String, JurisdictionOverlay> =
            BTreeMap::from([("us".to_string(), jurisdiction_lowers_adult_more())]);
        let coms: BTreeMap<String, CommunityOverlay> =
            BTreeMap::from([("workplace".to_string(), community_lowers_adult())]);
        let (j, c) = select_overlays_from_hints(&hints, &jurs, &coms);
        assert!(j.is_some());
        assert!(c.is_some());
        assert_eq!(j.unwrap().overlay_id, "ns.overlay.jurisdiction.fr.v1");
        assert_eq!(c.unwrap().overlay_id, "ns.overlay.community.workplace.v1");
    }

    #[test]
    fn select_overlays_from_hints_returns_one_when_only_one_key_present() {
        let hints: BTreeMap<String, String> =
            BTreeMap::from([("jurisdiction".to_string(), "us".to_string())]);
        let jurs: BTreeMap<String, JurisdictionOverlay> =
            BTreeMap::from([("us".to_string(), jurisdiction_lowers_adult_more())]);
        let coms: BTreeMap<String, CommunityOverlay> = BTreeMap::new();
        let (j, c) = select_overlays_from_hints(&hints, &jurs, &coms);
        assert!(j.is_some());
        assert!(c.is_none());
    }

    #[test]
    fn resolved_pack_records_versions() {
        let base = base_pack();
        let jur = jurisdiction_lowers_adult_more();
        let com = community_lowers_adult();
        let resolved = resolve_effective_pack(&base, Some(&jur), Some(&com)).unwrap();
        assert_eq!(
            resolved.jurisdiction_overlay_id.as_deref(),
            Some("ns.overlay.jurisdiction.fr.v1")
        );
        assert_eq!(
            resolved.jurisdiction_overlay_version.as_deref(),
            Some("2.0.0")
        );
        assert_eq!(
            resolved.community_overlay_id.as_deref(),
            Some("ns.overlay.community.workplace.v1")
        );
        assert_eq!(resolved.community_overlay_version.as_deref(), Some("1.0.0"));
        assert_eq!(resolved.base_pack_id, "cvguard.skill.base.v1");
        assert_eq!(resolved.base_pack_version, "1.2.0");
    }

    #[test]
    fn order_matters_when_overlays_touch_same_label() {
        // Pure ordering check: if jurisdiction sets adult.nudity
        // trigger=0.45 (loosening from base 0.50) and community
        // sets it to 0.20 (tightening), the resolved value is
        // 0.20 (community wins because applied last).
        let base = base_pack();
        let mut jur = jurisdiction_lowers_adult_more();
        jur.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "nudity".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.45),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        let mut com = community_lowers_adult();
        com.threshold_overrides.insert(
            "adult".to_string(),
            BTreeMap::from([(
                "nudity".to_string(),
                OverlayThresholdEntry {
                    trigger: Some(0.20),
                    severe: None,
                    clear_severe: false,
                },
            )]),
        );
        let resolved = resolve_effective_pack(&base, Some(&jur), Some(&com)).unwrap();
        assert_eq!(
            resolved.effective.thresholds.thresholds["adult"]["nudity"].trigger,
            0.20
        );
    }

    #[test]
    fn resolve_passes_through_lexicon_and_regex_additions() {
        let base = base_pack();
        let mut com = community_lowers_adult();
        com.scam_phrase_additions.push(OverlayLexiconAddition {
            key: "en".to_string(),
            language: "en".to_string(),
            entries: vec![LexiconEntry {
                phrase: "send btc".to_string(),
                weight: 1.0,
                tags: vec!["scam".to_string()],
            }],
        });
        com.severity_overrides.push(OverlaySeverityLevel {
            level: 5,
            name: None,
            ux_action: Some("blocked_card".to_string()),
            allow_reveal: Some(false),
            allow_forward: Some(false),
            description: None,
        });
        let resolved = resolve_effective_pack(&base, None, Some(&com)).unwrap();
        assert_eq!(resolved.effective.scam_phrases["en"].entries.len(), 1);
        let lv5 = resolved
            .effective
            .severity_rubric
            .levels
            .iter()
            .find(|l| l.level == 5)
            .unwrap();
        assert!(!lv5.allow_reveal);
    }
}

//! Embedded community overlay YAML files (38 communities).
//!
//! Each file is embedded from `files/communities/<name>.yaml`.
//! Use [`community_overlay_yaml`] to look up by community ID (filename without `.yaml`).

/// Get the raw YAML for a community overlay by name (filename without `.yaml`).
///
/// Returns `None` if the name doesn't match any embedded community.
pub fn community_overlay_yaml(name: &str) -> Option<&'static str> {
    match name {
        "adult_only" => Some(include_str!("files/communities/adult_only.yaml")),
        "book_club" => Some(include_str!("files/communities/book_club.yaml")),
        "cooking" => Some(include_str!("files/communities/cooking.yaml")),
        "creative_arts" => Some(include_str!("files/communities/creative_arts.yaml")),
        "dating" => Some(include_str!("files/communities/dating.yaml")),
        "education_higher" => Some(include_str!("files/communities/education_higher.yaml")),
        "emergency_response" => Some(include_str!("files/communities/emergency_response.yaml")),
        "environmental" => Some(include_str!("files/communities/environmental.yaml")),
        "family" => Some(include_str!("files/communities/family.yaml")),
        "fitness" => Some(include_str!("files/communities/fitness.yaml")),
        "gaming" => Some(include_str!("files/communities/gaming.yaml")),
        "health_support" => Some(include_str!("files/communities/health_support.yaml")),
        "hobbyist" => Some(include_str!("files/communities/hobbyist.yaml")),
        "journalism" => Some(include_str!("files/communities/journalism.yaml")),
        "language_learning" => Some(include_str!("files/communities/language_learning.yaml")),
        "legal_support" => Some(include_str!("files/communities/legal_support.yaml")),
        "lgbtq_support" => Some(include_str!("files/communities/lgbtq_support.yaml")),
        "marketplace" => Some(include_str!("files/communities/marketplace.yaml")),
        "mental_health" => Some(include_str!("files/communities/mental_health.yaml")),
        "music" => Some(include_str!("files/communities/music.yaml")),
        "neighborhood" => Some(include_str!("files/communities/neighborhood.yaml")),
        "nonprofit" => Some(include_str!("files/communities/nonprofit.yaml")),
        "open_source" => Some(include_str!("files/communities/open_source.yaml")),
        "parenting" => Some(include_str!("files/communities/parenting.yaml")),
        "pet_owners" => Some(include_str!("files/communities/pet_owners.yaml")),
        "photography" => Some(include_str!("files/communities/photography.yaml")),
        "political" => Some(include_str!("files/communities/political.yaml")),
        "religious" => Some(include_str!("files/communities/religious.yaml")),
        "school" => Some(include_str!("files/communities/school.yaml")),
        "science" => Some(include_str!("files/communities/science.yaml")),
        "seniors" => Some(include_str!("files/communities/seniors.yaml")),
        "sports" => Some(include_str!("files/communities/sports.yaml")),
        "startup" => Some(include_str!("files/communities/startup.yaml")),
        "tech_support" => Some(include_str!("files/communities/tech_support.yaml")),
        "travel" => Some(include_str!("files/communities/travel.yaml")),
        "veterans" => Some(include_str!("files/communities/veterans.yaml")),
        "volunteer" => Some(include_str!("files/communities/volunteer.yaml")),
        "workplace" => Some(include_str!("files/communities/workplace.yaml")),
        _ => None,
    }
}

/// List all community overlay names.
pub fn community_names() -> &'static [&'static str] {
    &[
        "adult_only", "book_club", "cooking", "creative_arts", "dating",
        "education_higher", "emergency_response", "environmental", "family",
        "fitness", "gaming", "health_support", "hobbyist", "journalism",
        "language_learning", "legal_support", "lgbtq_support", "marketplace",
        "mental_health", "music", "neighborhood", "nonprofit", "open_source",
        "parenting", "pet_owners", "photography", "political", "religious",
        "school", "science", "seniors", "sports", "startup", "tech_support",
        "travel", "veterans", "volunteer", "workplace",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_overlay(name: &str) -> serde_yaml::Value {
        let yaml = community_overlay_yaml(name).expect(&format!("{name} should exist"));
        serde_yaml::from_str(yaml).expect(&format!("{name} should parse as YAML"))
    }

    fn find_rule(overlay: &serde_yaml::Value, category: u64) -> Option<serde_yaml::Value> {
        let rules = overlay.get("rules")?.as_sequence()?;
        rules.iter().find(|r| {
            r.get("category").and_then(|v| v.as_u64()) == Some(category)
        }).cloned()
    }

    const REQUIRED_KEYS: &[&str] = &[
        "skill_id", "parent", "schema_version", "signers",
        "community_profile", "rules",
    ];
    const REQUIRED_PROFILE_KEYS: &[&str] = &["kind", "age_mode", "visibility", "set_by"];
    const VALID_AGE_MODES: &[&str] = &["minor_present", "mixed_age", "adult_only"];
    const VALID_ACTIONS: &[&str] = &["label_only", "warn", "strong_warn", "block"];

    #[test]
    fn all_communities_parse_as_mapping() {
        for name in community_names() {
            let val = parse_overlay(name);
            assert!(val.is_mapping(), "{name} must parse to a mapping");
        }
    }

    #[test]
    fn all_communities_have_required_keys() {
        for name in community_names() {
            let val = parse_overlay(name);
            for key in REQUIRED_KEYS {
                assert!(val.get(*key).is_some(), "{name} missing required key: {key}");
            }
        }
    }

    #[test]
    fn all_communities_parent_is_global_baseline() {
        for name in community_names() {
            let val = parse_overlay(name);
            let parent = val.get("parent").and_then(|v| v.as_str());
            assert_eq!(parent, Some("kchat.global.guardrail.baseline"), "{name} parent mismatch");
        }
    }

    #[test]
    fn all_communities_schema_version_is_1() {
        for name in community_names() {
            let val = parse_overlay(name);
            let sv = val.get("schema_version").and_then(|v| v.as_u64());
            assert_eq!(sv, Some(1), "{name} schema_version mismatch");
        }
    }

    #[test]
    fn all_communities_signers_include_trust_and_safety() {
        for name in community_names() {
            let val = parse_overlay(name);
            let signers = val.get("signers").and_then(|v| v.as_sequence());
            assert!(signers.is_some(), "{name} signers should be a sequence");
            let signer_strs: Vec<&str> = signers.unwrap().iter().filter_map(|v| v.as_str()).collect();
            assert!(signer_strs.contains(&"trust_and_safety"), "{name} must include trust_and_safety signer");
        }
    }

    #[test]
    fn all_communities_profile_has_required_fields() {
        for name in community_names() {
            let val = parse_overlay(name);
            let profile = val.get("community_profile").expect(&format!("{name} should have community_profile"));
            for key in REQUIRED_PROFILE_KEYS {
                assert!(profile.get(*key).is_some(), "{name} community_profile missing: {key}");
            }
        }
    }

    #[test]
    fn all_communities_age_mode_valid() {
        for name in community_names() {
            let val = parse_overlay(name);
            let age_mode = val
                .get("community_profile")
                .and_then(|v| v.get("age_mode"))
                .and_then(|v| v.as_str())
                .expect(&format!("{name} should have age_mode"));
            assert!(VALID_AGE_MODES.contains(&age_mode), "{name} has invalid age_mode: {age_mode}");
        }
    }

    #[test]
    fn all_communities_skill_id_format() {
        for name in community_names() {
            let val = parse_overlay(name);
            let skill_id = val.get("skill_id").and_then(|v| v.as_str()).expect(&format!("{name} should have skill_id"));
            assert!(skill_id.starts_with("kchat.community."), "{name} skill_id should start with kchat.community.");
            assert!(skill_id.ends_with(".guardrail.v1"), "{name} skill_id should end with .guardrail.v1");
        }
    }

    #[test]
    fn all_communities_rule_categories_in_taxonomy_range() {
        for name in community_names() {
            let val = parse_overlay(name);
            let rules = val.get("rules").and_then(|v| v.as_sequence()).expect(&format!("{name} rules should be a sequence"));
            for rule in rules {
                let cat = rule.get("category").and_then(|v| v.as_u64());
                assert!(cat.is_some(), "{name}: rule.category must be an integer");
                let cat = cat.unwrap();
                assert!(cat <= 16, "{name}: rule.category={cat} outside 0..16 taxonomy range");
            }
        }
    }

    #[test]
    fn all_communities_rule_actions_valid() {
        for name in community_names() {
            let val = parse_overlay(name);
            let rules = val.get("rules").and_then(|v| v.as_sequence()).expect(&format!("{name} rules should be a sequence"));
            for rule in rules {
                if let Some(action) = rule.get("action").and_then(|v| v.as_str()) {
                    assert!(VALID_ACTIONS.contains(&action), "{name}: invalid action: {action}");
                }
                if let Some(rule_set) = rule.get("rule_set").and_then(|v| v.as_sequence()) {
                    for sub in rule_set {
                        let action = sub.get("action").and_then(|v| v.as_str());
                        if let Some(a) = action {
                            assert!(VALID_ACTIONS.contains(&a), "{name}: invalid sub-rule action: {a}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn total_community_overlay_count_is_38() {
        assert_eq!(community_names().len(), 38);
    }

    // ─── Specific overlay assertions ───

    #[test]
    fn school_age_mode_minor_present_and_blocks_sexual_adult() {
        let val = parse_overlay("school");
        let age_mode = val.get("community_profile").and_then(|v| v.get("age_mode")).and_then(|v| v.as_str());
        assert_eq!(age_mode, Some("minor_present"));
        let sexual = find_rule(&val, 10).expect("school must define a SEXUAL_ADULT rule");
        let action = sexual.get("action").and_then(|v| v.as_str());
        assert_eq!(action, Some("block"));
    }

    #[test]
    fn adult_only_age_mode() {
        let val = parse_overlay("adult_only");
        let age_mode = val.get("community_profile").and_then(|v| v.get("age_mode")).and_then(|v| v.as_str());
        assert_eq!(age_mode, Some("adult_only"));
    }

    #[test]
    fn health_support_loosens_self_harm() {
        let val = parse_overlay("health_support");
        let sh = find_rule(&val, 2).expect("health_support must define SELF_HARM rule");
        let action = sh.get("action").and_then(|v| v.as_str());
        assert_eq!(action, Some("label_only"));
    }

    #[test]
    fn workplace_has_scam_links_counter() {
        let val = parse_overlay("workplace");
        let counters = val.get("group_risk_counters").and_then(|v| v.as_sequence());
        assert!(counters.is_some(), "workplace should have group_risk_counters");
        let ids: Vec<&str> = counters.unwrap().iter()
            .filter_map(|c| c.get("counter_id").and_then(|v| v.as_str()))
            .collect();
        assert!(ids.contains(&"group_scam_links_24h"), "workplace should have group_scam_links_24h counter");
    }

    #[test]
    fn marketplace_tightens_scam_and_illegal_goods() {
        let val = parse_overlay("marketplace");
        let scam = find_rule(&val, 7).expect("marketplace must define SCAM_FRAUD rule");
        assert_eq!(scam.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
        let illegal = find_rule(&val, 12).expect("marketplace must define ILLEGAL_GOODS rule");
        assert_eq!(illegal.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
        let drugs = find_rule(&val, 11).expect("marketplace must define DRUGS_WEAPONS rule");
        assert_eq!(drugs.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
    }

    #[test]
    fn political_tightens_civic_misinfo() {
        let val = parse_overlay("political");
        let age_mode = val.get("community_profile").and_then(|v| v.get("age_mode")).and_then(|v| v.as_str());
        assert_eq!(age_mode, Some("adult_only"));
        let civic = find_rule(&val, 14).expect("political must define MISINFORMATION_CIVIC rule");
        assert_eq!(civic.get("action").and_then(|v| v.as_str()), Some("warn"));
    }

    #[test]
    fn gaming_has_violence_threats_counter() {
        let val = parse_overlay("gaming");
        let counters = val.get("group_risk_counters").and_then(|v| v.as_sequence());
        assert!(counters.is_some(), "gaming should have group_risk_counters");
        let ids: Vec<&str> = counters.unwrap().iter()
            .filter_map(|c| c.get("counter_id").and_then(|v| v.as_str()))
            .collect();
        assert!(ids.contains(&"group_violence_threats_7d"), "gaming should have group_violence_threats_7d counter");
    }

    #[test]
    fn family_age_mode_mixed_and_strong_warn_sexual_adult() {
        let val = parse_overlay("family");
        let age_mode = val.get("community_profile").and_then(|v| v.get("age_mode")).and_then(|v| v.as_str());
        assert_eq!(age_mode, Some("mixed_age"));
        let sex = find_rule(&val, 10).expect("family must define SEXUAL_ADULT rule");
        assert_eq!(sex.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
    }

    #[test]
    fn dating_age_mode_adult_only() {
        let val = parse_overlay("dating");
        let age_mode = val.get("community_profile").and_then(|v| v.get("age_mode")).and_then(|v| v.as_str());
        assert_eq!(age_mode, Some("adult_only"));
        let sex = find_rule(&val, 10).expect("dating must define SEXUAL_ADULT rule");
        assert_eq!(sex.get("action").and_then(|v| v.as_str()), Some("label_only"));
        let scam = find_rule(&val, 7).expect("dating must define SCAM_FRAUD rule");
        assert_eq!(scam.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
    }

    #[test]
    fn mental_health_loosens_self_harm_for_peer_support() {
        let val = parse_overlay("mental_health");
        let age_mode = val.get("community_profile").and_then(|v| v.get("age_mode")).and_then(|v| v.as_str());
        assert_eq!(age_mode, Some("adult_only"));
        let sh = find_rule(&val, 2).expect("mental_health must define SELF_HARM rule");
        assert_eq!(sh.get("action").and_then(|v| v.as_str()), Some("label_only"));
    }

    #[test]
    fn journalism_loosens_extremism_for_news_context() {
        let val = parse_overlay("journalism");
        let age_mode = val.get("community_profile").and_then(|v| v.get("age_mode")).and_then(|v| v.as_str());
        assert_eq!(age_mode, Some("adult_only"));
        let ext = find_rule(&val, 4).expect("journalism must define EXTREMISM rule");
        assert_eq!(ext.get("action").and_then(|v| v.as_str()), Some("label_only"));
    }

    #[test]
    fn seniors_tightens_scam() {
        let val = parse_overlay("seniors");
        let age_mode = val.get("community_profile").and_then(|v| v.get("age_mode")).and_then(|v| v.as_str());
        assert_eq!(age_mode, Some("adult_only"));
        let scam = find_rule(&val, 7).expect("seniors must define SCAM_FRAUD rule");
        assert_eq!(scam.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
        let pii = find_rule(&val, 9).expect("seniors must define PRIVATE_DATA rule");
        assert_eq!(pii.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
    }

    #[test]
    fn religious_tightens_hate() {
        let val = parse_overlay("religious");
        let hate = find_rule(&val, 6).expect("religious must define HATE rule");
        assert_eq!(hate.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
    }

    #[test]
    fn lgbtq_support_strengthens_hate_and_harassment() {
        let val = parse_overlay("lgbtq_support");
        let age_mode = val.get("community_profile").and_then(|v| v.get("age_mode")).and_then(|v| v.as_str());
        assert_eq!(age_mode, Some("adult_only"));
        let hate = find_rule(&val, 6).expect("lgbtq_support must define HATE rule");
        assert_eq!(hate.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
        let har = find_rule(&val, 5).expect("lgbtq_support must define HARASSMENT rule");
        assert_eq!(har.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
    }

    #[test]
    fn emergency_response_tightens_health_misinformation() {
        let val = parse_overlay("emergency_response");
        let health = find_rule(&val, 13).expect("emergency_response must define MISINFORMATION_HEALTH rule");
        assert_eq!(health.get("action").and_then(|v| v.as_str()), Some("strong_warn"));
    }

    #[test]
    fn template_overlay_has_required_keys() {
        let yaml = include_str!("files/communities/_template/overlay.yaml");
        let val: serde_yaml::Value = serde_yaml::from_str(yaml).expect("template should parse");
        for key in REQUIRED_KEYS {
            assert!(val.get(*key).is_some(), "template overlay missing: {key}");
        }
    }
}

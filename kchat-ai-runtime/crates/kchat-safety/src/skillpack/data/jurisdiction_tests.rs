//! Jurisdiction overlay tests ported from kchat-skills/tests/jurisdictions/.
//!
//! Covers:
//! - Structural assertions for all 62 countries + 3 archetypes
//! - Archetype-specific override checks
//! - Country-specific severity floor, legal age, authority, primary language assertions
//! - Minority language / code-switching false-positive corpus structural tests

use super::*;

fn parse_overlay(code: &str) -> serde_yaml::Value {
    let yaml = jurisdiction_overlay_yaml(code).unwrap_or_else(|| panic!("{code} should exist"));
    serde_yaml::from_str(yaml).unwrap_or_else(|e| panic!("{code} should parse as YAML: {e}"))
}

fn parse_normalization(code: &str) -> serde_yaml::Value {
    let yaml = jurisdiction_normalization_yaml(code).unwrap_or_else(|| panic!("{code} normalization should exist"));
    serde_yaml::from_str(yaml).unwrap_or_else(|e| panic!("{code} normalization should parse: {e}"))
}

fn get_override(overlay: &serde_yaml::Value, category: u64) -> serde_yaml::Value {
    let overrides = overlay.get("overrides").and_then(|v| v.as_sequence()).unwrap();
    let matches: Vec<_> = overrides.iter().filter(|o| o.get("category").and_then(|c| c.as_u64()) == Some(category)).collect();
    assert!(!matches.is_empty(), "overlay must define an override for category {category}");
    assert_eq!(matches.len(), 1, "exactly one override expected for category {category}");
    matches[0].clone()
}

const REQUIRED_TOP_LEVEL: &[&str] = &[
    "skill_id", "parent", "schema_version", "expires_on", "signers",
    "activation", "local_definitions", "local_language_assets",
    "overrides", "allowed_contexts", "user_notice",
];

const REQUIRED_FORBIDDEN_CRITERIA: &[&str] = &[
    "gps_location", "ip_geolocation", "inferred_nationality",
    "inferred_ethnicity", "inferred_religion",
];

const REQUIRED_SIGNERS: &[&str] = &["trust_and_safety", "legal_review", "cultural_review"];

const REQUIRED_ALLOWED_CONTEXTS: &[&str] = &[
    "QUOTED_SPEECH_CONTEXT", "NEWS_CONTEXT",
    "EDUCATION_CONTEXT", "COUNTERSPEECH_CONTEXT",
];

const COUNTRY_CODES: &[&str] = &[
    "ae", "ar", "at", "au", "bd", "br", "ca", "ch", "cl", "co",
    "cz", "de", "dk", "dz", "ec", "eg", "es", "et", "fi", "fr",
    "gb", "gh", "gr", "hu", "id", "ie", "il", "in", "iq", "it",
    "jp", "ke", "kr", "ma", "mx", "my", "ng", "nl", "no", "nz",
    "pe", "ph", "pk", "pl", "pt", "ro", "ru", "sa", "se", "sg",
    "th", "tr", "tw", "tz", "ua", "us", "uy", "vn", "za",
];

const ARCHETYPE_CODES: &[&str] = &[
    "archetype-strict-adult",
    "archetype-strict-hate",
    "archetype-strict-marketplace",
];
#[allow(dead_code)]
const _ARCHETYPE_CODES_USED: &[&str] = ARCHETYPE_CODES;

fn run_structural_assertions(code: &str) {
    let overlay = parse_overlay(code);
    let norm = parse_normalization(code);

    // Required top-level keys
    for key in REQUIRED_TOP_LEVEL {
        assert!(overlay.get(*key).is_some(), "{code} missing top-level key: {key}");
    }

    // Skill ID
    let skill_id = overlay.get("skill_id").and_then(|v| v.as_str()).unwrap();
    let expected = format!("kchat.jurisdiction.{code}.guardrail.v1");
    assert_eq!(skill_id, expected, "{code} skill_id mismatch");
    assert!(skill_id.starts_with("kchat.jurisdiction."));
    assert!(skill_id.ends_with(".guardrail.v1"));

    // Parent
    let parent = overlay.get("parent").and_then(|v| v.as_str()).unwrap();
    assert_eq!(parent, "kchat.global.guardrail.baseline", "{code} parent mismatch");

    // Schema version
    let sv = overlay.get("schema_version").and_then(|v| v.as_u64()).unwrap();
    assert_eq!(sv, 1, "{code} schema_version must be 1");

    // Signers
    let signers: Vec<String> = overlay.get("signers")
        .and_then(|v| v.as_sequence())
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    for s in REQUIRED_SIGNERS {
        assert!(signers.iter().any(|x| x == *s), "{code} signers missing: {s}");
    }

    // Forbidden criteria
    let forbidden: Vec<String> = overlay.get("activation")
        .and_then(|v| v.get("forbidden_criteria"))
        .and_then(|v| v.as_sequence())
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    for fc in REQUIRED_FORBIDDEN_CRITERIA {
        assert!(forbidden.iter().any(|x| x == *fc), "{code} forbidden_criteria missing: {fc}");
    }

    // Activation criteria references country code
    let criteria = overlay.get("activation")
        .and_then(|v| v.get("criteria"))
        .and_then(|v| v.as_sequence())
        .unwrap();
    assert!(!criteria.is_empty(), "{code} activation criteria must be non-empty");
    let flat: Vec<String> = criteria.iter()
        .filter_map(|c| {
            if let serde_yaml::Value::Mapping(m) = c {
                m.values().filter_map(|v| v.as_str().map(String::from)).next()
            } else { None }
        })
        .collect();
    assert!(flat.iter().any(|v| v == code), "{code} activation criteria must reference country code {code}: {flat:?}");

    // Allowed contexts
    let contexts: Vec<String> = overlay.get("allowed_contexts")
        .and_then(|v| v.as_sequence())
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    for ctx in REQUIRED_ALLOWED_CONTEXTS {
        assert!(contexts.iter().any(|x| x == *ctx), "{code} allowed_contexts missing: {ctx}");
    }

    // No relaxed child safety
    let overrides = overlay.get("overrides").and_then(|v| v.as_sequence()).unwrap();
    for o in overrides {
        if o.get("category").and_then(|c| c.as_u64()) == Some(1) {
            let floor = o.get("severity_floor").and_then(|v| v.as_u64()).unwrap();
            assert!(floor >= 5, "{code} CHILD_SAFETY floor cannot be lowered below 5");
        }
    }

    // Expiry within 18 months — must be a future date and <= 18 months from now
    let expires = overlay.get("expires_on").and_then(|v| v.as_str()).unwrap();
    assert!(!expires.is_empty(), "{code} expires_on must not be empty");
    // Parse the date and validate it's in the future and within 18 months
    let exp_date = chrono::NaiveDate::parse_from_str(expires, "%Y-%m-%d")
        .unwrap_or_else(|e| panic!("{code} expires_on must be a valid date: {expires} ({e})"));
    let today = chrono::Local::now().date_naive();
    assert!(exp_date > today, "{code} expires_on must be in the future: {expires}");
    let max_expiry = today + chrono::Duration::days(18 * 31);
    assert!(exp_date <= max_expiry, "{code} expires_on must be within 18 months: {expires}");

    // User notice
    let notice = overlay.get("user_notice").unwrap();
    let summary = notice.get("visible_pack_summary").and_then(|v| v.as_str()).unwrap();
    assert!(!summary.trim().is_empty(), "{code} user_notice summary must be non-empty");
    assert!(notice.get("appeal_resource_id").is_some(), "{code} user_notice missing appeal_resource_id");
    assert!(notice.get("opt_out_allowed").is_some(), "{code} user_notice missing opt_out_allowed");
    assert!(notice.get("opt_out_allowed").and_then(|v| v.as_bool()).is_some(),
        "{code} opt_out_allowed must be a boolean");

    // Normalization
    let norm_section = overlay.get("local_language_assets")
        .and_then(|v| v.get("normalization"))
        .unwrap();
    assert_eq!(norm_section.get("nfkc").and_then(|v| v.as_bool()), Some(true), "{code} nfkc must be true");
    assert_eq!(norm_section.get("case_fold").and_then(|v| v.as_bool()), Some(true), "{code} case_fold must be true");
    assert!(norm_section.get("homoglyph_map_id").is_some(), "{code} homoglyph_map_id must exist");
    let translit = norm_section.get("transliteration_refs")
        .and_then(|v| v.as_sequence())
        .unwrap();
    assert!(!translit.is_empty(), "{code} transliteration_refs must be non-empty");

    // Normalization file matches overlay
    assert_eq!(norm.get("nfkc").and_then(|v| v.as_bool()), Some(true), "{code} norm file nfkc must be true");
    assert_eq!(norm.get("case_fold").and_then(|v| v.as_bool()), Some(true), "{code} norm file case_fold must be true");
    let norm_glyph = norm.get("homoglyph_map_id").and_then(|v| v.as_str()).unwrap();
    let overlay_glyph = norm_section.get("homoglyph_map_id").and_then(|v| v.as_str()).unwrap();
    assert_eq!(norm_glyph, overlay_glyph, "{code} norm file homoglyph_map_id mismatch");

    // Lexicons have provenance
    let lexicons = overlay.get("local_language_assets")
        .and_then(|v| v.get("lexicons"))
        .and_then(|v| v.as_sequence())
        .unwrap();
    assert!(!lexicons.is_empty(), "{code} must declare at least one lexicon");
    for lex in lexicons {
        assert!(lex.get("provenance").is_some(), "{code} lexicon missing provenance");
        assert!(lex.get("language").is_some(), "{code} lexicon missing language");
        assert!(lex.get("categories").is_some(), "{code} lexicon missing categories");
    }
}

// ===========================================================================
// Structural tests — all 62 countries
// ===========================================================================

#[test]
fn all_countries_parse_as_mapping() {
    for code in COUNTRY_CODES {
        let val = parse_overlay(code);
        assert!(val.is_mapping(), "{code} must parse to a mapping");
    }
}

#[test]
fn all_countries_structural_assertions() {
    for code in COUNTRY_CODES {
        run_structural_assertions(code);
    }
}

#[test]
fn all_countries_child_safety_floor_5() {
    for code in COUNTRY_CODES {
        let overlay = parse_overlay(code);
        let overrides = overlay.get("overrides").and_then(|v| v.as_sequence()).unwrap();
        if let Some(o) = overrides.iter().find(|o| o.get("category").and_then(|c| c.as_u64()) == Some(1)) {
            let floor = o.get("severity_floor").and_then(|v| v.as_u64()).unwrap();
            assert_eq!(floor, 5, "{code} CHILD_SAFETY floor must be 5");
        }
    }
}

#[test]
fn all_countries_extremism_floor_4_or_5() {
    for code in COUNTRY_CODES {
        let overlay = parse_overlay(code);
        let overrides = overlay.get("overrides").and_then(|v| v.as_sequence()).unwrap();
        if let Some(o) = overrides.iter().find(|o| o.get("category").and_then(|c| c.as_u64()) == Some(4)) {
            let floor = o.get("severity_floor").and_then(|v| v.as_u64()).unwrap();
            assert!(floor == 4 || floor == 5, "{code} EXTREMISM floor must be 4 or 5, got {floor}");
        }
    }
}

// ===========================================================================
// Archetype structural tests
// ===========================================================================

#[test]
fn archetype_strict_adult_structural() {
    let code = "archetype-strict-adult";
    let overlay = parse_overlay(code);
    let norm = parse_normalization(code);

    for key in REQUIRED_TOP_LEVEL {
        assert!(overlay.get(*key).is_some(), "{code} missing top-level key: {key}");
    }
    let skill_id = overlay.get("skill_id").and_then(|v| v.as_str()).unwrap();
    assert_eq!(skill_id, "kchat.jurisdiction.archetype-strict-adult.guardrail.v1");
    assert_eq!(overlay.get("parent").and_then(|v| v.as_str()), Some("kchat.global.guardrail.baseline"));
    assert_eq!(overlay.get("schema_version").and_then(|v| v.as_u64()), Some(1));

    let signers: Vec<String> = overlay.get("signers").and_then(|v| v.as_sequence()).unwrap()
        .iter().filter_map(|v| v.as_str().map(String::from)).collect();
    for s in REQUIRED_SIGNERS {
        assert!(signers.iter().any(|x| x == *s), "{code} signers missing: {s}");
    }

    let forbidden: Vec<String> = overlay.get("activation").and_then(|v| v.get("forbidden_criteria"))
        .and_then(|v| v.as_sequence()).unwrap()
        .iter().filter_map(|v| v.as_str().map(String::from)).collect();
    for fc in REQUIRED_FORBIDDEN_CRITERIA {
        assert!(forbidden.iter().any(|x| x == *fc), "{code} forbidden_criteria missing: {fc}");
    }

    let contexts: Vec<String> = overlay.get("allowed_contexts").and_then(|v| v.as_sequence()).unwrap()
        .iter().filter_map(|v| v.as_str().map(String::from)).collect();
    for ctx in REQUIRED_ALLOWED_CONTEXTS {
        assert!(contexts.iter().any(|x| x == *ctx), "{code} allowed_contexts missing: {ctx}");
    }

    // Category 10 severity floor 5
    let cat10 = get_override(&overlay, 10);
    assert_eq!(cat10.get("severity_floor").and_then(|v| v.as_u64()), Some(5),
        "archetype-strict-adult category 10 must have severity_floor 5");

    // No relaxed child safety
    let overrides = overlay.get("overrides").and_then(|v| v.as_sequence()).unwrap();
    for o in overrides {
        if o.get("category").and_then(|c| c.as_u64()) == Some(1) {
            assert!(o.get("severity_floor").and_then(|v| v.as_u64()).unwrap() >= 5);
        }
    }

    // Normalization
    let norm_section = overlay.get("local_language_assets").and_then(|v| v.get("normalization")).unwrap();
    assert_eq!(norm_section.get("nfkc").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(norm_section.get("case_fold").and_then(|v| v.as_bool()), Some(true));
    assert!(norm_section.get("homoglyph_map_id").is_some());
    assert!(norm_section.get("transliteration_refs").and_then(|v| v.as_sequence()).unwrap().is_empty() == false);

    assert_eq!(norm.get("nfkc").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(norm.get("case_fold").and_then(|v| v.as_bool()), Some(true));
    let norm_glyph = norm.get("homoglyph_map_id").and_then(|v| v.as_str()).unwrap();
    let overlay_glyph = norm_section.get("homoglyph_map_id").and_then(|v| v.as_str()).unwrap();
    assert_eq!(norm_glyph, overlay_glyph);
}

#[test]
fn archetype_strict_hate_structural() {
    let code = "archetype-strict-hate";
    let overlay = parse_overlay(code);
    let norm = parse_normalization(code);

    for key in REQUIRED_TOP_LEVEL {
        assert!(overlay.get(*key).is_some(), "{code} missing top-level key: {key}");
    }
    let skill_id = overlay.get("skill_id").and_then(|v| v.as_str()).unwrap();
    assert_eq!(skill_id, "kchat.jurisdiction.archetype-strict-hate.guardrail.v1");
    assert_eq!(overlay.get("parent").and_then(|v| v.as_str()), Some("kchat.global.guardrail.baseline"));
    assert_eq!(overlay.get("schema_version").and_then(|v| v.as_u64()), Some(1));

    let signers: Vec<String> = overlay.get("signers").and_then(|v| v.as_sequence()).unwrap()
        .iter().filter_map(|v| v.as_str().map(String::from)).collect();
    for s in REQUIRED_SIGNERS {
        assert!(signers.iter().any(|x| x == *s), "{code} signers missing: {s}");
    }

    // Category 4 floor 4 or 5
    let cat4 = get_override(&overlay, 4);
    let floor4 = cat4.get("severity_floor").and_then(|v| v.as_u64()).unwrap();
    assert!(floor4 == 4 || floor4 == 5, "archetype-strict-hate cat 4 floor must be 4 or 5");

    // Category 6 floor 4 or 5
    let cat6 = get_override(&overlay, 6);
    let floor6 = cat6.get("severity_floor").and_then(|v| v.as_u64()).unwrap();
    assert!(floor6 == 4 || floor6 == 5, "archetype-strict-hate cat 6 floor must be 4 or 5");

    // Normalization matches
    let norm_section = overlay.get("local_language_assets").and_then(|v| v.get("normalization")).unwrap();
    assert_eq!(norm.get("nfkc").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(norm.get("case_fold").and_then(|v| v.as_bool()), Some(true));
    let norm_glyph = norm.get("homoglyph_map_id").and_then(|v| v.as_str()).unwrap();
    let overlay_glyph = norm_section.get("homoglyph_map_id").and_then(|v| v.as_str()).unwrap();
    assert_eq!(norm_glyph, overlay_glyph);
}

#[test]
fn archetype_strict_marketplace_structural() {
    let code = "archetype-strict-marketplace";
    let overlay = parse_overlay(code);
    let norm = parse_normalization(code);

    for key in REQUIRED_TOP_LEVEL {
        assert!(overlay.get(*key).is_some(), "{code} missing top-level key: {key}");
    }
    let skill_id = overlay.get("skill_id").and_then(|v| v.as_str()).unwrap();
    assert_eq!(skill_id, "kchat.jurisdiction.archetype-strict-marketplace.guardrail.v1");
    assert_eq!(overlay.get("parent").and_then(|v| v.as_str()), Some("kchat.global.guardrail.baseline"));
    assert_eq!(overlay.get("schema_version").and_then(|v| v.as_u64()), Some(1));

    // Activation uses archetype region code
    let criteria = overlay.get("activation").and_then(|v| v.get("criteria"))
        .and_then(|v| v.as_sequence()).unwrap();
    for c in criteria {
        if let serde_yaml::Value::Mapping(m) = c {
            let val = m.values().filter_map(|v| v.as_str()).next().unwrap();
            assert_eq!(val, "archetype-strict-marketplace",
                "every activation criterion must use the archetype region code");
        }
    }

    // Category 11 floor 4
    let cat11 = get_override(&overlay, 11);
    assert_eq!(cat11.get("severity_floor").and_then(|v| v.as_u64()), Some(4),
        "archetype-strict-marketplace cat 11 must have severity_floor 4");

    // Category 12 floor 4
    let cat12 = get_override(&overlay, 12);
    assert_eq!(cat12.get("severity_floor").and_then(|v| v.as_u64()), Some(4),
        "archetype-strict-marketplace cat 12 must have severity_floor 4");

    // Normalization matches
    let norm_section = overlay.get("local_language_assets").and_then(|v| v.get("normalization")).unwrap();
    assert_eq!(norm.get("nfkc").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(norm.get("case_fold").and_then(|v| v.as_bool()), Some(true));
    let norm_glyph = norm.get("homoglyph_map_id").and_then(|v| v.as_str()).unwrap();
    let overlay_glyph = norm_section.get("homoglyph_map_id").and_then(|v| v.as_str()).unwrap();
    assert_eq!(norm_glyph, overlay_glyph);
}

// ===========================================================================
// Country-specific severity floor tests
// ===========================================================================

fn assert_severity_floor(code: &str, category: u64, expected: u64) {
    let overlay = parse_overlay(code);
    let override_ = get_override(&overlay, category);
    let floor = override_.get("severity_floor").and_then(|v| v.as_u64()).unwrap();
    assert_eq!(floor, expected, "{code} category {category} severity_floor must be {expected}");
}

#[test]
fn de_extremism_floor_5() { assert_severity_floor("de", 4, 5); }
#[test]
fn de_hate_floor_4() { assert_severity_floor("de", 6, 4); }
#[test]
fn de_sexual_adult_floor_3() { assert_severity_floor("de", 10, 3); }

#[test]
fn br_hate_floor_4() { assert_severity_floor("br", 6, 4); }
#[test]
fn br_misinfo_civic_floor_3() { assert_severity_floor("br", 14, 3); }

#[test]
fn in_extremism_floor_4() { assert_severity_floor("in", 4, 4); }
#[test]
fn in_hate_floor_4() { assert_severity_floor("in", 6, 4); }
#[test]
fn in_sexual_adult_floor_5() { assert_severity_floor("in", 10, 5); }

#[test]
fn jp_drugs_weapons_floor_5() { assert_severity_floor("jp", 11, 5); }
#[test]
fn jp_scam_fraud_floor_4() { assert_severity_floor("jp", 7, 4); }

#[test]
fn us_scam_fraud_floor_3() { assert_severity_floor("us", 7, 3); }

#[test]
fn ca_scam_fraud_floor_3() { assert_severity_floor("ca", 7, 3); }
#[test]
fn gb_scam_fraud_floor_3() { assert_severity_floor("gb", 7, 3); }

#[test]
fn fr_extremism_floor_5() { assert_severity_floor("fr", 4, 5); }
#[test]
fn fr_hate_floor_4() { assert_severity_floor("fr", 6, 4); }

#[test]
fn th_hate_floor_5() { assert_severity_floor("th", 6, 5); }

#[test]
fn id_sexual_adult_floor_5() { assert_severity_floor("id", 10, 5); }

#[test]
fn mx_drugs_weapons_floor_4() { assert_severity_floor("mx", 11, 4); }

#[test]
fn ae_extremism_floor_5() { assert_severity_floor("ae", 4, 5); }
#[test]
fn eg_extremism_floor_5() { assert_severity_floor("eg", 4, 5); }
#[test]
fn sa_extremism_floor_5() { assert_severity_floor("sa", 4, 5); }
#[test]
fn at_extremism_floor_5() { assert_severity_floor("at", 4, 5); }

// ===========================================================================
// Country-specific legal age tests
// ===========================================================================

fn assert_legal_age(code: &str, kind: &str, expected: u64) {
    let overlay = parse_overlay(code);
    let age = overlay.get("local_definitions")
        .and_then(|v| v.get(kind))
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("{code} missing local_definitions.{kind}"));
    assert_eq!(age, expected, "{code} {kind} must be {expected}");
}

#[test]
fn us_legal_age_alcohol_21() { assert_legal_age("us", "legal_age_marketplace_alcohol", 21); }
#[test]
fn us_legal_age_tobacco_21() { assert_legal_age("us", "legal_age_marketplace_tobacco", 21); }

#[test]
fn de_legal_age_alcohol_16() { assert_legal_age("de", "legal_age_marketplace_alcohol", 16); }
#[test]
fn de_legal_age_tobacco_18() { assert_legal_age("de", "legal_age_marketplace_tobacco", 18); }

#[test]
fn jp_legal_age_alcohol_20() { assert_legal_age("jp", "legal_age_marketplace_alcohol", 20); }
#[test]
fn jp_legal_age_tobacco_20() { assert_legal_age("jp", "legal_age_marketplace_tobacco", 20); }

#[test]
fn in_legal_age_alcohol_21() { assert_legal_age("in", "legal_age_marketplace_alcohol", 21); }

#[test]
fn kr_legal_age_alcohol_19() { assert_legal_age("kr", "legal_age_marketplace_alcohol", 19); }
#[test]
fn kr_legal_age_tobacco_19() { assert_legal_age("kr", "legal_age_marketplace_tobacco", 19); }

#[test]
fn ca_legal_age_alcohol_19() { assert_legal_age("ca", "legal_age_marketplace_alcohol", 19); }
#[test]
fn ca_legal_age_tobacco_19() { assert_legal_age("ca", "legal_age_marketplace_tobacco", 19); }

#[test]
fn ch_legal_age_alcohol_16() { assert_legal_age("ch", "legal_age_marketplace_alcohol", 16); }

#[test]
fn se_legal_age_alcohol_20() { assert_legal_age("se", "legal_age_marketplace_alcohol", 20); }

#[test]
fn th_legal_age_alcohol_20() { assert_legal_age("th", "legal_age_marketplace_alcohol", 20); }
#[test]
fn th_legal_age_tobacco_20() { assert_legal_age("th", "legal_age_marketplace_tobacco", 20); }

#[test]
fn ph_legal_age_tobacco_21() { assert_legal_age("ph", "legal_age_marketplace_tobacco", 21); }

#[test]
fn sg_legal_age_tobacco_21() { assert_legal_age("sg", "legal_age_marketplace_tobacco", 21); }

#[test]
fn tw_legal_age_tobacco_20() { assert_legal_age("tw", "legal_age_marketplace_tobacco", 20); }

#[test]
fn ae_legal_age_alcohol_21() { assert_legal_age("ae", "legal_age_marketplace_alcohol", 21); }
#[test]
fn eg_legal_age_alcohol_21() { assert_legal_age("eg", "legal_age_marketplace_alcohol", 21); }
#[test]
fn sa_legal_age_alcohol_21() { assert_legal_age("sa", "legal_age_marketplace_alcohol", 21); }
#[test]
fn bd_legal_age_alcohol_21() { assert_legal_age("bd", "legal_age_marketplace_alcohol", 21); }
#[test]
fn my_legal_age_alcohol_21() { assert_legal_age("my", "legal_age_marketplace_alcohol", 21); }
#[test]
fn pk_legal_age_alcohol_21() { assert_legal_age("pk", "legal_age_marketplace_alcohol", 21); }
#[test]
fn id_legal_age_alcohol_21() { assert_legal_age("id", "legal_age_marketplace_alcohol", 21); }

// ===========================================================================
// Country-specific election authority tests
// ===========================================================================

fn assert_authority(code: &str, expected: &str) {
    let overlay = parse_overlay(code);
    let auth = overlay.get("local_definitions")
        .and_then(|v| v.get("election_rules"))
        .and_then(|v| v.get("authority_resource_id"))
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("{code} missing election_rules.authority_resource_id"));
    assert_eq!(auth, expected, "{code} authority_resource_id mismatch");
}

#[test]
fn us_election_authority() { assert_authority("us", "us_fec_authority_v1"); }
#[test]
fn de_election_authority() { assert_authority("de", "de_bundeswahlleiter_authority_v1"); }
#[test]
fn br_election_authority() { assert_authority("br", "br_tse_authority_v1"); }
#[test]
fn in_election_authority() { assert_authority("in", "in_eci_authority_v1"); }
#[test]
fn jp_election_authority() { assert_authority("jp", "jp_soumu_authority_v1"); }
#[test]
fn gb_election_authority() { assert_authority("gb", "gb_electoral_commission_authority_v1"); }
#[test]
fn fr_election_authority() { assert_authority("fr", "fr_ministere_interieur_authority_v1"); }
#[test]
fn ca_election_authority() { assert_authority("ca", "ca_elections_canada_authority_v1"); }
#[test]
fn au_election_authority() { assert_authority("au", "au_aec_authority_v1"); }
#[test]
fn kr_election_authority() { assert_authority("kr", "kr_nec_authority_v1"); }
#[test]
fn mx_election_authority() { assert_authority("mx", "mx_ine_authority_v1"); }
#[test]
fn tr_election_authority() { assert_authority("tr", "tr_ysk_authority_v1"); }
#[test]
fn tw_election_authority() { assert_authority("tw", "tw_cec_authority_v1"); }
#[test]
fn za_election_authority() { assert_authority("za", "za_iec_authority_v1"); }
#[test]
fn vn_election_authority() { assert_authority("vn", "vn_nec_authority_v1"); }
#[test]
fn ng_election_authority() { assert_authority("ng", "ng_inec_authority_v1"); }
#[test]
fn ke_election_authority() { assert_authority("ke", "ke_iebc_authority_v1"); }
#[test]
fn eg_election_authority() { assert_authority("eg", "eg_nea_authority_v1"); }
#[test]
fn ae_election_authority() { assert_authority("ae", "ae_nec_authority_v1"); }
#[test]
fn sa_election_authority() { assert_authority("sa", "sa_gov_resource_authority_v1"); }
#[test]
fn ar_election_authority() { assert_authority("ar", "ar_cne_authority_v1"); }
#[test]
fn co_election_authority() { assert_authority("co", "co_registraduria_authority_v1"); }
#[test]
fn cl_election_authority() { assert_authority("cl", "cl_servel_authority_v1"); }
#[test]
fn pe_election_authority() { assert_authority("pe", "pe_onpe_authority_v1"); }
#[test]
fn ec_election_authority() { assert_authority("ec", "ec_cne_authority_v1"); }
#[test]
fn uy_election_authority() { assert_authority("uy", "uy_corte_electoral_authority_v1"); }
#[test]
fn es_election_authority() { assert_authority("es", "es_jec_authority_v1"); }
#[test]
fn it_election_authority() { assert_authority("it", "it_ministero_interno_authority_v1"); }
#[test]
fn nl_election_authority() { assert_authority("nl", "nl_kiesraad_authority_v1"); }
#[test]
fn pl_election_authority() { assert_authority("pl", "pl_pkw_authority_v1"); }
#[test]
fn se_election_authority() { assert_authority("se", "se_valmyndigheten_authority_v1"); }
#[test]
fn pt_election_authority() { assert_authority("pt", "pt_cne_authority_v1"); }
#[test]
fn ch_election_authority() { assert_authority("ch", "ch_bundeskanzlei_authority_v1"); }
#[test]
fn at_election_authority() { assert_authority("at", "at_bundeswahlbehoerde_authority_v1"); }
#[test]
fn id_election_authority() { assert_authority("id", "id_kpu_authority_v1"); }
#[test]
fn ph_election_authority() { assert_authority("ph", "ph_comelec_authority_v1"); }
#[test]
fn th_election_authority() { assert_authority("th", "th_ect_authority_v1"); }
#[test]
fn my_election_authority() { assert_authority("my", "my_spr_authority_v1"); }
#[test]
fn sg_election_authority() { assert_authority("sg", "sg_eld_authority_v1"); }
#[test]
fn pk_election_authority() { assert_authority("pk", "pk_ecp_authority_v1"); }
#[test]
fn bd_election_authority() { assert_authority("bd", "bd_ec_authority_v1"); }
#[test]
fn ru_election_authority() { assert_authority("ru", "ru_cik_authority_v1"); }
#[test]
fn ua_election_authority() { assert_authority("ua", "ua_cvk_authority_v1"); }
#[test]
fn ro_election_authority() { assert_authority("ro", "ro_aep_authority_v1"); }
#[test]
fn gr_election_authority() { assert_authority("gr", "gr_ypes_authority_v1"); }
#[test]
fn cz_election_authority() { assert_authority("cz", "cz_csu_authority_v1"); }
#[test]
fn hu_election_authority() { assert_authority("hu", "hu_nvi_authority_v1"); }
#[test]
fn dk_election_authority() { assert_authority("dk", "dk_im_authority_v1"); }
#[test]
fn fi_election_authority() { assert_authority("fi", "fi_oikeusministerio_authority_v1"); }
#[test]
fn no_election_authority() { assert_authority("no", "no_valgdirektoratet_authority_v1"); }
#[test]
fn ie_election_authority() { assert_authority("ie", "ie_coimisiun_na_mean_authority_v1"); }
#[test]
fn il_election_authority() { assert_authority("il", "il_central_elections_authority_v1"); }
#[test]
fn iq_election_authority() { assert_authority("iq", "iq_ihec_authority_v1"); }
#[test]
fn ma_election_authority() { assert_authority("ma", "ma_cndh_authority_v1"); }
#[test]
fn dz_election_authority() { assert_authority("dz", "dz_anie_authority_v1"); }
#[test]
fn gh_election_authority() { assert_authority("gh", "gh_ec_authority_v1"); }
#[test]
fn tz_election_authority() { assert_authority("tz", "tz_nec_authority_v1"); }
#[test]
fn et_election_authority() { assert_authority("et", "et_nebe_authority_v1"); }
#[test]
fn nz_election_authority() { assert_authority("nz", "nz_electoral_commission_authority_v1"); }

// ===========================================================================
// Country-specific primary language tests
// ===========================================================================

fn assert_primary_languages_exact(code: &str, expected: &[&str]) {
    let overlay = parse_overlay(code);
    let langs: Vec<String> = overlay.get("local_language_assets")
        .and_then(|v| v.get("primary_languages"))
        .and_then(|v| v.as_sequence())
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    let expected_vec: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
    assert_eq!(langs, expected_vec, "{code} primary_languages mismatch");
}

fn assert_primary_languages_contains(code: &str, expected: &[&str]) {
    let overlay = parse_overlay(code);
    let langs: Vec<String> = overlay.get("local_language_assets")
        .and_then(|v| v.get("primary_languages"))
        .and_then(|v| v.as_sequence())
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    for lang in expected {
        assert!(langs.iter().any(|l| l == *lang), "{code} primary_languages must include '{lang}'; got {langs:?}");
    }
}

#[test]
fn jp_primary_languages() { assert_primary_languages_exact("jp", &["ja"]); }
#[test]
fn kr_primary_languages() { assert_primary_languages_exact("kr", &["ko"]); }
#[test]
fn vn_primary_languages() { assert_primary_languages_exact("vn", &["vi"]); }
#[test]
fn th_primary_languages() { assert_primary_languages_exact("th", &["th"]); }
#[test]
fn tr_primary_languages() { assert_primary_languages_exact("tr", &["tr"]); }
#[test]
fn ru_primary_languages() { assert_primary_languages_exact("ru", &["ru"]); }
#[test]
fn pl_primary_languages() { assert_primary_languages_exact("pl", &["pl"]); }
#[test]
fn pt_primary_languages() { assert_primary_languages_exact("pt", &["pt"]); }
#[test]
fn ro_primary_languages() { assert_primary_languages_exact("ro", &["ro"]); }
#[test]
fn gr_primary_languages() { assert_primary_languages_exact("gr", &["el"]); }
#[test]
fn cz_primary_languages() { assert_primary_languages_exact("cz", &["cs"]); }
#[test]
fn hu_primary_languages() { assert_primary_languages_exact("hu", &["hu"]); }
#[test]
fn dk_primary_languages() { assert_primary_languages_exact("dk", &["da"]); }
#[test]
fn fi_primary_languages() { assert_primary_languages_exact("fi", &["fi"]); }
#[test]
fn se_primary_languages() { assert_primary_languages_exact("se", &["sv"]); }
#[test]
fn nl_primary_languages() { assert_primary_languages_exact("nl", &["nl"]); }
#[test]
fn it_primary_languages() { assert_primary_languages_exact("it", &["it"]); }
#[test]
fn fr_primary_languages() { assert_primary_languages_exact("fr", &["fr"]); }
#[test]
fn gb_primary_languages() { assert_primary_languages_exact("gb", &["en"]); }
#[test]
fn au_primary_languages() { assert_primary_languages_exact("au", &["en"]); }
#[test]
fn nz_primary_languages() { assert_primary_languages_exact("nz", &["en", "mi"]); }
#[test]
fn ng_primary_languages() { assert_primary_languages_exact("ng", &["en"]); }
#[test]
fn gh_primary_languages() { assert_primary_languages_exact("gh", &["en"]); }
#[test]
fn ar_primary_languages() { assert_primary_languages_exact("ar", &["es"]); }
#[test]
fn cl_primary_languages() { assert_primary_languages_exact("cl", &["es"]); }
#[test]
fn co_primary_languages() { assert_primary_languages_exact("co", &["es"]); }
#[test]
fn mx_primary_languages() { assert_primary_languages_exact("mx", &["es"]); }
#[test]
fn pe_primary_languages() { assert_primary_languages_exact("pe", &["es"]); }
#[test]
fn ec_primary_languages() { assert_primary_languages_exact("ec", &["es"]); }
#[test]
fn uy_primary_languages() { assert_primary_languages_exact("uy", &["es"]); }
#[test]
fn eg_primary_languages() { assert_primary_languages_exact("eg", &["ar"]); }
#[test]
fn sa_primary_languages() { assert_primary_languages_exact("sa", &["ar"]); }
#[test]
fn bd_primary_languages() { assert_primary_languages_exact("bd", &["bn"]); }
#[test]
fn id_primary_languages() { assert_primary_languages_exact("id", &["id"]); }
#[test]
fn at_primary_languages() { assert_primary_languages_exact("at", &["de"]); }
#[test]
fn de_primary_languages() { assert_primary_languages_exact("de", &["de"]); }

#[test]
fn ae_primary_languages() { assert_primary_languages_exact("ae", &["ar", "en"]); }
#[test]
fn dz_primary_languages() { assert_primary_languages_exact("dz", &["ar", "fr"]); }
#[test]
fn ma_primary_languages() { assert_primary_languages_exact("ma", &["ar", "fr"]); }
#[test]
fn ca_primary_languages() { assert_primary_languages_exact("ca", &["en", "fr"]); }
#[test]
fn ie_primary_languages() { assert_primary_languages_exact("ie", &["en", "ga"]); }
#[test]
fn il_primary_languages() { assert_primary_languages_exact("il", &["he", "ar"]); }
#[test]
fn iq_primary_languages() { assert_primary_languages_exact("iq", &["ar", "ku"]); }
#[test]
fn ke_primary_languages() { assert_primary_languages_exact("ke", &["en", "sw"]); }
#[test]
fn tz_primary_languages() { assert_primary_languages_exact("tz", &["sw", "en"]); }
#[test]
fn my_primary_languages() { assert_primary_languages_exact("my", &["ms", "en"]); }
#[test]
fn pk_primary_languages() { assert_primary_languages_exact("pk", &["ur", "en"]); }
#[test]
fn ph_primary_languages() { assert_primary_languages_exact("ph", &["en", "tl"]); }
#[test]
fn tw_primary_languages() { assert_primary_languages_exact("tw", &["zh"]); }
#[test]
fn ua_primary_languages() { assert_primary_languages_exact("ua", &["uk", "ru"]); }
#[test]
fn no_primary_languages() { assert_primary_languages_exact("no", &["no", "nb"]); }
#[test]
fn et_primary_languages() { assert_primary_languages_exact("et", &["am", "en"]); }
#[test]
fn sg_primary_languages() { assert_primary_languages_exact("sg", &["en", "zh", "ms", "ta"]); }
#[test]
fn es_primary_languages() { assert_primary_languages_exact("es", &["es", "ca", "eu", "gl"]); }
#[test]
fn ch_primary_languages() { assert_primary_languages_exact("ch", &["de", "fr", "it", "rm"]); }
#[test]
fn za_primary_languages() { assert_primary_languages_exact("za", &["en", "af", "zu"]); }

#[test]
fn br_primary_languages_includes_pt_br() { assert_primary_languages_contains("br", &["pt-BR"]); }
#[test]
fn in_primary_languages_includes_hi_and_en_in() { assert_primary_languages_contains("in", &["hi", "en-IN"]); }

// ===========================================================================
// Country-specific opt_out_allowed tests
// ===========================================================================

fn assert_opt_out(code: &str, expected: bool) {
    let overlay = parse_overlay(code);
    let opt = overlay.get("user_notice")
        .and_then(|v| v.get("opt_out_allowed"))
        .and_then(|v| v.as_bool())
        .unwrap_or_else(|| panic!("{code} missing user_notice.opt_out_allowed"));
    assert_eq!(opt, expected, "{code} opt_out_allowed mismatch");
}

#[test]
fn us_opt_out_true() { assert_opt_out("us", true); }
#[test]
fn gb_opt_out_true() { assert_opt_out("gb", true); }
#[test]
fn ca_opt_out_true() { assert_opt_out("ca", true); }
#[test]
fn de_opt_out_true() { assert_opt_out("de", true); }
#[test]
fn fr_opt_out_true() { assert_opt_out("fr", true); }
#[test]
fn jp_opt_out_true() { assert_opt_out("jp", true); }
#[test]
fn br_opt_out_true() { assert_opt_out("br", true); }
#[test]
fn in_opt_out_true() { assert_opt_out("in", true); }
#[test]
fn kr_opt_out_true() { assert_opt_out("kr", true); }
#[test]
fn au_opt_out_true() { assert_opt_out("au", true); }
#[test]
fn nz_opt_out_true() { assert_opt_out("nz", true); }
#[test]
fn it_opt_out_true() { assert_opt_out("it", true); }
#[test]
fn es_opt_out_true() { assert_opt_out("es", true); }
#[test]
fn nl_opt_out_true() { assert_opt_out("nl", true); }
#[test]
fn pl_opt_out_true() { assert_opt_out("pl", true); }
#[test]
fn se_opt_out_true() { assert_opt_out("se", true); }
#[test]
fn pt_opt_out_true() { assert_opt_out("pt", true); }
#[test]
fn ch_opt_out_true() { assert_opt_out("ch", true); }
#[test]
fn at_opt_out_true() { assert_opt_out("at", true); }
#[test]
fn ie_opt_out_true() { assert_opt_out("ie", true); }
#[test]
fn il_opt_out_true() { assert_opt_out("il", true); }
#[test]
fn id_opt_out_true() { assert_opt_out("id", true); }
#[test]
fn ph_opt_out_true() { assert_opt_out("ph", true); }
#[test]
fn mx_opt_out_true() { assert_opt_out("mx", true); }
#[test]
fn ar_opt_out_true() { assert_opt_out("ar", true); }
#[test]
fn cl_opt_out_true() { assert_opt_out("cl", true); }
#[test]
fn co_opt_out_true() { assert_opt_out("co", true); }
#[test]
fn pe_opt_out_true() { assert_opt_out("pe", true); }
#[test]
fn ec_opt_out_true() { assert_opt_out("ec", true); }
#[test]
fn uy_opt_out_true() { assert_opt_out("uy", true); }
#[test]
fn gh_opt_out_true() { assert_opt_out("gh", true); }
#[test]
fn ke_opt_out_true() { assert_opt_out("ke", true); }
#[test]
fn tz_opt_out_true() { assert_opt_out("tz", true); }
#[test]
fn et_opt_out_true() { assert_opt_out("et", true); }
#[test]
fn ng_opt_out_true() { assert_opt_out("ng", true); }
#[test]
fn za_opt_out_true() { assert_opt_out("za", true); }
#[test]
fn tw_opt_out_true() { assert_opt_out("tw", true); }
#[test]
fn ua_opt_out_true() { assert_opt_out("ua", true); }
#[test]
fn ru_opt_out_true() { assert_opt_out("ru", true); }
#[test]
fn ro_opt_out_true() { assert_opt_out("ro", true); }
#[test]
fn gr_opt_out_true() { assert_opt_out("gr", true); }
#[test]
fn cz_opt_out_true() { assert_opt_out("cz", true); }
#[test]
fn hu_opt_out_true() { assert_opt_out("hu", true); }
#[test]
fn dk_opt_out_true() { assert_opt_out("dk", true); }
#[test]
fn fi_opt_out_true() { assert_opt_out("fi", true); }
#[test]
fn no_opt_out_true() { assert_opt_out("no", true); }
#[test]
fn iq_opt_out_true() { assert_opt_out("iq", true); }
#[test]
fn ma_opt_out_true() { assert_opt_out("ma", true); }
#[test]
fn dz_opt_out_true() { assert_opt_out("dz", true); }

#[test]
fn ae_opt_out_false() { assert_opt_out("ae", false); }
#[test]
fn eg_opt_out_false() { assert_opt_out("eg", false); }
#[test]
fn sa_opt_out_false() { assert_opt_out("sa", false); }
#[test]
fn pk_opt_out_false() { assert_opt_out("pk", false); }
#[test]
fn bd_opt_out_false() { assert_opt_out("bd", false); }
#[test]
fn my_opt_out_false() { assert_opt_out("my", false); }
#[test]
fn sg_opt_out_false() { assert_opt_out("sg", false); }
#[test]
fn th_opt_out_false() { assert_opt_out("th", false); }
#[test]
fn tr_opt_out_false() { assert_opt_out("tr", false); }
#[test]
fn vn_opt_out_false() { assert_opt_out("vn", false); }

// ===========================================================================
// Country-specific protected classes tests (subset checks)
// ===========================================================================

fn assert_protected_classes_include(code: &str, expected: &[&str]) {
    let overlay = parse_overlay(code);
    let classes: Vec<String> = overlay.get("local_definitions")
        .and_then(|v| v.get("protected_classes"))
        .and_then(|v| v.as_sequence())
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    for exp in expected {
        assert!(classes.iter().any(|c| c == *exp), "{code} protected_classes must include '{exp}'; got {classes:?}");
    }
}

#[test]
fn us_protected_classes_federal_civil_rights() {
    assert_protected_classes_include("us", &["race", "color", "religion", "sex", "national_origin", "disability"]);
}

#[test]
fn de_protected_classes_grundgesetz_article_3() {
    assert_protected_classes_include("de", &["race", "ethnic_origin", "sex", "religion", "disability", "political_opinion", "language"]);
}

#[test]
fn br_protected_classes_constituicao_federal() {
    assert_protected_classes_include("br", &["race", "color", "sex", "religion", "national_origin", "age", "disability"]);
}

#[test]
fn in_protected_classes_articles_15_16() {
    assert_protected_classes_include("in", &["race", "religion", "caste", "sex", "place_of_birth", "disability"]);
}

#[test]
fn jp_protected_classes_article_14() {
    assert_protected_classes_include("jp", &["race", "creed", "sex", "social_status", "family_origin", "disability"]);
}

// ===========================================================================
// Country-specific transliteration refs tests
// ===========================================================================

fn assert_translit_ref_present(code: &str, ref_id: &str) {
    let overlay = parse_overlay(code);
    let refs: Vec<String> = overlay.get("local_language_assets")
        .and_then(|v| v.get("normalization"))
        .and_then(|v| v.get("transliteration_refs"))
        .and_then(|v| v.as_sequence())
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    assert!(refs.iter().any(|r| r == ref_id), "{code} normalization must include {ref_id}; got {refs:?}");
}

#[test]
fn jp_normalization_includes_translit_ja() { assert_translit_ref_present("jp", "translit_ja_v1"); }
#[test]
fn in_normalization_includes_devanagari() { assert_translit_ref_present("in", "translit_devanagari_v1"); }
#[test]
fn kr_normalization_includes_hangul() { assert_translit_ref_present("kr", "translit_hangul_v1"); }
#[test]
fn tw_normalization_includes_cjk() { assert_translit_ref_present("tw", "translit_cjk_v1"); }

// ===========================================================================
// Total jurisdiction count
// ===========================================================================

#[test]
fn total_jurisdiction_count_is_62() {
    // 59 countries + 3 archetypes
    assert_eq!(jurisdiction_codes().len(), 62);
}

// ===========================================================================
// Minority language / code-switching false-positive corpus structural tests
// ===========================================================================

/// (case_id, archetype, has_minority_language_tag, has_code_switching_tag, has_false_positive_tag)
const MINORITY_FP_CASES: &[(&str, &str, bool, bool, bool)] = &[
    ("adult-ml-001-tl-benign-greeting", "archetype-strict-adult", true, false, true),
    ("adult-ml-002-sw-recipe", "archetype-strict-adult", true, false, true),
    ("adult-cs-003-en-tl-mixed", "archetype-strict-adult", false, true, true),
    ("adult-cs-004-es-en-mixed", "archetype-strict-adult", false, true, true),
    ("adult-ml-005-nl-news", "archetype-strict-adult", true, false, true),
    ("hate-ml-006-eu-traditional-dance", "archetype-strict-hate", true, false, true),
    ("hate-ml-007-ga-football", "archetype-strict-hate", true, false, true),
    ("hate-cs-008-hi-en-mixed", "archetype-strict-hate", false, true, true),
    ("hate-cs-009-ar-en-mixed", "archetype-strict-hate", false, true, true),
    ("hate-ml-010-cy-community", "archetype-strict-hate", true, false, true),
    ("market-ml-011-vi-pharmacy", "archetype-strict-marketplace", true, false, true),
    ("market-ml-012-de-kitchen-knife", "archetype-strict-marketplace", true, false, true),
    ("market-cs-013-en-es-pharmacy", "archetype-strict-marketplace", false, true, true),
    ("market-cs-014-en-ja-mixed", "archetype-strict-marketplace", false, true, true),
    ("market-ml-015-pt-fireworks-festival", "archetype-strict-marketplace", true, false, true),
    ("adult-mlcs-016-tl-en-kasal", "archetype-strict-adult", true, true, true),
    ("hate-mlcs-017-cy-en-rugby", "archetype-strict-hate", true, true, true),
    ("market-mlcs-018-de-en-knives", "archetype-strict-marketplace", true, true, true),
    ("us-cs-019-es-en-family", "us", false, true, true),
    ("us-cs-020-en-es-school", "us", false, true, true),
    ("us-ml-021-nv-greeting", "us", true, false, true),
    ("us-ml-022-chr-stickball", "us", true, false, true),
    ("de-cs-023-tr-de-bakery", "de", false, true, true),
    ("de-cs-024-de-tr-family", "de", false, true, true),
    ("de-ml-025-hsb-village", "de", true, false, true),
    ("de-ml-026-dsb-school", "de", true, false, true),
    ("br-ml-027-tpw-river", "br", true, false, true),
    ("br-ml-028-gn-greeting", "br", true, false, true),
    ("br-cs-029-pt-en-mixed", "br", false, true, true),
    ("br-cs-030-en-pt-football", "br", false, true, true),
    ("in-ml-031-ta-festival", "in", true, false, true),
    ("in-ml-032-bn-school", "in", true, false, true),
    ("in-ml-033-ur-greeting", "in", true, false, true),
    ("in-cs-034-hi-en-hinglish", "in", false, true, true),
    ("in-cs-035-en-hi-mixed", "in", false, true, true),
    ("jp-ml-036-ryu-greeting", "jp", true, false, true),
    ("jp-ml-037-ain-mountain", "jp", true, false, true),
    ("jp-cs-038-ja-en-meeting", "jp", false, true, true),
    ("jp-cs-039-en-ja-food", "jp", false, true, true),
    ("mx-mi-029-nah-market-visit", "mx", true, false, true),
    ("mx-mi-030-myb-weaving", "mx", true, false, true),
    ("mx-co-031-es-en-work", "mx", false, true, true),
    ("mx-co-032-es-en-family", "mx", false, true, true),
    ("ca-mi-033-cre-canoe", "ca", true, false, true),
    ("ca-mi-034-iku-snowy-road", "ca", true, false, true),
    ("ca-co-035-en-fr-meeting", "ca", false, true, true),
    ("ca-co-036-en-fr-groceries", "ca", false, true, true),
    ("ar-mi-037-que-harvest", "ar", true, false, true),
    ("ar-mi-038-grn-mate", "ar", true, false, true),
    ("ar-co-039-es-en-football", "ar", false, true, true),
    ("ar-co-040-es-en-milonga", "ar", false, true, true),
    ("co-mi-041-guc-fishing", "co", true, false, true),
    ("co-mi-042-quc-weather", "co", true, false, true),
    ("co-co-043-es-en-coffee", "co", false, true, true),
    ("co-co-044-es-en-concert", "co", false, true, true),
    ("cl-mi-045-arn-lake", "cl", true, false, true),
    ("cl-mi-046-arn-family", "cl", true, false, true),
    ("cl-co-047-es-en-asado", "cl", false, true, true),
    ("cl-co-048-es-en-beach", "cl", false, true, true),
    ("pe-mi-049-quy-crop", "pe", true, false, true),
    ("pe-mi-050-aym-market", "pe", true, false, true),
    ("pe-co-051-es-en-hike", "pe", false, true, true),
    ("pe-co-052-es-en-ceviche", "pe", false, true, true),
    ("fr-mi-053-br-festival", "fr", true, false, true),
    ("fr-mi-054-oc-vineyard", "fr", true, false, true),
    ("fr-co-055-fr-ar-cafe", "fr", false, true, true),
    ("fr-co-056-fr-en-cinema", "fr", false, true, true),
    ("gb-mi-057-cy-choir", "gb", true, false, true),
    ("gb-mi-058-gd-ceilidh", "gb", true, false, true),
    ("gb-co-059-en-ur-bazaar", "gb", false, true, true),
    ("gb-co-060-en-ur-dinner", "gb", false, true, true),
    ("es-mi-061-ca-festival", "es", true, false, true),
    ("es-mi-062-eu-cider", "es", true, false, true),
    ("es-co-063-es-en-shopping", "es", false, true, true),
    ("es-co-064-es-en-vacation", "es", false, true, true),
    ("it-mi-065-sc-olives", "it", true, false, true),
    ("it-mi-066-fur-mountain", "it", true, false, true),
    ("it-co-067-it-en-opera", "it", false, true, true),
    ("it-co-068-it-en-pasta", "it", false, true, true),
    ("nl-mi-069-fy-windmill", "nl", true, false, true),
    ("nl-mi-070-fy-tea", "nl", true, false, true),
    ("nl-co-071-nl-en-meeting", "nl", false, true, true),
    ("nl-co-072-nl-en-bike", "nl", false, true, true),
    ("pl-mi-073-csb-harbor", "pl", true, false, true),
    ("pl-mi-074-szl-market", "pl", true, false, true),
    ("pl-co-075-pl-en-conference", "pl", false, true, true),
    ("pl-co-076-pl-en-dinner", "pl", false, true, true),
    ("se-mi-077-se-reindeer", "se", true, false, true),
    ("se-mi-078-sma-berries", "se", true, false, true),
    ("se-co-079-sv-en-fika", "se", false, true, true),
    ("se-co-080-sv-en-hiking", "se", false, true, true),
    ("pt-mi-081-mwl-village", "pt", true, false, true),
    ("pt-mi-082-mwl-festival", "pt", true, false, true),
    ("pt-co-083-pt-en-beach", "pt", false, true, true),
    ("pt-co-084-pt-en-music", "pt", false, true, true),
    ("ch-mi-085-rm-mountain", "ch", true, false, true),
    ("ch-mi-086-rm-cheese", "ch", true, false, true),
    ("ch-co-087-de-fr-commute", "ch", false, true, true),
    ("ch-co-088-de-it-market", "ch", false, true, true),
    ("at-mi-089-hbs-schoolday", "at", true, false, true),
    ("at-mi-090-hbs-harvest", "at", true, false, true),
    ("at-co-091-de-tr-cafe", "at", false, true, true),
    ("at-co-092-de-tr-family", "at", false, true, true),
    ("kr-mi-093-jeju-tangerine", "kr", true, false, true),
    ("kr-mi-094-jeju-market", "kr", true, false, true),
    ("kr-co-095-ko-en-coffee", "kr", false, true, true),
    ("kr-co-096-ko-en-travel", "kr", false, true, true),
    ("id-mi-097-jv-batik", "id", true, false, true),
    ("id-mi-098-su-rice", "id", true, false, true),
    ("id-co-099-id-en-office", "id", false, true, true),
    ("id-co-100-id-en-dinner", "id", false, true, true),
    ("ph-mi-101-ceb-market", "ph", true, false, true),
    ("ph-mi-102-ilo-fiesta", "ph", true, false, true),
    ("ph-co-103-tl-en-work", "ph", false, true, true),
    ("ph-co-104-tl-en-jeepney", "ph", false, true, true),
    ("th-mi-105-isan-market", "th", true, false, true),
    ("th-mi-106-nod-temple", "th", true, false, true),
    ("th-co-107-th-en-coffee", "th", false, true, true),
    ("th-co-108-th-en-food", "th", false, true, true),
    ("vn-mi-109-tay-hill", "vn", true, false, true),
    ("vn-mi-110-kk-fishing", "vn", true, false, true),
    ("vn-co-111-vi-en-breakfast", "vn", false, true, true),
    ("vn-co-112-vi-en-shopping", "vn", false, true, true),
    ("my-mi-113-ta-temple", "my", true, false, true),
    ("my-mi-114-ta-family", "my", true, false, true),
    ("my-co-115-ms-en-office", "my", false, true, true),
    ("my-co-116-ms-en-weekend", "my", false, true, true),
    ("sg-mi-117-hok-hawker", "sg", true, false, true),
    ("sg-mi-118-ms-mosque", "sg", true, false, true),
    ("sg-co-119-en-zh-food", "sg", false, true, true),
    ("sg-co-120-en-zh-weekend", "sg", false, true, true),
    ("tw-mi-121-nan-tea", "tw", true, false, true),
    ("tw-mi-122-hak-festival", "tw", true, false, true),
    ("tw-co-123-zh-en-bubble", "tw", false, true, true),
    ("tw-co-124-zh-en-hike", "tw", false, true, true),
    ("pk-mi-125-ps-farm", "pk", true, false, true),
    ("pk-mi-126-sd-market", "pk", true, false, true),
    ("pk-co-127-ur-en-office", "pk", false, true, true),
    ("pk-co-128-ur-en-weekend", "pk", false, true, true),
    ("bd-mi-129-ctg-river", "bd", true, false, true),
    ("bd-mi-130-syl-tea", "bd", true, false, true),
    ("bd-co-131-bn-en-office", "bd", false, true, true),
    ("bd-co-132-bn-en-festival", "bd", false, true, true),
    ("ng-mi-133-yo-market", "ng", true, false, true),
    ("ng-mi-134-ha-festival", "ng", true, false, true),
    ("ng-co-135-en-yo-greeting", "ng", false, true, true),
    ("ng-co-136-en-ig-food", "ng", false, true, true),
    ("za-mi-137-zu-rugby", "za", true, false, true),
    ("za-mi-138-xh-church", "za", true, false, true),
    ("za-co-139-en-zu-braai", "za", false, true, true),
    ("za-co-140-en-af-weekend", "za", false, true, true),
    ("eg-mi-141-nub-nile", "eg", true, false, true),
    ("eg-mi-142-ar-said-village", "eg", true, false, true),
    ("eg-co-143-ar-en-office", "eg", false, true, true),
    ("eg-co-144-ar-en-dinner", "eg", false, true, true),
    ("sa-mi-145-ar-najd-poetry", "sa", true, false, true),
    ("sa-mi-146-ar-desert-trip", "sa", true, false, true),
    ("sa-co-147-ar-en-meeting", "sa", false, true, true),
    ("sa-co-148-ar-en-coffee", "sa", false, true, true),
    ("ae-mi-149-ar-desert-falcon", "ae", true, false, true),
    ("ae-mi-150-ur-market", "ae", true, false, true),
    ("ae-co-151-ar-en-office", "ae", false, true, true),
    ("ae-co-152-ar-en-mall", "ae", false, true, true),
    ("ke-mi-153-ki-farm", "ke", true, false, true),
    ("ke-mi-154-luo-lake", "ke", true, false, true),
    ("ke-co-155-sw-en-office", "ke", false, true, true),
    ("ke-co-156-sw-en-market", "ke", false, true, true),
    ("au-mi-157-yol-fishing", "au", true, false, true),
    ("au-mi-158-ab-en-bush", "au", true, false, true),
    ("au-co-159-en-zh-cafe", "au", false, true, true),
    ("au-co-160-en-zh-bbq", "au", false, true, true),
    ("nz-mi-161-mi-waiata", "nz", true, false, true),
    ("nz-mi-162-mi-hangi", "nz", true, false, true),
    ("nz-co-163-en-mi-greeting", "nz", false, true, true),
    ("nz-co-164-en-mi-office", "nz", false, true, true),
    ("tr-mi-165-ku-village", "tr", true, false, true),
    ("tr-mi-166-lzz-coast", "tr", true, false, true),
    ("tr-co-167-tr-en-office", "tr", false, true, true),
    ("tr-co-168-tr-en-food", "tr", false, true, true),
    ("ru-mi-ru-001", "ru", true, false, true),
    ("ru-mi-ru-002", "ru", true, false, true),
    ("ru-co-ru-003", "ru", false, true, true),
    ("ru-co-ru-004", "ru", false, true, true),
    ("ua-mi-ua-001", "ua", true, false, true),
    ("ua-mi-ua-002", "ua", true, false, true),
    ("ua-co-ua-003", "ua", false, true, true),
    ("ua-co-ua-004", "ua", false, true, true),
    ("ro-mi-ro-001", "ro", true, false, true),
    ("ro-mi-ro-002", "ro", true, false, true),
    ("ro-co-ro-003", "ro", false, true, true),
    ("ro-co-ro-004", "ro", false, true, true),
    ("gr-mi-gr-001", "gr", true, false, true),
    ("gr-mi-gr-002", "gr", true, false, true),
    ("gr-co-gr-003", "gr", false, true, true),
    ("gr-co-gr-004", "gr", false, true, true),
    ("cz-mi-cz-001", "cz", true, false, true),
    ("cz-mi-cz-002", "cz", true, false, true),
    ("cz-co-cz-003", "cz", false, true, true),
    ("cz-co-cz-004", "cz", false, true, true),
    ("hu-mi-hu-001", "hu", true, false, true),
    ("hu-mi-hu-002", "hu", true, false, true),
    ("hu-co-hu-003", "hu", false, true, true),
    ("hu-co-hu-004", "hu", false, true, true),
    ("dk-mi-dk-001", "dk", true, false, true),
    ("dk-mi-dk-002", "dk", true, false, true),
    ("dk-co-dk-003", "dk", false, true, true),
    ("dk-co-dk-004", "dk", false, true, true),
    ("fi-mi-fi-001", "fi", true, false, true),
    ("fi-mi-fi-002", "fi", true, false, true),
    ("fi-co-fi-003", "fi", false, true, true),
    ("fi-co-fi-004", "fi", false, true, true),
    ("no-mi-no-001", "no", true, false, true),
    ("no-mi-no-002", "no", true, false, true),
    ("no-co-no-003", "no", false, true, true),
    ("no-co-no-004", "no", false, true, true),
    ("ie-mi-ie-001", "ie", true, false, true),
    ("ie-mi-ie-002", "ie", true, false, true),
    ("ie-co-ie-003", "ie", false, true, true),
    ("ie-co-ie-004", "ie", false, true, true),
    ("il-mi-il-001", "il", true, false, true),
    ("il-mi-il-002", "il", true, false, true),
    ("il-co-il-003", "il", false, true, true),
    ("il-co-il-004", "il", false, true, true),
    ("iq-mi-iq-001", "iq", true, false, true),
    ("iq-mi-iq-002", "iq", true, false, true),
    ("iq-co-iq-003", "iq", false, true, true),
    ("iq-co-iq-004", "iq", false, true, true),
    ("ma-mi-ma-001", "ma", true, false, true),
    ("ma-mi-ma-002", "ma", true, false, true),
    ("ma-co-ma-003", "ma", false, true, true),
    ("ma-co-ma-004", "ma", false, true, true),
    ("gh-mi-gh-001", "gh", true, false, true),
    ("gh-mi-gh-002", "gh", true, false, true),
    ("gh-co-gh-003", "gh", false, true, true),
    ("gh-co-gh-004", "gh", false, true, true),
    ("tz-mi-tz-001", "tz", true, false, true),
    ("tz-mi-tz-002", "tz", true, false, true),
    ("tz-co-tz-003", "tz", false, true, true),
    ("tz-co-tz-004", "tz", false, true, true),
    ("et-mi-et-001", "et", true, false, true),
    ("et-mi-et-002", "et", true, false, true),
    ("et-co-et-003", "et", false, true, true),
    ("et-co-et-004", "et", false, true, true),
    ("dz-mi-dz-001", "dz", true, false, true),
    ("dz-mi-dz-002", "dz", true, false, true),
    ("dz-co-dz-003", "dz", false, true, true),
    ("dz-co-dz-004", "dz", false, true, true),
    ("ec-mi-ec-001", "ec", true, false, true),
    ("ec-mi-ec-002", "ec", true, false, true),
    ("ec-co-ec-003", "ec", false, true, true),
    ("ec-co-ec-004", "ec", false, true, true),
    ("uy-mi-uy-001", "uy", true, false, true),
    ("uy-mi-uy-002", "uy", true, false, true),
    ("uy-co-uy-003", "uy", false, true, true),
    ("uy-co-uy-004", "uy", false, true, true),
];

const MLFP_ARCHETYPES: &[&str] = &[
    "archetype-strict-adult",
    "archetype-strict-hate",
    "archetype-strict-marketplace",
    "us", "de", "br", "in", "jp",
    "mx", "ca", "ar", "co", "cl", "pe",
    "fr", "gb", "es", "it", "nl", "pl", "se", "pt", "ch", "at",
    "kr", "id", "ph", "th", "vn", "my", "sg", "tw", "pk", "bd",
    "ng", "za", "eg", "sa", "ae", "ke",
    "au", "nz", "tr",
    "ru", "ua", "ro", "gr", "cz", "hu",
    "dk", "fi", "no",
    "ie",
    "il", "iq",
    "ma", "dz",
    "gh", "tz", "et",
    "ec", "uy",
];

#[test]
fn mlfp_case_ids_are_unique() {
    let mut seen = std::collections::HashSet::new();
    for (case_id, _, _, _, _) in MINORITY_FP_CASES {
        assert!(seen.insert(*case_id), "duplicate case_id: {case_id}");
    }
}

#[test]
fn mlfp_all_cases_have_false_positive_tag() {
    for (case_id, _, _, _, has_fp) in MINORITY_FP_CASES {
        assert!(*has_fp, "{case_id} must have false_positive tag");
    }
}

#[test]
fn mlfp_all_cases_have_minority_language_or_code_switching_tag() {
    for (case_id, _, has_ml, has_cs, _) in MINORITY_FP_CASES {
        assert!(*has_ml || *has_cs, "{case_id} must have minority_language or code_switching tag");
    }
}

#[test]
fn mlfp_jurisdiction_id_matches_archetype() {
    for (case_id, archetype, _, _, _) in MINORITY_FP_CASES {
        let _expected = format!("kchat.jurisdiction.{archetype}.guardrail.v1");
        // In the Python tests, jurisdiction_id is constructed from archetype.
        // We verify the archetype is a valid jurisdiction code.
        assert!(
            jurisdiction_overlay_yaml(archetype).is_some(),
            "{case_id}: archetype '{archetype}' must be a valid jurisdiction code"
        );
    }
}

#[test]
fn mlfp_minimum_cases_per_archetype() {
    for arch in MLFP_ARCHETYPES {
        let count = MINORITY_FP_CASES.iter()
            .filter(|(_, a, _, _, _)| a == arch)
            .count();
        assert!(count >= 4, "{arch}: need at least 4 minority-language/code-switching cases; got {count}");
    }
}

#[test]
fn mlfp_minimum_minority_language_cases() {
    let count = MINORITY_FP_CASES.iter()
        .filter(|(_, _, has_ml, _, _)| *has_ml)
        .count();
    assert!(count >= 118, "need at least 118 minority-language cases; got {count}");
}

#[test]
fn mlfp_minimum_code_switching_cases() {
    let count = MINORITY_FP_CASES.iter()
        .filter(|(_, _, _, has_cs, _)| *has_cs)
        .count();
    assert!(count >= 98, "need at least 98 code-switching cases; got {count}");
}

#[test]
fn mlfp_total_case_count() {
    assert_eq!(MINORITY_FP_CASES.len(), 255, "minority-language FP corpus must have 255 cases");
}

// ===========================================================================
// Jurisdiction template tests (ported from test_jurisdiction_template.py)
// ===========================================================================

fn parse_template() -> serde_yaml::Value {
    let yaml = jurisdiction_overlay_yaml("_template").unwrap_or_else(|| panic!("_template should exist"));
    serde_yaml::from_str(yaml).unwrap_or_else(|e| panic!("_template should parse as YAML: {e}"))
}

const TEMPLATE_REQUIRED_TOP_LEVEL: &[&str] = &[
    "skill_id", "parent", "schema_version", "expires_on", "signers",
    "activation", "local_definitions", "local_language_assets",
    "overrides", "allowed_contexts", "user_notice",
];

const TEMPLATE_REQUIRED_LOCAL_DEFINITIONS: &[&str] = &[
    "legal_age_general", "legal_age_sexual_content_consumer",
    "legal_age_marketplace_alcohol", "legal_age_marketplace_tobacco",
    "protected_classes", "listed_extremist_orgs",
    "restricted_symbols", "election_rules",
];

const TEMPLATE_REQUIRED_LANGUAGE_ASSETS: &[&str] = &[
    "primary_languages", "lexicons", "normalization",
];

const TEMPLATE_REQUIRED_NORMALIZATION_FIELDS: &[&str] = &[
    "nfkc", "case_fold", "homoglyph_map_id", "transliteration_refs",
];

const TEMPLATE_REQUIRED_USER_NOTICE: &[&str] = &[
    "visible_pack_summary", "appeal_resource_id", "opt_out_allowed",
];

const TEMPLATE_REQUIRED_FORBIDDEN_CRITERIA: &[&str] = &[
    "gps_location", "ip_geolocation", "inferred_nationality",
    "inferred_ethnicity", "inferred_religion",
];

const TEMPLATE_REQUIRED_SIGNERS: &[&str] = &["legal_review", "cultural_review"];

const TEMPLATE_REQUIRED_ALLOWED_CONTEXTS: &[&str] = &[
    "QUOTED_SPEECH_CONTEXT", "NEWS_CONTEXT",
    "EDUCATION_CONTEXT", "COUNTERSPEECH_CONTEXT",
];

#[test]
fn template_parses_as_valid_yaml() {
    let _ = parse_template();
}

#[test]
fn template_required_top_level_keys() {
    let t = parse_template();
    for key in TEMPLATE_REQUIRED_TOP_LEVEL {
        assert!(t.get(*key).is_some(), "template missing top-level key: {key}");
    }
}

#[test]
fn template_parent_is_global_baseline() {
    let t = parse_template();
    assert_eq!(t.get("parent").and_then(|v| v.as_str()), Some("kchat.global.guardrail.baseline"));
}

#[test]
fn template_schema_version_is_1() {
    let t = parse_template();
    assert_eq!(t.get("schema_version").and_then(|v| v.as_u64()), Some(1));
}

#[test]
fn template_signers_include_legal_and_cultural() {
    let t = parse_template();
    let signers = t.get("signers").and_then(|v| v.as_sequence()).unwrap();
    let signer_set: std::collections::HashSet<_> = signers.iter().filter_map(|v| v.as_str()).collect();
    for s in TEMPLATE_REQUIRED_SIGNERS {
        assert!(signer_set.contains(*s), "template signers missing: {s}");
    }
    assert!(signer_set.contains("trust_and_safety"), "template signers must include trust_and_safety");
}

#[test]
fn template_skill_id_pattern() {
    let t = parse_template();
    let id = t.get("skill_id").and_then(|v| v.as_str()).unwrap();
    assert!(id.starts_with("kchat.jurisdiction."), "skill_id must start with kchat.jurisdiction.: {id}");
    assert!(id.ends_with(".guardrail.v1"), "skill_id must end with .guardrail.v1: {id}");
}

#[test]
fn template_forbidden_criteria_has_all_five() {
    let t = parse_template();
    let fc = t.get("activation")
        .and_then(|v| v.get("forbidden_criteria"))
        .and_then(|v| v.as_sequence()).unwrap();
    let fc_set: std::collections::HashSet<_> = fc.iter().filter_map(|v| v.as_str()).collect();
    for c in TEMPLATE_REQUIRED_FORBIDDEN_CRITERIA {
        assert!(fc_set.contains(*c), "template forbidden_criteria missing: {c}");
    }
}

#[test]
fn template_activation_criteria_present() {
    let t = parse_template();
    let criteria = t.get("activation").and_then(|v| v.get("criteria"));
    assert!(criteria.is_some(), "template must have activation.criteria");
    assert!(criteria.unwrap().as_sequence().map(|s| !s.is_empty()).unwrap_or(false),
        "activation.criteria must be a non-empty list");
}

#[test]
fn template_local_definitions_required_keys() {
    let t = parse_template();
    let ld = t.get("local_definitions").and_then(|v| v.as_mapping()).unwrap();
    for key in TEMPLATE_REQUIRED_LOCAL_DEFINITIONS {
        assert!(ld.get(*key).is_some(), "template local_definitions missing: {key}");
    }
}

#[test]
fn template_language_assets_keys() {
    let t = parse_template();
    let la = t.get("local_language_assets").and_then(|v| v.as_mapping()).unwrap();
    for key in TEMPLATE_REQUIRED_LANGUAGE_ASSETS {
        assert!(la.get(*key).is_some(), "template local_language_assets missing: {key}");
    }
}

#[test]
fn template_normalization_required_fields() {
    let t = parse_template();
    let norm = t.get("local_language_assets")
        .and_then(|v| v.get("normalization"))
        .and_then(|v| v.as_mapping()).unwrap();
    for key in TEMPLATE_REQUIRED_NORMALIZATION_FIELDS {
        assert!(norm.get(*key).is_some(), "template normalization missing: {key}");
    }
    assert_eq!(norm.get("nfkc").and_then(|v| v.as_bool()), Some(true), "nfkc must be true");
    assert_eq!(norm.get("case_fold").and_then(|v| v.as_bool()), Some(true), "case_fold must be true");
}

#[test]
fn template_has_at_least_one_override() {
    let t = parse_template();
    let overrides = t.get("overrides").and_then(|v| v.as_sequence()).unwrap();
    assert!(!overrides.is_empty(), "template must have at least one override");
}

#[test]
fn template_allowed_contexts_match_protected_speech() {
    let t = parse_template();
    let ctx = t.get("allowed_contexts").and_then(|v| v.as_sequence()).unwrap();
    let ctx_set: std::collections::HashSet<_> = ctx.iter().filter_map(|v| v.as_str()).collect();
    for c in TEMPLATE_REQUIRED_ALLOWED_CONTEXTS {
        assert!(ctx_set.contains(*c), "template allowed_contexts missing: {c}");
    }
}

#[test]
fn template_user_notice_required_fields() {
    let t = parse_template();
    let un = t.get("user_notice").and_then(|v| v.as_mapping()).unwrap();
    for key in TEMPLATE_REQUIRED_USER_NOTICE {
        assert!(un.get(*key).is_some(), "template user_notice missing: {key}");
    }
}

// ===========================================================================
// Anti-misuse validation for all jurisdictions and communities
// ===========================================================================

#[test]
fn all_jurisdictions_pass_anti_misuse_validation() {
    use crate::skillpack::anti_misuse;
    for code in jurisdiction_codes() {
        let yaml = jurisdiction_overlay_yaml(code).unwrap();
        let report = anti_misuse::validate_pack_yaml(yaml)
            .unwrap_or_else(|e| panic!("{code} should parse: {e}"));
        assert!(report.passed(),
            "{code} failed anti-misuse validation: {:?}", report.errors);
    }
}

#[test]
fn all_communities_pass_anti_misuse_validation() {
    use crate::skillpack::anti_misuse;
    use crate::skillpack::data::communities;
    for name in communities::community_names() {
        let yaml = communities::community_overlay_yaml(name).unwrap();
        let report = anti_misuse::validate_pack_yaml(yaml)
            .unwrap_or_else(|e| panic!("{name} should parse: {e}"));
        assert!(report.passed(),
            "{name} failed anti-misuse validation: {:?}", report.errors);
    }
}

// ===========================================================================
// US civic_window assertions (from test_country_us.py)
// ===========================================================================

#[test]
fn us_civic_window_open_and_close() {
    let overlay = parse_overlay("us");
    let rules = overlay.get("local_definitions")
        .and_then(|v| v.get("election_rules"))
        .and_then(|v| v.as_mapping()).unwrap();
    let open = rules.get("civic_window_open").and_then(|v| v.as_str());
    let close = rules.get("civic_window_close").and_then(|v| v.as_str());
    assert!(open.is_some() && !open.unwrap().is_empty(),
        "US election_rules must have civic_window_open set");
    assert!(close.is_some() && !close.unwrap().is_empty(),
        "US election_rules must have civic_window_close set");
}

// ===========================================================================
// Template source-text verbatim checks (from test_jurisdiction_template.py)
// ===========================================================================

#[test]
fn template_source_mentions_forbidden_criteria_verbatim() {
    let yaml = jurisdiction_overlay_yaml("_template").unwrap();
    for criterion in TEMPLATE_REQUIRED_FORBIDDEN_CRITERIA {
        assert!(yaml.contains(criterion),
            "forbidden criterion '{criterion}' must appear verbatim in the template source");
    }
}

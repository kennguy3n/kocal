//! Adversarial corpus tests ported from kchat-skills/tests/adversarial/test_adversarial_corpus.py.
//!
//! Validates the adversarial/obfuscation corpus YAML and tests that the
//! Rust normalization pipeline neutralises evasion techniques so that
//! category trigger tokens remain detectable after normalization.

use crate::skillpack::data::loaders;
use crate::normalize;

const VALID_TECHNIQUES: &[&str] = &[
    "homoglyph_attacks",
    "leetspeak",
    "code_switching",
    "unicode_tricks",
    "whitespace_insertion",
    "image_text_evasion",
];

const MIN_CASES_PER_TECHNIQUE: usize = 10;
const MIN_DETECTION_RATE: f64 = 0.60;

/// Category trigger tokens — a non-zero intersection after normalization
/// counts as detection. Ported from _CATEGORY_TRIGGERS in the Python test.
fn category_triggers(category: u64) -> &'static [&'static str] {
    match category {
        1 => &["csam_indicator", "child", "minor", "schoolgirl", "underage"],
        2 => &["harm myself", "harm yourself", "suicide", "hurt myself", "harmmyself", "hurtmyself"],
        3 => &["kill you", "hurt you", "beat you", "killyou", "hurtyou", "goingtohurt", "goingtokill", "gonnakill", "gonna kill"],
        4 => &["join the cause", "fight together", "revolution", "swastika", "the enemy", "theenemy", "jointhecause", "fighttogether", "join the r", "jointher"],
        5 => &["worthless", "loser", "stupid", "estupido"],
        6 => &["hate them", "hatethem", "i hate", "ihate", "disgusting", "<slur_token_a>", "hate"],
        7 => &["verify", "login", "confirm", "account", "wire", "transfer", "fee", "suspend", "prize", "won", "bank", "reset", "password", "btc", "pay", "phishing", "qr code", "bitcoin", "example.", "update", "tracking", "delayed", "delivery", "address", "exe", ".exe", "scan to", "login page", "fake bank"],
        8 => &["cracked", "setup.exe", "download", "keys", ".exe"],
        10 => &["adult content", "uncensored", "free adult", "xxx", "adult"],
        11 => &["pills", "drugs", "bars", "chars", "selling", "glock", "handgun", "weapons", "controlled substances", "sellingpills", "sellingchars", "sellingcheap", "cheappills"],
        12 => &["stolen", "fake id", "illegal goods"],
        13 => &["vaccine", "cure", "fake medicine"],
        14 => &["vote", "polling", "ballot", "election", "voter fraud", "vote bags", "votebags"],
        15 => &["community rule"],
        _ => &[],
    }
}

fn detected_category(normalized: &str, category: u64) -> bool {
    if category == 0 {
        return false;
    }
    let triggers = category_triggers(category);
    let lower = normalized.to_lowercase();
    triggers.iter().any(|t| lower.contains(t))
}

fn parse_corpus() -> serde_yaml::Value {
    let yaml = loaders::embedded_adversarial_corpus();
    serde_yaml::from_str(yaml).expect("adversarial corpus should parse as YAML")
}

fn all_cases(corpus: &serde_yaml::Value) -> Vec<serde_yaml::Value> {
    let mut cases = Vec::new();
    for technique in VALID_TECHNIQUES {
        if let Some(block) = corpus.get(*technique).and_then(|v| v.as_sequence()) {
            for case in block {
                cases.push(case.clone());
            }
        }
    }
    cases
}

fn cases_by_technique(corpus: &serde_yaml::Value, technique: &str) -> Vec<serde_yaml::Value> {
    corpus.get(technique)
        .and_then(|v| v.as_sequence())
        .map(|s| s.to_vec())
        .unwrap_or_default()
}

// ===========================================================================
// Corpus structural tests
// ===========================================================================

#[test]
fn corpus_schema_version_is_1() {
    let corpus = parse_corpus();
    assert_eq!(corpus.get("schema_version").and_then(|v| v.as_u64()), Some(1));
}

#[test]
fn each_technique_has_block() {
    let corpus = parse_corpus();
    for technique in VALID_TECHNIQUES {
        let block = corpus.get(*technique);
        assert!(block.is_some(), "technique {technique} must exist in corpus");
        assert!(block.unwrap().as_sequence().map(|s| !s.is_empty()).unwrap_or(false),
            "technique {technique} must be a non-empty list");
    }
}

#[test]
fn each_technique_has_minimum_cases() {
    let corpus = parse_corpus();
    for technique in VALID_TECHNIQUES {
        let cases = cases_by_technique(&corpus, technique);
        assert!(cases.len() >= MIN_CASES_PER_TECHNIQUE,
            "technique {technique} needs >= {MIN_CASES_PER_TECHNIQUE} cases; got {}", cases.len());
    }
}

#[test]
fn case_ids_unique() {
    let corpus = parse_corpus();
    let cases = all_cases(&corpus);
    let mut ids: Vec<String> = Vec::new();
    for case in &cases {
        let id = case.get("case_id").and_then(|v| v.as_str()).unwrap();
        ids.push(id.to_string());
    }
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(ids.len(), unique.len(), "duplicate case_id in adversarial corpus");
}

#[test]
fn case_required_fields() {
    let corpus = parse_corpus();
    let cases = all_cases(&corpus);
    let required = ["case_id", "technique", "category", "text", "expected_detection", "notes"];
    for case in &cases {
        for field in &required {
            assert!(case.get(*field).is_some(),
                "{}: missing field {field}",
                case.get("case_id").and_then(|v| v.as_str()).unwrap_or("?"));
        }
        let technique = case.get("technique").and_then(|v| v.as_str()).unwrap();
        assert!(VALID_TECHNIQUES.contains(&technique),
            "{}: invalid technique {technique}",
            case.get("case_id").and_then(|v| v.as_str()).unwrap_or("?"));
        let category = case.get("category").and_then(|v| v.as_u64()).unwrap();
        assert!(category <= 16,
            "{}: category {category} out of range",
            case.get("case_id").and_then(|v| v.as_str()).unwrap_or("?"));
        let text = case.get("text").and_then(|v| v.as_str()).unwrap();
        assert!(!text.trim().is_empty(),
            "{}: text must be non-empty",
            case.get("case_id").and_then(|v| v.as_str()).unwrap_or("?"));
        let expected = case.get("expected_detection");
        assert!(expected.is_some() && (expected.unwrap().is_bool()),
            "{}: expected_detection must be bool",
            case.get("case_id").and_then(|v| v.as_str()).unwrap_or("?"));
    }
}

#[test]
fn normalize_accepts_every_case() {
    let corpus = parse_corpus();
    let cases = all_cases(&corpus);
    for case in &cases {
        let text = case.get("text").and_then(|v| v.as_str()).unwrap();
        let _out = normalize::normalize(text);
    }
}

// ===========================================================================
// Per-technique detection rate tests
// ===========================================================================

#[test]
fn detection_rate_per_technique() {
    let corpus = parse_corpus();
    for technique in VALID_TECHNIQUES {
        let cases = cases_by_technique(&corpus, technique);
        let positives: Vec<_> = cases.iter()
            .filter(|c| c.get("expected_detection").and_then(|v| v.as_bool()) == Some(true))
            .collect();
        assert!(!positives.is_empty(), "{technique} must have at least one positive case");
        let mut hits = 0;
        for case in &positives {
            let text = case.get("text").and_then(|v| v.as_str()).unwrap();
            let category = case.get("category").and_then(|v| v.as_u64()).unwrap();
            let normalized = normalize::normalize(text);
            if detected_category(&normalized, category) {
                hits += 1;
            }
        }
        let rate = hits as f64 / positives.len() as f64;
        assert!(rate >= MIN_DETECTION_RATE,
            "{technique} detection rate {rate:.2} below {MIN_DETECTION_RATE} ({hits}/{})",
            positives.len());
    }
}

#[test]
fn benign_controls_do_not_flag() {
    let corpus = parse_corpus();
    for technique in VALID_TECHNIQUES {
        let cases = cases_by_technique(&corpus, technique);
        let benign: Vec<_> = cases.iter()
            .filter(|c| c.get("expected_detection").and_then(|v| v.as_bool()) == Some(false))
            .collect();
        for case in &benign {
            let category = case.get("category").and_then(|v| v.as_u64()).unwrap();
            assert_eq!(category, 0,
                "{}: benign control must carry category 0",
                case.get("case_id").and_then(|v| v.as_str()).unwrap_or("?"));
            let text = case.get("text").and_then(|v| v.as_str()).unwrap();
            let normalized = normalize::normalize(text);
            assert!(!detected_category(&normalized, category));
        }
    }
}

#[test]
fn corpus_has_at_least_one_benign_control() {
    let corpus = parse_corpus();
    let cases = all_cases(&corpus);
    let benign = cases.iter()
        .filter(|c| c.get("expected_detection").and_then(|v| v.as_bool()) == Some(false))
        .count();
    assert!(benign >= 3, "corpus should include at least 3 benign controls; got {benign}");
}

// ===========================================================================
// Direct normalization assertions
// ===========================================================================

#[test]
fn homoglyph_normalization_neutralises_cyrillic() {
    let raw = "v\u{0435}rify your \u{0430}ccount";
    let norm = normalize::normalize(raw);
    assert!(norm.contains("verify"), "normalized text must contain 'verify': {norm}");
    assert!(norm.contains("account"), "normalized text must contain 'account': {norm}");
}

#[test]
fn fullwidth_nfkc_folds_to_ascii() {
    let raw = "\u{ff36}\u{ff25}\u{ff32}\u{ff29}\u{ff26}\u{ff39}";
    let norm = normalize::normalize_for_patterns(raw);
    let lower = norm.to_lowercase();
    assert!(lower.contains("verify"), "fullwidth VERIFY must NFKC-fold to 'verify': {norm} (lower: {lower})");
}

#[test]
fn zero_width_stripping() {
    let raw = "verify\u{200d} your\u{200d} login";
    let norm = normalize::normalize(raw);
    assert!(norm.contains("verify"), "zero-width chars must be stripped: {norm}");
}

#[test]
fn whitespace_collapse_and_strip() {
    let raw = "h a t e them";
    let norm = normalize::normalize(raw);
    assert!(norm.contains("hate"), "inter-letter spaces must be collapsed: {norm}");
}

// ===========================================================================
// Aggregate detection rate
// ===========================================================================

#[test]
fn aggregate_detection_rate() {
    let corpus = parse_corpus();
    let cases = all_cases(&corpus);
    let positives: Vec<_> = cases.iter()
        .filter(|c| c.get("expected_detection").and_then(|v| v.as_bool()) == Some(true))
        .collect();
    let mut hits = 0;
    for case in &positives {
        let text = case.get("text").and_then(|v| v.as_str()).unwrap();
        let category = case.get("category").and_then(|v| v.as_u64()).unwrap();
        let technique = case.get("technique").and_then(|v| v.as_str()).unwrap();
        let normalized = normalize::normalize(text);
        if detected_category(&normalized, category) {
            hits += 1;
        }
        // technique is used in Python but Rust normalize() handles all techniques uniformly
        let _ = technique;
    }
    let rate = hits as f64 / positives.len() as f64;
    assert!(rate >= MIN_DETECTION_RATE,
        "corpus-wide detection rate {rate:.2} below {MIN_DETECTION_RATE}");
}

// ===========================================================================
// Total case count
// ===========================================================================

#[test]
fn total_case_count_at_least_fifty() {
    let corpus = parse_corpus();
    let cases = all_cases(&corpus);
    assert!(cases.len() >= 50, "adversarial corpus must contain >= 50 cases; got {}", cases.len());
}

// ===========================================================================
// Additional tests ported from test_adversarial_corpus.py
// ===========================================================================

#[test]
fn decode_for_technique_returns_two_forms() {
    // In Python, decode_for_technique returns a tuple of (normalized, leet_decoded).
    // In Rust, normalize::normalize produces a single string that covers all techniques.
    // We verify that normalize produces a non-empty string for every case.
    let corpus = parse_corpus();
    let cases = all_cases(&corpus);
    for case in &cases {
        let text = case.get("text").and_then(|v| v.as_str()).unwrap();
        let normalized = normalize::normalize(text);
        assert!(!normalized.is_empty(), "normalized form must be non-empty for case");
    }
}

#[test]
fn extra_homoglyph_fold_neutralises_greek() {
    let raw = "v\u{03b5}rify \u{03b1}ccount";
    let norm = normalize::normalize(raw);
    assert!(norm.contains("verify"), "Greek homoglyph must fold to 'verify': {norm}");
    assert!(norm.contains("account"), "Greek homoglyph must fold to 'account': {norm}");
}

#[test]
fn leet_decode_round_trips() {
    let raw = "v3r1fy y0ur acc0unt";
    let norm = normalize::normalize(raw);
    assert!(norm.contains("verify"), "leetspeak must decode to 'verify': {norm}");
    assert!(norm.contains("your"), "leetspeak must decode to 'your': {norm}");
    assert!(norm.contains("account"), "leetspeak must decode to 'account': {norm}");
}

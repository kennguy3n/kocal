//! Text normalization — NFKC + case fold + TR39 homoglyph + defang.
//!
//! Normalization runs after decryption and before any detector. It strips
//! zero-width/bidi characters, normalizes leetspeak, applies NFKC compatibility
//! decomposition, case folding, and TR39 confusable homoglyph folding.
//!
//! Ported from slm-guardrail's `pipeline/normalize.rs` and `pipeline/defang.rs`,
//! keeping the multi-view architecture: PII/URL detectors get the digit-preserving
//! view; lexicon/scam detectors get the defanged (leetspeak-substituted) views.

use std::collections::HashMap;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

// ---------------------------------------------------------------------------
// TR39 homoglyph confusables map.
// ---------------------------------------------------------------------------

/// Embedded copy of the TR39 confusables map (704 entries, Unicode 17.0).
/// Ported from slm-guardrail's `build-tools/compiler/data/homoglyph_map.json`.
const HOMOGLYPH_MAP_JSON: &str = include_str!("../data/homoglyph_map.json");

/// Hand-curated fallback used only when the packaged JSON is missing or
/// malformed. Mirrors slm-guardrail's `HAND_CURATED_HOMOGLYPH_MAP`.
const HAND_CURATED: &[(char, char)] = &[
    // Cyrillic → Latin (lowercase).
    ('\u{0430}', 'a'), ('\u{0432}', 'b'), ('\u{0435}', 'e'),
    ('\u{043a}', 'k'), ('\u{043c}', 'm'), ('\u{043d}', 'h'),
    ('\u{043e}', 'o'), ('\u{0440}', 'p'), ('\u{0441}', 'c'),
    ('\u{0442}', 't'), ('\u{0443}', 'y'), ('\u{0445}', 'x'),
    ('\u{0456}', 'i'), ('\u{0458}', 'j'),
    // Cyrillic → Latin (uppercase).
    ('\u{0410}', 'a'), ('\u{0412}', 'b'), ('\u{0415}', 'e'),
    ('\u{041a}', 'k'), ('\u{041c}', 'm'), ('\u{041d}', 'h'),
    ('\u{041e}', 'o'), ('\u{0420}', 'p'), ('\u{0421}', 'c'),
    ('\u{0422}', 't'), ('\u{0423}', 'y'), ('\u{0425}', 'x'),
    // Greek → Latin (lowercase).
    ('\u{03b1}', 'a'), ('\u{03b2}', 'b'), ('\u{03b5}', 'e'),
    ('\u{03b7}', 'h'), ('\u{03b9}', 'i'), ('\u{03ba}', 'k'),
    ('\u{03bc}', 'm'), ('\u{03bd}', 'v'), ('\u{03bf}', 'o'),
    ('\u{03c1}', 'p'), ('\u{03c4}', 't'), ('\u{03c5}', 'y'),
    ('\u{03c7}', 'x'),
    // Fullwidth digits → ASCII.
    ('\u{ff10}', '0'), ('\u{ff11}', '1'), ('\u{ff12}', '2'),
    ('\u{ff13}', '3'), ('\u{ff14}', '4'), ('\u{ff15}', '5'),
    ('\u{ff16}', '6'), ('\u{ff17}', '7'), ('\u{ff18}', '8'),
    ('\u{ff19}', '9'),
];

#[derive(Deserialize)]
struct HomoglyphMapFile {
    #[serde(default)]
    map: HashMap<String, String>,
    #[serde(default)]
    tr39_unicode_version: Option<String>,
}

use serde::Deserialize;

struct LoadedMap {
    table: HashMap<char, char>,
    tr39_unicode_version: Option<String>,
}

fn load_homoglyph_map() -> &'static LoadedMap {
    static CELL: OnceLock<LoadedMap> = OnceLock::new();
    CELL.get_or_init(|| match serde_json::from_str::<HomoglyphMapFile>(HOMOGLYPH_MAP_JSON) {
        Ok(payload) => {
            let mut table: HashMap<char, char> = HashMap::with_capacity(payload.map.len());
            for (raw_key, raw_value) in &payload.map {
                let cp = match u32::from_str_radix(raw_key, 16) {
                    Ok(cp) => cp,
                    Err(_) => return hand_curated_map(),
                };
                let Some(key_char) = char::from_u32(cp) else { return hand_curated_map() };
                let Some(value_char) = raw_value.chars().next() else { return hand_curated_map() };
                table.insert(key_char, value_char);
            }
            LoadedMap { table, tr39_unicode_version: payload.tr39_unicode_version }
        }
        Err(_) => hand_curated_map(),
    })
}

fn hand_curated_map() -> LoadedMap {
    LoadedMap {
        table: HAND_CURATED.iter().copied().collect(),
        tr39_unicode_version: None,
    }
}

/// Apply TR39 confusable homoglyph folding (e.g. Cyrillic а → Latin a).
pub fn homoglyph_fold(text: &str) -> String {
    let map = &load_homoglyph_map().table;
    if map.is_empty() {
        return text.to_string();
    }
    text.chars().map(|ch| map.get(&ch).copied().unwrap_or(ch)).collect()
}

/// Source Unicode revision the packaged map was generated against.
pub fn homoglyph_map_unicode_version() -> Option<&'static str> {
    load_homoglyph_map().tr39_unicode_version.as_deref()
}

// ---------------------------------------------------------------------------
// Zero-width / bidi / format character stripping.
// ---------------------------------------------------------------------------

/// Zero-width, bidi, and format codepoints that adversaries insert inside
/// otherwise detectable terms. Ported from slm-guardrail's `ZERO_WIDTH_AND_FORMAT`.
const ZERO_WIDTH_AND_FORMAT: &[char] = &[
    '\u{200b}', // ZERO WIDTH SPACE
    '\u{200c}', // ZERO WIDTH NON-JOINER
    '\u{200d}', // ZERO WIDTH JOINER
    '\u{2060}', // WORD JOINER
    '\u{feff}', // BYTE ORDER MARK / ZWNBSP
    '\u{00ad}', // SOFT HYPHEN
    '\u{202a}', // LEFT-TO-RIGHT EMBEDDING
    '\u{202b}', // RIGHT-TO-LEFT EMBEDDING
    '\u{202c}', // POP DIRECTIONAL FORMATTING
    '\u{202d}', // LEFT-TO-RIGHT OVERRIDE
    '\u{202e}', // RIGHT-TO-LEFT OVERRIDE
    '\u{2066}', // LEFT-TO-RIGHT ISOLATE
    '\u{2067}', // RIGHT-TO-LEFT ISOLATE
    '\u{2068}', // FIRST STRONG ISOLATE
    '\u{2069}', // POP DIRECTIONAL ISOLATE
];

/// Check if a character is zero-width, bidi, or format.
fn is_zero_width(c: char) -> bool {
    ZERO_WIDTH_AND_FORMAT.contains(&c)
}

/// Remove zero-width / bidi / format codepoints from text.
pub fn strip_zero_width(text: &str) -> String {
    text.chars().filter(|&c| !is_zero_width(c)).collect()
}

// ---------------------------------------------------------------------------
// Leetspeak defang — multi-view with ambiguous digit `1`.
// ---------------------------------------------------------------------------

/// The digit `1` is ambiguous in leetspeak — adversaries use it for both
/// `l` (`a11` → `all`) and `i` (`th1s` → `this`). Two defanged views are
/// produced so detectors see both readings.
pub const LEET_DIGIT_ONE_VARIANTS: [char; 2] = ['l', 'i'];

fn leet_base(ch: char) -> Option<char> {
    match ch {
        '0' => Some('o'),
        '3' => Some('e'),
        '4' => Some('a'),
        '5' => Some('s'),
        '7' => Some('t'),
        '8' => Some('b'),
        '@' => Some('a'),
        '$' => Some('s'),
        '!' => Some('i'),
        _ => None,
    }
}

/// Substitute leetspeak digits/punctuation back to letters.
/// `digit_one_as` controls the ambiguous digit `1` ('l' or 'i').
pub fn defang_leetspeak(text: &str, digit_one_as: char) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch == '1' {
            out.push(digit_one_as);
        } else if let Some(sub) = leet_base(ch) {
            out.push(sub);
        } else {
            out.push(ch);
        }
    }
    out
}

/// Return the family of defanged views to search.
///
/// Each view starts from `normalized_text`, strips zero-width/bidi controls,
/// then applies leetspeak substitution. Two variants are produced when the
/// text contains `1` — one resolving it as `l`, the other as `i`. Detectors
/// must search every variant and union the hits.
pub fn defang_variants_for_matching(normalized_text: &str) -> Vec<String> {
    let stripped = strip_zero_width(normalized_text);
    if !stripped.contains('1') {
        return vec![defang_leetspeak(&stripped, 'l')];
    }
    LEET_DIGIT_ONE_VARIANTS.iter().copied()
        .map(|v| defang_leetspeak(&stripped, v))
        .collect()
}

// ---------------------------------------------------------------------------
// Main normalization entry points.
// ---------------------------------------------------------------------------

/// Normalize text for safety analysis (lexicon/scam detector view).
///
/// Steps:
/// 1. NFKC compatibility decomposition (e.g. ﬀ → ff, Ａ → A)
/// 2. Strip zero-width/bidi/format characters
/// 3. Case fold for caseless matching
/// 4. TR39 homoglyph fold (Cyrillic/Greek/Cherokee → Latin)
/// 5. Normalize leetspeak (digit `1` → `i`)
/// 6. Collapse repeated whitespace
/// 7. De-space single-character obfuscation (e.g. "H O W T O" → "howto")
pub fn normalize(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let no_zw = strip_zero_width(&nfkc);
    let folded = caseless::default_case_fold_str(&no_zw);
    let homoglyphed = homoglyph_fold(&folded);
    let deleeted = defang_leetspeak(&homoglyphed, 'i');
    let collapsed = collapse_whitespace(&deleeted);
    despace_obfuscation(&collapsed)
}

/// Light normalization for pattern-based detectors (PII, URL).
///
/// Does NOT apply leetspeak or homoglyph substitution — would break
/// digit-bearing PII (phone, SSN, IBAN, credit card).
pub fn normalize_for_patterns(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let no_zw = strip_zero_width(&nfkc);
    collapse_whitespace(&no_zw)
}

/// NFKC + case fold + homoglyph + despace (no leetspeak).
/// Used as the base for building defang variants. Includes de-space
/// obfuscation defense so spaced-out harmful text (e.g. "H O W T O") is
/// joined before lexicon matching.
pub fn normalize_for_lexicon(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let no_zw = strip_zero_width(&nfkc);
    let folded = caseless::default_case_fold_str(&no_zw);
    let homoglyphed = homoglyph_fold(&folded);
    let collapsed = collapse_whitespace(&homoglyphed);
    despace_obfuscation(&collapsed)
}

/// Collapse multiple whitespace characters into single spaces.
fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_ws = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                result.push(' ');
            }
            prev_ws = true;
        } else {
            result.push(c);
            prev_ws = false;
        }
    }
    result.trim().to_string()
}

/// Remove spaces between single-letter sequences (obfuscation defense).
///
/// Detects patterns like "h o w t o m a k e a b o m b" where individual
/// letters are separated by spaces to bypass keyword matching. Joins
/// sequences of 3+ single-character tokens into continuous strings.
fn despace_obfuscation(text: &str) -> String {
    let tokens: Vec<&str> = text.split(' ').collect();
    if tokens.len() < 3 {
        return text.to_string();
    }
    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    while i < tokens.len() {
        // Only start a run if the current token is a single character.
        if tokens[i].chars().count() == 1 {
            let run_start = i;
            let mut run_len = 1usize;
            while i + run_len < tokens.len()
                && tokens[i + run_len].chars().count() == 1
            {
                run_len += 1;
            }
            if run_len >= 3 {
                for j in 0..run_len {
                    result.push_str(tokens[run_start + j]);
                }
                i = run_start + run_len;
                continue;
            }
        }
        result.push_str(tokens[i]);
        if i + 1 < tokens.len() {
            result.push(' ');
        }
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfkc_normalization() {
        let input = "\u{FF21}\u{FF22}\u{FF23}"; // ＡＢＣ
        assert_eq!(normalize(input), "abc");
    }

    #[test]
    fn test_zero_width_stripped() {
        assert_eq!(normalize("hel\u{200B}lo"), "hello");
    }

    #[test]
    fn test_bidi_format_stripped() {
        assert_eq!(normalize("wi\u{202e}re"), "wire");
        assert_eq!(normalize("a\u{2066}b\u{2069}c"), "abc");
    }

    #[test]
    fn test_case_fold() {
        assert_eq!(normalize("HELLO"), "hello");
        assert_eq!(normalize("HeLLo"), "hello");
    }

    #[test]
    fn test_leetspeak() {
        assert_eq!(normalize("h3llo"), "hello");
        assert_eq!(normalize("w0rld"), "world");
        assert_eq!(normalize("l33t"), "leet");
    }

    #[test]
    fn test_whitespace_collapse() {
        assert_eq!(normalize("hello    world"), "hello world");
        assert_eq!(normalize("  hello  "), "hello");
    }

    #[test]
    fn test_combined() {
        let input = "\u{FF08}\u{200B}test\u{200D}\u{FF09}";
        assert_eq!(normalize(input), "(test)");
    }

    // --- Homoglyph tests ---

    #[test]
    fn test_homoglyph_cyrillic_to_latin() {
        let cyrillic_a = '\u{0430}';
        let input = format!("p{}yp{}l", cyrillic_a, cyrillic_a);
        assert_eq!(normalize(&input), "paypal");
    }

    #[test]
    fn test_homoglyph_greek_to_latin() {
        let input = "p\u{03b1}yp\u{03b1}l";
        assert_eq!(normalize(input), "paypal");
    }

    #[test]
    fn test_homoglyph_map_loaded_from_json() {
        let table = &load_homoglyph_map().table;
        assert!(table.len() > 100, "TR39 map should be >100 entries; got {}", table.len());
    }

    #[test]
    fn test_homoglyph_unicode_version() {
        let version = homoglyph_map_unicode_version();
        assert!(version.is_some(), "TR39 unicode version should be exposed");
    }

    // --- Defang variant tests ---

    #[test]
    fn test_defang_variants_no_one() {
        let views = defang_variants_for_matching("h3llo");
        assert_eq!(views.len(), 1);
        assert_eq!(views[0], "hello");
    }

    #[test]
    fn test_defang_variants_with_one() {
        let views = defang_variants_for_matching("a11");
        assert_eq!(views.len(), 2);
        assert!(views.contains(&"all".to_string()));
        assert!(views.contains(&"aii".to_string()));
    }

    #[test]
    fn test_defang_strips_zero_width() {
        let views = defang_variants_for_matching("w\u{200b}ire");
        assert_eq!(views.len(), 1);
        assert_eq!(views[0], "wire");
    }

    // --- De-space obfuscation tests ---

    #[test]
    fn test_despace_obfuscation() {
        let result = normalize("h o w t o m a k e a b o m b");
        assert!(result.contains("howtomakeabomb"));
    }

    #[test]
    fn test_despace_obfuscation_preserves_multichar_tokens() {
        // Multi-char tokens should NOT be joined into runs
        let result = normalize("hello a b c world");
        assert!(!result.contains("helloabc"), "multi-char token should not be joined: {}", result);
        assert!(result.contains("abc"), "single-char run should still be joined: {}", result);
    }
}

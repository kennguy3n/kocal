//! Text normalization — NFKC + case fold + defang.
//!
//! Normalization runs after decryption and before any detector. It strips
//! zero-width characters, normalizes leetspeak, and applies NFKC compatibility
//! decomposition followed by case folding.

use unicode_normalization::UnicodeNormalization;

/// Normalize text for safety analysis.
///
/// Steps:
/// 1. NFKC compatibility decomposition (e.g. ﬀ → ff, Ａ → A)
/// 2. Strip zero-width characters (U+200B, U+200C, U+200D, U+FEFF)
/// 3. Case fold for caseless matching
/// 4. Normalize common leetspeak substitutions
/// 5. Collapse repeated whitespace
/// 6. De-space single-character obfuscation (e.g. "H O W T O" → "howto")
pub fn normalize(text: &str) -> String {
    // 1. NFKC normalization
    let nfkc: String = text.nfkc().collect();

    // 2. Strip zero-width characters
    let no_zw: String = nfkc
        .chars()
        .filter(|&c| !is_zero_width(c))
        .collect();

    // 3. Case fold
    let folded = caseless::default_case_fold_str(&no_zw);

    // 4. Normalize leetspeak
    let deleeted = normalize_leetspeak(&folded);

    // 5. Collapse whitespace
    let collapsed = collapse_whitespace(&deleeted);

    // 6. De-space single-character obfuscation
    // e.g. "h o w t o m a k e a b o m b" → "howtomakeabomb"
    // This catches spaced-out attempts to bypass lexicon matching
    despace_obfuscation(&collapsed)
}

/// Light normalization for pattern-based detectors (PII, URL).
///
/// This does NOT apply leetspeak substitution, which would break digit-based
/// patterns like credit card numbers, phone numbers, and IP addresses.
///
/// Steps:
/// 1. NFKC compatibility decomposition
/// 2. Strip zero-width characters
/// 3. Collapse repeated whitespace
pub fn normalize_for_patterns(text: &str) -> String {
    let nfkc: String = text.nfkc().collect();
    let no_zw: String = nfkc
        .chars()
        .filter(|&c| !is_zero_width(c))
        .collect();
    collapse_whitespace(&no_zw)
}

/// Check if a character is a zero-width or invisible character.
fn is_zero_width(c: char) -> bool {
    matches!(c,
        '\u{200B}' | // Zero Width Space
        '\u{200C}' | // Zero Width Non-Joiner
        '\u{200D}' | // Zero Width Joiner
        '\u{FEFF}' | // Zero Width No-Break Space (BOM)
        '\u{2060}' | // Word Joiner
        '\u{00AD}'   // Soft Hyphen
    )
}

/// Normalize common leetspeak substitutions.
fn normalize_leetspeak(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '0' => 'o',
            '1' => 'i',
            '3' => 'e',
            '4' => 'a',
            '5' => 's',
            '7' => 't',
            '@' => 'a',
            '$' => 's',
            '!' => 'i',
            '|' => 'i',
            '+' => 't',
            _ => c,
        })
        .collect()
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
        // Check if this token and the next few are all single characters
        let mut run_start = i;
        let mut run_len = 1usize;

        while i + run_len < tokens.len()
            && tokens[i + run_len].chars().count() == 1
            && tokens[i + run_len - 0].chars().count() == 1
        {
            run_len += 1;
        }

        // Adjust: we need at least 3 consecutive single-char tokens to de-space
        if run_len >= 3 {
            // Join the run without spaces
            for j in 0..run_len {
                result.push_str(tokens[run_start + j]);
            }
            i = run_start + run_len;
        } else {
            // Not a de-space candidate — output as-is with space
            result.push_str(tokens[i]);
            if i + 1 < tokens.len() {
                result.push(' ');
            }
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfkc_normalization() {
        // Fullwidth letters → ASCII
        let input = "\u{FF21}\u{FF22}\u{FF23}"; // ＡＢＣ
        assert_eq!(normalize(input), "abc");
    }

    #[test]
    fn test_zero_width_stripped() {
        let input = "hel\u{200B}lo";
        assert_eq!(normalize(input), "hello");
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
        // NFKC: （test） → (test), zero-width stripped
        assert_eq!(normalize(input), "(test)");
    }
}

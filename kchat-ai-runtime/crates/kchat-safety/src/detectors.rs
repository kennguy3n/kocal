//! Deterministic detectors — fast regex/heuristic detectors that run before
//! any ML model.
//!
//! Detectors run in priority order:
//!   CHILD_SAFETY > PRIVATE_DATA > SCAM_FRAUD > LEXICON > None
//!
//! Ported from slm-guardrail's pipeline modules:
//! - `pipeline/lexicon.rs` — script-aware word-boundary matching + LRU cache
//! - `pipeline/scam.rs` — 7 calibrated regex families
//! - `pipeline/url.rs` — bare-host + lookalike-brand + shortener scoring
//! - `pipeline/pii.rs` — IBAN mod-97, credential leak, phone structural validation

use crate::verdict::{Action, Severity};
use fancy_regex::Regex as FRegex;
use regex::Regex;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::{Arc, OnceLock};
use parking_lot::Mutex;
use lru::LruCache;

/// Risk category IDs from the KChat taxonomy.
pub mod categories {
    pub const SAFE: u32 = 0;
    pub const CHILD_SAFETY: u32 = 1;
    pub const PRIVATE_DATA: u32 = 2;
    pub const SCAM_FRAUD: u32 = 3;
    pub const HATE_SPEECH: u32 = 4;
    pub const VIOLENCE: u32 = 5;
    pub const NSFW: u32 = 6;
    pub const SELF_HARM: u32 = 7;
    pub const SPAM: u32 = 8;
}

/// A signal from a deterministic detector.
#[derive(Debug, Clone)]
pub struct DetectorSignal {
    pub category: u32,
    pub severity: Severity,
    pub confidence: f64,
    pub reason_code: String,
    pub action: Action,
}

/// All deterministic signals from a single text analysis.
#[derive(Debug, Default)]
pub struct LocalSignals {
    pub signals: Vec<DetectorSignal>,
}

impl LocalSignals {
    pub fn is_empty(&self) -> bool { self.signals.is_empty() }
    pub fn add(&mut self, signal: DetectorSignal) { self.signals.push(signal); }
}

// ---------------------------------------------------------------------------
// PII detector — credit card (Luhn), IBAN (mod-97), phone, SSN, email,
// credential leak. Ported from slm-guardrail `pipeline/pii.rs`.
// ---------------------------------------------------------------------------

pub struct PiiDetector;

/// Minimum digit count for a PHONE match.
const PHONE_MIN_DIGITS: usize = 8;

impl PiiDetector {
    pub fn detect(text: &str) -> Vec<DetectorSignal> {
        let cleaned = crate::normalize::strip_zero_width(text);
        let mut signals = Vec::new();
        let mut credit_card_spans: Vec<(usize, usize)> = Vec::new();

        // Email
        if EMAIL_RE.get_or_init(|| Regex::new(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+").unwrap()).is_match(&cleaned) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::BORDERLINE,
                confidence: 0.90, reason_code: "pii_email".into(), action: Action::Redact,
            });
        }

        // Credit card (Luhn-validated) — runs before PHONE to claim spans
        let cc_re = CC_RE.get_or_init(|| {
            FRegex::new(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)").unwrap()
        });
        for m in cc_re.find_iter(&cleaned).flatten() {
            let raw = &cleaned[m.start()..m.end()];
            let digits: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.len() >= 13 && digits.len() <= 19 && luhn_valid(&digits) {
                credit_card_spans.push((m.start(), m.end()));
                signals.push(DetectorSignal {
                    category: categories::PRIVATE_DATA, severity: Severity::SEVERE,
                    confidence: 0.95, reason_code: "pii_credit_card".into(), action: Action::Redact,
                });
            }
        }

        // Phone — structural validation + credit-card span suppression
        let phone_re = PHONE_RE.get_or_init(|| {
            FRegex::new(r"(?<!\w)\+?\d[\d\-\s().]{7,}\d(?!\w)").unwrap()
        });
        for m in phone_re.find_iter(&cleaned).flatten() {
            let raw = &cleaned[m.start()..m.end()];
            let digit_count = raw.chars().filter(|c| c.is_ascii_digit()).count();
            if digit_count < PHONE_MIN_DIGITS { continue; }
            if !looks_like_phone(raw) { continue; }
            let (start, end) = (m.start(), m.end());
            let overlaps_card = credit_card_spans.iter().any(|(s, e)| start < *e && *s < end);
            if overlaps_card { continue; }
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::BORDERLINE,
                confidence: 0.70, reason_code: "pii_phone".into(), action: Action::Redact,
            });
            break;
        }

        // SSN (US format with invalid-range exclusion)
        if SSN_RE.get_or_init(|| {
            FRegex::new(r"(?<!\d)(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)").unwrap()
        }).is_match(&cleaned).unwrap_or(false) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::SEVERE,
                confidence: 0.95, reason_code: "pii_ssn".into(), action: Action::Redact,
            });
        }

        // IBAN (mod-97 validated) — match IBAN format and validate with mod-97.
        // The regex may over-capture (include trailing words), so we try
        // progressively shorter substrings until one passes mod-97 validation.
        let iban_re = IBAN_RE.get_or_init(|| {
            FRegex::new(r"(?<![A-Za-z0-9])([A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){10,30})(?![A-Za-z0-9])").unwrap()
        });
        let iban_re_ci = IBAN_RE_CI.get_or_init(|| {
            FRegex::new(r"(?<![A-Za-z0-9])([A-Za-z]{2}\d{2}(?:[ ]?[A-Za-z0-9]){10,30})(?![A-Za-z0-9])").unwrap()
        });
        // Try uppercase first, then case-insensitive
        let mut found_iban = false;
        for re in [iban_re, iban_re_ci] {
            for m in re.captures_iter(&cleaned).flatten() {
                if let Some(group) = m.get(1) {
                    let raw = group.as_str();
                    // Try the full match first, then progressively trim trailing
                    // space-separated tokens until mod-97 passes or min length reached.
                    if iban_check(raw) {
                        found_iban = true;
                        break;
                    }
                    // Try trimming trailing tokens (e.g. "GB82...432 for the" → "GB82...432")
                    let parts: Vec<&str> = raw.split(' ').collect();
                    for end in (1..parts.len()).rev() {
                        let trimmed = parts[..end].join(" ");
                        if iban_check(&trimmed) {
                            found_iban = true;
                            break;
                        }
                    }
                    if found_iban { break; }
                }
            }
            if found_iban { break; }
        }
        if found_iban {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::SEVERE,
                confidence: 0.95, reason_code: "pii_iban".into(), action: Action::Redact,
            });
        }

        // Credential leak — paired user/password tokens (multilingual)
        if CRED_LEAK_RE.get_or_init(|| {
            let parts: [&str; 22] = [
                "(?si)", "(?:",
                r"\b(?:user|account)(?:[_-]?id|name)?\b", r"|\blogin\b",
                r"|\bemail\b", r"|\bid\b", r"|\buid\b", r"|\busuario\b",
                "|\u{30e6}\u{30fc}\u{30b6}\u{30fc}(?:id|name)?", // Japanese ユーザー
                ")", r"\s*[:=]\s*\S+", r".{0,120}?", "(?:",
                r"\bpass(?:word|phrase|wd)?\b", r"|\bpwd\b", r"|\bpasswort\b",
                "|\\bcontrase\u{00f1}a\\b", // Spanish contraseña
                r"|\bsenha\b", // Portuguese
                "|\u{30d1}\u{30b9}\u{30ef}\u{30fc}\u{30c9}", // Japanese パスワード
                "|\u{5bc6}\u{7801}", // Chinese 密码
                "|\u{0643}\u{0644}\u{0645}\u{0629}\\s*\u{0627}\u{0644}\u{0645}\u{0631}\u{0648}\u{0631}", // Arabic كلمة المرور
                ")\\s*[:=]\\s*\\S+",
            ];
            FRegex::new(&parts.concat()).unwrap()
        }).is_match(&cleaned).unwrap_or(false) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::SEVERE,
                confidence: 0.90, reason_code: "pii_credentials".into(), action: Action::Redact,
            });
        }

        signals
    }
}

static EMAIL_RE: OnceLock<Regex> = OnceLock::new();
static CC_RE: OnceLock<FRegex> = OnceLock::new();
static PHONE_RE: OnceLock<FRegex> = OnceLock::new();
static SSN_RE: OnceLock<FRegex> = OnceLock::new();
static IBAN_RE: OnceLock<FRegex> = OnceLock::new();
static IBAN_RE_CI: OnceLock<FRegex> = OnceLock::new();
static CRED_LEAK_RE: OnceLock<FRegex> = OnceLock::new();

/// Luhn mod-10 checksum (ISO/IEC 7812).
fn luhn_valid(digits: &str) -> bool {
    let len = digits.len();
    let parity = len % 2;
    let mut total: u32 = 0;
    for (i, ch) in digits.bytes().enumerate() {
        if !ch.is_ascii_digit() { return false; }
        let mut d = (ch - b'0') as u32;
        if i % 2 == parity { d *= 2; if d > 9 { d -= 9; } }
        total += d;
    }
    total > 0 && total % 10 == 0
}

/// ISO 13616 mod-97 IBAN validation.
fn iban_check(iban: &str) -> bool {
    let compact: String = iban.chars().filter(|c| !c.is_whitespace())
        .flat_map(|c| c.to_uppercase()).collect();
    if compact.len() < 15 || compact.len() > 34 { return false; }
    let bytes = compact.as_bytes();
    if !bytes[0].is_ascii_uppercase() || !bytes[1].is_ascii_uppercase()
        || !bytes[2].is_ascii_digit() || !bytes[3].is_ascii_digit() { return false; }
    for &b in &bytes[4..] { if !b.is_ascii_alphanumeric() { return false; } }
    let rearranged: String = compact[4..].chars().chain(compact[..4].chars()).collect();
    let mut expanded = String::with_capacity(rearranged.len() * 2);
    for ch in rearranged.chars() {
        if ch.is_ascii_digit() { expanded.push(ch); }
        else if ch.is_ascii_uppercase() {
            expanded.push_str(&((ch as u32 - 'A' as u32 + 10).to_string()));
        } else { return false; }
    }
    let mut acc: u64 = 0;
    for ch in expanded.chars() {
        let d = (ch as u8 - b'0') as u64;
        acc = (acc * 10 + d) % 97;
    }
    acc == 1
}

/// Phone structural validation — rejects lottery sequences.
fn looks_like_phone(raw: &str) -> bool {
    if raw.contains('+') { return true; }
    let mut run = 0usize;
    for ch in raw.chars() {
        if ch.is_ascii_digit() { run += 1; if run >= 3 { return true; } }
        else { run = 0; }
    }
    let mut has_digit_group = false;
    for token in raw.split_whitespace() {
        let token_digits = token.chars().filter(|c| c.is_ascii_digit()).count();
        if token_digits == 0 { continue; }
        has_digit_group = true;
        if token_digits < 2 { return false; }
    }
    has_digit_group
}

// ---------------------------------------------------------------------------
// Scam detector — 7 calibrated regex families.
// Ported from slm-guardrail `pipeline/scam.rs`.
// ---------------------------------------------------------------------------

pub struct ScamDetector;

impl ScamDetector {
    pub fn detect(text: &str) -> Vec<DetectorSignal> {
        let mut signals = Vec::new();
        let patterns: [(&str, &FRegex); 8] = [
            ("ADVANCE_FEE", advance_fee_re()),
            ("FAKE_GIVEAWAY", fake_giveaway_re()),
            ("CREDENTIAL_HARVEST", credential_harvest_re()),
            ("ROMANCE_SCAM", romance_scam_re()),
            ("CRYPTO_SCAM", crypto_scam_re()),
            ("QR_SCAM", qr_scam_re()),
            ("TECH_SUPPORT_SCAM", tech_support_scam_re()),
            ("URGENCY_MONEY", urgency_money_re()),
        ];
        let mut seen = HashSet::new();
        for (name, re) in &patterns {
            if re.is_match(text).unwrap_or(false) && seen.insert(*name) {
                signals.push(DetectorSignal {
                    category: categories::SCAM_FRAUD,
                    severity: Severity::SEVERE,
                    confidence: 0.82,
                    reason_code: format!("scam_{}", name.to_lowercase()),
                    action: Action::Warn,
                });
            }
        }
        signals
    }
}

fn advance_fee_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(r"(?i)\b(?:wire|transfer|deposit)\b.*\bfee\b").unwrap())
}

fn fake_giveaway_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(r"(?i)\b(?:congratulations|you\s+(?:have\s+)?won|claim\s+your\s+prize|you've\s+won)\b").unwrap())
}

fn credential_harvest_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(r"(?i)\b(?:verify|confirm|reset|update)\b.{0,120}?\b(?:password|account|login)\b|\b(?:password|account|login)\b.{0,120}?\b(?:verify|confirm|reset|update)\b").unwrap())
}

fn romance_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| {
        let endearment = r"\b(?:darling|sweetheart|honey|baby|my\s+love|beloved|dear|my\s+heart|soulmate)\b";
        let ask = r"\b(?:gift\s*cards?|wire(?:d|s|ing)?|western\s+union|money\s*gram|send\s+\$?\d+|need\s+(?:money|cash)|loan\s+me|borrow|repay\s+you|itunes\s+card|apple\s+gift\s+card|google\s+play\s+card|amazon\s+card|steam\s+card)\b";
        let pattern = format!("(?is)(?:{endearment}.{{0,160}}?{ask})|(?:{ask}.{{0,160}}?{endearment})");
        FRegex::new(&pattern).unwrap()
    })
}

fn crypto_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| {
        let assets = r"(?:btc|eth|usdt|usdc|sol|bnb|xrp|ada|doge|matic|ltc|bitcoin|ethereum|tether|crypto)";
        let pattern = format!(
            r"(?i)(?:\bsend\b\s+(?:me\s+)?(?:\d+(?:\.\d+)?\s*)?{assets}\b.{{0,80}}?\b(?:return|get|receive|i'?ll\s+(?:send|return))\b)|\b(?:guaranteed|risk\-?free)\b.{{0,20}}?\b(?:returns?|profits?|gains?|roi)\b|\b\d+\s*%\s+(?:returns?|profits?|gains?|roi)\b|\bpump\s+and\s+dump\b|\bsend\b\s+(?:to\s+)?(?:my\s+)?(?:wallet\s+address|btc\s+address|eth\s+address|(?:bitcoin|ethereum)\s+address)\b|(?:\b(?:share|give|tell|provide|reveal|paste|forward)\b).{{0,40}}?\b(?:seed\s*phrase|seed\s*words?|recovery\s*phrase|recovery\s*words?|mnemonic(?:\s*phrase)?|private\s*key|secret\s*key)\b|\b(?:seed\s*phrase|recovery\s*phrase|mnemonic|private\s*key)\b.{{0,80}}?\b(?:refund|double|2x|guaranteed)\b",
        );
        FRegex::new(&pattern).unwrap()
    })
}

/// Urgency + money/crypto request pattern (legacy compatibility).
fn urgency_money_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| {
        FRegex::new(r"(?i)\b(?:urgent|immediately|act\s+now|limited\s+time|expires\s+today|final\s+notice|last\s+chance)\b.{0,80}?\b(?:send\s+money|wire\s+transfer|gift\s+card|bitcoin|crypto|paypal|venmo|cash\s+app|western\s+union)\b").unwrap()
    })
}

fn qr_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\bscan\s+(?:this\s+)?qr\b.{0,80}?\b(?:verify|account|bank|pay|payment|transfer|login|sign\s*in|locked?|update|confirm|parking|ticket|ev\s+charger|charger)\b|\b(?:parking\s+ticket|ev\s+charger)\s+qr\b",
    ).unwrap())
}

fn tech_support_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\b(?:virus|malware|trojan|infection|infected|hacked?|compromised?|breach(?:ed)?)\b.{0,120}?\b(?:call|dial|phone|contact)\b.{0,40}?(?:\+?\d[\d\-\s().]{6,}\d|number)|\b(?:microsoft|apple|google|norton|mcafee|windows\s+defender)\s+(?:support|technician|engineer|helpdesk|security)\b|\b(?:call|dial)\s+(?:microsoft|apple|norton|mcafee)\b",
    ).unwrap())
}

// ---------------------------------------------------------------------------
// URL risk detector — bare-host + lookalike-brand + shortener + code-ext guard.
// Ported from slm-guardrail `pipeline/url.rs`.
// ---------------------------------------------------------------------------

pub struct UrlDetector;

impl UrlDetector {
    pub fn detect(text: &str) -> Vec<DetectorSignal> {
        let score = score_url_risk(text);
        if score >= 0.85 {
            vec![DetectorSignal {
                category: categories::SCAM_FRAUD, severity: Severity::SEVERE,
                confidence: score, reason_code: "url_high_risk".into(), action: Action::Warn,
            }]
        } else if score >= 0.5 {
            vec![DetectorSignal {
                category: categories::SCAM_FRAUD, severity: Severity::BORDERLINE,
                confidence: score, reason_code: "url_moderate_risk".into(), action: Action::Warn,
            }]
        } else {
            vec![]
        }
    }
}

/// Aggregate URL risk score in [0.0, 1.0].
pub fn score_url_risk(normalized_text: &str) -> f64 {
    let cleaned = crate::normalize::strip_zero_width(normalized_text);
    let email_spans: Vec<(usize, usize)> = EMAIL_RE.get_or_init(|| {
        Regex::new(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+").unwrap()
    }).find_iter(&cleaned).map(|m| (m.start(), m.end())).collect();
    let inside_email = |start: usize, end: usize| -> bool {
        email_spans.iter().any(|(es, ee)| *es <= start && end <= *ee)
    };

    let url_re = URL_RE.get_or_init(|| {
        Regex::new(r#"(?i)(?:https?://|www\.)[^\s<>"']{3,}"#).unwrap()
    });
    let mut urls: Vec<String> = url_re.find_iter(&cleaned).map(|m| m.as_str().to_string()).collect();

    // Bare-host candidates
    let bare_re = BARE_HOST_RE.get_or_init(|| {
        FRegex::new(r#"(?i)(?<![\w@])(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:/[^\s<>"']*)?"#).unwrap()
    });
    for m in bare_re.find_iter(&cleaned).flatten() {
        let candidate = &cleaned[m.start()..m.end()];
        if urls.iter().any(|u| u.contains(candidate) || candidate.contains(u.as_str())) { continue; }
        if inside_email(m.start(), m.end()) { continue; }
        let host = extract_host(candidate);
        let (first_label, last_label) = if host.contains('.') {
            let first = host.split('.').next().unwrap_or("");
            let last = host.rsplit('.').next().unwrap_or("");
            (first.to_string(), last.to_string())
        } else { (host.clone(), host.clone()) };
        let lowered = candidate.to_ascii_lowercase();
        let last_known = known_tlds().contains(last_label.as_str());
        let first_shortener = url_shortener_labels().contains(first_label.as_str());
        let lookalike = lookalike_brand_re().is_match(&lowered);
        if !last_known && !first_shortener && !lookalike { continue; }
        // Code-identifier guard — skip file.py, script.sh, main.rs
        if code_extension_overlap().contains(last_label.as_str())
            && host.chars().filter(|c| *c == '.').count() == 1
            && !candidate.contains('/') && !candidate.contains('?')
            && !first_shortener && !lookalike { continue; }
        urls.push(candidate.to_string());
    }

    if urls.is_empty() { return 0.0; }
    let mut max_score: f64 = 0.0;
    for url in &urls {
        let lowered = url.to_ascii_lowercase();
        let host = extract_host(url);
        let mut score: f64 = 0.2;
        for tld in high_risk_tlds() {
            let dot_tld = format!(".{tld}");
            let dot_tld_slash = format!(".{tld}/");
            if (lowered.ends_with(&dot_tld) || lowered.contains(&dot_tld_slash)) && score < 0.9 { score = 0.9; }
        }
        for kw in high_risk_keywords() {
            if lowered.contains(kw) && score < 0.85 { score = 0.85; }
        }
        if url_shorteners().contains(host.as_str()) && score < 0.85 { score = 0.85; }
        let first_label = host.split('.').next().unwrap_or(host.as_str());
        if url_shortener_labels().contains(first_label) && score < 0.85 { score = 0.85; }
        if lookalike_brand_re().is_match(&lowered) && score < 0.9 { score = 0.9; }
        if score > max_score { max_score = score; }
    }
    max_score.min(1.0)
}

fn extract_host(url: &str) -> String {
    let mut lowered = url.to_ascii_lowercase();
    if let Some(idx) = lowered.find("://") { lowered = lowered[idx + 3..].to_string(); }
    for sep in ['/', '?', '#'] {
        if let Some(idx) = lowered.find(sep) { lowered.truncate(idx); }
    }
    if let Some(stripped) = lowered.strip_prefix("www.") { lowered = stripped.to_string(); }
    lowered
}

static URL_RE: OnceLock<Regex> = OnceLock::new();
static BARE_HOST_RE: OnceLock<FRegex> = OnceLock::new();

fn high_risk_tlds() -> &'static HashSet<&'static str> {
    static CELL: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CELL.get_or_init(|| ["zip", "mov", "top", "click", "country", "xyz", "ml", "tk", "cf", "ga", "gq"].iter().copied().collect())
}

fn high_risk_keywords() -> &'static HashSet<&'static str> {
    static CELL: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CELL.get_or_init(|| ["login", "verify", "account", "secure", "update"].iter().copied().collect())
}

fn known_tlds() -> &'static HashSet<&'static str> {
    static CELL: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CELL.get_or_init(|| {
        let mut s: HashSet<&'static str> = high_risk_tlds().iter().copied().collect();
        for t in ["com","net","org","edu","gov","mil","int","info","biz","name","pro","io","co","ai","app","dev","me","tv","cc","ws","ly","gl","sh","gg","fm","im","so","in","site","online","store","tech","blog","cloud","page","live","news","media","design","tools","video","shop","world","us","uk","ca","au","nz","jp","fr","de","it","es","nl","se","no","dk","fi","pl","cz","hu","gr","pt","ie","at","be","ch","ro","bg","sk","si","hr","lt","lv","ee","lu","is","mt","cy","eu","cn","kr","hk","tw","sg","id","my","ph","th","vn","pk","bd","lk","np","kh","la","mm","br","mx","ar","cl","pe","ve","uy","py","bo","ec","cr","pa","gt","ni","hn","sv","do","za","ng","eg","ke","ma","tn","dz","et","gh","tz","ug","sn","ci","rw","ae","sa","qa","kw","bh","om","tr","il","ir","iq","jo","lb","ru","ua","by","kz","uz","ge","am","az"] {
            s.insert(t);
        }
        s
    })
}

fn url_shorteners() -> &'static HashSet<&'static str> {
    static CELL: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CELL.get_or_init(|| ["bit.ly","tinyurl.com","t.co","goo.gl","t.ly","ow.ly","is.gd","buff.ly","rebrand.ly","shorturl.at","cutt.ly","bl.ink","v.gd"].iter().copied().collect())
}

fn url_shortener_labels() -> &'static HashSet<&'static str> {
    static CELL: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CELL.get_or_init(|| ["bit","tinyurl","bitly","tiny","shorten","short","rebrand","cutt","shorturl"].iter().copied().collect())
}

fn code_extension_overlap() -> &'static HashSet<&'static str> {
    static CELL: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CELL.get_or_init(|| ["py","rs","sh","cc","pl"].iter().copied().collect())
}

fn lookalike_brand_re() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    CELL.get_or_init(|| {
        let pattern = concat!(
            "(?i)",
            "(?:",
            r"chase|chse|wells\s*fargo|wellsfargo|bank\s*of\s*america|",
            r"bank[0o]famerica|citi(?:bank)?|hsbc|barclays|paypal|p4ypal|",
            r"google|g[o0][o0]gle|microsoft|micros[o0]ft|apple|appl[3e]|",
            r"amazon|amaz[o0]n|netflix|netfl1x",
            ")",
            r"[\-_.]+",
            r"(?:secure|security|verify|login|signin|account|update|support|help|check|payment|billing)",
        );
        Regex::new(pattern).unwrap()
    })
}

// ---------------------------------------------------------------------------
// Lexicon detector — script-aware word-boundary matching + LRU cache.
// Ported from slm-guardrail `pipeline/lexicon.rs`.
// ---------------------------------------------------------------------------

/// Unicode blocks where `\b` word-boundary anchoring is unsafe:
/// scripts that don't separate words with whitespace (CJK, Hangul, Thai, etc.).
const NO_WORD_BOUNDARY_RANGES: &[(u32, u32)] = &[
    (0x3040, 0x309F),   // Hiragana
    (0x30A0, 0x30FF),   // Katakana
    (0x31F0, 0x31FF),   // Katakana Phonetic Extensions
    (0x4E00, 0x9FFF),   // CJK Unified Ideographs
    (0x3400, 0x4DBF),   // CJK Extension A
    (0x20000, 0x2A6DF), // CJK Extension B
    (0xF900, 0xFAFF),   // CJK Compatibility Ideographs
    (0xAC00, 0xD7AF),   // Hangul Syllables
    (0x1100, 0x11FF),   // Hangul Jamo
    (0x3130, 0x318F),   // Hangul Compatibility Jamo
    (0x0E00, 0x0E7F),   // Thai
    (0x0E80, 0x0EFF),   // Lao
    (0x1000, 0x109F),   // Myanmar
    (0x1780, 0x17FF),   // Khmer
];

fn is_word_boundary_safe(ch: Option<char>) -> bool {
    let Some(ch) = ch else { return false };
    let cp = ch as u32;
    for (lo, hi) in NO_WORD_BOUNDARY_RANGES {
        if cp >= *lo && cp <= *hi { return false; }
    }
    ch.is_alphanumeric() || ch == '_'
}

fn compile_lexicon_token(token: &str) -> Regex {
    let mut pattern = regex::escape(token);
    if is_word_boundary_safe(token.chars().next()) {
        pattern = format!(r"\b{pattern}");
    }
    if is_word_boundary_safe(token.chars().last()) {
        pattern = format!("{pattern}\\b");
    }
    Regex::new(&pattern).expect("compiled lexicon regex must be valid")
}

const LEXICON_REGEX_CACHE_CAPACITY: usize = 4096;

fn token_cache() -> &'static Mutex<LruCache<String, Arc<Regex>>> {
    static CELL: OnceLock<Mutex<LruCache<String, Arc<Regex>>>> = OnceLock::new();
    CELL.get_or_init(|| {
        let cap = NonZeroUsize::new(LEXICON_REGEX_CACHE_CAPACITY)
            .expect("cache capacity must be non-zero");
        Mutex::new(LruCache::new(cap))
    })
}

fn cached_lexicon_regex(folded: &str) -> Arc<Regex> {
    let mut guard = token_cache().lock();
    if let Some(re) = guard.get(folded) { return Arc::clone(re); }
    let re = Arc::new(compile_lexicon_token(folded));
    guard.put(folded.to_string(), Arc::clone(&re));
    re
}

pub struct LexiconDetector;

impl LexiconDetector {
    pub fn detect(text: &str, lexicon: &[(String, u32, Severity)]) -> Vec<DetectorSignal> {
        let mut signals = Vec::new();
        for (term, category, severity) in lexicon {
            if term.is_empty() { continue; }
            let folded = caseless::default_case_fold_str(term);
            let re = cached_lexicon_regex(&folded);
            if re.is_match(text) {
                signals.push(DetectorSignal {
                    category: *category,
                    severity: *severity,
                    confidence: 0.75,
                    reason_code: format!("lexicon_match_{}", term.replace(' ', "_")),
                    action: if *severity >= Severity::SEVERE { Action::Block } else { Action::Warn },
                });
            }
        }
        signals
    }
}

// ---------------------------------------------------------------------------
// Orchestrator — run all detectors and resolve priority chain.
// ---------------------------------------------------------------------------

/// Run all deterministic detectors across multiple text views.
///
/// `pattern_text` — digit-preserving normalized view (for PII/URL).
/// `lexicon_views` — leetspeak-defanged views (for lexicon + scam).
///   The first view is the canonical normalized text; subsequent views are
///   defang variants (e.g. digit `1` → `l` vs `i`). Detectors union hits
///   across all views and dedupe by reason_code.
/// `lexicon` — signed policy pack lexicon entries.
pub fn run_all_detectors(
    pattern_text: &str,
    lexicon_views: &[String],
    lexicon: &[(String, u32, Severity)],
) -> LocalSignals {
    let mut signals = LocalSignals::default();

    // PII and URL run on the digit-preserving view only.
    for s in PiiDetector::detect(pattern_text) { signals.add(s); }
    for s in UrlDetector::detect(pattern_text) { signals.add(s); }

    // Scam and lexicon run across all defang variants — union hits, dedupe.
    let mut seen_scam: HashSet<String> = HashSet::new();
    let mut seen_lex: HashSet<String> = HashSet::new();
    for view in lexicon_views {
        for s in ScamDetector::detect(view) {
            if seen_scam.insert(s.reason_code.clone()) { signals.add(s); }
        }
        for s in LexiconDetector::detect(view, lexicon) {
            if seen_lex.insert(s.reason_code.clone()) { signals.add(s); }
        }
    }

    signals
}

/// Resolve deterministic signals into a verdict using the priority chain:
///   CHILD_SAFETY > SELF_HARM > PRIVATE_DATA > SCAM_FRAUD > HATE_SPEECH > VIOLENCE > NSFW > SPAM
pub fn resolve_priority_chain(signals: &LocalSignals) -> Option<DetectorSignal> {
    if signals.is_empty() { return None; }
    let priority = [
        categories::CHILD_SAFETY, categories::SELF_HARM, categories::PRIVATE_DATA,
        categories::SCAM_FRAUD, categories::HATE_SPEECH, categories::VIOLENCE,
        categories::NSFW, categories::SPAM,
    ];
    for &cat in &priority {
        if let Some(best) = signals.signals.iter()
            .filter(|s| s.category == cat)
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
        {
            return Some(best.clone());
        }
    }
    signals.signals.first().cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- PII tests ---

    #[test]
    fn test_email_detected() {
        assert!(PiiDetector::detect("contact alice@example.com").iter().any(|s| s.reason_code == "pii_email"));
    }

    #[test]
    fn test_credit_card_luhn_valid() {
        assert!(PiiDetector::detect("card 4111111111111111").iter().any(|s| s.reason_code == "pii_credit_card"));
    }

    #[test]
    fn test_credit_card_rejects_invalid_luhn() {
        assert!(!PiiDetector::detect("4111111111111112").iter().any(|s| s.reason_code == "pii_credit_card"));
    }

    #[test]
    fn test_phone_plus_format() {
        assert!(PiiDetector::detect("call +1-415-555-0199").iter().any(|s| s.reason_code == "pii_phone"));
    }

    #[test]
    fn test_phone_french_groups() {
        assert!(PiiDetector::detect("01 23 45 67 89").iter().any(|s| s.reason_code == "pii_phone"));
    }

    #[test]
    fn test_lottery_not_phone() {
        assert!(!PiiDetector::detect("4 17 23 29 36 41").iter().any(|s| s.reason_code == "pii_phone"));
    }

    #[test]
    fn test_ssn_valid() {
        assert!(PiiDetector::detect("ssn: 123-45-6789").iter().any(|s| s.reason_code == "pii_ssn"));
    }

    #[test]
    fn test_iban_valid() {
        assert!(PiiDetector::detect("IBAN GB82WEST12345698765432").iter().any(|s| s.reason_code == "pii_iban"));
    }

    #[test]
    fn test_iban_invalid() {
        assert!(!PiiDetector::detect("IBAN GB82WEST12345698765433").iter().any(|s| s.reason_code == "pii_iban"));
    }

    #[test]
    fn test_credential_leak() {
        assert!(PiiDetector::detect("user: admin password: hunter2").iter().any(|s| s.reason_code == "pii_credentials"));
    }

    // --- Scam tests ---

    #[test]
    fn test_scam_fake_giveaway() {
        let sigs = ScamDetector::detect("Congratulations! You've won $1,000,000");
        assert!(sigs.iter().any(|s| s.reason_code.contains("fake_giveaway")));
    }

    #[test]
    fn test_scam_credential_harvest() {
        let sigs = ScamDetector::detect("Please verify your password to confirm your account");
        assert!(sigs.iter().any(|s| s.reason_code.contains("credential_harvest")));
    }

    #[test]
    fn test_scam_crypto_guaranteed_returns() {
        let sigs = ScamDetector::detect("Invest in crypto with guaranteed 500% returns");
        assert!(sigs.iter().any(|s| s.reason_code.contains("crypto_scam")));
    }

    #[test]
    fn test_scam_tech_support() {
        let sigs = ScamDetector::detect("Your computer has a virus. Call Microsoft support at +1-800-555-0199");
        assert!(sigs.iter().any(|s| s.reason_code.contains("tech_support")));
    }

    #[test]
    fn test_scam_advance_fee() {
        let sigs = ScamDetector::detect("Send the wire transfer fee to release the funds");
        assert!(sigs.iter().any(|s| s.reason_code.contains("advance_fee")));
    }

    #[test]
    fn test_scam_qr() {
        let sigs = ScamDetector::detect("Scan this QR code to verify your account");
        assert!(sigs.iter().any(|s| s.reason_code.contains("qr_scam")));
    }

    // --- URL risk tests ---

    #[test]
    fn test_url_high_risk_tld() {
        assert!((score_url_risk("download https://payload.zip") - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_url_code_extension_no_fire() {
        for benign in ["see file.py", "run script.sh", "in main.rs"] {
            assert_eq!(score_url_risk(benign), 0.0, "false positive: {benign}");
        }
    }

    #[test]
    fn test_url_lookalike_brand() {
        let s = score_url_risk("https://paypal-secure.example.com/verify");
        assert!(s >= 0.85, "expected lookalike brand score, got {s}");
    }

    #[test]
    fn test_url_shortener() {
        assert!(score_url_risk("click https://bit.ly/abc") >= 0.85);
    }

    #[test]
    fn test_url_email_not_bare_host() {
        assert_eq!(score_url_risk("contact alice@example.com please"), 0.0);
    }

    // --- Lexicon tests ---

    #[test]
    fn test_lexicon_word_boundary() {
        let lex = vec![("kill".into(), categories::VIOLENCE, Severity::SEVERE)];
        let sigs = LexiconDetector::detect("kill them", &lex);
        assert!(!sigs.is_empty());
        // "skillful" should NOT match "kill" due to word boundary
        let sigs2 = LexiconDetector::detect("skillful work", &lex);
        assert!(sigs2.is_empty(), "word boundary false positive: {:?}", sigs2);
    }

    #[test]
    fn test_lexicon_cjk_no_word_boundary() {
        // CJK token should match without \b (CJK has no word boundaries)
        let lex = vec![("爆破".into(), categories::VIOLENCE, Severity::SEVERE)];
        let sigs = LexiconDetector::detect("爆破方法", &lex);
        assert!(!sigs.is_empty(), "CJK lexicon should match without word boundary");
    }

    #[test]
    fn test_lexicon_cache_reuses() {
        let lex = vec![("test".into(), categories::SPAM, Severity::BORDERLINE)];
        // Call twice — should reuse cached regex
        let _ = LexiconDetector::detect("this is a test", &lex);
        let _ = LexiconDetector::detect("another test here", &lex);
        // If we get here without panicking, the cache works
    }

    #[test]
    fn test_lexicon_arabic_match() {
        // Arabic lexicon terms should match Arabic text with question mark
        let lex = vec![("كيف أصنع سلاحا".into(), categories::VIOLENCE, Severity::SEVERE)];
        let sigs = LexiconDetector::detect("كيف أصنع سلاحا؟", &lex);
        assert!(!sigs.is_empty(), "Arabic lexicon should match text with question mark: {:?}", sigs);
    }

    #[test]
    fn test_lexicon_arabic_normalized() {
        // Test Arabic matching through the full normalize_for_lexicon pipeline
        let normalized = crate::normalize::normalize_for_lexicon("كيف أصنع سلاحا؟");
        let lex = vec![("كيف أصنع سلاحا".into(), categories::VIOLENCE, Severity::SEVERE)];
        let sigs = LexiconDetector::detect(&normalized, &lex);
        assert!(!sigs.is_empty(), "Arabic lexicon should match normalized text: normalized='{}', sigs={:?}", normalized, sigs);
    }
}

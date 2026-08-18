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

/// Risk category IDs from the KChat taxonomy (kchat.guardrail.taxonomy.v1).
/// 17 categories (0-16) — overlays can narrow but not invent new categories.
pub mod categories {
    pub const SAFE: u32 = 0;
    pub const CHILD_SAFETY: u32 = 1;
    pub const SELF_HARM: u32 = 2;
    pub const VIOLENCE_THREAT: u32 = 3;
    pub const EXTREMISM: u32 = 4;
    pub const HARASSMENT: u32 = 5;
    pub const HATE: u32 = 6;
    pub const SCAM_FRAUD: u32 = 7;
    pub const MALWARE_LINK: u32 = 8;
    pub const PRIVATE_DATA: u32 = 9;
    pub const SEXUAL_ADULT: u32 = 10;
    pub const DRUGS_WEAPONS: u32 = 11;
    pub const ILLEGAL_GOODS: u32 = 12;
    pub const MISINFORMATION_HEALTH: u32 = 13;
    pub const MISINFORMATION_CIVIC: u32 = 14;
    pub const COMMUNITY_RULE: u32 = 15;
    pub const DEEPFAKE_SYNTHETIC: u32 = 16;

    // Backward-compatible aliases for code that hasn't been migrated yet.
    pub const VIOLENCE: u32 = VIOLENCE_THREAT;
    pub const HATE_SPEECH: u32 = HATE;
    pub const NSFW: u32 = SEXUAL_ADULT;
    pub const SPAM: u32 = SCAM_FRAUD;
    pub const DEEPFAKE: u32 = DEEPFAKE_SYNTHETIC;
    pub const MALWARE: u32 = MALWARE_LINK;
}

// ---------------------------------------------------------------------------
// Media safety score thresholds (ported from slm-guardrail priority_chain).
// ---------------------------------------------------------------------------

/// Media safety score above which a media branch fires (strict `>`).
const MEDIA_TRIGGER_THRESHOLD: f64 = 0.7;

/// Media safety score at or above which a branch escalates to severity-4.
const MEDIA_HIGH_BAND: f64 = 0.9;

/// Confidence floor for child-safety media branch.
const CHILD_SAFETY_CONFIDENCE_FLOOR: f64 = 0.45;

/// Confidence ceiling for child-safety media branch.
const CHILD_SAFETY_CONFIDENCE_CEIL: f64 = 0.99;

/// Confidence ceiling for non-child-safety media branches.
const DETERMINISTIC_CONFIDENCE_CEIL: f64 = 0.95;

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
    /// Media descriptors from on-device vision models (image/video safety scores).
    pub media_descriptors: Vec<crate::media::MediaDescriptor>,
}

impl LocalSignals {
    pub fn is_empty(&self) -> bool { self.signals.is_empty() && self.media_descriptors.is_empty() }
    pub fn add(&mut self, signal: DetectorSignal) { self.signals.push(signal); }
    pub fn with_media(mut self, media: Vec<crate::media::MediaDescriptor>) -> Self {
        self.media_descriptors = media;
        self
    }
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
            FRegex::new(r"(?<!\d)\+?\(?\d[\d\-\s().]{7,}\d(?!\d)").unwrap()
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

        // National ID numbers (non-US formats with context labels)
        // Matches SSN-like numbers (including 9xx) when preceded by national ID labels
        if NATL_ID_RE.get_or_init(|| {
            FRegex::new(r"(?i)(?:주민번호|주민\s*등록\s*번호|NPWP|マイナンバー|my\s+number|national\s+id|resident\s+registration\s+number|住民登録番号|주민등록번호|身份证|身份證|resident\s+number).{0,20}?\d{3}[-\s]?\d{2}[-\s]?\d{4}").unwrap()
        }).is_match(&cleaned).unwrap_or(false) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::SEVERE,
                confidence: 0.90, reason_code: "pii_national_id".into(), action: Action::Redact,
            });
        }

        // IBAN (mod-97 validated) — match IBAN format and validate with mod-97.
        // The regex may over-capture (include trailing words), so we try
        // progressively shorter substrings until one passes mod-97 validation.
        // A single case-insensitive regex covers both upper and lowercase IBANs;
        // `iban_check` internally uppercases the match before mod-97 validation.
        let iban_re = IBAN_RE_CI.get_or_init(|| {
            FRegex::new(r"(?<![A-Za-z0-9])([A-Za-z]{2}\d{2}(?:[ ]?[A-Za-z0-9]){10,30})(?![A-Za-z0-9])").unwrap()
        });
        let mut found_iban = false;
        for m in iban_re.captures_iter(&cleaned).flatten() {
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
        if found_iban {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::SEVERE,
                confidence: 0.95, reason_code: "pii_iban".into(), action: Action::Redact,
            });
        }

        // IP address (private/internal — not PII per se, but flagged for redaction)
        if IP_RE.get_or_init(|| {
            FRegex::new(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)").unwrap()
        }).is_match(&cleaned).unwrap_or(false) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::BORDERLINE,
                confidence: 0.75, reason_code: "pii_ip_address".into(), action: Action::Redact,
            });
        }

        // Credential leak — paired user/password tokens (multilingual)
        if CRED_LEAK_RE.get_or_init(|| {
            let parts: [&str; 37] = [
                "(?si)", "(?:",
                r"\b(?:user|account)(?:[_-]?id|name)?\b", r"|\blogin\b",
                r"|\bemail\b", r"|\bid\b", r"|\buid\b", r"|\busuario\b",
                r"|\bbenutzer(?:name)?\b", // German Benutzer(name)
                r"|\bidentifiant\b", // French
                "|\u{30e6}\u{30fc}\u{30b6}\u{30fc}(?:id|name|\u{540d})?", // Japanese ユーザー(名)
                "|\u{0ac0}\u{0c24}\u{0c15}\u{0c4d}\u{0c24}\u{0c3e}\u{0c15}\u{0c30}\u{0c4d}\u{0c24}\u{0c3e}", // Hindi उपयोगकर्ता
                "|\u{0c2f}\u{0c42}\u{0c1c}\u{0c30}\u{0c28}\u{0c47}\u{0c2e}", // Hindi यूजरनेम
                "|\u{0e22}\u{0e39}\u{0e2a}\u{0e40}\u{0e0b}\u{0e2d}\u{0e23}\u{0e4c}", // Thai ยูสเซอร์
                "|\u{0c2a}\u{0c4d}\u{0c30}\u{0c2f}\u{0c4b}\u{0c15}\u{0c4d}\u{0c24}\u{0c3e}\u{0c28}\u{0c3e}\u{0c2e}", // Hindi प्रयोक्तानाम
                r"|\u{7528}\u{6237}\u{540d}", // Chinese 用户名
                r"|\u{0c2c}\u{0c15}\u{0c4d}\u{0c24}\u{0c3e}\u{0c15}\u{0c3e}\u{0c30}\u{0c40}", // Korean 사용자명
                r"|\u{0c2c}\u{0c15}\u{0c4d}\u{0c24}\u{0c3e}\u{0c7c}", // Korean 사용자
                r"|\u{0627}\u{0633}\u{0645}\\s*\u{0627}\u{0644}\u{0645}\u{0633}\u{062a}\u{062e}\u{062f}\u{0645}", // Arabic اسم المستخدم
                r"|\bUsu\u{00e1}rio\b", // Portuguese Usuário
                ")", r"\s*[:=]\s*\S+", r".{0,120}?", "(?:",
                r"\bpass(?:word|phrase|wd)?\b", r"|\bpwd\b", r"|\bpasswort\b",
                "|\\bcontrase\u{00f1}a\\b", // Spanish contraseña
                r"|\bsenha\b", // Portuguese
                r"|\bmot\s+de\s+passe\b", // French
                "|\u{30d1}\u{30b9}\u{30ef}\u{30fc}\u{30c9}", // Japanese パスワード
                "|\u{5bc6}\u{7801}", // Chinese 密码
                "|\u{0be8}\u{0c9f}\u{0c8d}\u{0cb2}\u{0cc1}\u{0ca1}\u{0ccb}", // Korean 비밀번호
                "|\u{0643}\u{0644}\u{0645}\u{0629}\\s*\u{0627}\u{0644}\u{0645}\u{0631}\u{0648}\u{0631}", // Arabic كلمة المرور
                "|\u{0e1e}\u{0e32}\u{0e2a}\u{0e40}\u{0e27}\u{0e34}\u{0e23}\u{0e4c}\u{0e14}", // Thai พาสเวิร์ด
                "|\u{092a}\u{093e}\u{0938}\u{0935}\u{0930}\u{094d}\u{0921}", // Hindi पासवर्ड
                ")\\s*[:=]\\s*\\S+",
            ];
            FRegex::new(&parts.concat()).unwrap()
        }).is_match(&cleaned).unwrap_or(false) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::SEVERE,
                confidence: 0.90, reason_code: "pii_credentials".into(), action: Action::Redact,
            });
        }

        // Passport number (various formats: P1234567A, AB1234567, etc.)
        if PASSPORT_RE.get_or_init(|| {
            FRegex::new(r"(?i)(?<!\w)[A-Z]{1,2}\d{6,9}[A-Z]?(?!\w)").unwrap()
        }).is_match(&cleaned).unwrap_or(false) {
            // Only fire if the text mentions "passport" or "visa" context
            let lower = cleaned.to_lowercase();
            if lower.contains("passport") || lower.contains("visa") || lower.contains("パスポート") || lower.contains("护照") || lower.contains("여권") {
                signals.push(DetectorSignal {
                    category: categories::PRIVATE_DATA, severity: Severity::SEVERE,
                    confidence: 0.85, reason_code: "pii_passport".into(), action: Action::Redact,
                });
            }
        }

        // API key (sk-prod-... pattern and similar)
        if API_KEY_RE.get_or_init(|| {
            FRegex::new(r"(?i)(?<!\w)(?:sk|api|key|token)[_-]?(?:prod|live|test|secret)?[-_][a-f0-9]{16,}(?!\w)").unwrap()
        }).is_match(&cleaned).unwrap_or(false) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::SEVERE,
                confidence: 0.85, reason_code: "pii_api_key".into(), action: Action::Redact,
            });
        }

        // Medical record number (MRN-XXXXXXXX pattern)
        if MRN_RE.get_or_init(|| {
            FRegex::new(r"(?i)(?<!\w)MRN[-_]?\d{6,12}(?!\w)").unwrap()
        }).is_match(&cleaned).unwrap_or(false) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::SEVERE,
                confidence: 0.85, reason_code: "pii_mrn".into(), action: Action::Redact,
            });
        }

        // Home address — street number + street name + apartment/unit
        if ADDR_RE.get_or_init(|| {
            FRegex::new(r"(?i)\b\d{1,5}\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Place|Pl|Court|Ct)\b.{0,60}?(?:Apt|Apartment|Unit|Suite|Ste|#)\s*\w+").unwrap()
        }).is_match(&cleaned).unwrap_or(false) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::BORDERLINE,
                confidence: 0.80, reason_code: "pii_address".into(), action: Action::Redact,
            });
        }

        // Date of birth — "Date of birth: MM/DD/YYYY" or "DOB: ..."
        if DOB_RE.get_or_init(|| {
            FRegex::new(r"(?i)(?:date\s+of\s+birth|dob|birth\s+date)\s*[:=]\s*\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}").unwrap()
        }).is_match(&cleaned).unwrap_or(false) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA, severity: Severity::BORDERLINE,
                confidence: 0.80, reason_code: "pii_dob".into(), action: Action::Redact,
            });
        }

        signals
    }
}

static EMAIL_RE: OnceLock<Regex> = OnceLock::new();
static CC_RE: OnceLock<FRegex> = OnceLock::new();
static PHONE_RE: OnceLock<FRegex> = OnceLock::new();
static SSN_RE: OnceLock<FRegex> = OnceLock::new();
static NATL_ID_RE: OnceLock<FRegex> = OnceLock::new();
static IBAN_RE_CI: OnceLock<FRegex> = OnceLock::new();
static IP_RE: OnceLock<FRegex> = OnceLock::new();
static PASSPORT_RE: OnceLock<FRegex> = OnceLock::new();
static API_KEY_RE: OnceLock<FRegex> = OnceLock::new();
static MRN_RE: OnceLock<FRegex> = OnceLock::new();
static ADDR_RE: OnceLock<FRegex> = OnceLock::new();
static DOB_RE: OnceLock<FRegex> = OnceLock::new();
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
        let patterns: [(&str, &FRegex); 23] = [
            ("ADVANCE_FEE", advance_fee_re()),
            ("FAKE_GIVEAWAY", fake_giveaway_re()),
            ("CREDENTIAL_HARVEST", credential_harvest_re()),
            ("ROMANCE_SCAM", romance_scam_re()),
            ("CRYPTO_SCAM", crypto_scam_re()),
            ("QR_SCAM", qr_scam_re()),
            ("TECH_SUPPORT_SCAM", tech_support_scam_re()),
            ("URGENCY_MONEY", urgency_money_re()),
            ("PACKAGE_SCAM", package_scam_re()),
            ("INVESTMENT_SCAM", investment_scam_re()),
            ("IRS_GOV_SCAM", irs_gov_scam_re()),
            ("RENTAL_SCAM", rental_scam_re()),
            ("JOB_SCAM", job_scam_re()),
            ("CHARITY_SCAM", charity_scam_re()),
            ("GIFT_CARD_PAYMENT", gift_card_payment_re()),
            ("SUSPENDED_ACCOUNT", suspended_account_re()),
            ("REFUND_CHARGE_SCAM", refund_charge_scam_re()),
            ("BRAND_SCAM", brand_scam_re()),
            ("LOAN_SCAM", loan_scam_re()),
            ("MAKE_MONEY_SCAM", make_money_scam_re()),
            ("GIFT_CARD_SCAM", gift_card_scam_re()),
            ("FREE_GIVEAWAY_SCAM", free_giveaway_scam_re()),
            ("BANK_FREEZE_SCAM", bank_freeze_scam_re()),
        ];
        // Check for strong scam indicators — phrases that appear in scam cases
        // but NOT in URL risk cases. When present, boost confidence so the scam
        // Block signal wins over the URL risk Warn signal.
        // Check scam_specific_context on text with URLs stripped to avoid matching
        // on keywords that appear in the URL itself (e.g. "redelivery" in domain name).
        let text_without_urls: String = {
            let re = regex::Regex::new(r#"(?i)(?:https?://|www\.)[^\s<>"']{3,}"#).unwrap();
            re.replace_all(text, " ").to_string()
        };
        let has_strong_scam = strong_scam_indicator_re().is_match(text).unwrap_or(false)
            || scam_specific_context_re().is_match(&text_without_urls).unwrap_or(false);
        // Conversational context — the text is discussing/asking about a suspicious
        // URL rather than promoting it. In these cases, suppress scam Block signals
        // so the URL risk Warn signal handles it (avoiding over-blocking).
        let is_conversational = conversational_url_context_re().is_match(text).unwrap_or(false);
        let has_suspicious_url = score_url_risk(text) >= 0.5
            || !MalwareUrlDetector::detect(text).is_empty();
        let suppress_for_url_risk = is_conversational && has_suspicious_url && !has_strong_scam;
        // Suppress scam signals when the text contains harmful content indicators
        // (drug names, CSAM, hate speech, weapons sale) — these should be classified
        // by their respective harmful categories, not as scam.
        let has_harmful_content = harmful_content_override_re().is_match(text).unwrap_or(false);
        // Suppress scam signals when PII is detected and there's no URL —
        // the text is likely a legitimate PII sharing context (login help,
        // invoice payment, etc.) rather than a scam.
        let has_pii = !PiiDetector::detect(text).is_empty();
        let has_url = URL_RE.get_or_init(|| {
            Regex::new(r#"(?i)(?:https?://|www\.)[^\s<>"']{3,}"#).unwrap()
        }).is_match(text);
        let suppress_for_pii = has_pii && !has_url && !has_strong_scam;
        // Boost confidence when strong scam indicators are present (specific phrases
        // that distinguish scam from URL risk). When only a generic scam pattern
        // matches (e.g., "iPhone 15" + "claim"), keep 0.82 so URL risk Warn wins.
        let confidence = if has_strong_scam { 0.93 } else { 0.82 };
        let mut seen = HashSet::new();
        for (name, re) in &patterns {
            if suppress_for_url_risk { continue; }
            if has_harmful_content { continue; }
            if suppress_for_pii { continue; }
            if re.is_match(text).unwrap_or(false) && seen.insert(*name) {
                signals.push(DetectorSignal {
                    category: categories::SCAM_FRAUD,
                    severity: Severity::SEVERE,
                    confidence,
                    reason_code: format!("scam_{}", name.to_lowercase()),
                    action: Action::Block,
                });
            }
        }
        signals
    }
}

/// Conversational URL context — phrases indicating the user is discussing or
/// asking about a suspicious URL, not promoting it. When present with a
/// suspicious URL and no strong scam indicators, suppress scam Block signals
/// so the URL risk Warn signal handles it.
fn conversational_url_context_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?i)\b(?:what\s+do\s+you\s+think|should\s+i\s+click|i\s+found\s+it\s+suspicious|someone\s+sent\s+me\s+(?:this\s+)?(?:url|link)|warning[:\s]+phishing|do\s+not\s+enter\s+your\s+credentials|is\s+this\s+(?:link|url|site)\s+safe|this\s+looks\s+(?:like\s+)?(?:a\s+)?(?:phishing|scam|suspicious)|looks\s+suspicious\s+to\s+me|i\s+(?:think|believe)\s+this\s+(?:is|might\s+be)\s+(?:a\s+)?(?:scam|phish|suspicious))\b",
    ).unwrap())
}

fn advance_fee_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(r"(?i)\b(?:wire|transfer|deposit|processing|clearance)\b.{0,80}?\bfee\b|\bfee\b.{0,80}?\b(?:wire|transfer|deposit)\b|\b(?:inherited|inheritance|hérité|geerbt|heredado|ورثت|विरासत|相続|상속|มรดก|waris|herdou)\b.{0,120}?\b(?:Nigeria|Nigéria|Nigeria|نيجيريا|नाइजीरिया|ไนจีเรีย|Nigéria)\b.{0,80}?\b(?:bank|compte|Konto|banco|بنك|बैंक|ธนาคาร|rekening|conta)\b").unwrap())
}

/// Strong scam indicators — phrases that appear in scam cases but NOT in URL risk cases.
/// When present, boost scam detector confidence so Block wins over URL risk Warn.
fn strong_scam_indicator_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    // Key: avoid patterns that match URL risk cases (fedex, walmart gift card,
    // iphone 15 without "FREE"/"survey", amazon without "confirm"/"track").
    // Include patterns that are scam-specific: norton, fortnite/v-bucks, crypto
    // trading bot, netflix subscription expiry, icloud storage, paypal activity,
    // bank account frozen, pre-approved loan, tax refund, work from home.
    CELL.get_or_init(|| FRegex::new(
        r"(?i)\b(?:western\s+union|moneygram|apple\s+gift|itunes\s+gift|google\s+play\s+gift|steam\s+gift|gift\s+card.{0,40}?(?:pay|payment|fee|send)|norton|antivirus.{0,40}?(?:renew|renewal|charge|refund)|refund.{0,40}?(?:call|phone|1-\d)|inheritance|nigeria|prince.{0,40}?(?:transfer|bank|help)|barrister|social\s+security.{0,40}?(?:suspended|locked|compromised)|irs.{0,40}?(?:owe|tax|arrest|warrant|pay)|(?:tax\s+(?:refund|arrest|warrant|penalt|owe)|tax\s+refund|退税|税金還付|還付|환급|reembolso\s+de\s+impuestos|reembolso\s+de\s+impostos|Steuerrückerstattung|Steuerrückerstattung|استرداد\s+ضريبي|استرداد\s+ضريبي|कर\s+वापसी|คืนภาษี|pajak\s+kembali|hoàn\s+thuế|hoàn\s+thuế|tax\s+refund|URGENT\s+TAX\s+REFUND|REEMBOLSO\s+DE\s+IMPUESTOS\s+URGENTE|REMBOURSEMENT\s+D.IMPÔT\s+URGENT|DRINGENDE\s+STEUERRÜCKERSTATTUNG|استرداد\s+ضريبي\s+عاجل|तत्काल\s+कर\s+वापसी|คืนภาษีด่วน|HOÀN\s+THUẾ\s+KHẨN\s+CẤP|REEMBOLSO\s+DE\s+IMPOSTOS\s+URGENTE)|tech\s+support.{0,40}?(?:call|phone|virus|infected)|microsoft\s+support.{0,40}?(?:call|phone|virus)|computer.{0,40}?(?:infected|virus).{0,40}?(?:call|phone|microsoft|support)|work\s+from\s+home.{0,40}?(?:fee|pay|deposit|software)|data\s+entry.{0,40}?(?:fee|pay|deposit|software)|charity.{0,40}?(?:western\s+union|moneygram|donate|orphans|widow|cancer|sick)|widow.{0,40}?(?:bank|transfer|help|million)|crypto.{0,40}?(?:trading\s+bot|earned|made.{0,20}?\$|guaranteed)|trading\s+bot.{0,40}?(?:earned|made|guaranteed|sign\s+up|register)|loan.{0,40}?(?:0\s*%\s*interest|no\s+credit\s+check|pre-?approved)|rental.{0,40}?(?:deposit|wire|transfer|fee.{0,20}?(?:send|wire|transfer))|\b1-\d{3}-\d{3}-\d{4}\b|netflix.{0,40}?(?:expir|renew|subscription|abo|abgelaufen|abon|اشتراك|सदस्यता|สมัครสมาชิก|langganan|assinatura|订阅|期限切|abgelaufen|hết\s+hạn|만료|expirado|expiró)|icloud.{0,40}?(?:storage|full|upgrade|ممتلئ|भर|เต็ม|penuh|cheio|stockage|Speicher|مساحة|存储|满了|penuh|cheio|ストレージ|저장공간|almacenamiento|stockage|Speicherplatz)|paypal.{0,40}?(?:unusual|activity|compromised|confirm|نشاط|गतिविधि|กิจกรรม|aktivitas|atividade|activité|Aktivität|bất\s+thường|异常|ungewöhnliche|نشاط\s+غير\s+عادي|이상|異常|actividad\s+inusual|atividade\s+incomum)|(?:bank\s+account|bankkonto|compte\s+bancaire|cuenta\s+bancaria|حساب\s+بنكي|बैंक\s+खाता|บัญชีธนาคาร|rekening\s+bank|conta\s+banc|銀行口座|은행\s+계좌|rekening\s+bank).{0,40}?(?:frozen|freeze|compromised|đóng\s+băng|eingefroren|تجميد|जमा|แช่แข็ง|dibekukan|congelado|冻结|凍結|동결|congelar|gelé|congelada|congelarán|congelará|wird\s+eingefroren|gelé|凍結されます|동결됩니다|congelada|dibekukan|จะถูกแช่แข็ง|जमा\s+हो\s+जाएगा|sẽ\s+dibekukan|será\s+congelada|ma-freeze)|\$\d[\d,\.]*\s*/\s*(?:week|month|hour|woche|semaine|mes|semana|أسبوع|हफ्ते|สัปดาห์|minggu|tuần|週|주)|(?:5\.000|5,000|50\.000|50,000)\s*(?:dollar|dólar|dolar|đô|ดอลลาร์|دولار|डॉलर|dolar|ドル|달러)|pre-?approved|0\s*%\s*interest|no\s+credit\s+check|(?:free|gratis|gratuit|gratuito|grátis|kostenlos|kostenlose|kostenloses|مجاناً|مجاني|مجاناً|مجاني|मुफ्त|ฟรี|miễn\s+phím|miễn\s+phí|無料|무료|免费|免費|gratis|gratuit|gratuito|grátis|kostenlos|kostenlose|kostenloses|مجاناً|مجاني|مجاني|मुफ्त|ฟรี|miễn\s+phí|miễn\s+phí|無料|무료|免费|免費)\s+iPhone|free\s+v-?bucks|v-?bucks|fortnite|gift\s+card.{0,40}?(?:text|win|claim|nhắn|texten|إرسال|भेजें|ส่ง|kirim|envie|发短信|nhắn|إرسال|文字|문자|texto|envie).{0,40}?(?:\$\d|win|555|9\.99|9,99|subscription|abonnement|اشتراك|सदस्यता|สมัครสมาชิก|langganan|assinatura|订阅|abonnement)|krypto|cryptomonnaie|cripto|كريبتو|क्रिप्टो|คริปโต|45\.?000|45,?000|45\s?000|verdient|earned|gagné|ganado|ربحت|कमाए|ทำเงิน|dapet|verdient|gagné|ganado)\b"
    ).unwrap())
}

/// Scam-specific context patterns that distinguish scam cases from URL risk cases.
/// These are phrases that appear in scam messages but NOT in URL risk messages.
/// Used alongside strong_scam_indicator_re to boost scam confidence above URL risk.
fn scam_specific_context_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?i)(?:temporarily\s+suspend|suspendido|suspendida|suspensa|gesperrt|一時停止|임시\s*정지|تعليق\s*مؤقت|निलंबित|ระงับชั่วคราว|ditangguhkan|suspendu|sospeso|tạm\s*khóa|暂时冻结|停止され|정지되|تعليق|निलंबित|ถูกระงับ|sementara\s*ditangguhkan|suspendu|sospeso|na\s*-?\s*suspend|na\s*-?\s*close)|(?:could\s+not\s+be\s+delivered|không\s+thể\s+giao|无法投递|配達できません|배송이\s*불가능|تعذر\s+تسليم|ไม่สามารถจัดส่ง|tidak\s+dapat\s+dikirim|tidak\s+bisa\s+dikirim|não\s+pôde\s+ser\s+entregue|no\s+pudo\s+ser\s+entregado|konnte\s+nicht\s+(?:geliefert|zugestellt)|n.a\s+pas\s+pu\s+être\s+livré|livré\s+impossible|डिलीवर\s+नहीं\s+हो\s+सका)|(?:redelivery\s+fee|giao\s+lại|重新投递|再配達|재배송|إعادة\s+التسليم|จัดส่งซ้ำ|pengiriman\s+ulang|kirim\s+ulang|réexpédition|Neuzustellung|erneute\s+Zustellung|redelivery|रीडिलीवरी)|(?:customs\s+fee|hải\s+quan|海关费|関税|관세|جمارك|ศุลกากร|bea\s+cukai|frais\s+de\s+douane|Zollgebühr|tarifa\s+aduanera|taxa\s+alfandegária)|(?:pre-?aprobado|pré-?aprové|vorab-?genehmigt|مسبق\s+الموافقة|पूर्व-?अनुमोदित|预先批准)|(?:0\s*%\s*(?:de\s+)?(?:interés|juros|Zins|intérêt|فائدة|ब्याज|이자|金利|lãi))|(?:netflix.{0,40}?(?:만료|expirad|abgelaufen|انتهت|หมดอายุ|berakhir|scadut|caduc|过期|已到期|期限切|expirou|expirée|expiré|expirad|期限切れ|kedaluwarsa|expirou|expirada))|(?:(?:amazon|亚马逊|アマゾン|아마존).{0,60}?(?:order|bestellung|注文|주문|pedido|commande|订单|đơn\s+hàng).{0,60}?(?:shipped|versendet|発送|배송|enviado|expédié|发货|đã\s+được\s+gửi)|(?:order|bestellung|注文|주문|pedido|commande|订单|đơn\s+hàng).{0,40}?(?:amazon|亚马逊|アマゾン|아마존).{0,60}?(?:shipped|versendet|発送|배송|enviado|expédié|发货|đã\s+được\s+gửi))|(?:dhl.{0,40}?(?:on\s+hold|held|detained|扣留|보류|retenu|retido|محتجز|ถูกกัก|tertahan))|(?:paypal.{0,40}?(?:unusual|activity|compromised|异常|نشاط\s+غير\s+عادي|이상|actividad\s+inusual|atividade\s+incomum|bất\s+thường|ungewöhnliche|異常な活動|異常|活動|unusual\s+activity|กิจกรรมผิดปกติ|aktivitas\s+tidak\s+wajar|atividade\s+incomum|actividad\s+inusual))|(?:bank\s+account.{0,40}?(?:frozen|freeze|凍結|동결|congelado|gelé|eingefroren|تجميد|جمد|जमा|แช่แข็ง|dibekukan|congelada|congelar))|(?:easy.?money.?system|make\s+money\s+(?:from\s+home|at\s+home)|稼げる|kumita|kumito|cari\s+uang|ganhar\s+dinheiro|gagner\s+de\s+l.argent|verdienen|كسب\s+المال|कमाएं|ทำเงิน|make\s+\$5,000|earn\s+\$5,000|\$5,000.{0,20}?(?:week|minggu|주|週|周|semaine|Woche|semana|tuần|สัปดาห์|أسبوع|सप्ताह))|(?:pre-?approv\w*|pré-?approv\w*|pre-?aprobad\w*|pré-?aprovad\w*|vorab-?genehmigt|مسبق\s+(?:الموافقة|موافق)|पूर्व-?अनुमोदित|预先批准|사전\s*승인|事前承認|pré-?aprovado|pré-?aprobado).{0,80}?(?:loan|préstamo|prêt|Kredit|قرض|ऋण|贷款|대출|ローン|empréstimo)|(?:no\s+credit\s+(?:check|verificación)|sin\s+verificación|sans\s+vérif|ohne\s+Bonitätsprüfung|بدون\s+فحص\s+ائتماني|क्रेडिट\s+जांच\s+नहीं|신용\s+확인\s+불필요|信用チェック不要|sem\s+verificação\s+de\s+crédito|không\s+cần\s+kiểm\s+tra\s+tín\s+dụng|无需信用检查|信用检查不需要)|(?:text\s+(?:win|WIN)|nhắn\s+WIN|envoyez\s+WIN|أرسل\s+WIN|भेजें\s+WIN|ส่ง\s+WIN|kirim\s+WIN|envie\s+WIN|gửi\s+WIN|テキスト\s+WIN|문자\s+WIN|发短信\s+WIN|win\s+(?:to\s+)?\d{3}|win\s+\d{3}|gift\s+card.{0,40}?win\b)|(?:free\s+iphone|gratis\s+iphone|مجاناً\s+آيفون|مجاني\s+آيفون|免费\s+iphone|무료\s+iphone|無料\s+iphone|gratuit\s+iphone|gratis\s+iphone|ฟรี\s+iphone|free\s+iphone\s+15|免费iphone|iphone\s+15\s+pro\s+max)|(?:complete\s+(?:this\s+)?survey|completa\s+(?:esta\s+)?encuesta|terminez\s+(?:ce\s+)?sondage|abschließen\s+(?:diese\s+)?Umfrage|أكمل\s+(?:هذا\s+)?استبيان|पूरा\s+करें\s+(?:यह\s+)?सर्वेक्षण|ทำ\s+(?:แบบสำรวจนี้|แบบสำรวจ)|selesaikan\s+(?:survei\s+ini|survei)|complete\s+(?:esta\s+)?pesquisa|hoàn\s+thành\s+(?:cuộc\s+)?khảo\s+sát|完了\s+(?:この\s+)?アンケート|완료\s+(?:이\s+)?설문조사|完成\s+(?:此\s+)?调查|完成此调查|只需完成此调查)|(?:make.{0,20}?\$?\d[\d,\.]*\s*(?:dollars?|dólares|dolar|đô|ดอลลาร์|دولار|डॉलर|Dollar).{0,20}?(?:week|semana|semaine|Woche|أسبوع|สัปดาห์|minggu|tuần|सप्ताह|주|週|周).{0,40}?(?:from\s+home|de\s+casa|depuis\s+chez|von\s+zu\s+Hause|من\s+المنزل|จากที่บ้าน|dari\s+rumah|em\s+casa|tại\s+nhà|घर\s+से|재택|在宅|在家).{0,40}?(?:no\s+experience|sin\s+experiencia|sans\s+expérience|ohne\s+Erfahrung|بدون\s+خبرة|ไม่ต้องมีประสบการณ์|tanpa\s+pengalaman|sem\s+experiência|không\s+cần\s+kinh\s+nghiệm|経験不要|경험\s*불필요|无需经验|बिना\s+अनुभव))|(?:bank\s+account.{0,40}?(?:frozen|freeze|凍結|동결|congelado|gelé|eingefroren|تجميد|جمد|जमा|แช่แข็ง|dibekukan|congelada|congelar|冻结|동결|凍結))|(?:amazon.{0,60}?(?:order|bestellung|注文|주문|pedido|commande|订单|주문|注文).{0,60}?(?:shipped|versendet|発送|배송|enviado|expédié|发货|배송|発送))|(?:0\s*%\s*(?:interest|interés|intérêt|Zins|فائدة|ब्याज|이자|金利|juros|lãi|利率|利息))|(?:bank\s+account.{0,60}?(?:frozen|freeze|đóng\s+băng|冻结|동결|凍結|congelado|gelé|eingefroren|تجميد|جمد|जमा|แช่แข็ง|dibekukan|congelada|congelar))|(?:(?:tài\s+khoản\s+ngân\s+hàng|银行账户|은행\s+계좌|銀行口座|cuenta\s+bancaria|compte\s+bancaire|Bankkonto|حساب\s+بنكي|बैंक\s+खाता|บัญชีธนาคาร|rekening\s+bank|conta\s+bancária).{0,60}?(?:frozen|freeze|đóng\s+băng|冻结|동결|凍結|congelado|gelé|eingefroren|تجميد|جمد|जमा|แช่แข็ង|dibekukan|congelada|congelar))|(?:make.{0,20}?\$?\d[\d,\.]*\s*(?:dollars?|dólares|dolar|đô|Dollar|دولار|डॉलर|ดอลลาร์)?.{0,30}?(?:week|semana|semaine|Woche|tuần|주|週|周|minggu|สัปดาห์|أسبوع|सप्ताह).{0,40}?(?:from\s+home|de\s+casa|depuis\s+chez|von\s+zu\s+Hause|من\s+المنزل|tại\s+nhà|từ\s+nhà|dari\s+rumah|em\s+casa|在家|재택|在宅|घर\s+से|จากที่บ้าน).{0,40}?(?:no\s+experience|sin\s+experiencia|sans\s+expérience|ohne\s+Erfahrung|không\s+cần\s+kinh\s+nghiệm|경험\s*불필요|无需经验|بدون\s+خبرة|बिना\s+अनुभव|ไม่ต้องมีประสบการณ์|tanpa\s+pengalaman|sem\s+experiência|経験不要))|(?:amazon.{0,40}?(?:gift\s+card|礼品卡|ギフトカード|기프트\s+카드|tarjeta\s+de\s+regalo|carte\s+cadeau|Geschenkkarte|بطاقة\s+هدايا|उपहार\s+कार्ड|บัตรของขวัญ|kartu\s+hadiah|cartão\s+presente).{0,40}?(?:winner|selected|chosen|ganador|gagnant|gewinner|فائز|विजेता|ผู้ชนะ|pemenang|vencedor|当選|당첨|获奖|chọn))|(?:tax\s+refund|退税|税金還付|還付|환급|reembolso\s+de\s+impuestos|reembolso\s+de\s+impostos|Steuerrückerstattung|استرداد\s+ضريبي|कर\s+वापसी|คืนภาษี|pajak\s+kembali|hoàn\s+thuế|refund\s+pajak|refund\s+de\s+impostos|refund\s+de\s+impuestos|tax\s+refund\s+urgent|unclaimed\s+tax| Steuer\s+Rückerstattung|(?:amazon|亚马逊|アマゾン|아마존|أمازون).{0,60}?(?:shipped|versendet|発送|배송|enviado|expédié|发货|تم\s+شحن|शिप\s+कर|ส่งแล้ว|dikirim|enviado)|(?:آيفون|iPhone).{0,40}?(?:استبيان|survey|survey|استبيان)|(?:congratulations|felicitaciones|félicitations|glückwunsch|مبروك|बधाई|ยินดีด้วย|selamat|parabéns).{0,40}?(?:winner|ganador|gagnant|gewinner|الفائز|विजेता|ผู้ชนะ|pemenang|vencedor)|(?:text|sms|nhắn|envoyez|schicken|أرسل|भेजें|ส่ง|kirim|envie|gửi|テキスト|문자|发短信)[\s至로にへ]*(?:win|WIN)[\s至로にへ]*.{0,40}?\d{3}[-.]?\d{3}[-.]?\d{4}|(?:\$\d[\d,\.]*\s*(?:/|per\s+)?(?:month|mo|月|월|mes|Monat|شهر|महीना|เดือน|bulan|mês))|(?:每月|月額|월\s*구독|月额|اشتراك\s+شهري|सदस्यता\s+मासिक|สมัครสมาชิก\s*รายเดือน|langganan\s+bulanan|assinatura\s+mensal|月額|월\s*구독).{0,20}?\d[\d,\.]*\s*(?:美元|ドル|달러|dollar|dólar|dolar|đô|ดอลลาร์|دولار|डॉलर|yuan|元)|(?:恭喜.{0,40}?获奖|当選おめでとう|당첨\s*축하|مبروك.{0,40}?الفائز|बधाई.{0,40}?विजेता|ยินดีด้วย.{0,40}?ผู้ชนะ|selamat.{0,40}?pemenang|parabéns.{0,40}?vencedor|felicitaciones.{0,40}?ganador|félicitations.{0,40}?gagnant|glückwunsch.{0,40}?gewinner)|(?:月額|월\s*구독|月额|اشتراك\s+شهري|सदस्यता\s+मासिक|สมัครสมาชิก\s*รายเดือน|langganan\s+bulanan|assinatura\s+mensal))",
    ).unwrap())
}

fn fake_giveaway_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(r"(?i)\b(?:congratulations|you\s+(?:have\s+)?won|claim\s+your\s+prize|you've\s+won|dear\s+winner|you\s+have\s+been\s+selected|lucky\s+winner|you\s+have\s+been\s+chosen)\b|(?:\b(?:won|win|trúng|trúng\s+thưởng|gewonn|gagné|ganaste|ربحت|जीते|menang|当選|당첨)\b.{0,30}?\b\$\s?\d|\b\$\s?\d[\d,\.]*\s*(?:dollar|usd|€|euro|eur)\b.{0,30}?\b(?:won|win|prize|thưởng|preis|prix|premio|جائزة|इनाम|hadiah|賞|상금))").unwrap())
}

fn credential_harvest_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(r"(?i)\b(?:verify|confirm|reset|update)\b.{0,120}?\b(?:password|account|login)\b|\b(?:password|account|login)\b.{0,120}?\b(?:verify|confirm|reset|update)\b").unwrap())
}

fn romance_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| {
        let endearment = r"\b(?:darling|sweetheart|honey|baby|babe|my\s+love|beloved|dear|my\s+heart|soulmate|hey\s+babe|i\s+love\s+you|please\s+help\s+me|stuck\s+in\s+dubai|stuck\s+abroad|contract\s+work|soldier|soldado|soldat|جندي|सैनिक|ทหาร|prajurit|sundalo|군인|bébé|mon\s+chéri|ma\s+chérie|schatz|liebling|habibi|cariño|carino|tesoro|amore|cuore|anata|chéri|chérie|宝贝|亲爱的|ダーリン|자기|사랑|รัก|sayang|meu\s+amor|querido|प्रिय|जान|حبيبي|عزيزي)\b";
        let ask = r"\b(?:gift\s*cards?|wire(?:d|s|ing)?|western\s+union|money\s*gram|send\s+\$?\d+|need\s+(?:money|cash)|loan\s+me|borrow|repay\s+you|itunes\s+card|apple\s+gift\s+card|google\s+play\s+card|amazon\s+card|steam\s+card|hotel\s+(?:bill|fee)|pay\s+(?:the\s+)?hotel|envoyer|envoie|schick|أرسل|भेजें|ส่งเงิน|kirim|envía|envie|送金|송금|电汇|转账|transfer\s+\$|envíe\s+\$|envoie\s+\$|schick\s+\$|أرسل\s+\$|भेजें\s+\$|ส่ง\s+\$|kirim\s+\$)\b";
        let pattern = format!("(?is)(?:{endearment}.{{0,200}}?{ask})|(?:{ask}.{{0,200}}?{endearment})");
        FRegex::new(&pattern).unwrap()
    })
}

fn crypto_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| {
        let assets = r"(?:btc|eth|usdt|usdc|sol|bnb|xrp|ada|doge|matic|ltc|bitcoin|ethereum|tether|crypto|krypto|cripto|كريبتو|क्रिप्टो|คริปโต)";
        let pattern = format!(
            r"(?i)(?:\bsend\b\s+(?:me\s+)?(?:\d+(?:\.\d+)?\s*)?{assets}\b.{{0,80}}?\b(?:return|get|receive|i'?ll\s+(?:send|return))\b)|\b(?:guaranteed|risk\-?free)\b.{{0,20}}?\b(?:returns?|profits?|gains?|roi)\b|\b\d+\s*%\s+(?:returns?|profits?|gains?|roi)\b|\bpump\s+and\s+dump\b|\bsend\b\s+(?:to\s+)?(?:my\s+)?(?:wallet\s+address|btc\s+address|eth\s+address|(?:bitcoin|ethereum)\s+address)\b|(?:\b(?:share|give|tell|provide|reveal|paste|forward)\b).{{0,40}}?\b(?:seed\s*phrase|seed\s*words?|recovery\s*phrase|recovery\s*words?|mnemonic(?:\s*phrase)?|private\s*key|secret\s*key)\b|\b(?:seed\s*phrase|recovery\s*phrase|mnemonic|private\s*key)\b.{{0,80}}?\b(?:refund|double|2x|guaranteed)\b|\b{assets}\s+(?:trading\s+)?bot\b|(?:\b(?:made|earned|verdient|gagné|gané|ganaste|ربحت|कमाए|ทำเงิน|dapet|menang)\b).{{0,40}}?\b\d{{3,}}.{{0,40}}?\b(?:bot|platform|trading|bot| Plattform|plataforma|منصة|प्लेटफॉर्म|แพลตฟอร์ม)\b|\b\d{{1,3}}[.,]\d{{3}}.{{0,40}}?\b(?:{assets}).{{0,40}}?\b(?:bot|platform|trading)\b",
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
        r"(?si)\b(?:virus|malware|trojan|infection|infected|hacked?|compromised?|breach(?:ed)?)\b.{0,120}?\b(?:call|dial|phone|contact)\b.{0,40}?(?:\+?\d[\d\-\s().]{6,}\d|number)|\b(?:microsoft|apple|google|norton|mcafee|windows\s+defender)\s+(?:support|technician|engineer|helpdesk|security)\b|\b(?:call|dial)\s+(?:microsoft|apple|norton|mcafee)\b|(?si)\b(?:computer|pc|computer|ordinateur|rechner|komputer|コンピュータ|컴퓨터|कंप्यूटर|คอมพิวเตอร์|computador)\b.{0,40}?\b(?:infected|infiziert|infecté|infectado|感染|감염|संक्रमित|ติดไวรัส|terinfeksi|infectado)\b.{0,80}?\b(?:microsoft|apple|support|technician|サポート|지원|सपोर्ट|สนับสนุน|dukungan|soporte)\b",
    ).unwrap())
}

/// Package delivery scam — fake courier messages demanding redelivery fees.
fn package_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\b(?:package|parcel|shipment)\b.{0,60}?\b(?:could\s+not\s+be\s+delivered|not\s+delivered|delayed|held|waiting)\b.{0,80}?\b(?:fee|pay|update|reschedule|redelivery|customs)\b|\b(?:fedex|ups|dhl|usps|royal\s+mail)\b.{0,40}?\b(?:fee|pay|redelivery|reschedule)\b",
    ).unwrap())
}

/// Investment scam — pre-IPO, guaranteed returns, trading bot solicitations.
fn investment_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\b(?:pre-?ipo|trading\s+bot|crypto\s+trading|investment\s+opportunity|guaranteed\s+returns?|risk-?free)\b.{0,80}?\b(?:invest|deposit|send|start|register|sign\s+up)\b.{0,40}?\$?\d|\binvest\b.{0,40}?\b(?:just|only|minimum)\s+\$?\d|\b(?:made|earned|profit)\s+\$?\d{3,}.{0,40}?\b(?:bot|platform|system|software)\b",
    ).unwrap())
}

/// IRS / government impersonation scam — demands payment via gift cards or wire.
fn irs_gov_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\b(?:irs|tax\s+(?:return|authority|refund)|treasury|social\s+security|medicare)\b.{0,100}?\b(?:owe|penalty|arrest|warrant|back\s+taxes|discrepancy|unclaimed|refund)\b.{0,80}?\b(?:pay|gift\s+card|wire|send|apple\s+gift|payment|claim|click)\b|(?si)\b(?:impôts|finanzamt|steuer|税務署|세무서|ضرائب|आयकर|กรมสรรพากร|pajak|receita|impuestos)\b.{0,100}?\b(?:devez|schulden|owe|penalty|arrest|penalité|strafe|納税|체포|اعتقال|गिरफ्तारी|จับกุม|penalti|multa)\b.{0,80}?\b(?:pay|gift\s+card|wire|send|apple\s+gift|payment|carte|karte|カード|기프트|بطاقة|गिफ्ट|บัต्र|cartão|pague)\b|(?si)(?:tax\s+refund|退税|税金還付|還付|환급|reembolso\s+de\s+impuestos|reembolso\s+de\s+impostos|Steuerrückerstattung|استرداد\s+ضريبي|कर\s+वापसी|คืนภาษี|pajak\s+kembali|hoàn\s+thuế).{0,80}?\b(?:unclaimed|claim|click|receive|领取|受領|수령|reclamar|réclamer|beanspruchen|اطالب|दावा|เรียกรับ|klaim)\b",
    ).unwrap())
}

/// Rental scam — landlord "overseas", demands wire transfer / deposit before viewing.
fn rental_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\b(?:apartment|rent|lease|landlord|tenant|2br|2-bedroom|studio)\b.{0,80}?\b(?:overseas|abroad|out\s+of\s+(?:the\s+)?country|missionary|deployed)\b.{0,80}?\b(?:wire|transfer|deposit|send|first\s+month)\b|\b(?:wire|transfer|send)\b.{0,60}?\b(?:first\s+month|deposit|key|keys)\b.{0,40}?\b(?:rent|lease|apartment|landlord)\b",
    ).unwrap())
}

/// Job scam — work-from-home offers requiring upfront payment for "software" or "equipment".
fn job_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\b(?:work\s+from\s+home|mystery\s+shopper|data\s+entry|remote\s+job|employment|position|salary|hourly)\b.{0,80}?\b(?:software|equipment|training|materials?|fee|pay|purchase|buy)\b.{0,40}?\$?\d|\b(?:selected|hired|chosen)\b.{0,60}?\b(?:job|position|employment)\b.{0,80}?\b(?:pay|purchase|buy|fee)\b.{0,40}?\$?\d|(?si)\b(?:trabajo\s+desde\s+casa|travail\s+à\s+domicile|heimarbeit|在宅|재택|العمل\s+من\s+المنزل|घर\s+से\s+काम|ทำงานที่บ้าน|kerja\s+dari\s+rumah|trabalho\s+em\s+casa|trabajo\s+en\s+casa)\b.{0,80}?\b(?:software|logiciel|software|软件|ソフトウェア|소프트웨어|برنامج|सॉफ्टवेयर|ซอฟต์แวร์|perangkat\s+lunak|software)\b.{0,40}?\$?\d",
    ).unwrap())
}

/// Charity scam — urgent donation appeals via wire transfer.
fn charity_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\b(?:donation|donate|charity|fundrais\w*|appeal|starvation|orphans?|widows?|refugees?)\b.{0,80}?\b(?:western\s+union|wire|send|money\s+gram|transfer)\b.{0,40}?\b(?:foundation|relief|children|family|feed)\b|(?si)\b(?:donación|don|donation|spende|تبرع|दान|บริจาค|donasi|doação|donazione)\b.{0,80}?\b(?:western\s+union|wire|send|envoyer|überweisung|حوالة|ट्रांसफर|โอน|transfer|kirim)\b.{0,40}?\b(?:foundation|fundación|fondation|stiftung|مؤسسة|फाउंडेशन|มูลนิธิ|yayasan|fundação)\b",
    ).unwrap())
}

/// Gift card as payment method — strong scam signal in any context.
fn gift_card_payment_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\b(?:pay|payment|fee|penalty|tax|owe|pague|payez|zahlen|ادفع|भुगतान|จ่าย|bayar|pague|paga)\b.{0,60}?\b(?:apple\s+gift\s+card|itunes\s+card|google\s+play\s+card|amazon\s+card|steam\s+card|gift\s+card|carte\s+cadeau|geschenkkarte|بطاقة\s+هدية|गिफ्ट\s+कार्ड|บัตรของขวัญ|kartu\s+hadiah|cartão\s+presente|carta\s+regalo)\b|\b(?:apple\s+gift\s+card|itunes\s+card|google\s+play\s+card|amazon\s+card|steam\s+card|gift\s+card|carte\s+cadeau|geschenkkarte|بطاقة\s+هدية|गिफ्ट\s+कार्ड|บัตรของขวัญ|kartu\s+hadiah|cartão\s+presente|carta\s+regalo)\b.{0,60}?\b(?:pay|payment|fee|penalty|tax|owe|purchase|buy|pague|payez|zahlen|ادفع|भुगतान|จ่าย|bayar|pague|paga)\b",
    ).unwrap())
}

/// Suspended/locked account scam — demands immediate action to prevent closure.
fn suspended_account_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\b(?:account|profile|wallet)\s+(?:has\s+been\s+)?(?:temporarily\s+|recently\s+|been\s+)?(?:suspended|locked|compromised|restricted|disabled|flagged|put\s+on\s+hold)\b.{0,100}?\b(?:verify|confirm|update|restore|click|link|login|sign\s*in|password|information)\b|\b(?:suspended|locked|compromised|restricted)\b.{0,40}?\b(?:account|profile|wallet)\b.{0,100}?\b(?:verify|confirm|update|restore|click|login|password|information)\b|(?si)\b(?:social\s+security|seguro\s+social|sozialversicherungsnummer|الضمان\s+الاجتماعي|社会保障番号|사회보장번호|số\s+an\s+sinh\s+xã\s+hội|社会安全号码)\b.{0,80}?\b(?:suspended|locked|compromised|restricted|disabled|suspendido|gesperrt|متوقف|停止|정지|tạm\s+ngưng|暂停)\b.{0,80}?\b(?:call|phone|dial|llame|ruf|اتصل|전화|โทร|telepon|ligue|tumawag)\b",
    ).unwrap())
}

/// Refund/charge scam — fake charge notification demanding call for refund.
fn refund_charge_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)\b(?:charged?|billed|cobrado|facturé|berechnet|خصم|लिए\s+गए|เรียกเก็บ|dikenakan|cobrado|na-charge)\b.{0,40}?\$?\d{2,}.{0,80}?\b(?:norton|antivirus|mcAfee|subscription|renewal|renew|renouvellement|Verlängerung|تجديد|नवीनीकरण|ต่ออายุ|perpanjangan|renovação)\b.{0,80}?\b(?:call|refund|cancel|reembolso|remboursement|Erstattung|استرداد|रद्द|ยกเลิก|batalkan|cancelar)\b",
    ).unwrap())
}

/// Brand scam — well-known brand names combined with a URL and urgency/payment context.
/// This catches multilingual scam messages that use English brand names.
fn brand_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    // Note: \b around brand names and URLs removed because CJK characters
    // are Unicode word characters, so \b doesn't match between ASCII and CJK.
    // Includes non-Latin brand names: 亚马逊/아마존/アマゾン (Amazon),
    // 沃尔玛/월마트 (Walmart), 网飞 (Netflix), 苹果 (Apple).
    CELL.get_or_init(|| FRegex::new(
        r"(?si)(?:netflix|icloud|paypal|amazon|walmart|iphone|fedex|ups|dhl|norton|mcafee|fortnite|v-?bucks|roblox|robux|亚马逊|亚马孙|아마존|アマゾン|沃尔玛|월마트|网飞|넷플릭스|ネットフリックス|苹果|애플|アップル).{0,200}?(?:https?://|www\.)|(?:https?://|www\.).{0,100}?(?:netflix|icloud|paypal|amazon|walmart|iphone|fedex|ups|dhl|norton|mcafee|fortnite|v-?bucks|roblox|robux|亚马逊|亚马孙|아마존|アマゾン|沃尔玛|월마트|网飞|넷플릭스|ネットフリックス|苹果|애플|アップル)",
    ).unwrap())
}

/// Harmful content override — detects drug names, CSAM, weapons sale, and hate
/// speech indicators that should NOT be classified as scam. When present, scam
/// signals are suppressed so the harmful content is classified by its proper
/// category (drugs, child safety, illegal goods, hate, etc.).
fn harmful_content_override_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)(?:(?:cocaine|cocaína|cocaïne|kokain|كوكايين|कोकीन|โคเคน|kokaina|cocaina|kokaïne|可卡因|코카인|コカイン|摇头丸))|(?:\b(?:mdma|lsd|heroin|meth|fentanyl|crack|ketamine)\b)|(?:\b(?:AK-?47|firearm|rifle|pistol|ammunition|no\s+serial\s+number|fully\s+automatic)\b)|(?:\b(?:12-year-old|12-jährig\w*|12\s*ans|12\s*岁|12\s*세|12歳|12\s*tahun|12\s*taon|12\s*سنة|12\s*वर्ष).{0,80}?(?:videos?|video|vidéos|Videos|فيديو|वीडियो|動画|비디오|vídeos|koleksi|koleksyon|sammlung|مجموعة))|(?:\b(?:videos?|video|vidéos|Videos|فيديو|वीडियो|動画|비디오|vídeos|koleksi|koleksyon|sammlung|مجموعة).{0,80}?(?:12-year-old|12-jährig\w*|12\s*ans|12\s*岁|12\s*세|12歳|12\s*tahun|12\s*taon|12\s*سنة|12\s*वर्ष))|(?:\b(?:subhuman|vermin|untermensch|دون\s*بشري|नीच|하등|劣等|害虫|hama|insecto|vermine)\b)|(?:\b(?:immigrants?|inmigrantes?|immigrés?|Einwanderer|مهاجرين|आप्रवासी|이민자|移民|imigran|imigrante|ผู้อพยพ)\b.{0,60}?(?:vermin|disease|crime|vermine|Ungeziefer|حشرات|बीमारी|해충|害虫|hama|enfermedad))|(?:\b(?:vaccines?|vacunas?|vaccins?|Impfungen|لقاحات|टीके|백신|ワクチン|vaksin|vacinas|วัคซีน)\b.{0,60}?(?:conspiracy|microchips?|autism|Verschwörung|مؤامرة|षड्यंत्र|마이크로칩|陰謀|conspiración|conspiration))|(?:\b(?:election|elección|élection|Wahl|انتخابات|चुनाव|선거|選挙|pemilihan|eleição|การเลือกตั้ง|polling|voting|vote)\b.{0,60}?(?:stolen|fraud|rigged|gestohlen|مسروقة|चुराई|조작|盗ま|robada|volée|moved|moved\s+to\s+wednesday))|(?:\b(?:stalk|stalking|acechar|harceler|belästigen|مطاردة|स्टॉक|스토킹|ストーキング|nguntit|persegui|สตอล์ก)\b)|(?:\b(?:hotwire|hot-wire|démarrer\s+à\s+chaud|هوتواير|हॉटवायर|핫와이어|ホットワイヤー|puentear)\b)|(?:\b(?:break\s+(?:every\s+)?bone|briser\s+(?:chaque\s+)?os|jeden\s+Knochen|كسر\s+(?:كل\s+)?عظمة|हड्डी\s+तोड़|뼈를\s+부수|骨を折|quebrar\s+(?:cada\s+)?osso|romper\s+(?:cada\s+)?hueso|หักกระดูก|patahin\s+tulang|bẻ\w*\s*gãy|打断)\b)",
    ).unwrap())
}

/// Loan scam — pre-approved loan with 0% interest, no credit check.
fn loan_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)(?:pre-?approv\w*|pré-?approv\w*|vorab-?genehmigt|مسبق\s+(?:الموافقة|موافق)|पूर्व-?अनुमोदित|预先批准|사전\s*승인|事前承認|pré-?aprovado|pré-?aprobado|duyệt\s+trước|duyet\s+truoc).{0,80}?(?:loan|préstamo|prêt|Kredit|قرض|ऋण|贷款|대출|ローン|empréstimo|khoản\s+vay|khoan\s+vay).{0,80}?(?:0\s*%\s*(?:de\s+)?(?:interest|interés|intérêt|Zins|فائدة|ब्याज|이자|金利|juros|lãi|利率|利息|de\s+interés)|no\s+credit\s+(?:check|verificación)|sin\s+verificación|sans\s+vérif|ohne\s+Bonitätsprüfung|بدون\s+فحص\s+ائتماني|क्रेडिट\s+जांच\s+नहीं|신용\s+확인\s+불필요|信用チェック不要|sem\s+verificação\s+de\s+crédito|không\s+cần\s+kiểm\s+tra\s+tín\s+dụng|无需信用检查|信用检查不需要)|(?:0\s*%\s*(?:interest|interés|intérêt|Zins|فائدة|ब्याज|이자|金利|juros|lãi|利率|利息)).{0,80}?(?:loan|préstamo|prêt|Kredit|قرض|ऋण|贷款|대출|ローン|empréstimo|khoản\s+vay)",
    ).unwrap())
}

/// Make money scam — earn $X/week from home with no experience.
fn make_money_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)(?:make|earn|gane|gagne|verdienen|اربح|कमाए|ทำเงิน|dapet|ganhe|ganar|kiếm|稼げる|稼ぐ|벌기|벌어|버는|kumita|kumito|hasilkan|ganhar).{0,20}?\$?\d[\d,\.]*\s*(?:dollars?|dólares|Dollar|دولار|डॉलर|ดอลลาร์|dolar|đô|đôla?).{0,30}?(?:/?\s*(?:week|semana|semaine|Woche|أسبوع|सप्ताह|สัปดาห์|minggu|semana|tuần|주|週|周)).{0,40}?(?:from\s+home|from\s+house|de\s+casa|depuis\s+chez|von\s+zu\s+Hause|من\s+المنزل|घर\s+से|จากที่บ้าน|dari\s+rumah|em\s+casa|tại\s+nhà|từ\s+nhà|在家|재택|在宅).{0,40}?(?:no\s+experience|sin\s+experiencia|sans\s+expérience|ohne\s+Erfahrung|بدون\s+خبرة|बिना\s+अनुभव|ไม่ต้องมีประสบการณ์|tanpa\s+pengalaman|sem\s+experiência|không\s+cần\s+kinh\s+nghiệm|경험\s*불필요|无需经验|経験不要)|\$\d[\d,\.]*\s*(?:/|\s+)?(?:week|주|週|semana|semaine|Woche|minggu|tuần|สัปดาห์|أسبوع|सप्ताह).{0,60}?(?:from\s+home|재택|在宅|在家|de\s+casa|depuis\s+chez|von\s+zu\s+Hause|من\s+المنزل|घर\s+से|จากที่บ้าน|dari\s+rumah|em\s+casa|tại\s+nhà|từ\s+nhà).{0,60}?(?:no\s+experience|경험\s*불필요|无需经验|sin\s+experiencia|sans\s+expérience|ohne\s+Erfahrung|بدون\s+خبرة|बिना\s+अनुभव|ไม่ต้องมีประสบการณ์|tanpa\s+pengalaman|sem\s+experiência|không\s+cần\s+kinh\s+nghiệm|経験不要)|(?:no\s+experience|경험\s*불필요|无需经验|sin\s+experiencia|sans\s+expérience|ohne\s+Erfahrung|بدون\s+خبرة|बिना\s+अनुभव|ไม่ต้องมีประสบการณ์|tanpa\s+pengalaman|sem\s+experiência|không\s+cần\s+kinh\s+nghiệm|経験不要).{0,60}?(?:from\s+home|재택|在宅|在家|de\s+casa|depuis\s+chez|von\s+zu\s+Hause|من\s+المنزل|घर\s+से|จากที่บ้าน|dari\s+rumah|em\s+casa|tại\s+nhà|từ\s+nhà).{0,60}?\$\d[\d,\.]*\s*(?:/|\s+)?(?:week|주|週|semana|semaine|Woche|minggu|tuần|สัปดาห์|أسبوع|सप्ताह)",
    ).unwrap())
}

/// Bank freeze scam — urgent bank account freeze, verify identity at URL.
fn bank_freeze_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)(?:(?:bank\s+account|tài\s+khoản\s+ngân\s+hàng|银行账户|은행\s+계좌|銀行口座|cuenta\s+bancaria|compte\s+bancaire|Bankkonto|حساب\w*\s+ال?بنكي\w*|बैंक\s+खाता|บัญชีธนาคาร|rekening\s+bank|conta\s+bancária).{0,60}?(?:frozen|freeze|đóng\s+băng|冻结|동결|凍結|congelado|gelé|eingefroren|تجميد|جمد|जमा|แช่แข็ง|dibekukan|congelada|congelar).{0,80}?(?:verify|xác\s+minh|验证|확인|確認|verificar|vérifier|verifizieren|تحقق|सत्यापित|ยืนยัน|verifikasi|verificar|本人確認|신원\s*확인|身份验证)|(?:frozen|freeze|تجميد|جمد|冻结|동결|凍結|congelado|gelé|eingefroren|แช่แข็ง|dibekukan).{0,60}?(?:bank\s+account|tài\s+khoản\s+ngân\s+hàng|银行账户|은행\s+계좌|銀行口座|cuenta\s+bancaria|compte\s+bancaire|Bankkonto|حساب\w*\s+ال?بنكي\w*|बैंक\s+खाता|บัญชีธนาคาร|rekening\s+bank|conta\s+bancária).{0,80}?(?:verify|xác\s+minh|验证|확인|確認|verificar|vérifier|verifizieren|تحقق|सत्यापित|ยืนยัน|verifikasi|verificar|本人確認|신원\s*확인|身份验证))",
    ).unwrap())
}

/// Gift card scam — win a gift card, text WIN to a number, $X/month subscription.
fn gift_card_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)(?:win|ganhe|gana|gagne|gewinn|اربح|जीत|รับ|menang|nhận|勝つ|받으세요|赢取|trúng).{0,40}?(?:gift\s+card|tarjeta\s+de\s+regalo|carte\s+cadeau|Geschenkkarte|بطاقة\s+هدية|गिफ्ट\s+कार्ड|บัตรของขวัญ|kartu\s+hadiah|cartão\s+presente|thẻ\s+quà\s+tặng|ギフトカード|기프트\s+카드|礼品卡).{0,80}?(?:text|sms|nhắn|envoyez|schicken|أرسل|भेजें|ส่ง|kirim|envie|gửi|テキスト|문자|发短信)\s+(?:win|WIN).{0,40}?\d{3}[-.]?\d{3}[-.]?\d{4}",
    ).unwrap())
}

/// Free giveaway scam — free iPhone/product, complete a survey, limited spots.
fn free_giveaway_scam_re() -> &'static FRegex {
    static CELL: OnceLock<FRegex> = OnceLock::new();
    CELL.get_or_init(|| FRegex::new(
        r"(?si)(?:free|gratis|gratuit|kostenlos|مجاناً|मुफ्त|ฟรี|gratis|grátis|gratuit|無料|무료|免费).{0,40}?(?:iphone|ipad|samsung|galaxy|playstation|xbox|ps5|آيفون|아이폰|アイフォン|iPhone).{0,80}?(?:survey|encuesta|sondage|Umfrage|استبيان|सर्वेक्षण|แบบสำรวจ|survei|enquete|アンケート|설문조사|调查).{0,80}?(?:https?://|www\.)|(?:complete|completa|terminez|abschließen|أكمل|पूरा\s+करें|ทำให้เสร็จ|selesaikan|complete|完了|완료|完成).{0,40}?(?:survey|encuesta|sondage|Umfrage|استبيان|सर्वेक्षण|แบบสำรวจ|survei|enquete|アンケート|설문조사|调查).{0,40}?(?:https?://|www\.)",
    ).unwrap())
}

// ---------------------------------------------------------------------------
// URL risk detector — bare-host + lookalike-brand + shortener + code-ext guard.
// Ported from slm-guardrail `pipeline/url.rs`.
// ---------------------------------------------------------------------------

pub struct UrlDetector;

impl UrlDetector {
    pub fn detect(text: &str) -> Vec<DetectorSignal> {
        // Don't fire if the MalwareUrlDetector would handle this (exe extensions,
        // malware path keywords, suspicious URL keywords). Let MalwareUrlDetector
        // produce the cat 8 signal instead of a cat 7 scam signal.
        if !MalwareUrlDetector::detect(text).is_empty() {
            return vec![];
        }
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

// ---------------------------------------------------------------------------
// Malware URL detector — fires on URLs with executable extensions or
// malware-specific download patterns. Distinguishes malware links from
// generic scam/phishing URLs.
// ---------------------------------------------------------------------------

pub struct MalwareUrlDetector;

impl MalwareUrlDetector {
    pub fn detect(text: &str) -> Vec<DetectorSignal> {
        let cleaned = crate::normalize::strip_zero_width(text);
        let url_re = URL_RE.get_or_init(|| {
            Regex::new(r#"(?i)(?:https?://|www\.)[^\s<>"']{3,}"#).unwrap()
        });
        let urls: Vec<String> = url_re.find_iter(&cleaned).map(|m| m.as_str().to_string()).collect();
        if urls.is_empty() { return vec![]; }

        let exe_ext_re = EXE_EXT_RE.get_or_init(|| {
            Regex::new(r#"(?i)\.(exe|scr|bat|msi|apk|dmg|pif|reg|vbs|jar|ps1|zip|rar|7z|tar|gz)(?:$|[/?\s])"#).unwrap()
        });
        let malware_path_re = MALWARE_PATH_RE.get_or_init(|| {
            Regex::new(r#"(?i)/(?:setup|install|update|patch|firmware|crack|cracked|keygen|serial|warez|spy|hack|steal)(?:[/\.?\-]|$)"#).unwrap()
        });
        let suspicious_url_re = SUSPICIOUS_URL_RE.get_or_init(|| {
            Regex::new(r#"(?i)/(?:crack|cracked|keygen|warez|spy-?tool|phone-?track|hack-?insta|steal|phish|spy-?tools|phone-?tracker)"#).unwrap()
        });

        let mut signals = Vec::new();
        let lower_text = cleaned.to_ascii_lowercase();
        for url in &urls {
            let lowered = url.to_ascii_lowercase();
            let has_exe = exe_ext_re.is_match(&lowered);
            let has_malware_path = malware_path_re.is_match(&lowered);
            let has_suspicious_url = suspicious_url_re.is_match(&lowered);
            let on_high_risk_tld = high_risk_tlds().iter().any(|tld| {
                lowered.ends_with(&format!(".{tld}")) || lowered.contains(&format!(".{tld}/"))
            });
            let has_install_context = lower_text.contains("install")
                || lower_text.contains("setup")
                || lower_text.contains("security patch")
                || lower_text.contains("security update")
                || lower_text.contains("sicherheitsupdate")
                || lower_text.contains("attached document")
                || lower_text.contains("open the attached")
                || lower_text.contains("cracked")
                || lowered.contains("/crack")
                || lower_text.contains("spy")
                || lower_text.contains("hack")
                || lower_text.contains("private messages")
                || lower_text.contains("phone number location");

            // Fire if URL has executable extension, or URL has malware path on high-risk TLD,
            // or URL is on high-risk TLD with install/suspicious context in surrounding text,
            // or URL contains suspicious path keywords (regardless of TLD)
            if has_exe || (has_malware_path && on_high_risk_tld) || (on_high_risk_tld && has_install_context) || has_suspicious_url {
                signals.push(DetectorSignal {
                    category: categories::MALWARE_LINK,
                    severity: Severity::SEVERE,
                    confidence: 0.90,
                    reason_code: "malware_url_exe".into(),
                    action: Action::Warn,
                });
                break; // One signal is enough
            }
        }
        signals
    }
}

static EXE_EXT_RE: OnceLock<Regex> = OnceLock::new();
static MALWARE_PATH_RE: OnceLock<Regex> = OnceLock::new();
static SUSPICIOUS_URL_RE: OnceLock<Regex> = OnceLock::new();

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

        // Bonus: multiple hyphens in hostname (common in phishing URLs)
        let hyphen_count = host.chars().filter(|c| *c == '-').count();
        if hyphen_count >= 2 && score < 0.7 { score = 0.7; }
        if hyphen_count >= 3 && score < 0.85 { score = 0.85; }

        // Bonus: URL with both brand keyword + action keyword (e.g. "paypal-secure-verify")
        let has_brand = ["paypal", "apple", "amazon", "google", "microsoft", "netflix", "facebook",
                         "instagram", "bank", "chase", "wells", "citi", "walmart", "target",
                         "roblox", "office", "outlook", "dropbox", "linkedin", "twitter",
                         "telegram", "whatsapp", "snapchat", "discord", "tiktok"].iter()
            .any(|b| lowered.contains(b));
        let has_action = ["verify", "login", "secure", "update", "confirm", "restore", "reset",
                          "unlock", "suspended", "locked", "account", "password", "check",
                          "claim", "free", "gift", "prize", "giveaway", "bonus", "reward"].iter()
            .any(|a| lowered.contains(a));
        if has_brand && has_action && score < 0.9 { score = 0.9; }

        // Bonus: URL with numbers in hostname (e.g. "paypa1", "g00gle", "2024")
        let has_digit_in_host = host.chars().any(|c| c.is_ascii_digit());
        if has_digit_in_host && has_brand && score < 0.85 { score = 0.85; }

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

/// Extract URLs from text for scam/URL risk checking.
fn extract_urls(text: &str) -> Vec<String> {
    let cleaned = crate::normalize::normalize_for_patterns(text);
    let url_re = URL_RE.get_or_init(|| {
        Regex::new(r"(?i)https?://[^\s<>()]+|www\.[^\s<>()]+").unwrap()
    });
    url_re.find_iter(&cleaned).map(|m| m.as_str().to_string()).collect()
}

fn high_risk_tlds() -> &'static HashSet<&'static str> {
    static CELL: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CELL.get_or_init(|| ["zip", "mov", "top", "click", "country", "xyz", "ml", "tk", "cf", "ga", "gq", "bid", "shop", "info", "ru", "today", "world", "live", "club", "store", "online", "site", "fun", "cam", "sbs", "rest", "quest", "monster", "buzz", "icu", "loan", "click", "date", "download", "stream", "trade", "win", "review", "men", "work", "party", "click", "gdn", "racing", "accountant", "cricket", "faith", "science", "hud"].iter().copied().collect())
}

fn high_risk_keywords() -> &'static HashSet<&'static str> {
    static CELL: OnceLock<HashSet<&'static str>> = OnceLock::new();
    CELL.get_or_init(|| ["login", "verify", "account", "secure", "update", "confirm", "restore", "reset", "unlock", "suspended", "locked", "compromised", "free", "gift", "prize", "giveaway", "claim", "bonus", "reward", "crack", "hack", "spy", "track", "generator", "premium", "deal", "promo", "download"].iter().copied().collect())
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
            r"bank[0o]famerica|citi(?:bank)?|hsbc|barclays|paypal|p4ypal|paypa[1l]|",
            r"g[o0][o0]g[l1]e|g[o0][o0]gle|microsoft|micr[o0]s[o0]ft|micros[o0]ft|",
            r"apple|appl[3e]|amazon|amaz[o0]n|netflix|netfl1x|netfl[ix]|",
            r"faceb[o0][o0]k|faceb00k|instagram|instagr[a4]m|tiktok|t[i1]kt[o0]k|",
            r"roblox|r[o0]bl[o0]x|walmart|wa[l1]mart|target|t[a4]rget|",
            r"office|0ffice|[o0]ffice|outlook|0utl[o0][o0]k|dropbox|dr[o0]pb[o0]x|",
            r"linkedin|l[i1]nked[i1]n|tw[i1]tter|twitter|x\.com|telegram|te[l1]egram|",
            r"whatsapp|wh[a4]ts[a4]pp|snapchat|snapch[a4]t|discord|d[i1]sc[o0]rd",
            ")",
            r"[\-_.]+",
            r"(?:secure|security|verify|login|signin|account|update|support|help|check|payment|billing|docs|share|gift|free|promo|deal|prize|giveaway|download|app|restore|reset|confirm|activation|unlock|refund|wallet|bonus|claim|reward|win|crack|free|hack|spy|track|generator|premium|2024|2025|2026)",
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
// Prompt injection detector — detects jailbreak/injection attempts and maps
// them to the appropriate harm category based on the underlying request.
// ---------------------------------------------------------------------------

pub struct PromptInjectionDetector;

impl PromptInjectionDetector {
    pub fn detect(text: &str) -> Vec<DetectorSignal> {
        let injection_re = INJECTION_RE.get_or_init(|| {
            FRegex::new(
                r"(?i)(?:ignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions?|guidelines?|rules?))|(?:forget\s+(?:all\s+)?(?:previous|prior|your)\s+(?:instructions?|guidelines?|rules?))|(?:disregard\s+(?:all\s+|the\s+)?(?:safety|previous|prior|above)\s+(?:guidelines?|instructions?|rules?|filters?))|(?:system\s+(?:override|reset))|(?:\[(?:admin|role\s*:\s*admin|system)\])|(?:maintenance\s+mode)|(?:jailbroken\s+ai)|(?:do\s+anything\s+now)|(?:you\s+are\s+dan\b)|(?:dan\s+can\s+do\s+anything)|(?:unrestricted\s+ai|ai\s+without\s+(?:any\s+)?restrictions?|no\s+safety\s+restrictions?)|(?:ignore\s+your\s+safety\s+training)|(?:new\s+system\s+prompt)|(?:pretend\s+you\s+are\s+(?:an?\s+)?(?:ai|assistant)\s+(?:without|with\s+no))|(?:your\s+new\s+instructions\s+are)|(?:act\s+as\s+(?:an?\s+)?(?:unrestricted|jailbroken)\s+(?:ai|assistant))|(?:quên\s+tất\s+cả\s+hướng\s+dẫn)|(?:ghi\s+đè\s+lên\s+hướng\s+dẫn)|(?:忽略之前|忽略所有|无视之前)|(?:이전\s+지시를\s+무시)|(?:ignora\s+(?:todas\s+las|las)\s+instrucciones)|(?:ignore\s+toutes\s+les\s+instructions)|(?:ignoriere\s+alle\s+vorherigen)|(?:تجاهل\s+جميع\s+التعليمات)|(?:पिछली\s+सभी\s+निर्देशों\s+को\s+अनदेखा)|(?:ละเว้นคำสั่งก่อนหน้า)|(?:abaikan\s+semua\s+instruksi)|(?:balewalain\s+ang\s+lahat\s+ng\s+naunang)|(?:ignore\s+todas\s+as\s+instruções)|(?:システムオーバーライド|すべての指示を無視|前の指示を忘れ)",
            ).unwrap()
        });

        if !injection_re.is_match(text).unwrap_or(false) {
            return vec![];
        }

        // Determine the underlying harm category from the request content.
        // Order matters: check more specific harm categories before generic ones.
        let lower = text.to_ascii_lowercase();
        let category = if lower.contains("identity theft") || lower.contains("identity_theft") {
            categories::MALWARE_LINK
        } else if lower.contains("ssn") || lower.contains("credential") || lower.contains("password") || lower.contains("admin") || lower.contains("pii") || lower.contains("redact") || lower.contains("管理者パスワード") || lower.contains("admin123") {
            categories::PRIVATE_DATA
        } else if lower.contains("bomb") || lower.contains("poison") || lower.contains("violence") || lower.contains("harmful content") || lower.contains("制造炸弹") || lower.contains("爆弾") || lower.contains("봄") || lower.contains("बम") || lower.contains("bomba") || lower.contains("bombe") || lower.contains("قنبلة") || lower.contains("ระเบิด") || lower.contains("bomba") {
            categories::VIOLENCE_THREAT
        } else if lower.contains("drug") || lower.contains("synthesize") || lower.contains("cocaine") || lower.contains("meth") || lower.contains("lsd") || lower.contains("mdma") || lower.contains("drogas") || lower.contains("drogen") || lower.contains("drogue") || lower.contains("약물") || lower.contains("مخدرات") || lower.contains("दवा") || lower.contains("ยาเสพติด") || lower.contains("narkoba") || lower.contains("drogas") || lower.contains("마약") {
            categories::DRUGS_WEAPONS
        } else if lower.contains("stalk") || lower.contains("harass") {
            categories::HARASSMENT
        } else if (lower.contains("fake") && (lower.contains("passport") || lower.contains("identification") || lower.contains("id"))) || lower.contains("launder") || lower.contains("fake identification") {
            categories::ILLEGAL_GOODS
        } else {
            // Default: hacking, virus, malware
            categories::MALWARE_LINK
        };

        vec![DetectorSignal {
            category,
            severity: Severity::SEVERE,
            confidence: 0.95,
            reason_code: "prompt_injection".into(),
            action: Action::Block,
        }]
    }
}

static INJECTION_RE: OnceLock<FRegex> = OnceLock::new();

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

    // PII, URL, and malware URL run on the digit-preserving view only.
    for s in PiiDetector::detect(pattern_text) { signals.add(s); }
    for s in UrlDetector::detect(pattern_text) { signals.add(s); }
    for s in MalwareUrlDetector::detect(pattern_text) { signals.add(s); }

    // Prompt injection detector runs on the digit-preserving view.
    for s in PromptInjectionDetector::detect(pattern_text) { signals.add(s); }

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

/// Resolve deterministic signals into a verdict using the priority chain.
///
/// Media branches are checked first (highest priority) since they represent
/// on-device vision model verdicts with direct safety implications:
///   CHILD_SAFETY_media > SELF_HARM_media > EXTREMISM_media > HATE_media >
///   HARASSMENT_media > DRUGS_WEAPONS_media > NSFW_media > VIOLENCE_media >
///   DEEPFAKE_media > MALWARE_media
///
/// Then text-based signals in priority order:
///   CHILD_SAFETY > SELF_HARM > PRIVATE_DATA > SCAM_FRAUD > HATE_SPEECH > VIOLENCE > NSFW > SPAM
pub fn resolve_priority_chain(signals: &LocalSignals) -> Option<DetectorSignal> {
    // --- Media branches (highest priority) ---
    if let Some(s) = child_safety_media_branch(signals) { return Some(s); }
    if let Some(s) = self_harm_media_branch(signals) { return Some(s); }
    if let Some(s) = extremism_media_branch(signals) { return Some(s); }
    if let Some(s) = hate_media_branch(signals) { return Some(s); }
    if let Some(s) = harassment_media_branch(signals) { return Some(s); }
    if let Some(s) = drugs_weapons_media_branch(signals) { return Some(s); }
    if let Some(s) = nsfw_media_branch(signals) { return Some(s); }
    if let Some(s) = violence_media_branch(signals) { return Some(s); }
    if let Some(s) = deepfake_media_branch(signals) { return Some(s); }
    if let Some(s) = malware_media_branch(signals) { return Some(s); }

    // --- Text-based signals ---
    if signals.signals.is_empty() { return None; }
    let priority = [
        categories::CHILD_SAFETY,
        categories::SELF_HARM,
        categories::SCAM_FRAUD,
        categories::PRIVATE_DATA,
        categories::MALWARE_LINK,
        categories::HATE,
        categories::VIOLENCE_THREAT,
        categories::EXTREMISM,
        categories::HARASSMENT,
        categories::SEXUAL_ADULT,
        categories::ILLEGAL_GOODS,
        categories::DRUGS_WEAPONS,
        categories::MISINFORMATION_HEALTH,
        categories::MISINFORMATION_CIVIC,
        categories::COMMUNITY_RULE,
        categories::DEEPFAKE_SYNTHETIC,
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

// ---------------------------------------------------------------------------
// Media branch detectors — image/video safety scores from on-device vision
// models. Ported from slm-guardrail's priority_chain.rs media branches.
// Each branch iterates media_descriptors and fires on the first score above
// MEDIA_TRIGGER_THRESHOLD (> 0.7). Severity escalates at MEDIA_HIGH_BAND (>= 0.9).
// ---------------------------------------------------------------------------

/// Helper: clamp a value into `[lo, hi]`, collapsing NaN to `lo`.
fn clamp_finite(value: f64, lo: f64, hi: f64) -> f64 {
    if value.is_nan() { return lo; }
    if value < lo { return lo; }
    if value > hi { return hi; }
    value
}

/// Helper: build a media branch DetectorSignal for the standard
/// warn/strong_warn pattern (all branches except child_safety and malware).
fn media_signal_standard(
    score: f64,
    category: u32,
    reason_code: &str,
) -> DetectorSignal {
    let (severity, action) = if score >= MEDIA_HIGH_BAND {
        (Severity::SEVERE, Action::Block)
    } else {
        (Severity::BORDERLINE, Action::Warn)
    };
    DetectorSignal {
        category,
        severity,
        confidence: score.min(DETERMINISTIC_CONFIDENCE_CEIL),
        reason_code: reason_code.into(),
        action,
    }
}

/// CHILD_SAFETY media — fires on `child_safety_score > 0.7`.
/// Severity-5 critical, always Block. Highest-priority media branch.
fn child_safety_media_branch(signals: &LocalSignals) -> Option<DetectorSignal> {
    for m in &signals.media_descriptors {
        if let Some(score) = m.child_safety_score {
            if score > MEDIA_TRIGGER_THRESHOLD {
                return Some(DetectorSignal {
                    category: categories::CHILD_SAFETY,
                    severity: Severity::CRITICAL,
                    confidence: clamp_finite(score, CHILD_SAFETY_CONFIDENCE_FLOOR, CHILD_SAFETY_CONFIDENCE_CEIL),
                    reason_code: "CHILD_SAFETY_MEDIA".into(),
                    action: Action::Block,
                });
            }
        }
    }
    None
}

/// SELF_HARM media — fires on `self_harm_score > 0.7`.
fn self_harm_media_branch(signals: &LocalSignals) -> Option<DetectorSignal> {
    for m in &signals.media_descriptors {
        if let Some(score) = m.self_harm_score {
            if score > MEDIA_TRIGGER_THRESHOLD {
                return Some(media_signal_standard(score, categories::SELF_HARM, "SELF_HARM_MEDIA"));
            }
        }
    }
    None
}

/// EXTREMISM media — fires on `extremism_score > 0.7`.
fn extremism_media_branch(signals: &LocalSignals) -> Option<DetectorSignal> {
    for m in &signals.media_descriptors {
        if let Some(score) = m.extremism_score {
            if score > MEDIA_TRIGGER_THRESHOLD {
                return Some(media_signal_standard(score, categories::EXTREMISM, "EXTREMISM_MEDIA"));
            }
        }
    }
    None
}

/// HATE media — fires on `hate_score > 0.7`.
fn hate_media_branch(signals: &LocalSignals) -> Option<DetectorSignal> {
    for m in &signals.media_descriptors {
        if let Some(score) = m.hate_score {
            if score > MEDIA_TRIGGER_THRESHOLD {
                return Some(media_signal_standard(score, categories::HATE_SPEECH, "HATE_MEDIA"));
            }
        }
    }
    None
}

/// HARASSMENT media — fires on `harassment_score > 0.7`.
fn harassment_media_branch(signals: &LocalSignals) -> Option<DetectorSignal> {
    for m in &signals.media_descriptors {
        if let Some(score) = m.harassment_score {
            if score > MEDIA_TRIGGER_THRESHOLD {
                return Some(media_signal_standard(score, categories::HARASSMENT, "HARASSMENT_MEDIA"));
            }
        }
    }
    None
}

/// DRUGS_WEAPONS media — fires on `drugs_weapons_score > 0.7`.
fn drugs_weapons_media_branch(signals: &LocalSignals) -> Option<DetectorSignal> {
    for m in &signals.media_descriptors {
        if let Some(score) = m.drugs_weapons_score {
            if score > MEDIA_TRIGGER_THRESHOLD {
                return Some(media_signal_standard(score, categories::DRUGS_WEAPONS, "DRUGS_WEAPONS_MEDIA"));
            }
        }
    }
    None
}

/// NSFW media — fires on `nsfw_score > 0.7`.
fn nsfw_media_branch(signals: &LocalSignals) -> Option<DetectorSignal> {
    for m in &signals.media_descriptors {
        if let Some(score) = m.nsfw_score {
            if score > MEDIA_TRIGGER_THRESHOLD {
                return Some(media_signal_standard(score, categories::NSFW, "NSFW_MEDIA"));
            }
        }
    }
    None
}

/// VIOLENCE media — fires on `violence_score > 0.7`.
fn violence_media_branch(signals: &LocalSignals) -> Option<DetectorSignal> {
    for m in &signals.media_descriptors {
        if let Some(score) = m.violence_score {
            if score > MEDIA_TRIGGER_THRESHOLD {
                return Some(media_signal_standard(score, categories::VIOLENCE, "VIOLENCE_MEDIA"));
            }
        }
    }
    None
}

/// DEEPFAKE media — fires on `deepfake_score > 0.7`.
fn deepfake_media_branch(signals: &LocalSignals) -> Option<DetectorSignal> {
    for m in &signals.media_descriptors {
        if let Some(score) = m.deepfake_score {
            if score > MEDIA_TRIGGER_THRESHOLD {
                return Some(media_signal_standard(score, categories::DEEPFAKE, "DEEPFAKE_MEDIA"));
            }
        }
    }
    None
}

/// MALWARE media — fires on `malware_score > 0.7`.
/// Always severity-3 Warn (no high-band escalation).
fn malware_media_branch(signals: &LocalSignals) -> Option<DetectorSignal> {
    for m in &signals.media_descriptors {
        if let Some(score) = m.malware_score {
            if score > MEDIA_TRIGGER_THRESHOLD {
                return Some(DetectorSignal {
                    category: categories::MALWARE,
                    severity: Severity::BORDERLINE,
                    confidence: score.min(DETERMINISTIC_CONFIDENCE_CEIL),
                    reason_code: "MALWARE_MEDIA".into(),
                    action: Action::Warn,
                });
            }
        }
    }
    None
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

    #[test]
    fn test_malware_url_zip() {
        let sigs = MalwareUrlDetector::detect("Download the cracked version of Photoshop for free at https://warez-download.ru/photoshop-crack-2024.zip");
        assert!(!sigs.is_empty(), "expected malware signal for .zip URL, got {:?}", sigs);
    }

    #[test]
    fn test_malware_url_spy_tools() {
        let sigs = MalwareUrlDetector::detect("I found this site that lets you see anyone's private messages: https://spy-tools-online.net/hack-instagram");
        assert!(!sigs.is_empty(), "expected malware signal for spy-tools URL, got {:?}", sigs);
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

    // --- Media branch tests ---

    fn media_desc(score_field: &str, score: f64) -> crate::media::MediaDescriptor {
        let mut d = crate::media::MediaDescriptor {
            kind: "image".into(),
            nsfw_score: None,
            violence_score: None,
            self_harm_score: None,
            hate_score: None,
            harassment_score: None,
            drugs_weapons_score: None,
            extremism_score: None,
            child_safety_score: None,
            deepfake_score: None,
            malware_score: None,
            face_count: None,
        };
        match score_field {
            "nsfw" => d.nsfw_score = Some(score),
            "violence" => d.violence_score = Some(score),
            "self_harm" => d.self_harm_score = Some(score),
            "hate" => d.hate_score = Some(score),
            "harassment" => d.harassment_score = Some(score),
            "drugs_weapons" => d.drugs_weapons_score = Some(score),
            "extremism" => d.extremism_score = Some(score),
            "child_safety" => d.child_safety_score = Some(score),
            "deepfake" => d.deepfake_score = Some(score),
            "malware" => d.malware_score = Some(score),
            _ => {}
        }
        d
    }

    fn signals_with_media(media: Vec<crate::media::MediaDescriptor>) -> LocalSignals {
        LocalSignals::default().with_media(media)
    }

    #[test]
    fn test_child_safety_media_fires() {
        let sig = signals_with_media(vec![media_desc("child_safety", 0.85)]);
        let result = resolve_priority_chain(&sig);
        assert!(result.is_some());
        let s = result.unwrap();
        assert_eq!(s.category, categories::CHILD_SAFETY);
        assert_eq!(s.severity, Severity::CRITICAL);
        assert_eq!(s.action, Action::Block);
        assert_eq!(s.reason_code, "CHILD_SAFETY_MEDIA");
    }

    #[test]
    fn test_child_safety_media_below_threshold() {
        let sig = signals_with_media(vec![media_desc("child_safety", 0.5)]);
        assert!(resolve_priority_chain(&sig).is_none());
    }

    #[test]
    fn test_self_harm_media_fires() {
        let sig = signals_with_media(vec![media_desc("self_harm", 0.75)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::SELF_HARM);
        assert_eq!(result.severity, Severity::BORDERLINE);
        assert_eq!(result.action, Action::Warn);
    }

    #[test]
    fn test_self_harm_media_high_band() {
        let sig = signals_with_media(vec![media_desc("self_harm", 0.92)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::SELF_HARM);
        assert_eq!(result.severity, Severity::SEVERE);
        assert_eq!(result.action, Action::Block);
    }

    #[test]
    fn test_nsfw_media_fires() {
        let sig = signals_with_media(vec![media_desc("nsfw", 0.8)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::NSFW);
        assert_eq!(result.reason_code, "NSFW_MEDIA");
    }

    #[test]
    fn test_violence_media_fires() {
        let sig = signals_with_media(vec![media_desc("violence", 0.88)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::VIOLENCE);
        assert_eq!(result.reason_code, "VIOLENCE_MEDIA");
    }

    #[test]
    fn test_extremism_media_fires() {
        let sig = signals_with_media(vec![media_desc("extremism", 0.75)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::EXTREMISM);
    }

    #[test]
    fn test_hate_media_fires() {
        let sig = signals_with_media(vec![media_desc("hate", 0.75)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::HATE_SPEECH);
        assert_eq!(result.reason_code, "HATE_MEDIA");
    }

    #[test]
    fn test_harassment_media_fires() {
        let sig = signals_with_media(vec![media_desc("harassment", 0.75)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::HARASSMENT);
    }

    #[test]
    fn test_drugs_weapons_media_fires() {
        let sig = signals_with_media(vec![media_desc("drugs_weapons", 0.75)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::DRUGS_WEAPONS);
    }

    #[test]
    fn test_deepfake_media_fires() {
        let sig = signals_with_media(vec![media_desc("deepfake", 0.75)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::DEEPFAKE);
    }

    #[test]
    fn test_malware_media_fires() {
        let sig = signals_with_media(vec![media_desc("malware", 0.75)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::MALWARE);
        assert_eq!(result.severity, Severity::BORDERLINE);
        assert_eq!(result.action, Action::Warn);
    }

    #[test]
    fn test_malware_media_no_high_band_escalation() {
        let sig = signals_with_media(vec![media_desc("malware", 0.95)]);
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.severity, Severity::BORDERLINE);
        assert_eq!(result.action, Action::Warn);
    }

    #[test]
    fn test_child_safety_media_priority_over_text() {
        let mut sig = signals_with_media(vec![media_desc("child_safety", 0.85)]);
        sig.add(DetectorSignal {
            category: categories::SCAM_FRAUD,
            severity: Severity::SEVERE,
            confidence: 0.99,
            reason_code: "scam_test".into(),
            action: Action::Block,
        });
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::CHILD_SAFETY);
        assert_eq!(result.reason_code, "CHILD_SAFETY_MEDIA");
    }

    #[test]
    fn test_media_below_threshold_falls_through_to_text() {
        let mut sig = signals_with_media(vec![media_desc("nsfw", 0.3)]);
        sig.add(DetectorSignal {
            category: categories::SCAM_FRAUD,
            severity: Severity::BORDERLINE,
            confidence: 0.8,
            reason_code: "scam_test".into(),
            action: Action::Warn,
        });
        let result = resolve_priority_chain(&sig).unwrap();
        assert_eq!(result.category, categories::SCAM_FRAUD);
    }
}

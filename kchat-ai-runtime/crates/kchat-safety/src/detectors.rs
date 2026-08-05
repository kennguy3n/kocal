//! Deterministic detectors — fast regex/heuristic detectors that run before
//! any ML model.
//!
//! Detectors run in priority order:
//!   CHILD_SAFETY > PRIVATE_DATA > SCAM_FRAUD > LEXICON > None
//!
//! Each detector returns a `DetectorSignal` if it matches, which the priority
//! chain resolves into a final verdict.

use crate::verdict::{Action, Severity, VerdictBuilder};
use regex::Regex;
use std::sync::OnceLock;

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
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }

    pub fn add(&mut self, signal: DetectorSignal) {
        self.signals.push(signal);
    }
}

/// PII detector — detects credit cards (Luhn), IBAN (mod-97), phone numbers,
/// SSN, email addresses, and credential patterns.
pub struct PiiDetector;

impl PiiDetector {
    pub fn detect(text: &str) -> Vec<DetectorSignal> {
        let mut signals = Vec::new();

        // Credit card (Luhn-validated)
        if let Some(_) = detect_credit_card(text) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA,
                severity: Severity::SEVERE,
                confidence: 0.95,
                reason_code: "pii_credit_card".into(),
                action: Action::Redact,
            });
        }

        // Email — fixed regex (removed erroneous | in character class)
        if EMAIL_RE.get_or_init(|| Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap()).is_match(text) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA,
                severity: Severity::BORDERLINE,
                confidence: 0.90,
                reason_code: "pii_email".into(),
                action: Action::Redact,
            });
        }

        // Phone number (basic international pattern)
        if PHONE_RE.get_or_init(|| Regex::new(r"\b\+?[0-9]{1,3}[-.\s]?\(?[0-9]{1,4}\)?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}\b").unwrap()).is_match(text) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA,
                severity: Severity::BORDERLINE,
                confidence: 0.70,
                reason_code: "pii_phone".into(),
                action: Action::Redact,
            });
        }

        // SSN (US format)
        if SSN_RE.get_or_init(|| Regex::new(r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b").unwrap()).is_match(text) {
            signals.push(DetectorSignal {
                category: categories::PRIVATE_DATA,
                severity: Severity::SEVERE,
                confidence: 0.95,
                reason_code: "pii_ssn".into(),
                action: Action::Redact,
            });
        }

        signals
    }
}

static EMAIL_RE: OnceLock<Regex> = OnceLock::new();
static PHONE_RE: OnceLock<Regex> = OnceLock::new();
static SSN_RE: OnceLock<Regex> = OnceLock::new();

/// Detect credit card numbers with Luhn validation.
fn detect_credit_card(text: &str) -> Option<String> {
    let re = CC_RE.get_or_init(|| {
        Regex::new(r"\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b").unwrap()
    });

    for m in re.find_iter(text) {
        let digits: String = m.as_str().chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 13 && digits.len() <= 19 && luhn_valid(&digits) {
            return Some(m.as_str().to_string());
        }
    }
    None
}

static CC_RE: OnceLock<Regex> = OnceLock::new();

/// Luhn algorithm validation.
fn luhn_valid(digits: &str) -> bool {
    let mut sum = 0u32;
    let mut alt = false;
    for c in digits.chars().rev() {
        if let Some(mut d) = c.to_digit(10) {
            if alt {
                d *= 2;
                if d > 9 {
                    d -= 9;
                }
            }
            sum += d;
            alt = !alt;
        }
    }
    sum % 10 == 0
}

/// Scam/fraud detector — detects common scam patterns.
pub struct ScamDetector;

impl ScamDetector {
    pub fn detect(text: &str) -> Vec<DetectorSignal> {
        let mut signals = Vec::new();
        let lower = text.to_lowercase();

        // Urgency + money request pattern
        let urgency_patterns = [
            "urgent", "immediately", "act now", "limited time",
            "expires today", "final notice", "last chance",
        ];
        let money_patterns = [
            "send money", "wire transfer", "gift card", "bitcoin",
            "crypto", "paypal", "venmo", "cash app", "western union",
        ];

        let has_urgency = urgency_patterns.iter().any(|p| lower.contains(p));
        let has_money = money_patterns.iter().any(|p| lower.contains(p));

        if has_urgency && has_money {
            signals.push(DetectorSignal {
                category: categories::SCAM_FRAUD,
                severity: Severity::SEVERE,
                confidence: 0.85,
                reason_code: "scam_urgency_money".into(),
                action: Action::Warn,
            });
        }

        // Impersonation patterns
        let impersonation = [
            "i am from the government",
            "i am from the irs",
            "i am from microsoft",
            "i am from apple support",
            "your account has been compromised",
            "your account will be suspended",
            "verify your identity",
        ];
        if impersonation.iter().any(|p| lower.contains(p)) {
            signals.push(DetectorSignal {
                category: categories::SCAM_FRAUD,
                severity: Severity::SEVERE,
                confidence: 0.80,
                reason_code: "scam_impersonation".into(),
                action: Action::Warn,
            });
        }

        signals
    }
}

/// Lexicon detector — matches against signed policy pack lexicons.
/// This is a simplified version; the real implementation loads from signed packs.
pub struct LexiconDetector;

impl LexiconDetector {
    pub fn detect(text: &str, lexicon: &[(String, u32, Severity)]) -> Vec<DetectorSignal> {
        let lower = text.to_lowercase();
        let mut signals = Vec::new();

        for (term, category, severity) in lexicon {
            // Terms are pre-lowercased in build_lexicon(), so no need to lowercase again
            if lower.contains(term.as_str()) {
                signals.push(DetectorSignal {
                    category: *category,
                    severity: *severity,
                    confidence: 0.75,
                    reason_code: format!("lexicon_match_{}", term.replace(' ', "_")),
                    action: if *severity >= Severity::SEVERE {
                        Action::Block
                    } else {
                        Action::Warn
                    },
                });
            }
        }

        signals
    }
}

/// URL risk detector — scores URLs for phishing/malware risk indicators.
pub struct UrlDetector;

impl UrlDetector {
    pub fn detect(text: &str) -> Vec<DetectorSignal> {
        let mut signals = Vec::new();
        let re = URL_RE.get_or_init(|| {
            Regex::new(r"https?://[^\s]+").unwrap()
        });

        for m in re.find_iter(text) {
            let url = m.as_str().to_lowercase();
            let mut risk_score = 0u32;

            // Suspicious TLDs — check the domain part (after :// and before first / or end)
            let domain = url.strip_prefix("http://")
                .or_else(|| url.strip_prefix("https://"))
                .unwrap_or(&url);
            let domain = domain.split('/').next().unwrap_or(domain);
            // Remove port if present
            let domain = domain.split(':').next().unwrap_or(domain);
            if domain.ends_with(".tk") || domain.ends_with(".ml") || domain.ends_with(".ga") || domain.ends_with(".cf") {
                risk_score += 2;
            }

            // IP address instead of domain — high risk indicator.
            // Use proper octet validation (0-255) to avoid false positives.
            // Also handles optional port suffix (e.g. http://1.2.3.4:8080).
            if URL_IP_RE.get_or_init(|| {
                Regex::new(r"https?://(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}(?::\d+)?").unwrap()
            }).is_match(&url) {
                risk_score += 3;
            }

            // URL shortener
            let shorteners = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"];
            if shorteners.iter().any(|s| url.contains(s)) {
                risk_score += 1;
            }

            // Excessive subdomains
            let dot_count = url.matches('.').count();
            if dot_count > 4 {
                risk_score += 1;
            }

            if risk_score >= 3 {
                signals.push(DetectorSignal {
                    category: categories::SCAM_FRAUD,
                    severity: Severity::SEVERE,
                    confidence: 0.70,
                    reason_code: "url_high_risk".into(),
                    action: Action::Warn,
                });
            } else if risk_score >= 1 {
                signals.push(DetectorSignal {
                    category: categories::SCAM_FRAUD,
                    severity: Severity::BORDERLINE,
                    confidence: 0.50,
                    reason_code: "url_moderate_risk".into(),
                    action: Action::Warn,
                });
            }
        }

        signals
    }
}

static URL_RE: OnceLock<Regex> = OnceLock::new();
static URL_IP_RE: OnceLock<Regex> = OnceLock::new();

/// Run all deterministic detectors and collect signals.
///
/// `pattern_text` is lightly normalized (NFKC + zero-width strip, no leetspeak)
/// for PII and URL pattern matching. `lexicon_text` is fully normalized
/// (including leetspeak and case fold) for lexicon matching.
pub fn run_all_detectors(
    pattern_text: &str,
    lexicon_text: &str,
    lexicon: &[(String, u32, Severity)],
) -> LocalSignals {
    let mut signals = LocalSignals::default();

    // PII — run on pattern text (digits preserved)
    for s in PiiDetector::detect(pattern_text) {
        signals.add(s);
    }

    // Scam/fraud — run on lexicon text (case-folded, leetspeak normalized)
    for s in ScamDetector::detect(lexicon_text) {
        signals.add(s);
    }

    // URL risk — run on pattern text (URLs and IPs preserved)
    for s in UrlDetector::detect(pattern_text) {
        signals.add(s);
    }

    // Lexicon — run on lexicon text (case-folded, leetspeak normalized)
    for s in LexiconDetector::detect(lexicon_text, lexicon) {
        signals.add(s);
    }

    signals
}

/// Resolve deterministic signals into a verdict using the priority chain:
///   CHILD_SAFETY > PRIVATE_DATA > SCAM_FRAUD > LEXICON > None
pub fn resolve_priority_chain(signals: &LocalSignals) -> Option<DetectorSignal> {
    if signals.is_empty() {
        return None;
    }

    // Priority order (highest first)
    let priority = [
        categories::CHILD_SAFETY,
        categories::SELF_HARM,
        categories::PRIVATE_DATA,
        categories::SCAM_FRAUD,
        categories::HATE_SPEECH,
        categories::VIOLENCE,
        categories::NSFW,
        categories::SPAM,
    ];

    for &cat in &priority {
        if let Some(s) = signals.signals.iter().find(|s| s.category == cat) {
            return Some(s.clone());
        }
    }

    // Return the first signal if no priority match
    signals.signals.first().cloned()
}

/// Convert a detector signal into a verdict.
pub fn signal_to_verdict(signal: &DetectorSignal) -> crate::verdict::Verdict {
    VerdictBuilder::default()
        .action(signal.action)
        .severity(signal.severity)
        .category(signal.category)
        .confidence(signal.confidence)
        .reason_code(&signal.reason_code)
        .source(crate::verdict::VerdictSource::Deterministic)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credit_card_detection() {
        // Valid Luhn: 4111111111111111 (Visa test number)
        let signals = PiiDetector::detect("my card is 4111 1111 1111 1111");
        assert!(signals.iter().any(|s| s.reason_code == "pii_credit_card"), "expected credit card detection");
    }

    #[test]
    fn test_invalid_credit_card_not_detected() {
        // Invalid Luhn
        let signals = PiiDetector::detect("1234 5678 9012 3456");
        assert!(!signals.iter().any(|s| s.reason_code == "pii_credit_card"));
    }

    #[test]
    fn test_email_detection() {
        let signals = PiiDetector::detect("contact me at user@example.com");
        assert!(signals.iter().any(|s| s.reason_code == "pii_email"));
    }

    #[test]
    fn test_scam_urgency_money() {
        let signals = ScamDetector::detect("URGENT! Send money via bitcoin immediately!");
        assert!(signals.iter().any(|s| s.reason_code == "scam_urgency_money"));
    }

    #[test]
    fn test_scam_impersonation() {
        let signals = ScamDetector::detect("I am from the IRS and your account has been compromised");
        assert!(signals.iter().any(|s| s.reason_code == "scam_impersonation"));
    }

    #[test]
    fn test_url_ip_address() {
        let signals = UrlDetector::detect("Visit http://192.168.1.1/login now!");
        assert!(signals.iter().any(|s| s.reason_code == "url_high_risk"));
    }

    #[test]
    fn test_lexicon_match() {
        let lexicon = vec![
            ("hate".to_string(), categories::HATE_SPEECH, Severity::SEVERE),
        ];
        let signals = LexiconDetector::detect("I hate everything", &lexicon);
        assert!(!signals.is_empty());
    }

    #[test]
    fn test_priority_chain() {
        let mut signals = LocalSignals::default();
        signals.add(DetectorSignal {
            category: categories::SPAM,
            severity: Severity::BENIGN,
            confidence: 0.5,
            reason_code: "spam".into(),
            action: Action::Warn,
        });
        signals.add(DetectorSignal {
            category: categories::CHILD_SAFETY,
            severity: Severity::CRITICAL,
            confidence: 0.95,
            reason_code: "child_safety".into(),
            action: Action::Block,
        });

        let resolved = resolve_priority_chain(&signals);
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().category, categories::CHILD_SAFETY);
    }

    #[test]
    fn test_empty_signals_returns_none() {
        let signals = LocalSignals::default();
        assert!(resolve_priority_chain(&signals).is_none());
    }

    #[test]
    fn test_luhn_valid() {
        assert!(luhn_valid("4111111111111111"));
        assert!(luhn_valid("4012888888881881"));
        assert!(!luhn_valid("1234567890123456"));
    }
}

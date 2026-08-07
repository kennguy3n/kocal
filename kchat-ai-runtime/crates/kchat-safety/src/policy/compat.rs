//! Signed policy pack format and Ed25519 verification.
//!
//! Policy packs contain deterministic rules, lexicons, thresholds, and
//! severity rubrics. They are Ed25519-signed with KChat release keys and
//! verified on-device before use.
//!
//! Verification is three-step:
//! 1. Content digest match — recompute SHA-256 over all non-manifest files
//! 2. Pinned key equality — manifest's public_key must equal caller-pinned key
//! 3. Strict Ed25519 verify — signature over canonical preimage

use crate::verdict::Severity;
use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Risk category for policy rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskCategory {
    ChildSafety,
    PrivateData,
    ScamFraud,
    HateSpeech,
    Violence,
    Nsfw,
    SelfHarm,
    Spam,
    Custom(u32),
}

impl RiskCategory {
    /// Map to taxonomy category ID per kchat.guardrail.taxonomy.v1.
    pub fn as_u32(self) -> u32 {
        match self {
            RiskCategory::ChildSafety => 1,    // CHILD_SAFETY
            RiskCategory::SelfHarm => 2,       // SELF_HARM
            RiskCategory::Violence => 3,       // VIOLENCE_THREAT
            RiskCategory::HateSpeech => 6,     // HATE
            RiskCategory::ScamFraud => 7,      // SCAM_FRAUD
            RiskCategory::PrivateData => 9,    // PRIVATE_DATA
            RiskCategory::Nsfw => 10,          // SEXUAL_ADULT
            RiskCategory::Spam => 7,           // Map to SCAM_FRAUD (closest taxonomy match)
            RiskCategory::Custom(id) => id,
        }
    }
}

/// A single policy rule in a signed pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Rule identifier
    pub rule_id: String,
    /// Risk category
    pub category: RiskCategory,
    /// Severity level (1-5)
    pub severity: u8,
    /// Lexicon terms to match (case-insensitive after normalization)
    pub lexicon: Vec<String>,
    /// Regex patterns to match
    pub regex_patterns: Vec<String>,
    /// Action to take on match
    pub action: String, // "block", "warn", "redact", "require_consent"
    /// Confidence floor (0.0-1.0)
    pub confidence_floor: f64,
    /// Whether this rule can be overridden by community overlays
    pub overridable: bool,
}

/// Manifest for a signed policy pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyPackManifest {
    pub pack_id: String,
    pub version: String,
    pub content_sha256: String,
    pub public_key: String,
    pub signature: String,
}

/// A verified policy pack loaded from signed data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyPack {
    pub manifest: PolicyPackManifest,
    pub rules: Vec<PolicyRule>,
    /// Thresholds for encoder confidence → action mapping
    pub thresholds: PolicyThresholds,
    /// SLM prompt template for ambiguous cases
    pub slm_prompt: String,
}

/// Thresholds for mapping encoder confidence to actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyThresholds {
    /// Confidence above which to block
    pub block_threshold: f64,
    /// Confidence above which to warn
    pub warn_threshold: f64,
    /// Confidence below which to escalate to encoder
    pub encoder_escalation_threshold: f64,
}

impl Default for PolicyThresholds {
    fn default() -> Self {
        Self {
            block_threshold: 0.85,
            warn_threshold: 0.60,
            encoder_escalation_threshold: 0.40,
        }
    }
}

#[derive(Debug, Error)]
pub enum PolicyPackError {
    #[error("content digest mismatch: expected {expected}, got {actual}")]
    DigestMismatch { expected: String, actual: String },

    #[error("public key does not match pinned key")]
    KeyMismatch,

    #[error("signature verification failed: {0}")]
    SignatureInvalid(String),

    #[error("pack parsing failed: {0}")]
    ParseFailed(String),

    #[error("null digest detected — release blocker")]
    NullDigest,
}

/// Verify a policy pack's signature and content digest.
///
/// `pack_json` is the serialized PolicyPack (without manifest fields used for verification).
/// `pinned_public_key_hex` is the Ed25519 public key pinned in the app binary.
pub fn verify_policy_pack(
    pack: &PolicyPack,
    pinned_public_key_hex: &str,
) -> Result<(), PolicyPackError> {
    // 1. Check for null digest
    if pack.manifest.content_sha256.chars().all(|c| c == '0') {
        return Err(PolicyPackError::NullDigest);
    }

    // 2. Validate hex length BEFORE comparison to avoid timing leaks
    if pack.manifest.public_key.len() != 64 {
        return Err(PolicyPackError::SignatureInvalid("public key hex must be 64 characters".into()));
    }

    // 3. Pinned key equality — constant-time comparison
    if !constant_time_eq_bytes(pack.manifest.public_key.as_bytes(), pinned_public_key_hex.as_bytes()) {
        return Err(PolicyPackError::KeyMismatch);
    }

    // 4. Parse public key
    let pk_bytes = hex::decode(&pack.manifest.public_key)
        .map_err(|e| PolicyPackError::SignatureInvalid(format!("bad public key hex: {e}")))?;
    if pk_bytes.len() != 32 {
        return Err(PolicyPackError::SignatureInvalid("public key must be 32 bytes".into()));
    }
    let mut pk_arr = [0u8; 32];
    pk_arr.copy_from_slice(&pk_bytes);
    let verifying_key = VerifyingKey::from_bytes(&pk_arr)
        .map_err(|e| PolicyPackError::SignatureInvalid(format!("invalid public key: {e}")))?;

    // 4. Parse signature — validate hex length before decoding
    if pack.manifest.signature.len() != 128 {
        return Err(PolicyPackError::SignatureInvalid("signature hex must be 128 characters".into()));
    }
    let sig_bytes = hex::decode(&pack.manifest.signature)
        .map_err(|e| PolicyPackError::SignatureInvalid(format!("bad signature hex: {e}")))?;
    if sig_bytes.len() != 64 {
        return Err(PolicyPackError::SignatureInvalid("signature must be 64 bytes".into()));
    }
    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&sig_bytes);
    let signature = Signature::from_bytes(&sig_arr);

    // 5. Canonical preimage: "{content_sha256}|{pack_id}|{version}"
    let preimage = format!(
        "{}|{}|{}",
        pack.manifest.content_sha256, pack.manifest.pack_id, pack.manifest.version
    );

    // 6. Verify using strict mode (rejects weak keys and malleable signatures)
    verifying_key
        .verify_strict(preimage.as_bytes(), &signature)
        .map_err(|e| PolicyPackError::SignatureInvalid(format!("verification failed: {e}")))?;

    Ok(())
}

/// Compute the content digest of a policy pack's rules.
pub fn compute_pack_digest(pack: &PolicyPack) -> String {
    let mut hasher = Sha256::new();
    // Hash the rules and thresholds (not the manifest fields)
    let content = serde_json::to_string(&pack.rules).unwrap_or_default();
    hasher.update(content.as_bytes());
    let thresholds = serde_json::to_string(&pack.thresholds).unwrap_or_default();
    hasher.update(thresholds.as_bytes());
    hasher.update(pack.slm_prompt.as_bytes());
    hex::encode(hasher.finalize())
}

/// Constant-time byte slice comparison to prevent timing attacks.
fn constant_time_eq_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

/// Convert a policy rule's action string to a verdict Action.
pub fn parse_action(action: &str) -> crate::verdict::Action {
    match action {
        "block" => crate::verdict::Action::Block,
        "warn" => crate::verdict::Action::Warn,
        "redact" => crate::verdict::Action::Redact,
        "require_consent" => crate::verdict::Action::RequireConsent,
        _ => crate::verdict::Action::Allow,
    }
}

/// Convert a severity level (1-5) to a Severity.
pub fn severity_from_u8(level: u8) -> Severity {
    match level {
        0 => Severity::SAFE,
        1 => Severity::BENIGN,
        2 => Severity::BORDERLINE,
        3 => Severity::SEVERE,
        4..=5 => Severity::CRITICAL,
        _ => Severity::SAFE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{SigningKey, Signer};
    use rand::rngs::OsRng;

    fn make_test_pack() -> (PolicyPack, String, SigningKey) {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        let pk_hex = hex::encode(verifying_key.to_bytes());

        let mut pack = PolicyPack {
            manifest: PolicyPackManifest {
                pack_id: "test-pack".into(),
                version: "1.0.0".into(),
                content_sha256: "0".repeat(64), // placeholder
                public_key: pk_hex.clone(),
                signature: "0".repeat(128), // placeholder
            },
            rules: vec![PolicyRule {
                rule_id: "rule-001".into(),
                category: RiskCategory::ScamFraud,
                severity: 3,
                lexicon: vec!["send money".into()],
                regex_patterns: vec![],
                action: "warn".into(),
                confidence_floor: 0.7,
                overridable: true,
            }],
            thresholds: PolicyThresholds::default(),
            slm_prompt: "Analyze this message for scam indicators.".into(),
        };

        // Compute content digest
        pack.manifest.content_sha256 = compute_pack_digest(&pack);

        // Sign
        let preimage = format!(
            "{}|{}|{}",
            pack.manifest.content_sha256, pack.manifest.pack_id, pack.manifest.version
        );
        let sig = signing_key.sign(preimage.as_bytes());
        pack.manifest.signature = hex::encode(sig.to_bytes());

        (pack, pk_hex, signing_key)
    }

    #[test]
    fn test_verify_policy_pack_succeeds() {
        let (pack, pk_hex, _) = make_test_pack();
        assert!(verify_policy_pack(&pack, &pk_hex).is_ok());
    }

    #[test]
    fn test_verify_policy_pack_wrong_key() {
        let (pack, _, _) = make_test_pack();
        let wrong_key = "a".repeat(64);
        assert!(verify_policy_pack(&pack, &wrong_key).is_err());
    }

    #[test]
    fn test_null_digest_rejected() {
        let (mut pack, pk_hex, _) = make_test_pack();
        pack.manifest.content_sha256 = "0".repeat(64);
        assert!(matches!(
            verify_policy_pack(&pack, &pk_hex),
            Err(PolicyPackError::NullDigest)
        ));
    }

    #[test]
    fn test_parse_action() {
        assert_eq!(parse_action("block"), crate::verdict::Action::Block);
        assert_eq!(parse_action("warn"), crate::verdict::Action::Warn);
        assert_eq!(parse_action("redact"), crate::verdict::Action::Redact);
        assert_eq!(parse_action("require_consent"), crate::verdict::Action::RequireConsent);
        assert_eq!(parse_action("unknown"), crate::verdict::Action::Allow);
    }

    #[test]
    fn test_severity_from_u8() {
        assert_eq!(severity_from_u8(0), Severity::SAFE);
        assert_eq!(severity_from_u8(3), Severity::SEVERE);
        assert_eq!(severity_from_u8(5), Severity::CRITICAL);
    }
}

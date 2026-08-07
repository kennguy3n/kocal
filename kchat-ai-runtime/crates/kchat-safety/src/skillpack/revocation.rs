//! Skill-pack revocation list.
//!
//! Mirrors `build-tools/compiler/skill_passport.py::RevocationList`
//! and `RevocationEntry`, including the deterministic JSON signing
//! payload, ed25519 verification, expiry check, and `is_revoked`
//! / `lookup` lookup helpers.
//!
//! The list is itself authenticated by an ed25519 signature over a
//! deterministic JSON serialisation of every non-signature field
//! so a tampered revocation list cannot silently un-revoke a
//! known-bad pack.
//!
//! ### Why this lives on-device
//!
//! Revocation list checking — Python lines 279–302 in the
//! upstream reference — ports to Rust so a signed pack that has since been revoked
//! is rejected at load time on the device, not only at compile
//! time on the server. This module is the on-device portion of
//! that contract.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::crypto::canonical_json::{canonical_json_bytes, CanonicalJsonError};
use crate::crypto::ed25519::{Ed25519PublicKey, Ed25519Signature, Ed25519VerifyError};

/// Pinned ed25519 signature algorithm tag accepted by
/// [`RevocationList::verify_signature`].
///
/// Mirrors Python's `SIGNATURE_ALGORITHM = "ed25519"` constant.
pub const SIGNATURE_ALGORITHM: &str = "ed25519";

/// Signature envelope on a [`RevocationList`].
///
/// Mirrors Python's `Signature` dataclass field-for-field. Stored
/// as `algorithm` + `key_id` + base64-encoded `value` so the
/// envelope is self-describing on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature {
    pub algorithm: String,
    pub key_id: String,
    pub value: String,
}

/// One `(skill_id, skill_version)` revoked tuple.
///
/// Mirrors Python's `RevocationEntry` dataclass. `revoked_on` is
/// kept as an ISO-8601 date *string* (e.g. `"2024-06-30"`) rather
/// than a `chrono::NaiveDate` so the Rust crate doesn't pull in
/// `chrono` as a hard dependency. Hosts that need a typed date
/// can parse it at the boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevocationEntry {
    pub skill_id: String,
    pub skill_version: String,
    /// ISO-8601 date (`YYYY-MM-DD`).
    pub revoked_on: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub revoked_by: String,
}

/// Signed list of revoked passports.
///
/// Mirrors Python's `RevocationList` dataclass and provides every
/// method called by the revocation-check contract:
///
/// * [`Self::is_revoked`] / [`Self::lookup`] — `O(N)` scan over
///   `entries` (matching Python's behaviour exactly; the
///   reference implementation keeps lists short and trusts the
///   caller to keep them so).
/// * [`Self::signing_payload`] — deterministic, sorted-keys,
///   minified JSON of every non-signature field (the byte-stream
///   the ed25519 signature commits to).
/// * [`Self::verify_signature`] — pinned-key ed25519
///   verification + expiry check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevocationList {
    pub entries: Vec<RevocationEntry>,
    /// ISO-8601 date.
    pub issued_on: String,
    /// ISO-8601 date.
    pub expires_on: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<Signature>,
}

/// Errors surfaced by revocation-list / passport revocation
/// checks. Mirrors Python's `PassportValidationError` failure
/// modes one-for-one — every Python `raise PassportValidationError(...)`
/// site in `skill_passport.py` lines 279–302 (revocation block) and
/// in `RevocationList.verify_signature` is represented as a
/// distinct variant so cross-platform error routing stays
/// structured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassportValidationError {
    /// Either `(skill_id, skill_version)` matched an entry in the
    /// revocation list, or the list itself failed a precondition.
    PassportRevoked {
        skill_id: String,
        skill_version: String,
        reason: String,
    },
    RevocationListUnsigned,
    RevocationListUnsupportedAlgorithm(String),
    RevocationListSignatureMismatch,
    RevocationListExpired {
        expires_on: String,
    },
    InvalidIsoDate(String),
    BadSignaturePayload(String),
    CanonicalJson(String),
}

impl fmt::Display for PassportValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PassportRevoked {
                skill_id,
                skill_version,
                reason,
            } => {
                if reason.is_empty() {
                    write!(f, "passport revoked: {skill_id}@{skill_version}")
                } else {
                    write!(f, "passport revoked: {skill_id}@{skill_version} ({reason})")
                }
            }
            Self::RevocationListUnsigned => write!(f, "revocation list is unsigned"),
            Self::RevocationListUnsupportedAlgorithm(algo) => {
                write!(f, "unsupported signature algorithm: {algo}")
            }
            Self::RevocationListSignatureMismatch => {
                write!(f, "revocation list signature mismatch")
            }
            Self::RevocationListExpired { expires_on } => {
                write!(f, "revocation list expired on {expires_on}")
            }
            Self::InvalidIsoDate(s) => write!(f, "invalid ISO-8601 date: {s}"),
            Self::BadSignaturePayload(detail) => {
                write!(f, "bad revocation list signature payload: {detail}")
            }
            Self::CanonicalJson(detail) => {
                write!(f, "canonical JSON serialisation failed: {detail}")
            }
        }
    }
}

impl std::error::Error for PassportValidationError {}

impl From<CanonicalJsonError> for PassportValidationError {
    fn from(err: CanonicalJsonError) -> Self {
        Self::CanonicalJson(err.to_string())
    }
}

impl From<Ed25519VerifyError> for PassportValidationError {
    fn from(err: Ed25519VerifyError) -> Self {
        // `Ed25519VerifyError` is `#[non_exhaustive]` for *downstream*
        // crates but we match it exhaustively here because it lives
        // in the same crate (`crate::crypto::ed25519`). That means
        // adding a new variant in `ed25519.rs` will produce a
        // compile error here, forcing an explicit triage decision
        // — `BadSignaturePayload` (envelope / encoding problem)
        // vs. `RevocationListSignatureMismatch` (proof-level
        // failure) — rather than silently funnelling the new
        // failure mode into a generic wildcard. The previous
        // `_ => SignatureMismatch` arm masked exactly that
        // distinction.
        match err {
            Ed25519VerifyError::PublicKeyWrongLength { .. }
            | Ed25519VerifyError::SignatureWrongLength { .. }
            | Ed25519VerifyError::InvalidPublicKey
            | Ed25519VerifyError::InvalidSignature
            | Ed25519VerifyError::InvalidHex(_) => Self::BadSignaturePayload(err.to_string()),
            Ed25519VerifyError::VerificationFailed => Self::RevocationListSignatureMismatch,
        }
    }
}

impl RevocationList {
    /// Construct an unsigned list. Hosts that want to sign it
    /// should produce the signing payload via
    /// [`Self::signing_payload`] and attach a [`Signature`] via
    /// [`Self::with_signature`] (or set the field directly).
    pub fn new(
        entries: Vec<RevocationEntry>,
        issued_on: impl Into<String>,
        expires_on: impl Into<String>,
    ) -> Self {
        Self {
            entries,
            issued_on: issued_on.into(),
            expires_on: expires_on.into(),
            signature: None,
        }
    }

    /// Attach a signature envelope. Idempotent and `&mut self`-free
    /// to keep `RevocationList` ergonomically `Clone`able through
    /// a builder chain.
    pub fn with_signature(mut self, signature: Signature) -> Self {
        self.signature = Some(signature);
        self
    }

    /// `True` when an entry exists for `(skill_id, skill_version)`.
    /// Mirrors Python's `is_revoked` exactly.
    pub fn is_revoked(&self, skill_id: &str, skill_version: &str) -> bool {
        self.lookup(skill_id, skill_version).is_some()
    }

    /// Return the matching entry, or `None`. Mirrors Python's
    /// `lookup` exactly.
    pub fn lookup(&self, skill_id: &str, skill_version: &str) -> Option<&RevocationEntry> {
        self.entries
            .iter()
            .find(|e| e.skill_id == skill_id && e.skill_version == skill_version)
    }

    /// Return the deterministic JSON byte stream the ed25519
    /// signature commits to.
    ///
    /// Mirrors Python's `signing_payload` exactly: every field is
    /// emitted, the signature field is excluded, keys are sorted,
    /// and the separator pair is `(",",":")` — no whitespace.
    pub fn signing_payload(&self) -> Result<Vec<u8>, PassportValidationError> {
        // Build the same dict shape Python emits via `to_dict(...)`.
        let mut map: Map<String, Value> = Map::new();
        let mut entries_json: Vec<Value> = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            let mut e_map: Map<String, Value> = Map::new();
            e_map.insert(
                "skill_id".to_string(),
                Value::String(entry.skill_id.clone()),
            );
            e_map.insert(
                "skill_version".to_string(),
                Value::String(entry.skill_version.clone()),
            );
            e_map.insert(
                "revoked_on".to_string(),
                Value::String(entry.revoked_on.clone()),
            );
            e_map.insert("reason".to_string(), Value::String(entry.reason.clone()));
            e_map.insert(
                "revoked_by".to_string(),
                Value::String(entry.revoked_by.clone()),
            );
            entries_json.push(Value::Object(e_map));
        }
        map.insert("entries".to_string(), Value::Array(entries_json));
        map.insert(
            "issued_on".to_string(),
            Value::String(self.issued_on.clone()),
        );
        map.insert(
            "expires_on".to_string(),
            Value::String(self.expires_on.clone()),
        );
        Ok(canonical_json_bytes(&Value::Object(map))?)
    }

    /// Verify the list's own ed25519 signature and expiry. Mirrors
    /// Python's `verify_signature` byte-for-byte.
    ///
    /// `today` is the caller-supplied ISO-8601 date string for the
    /// expiry check. Passing `None` will trip
    /// [`PassportValidationError::InvalidIsoDate`] because the
    /// Rust crate intentionally does not pull in `chrono`; the
    /// caller is responsible for sourcing the current date in the
    /// timezone-correct way.
    pub fn verify_signature(
        &self,
        public_key: &Ed25519PublicKey,
        today: &str,
    ) -> Result<(), PassportValidationError> {
        let sig = self
            .signature
            .as_ref()
            .ok_or(PassportValidationError::RevocationListUnsigned)?;
        if sig.algorithm != SIGNATURE_ALGORITHM {
            return Err(PassportValidationError::RevocationListUnsupportedAlgorithm(
                sig.algorithm.clone(),
            ));
        }
        let sig_bytes = base64_decode(&sig.value).map_err(|e| {
            PassportValidationError::BadSignaturePayload(format!("base64 decode: {e}"))
        })?;
        let signature = Ed25519Signature::from_bytes(&sig_bytes)?;
        let payload = self.signing_payload()?;
        // Use `?` (which goes through `From<Ed25519VerifyError>`)
        // instead of `.map_err(|_| RevocationListSignatureMismatch)`.
        // The earlier `map_err` arm collapsed *every* verify failure —
        // including envelope-shape errors like `PublicKeyWrongLength`,
        // `SignatureWrongLength`, `InvalidPublicKey`, `InvalidSignature`,
        // and `InvalidHex` — onto the proof-level `SignatureMismatch`
        // variant, defeating the whole point of the exhaustive
        // `From<Ed25519VerifyError>` impl above (which routes envelope
        // errors to `BadSignaturePayload` and only the genuine
        // `VerificationFailed` to `RevocationListSignatureMismatch`).
        // Bubbling via `?` lets the `From` impl do its job and gives
        // operators the precise diagnostic they need.
        public_key.verify(&payload, &signature)?;
        if !is_iso_date(today) {
            return Err(PassportValidationError::InvalidIsoDate(today.to_string()));
        }
        if !is_iso_date(&self.expires_on) {
            return Err(PassportValidationError::InvalidIsoDate(
                self.expires_on.clone(),
            ));
        }
        if today > self.expires_on.as_str() {
            return Err(PassportValidationError::RevocationListExpired {
                expires_on: self.expires_on.clone(),
            });
        }
        Ok(())
    }

    /// Run the on-device revocation check for a `(skill_id,
    /// skill_version)`. Mirrors Python's `skill_passport.py:285-302`
    /// block exactly.
    ///
    /// The `verify` parameter is a [`Verify`] enum that pairs
    /// `public_key` and `today` so the caller cannot accidentally
    /// pass a public key without a date (which the previous
    /// `Option<&Ed25519PublicKey>` + `Option<&str>` signature
    /// silently turned into an `InvalidIsoDate("")` deep inside
    /// `verify_signature`).
    ///
    /// Semantics:
    /// * [`Verify::WithKey { key, today }`]: verify the list's
    ///   own signature + expiry first. A signature failure or an
    ///   expired list is rejected before any `is_revoked` check
    ///   can be trusted.
    /// * [`Verify::Skip`]: skip the list-level signature check
    ///   (the caller has already verified it out-of-band).
    /// * If `(skill_id, skill_version)` is in the list, return
    ///   [`PassportValidationError::PassportRevoked`].
    /// * Otherwise return `Ok(())`.
    pub fn check_passport(
        &self,
        skill_id: &str,
        skill_version: &str,
        verify: Verify<'_>,
    ) -> Result<(), PassportValidationError> {
        if let Verify::WithKey { key, today } = verify {
            self.verify_signature(key, today)?;
        }
        if let Some(entry) = self.lookup(skill_id, skill_version) {
            return Err(PassportValidationError::PassportRevoked {
                skill_id: skill_id.to_string(),
                skill_version: skill_version.to_string(),
                reason: entry.reason.clone(),
            });
        }
        Ok(())
    }
}

/// Verification mode for [`RevocationList::check_passport`].
///
/// Pairs `public_key` and `today` in a single value so the
/// caller cannot accidentally pass one without the other. The
/// older `Option<&Ed25519PublicKey>` + `Option<&str>` API
/// allowed the nonsensical combination `(Some(key), None)`,
/// which silently turned into an `InvalidIsoDate("")` deep
/// inside `verify_signature`; the enum makes the misuse
/// impossible at the type level.
#[derive(Debug, Clone, Copy)]
pub enum Verify<'a> {
    /// Skip the list-level signature check. The caller has
    /// already verified the list's authenticity out-of-band
    /// (or doesn't need to, e.g. for read-only diagnostics).
    Skip,
    /// Verify the list's signature against `key` and the expiry
    /// against `today`. `today` is an ISO-8601 `YYYY-MM-DD`
    /// string the caller is responsible for sourcing in the
    /// timezone-correct way (the crate intentionally does not
    /// pull in `chrono`).
    WithKey {
        key: &'a Ed25519PublicKey,
        today: &'a str,
    },
}

fn is_iso_date(s: &str) -> bool {
    // Strict ISO-8601 calendar-date validation matching Python's
    // `date.fromisoformat(...)` surface for the narrow `YYYY-MM-DD`
    // shape we accept. Covers:
    // * exact length 10
    // * hyphens at positions 4 and 7
    // * digits-only elsewhere
    // * month in 01..=12
    // * day in 01..=days_in_month(year, month)
    //   (including the Gregorian leap-year rule for February)
    //
    // The earlier byte-shape-only check accepted semantically
    // invalid strings like `"2024-99-99"`, which diverged from
    // Python's parity behaviour and let a host pass a corrupted
    // `today` past the expiry comparison. The full check is
    // still O(1) and dependency-free.
    if s.len() != 10 {
        return false;
    }
    let bytes = s.as_bytes();
    if bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    let digit_positions = [0, 1, 2, 3, 5, 6, 8, 9];
    if !digit_positions.iter().all(|&p| bytes[p].is_ascii_digit()) {
        return false;
    }
    let year = match s[0..4].parse::<u16>() {
        Ok(y) => y,
        Err(_) => return false,
    };
    let month = match s[5..7].parse::<u8>() {
        Ok(m) if (1..=12).contains(&m) => m,
        _ => return false,
    };
    let day = match s[8..10].parse::<u8>() {
        Ok(d) if d >= 1 => d,
        _ => return false,
    };
    day <= days_in_month(year, month)
}

/// Days in a Gregorian calendar month. Implements the same
/// rule Python's `calendar.monthrange` returns: 28/29 for
/// February (leap-year aware), 30 for Apr/Jun/Sep/Nov, 31
/// otherwise. Caller must have already validated
/// `month ∈ 1..=12`.
fn days_in_month(year: u16, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            // Gregorian leap-year: divisible by 4, except
            // century years not divisible by 400.
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            if leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Minimal RFC-4648 §4 ("base64", **standard alphabet only**)
/// decoder used by [`RevocationList::verify_signature`]. We
/// avoid pulling in `base64` as a new dep for one call site —
/// `canonical_json` already commits us to a couple of byte-level
/// utilities like this one.
///
/// Accepts:
/// * the standard alphabet `A–Z a–z 0–9 + /`
/// * trailing `=` padding (1 or 2)
/// * embedded `\r` / `\n` whitespace (PEM-style line wraps)
///
/// **Explicitly rejects** RFC-4648 §5 ("base64url", with `-`
/// and `_` in place of `+` and `/`). A URL-safe-encoded payload
/// is reported with a dedicated diagnostic so a future caller
/// that confuses the two alphabets gets a self-describing
/// failure instead of silently-corrupted bytes. Python's
/// `base64.b64decode` shows the same behaviour; the parity is
/// intentional.
fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    let table: BTreeMap<char, u8> = (b'A'..=b'Z')
        .chain(b'a'..=b'z')
        .chain(b'0'..=b'9')
        .map(|b| b as char)
        .chain(['+', '/'].iter().copied())
        .enumerate()
        .map(|(i, c)| (c, i as u8))
        .collect();
    let cleaned: String = input.chars().filter(|c| *c != '\n' && *c != '\r').collect();
    if cleaned.is_empty() {
        return Ok(Vec::new());
    }
    let mut bytes = Vec::with_capacity(cleaned.len() * 3 / 4);
    // Sliding-window accumulator: `buf` holds the *unconsumed*
    // bits from the prior sextet(s); `bits` is its current
    // width. After each push we mask `buf` down to those `bits`
    // remaining low-order bits so the accumulator can never
    // hold stale high-order bits between iterations.
    //
    // The maximum window width is small — after a fresh sextet
    // is shifted in (`bits + 6`) the inner branch immediately
    // pushes a byte and drops `bits` back to `< 8`, so `buf`
    // never carries more than 13 bits of live state. Even
    // without the mask the u32 width is far from overflow; the
    // explicit `buf &= (1 << bits) - 1` is for *readability*
    // — it makes the "low `bits` bits are the residual" invariant
    // obvious to anyone reading the loop, instead of leaving a
    // reviewer to derive the bound from the surrounding shifts.
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for c in cleaned.chars() {
        if c == '=' {
            break;
        }
        if c == '-' || c == '_' {
            return Err(format!(
                "url-safe base64 alphabet not supported (got {c:?}); pass the RFC-4648 standard alphabet (`+`/`/`) instead"
            ));
        }
        let v = *table
            .get(&c)
            .ok_or_else(|| format!("invalid base64 character: {c:?}"))?;
        buf = (buf << 6) | (v as u32);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push(((buf >> bits) & 0xFF) as u8);
            // Drop the byte we just emitted; keep only the
            // `bits` low-order bits as the residual for the
            // next sextet.
            buf &= (1u32 << bits).wrapping_sub(1);
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::ed25519::Ed25519PublicKey;

    fn entry(skill_id: &str, skill_version: &str, reason: &str) -> RevocationEntry {
        RevocationEntry {
            skill_id: skill_id.to_string(),
            skill_version: skill_version.to_string(),
            revoked_on: "2024-06-30".to_string(),
            reason: reason.to_string(),
            revoked_by: "ops".to_string(),
        }
    }

    fn unsigned_list() -> RevocationList {
        RevocationList::new(
            vec![entry("kchat.global.baseline", "1.0.0", "safety-regression")],
            "2024-06-01",
            "2026-06-01",
        )
    }

    #[test]
    fn is_revoked_finds_exact_match() {
        let list = unsigned_list();
        assert!(list.is_revoked("kchat.global.baseline", "1.0.0"));
    }

    #[test]
    fn is_revoked_distinguishes_version() {
        let list = unsigned_list();
        assert!(!list.is_revoked("kchat.global.baseline", "1.0.1"));
    }

    #[test]
    fn is_revoked_distinguishes_skill_id() {
        let list = unsigned_list();
        assert!(!list.is_revoked("kchat.global.other", "1.0.0"));
    }

    #[test]
    fn lookup_returns_entry() {
        let list = unsigned_list();
        let found = list
            .lookup("kchat.global.baseline", "1.0.0")
            .expect("entry exists");
        assert_eq!(found.reason, "safety-regression");
    }

    #[test]
    fn signing_payload_is_deterministic() {
        let list = unsigned_list();
        let a = list.signing_payload().expect("payload a");
        let b = list.signing_payload().expect("payload b");
        assert_eq!(a, b);
    }

    #[test]
    fn signing_payload_always_includes_reason_and_revoked_by_even_when_empty() {
        // Parity guard: Python's `RevocationEntry.to_dict` emits
        // `reason` and `revoked_by` unconditionally (they are
        // required `str` fields on the dataclass — no default), so
        // the Rust signing_payload MUST also emit them as empty
        // strings when absent. If a future refactor introduces
        // `skip_serializing_if = Option::is_none`-style omission
        // on those fields, cross-platform ed25519 signature
        // verification will silently break: the Python signer
        // will sign `..."reason":""...` and the Rust verifier
        // will hash a payload that lacks the key entirely.
        let entry = RevocationEntry {
            skill_id: "kchat.global.x".to_string(),
            skill_version: "1.0.0".to_string(),
            revoked_on: "2024-06-30".to_string(),
            reason: String::new(),
            revoked_by: String::new(),
        };
        let list = RevocationList::new(vec![entry], "2024-06-01", "2026-06-01");
        let payload = list.signing_payload().expect("payload");
        let s = std::str::from_utf8(&payload).expect("utf8");
        assert!(
            s.contains(r#""reason":"""#),
            "signing payload must include empty `reason` field: {s}"
        );
        assert!(
            s.contains(r#""revoked_by":"""#),
            "signing payload must include empty `revoked_by` field: {s}"
        );
    }

    #[test]
    fn signing_payload_excludes_signature() {
        let mut list = unsigned_list();
        let before = list.signing_payload().expect("before");
        list.signature = Some(Signature {
            algorithm: "ed25519".to_string(),
            key_id: "k".to_string(),
            value: "AAAA".to_string(),
        });
        let after = list.signing_payload().expect("after");
        assert_eq!(before, after, "signature field must not enter payload");
    }

    #[test]
    fn unsigned_list_verify_signature_rejected() {
        let list = unsigned_list();
        let pk_bytes = [0u8; 32];
        let pk = Ed25519PublicKey::from_bytes(&pk_bytes).expect("zero key ok for our error path");
        assert!(matches!(
            list.verify_signature(&pk, "2024-07-01"),
            Err(PassportValidationError::RevocationListUnsigned)
        ));
    }

    #[test]
    fn check_passport_no_match_skip_ok() {
        let list = unsigned_list();
        assert!(list
            .check_passport("kchat.global.other", "1.0.0", Verify::Skip)
            .is_ok());
    }

    #[test]
    fn check_passport_match_skip_rejects() {
        let list = unsigned_list();
        let err = list
            .check_passport("kchat.global.baseline", "1.0.0", Verify::Skip)
            .unwrap_err();
        assert!(matches!(
            err,
            PassportValidationError::PassportRevoked { .. }
        ));
        let msg = err.to_string();
        assert!(msg.contains("kchat.global.baseline@1.0.0"));
        assert!(msg.contains("safety-regression"));
    }

    #[test]
    fn check_passport_with_key_requires_signature() {
        let list = unsigned_list();
        let pk_bytes = [0u8; 32];
        let pk = Ed25519PublicKey::from_bytes(&pk_bytes).expect("zero key ok for error path");
        // Even on a clean (non-revoked) ID, the unsigned list must
        // be rejected so a forged-unsigned list can't pretend to be
        // a fresh empty list.
        let err = list
            .check_passport(
                "kchat.global.other",
                "1.0.0",
                Verify::WithKey {
                    key: &pk,
                    today: "2024-07-01",
                },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            PassportValidationError::RevocationListUnsigned
        ));
    }

    /// Base-64 encoding of 64 zero bytes — the canonical shape
    /// of a (fake) ed25519 signature value used to exercise the
    /// "valid envelope, wrong key" path without depending on a
    /// real signing key.
    const ZERO_SIG_64_B64: &str =
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";

    #[test]
    fn expired_list_rejected_when_verifying_with_pubkey() {
        // Build a list whose signature envelope is parseable but
        // whose signature value cannot verify under the zero key.
        // The contract is "signature must succeed before expiry
        // is even considered", so we expect a signature-mismatch
        // failure for the zero-byte fake.
        let mut list = unsigned_list();
        list.expires_on = "2024-01-01".to_string();
        list.signature = Some(Signature {
            algorithm: "ed25519".to_string(),
            key_id: "k".to_string(),
            value: ZERO_SIG_64_B64.to_string(),
        });
        let pk = Ed25519PublicKey::from_bytes(&[0u8; 32]).expect("ok");
        let err = list.verify_signature(&pk, "2024-07-01").unwrap_err();
        assert!(matches!(
            err,
            PassportValidationError::RevocationListSignatureMismatch
        ));
    }

    #[test]
    fn unsupported_algorithm_rejected() {
        let mut list = unsigned_list();
        list.signature = Some(Signature {
            algorithm: "rsa-pss".to_string(),
            key_id: "k".to_string(),
            value: "AAAA".to_string(),
        });
        let pk = Ed25519PublicKey::from_bytes(&[0u8; 32]).expect("ok");
        let err = list.verify_signature(&pk, "2024-07-01").unwrap_err();
        assert!(matches!(
            err,
            PassportValidationError::RevocationListUnsupportedAlgorithm(ref a)
                if a == "rsa-pss"
        ));
    }

    #[test]
    fn invalid_today_iso_date_rejected_or_signature_fails_first() {
        // The verify path checks signature FIRST, so a garbage
        // date string only surfaces when a real signature would
        // otherwise verify. With the zero-key fake we still hit
        // RevocationListSignatureMismatch first. The date helper
        // itself is asserted independently below to ensure the
        // contract is exercised.
        let mut list = unsigned_list();
        list.signature = Some(Signature {
            algorithm: "ed25519".to_string(),
            key_id: "k".to_string(),
            value: ZERO_SIG_64_B64.to_string(),
        });
        let pk = Ed25519PublicKey::from_bytes(&[0u8; 32]).expect("ok");
        let err = list.verify_signature(&pk, "garbage").unwrap_err();
        assert!(!is_iso_date("garbage"));
        assert!(matches!(
            err,
            PassportValidationError::RevocationListSignatureMismatch
        ));
    }

    #[test]
    fn base64_decode_basic() {
        // "abc" base64-encoded is "YWJj".
        assert_eq!(base64_decode("YWJj").unwrap(), b"abc".to_vec());
        // 64 zero bytes encodes to 88 chars including `==`
        // padding; the helper honours the padding by stopping
        // at the first `=`.
        let zeros = vec![0u8; 64];
        assert_eq!(base64_decode(ZERO_SIG_64_B64).unwrap(), zeros);
    }

    #[test]
    fn iso_date_helper_accepts_correct_shape() {
        assert!(is_iso_date("2024-06-30"));
        assert!(is_iso_date("1999-01-01"));
    }

    #[test]
    fn iso_date_helper_rejects_drift() {
        assert!(!is_iso_date(""));
        assert!(!is_iso_date("2024-6-30"));
        assert!(!is_iso_date("2024/06/30"));
        assert!(!is_iso_date("2024-06-30T00:00:00Z"));
        assert!(!is_iso_date("yyyy-mm-dd"));
    }

    #[test]
    fn iso_date_helper_rejects_invalid_month_or_day() {
        // Month / day range checks, matching Python's
        // `date.fromisoformat`.
        assert!(!is_iso_date("2024-00-15"), "month 00 must be rejected");
        assert!(!is_iso_date("2024-13-15"), "month 13 must be rejected");
        assert!(!is_iso_date("2024-99-99"), "fully out-of-range");
        assert!(!is_iso_date("2024-06-00"), "day 00 must be rejected");
        assert!(!is_iso_date("2024-06-31"), "June has 30 days");
        assert!(!is_iso_date("2024-04-31"), "April has 30 days");
        assert!(!is_iso_date("2023-02-29"), "2023 was not a leap year");
    }

    #[test]
    fn iso_date_helper_handles_leap_year_correctly() {
        // 2024 is divisible by 4, not by 100 -> leap.
        assert!(is_iso_date("2024-02-29"));
        // 2100 is divisible by 100, not by 400 -> NOT leap.
        assert!(!is_iso_date("2100-02-29"));
        // 2000 is divisible by 400 -> leap.
        assert!(is_iso_date("2000-02-29"));
        // Feb 30 never exists.
        assert!(!is_iso_date("2024-02-30"));
    }

    #[test]
    fn base64_decode_rejects_url_safe_alphabet() {
        // `-` and `_` are the RFC-4648 §5 ("base64url") alphabet
        // and must be rejected by our standard-alphabet decoder
        // with a self-describing diagnostic, not silently
        // misdecoded.
        let err_dash = base64_decode("ab-d").unwrap_err();
        assert!(
            err_dash.contains("url-safe base64 alphabet not supported"),
            "expected url-safe diagnostic, got: {err_dash}"
        );
        let err_underscore = base64_decode("ab_d").unwrap_err();
        assert!(
            err_underscore.contains("url-safe base64 alphabet not supported"),
            "expected url-safe diagnostic, got: {err_underscore}"
        );
    }

    // Ensure signature struct is round-trippable through JSON,
    // and serde-aware so a host can persist + reload a
    // RevocationList.
    #[test]
    fn signature_json_round_trip() {
        let sig = Signature {
            algorithm: "ed25519".to_string(),
            key_id: "key-1".to_string(),
            value: "AAAA".to_string(),
        };
        let s = serde_json::to_string(&sig).expect("serialize");
        let back: Signature = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(sig, back);
    }

    #[test]
    fn revocation_list_json_round_trip() {
        let list = unsigned_list();
        let s = serde_json::to_string(&list).expect("serialize");
        let back: RevocationList = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(list, back);
    }
}

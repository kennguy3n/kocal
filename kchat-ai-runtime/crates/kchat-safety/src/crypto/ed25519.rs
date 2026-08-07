//! Strict (RFC 8032) Ed25519 signature verification.
//!
//! The on-device runtime is a *verifier only* — every helper here
//! accepts a pinned 32-byte public key and a 64-byte signature
//! over a caller-provided message, returns `Ok(())` on a valid
//! signature, and a typed [`Ed25519VerifyError`] otherwise. There
//! are no signing entry points and the underlying `ed25519-dalek`
//! crate is wired with `default-features = false` so the private-
//! key code path is not even compiled into the binary.
//!
//! ### Why `verify_strict` (not `verify`)?
//!
//! `ed25519-dalek` exposes two verification entry points:
//!
//! * [`VerifyingKey::verify`] — accepts every signature that
//!   passes the RFC 8032 algebraic check, including signatures
//!   constructed using the legacy "non-canonical scalar" trick
//!   and signatures against small-order public keys.
//! * [`VerifyingKey::verify_strict`] — additionally rejects:
//!   * non-canonical encodings of the `R` and `S` scalars
//!     (signatures Tonge-malleable),
//!   * small-order public keys (keys whose verification would
//!     succeed against more than one signature for the same
//!     message — a curve-order attack surface),
//!   * negative `s` components (signatures outside `[0, ℓ)`).
//!
//! The Python reference (`cv-guard/shared/skillpack/verifier.py`
//! and `cv-guard/shared/manifest/manifest_verifier.py`) uses
//! `cryptography.hazmat.primitives.asymmetric.ed25519.
//! Ed25519PublicKey.verify`, which under the hood calls OpenSSL's
//! `EVP_DigestVerify` against the modern EVP API. OpenSSL's
//! Ed25519 implementation also rejects non-canonical encodings
//! and small-order keys, so `verify_strict` is the Rust-side
//! analogue that preserves cross-platform reject-set parity.
//! Anything `verify` accepts that `verify_strict` rejects is a
//! signature the Python reference would also reject, so using
//! `verify_strict` here closes the parity gap rather than
//! widening the accept set.
//!
//! ### Constant-time guarantees
//!
//! `ed25519-dalek` runs the algebraic verification in constant
//! time relative to the signature bits — there is no early exit
//! on partial-mismatch. Any timing comparison the caller might
//! want to layer on top (e.g. comparing two hex-encoded
//! signatures) must use [`subtle::ConstantTimeEq`] explicitly;
//! [`Ed25519Signature`] / [`Ed25519PublicKey`] do **not**
//! implement `Eq` or `PartialEq` to avoid accidentally leaking
//! comparison timing into application code. Use the dedicated
//! [`Ed25519PublicKey::constant_time_eq`] helper instead.
//!
//! ### Cross-platform contract
//!
//! Every [`verify_signature_hex`] call is paired with a Python
//! `cryptography.hazmat.primitives.asymmetric.ed25519.
//! Ed25519PublicKey(pub_hex).verify(sig_bytes, msg)` call in the
//! fixture oracle (`tools/gen_crypto_fixtures.py`). The replay
//! test (`tests/crypto_parity.rs`) covers the RFC 8032 test
//! vectors plus skill-pack / passport / manifest signature
//! triples generated against ephemeral keys, so any drift
//! between the on-device verifier and the server-side signer is
//! caught at CI time.

use std::fmt;

use ed25519_dalek::{
    Signature as DalekSignature, SignatureError as DalekSignatureError, VerifyingKey,
    PUBLIC_KEY_LENGTH, SIGNATURE_LENGTH,
};
use subtle::ConstantTimeEq;

/// Length of an Ed25519 public key in raw bytes (32).
pub const ED25519_PUBLIC_KEY_LEN: usize = PUBLIC_KEY_LENGTH;
/// Length of an Ed25519 signature in raw bytes (64).
pub const ED25519_SIGNATURE_LEN: usize = SIGNATURE_LENGTH;

/// Errors surfaced by the verifier.
///
/// `#[non_exhaustive]` so adding a variant is not a breaking
/// change for downstream `match` sites.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Ed25519VerifyError {
    /// Public key bytes were not exactly 32 bytes long.
    PublicKeyWrongLength {
        /// Number of bytes provided.
        actual: usize,
    },
    /// Signature bytes were not exactly 64 bytes long.
    SignatureWrongLength {
        /// Number of bytes provided.
        actual: usize,
    },
    /// Public key bytes failed canonicality / point-on-curve
    /// checks. This is the `dalek` `VerifyingKey::from_bytes`
    /// failure path — e.g. encoding of a point that is not on
    /// the Edwards 25519 curve.
    InvalidPublicKey,
    /// Signature bytes failed canonicality / encoding checks.
    /// This is the `dalek` `Signature::from_bytes` failure path
    /// — e.g. an `S` scalar outside `[0, ℓ)` after the
    /// `verify_strict` extra checks.
    InvalidSignature,
    /// Public key + signature parsed cleanly but the strict
    /// RFC-8032 verification did not pass under the supplied
    /// public key and message.
    VerificationFailed,
    /// Hex-encoded input contained a non-hex character or had
    /// an odd number of hex digits.
    InvalidHex(String),
}

impl fmt::Display for Ed25519VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PublicKeyWrongLength { actual } => write!(
                f,
                "ed25519 public key must be exactly {ED25519_PUBLIC_KEY_LEN} bytes (got {actual})"
            ),
            Self::SignatureWrongLength { actual } => write!(
                f,
                "ed25519 signature must be exactly {ED25519_SIGNATURE_LEN} bytes (got {actual})"
            ),
            Self::InvalidPublicKey => f.write_str("ed25519 public key is not on the curve"),
            Self::InvalidSignature => f.write_str("ed25519 signature is not canonically encoded"),
            Self::VerificationFailed => f.write_str("ed25519 signature verification failed"),
            Self::InvalidHex(reason) => write!(f, "invalid hex encoding: {reason}"),
        }
    }
}

impl std::error::Error for Ed25519VerifyError {}

/// 32-byte pinned Ed25519 public key.
///
/// Construction parses the raw bytes and runs the `dalek` point-
/// on-curve check, so the resulting value is guaranteed to be a
/// valid Edwards 25519 point. The struct does NOT implement `Eq`
/// / `PartialEq` — use [`Ed25519PublicKey::constant_time_eq`] for
/// timing-safe equality.
#[derive(Clone)]
pub struct Ed25519PublicKey {
    inner: VerifyingKey,
    raw: [u8; ED25519_PUBLIC_KEY_LEN],
}

impl Ed25519PublicKey {
    /// Parse a raw 32-byte Ed25519 public key.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Ed25519VerifyError> {
        if bytes.len() != ED25519_PUBLIC_KEY_LEN {
            return Err(Ed25519VerifyError::PublicKeyWrongLength {
                actual: bytes.len(),
            });
        }
        let mut arr = [0u8; ED25519_PUBLIC_KEY_LEN];
        arr.copy_from_slice(bytes);
        let inner = VerifyingKey::from_bytes(&arr)
            .map_err(|_: DalekSignatureError| Ed25519VerifyError::InvalidPublicKey)?;
        Ok(Self { inner, raw: arr })
    }

    /// Parse a 64-character lowercase-hex Ed25519 public key.
    ///
    /// This is the encoding the Python skill-pack compiler emits
    /// for the manifest's `public_key` field.
    pub fn from_hex(s: &str) -> Result<Self, Ed25519VerifyError> {
        let bytes = decode_hex(s)?;
        Self::from_bytes(&bytes)
    }

    /// Hand back the raw 32-byte encoding.
    ///
    /// Returned by value (not by reference) so callers cannot
    /// mutate the internal state and end up with an
    /// `Ed25519PublicKey` whose `raw` no longer matches its
    /// parsed `inner`.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; ED25519_PUBLIC_KEY_LEN] {
        self.raw
    }

    /// Hex-encode the raw key (64 lowercase hex chars).
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.raw)
    }

    /// Constant-time equality against another public key. Use
    /// this rather than `==` so a side-channel cannot leak which
    /// byte mismatched first.
    #[must_use]
    pub fn constant_time_eq(&self, other: &Ed25519PublicKey) -> bool {
        self.raw.ct_eq(&other.raw).into()
    }

    /// Constant-time equality against a raw 32-byte slice.
    /// Returns `false` if the slice is the wrong length.
    #[must_use]
    pub fn constant_time_eq_bytes(&self, other: &[u8]) -> bool {
        if other.len() != ED25519_PUBLIC_KEY_LEN {
            return false;
        }
        self.raw.ct_eq(other).into()
    }

    /// Constant-time equality against a hex-encoded 64-char
    /// string. Returns `false` for any non-hex / wrong-length
    /// input.
    #[must_use]
    pub fn constant_time_eq_hex(&self, other_hex: &str) -> bool {
        let Ok(bytes) = decode_hex(other_hex) else {
            return false;
        };
        self.constant_time_eq_bytes(&bytes)
    }

    /// Verify a 64-byte signature over `message`. Runs the
    /// `ed25519-dalek::VerifyingKey::verify_strict` check.
    pub fn verify(
        &self,
        message: &[u8],
        signature: &Ed25519Signature,
    ) -> Result<(), Ed25519VerifyError> {
        self.inner
            .verify_strict(message, &signature.inner)
            .map_err(|_: DalekSignatureError| Ed25519VerifyError::VerificationFailed)
    }
}

impl fmt::Debug for Ed25519PublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Public keys are not secret — printing the hex
        // encoding is fine and useful for diagnostics. Kept as
        // a manual impl so a future refactor that switches the
        // representation does not silently change the Debug
        // shape.
        f.debug_struct("Ed25519PublicKey")
            .field("hex", &self.to_hex())
            .finish()
    }
}

/// 64-byte Ed25519 signature.
///
/// Construction runs the `ed25519-dalek::Signature::from_bytes`
/// shape check — the value is guaranteed to be a well-formed
/// encoding of an `(R, S)` pair, but no curve-level checks happen
/// until the signature is paired with a public key in
/// [`Ed25519PublicKey::verify`].
#[derive(Clone)]
pub struct Ed25519Signature {
    inner: DalekSignature,
    raw: [u8; ED25519_SIGNATURE_LEN],
}

impl Ed25519Signature {
    /// Parse a raw 64-byte signature.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Ed25519VerifyError> {
        if bytes.len() != ED25519_SIGNATURE_LEN {
            return Err(Ed25519VerifyError::SignatureWrongLength {
                actual: bytes.len(),
            });
        }
        let mut arr = [0u8; ED25519_SIGNATURE_LEN];
        arr.copy_from_slice(bytes);
        // `Signature::from_bytes` is infallible on a 64-byte
        // slice — it only enforces shape, not the strict-form
        // S-scalar reduction. The strict-form reject happens in
        // `verify_strict`. Kept the `Result` shape here to keep
        // a clean error path in case the dalek API tightens in
        // a future major version.
        let inner = DalekSignature::from_bytes(&arr);
        Ok(Self { inner, raw: arr })
    }

    /// Parse a 128-character lowercase-hex signature. This is
    /// the encoding the Python skill-pack compiler emits for
    /// the manifest's `signature` field.
    pub fn from_hex(s: &str) -> Result<Self, Ed25519VerifyError> {
        let bytes = decode_hex(s)?;
        Self::from_bytes(&bytes)
    }

    /// Raw 64-byte signature.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; ED25519_SIGNATURE_LEN] {
        self.raw
    }

    /// 128 lowercase-hex chars.
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.raw)
    }
}

impl fmt::Debug for Ed25519Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ed25519Signature")
            .field("hex", &self.to_hex())
            .finish()
    }
}

/// Verify a signature against a pinned public key and message.
///
/// This is the lowest-level convenience entry point — it parses
/// the public key + signature inline and applies strict RFC-8032
/// verification. Higher-level helpers ([`verify_signature_hex`])
/// thin-wrap this with hex decoding for the manifest path.
pub fn verify_signature(
    public_key_bytes: &[u8],
    message: &[u8],
    signature_bytes: &[u8],
) -> Result<(), Ed25519VerifyError> {
    let key = Ed25519PublicKey::from_bytes(public_key_bytes)?;
    let sig = Ed25519Signature::from_bytes(signature_bytes)?;
    key.verify(message, &sig)
}

/// Verify a hex-encoded signature against a hex-encoded public
/// key and a byte message.
///
/// This is the entry point the skill-pack / overlay / manifest
/// verifiers will call — all three artefacts carry their public
/// key and signature as lowercase hex in the manifest JSON, and
/// the on-device runtime decodes them once at load time.
pub fn verify_signature_hex(
    public_key_hex: &str,
    message: &[u8],
    signature_hex: &str,
) -> Result<(), Ed25519VerifyError> {
    let key = Ed25519PublicKey::from_hex(public_key_hex)?;
    let sig = Ed25519Signature::from_hex(signature_hex)?;
    key.verify(message, &sig)
}

fn decode_hex(s: &str) -> Result<Vec<u8>, Ed25519VerifyError> {
    hex::decode(s).map_err(|e| Ed25519VerifyError::InvalidHex(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // RFC 8032 §7.1 test vectors — pinned by the parity oracle, replayed
    // here as in-module tests so the cargo build alone exercises the
    // verifier without needing the parity fixture file.
    // ------------------------------------------------------------------

    /// RFC 8032 §7.1 test vector 1: empty message.
    const RFC8032_TV1_PUBLIC_KEY: &str =
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
    const RFC8032_TV1_MESSAGE: &[u8] = b"";
    const RFC8032_TV1_SIGNATURE: &str =
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b";

    /// RFC 8032 §7.1 test vector 2: 1-byte message.
    const RFC8032_TV2_PUBLIC_KEY: &str =
        "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c";
    const RFC8032_TV2_MESSAGE: &[u8] = &[0x72];
    const RFC8032_TV2_SIGNATURE: &str =
        "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00";

    /// RFC 8032 §7.1 test vector 3: 2-byte message.
    const RFC8032_TV3_PUBLIC_KEY: &str =
        "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025";
    const RFC8032_TV3_MESSAGE: &[u8] = &[0xaf, 0x82];
    const RFC8032_TV3_SIGNATURE: &str =
        "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a";

    #[test]
    fn rfc8032_vector_1_verifies() {
        verify_signature_hex(
            RFC8032_TV1_PUBLIC_KEY,
            RFC8032_TV1_MESSAGE,
            RFC8032_TV1_SIGNATURE,
        )
        .expect("RFC 8032 TV1 must verify");
    }

    #[test]
    fn rfc8032_vector_2_verifies() {
        verify_signature_hex(
            RFC8032_TV2_PUBLIC_KEY,
            RFC8032_TV2_MESSAGE,
            RFC8032_TV2_SIGNATURE,
        )
        .expect("RFC 8032 TV2 must verify");
    }

    #[test]
    fn rfc8032_vector_3_verifies() {
        verify_signature_hex(
            RFC8032_TV3_PUBLIC_KEY,
            RFC8032_TV3_MESSAGE,
            RFC8032_TV3_SIGNATURE,
        )
        .expect("RFC 8032 TV3 must verify");
    }

    #[test]
    fn rfc8032_vector_1_rejects_wrong_message() {
        let err = verify_signature_hex(RFC8032_TV1_PUBLIC_KEY, b"tampered", RFC8032_TV1_SIGNATURE)
            .unwrap_err();
        assert_eq!(err, Ed25519VerifyError::VerificationFailed);
    }

    #[test]
    fn rfc8032_vector_1_rejects_wrong_key() {
        let err = verify_signature_hex(
            RFC8032_TV2_PUBLIC_KEY,
            RFC8032_TV1_MESSAGE,
            RFC8032_TV1_SIGNATURE,
        )
        .unwrap_err();
        assert_eq!(err, Ed25519VerifyError::VerificationFailed);
    }

    #[test]
    fn rfc8032_vector_1_rejects_flipped_signature_bit() {
        // Flip the first bit of the signature's R component.
        let mut sig_bytes = hex::decode(RFC8032_TV1_SIGNATURE).unwrap();
        sig_bytes[0] ^= 0x01;
        let tampered_hex = hex::encode(&sig_bytes);
        let err = verify_signature_hex(RFC8032_TV1_PUBLIC_KEY, RFC8032_TV1_MESSAGE, &tampered_hex)
            .unwrap_err();
        // Flipping a bit lands us in one of two failure modes
        // depending on whether the new bytes happen to form a
        // valid signature shape — both are rejects.
        assert!(
            matches!(
                err,
                Ed25519VerifyError::VerificationFailed | Ed25519VerifyError::InvalidSignature
            ),
            "expected VerificationFailed or InvalidSignature, got {err:?}"
        );
    }

    // ------------------------------------------------------------------
    // Length / encoding validators.
    // ------------------------------------------------------------------

    #[test]
    fn public_key_wrong_length_is_typed_error() {
        let err = Ed25519PublicKey::from_bytes(&[0u8; 31]).unwrap_err();
        assert_eq!(err, Ed25519VerifyError::PublicKeyWrongLength { actual: 31 });
        let err = Ed25519PublicKey::from_bytes(&[0u8; 33]).unwrap_err();
        assert_eq!(err, Ed25519VerifyError::PublicKeyWrongLength { actual: 33 });
        let err = Ed25519PublicKey::from_bytes(&[]).unwrap_err();
        assert_eq!(err, Ed25519VerifyError::PublicKeyWrongLength { actual: 0 });
    }

    #[test]
    fn signature_wrong_length_is_typed_error() {
        let err = Ed25519Signature::from_bytes(&[0u8; 63]).unwrap_err();
        assert_eq!(err, Ed25519VerifyError::SignatureWrongLength { actual: 63 });
        let err = Ed25519Signature::from_bytes(&[0u8; 65]).unwrap_err();
        assert_eq!(err, Ed25519VerifyError::SignatureWrongLength { actual: 65 });
        let err = Ed25519Signature::from_bytes(&[]).unwrap_err();
        assert_eq!(err, Ed25519VerifyError::SignatureWrongLength { actual: 0 });
    }

    #[test]
    fn public_key_invalid_hex_is_typed_error() {
        let err = Ed25519PublicKey::from_hex("nothex").unwrap_err();
        assert!(matches!(err, Ed25519VerifyError::InvalidHex(_)));
        // Odd-length hex.
        let err = Ed25519PublicKey::from_hex("ab0").unwrap_err();
        assert!(matches!(err, Ed25519VerifyError::InvalidHex(_)));
        // Right length but non-hex chars.
        let bad: String = "zz".repeat(32);
        let err = Ed25519PublicKey::from_hex(&bad).unwrap_err();
        assert!(matches!(err, Ed25519VerifyError::InvalidHex(_)));
    }

    #[test]
    fn signature_invalid_hex_is_typed_error() {
        let err = Ed25519Signature::from_hex("zz").unwrap_err();
        assert!(matches!(err, Ed25519VerifyError::InvalidHex(_)));
    }

    #[test]
    fn malformed_public_key_bytes_are_typed_error() {
        // y = 2 (little-endian, all zeros except byte 0) is not
        // a valid Edwards point — there is no x in the field such
        // that `x² = (y² - 1) / (d·y² + 1)`. `ed25519-dalek`'s
        // `VerifyingKey::from_bytes` correctly rejects this and we
        // re-export the failure as a typed [`Ed25519VerifyError::
        // InvalidPublicKey`]. (Many small y values land on the
        // curve — y = 2, 7, 8, 11, 12, 13 are among the first
        // off-curve values reachable by toggling a single byte.)
        let mut not_on_curve = [0u8; 32];
        not_on_curve[0] = 2;
        let err = Ed25519PublicKey::from_bytes(&not_on_curve).unwrap_err();
        assert_eq!(err, Ed25519VerifyError::InvalidPublicKey);

        // Same input through the hex entry point produces the
        // same typed error.
        let err_hex = Ed25519PublicKey::from_hex(&hex::encode(not_on_curve)).unwrap_err();
        assert_eq!(err_hex, Ed25519VerifyError::InvalidPublicKey);
    }

    // ------------------------------------------------------------------
    // Constant-time equality helpers.
    // ------------------------------------------------------------------

    #[test]
    fn constant_time_eq_matches_for_equal_keys() {
        let k1 = Ed25519PublicKey::from_hex(RFC8032_TV1_PUBLIC_KEY).unwrap();
        let k2 = Ed25519PublicKey::from_hex(RFC8032_TV1_PUBLIC_KEY).unwrap();
        assert!(k1.constant_time_eq(&k2));
        assert!(k1.constant_time_eq_hex(RFC8032_TV1_PUBLIC_KEY));
        let raw = hex::decode(RFC8032_TV1_PUBLIC_KEY).unwrap();
        assert!(k1.constant_time_eq_bytes(&raw));
    }

    #[test]
    fn constant_time_eq_rejects_unequal_keys() {
        let k1 = Ed25519PublicKey::from_hex(RFC8032_TV1_PUBLIC_KEY).unwrap();
        let k2 = Ed25519PublicKey::from_hex(RFC8032_TV2_PUBLIC_KEY).unwrap();
        assert!(!k1.constant_time_eq(&k2));
        assert!(!k1.constant_time_eq_hex(RFC8032_TV2_PUBLIC_KEY));
    }

    #[test]
    fn constant_time_eq_rejects_wrong_length_bytes() {
        let k1 = Ed25519PublicKey::from_hex(RFC8032_TV1_PUBLIC_KEY).unwrap();
        assert!(!k1.constant_time_eq_bytes(&[0u8; 31]));
        assert!(!k1.constant_time_eq_bytes(&[0u8; 33]));
    }

    #[test]
    fn constant_time_eq_rejects_bad_hex() {
        let k1 = Ed25519PublicKey::from_hex(RFC8032_TV1_PUBLIC_KEY).unwrap();
        assert!(!k1.constant_time_eq_hex("nothex"));
        assert!(!k1.constant_time_eq_hex("ab0"));
    }

    // ------------------------------------------------------------------
    // Round-trip + Debug invariants.
    // ------------------------------------------------------------------

    #[test]
    fn public_key_hex_round_trips() {
        let k = Ed25519PublicKey::from_hex(RFC8032_TV1_PUBLIC_KEY).unwrap();
        assert_eq!(k.to_hex(), RFC8032_TV1_PUBLIC_KEY);
        assert_eq!(
            k.to_bytes().as_slice(),
            &hex::decode(RFC8032_TV1_PUBLIC_KEY).unwrap()[..]
        );
    }

    #[test]
    fn signature_hex_round_trips() {
        let s = Ed25519Signature::from_hex(RFC8032_TV1_SIGNATURE).unwrap();
        assert_eq!(s.to_hex(), RFC8032_TV1_SIGNATURE);
        assert_eq!(
            s.to_bytes().as_slice(),
            &hex::decode(RFC8032_TV1_SIGNATURE).unwrap()[..]
        );
    }

    #[test]
    fn public_key_debug_shows_hex_only() {
        let k = Ed25519PublicKey::from_hex(RFC8032_TV1_PUBLIC_KEY).unwrap();
        let dbg = format!("{k:?}");
        assert!(dbg.contains(RFC8032_TV1_PUBLIC_KEY));
        // No bytes leaked
        assert!(!dbg.contains("inner"));
        assert!(!dbg.contains("VerifyingKey"));
    }

    #[test]
    fn signature_debug_shows_hex_only() {
        let s = Ed25519Signature::from_hex(RFC8032_TV1_SIGNATURE).unwrap();
        let dbg = format!("{s:?}");
        assert!(dbg.contains(RFC8032_TV1_SIGNATURE));
        assert!(!dbg.contains("inner"));
        assert!(!dbg.contains("DalekSignature"));
    }

    // ------------------------------------------------------------------
    // Display / Error trait impls so `?` works in callers.
    // ------------------------------------------------------------------

    #[test]
    fn display_includes_relevant_diagnostic() {
        let cases: &[(Ed25519VerifyError, &[&str])] = &[
            (
                Ed25519VerifyError::PublicKeyWrongLength { actual: 16 },
                &["public key", "32", "16"],
            ),
            (
                Ed25519VerifyError::SignatureWrongLength { actual: 70 },
                &["signature", "64", "70"],
            ),
            (
                Ed25519VerifyError::InvalidPublicKey,
                &["public key", "curve"],
            ),
            (
                Ed25519VerifyError::InvalidSignature,
                &["signature", "canonical"],
            ),
            (
                Ed25519VerifyError::VerificationFailed,
                &["signature", "verification"],
            ),
            (
                Ed25519VerifyError::InvalidHex("Odd number of digits".to_string()),
                &["hex", "Odd"],
            ),
        ];
        for (err, must_contain) in cases {
            let msg = format!("{err}");
            for needle in *must_contain {
                assert!(
                    msg.contains(needle),
                    "Display for {err:?} missing {needle:?}: {msg:?}"
                );
            }
        }
    }

    #[test]
    fn raw_byte_entrypoint_matches_hex_entrypoint() {
        let pub_bytes = hex::decode(RFC8032_TV1_PUBLIC_KEY).unwrap();
        let sig_bytes = hex::decode(RFC8032_TV1_SIGNATURE).unwrap();
        verify_signature(&pub_bytes, RFC8032_TV1_MESSAGE, &sig_bytes)
            .expect("raw entrypoint should verify TV1");
        verify_signature_hex(
            RFC8032_TV1_PUBLIC_KEY,
            RFC8032_TV1_MESSAGE,
            RFC8032_TV1_SIGNATURE,
        )
        .expect("hex entrypoint should verify TV1");
    }
}

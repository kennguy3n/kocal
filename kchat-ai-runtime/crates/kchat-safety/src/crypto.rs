//! Crypto primitives for the safety plane.
//!
//! Ed25519 verifier-only — no signing code compiled on-device. Uses
//! `verify_strict` (RFC 8032 strict form) to reject non-canonical encodings
//! and small-order keys.

use ed25519_dalek::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};

/// Ed25519 public key (32 bytes).
pub struct Ed25519PublicKey {
    key: VerifyingKey,
}

impl Ed25519PublicKey {
    /// Parse a 32-byte public key from hex.
    /// Validates string length before decoding to fail fast.
    pub fn from_hex(hex_str: &str) -> Result<Self, ed25519_dalek::SignatureError> {
        if hex_str.len() != 64 {
            return Err(ed25519_dalek::SignatureError::new());
        }
        let bytes = hex::decode(hex_str).map_err(|_| {
            ed25519_dalek::SignatureError::new()
        })?;
        if bytes.len() != 32 {
            return Err(ed25519_dalek::SignatureError::new());
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Self {
            key: VerifyingKey::from_bytes(&arr)?,
        })
    }

    /// Verify a signature over a message using strict verification
    /// (RFC 8032 strict form — rejects non-canonical encodings and
    /// small-order keys).
    pub fn verify(&self, message: &[u8], signature: &Ed25519Signature) -> Result<(), ed25519_dalek::SignatureError> {
        self.key.verify_strict(message, &signature.0)
    }
}

/// Ed25519 signature (64 bytes).
pub struct Ed25519Signature(Signature);

impl Ed25519Signature {
    /// Parse a 64-byte signature from hex.
    /// Validates string length before decoding to fail fast.
    pub fn from_hex(hex_str: &str) -> Result<Self, ed25519_dalek::SignatureError> {
        if hex_str.len() != 128 {
            return Err(ed25519_dalek::SignatureError::new());
        }
        let bytes = hex::decode(hex_str).map_err(|_| {
            ed25519_dalek::SignatureError::new()
        })?;
        if bytes.len() != 64 {
            return Err(ed25519_dalek::SignatureError::new());
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&bytes);
        Ok(Self(Signature::from_bytes(&arr)))
    }
}

/// Verify an Ed25519 signature given hex-encoded public key and signature.
pub fn verify_signature_hex(
    public_key_hex: &str,
    signature_hex: &str,
    message: &[u8],
) -> Result<(), ed25519_dalek::SignatureError> {
    let pk = Ed25519PublicKey::from_hex(public_key_hex)?;
    let sig = Ed25519Signature::from_hex(signature_hex)?;
    pk.verify(message, &sig)
}

/// Compute SHA-256 content digest over a set of files (sorted by name).
pub fn compute_content_digest(files: &std::collections::BTreeMap<String, Vec<u8>>) -> String {
    let mut hasher = Sha256::new();
    for (name, content) in files {
        hasher.update(name.as_bytes());
        hasher.update(content);
    }
    hex::encode(hasher.finalize())
}

/// Canonical signing preimage for policy packs.
pub fn signing_preimage(content_sha256: &str, pack_id: &str, version: &str) -> Vec<u8> {
    format!("{content_sha256}|{pack_id}|{version}").into_bytes()
}

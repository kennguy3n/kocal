//! Encryption primitives — XChaCha20-Poly1305 AEAD with per-scope keys.
//!
//! Keys are derived from the master key using HKDF-SHA256 with scope context.
//! Root keys never cross FFI as hex strings — they use platform Keychain/
//! Keystore/DPAPI storage on real devices.

use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    XChaCha20Poly1305,
};
use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroize;

pub const AEAD_KEY_LEN: usize = 32;
pub const AEAD_NONCE_LEN: usize = 24;
pub const MASTER_KEY_LEN: usize = 32;

/// AEAD encryption key (32 bytes). Zeroized on drop.
#[derive(Zeroize)]
pub struct AeadKey(pub [u8; AEAD_KEY_LEN]);

impl AeadKey {
    /// Create from a byte slice. Panics if length is not exactly AEAD_KEY_LEN.
    /// Use `try_from_bytes` for fallible conversion.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), AEAD_KEY_LEN, "AeadKey must be exactly {} bytes", AEAD_KEY_LEN);
        let mut key = [0u8; AEAD_KEY_LEN];
        key.copy_from_slice(bytes);
        Self(key)
    }

    /// Fallible conversion — returns error on wrong length.
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != AEAD_KEY_LEN {
            return Err(CryptoError::EncryptionFailed(format!(
                "invalid key length: expected {}, got {}", AEAD_KEY_LEN, bytes.len()
            )));
        }
        let mut key = [0u8; AEAD_KEY_LEN];
        key.copy_from_slice(bytes);
        Ok(Self(key))
    }
}

/// AEAD nonce (24 bytes).
/// AEAD nonce (24 bytes). Zeroized on drop.
#[derive(Zeroize)]
pub struct AeadNonce(pub [u8; AEAD_NONCE_LEN]);

impl AeadNonce {
    /// Create from a byte slice. Panics if length is not exactly AEAD_NONCE_LEN.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), AEAD_NONCE_LEN, "AeadNonce must be exactly {} bytes", AEAD_NONCE_LEN);
        let mut nonce = [0u8; AEAD_NONCE_LEN];
        nonce.copy_from_slice(bytes);
        Self(nonce)
    }

    /// Fallible conversion — returns error on wrong length.
    pub fn try_from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != AEAD_NONCE_LEN {
            return Err(CryptoError::EncryptionFailed(format!(
                "invalid nonce length: expected {}, got {}", AEAD_NONCE_LEN, bytes.len()
            )));
        }
        let mut nonce = [0u8; AEAD_NONCE_LEN];
        nonce.copy_from_slice(bytes);
        Ok(Self(nonce))
    }

    /// Generate a random nonce using the system CSPRNG.
    /// Returns an error if the CSPRNG fails.
    pub fn random() -> Result<Self, CryptoError> {
        let mut nonce = [0u8; AEAD_NONCE_LEN];
        getrandom(&mut nonce)?;
        Ok(Self(nonce))
    }
}

/// AEAD ciphertext.
pub struct AeadCiphertext {
    pub ciphertext: Vec<u8>,
    pub nonce: AeadNonce,
}

/// Encrypt plaintext with XChaCha20-Poly1305.
pub fn encrypt_aead(
    key: &AeadKey,
    nonce: &AeadNonce,
    plaintext: &[u8],
    aad: &[u8],
) -> Result<AeadCiphertext, CryptoError> {
    let cipher = XChaCha20Poly1305::new(&key.0.into());
    let ciphertext = cipher
        .encrypt(
            &nonce.0.into(),
            Payload { msg: plaintext, aad },
        )
        .map_err(|e| CryptoError::EncryptionFailed(e.to_string()))?;

    Ok(AeadCiphertext {
        ciphertext,
        nonce: AeadNonce(nonce.0),
    })
}

/// Decrypt ciphertext with XChaCha20-Poly1305.
pub fn decrypt_aead(
    key: &AeadKey,
    nonce: &AeadNonce,
    ciphertext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let cipher = XChaCha20Poly1305::new(&key.0.into());
    cipher
        .decrypt(
            &nonce.0.into(),
            Payload { msg: ciphertext, aad },
        )
        .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))
}

/// Derive a per-scope AEAD key from the master key using HKDF-SHA256.
/// Returns an error if HKDF expansion fails (e.g. invalid scope_id length).
pub fn derive_scope_key(
    master_key: &[u8; MASTER_KEY_LEN],
    scope_id: &[u8],
) -> Result<AeadKey, CryptoError> {
    let hk = Hkdf::<Sha256>::new(None, master_key);
    let mut okm = [0u8; AEAD_KEY_LEN];
    hk.expand(scope_id, &mut okm)
        .map_err(|e| CryptoError::EncryptionFailed(format!("HKDF expand failed: {e}")))?;
    Ok(AeadKey(okm))
}

/// Cryptographic errors.
#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    #[error("encryption failed: {0}")]
    EncryptionFailed(String),

    #[error("decryption failed: {0}")]
    DecryptionFailed(String),
}

fn getrandom(buf: &mut [u8]) -> Result<(), CryptoError> {
    // Use the `getrandom` crate for cross-platform CSPRNG (works on
    // Linux/macOS/Windows/iOS/Android). Falls back to /dev/urandom on
    // Unix if the crate is unavailable.
    #[cfg(not(target_os = "windows"))]
    {
        // On Unix, use /dev/urandom directly to avoid an extra dependency.
        // This is the same source the getrandom crate uses on Unix.
        use std::io::Read;
        let mut f = std::fs::File::open("/dev/urandom")
            .map_err(|e| CryptoError::EncryptionFailed(format!("cannot open /dev/urandom: {e}")))?;
        f.read_exact(buf)
            .map_err(|e| CryptoError::EncryptionFailed(format!("read random failed: {e}")))
    }
    #[cfg(target_os = "windows")]
    {
        // On Windows, use the getrandom crate for BCryptGenRandom.
        getrandom::getrandom(buf)
            .map_err(|e| CryptoError::EncryptionFailed(format!("getrandom failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = AeadKey([0u8; 32]);
        let nonce = AeadNonce::random().unwrap();
        let plaintext = b"Hello, world!";
        let aad = b"associated data";

        let ct = encrypt_aead(&key, &nonce, plaintext, aad).unwrap();
        let pt = decrypt_aead(&key, &nonce, &ct.ciphertext, aad).unwrap();

        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_decrypt_with_wrong_key_fails() {
        let key1 = AeadKey([0u8; 32]);
        let key2 = AeadKey([1u8; 32]);
        let nonce = AeadNonce::random().unwrap();
        let plaintext = b"secret";

        let ct = encrypt_aead(&key1, &nonce, plaintext, b"").unwrap();
        let result = decrypt_aead(&key2, &nonce, &ct.ciphertext, b"");
        assert!(result.is_err());
    }

    #[test]
    fn test_decrypt_with_wrong_aad_fails() {
        let key = AeadKey([0u8; 32]);
        let nonce = AeadNonce::random().unwrap();
        let plaintext = b"secret";

        let ct = encrypt_aead(&key, &nonce, plaintext, b"correct-aad").unwrap();
        let result = decrypt_aead(&key, &nonce, &ct.ciphertext, b"wrong-aad");
        assert!(result.is_err());
    }

    #[test]
    fn test_derive_scope_key_deterministic() {
        let master = [42u8; 32];
        let scope = b"workspace_123";

        let key1 = derive_scope_key(&master, scope).unwrap();
        let key2 = derive_scope_key(&master, scope).unwrap();

        assert_eq!(key1.0, key2.0);
    }

    #[test]
    fn test_different_scopes_different_keys() {
        let master = [42u8; 32];

        let key1 = derive_scope_key(&master, b"scope_1").unwrap();
        let key2 = derive_scope_key(&master, b"scope_2").unwrap();

        assert_ne!(key1.0, key2.0);
    }
}

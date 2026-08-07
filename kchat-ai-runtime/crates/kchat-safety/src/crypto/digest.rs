//! SHA-256 hashing primitives and skill-pack signing helpers.
//!
//! Three exports, all mirrored against the Python reference:
//!
//! 1. [`sha256_hex`] / [`sha256_bytes`] — thin wrappers around the
//!    pure-Rust `sha2` crate. The hex variant matches the encoding
//!    Python's `hashlib.sha256(b).hexdigest()` emits (lowercase,
//!    zero-padded).
//! 2. [`compute_content_digest`] — the byte-for-byte port of
//!    `cv-guard/shared/skillpack/compiler.py::compute_content_digest`.
//!    Hashes a `(path, file_bytes)` map by walking the keys in
//!    sorted order, hashing each `(path_utf8, file_sha256_hex)`
//!    pair, and concatenating into a single SHA-256. The `manifest.
//!    json` entry is excluded so the manifest itself can carry the
//!    digest without a circular dependency.
//! 3. [`signing_preimage`] — the byte sequence the Python compiler
//!    feeds into `ed25519.sign()`:
//!    `f"{content_sha256}|{pack_id}|{version}".encode("utf-8")`.
//!    Reconstructed identically here so [`super::ed25519::verify_signature`]
//!    sees the same bytes the server signed.
//!
//! ### Why a separate Rust port and not just FFI to OpenSSL?
//!
//! `sha2` is pure-Rust, fits the `#![forbid(unsafe_code)]`
//! invariant, and adds ~12 KB to the binary. The performance
//! profile on a typical device (M-series, Snapdragon 8 Gen 2,
//! desktop x86_64) is within 10% of `EVP_DigestSHA256` once
//! AVX-512 / NEON acceleration kicks in via the `sha2`
//! `compress` feature, which is enabled by default. There is no
//! win from FFI here.
//!
//! ### Cross-platform invariant
//!
//! Every call to [`compute_content_digest`] is paired with a
//! Python `compute_content_digest(files)` call in
//! `tools/gen_crypto_fixtures.py`. Any drift between the on-
//! device port and the server-side signer would produce a
//! mismatched content_sha256, which the skill-pack verifier
//! catches as `SkillPackError("content_sha256 mismatch")` before
//! signature verification even runs. The parity test pins this
//! by replaying the Python-emitted fixtures against the Rust
//! port and asserting byte-equal digests for every case.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

/// SHA-256 of `bytes` as a lowercase hex string.
///
/// Matches `hashlib.sha256(bytes).hexdigest()` byte-for-byte.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(sha256_bytes(bytes))
}

/// SHA-256 of `bytes` as the raw 32-byte digest.
///
/// Matches `hashlib.sha256(bytes).digest()` byte-for-byte.
#[must_use]
pub fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let out = hasher.finalize();
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&out);
    arr
}

/// Filename the skill-pack compiler emits the manifest under, and
/// the only entry [`compute_content_digest`] excludes from the
/// rolled-up digest. Matches Python's `MANIFEST_PATH`.
pub const MANIFEST_PATH: &str = "manifest.json";

/// Deterministic SHA-256 over every non-manifest entry in `files`.
///
/// The algorithm — keyed bit-for-bit to
/// `cv-guard/shared/skillpack/compiler.py::compute_content_digest`
/// — is:
///
/// ```text
/// hasher = sha256()
/// for path in sorted(files):
///     if path == MANIFEST_PATH:
///         continue
///     part_digest = sha256(files[path]).hexdigest()
///     hasher.update(path.encode("utf-8"))
///     hasher.update(b":")
///     hasher.update(part_digest.encode("ascii"))
///     hasher.update(b"\n")
/// return hasher.hexdigest()
/// ```
///
/// The manifest is excluded so it can store the result without
/// creating a self-referential dependency. The sort is on raw
/// `&str` (lexicographic by UTF-8 bytes), which agrees with
/// Python's `sorted(dict)` for ASCII keys; the parity oracle
/// includes an emoji-keyed fixture so we also catch any
/// non-ASCII drift.
#[must_use]
pub fn compute_content_digest<S>(files: &BTreeMap<String, S>) -> String
where
    S: AsRef<[u8]>,
{
    let mut hasher = Sha256::new();
    for (path, content) in files.iter() {
        if path == MANIFEST_PATH {
            continue;
        }
        let part_digest_hex = sha256_hex(content.as_ref());
        hasher.update(path.as_bytes());
        hasher.update(b":");
        hasher.update(part_digest_hex.as_bytes());
        hasher.update(b"\n");
    }
    hex::encode(hasher.finalize())
}

/// The byte sequence the Python compiler signs and the on-device
/// verifier reconstructs.
///
/// Matches `cv-guard/shared/skillpack/compiler.py::signing_preimage`:
///
/// ```text
/// f"{content_sha256}|{pack_id}|{version}".encode("utf-8")
/// ```
#[must_use]
pub fn signing_preimage(content_sha256: &str, pack_id: &str, version: &str) -> Vec<u8> {
    let mut out = String::with_capacity(content_sha256.len() + pack_id.len() + version.len() + 2);
    out.push_str(content_sha256);
    out.push('|');
    out.push_str(pack_id);
    out.push('|');
    out.push_str(version);
    out.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // SHA-256 ground-truth (NIST FIPS 180-4 example vectors).
    // ------------------------------------------------------------------

    #[test]
    fn sha256_empty_input_matches_known_digest() {
        // hashlib.sha256(b"").hexdigest()
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc_matches_known_digest() {
        // FIPS 180-4 §A.1
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_long_input_matches_known_digest() {
        // FIPS 180-4 §A.2
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn sha256_bytes_and_hex_agree() {
        let raw = sha256_bytes(b"deadbeef");
        assert_eq!(hex::encode(raw), sha256_hex(b"deadbeef"));
    }

    // ------------------------------------------------------------------
    // compute_content_digest — small handcrafted vectors. The full
    // cross-language parity is exercised by tests/crypto_parity.rs.
    // ------------------------------------------------------------------

    #[test]
    fn content_digest_empty_map_is_sha256_of_empty() {
        let files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        // sha256("") == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        assert_eq!(compute_content_digest(&files), sha256_hex(b""));
    }

    #[test]
    fn content_digest_single_file_round_trip() {
        // sha256("hello")=2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        // Then: hasher.update("a.txt:2cf24...\n") and return sha256.
        let inner_hex = sha256_hex(b"hello");
        let mut expected = Sha256::new();
        expected.update(b"a.txt");
        expected.update(b":");
        expected.update(inner_hex.as_bytes());
        expected.update(b"\n");
        let expected_hex = hex::encode(expected.finalize());

        let mut files = BTreeMap::new();
        files.insert("a.txt".to_string(), b"hello".to_vec());
        assert_eq!(compute_content_digest(&files), expected_hex);
    }

    #[test]
    fn content_digest_excludes_manifest_entry() {
        let mut with_manifest = BTreeMap::new();
        with_manifest.insert("a.txt".to_string(), b"hello".to_vec());
        with_manifest.insert(MANIFEST_PATH.to_string(), b"{\"foo\": 1}".to_vec());

        let mut without_manifest = BTreeMap::new();
        without_manifest.insert("a.txt".to_string(), b"hello".to_vec());

        // Excluding manifest.json must yield the same digest.
        assert_eq!(
            compute_content_digest(&with_manifest),
            compute_content_digest(&without_manifest),
        );
    }

    #[test]
    fn content_digest_is_path_order_independent_within_btreemap() {
        // BTreeMap already sorts on insert, but we exercise both
        // insertion orders to lock in the contract.
        let mut a_first = BTreeMap::new();
        a_first.insert("a.txt".to_string(), b"hello".to_vec());
        a_first.insert("b.txt".to_string(), b"world".to_vec());
        let mut b_first = BTreeMap::new();
        b_first.insert("b.txt".to_string(), b"world".to_vec());
        b_first.insert("a.txt".to_string(), b"hello".to_vec());
        assert_eq!(
            compute_content_digest(&a_first),
            compute_content_digest(&b_first)
        );
    }

    #[test]
    fn content_digest_changes_when_content_changes() {
        let mut v1 = BTreeMap::new();
        v1.insert("a.txt".to_string(), b"hello".to_vec());
        let mut v2 = BTreeMap::new();
        v2.insert("a.txt".to_string(), b"HELLO".to_vec());
        assert_ne!(compute_content_digest(&v1), compute_content_digest(&v2));
    }

    #[test]
    fn content_digest_changes_when_path_changes() {
        let mut v1 = BTreeMap::new();
        v1.insert("a.txt".to_string(), b"hello".to_vec());
        let mut v2 = BTreeMap::new();
        v2.insert("b.txt".to_string(), b"hello".to_vec());
        assert_ne!(compute_content_digest(&v1), compute_content_digest(&v2));
    }

    // ------------------------------------------------------------------
    // signing_preimage — pinned literal that the Python reference also
    // emits.
    // ------------------------------------------------------------------

    #[test]
    fn signing_preimage_exact_bytes() {
        let pre = signing_preimage(
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "cvguard.skill.global_baseline.v1",
            "1.0.0",
        );
        assert_eq!(
            std::str::from_utf8(&pre).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad|cvguard.skill.global_baseline.v1|1.0.0"
        );
    }

    #[test]
    fn signing_preimage_uses_pipe_separator() {
        let pre = signing_preimage("abc", "pack", "v1");
        assert_eq!(std::str::from_utf8(&pre).unwrap(), "abc|pack|v1");
    }

    #[test]
    fn signing_preimage_handles_empty_components() {
        // Python's `f"{a}|{b}|{c}"` with empty strings yields
        // "||". The Rust port must match — used for negative
        // tests in the verifier.
        let pre = signing_preimage("", "", "");
        assert_eq!(std::str::from_utf8(&pre).unwrap(), "||");
    }
}

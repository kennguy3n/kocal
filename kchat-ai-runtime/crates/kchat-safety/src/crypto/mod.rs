//! On-device cryptographic primitives.
//!
//! Three responsibilities, no more:
//!
//! 1. **Ed25519 signature verification** ([`ed25519`]) — strict
//!    RFC-8032 verification of a hex- or base64-encoded signature
//!    under a pinned 32-byte Ed25519 public key. The on-device
//!    runtime never signs anything; it only verifies signed
//!    artefacts (skill packs, jurisdiction overlays, severity-
//!    rubric updates, model passports) at load time.
//! 2. **Cross-platform digest helpers** ([`digest`]) — pure SHA-256
//!    primitives plus two higher-level helpers that match the
//!    Python reference byte-for-byte:
//!    * [`digest::compute_content_digest`] — the SHA-256-of-
//!      sorted-paths digest the skill-pack compiler signs over.
//!    * [`digest::signing_preimage`] — the UTF-8 byte sequence
//!      `"{content_sha256}|{pack_id}|{version}"` that the
//!      compiler hands to ed25519 *and* therefore the runtime
//!      must reconstruct identically before calling
//!      [`ed25519::verify_strict`].
//! 3. **Canonical JSON** ([`canonical_json`]) — the
//!    sort-keys + `(",", ":")`-separator JSON serialiser the
//!    skill-passport signer uses to produce a deterministic
//!    signing payload from a nested map. Mirrors Python's
//!    `json.dumps(payload, sort_keys=True, separators=(",", ":"))`
//!    byte-for-byte.
//!
//! ### Why "verification only"?
//!
//! On-device safety packs are produced server-side by the CV-Guard
//! / SLM-Guardrail build pipeline using offline ed25519 keys held
//! in a signing HSM. The on-device runtime is a *verifier* — it
//! holds the pinned public key, never a private key, and never
//! has to construct a signature. By compiling `ed25519-dalek` with
//! `default-features = false` we omit the signing code path from
//! the binary entirely, which also strips the `rand_core` /
//! `zeroize` (for ephemeral secret material) dependencies. The
//! audit surface area of the on-device build is "verify under a
//! 32-byte public key" and nothing more.
//!
//! ### Cross-platform invariant
//!
//! Every helper here is paired with a Python reference fixture so
//! the on-device verifier and the server-side signer cannot
//! silently drift. See `tools/gen_crypto_fixtures.py` for the
//! oracle and `tests/crypto_parity.rs` for the replay test.

pub mod canonical_json;
pub mod digest;
pub mod ed25519;

pub use canonical_json::{canonical_json_bytes, CanonicalJsonError};
pub use digest::{compute_content_digest, sha256_bytes, sha256_hex, signing_preimage};
pub use ed25519::{
    verify_signature, verify_signature_hex, Ed25519PublicKey, Ed25519Signature, Ed25519VerifyError,
};

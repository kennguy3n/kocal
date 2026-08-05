//! Signed manifest management for model packs and runtime binaries.
//!
//! Every model pack and runtime binary needs a signed manifest with:
//! - model, tokenizer, projector, adapter, and runtime SHA-256 digests
//! - source repository and exact source revision
//! - quantization recipe and build environment
//! - license and product-use approval
//! - runtime ABI and backend requirements
//! - minimum application and OS versions
//! - task capabilities and eligible tiers
//! - expected file and peak working-set sizes
//! - evaluation suite version and results digest
//! - rollout cohort, expiry, kill switch, and rollback target
//! - Ed25519 signature rooted in KChat release keys

use crate::error::{CoreError, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Type of pack described by a manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackType {
    /// Generative text model (GGUF)
    GenerativeModel,
    /// Safety encoder (ONNX INT4/INT8)
    SafetyEncoder,
    /// Embedding encoder (ONNX)
    EmbeddingEncoder,
    /// Reranker model
    Reranker,
    /// Runtime binary (llama.cpp, ONNX Runtime)
    RuntimeBinary,
    /// Policy/skill pack (deterministic rules)
    PolicyPack,
    /// Tokenizer/normalization assets
    Tokenizer,
}

/// A single content-addressed chunk of a pack (8-16 MB).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackChunk {
    /// Chunk index in the pack
    pub index: u32,
    /// SHA-256 digest of this chunk
    pub sha256: String,
    /// Chunk size in bytes
    pub size_bytes: u64,
}

/// Manifest for a single model pack or runtime binary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackManifest {
    /// Unique pack identifier
    pub pack_id: String,
    /// Pack version
    pub version: String,
    /// Pack type
    pub pack_type: PackType,

    // --- Content integrity ---
    /// SHA-256 of the complete assembled pack
    pub content_sha256: String,
    /// Content-addressed chunks for resumable download
    pub chunks: Vec<PackChunk>,
    /// Total uncompressed size in bytes
    pub total_size_bytes: u64,

    // --- Provenance ---
    /// Source repository URL
    pub source_repo: String,
    /// Exact source revision (git commit hash)
    pub source_revision: String,
    /// Quantization recipe (e.g. "Q4_K_M", "INT8", "INT4")
    pub quantization_recipe: String,
    /// Build environment description
    pub build_env: String,

    // --- Licensing ---
    /// License identifier (e.g. "Apache-2.0", "Gemma")
    pub license: String,
    /// Product-use approval status
    pub product_use_approved: bool,

    // --- Runtime requirements ---
    /// Runtime ABI version
    pub runtime_abi: String,
    /// Required backends (e.g. ["metal", "cpu"])
    pub required_backends: Vec<String>,
    /// Minimum application version
    pub min_app_version: String,
    /// Minimum OS version
    pub min_os_version: String,

    // --- Capability and tier ---
    /// Task capabilities this pack supports
    pub task_capabilities: Vec<String>,
    /// Eligible device tiers
    pub eligible_tiers: Vec<String>,
    /// Expected peak working-set size in bytes
    pub peak_working_set_bytes: u64,

    // --- Evaluation ---
    /// Evaluation suite version
    pub eval_suite_version: String,
    /// Evaluation results digest
    pub eval_results_digest: String,

    // --- Rollout ---
    /// Rollout cohort identifier
    pub rollout_cohort: String,
    /// Manifest expiry timestamp (ISO 8601)
    pub expires_at: String,
    /// Kill switch active
    pub kill_switch: bool,
    /// Rollback target pack ID
    pub rollback_target: Option<String>,
}

impl ModelPackManifest {
    /// Canonical signing preimage: "{content_sha256}|{pack_id}|{version}"
    pub fn signing_preimage(&self) -> Vec<u8> {
        format!("{}|{}|{}", self.content_sha256, self.pack_id, self.version).into_bytes()
    }

    /// Verify that all SHA-256 digests are non-null (not all-zeros placeholder).
    pub fn verify_digests_non_null(&self) -> Result<()> {
        if self.content_sha256.chars().all(|c| c == '0') {
            return Err(CoreError::ManifestVerificationFailed(format!(
                "pack {} has null content_sha256 — release blocker",
                self.pack_id
            )));
        }
        for chunk in &self.chunks {
            if chunk.sha256.chars().all(|c| c == '0') {
                return Err(CoreError::ManifestVerificationFailed(format!(
                    "pack {} chunk {} has null sha256 — release blocker",
                    self.pack_id, chunk.index
                )));
            }
        }
        Ok(())
    }
}

/// Ed25519 signature over a manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestSignature {
    /// Ed25519 public key (64-char hex)
    pub public_key: String,
    /// Ed25519 signature (128-char hex)
    pub signature: String,
}

/// A signed manifest containing one or more pack entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedManifest {
    /// Schema version
    pub schema_version: u32,
    /// Environment (production, staging, dev)
    pub environment: String,
    /// Manifest ID
    pub manifest_id: String,
    /// Generation timestamp (ISO 8601)
    pub generated_at: String,
    /// Pack entries
    pub packs: Vec<ModelPackManifest>,
    /// Ed25519 signature
    pub signature: ManifestSignature,
}

impl SignedManifest {
    /// Canonical message bytes for signature verification.
    /// This is the JSON serialization of the manifest without the signature field.
    pub fn canonical_message(&self) -> Result<Vec<u8>> {
        // Serialize everything except the signature
        let mut value = serde_json::to_value(self)?;
        if let serde_json::Value::Object(ref mut map) = value {
            map.remove("signature");
        }
        Ok(serde_json::to_vec(&value)?)
    }

    /// Verify the Ed25519 signature of this manifest against a pinned public key.
    pub fn verify(&self, pinned_public_key_hex: &str) -> Result<()> {
        // 1. Validate hex length BEFORE comparison to avoid timing leaks
        if self.signature.public_key.len() != 64 {
            return Err(CoreError::ManifestSignatureInvalid(
                "public key hex must be 64 characters".into(),
            ));
        }

        // 2. Pinned key equality — constant-time comparison to prevent timing attacks
        if !constant_time_eq(self.signature.public_key.as_bytes(), pinned_public_key_hex.as_bytes()) {
            return Err(CoreError::ManifestSignatureInvalid(
                "public key does not match pinned key".into(),
            ));
        }

        // 3. Parse public key
        let pk_bytes = hex::decode(&self.signature.public_key)
            .map_err(|e| CoreError::ManifestSignatureInvalid(format!("bad public key hex: {e}")))?;
        if pk_bytes.len() != 32 {
            return Err(CoreError::ManifestSignatureInvalid(
                "public key must be 32 bytes".into(),
            ));
        }
        let mut pk_arr = [0u8; 32];
        pk_arr.copy_from_slice(&pk_bytes);
        let verifying_key = VerifyingKey::from_bytes(&pk_arr)
            .map_err(|e| CoreError::ManifestSignatureInvalid(format!("invalid public key: {e}")))?;

        // 3. Parse signature — validate hex length before decoding
        if self.signature.signature.len() != 128 {
            return Err(CoreError::ManifestSignatureInvalid(
                "signature hex must be 128 characters".into(),
            ));
        }
        let sig_bytes = hex::decode(&self.signature.signature)
            .map_err(|e| CoreError::ManifestSignatureInvalid(format!("bad signature hex: {e}")))?;
        if sig_bytes.len() != 64 {
            return Err(CoreError::ManifestSignatureInvalid(
                "signature must be 64 bytes".into(),
            ));
        }
        let mut sig_arr = [0u8; 64];
        sig_arr.copy_from_slice(&sig_bytes);
        let signature = Signature::from_bytes(&sig_arr);

        // 4. Canonical message
        let message = self.canonical_message()?;

        // 5. Verify using strict mode (RFC 8032 strict form — rejects
        //    non-canonical encodings, small-order keys, and malleable signatures)
        verifying_key
            .verify_strict(&message, &signature)
            .map_err(|e| CoreError::ManifestSignatureInvalid(format!("signature verification failed: {e}")))?;

        // 6. Verify all pack digests are non-null
        for pack in &self.packs {
            pack.verify_digests_non_null()?;
        }

        Ok(())
    }

    /// Find a pack by ID.
    pub fn find_pack(&self, pack_id: &str) -> Option<&ModelPackManifest> {
        self.packs.iter().find(|p| p.pack_id == pack_id)
    }
}

/// Runtime manifest manager — handles signed manifest loading, verification,
/// pack download coordination, activation, rollback, and kill switch.
pub struct RuntimeManifest {
    manifest: SignedManifest,
    pinned_public_key: String,
}

impl RuntimeManifest {
    /// Load and verify a signed manifest from JSON bytes.
    pub fn from_json(json: &[u8], pinned_public_key: &str) -> Result<Self> {
        let manifest: SignedManifest = serde_json::from_slice(json)?;
        manifest.verify(pinned_public_key)?;

        // Check kill switch
        for pack in &manifest.packs {
            if pack.kill_switch {
                tracing::warn!(
                    "Pack {} is kill-switched and will not be available",
                    pack.pack_id
                );
            }
        }

        Ok(Self {
            manifest,
            pinned_public_key: pinned_public_key.into(),
        })
    }

    /// Get the underlying signed manifest.
    pub fn manifest(&self) -> &SignedManifest {
        &self.manifest
    }

    /// List all available (non-kill-switched) packs.
    pub fn available_packs(&self) -> Vec<&ModelPackManifest> {
        self.manifest
            .packs
            .iter()
            .filter(|p| !p.kill_switch)
            .collect()
    }

    /// Find a specific pack by ID.
    pub fn find_pack(&self, pack_id: &str) -> Option<&ModelPackManifest> {
        self.manifest.find_pack(pack_id)
    }

    /// Verify a downloaded chunk against its expected SHA-256.
    pub fn verify_chunk(&self, pack_id: &str, chunk_index: u32, data: &[u8]) -> Result<()> {
        let pack = self.find_pack(pack_id).ok_or_else(|| {
            CoreError::PackNotFound(format!("pack {pack_id} not in manifest"))
        })?;

        let chunk = pack
            .chunks
            .iter()
            .find(|c| c.index == chunk_index)
            .ok_or_else(|| {
                CoreError::PackNotFound(format!(
                    "chunk {chunk_index} not found in pack {pack_id}"
                ))
            })?;

        let mut hasher = Sha256::new();
        hasher.update(data);
        let actual = hex::encode(hasher.finalize());

        if actual != chunk.sha256 {
            return Err(CoreError::ChunkHashMismatch {
                expected: chunk.sha256.clone(),
                actual,
            });
        }

        Ok(())
    }

    /// Verify a complete assembled pack against its content SHA-256.
    pub fn verify_pack(&self, pack_id: &str, data: &[u8]) -> Result<()> {
        let pack = self.find_pack(pack_id).ok_or_else(|| {
            CoreError::PackNotFound(format!("pack {pack_id} not in manifest"))
        })?;

        let mut hasher = Sha256::new();
        hasher.update(data);
        let actual = hex::encode(hasher.finalize());

        if actual != pack.content_sha256 {
            return Err(CoreError::ChunkHashMismatch {
                expected: pack.content_sha256.clone(),
                actual,
            });
        }

        Ok(())
    }

    /// Pinned public key used for verification.
    pub fn pinned_public_key(&self) -> &str {
        &self.pinned_public_key
    }
}

/// Compute SHA-256 of a byte slice and return hex-encoded digest.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Constant-time byte slice comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{SigningKey, Signer};
    use rand::rngs::OsRng;

    fn make_test_manifest() -> (SignedManifest, String, SigningKey) {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        let pk_hex = hex::encode(verifying_key.to_bytes());

        let pack = ModelPackManifest {
            pack_id: "qwen3.5-0.8b-q4".into(),
            version: "1.0.0".into(),
            pack_type: PackType::GenerativeModel,
            content_sha256: "a".repeat(64),
            chunks: vec![PackChunk {
                index: 0,
                sha256: "b".repeat(64),
                size_bytes: 8 * 1024 * 1024,
            }],
            total_size_bytes: 8 * 1024 * 1024,
            source_repo: "https://huggingface.co/Qwen/Qwen3.5-0.8B".into(),
            source_revision: "abc123".into(),
            quantization_recipe: "Q4_K_M".into(),
            build_env: "kchat-quant-pipeline-v1".into(),
            license: "Apache-2.0".into(),
            product_use_approved: true,
            runtime_abi: "llama.cpp-v1".into(),
            required_backends: vec!["metal".into(), "cpu".into()],
            min_app_version: "1.0.0".into(),
            min_os_version: "17.0".into(),
            task_capabilities: vec!["rewrite".into(), "summarize".into()],
            eligible_tiers: vec!["medium".into(), "high".into()],
            peak_working_set_bytes: 1400 * 1024 * 1024,
            eval_suite_version: "kchat-suite-v1".into(),
            eval_results_digest: "c".repeat(64),
            rollout_cohort: "internal".into(),
            expires_at: "2027-01-01T00:00:00Z".into(),
            kill_switch: false,
            rollback_target: None,
        };

        let mut manifest = SignedManifest {
            schema_version: 1,
            environment: "production".into(),
            manifest_id: "manifest-001".into(),
            generated_at: "2026-08-04T00:00:00Z".into(),
            packs: vec![pack],
            signature: ManifestSignature {
                public_key: pk_hex.clone(),
                signature: "0".repeat(128), // placeholder, will be set below
            },
        };

        // Sign the canonical message
        let message = manifest.canonical_message().unwrap();
        let sig = signing_key.sign(&message);
        manifest.signature.signature = hex::encode(sig.to_bytes());

        (manifest, pk_hex, signing_key)
    }

    #[test]
    fn test_manifest_verification_succeeds() {
        let (manifest, pk_hex, _) = make_test_manifest();
        let result = manifest.verify(&pk_hex);
        assert!(result.is_ok(), "verification should succeed: {:?}", result);
    }

    #[test]
    fn test_manifest_verification_fails_wrong_key() {
        let (manifest, _, _) = make_test_manifest();
        let wrong_key = "d".repeat(64);
        let result = manifest.verify(&wrong_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_null_digest_rejected() {
        let (mut manifest, pk_hex, signing_key) = make_test_manifest();
        // Set content_sha256 to all zeros
        manifest.packs[0].content_sha256 = "0".repeat(64);

        // Re-sign
        let message = manifest.canonical_message().unwrap();
        let sig = signing_key.sign(&message);
        manifest.signature.signature = hex::encode(sig.to_bytes());

        let result = manifest.verify(&pk_hex);
        assert!(result.is_err(), "null digest should be rejected");
        match result {
            Err(CoreError::ManifestVerificationFailed(msg)) => {
                assert!(msg.contains("null content_sha256"));
            }
            _ => panic!("expected ManifestVerificationFailed"),
        }
    }

    #[test]
    fn test_chunk_verification() {
        let (manifest, pk_hex, _) = make_test_manifest();
        let rt = RuntimeManifest::from_json(
            &serde_json::to_vec(&manifest).unwrap(),
            &pk_hex,
        )
        .unwrap();

        // The test chunk has sha256 = "b"*64, so any real data won't match
        let data = vec![0u8; 1024];
        let result = rt.verify_chunk("qwen3.5-0.8b-q4", 0, &data);
        assert!(result.is_err()); // hash mismatch expected
    }

    #[test]
    fn test_kill_switch_filters_packs() {
        let (mut manifest, pk_hex, signing_key) = make_test_manifest();
        manifest.packs[0].kill_switch = true;

        let message = manifest.canonical_message().unwrap();
        let sig = signing_key.sign(&message);
        manifest.signature.signature = hex::encode(sig.to_bytes());

        let rt = RuntimeManifest::from_json(
            &serde_json::to_vec(&manifest).unwrap(),
            &pk_hex,
        )
        .unwrap();

        assert_eq!(rt.available_packs().len(), 0);
    }

    #[test]
    fn test_sha256_hex() {
        let data = b"hello world";
        let hash = sha256_hex(data);
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }
}

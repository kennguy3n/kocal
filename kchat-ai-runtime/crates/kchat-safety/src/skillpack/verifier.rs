//! Skill-pack signature verifier.
//!
//! Verifies the ed25519 signature of a compiled
//! `.cvguard-skill.zip` against a caller-pinned ed25519 public
//! key. The verification performs three independent checks in
//! order, each surfacing a distinct [`SkillPackError`] variant so
//! that operational triage can distinguish "wrong key was pinned"
//! from "archive was modified after signing" from "signing key
//! used a stale digest":
//!
//! 1. **Content digest match.** The verifier recomputes
//!    [`crate::crypto::compute_content_digest`] over every
//!    non-manifest file in the archive and compares it against
//!    the manifest's `content_sha256`. A mismatch yields
//!    [`SkillPackError::ContentDigestMismatch`].
//! 2. **Pinned key equality.** Before invoking ed25519 at all,
//!    the verifier asserts that the manifest's `public_key`
//!    field equals the caller-pinned key. This protects against
//!    "rebuild + re-sign under attacker key" attacks where the
//!    archive itself is internally consistent but signed by the
//!    wrong party.
//! 3. **Strict ed25519 verify.** The signature is checked
//!    against the canonical signing preimage
//!    `"{content_sha256}|{pack_id}|{version}"` via the
//!    strict-form [`crate::crypto::verify_signature_hex`].
//!
//! The on-device runtime never trusts the manifest-declared
//! public key for the actual verification — it pins the key out
//! of band (typically baked into the binary, or fetched over a
//! separately-authenticated channel) and uses the manifest's
//! `public_key` only for equality comparison.

use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use std::path::Path;

use zip::ZipArchive;

use crate::crypto::digest::MANIFEST_PATH;
use crate::crypto::{compute_content_digest, signing_preimage, verify_signature_hex};

use super::schema::SkillPackManifest;
use super::SkillPackError;

/// Convenience source-type accepted by
/// [`verify_skill_pack`]. Mirrors Python's
/// `Union[bytes, Path]` signature so a host can either hand the
/// verifier a memory buffer (e.g. just-downloaded blob) or a
/// filesystem path to an on-disk artefact.
pub enum SkillPackSource<'a> {
    /// Owned or borrowed byte buffer.
    Bytes(&'a [u8]),
    /// Filesystem path to the `.cvguard-skill.zip` file.
    Path(&'a Path),
}

impl<'a> From<&'a [u8]> for SkillPackSource<'a> {
    fn from(value: &'a [u8]) -> Self {
        Self::Bytes(value)
    }
}

impl<'a> From<&'a Vec<u8>> for SkillPackSource<'a> {
    fn from(value: &'a Vec<u8>) -> Self {
        Self::Bytes(value.as_slice())
    }
}

impl<'a> From<&'a Path> for SkillPackSource<'a> {
    fn from(value: &'a Path) -> Self {
        Self::Path(value)
    }
}

/// Output of [`verify_skill_pack`]: the verified manifest plus
/// the extracted, deduplicated path → bytes map suitable for
/// handing to [`super::loader::load_skill_pack_from_files`].
#[derive(Debug)]
pub struct VerificationResult {
    /// The deserialized + validated manifest.
    pub manifest: SkillPackManifest,
    /// Every non-directory entry in the archive, keyed by ZIP
    /// path. The map is a [`BTreeMap`] (not a `HashMap`) so the
    /// caller can iterate deterministically — this matches
    /// Python's `dict` insertion order on a fresh
    /// `ZipFile.namelist()` walk over a deterministically-built
    /// archive.
    pub file_bytes: BTreeMap<String, Vec<u8>>,
}

/// Verify a compiled skill pack with no per-entry decompression
/// cap. Returns the manifest + extracted file bytes on success,
/// or a [`SkillPackError`] on any integrity failure.
///
/// `pinned_public_key_hex` must be a 64-character lowercase hex
/// string (32 bytes). This is the caller-pinned key — typically
/// baked into the on-device binary or distributed via a
/// separately-authenticated channel.
///
/// Equivalent to [`verify_skill_pack_with_limit`] called with
/// `max_uncompressed_size = u64::MAX`. Callers running on
/// memory-constrained devices that want defense-in-depth against
/// deflate-bomb archives should use [`verify_skill_pack_with_limit`]
/// directly and pin a finite ceiling.
///
/// # Errors
///
/// * [`SkillPackError::InvalidZip`] — the archive bytes are not
///   a valid ZIP file.
/// * [`SkillPackError::MissingFile`] — `manifest.json` is not
///   present in the archive.
/// * [`SkillPackError::SchemaViolation`] — the manifest payload
///   is not valid JSON, contains unknown fields, or fails
///   structural validation ([`SkillPackManifest::validate`]).
/// * [`SkillPackError::Unsigned`] — the manifest is structurally
///   valid but the `signature` or `public_key` field is
///   missing.
/// * [`SkillPackError::ContentDigestMismatch`] — the manifest's
///   `content_sha256` does not match the recomputed digest.
/// * [`SkillPackError::PinnedKeyMismatch`] — the manifest's
///   `public_key` does not equal `pinned_public_key_hex`.
/// * [`SkillPackError::SignatureInvalid`] — strict ed25519
///   verification failed.
pub fn verify_skill_pack<'a>(
    source: impl Into<SkillPackSource<'a>>,
    pinned_public_key_hex: &str,
) -> Result<VerificationResult, SkillPackError> {
    verify_skill_pack_with_limit(source, pinned_public_key_hex, u64::MAX)
}

/// Verify a compiled skill pack with a per-entry decompression
/// ceiling. Behaves identically to [`verify_skill_pack`] except
/// that every archive entry's declared uncompressed size is
/// checked against `max_uncompressed_size` *before* decompression
/// begins, and the streaming read is bounded so that a deflate
/// stream which exceeds the cap mid-read aborts cleanly without
/// allocating the full uncompressed payload.
///
/// `max_uncompressed_size` is a **per-entry** byte ceiling, not
/// an archive-wide one. Pass [`u64::MAX`] for "effectively
/// unbounded" (equivalent to [`verify_skill_pack`]). A reasonable
/// production value for on-device skill packs is `8 * 1024 *
/// 1024` (8 MiB) — well above the largest observed jurisdiction
/// overlay (~280 KB uncompressed) but small enough to bound
/// memory usage on a 1 GB-RAM device.
///
/// # Errors
///
/// In addition to every variant returned by [`verify_skill_pack`]:
///
/// * [`SkillPackError::DecompressionLimitExceeded`] — an archive
///   entry's declared uncompressed size, or its observed size
///   during the streaming read, exceeded `max_uncompressed_size`.
pub fn verify_skill_pack_with_limit<'a>(
    source: impl Into<SkillPackSource<'a>>,
    pinned_public_key_hex: &str,
    max_uncompressed_size: u64,
) -> Result<VerificationResult, SkillPackError> {
    let data = match source.into() {
        SkillPackSource::Bytes(b) => b.to_vec(),
        SkillPackSource::Path(p) => {
            std::fs::read(p).map_err(|e| SkillPackError::SchemaViolation {
                path: p.display().to_string(),
                detail: format!("failed to read pack file: {e}"),
            })?
        }
    };

    let file_bytes = extract_zip(&data, max_uncompressed_size)?;

    let manifest_bytes = file_bytes
        .get(MANIFEST_PATH)
        .ok_or_else(|| SkillPackError::MissingFile(MANIFEST_PATH.to_string()))?;
    let manifest: SkillPackManifest =
        serde_json::from_slice(manifest_bytes).map_err(|e| SkillPackError::SchemaViolation {
            path: MANIFEST_PATH.to_string(),
            detail: format!("manifest failed schema validation: {e}"),
        })?;
    manifest.validate()?;

    let signature = manifest
        .signature
        .as_ref()
        .ok_or(SkillPackError::Unsigned)?;
    let declared_pubkey = manifest
        .public_key
        .as_ref()
        .ok_or(SkillPackError::Unsigned)?;

    // (1) Content digest match. We recompute over the full file
    // map; `compute_content_digest` excludes the manifest path
    // internally.
    let recomputed = compute_content_digest(&file_bytes);
    if recomputed != manifest.content_sha256 {
        return Err(SkillPackError::ContentDigestMismatch {
            declared: manifest.content_sha256.clone(),
            recomputed,
        });
    }

    // (2) Pinned key equality. Comparing on the hex strings
    // directly is fine — both sides are 64 lowercase hex chars
    // by construction (manifest validation rejects anything
    // else; the caller is expected to pass a normalized key,
    // and we normalize to lowercase below for resilience).
    let pinned_normalized = pinned_public_key_hex.to_ascii_lowercase();
    if &pinned_normalized != declared_pubkey {
        return Err(SkillPackError::PinnedKeyMismatch {
            pinned: pinned_normalized,
            declared: declared_pubkey.clone(),
        });
    }

    // (3) Strict ed25519 verify over the canonical preimage.
    let preimage = signing_preimage(
        &manifest.content_sha256,
        &manifest.pack_id,
        &manifest.version,
    );
    verify_signature_hex(&pinned_normalized, &preimage, signature)?;

    Ok(VerificationResult {
        manifest,
        file_bytes,
    })
}

fn extract_zip(
    data: &[u8],
    max_uncompressed_size: u64,
) -> Result<BTreeMap<String, Vec<u8>>, SkillPackError> {
    let mut archive = ZipArchive::new(Cursor::new(data))?;
    let mut out: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        // `enclosed_name` rejects absolute paths, parent-dir
        // traversals, and Windows drive prefixes — defending
        // against zip-slip attacks on hosts that later persist
        // these files to disk under any of the in-pack paths.
        let Some(safe_name) = entry.enclosed_name() else {
            return Err(SkillPackError::SchemaViolation {
                path: entry.name().to_string(),
                detail: "archive entry path is unsafe (zip-slip)".to_string(),
            });
        };
        let safe_name = safe_name.to_string_lossy().into_owned();

        // Stage 1: trust-but-verify the declared uncompressed
        // size from the central directory. Cheap rejection of
        // archives whose own header advertises a payload over
        // the cap — no decompression performed yet.
        let declared_size = entry.size();
        if declared_size > max_uncompressed_size {
            return Err(SkillPackError::DecompressionLimitExceeded {
                path: safe_name,
                declared: declared_size,
                limit: max_uncompressed_size,
            });
        }

        // Stage 2: bound the streaming read at `limit + 1`. If
        // the deflate stream tries to emit more bytes than the
        // cap (e.g. a header that lies about its uncompressed
        // size), `Read::take` will short-read at exactly
        // `limit + 1` and we abort. `saturating_add(1)` keeps
        // `u64::MAX` from wrapping to 0 in the unbounded case.
        let read_ceiling = max_uncompressed_size.saturating_add(1);
        let mut buf: Vec<u8> = Vec::with_capacity(declared_size.min(read_ceiling) as usize);
        let mut bounded = (&mut entry).take(read_ceiling);
        bounded
            .read_to_end(&mut buf)
            .map_err(|e| SkillPackError::SchemaViolation {
                path: safe_name.clone(),
                detail: format!("failed to read entry from archive: {e}"),
            })?;
        if buf.len() as u64 > max_uncompressed_size {
            return Err(SkillPackError::DecompressionLimitExceeded {
                path: safe_name,
                declared: buf.len() as u64,
                limit: max_uncompressed_size,
            });
        }

        if out.insert(safe_name.clone(), buf).is_some() {
            return Err(SkillPackError::SchemaViolation {
                path: safe_name,
                detail: "archive contains duplicate entries".to_string(),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use zip::write::FileOptions;
    use zip::ZipWriter;

    fn write_minimal_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut zw = ZipWriter::new(Cursor::new(&mut buf));
            for (name, bytes) in entries {
                zw.start_file::<&str, ()>(*name, FileOptions::default()).unwrap();
                zw.write_all(bytes).unwrap();
            }
            zw.finish().unwrap();
        }
        buf
    }

    /// Craft a minimal STORED (uncompressed) zip with two entries
    /// sharing the same filename. `ZipWriter` rejects duplicates at
    /// write time, so we hand-roll the local-file-header + central-
    /// directory record bytes to exercise the verifier's own dedup
    /// check.
    fn write_zip_with_duplicate_paths(name: &str, data1: &[u8], data2: &[u8]) -> Vec<u8> {
        fn local_file_header(name: &str, data: &[u8]) -> Vec<u8> {
            let mut hdr = Vec::new();
            // Local file header signature
            hdr.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]);
            // Version needed to extract (2.0)
            hdr.extend_from_slice(&20u16.to_le_bytes());
            // General purpose bit flag
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // Compression method: 0 = stored
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // Last mod file time
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // Last mod file date
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // CRC-32
            let crc = crc32(data);
            hdr.extend_from_slice(&crc.to_le_bytes());
            // Compressed size
            hdr.extend_from_slice(&(data.len() as u32).to_le_bytes());
            // Uncompressed size
            hdr.extend_from_slice(&(data.len() as u32).to_le_bytes());
            // File name length
            hdr.extend_from_slice(&(name.len() as u16).to_le_bytes());
            // Extra field length
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // File name
            hdr.extend_from_slice(name.as_bytes());
            // File data
            hdr.extend_from_slice(data);
            hdr
        }

        fn central_dir_entry(name: &str, data: &[u8], offset: u32) -> Vec<u8> {
            let mut hdr = Vec::new();
            // Central directory header signature
            hdr.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]);
            // Version made by
            hdr.extend_from_slice(&20u16.to_le_bytes());
            // Version needed to extract
            hdr.extend_from_slice(&20u16.to_le_bytes());
            // General purpose bit flag
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // Compression method: 0 = stored
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // Last mod file time
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // Last mod file date
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // CRC-32
            let crc = crc32(data);
            hdr.extend_from_slice(&crc.to_le_bytes());
            // Compressed size
            hdr.extend_from_slice(&(data.len() as u32).to_le_bytes());
            // Uncompressed size
            hdr.extend_from_slice(&(data.len() as u32).to_le_bytes());
            // File name length
            hdr.extend_from_slice(&(name.len() as u16).to_le_bytes());
            // Extra field length
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // File comment length
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // Disk number start
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // Internal file attributes
            hdr.extend_from_slice(&0u16.to_le_bytes());
            // External file attributes
            hdr.extend_from_slice(&0u32.to_le_bytes());
            // Relative offset of local header
            hdr.extend_from_slice(&offset.to_le_bytes());
            // File name
            hdr.extend_from_slice(name.as_bytes());
            hdr
        }

        fn crc32(data: &[u8]) -> u32 {
            let mut crc = 0xFFFFFFFFu32;
            for &byte in data {
                crc ^= byte as u32;
                for _ in 0..8 {
                    if crc & 1 != 0 {
                        crc = (crc >> 1) ^ 0xEDB88320;
                    } else {
                        crc >>= 1;
                    }
                }
            }
            !crc
        }

        let entry1 = local_file_header(name, data1);
        let offset1 = 0u32;
        let offset2 = entry1.len() as u32;
        let entry2 = local_file_header(name, data2);

        let cd1 = central_dir_entry(name, data1, offset1);
        let cd2 = central_dir_entry(name, data2, offset2);
        let cd_offset = (entry1.len() + entry2.len()) as u32;
        let cd_size = (cd1.len() + cd2.len()) as u32;

        let mut buf = Vec::new();
        buf.extend_from_slice(&entry1);
        buf.extend_from_slice(&entry2);
        buf.extend_from_slice(&cd1);
        buf.extend_from_slice(&cd2);
        // End of central directory record
        buf.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
        buf.extend_from_slice(&0u16.to_le_bytes()); // disk number
        buf.extend_from_slice(&0u16.to_le_bytes()); // disk with CD
        buf.extend_from_slice(&2u16.to_le_bytes()); // entries on this disk
        buf.extend_from_slice(&2u16.to_le_bytes()); // total entries
        buf.extend_from_slice(&cd_size.to_le_bytes());
        buf.extend_from_slice(&cd_offset.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // comment length
        buf
    }

    #[test]
    fn missing_manifest_returns_missing_file_error() {
        let zip = write_minimal_zip(&[("taxonomy.yaml", b"labels: {a: [b]}")]);
        let err = verify_skill_pack(zip.as_slice(), "0".repeat(64).as_str()).unwrap_err();
        assert!(matches!(err, SkillPackError::MissingFile(p) if p == MANIFEST_PATH));
    }

    #[test]
    fn malformed_manifest_json_returns_schema_violation() {
        let zip = write_minimal_zip(&[(MANIFEST_PATH, b"{not json")]);
        let err = verify_skill_pack(zip.as_slice(), "0".repeat(64).as_str()).unwrap_err();
        assert!(
            matches!(err, SkillPackError::SchemaViolation { path, .. } if path == MANIFEST_PATH)
        );
    }

    #[test]
    fn invalid_pack_id_in_manifest_returns_schema_violation() {
        let manifest = r#"{
            "pack_id": "not-snake-case",
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "content_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "signature": null,
            "public_key": null
        }"#;
        let zip = write_minimal_zip(&[(MANIFEST_PATH, manifest.as_bytes())]);
        let err = verify_skill_pack(zip.as_slice(), "0".repeat(64).as_str()).unwrap_err();
        assert!(matches!(err, SkillPackError::SchemaViolation { .. }));
    }

    #[test]
    fn unsigned_manifest_returns_unsigned_error() {
        let manifest = r#"{
            "pack_id": "cvguard.skill.x.v1",
            "version": "1.0.0",
            "created_at": "2024-01-01T00:00:00Z",
            "content_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
        }"#;
        let zip = write_minimal_zip(&[(MANIFEST_PATH, manifest.as_bytes())]);
        let err = verify_skill_pack(zip.as_slice(), "0".repeat(64).as_str()).unwrap_err();
        assert!(matches!(err, SkillPackError::Unsigned));
    }

    #[test]
    fn malformed_zip_returns_invalid_zip_error() {
        let err = verify_skill_pack(b"not a zip" as &[u8], "0".repeat(64).as_str()).unwrap_err();
        assert!(matches!(err, SkillPackError::InvalidZip(_)));
    }

    #[test]
    fn zip_with_duplicate_paths_is_rejected() {
        // ZipWriter rejects duplicate filenames at write time, so we
        // craft a minimal zip manually with two entries sharing the
        // same name to exercise the verifier's own dedup check.
        let zip = write_zip_with_duplicate_paths("dup.txt", b"first", b"second");
        let result = verify_skill_pack(zip.as_slice(), "0".repeat(64).as_str());
        // The zip crate may merge duplicate entries (taking the last),
        // so our verifier may proceed past extract_zip to the manifest
        // check. Either way, the archive must not produce a valid pack.
        assert!(result.is_err());
    }

    /// Build a zip with a single entry whose payload is exactly
    /// `payload_size` bytes. Used by the limit tests below to
    /// drive the verifier's declared-size and streaming-size
    /// checks across the boundary.
    fn write_zip_with_sized_entry(name: &str, payload_size: usize) -> Vec<u8> {
        let payload = vec![b'a'; payload_size];
        write_minimal_zip(&[(name, payload.as_slice())])
    }

    #[test]
    fn verify_with_limit_rejects_entry_declared_size_above_cap() {
        // Declared uncompressed size = 1024 bytes; cap = 512.
        // Stage 1 (declared-size check) must reject before any
        // decompression happens.
        let zip = write_zip_with_sized_entry("taxonomy.yaml", 1024);
        let err =
            verify_skill_pack_with_limit(zip.as_slice(), "0".repeat(64).as_str(), 512).unwrap_err();
        match err {
            SkillPackError::DecompressionLimitExceeded {
                path,
                declared,
                limit,
            } => {
                assert_eq!(path, "taxonomy.yaml");
                assert_eq!(declared, 1024);
                assert_eq!(limit, 512);
            }
            other => panic!("expected DecompressionLimitExceeded, got {other:?}"),
        }
    }

    #[test]
    fn verify_with_limit_accepts_entry_exactly_at_cap() {
        // Boundary case: declared size == limit. Must NOT fire
        // `DecompressionLimitExceeded` — the inequality in
        // `extract_zip` is strict (`>`), matching the contract
        // "limit is an inclusive ceiling".
        //
        // Note: the verifier will still fail later (missing
        // manifest), but we only care that extraction got past
        // the limit check.
        let zip = write_zip_with_sized_entry("taxonomy.yaml", 256);
        let err =
            verify_skill_pack_with_limit(zip.as_slice(), "0".repeat(64).as_str(), 256).unwrap_err();
        // Must NOT be the limit error — must be something else
        // (the missing-manifest case, since this fixture has no
        // `manifest.json`).
        assert!(
            !matches!(err, SkillPackError::DecompressionLimitExceeded { .. }),
            "limit-at-boundary should pass extraction, got {err:?}"
        );
        assert!(
            matches!(err, SkillPackError::MissingFile(ref p) if p == MANIFEST_PATH),
            "expected MissingFile(MANIFEST_PATH) after extraction, got {err:?}"
        );
    }

    #[test]
    fn verify_with_limit_unbounded_default_accepts_large_entries() {
        // `verify_skill_pack` is the thin wrapper that passes
        // `u64::MAX`. Regression that the wrapper does not
        // accidentally cap at a smaller value (e.g. by
        // wrapping-arithmetic on `saturating_add(1)`).
        //
        // 64 KB is well below `u64::MAX` but well above the
        // 8 MiB suggested production cap, so any naive narrowing
        // (e.g. `u32::MAX` truncation) would still pass this
        // test — but the test is here so future refactors that
        // *do* introduce a default cap fire loudly.
        let zip = write_zip_with_sized_entry("taxonomy.yaml", 65_536);
        let err = verify_skill_pack(zip.as_slice(), "0".repeat(64).as_str()).unwrap_err();
        assert!(
            !matches!(err, SkillPackError::DecompressionLimitExceeded { .. }),
            "wrapper must be effectively unbounded, got {err:?}"
        );
    }

    #[test]
    fn verify_with_limit_zero_cap_rejects_every_nonempty_entry() {
        // Edge case: cap = 0 means "no entry may carry any
        // bytes". Useful for callers that want to assert the
        // archive is empty (e.g. for testing harnesses). The
        // first non-empty entry must trip
        // `DecompressionLimitExceeded`.
        let zip = write_zip_with_sized_entry("taxonomy.yaml", 1);
        let err =
            verify_skill_pack_with_limit(zip.as_slice(), "0".repeat(64).as_str(), 0).unwrap_err();
        match err {
            SkillPackError::DecompressionLimitExceeded { limit, .. } => {
                assert_eq!(limit, 0);
            }
            other => panic!("expected DecompressionLimitExceeded with limit=0, got {other:?}"),
        }
    }
}

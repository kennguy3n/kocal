//! Skill-pack-specific error type.
//!
//! Every fallible operation in [`super::loader`] and
//! [`super::verifier`] returns a [`SkillPackError`]. The variants
//! are intentionally narrow and structured so that hosts can do
//! source-attribution (e.g. "the manifest failed schema validation"
//! vs "the signature did not verify") without parsing the display
//! text. Mirrors Python's `SkillPackError(ValueError)` plus the
//! ad-hoc error messages used by `shared/skillpack/loader.py` and
//! `shared/skillpack/verifier.py`.

use std::error::Error;
use std::fmt;

use crate::crypto::Ed25519VerifyError;

/// Top-level error type for skill-pack operations.
#[derive(Debug)]
pub enum SkillPackError {
    /// The raw archive bytes failed to parse as a ZIP file. Wraps
    /// the underlying [`zip::result::ZipError`].
    InvalidZip(zip::result::ZipError),
    /// A required file is missing from the archive. The string
    /// carries the canonical path that was looked up (e.g.
    /// `"manifest.json"`, `"taxonomy.yaml"`).
    MissingFile(String),
    /// A YAML / JSON document inside the archive could not be
    /// deserialized into its closed-shape Rust type.
    SchemaViolation {
        /// Path of the offending file inside the archive.
        path: String,
        /// Human-readable detail about the parse / validation
        /// failure.
        detail: String,
    },
    /// The manifest claimed a content digest that did not match
    /// the recomputed digest of the archive's file bytes. Carries
    /// the hex-encoded manifest-declared digest and the
    /// recomputed digest so the host can log both for incident
    /// triage.
    ContentDigestMismatch {
        /// The `content_sha256` value asserted by the manifest.
        declared: String,
        /// The SHA-256 the verifier recomputed over the actual
        /// pack contents.
        recomputed: String,
    },
    /// The manifest declared a public key that did not match the
    /// pinned key passed to the verifier. Strings carry the
    /// 64-char hex encoding of each key.
    PinnedKeyMismatch {
        /// The hex public key the caller pinned.
        pinned: String,
        /// The hex public key the manifest declared.
        declared: String,
    },
    /// The ed25519 signature did not verify against the pinned
    /// key. Wraps the underlying [`Ed25519VerifyError`] for
    /// detailed logging.
    SignatureInvalid(Ed25519VerifyError),
    /// The manifest is structurally well-formed but lacks the
    /// `signature` or `public_key` field — i.e. it is an
    /// unsigned development build.
    Unsigned,
    /// A skill pack referenced a label / category / path that
    /// violates a cross-file invariant (e.g. taxonomy declares
    /// `adult.nudity` but `thresholds.yaml` carries no entry for
    /// `adult.nudity`). These checks are deferred to per-deserializer
    /// validators; this variant is reserved for the eventual
    /// cross-file validator pass.
    CrossFileViolation(String),
    /// A single zip entry's declared uncompressed size, or the
    /// observed decompressed bytes, exceeded the caller-supplied
    /// limit passed to
    /// [`super::verify_skill_pack_with_limit`]. Defense in depth
    /// against deflate-bomb archives — only fires when the
    /// caller opts in by passing a finite cap. Today's signed
    /// skill packs are well below any practical limit, but a
    /// host running on a memory-constrained device can pin a
    /// ceiling here without trusting the archive's own size
    /// declaration.
    DecompressionLimitExceeded {
        /// Path of the offending entry inside the archive.
        path: String,
        /// The uncompressed size the entry declared (or the
        /// running total observed during streaming read,
        /// whichever crossed the limit first).
        declared: u64,
        /// The caller-supplied byte cap.
        limit: u64,
    },
    /// A community or jurisdiction overlay attempted to weaken a
    /// threshold inside a *protected* category (currently
    /// `child_safety`). The floor is PROPOSAL §10 — jurisdictions
    /// and communities are allowed
    /// to *tighten* child safety (lower `trigger` / `severe`), but
    /// never to relax it past the global baseline. Mirrors the
    /// `SkillPackError` raised inside Python's
    /// `_check_protected_floor`.
    OverlayFloorViolation {
        /// Protected category that was being overridden
        /// (e.g. `"child_safety"`).
        category: String,
        /// Label inside the category (e.g. `"any_hit"`).
        label: String,
        /// Human-readable detail about which side of the floor
        /// was crossed and by how much (e.g. `"trigger 0.40 > base
        /// 0.20"` or `"cannot clear severe floor on child_safety.any_hit"`).
        detail: String,
    },
}

impl fmt::Display for SkillPackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidZip(e) => write!(f, "skill pack zip is malformed: {e}"),
            Self::MissingFile(path) => {
                write!(f, "skill pack is missing required file: {path}")
            }
            Self::SchemaViolation { path, detail } => {
                write!(f, "schema violation in {path}: {detail}")
            }
            Self::ContentDigestMismatch {
                declared,
                recomputed,
            } => write!(
                f,
                "content_sha256 mismatch: manifest={declared}, recomputed={recomputed}"
            ),
            Self::PinnedKeyMismatch { pinned, declared } => write!(
                f,
                "pinned public key mismatch: pinned={pinned}, manifest_declared={declared}"
            ),
            Self::SignatureInvalid(e) => write!(f, "ed25519 signature verification failed: {e}"),
            Self::Unsigned => write!(
                f,
                "skill pack manifest is not signed (signature or public_key missing)"
            ),
            Self::CrossFileViolation(detail) => {
                write!(f, "cross-file skill-pack invariant violated: {detail}")
            }
            Self::DecompressionLimitExceeded {
                path,
                declared,
                limit,
            } => write!(
                f,
                "decompression limit exceeded for {path}: declared/observed={declared} bytes, limit={limit} bytes"
            ),
            Self::OverlayFloorViolation {
                category,
                label,
                detail,
            } => write!(
                f,
                "overlay would loosen {category} floor on {category}.{label}: {detail}"
            ),
        }
    }
}

impl Error for SkillPackError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidZip(e) => Some(e),
            Self::SignatureInvalid(e) => Some(e),
            _ => None,
        }
    }
}

impl From<zip::result::ZipError> for SkillPackError {
    fn from(value: zip::result::ZipError) -> Self {
        Self::InvalidZip(value)
    }
}

impl From<Ed25519VerifyError> for SkillPackError {
    fn from(value: Ed25519VerifyError) -> Self {
        Self::SignatureInvalid(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_display_includes_path() {
        let err = SkillPackError::MissingFile("manifest.json".to_string());
        let s = format!("{err}");
        assert!(s.contains("manifest.json"), "got: {s}");
        assert!(s.contains("missing"), "got: {s}");
    }

    #[test]
    fn schema_violation_carries_path_and_detail() {
        let err = SkillPackError::SchemaViolation {
            path: "taxonomy.yaml".to_string(),
            detail: "labels must be non-empty".to_string(),
        };
        let s = format!("{err}");
        assert!(s.contains("taxonomy.yaml"), "got: {s}");
        assert!(s.contains("labels must be non-empty"), "got: {s}");
    }

    #[test]
    fn content_digest_mismatch_shows_both_hashes() {
        let err = SkillPackError::ContentDigestMismatch {
            declared: "aa".to_string(),
            recomputed: "bb".to_string(),
        };
        let s = format!("{err}");
        assert!(s.contains("manifest=aa"), "got: {s}");
        assert!(s.contains("recomputed=bb"), "got: {s}");
    }

    #[test]
    fn pinned_key_mismatch_shows_both_keys() {
        let err = SkillPackError::PinnedKeyMismatch {
            pinned: "00".to_string(),
            declared: "ff".to_string(),
        };
        let s = format!("{err}");
        assert!(s.contains("pinned=00"), "got: {s}");
        assert!(s.contains("manifest_declared=ff"), "got: {s}");
    }

    #[test]
    fn unsigned_variant_is_distinct() {
        let err = SkillPackError::Unsigned;
        let s = format!("{err}");
        assert!(s.contains("not signed"), "got: {s}");
    }

    #[test]
    fn error_source_chain_includes_zip_error() {
        // construct a zip error by reading garbage bytes
        use std::io::Cursor;
        let res = zip::ZipArchive::new(Cursor::new(b"not a zip"));
        let zip_err = res.expect_err("invalid zip should error");
        let wrapped: SkillPackError = zip_err.into();
        // The `source()` chain should expose the inner zip error.
        assert!(wrapped.source().is_some(), "no source on InvalidZip");
    }

    #[test]
    fn cross_file_violation_renders_detail() {
        let err =
            SkillPackError::CrossFileViolation("threshold references unknown label".to_string());
        let s = format!("{err}");
        assert!(s.contains("threshold references unknown label"), "got: {s}");
    }

    #[test]
    fn debug_format_includes_variant_name() {
        let err = SkillPackError::Unsigned;
        let s = format!("{err:?}");
        assert!(s.contains("Unsigned"), "got: {s}");
    }
}

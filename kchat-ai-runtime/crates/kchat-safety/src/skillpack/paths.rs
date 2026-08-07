//! Canonical filesystem layout of a compiled `.cvguard-skill.zip` archive.
//!
//! These constants mirror the path strings hard-coded in
//! cv-guard's `shared/skillpack/compiler.py` exactly. They are
//! re-exported through the crate root so callers that want to
//! introspect a pack archive (for diagnostics or a custom loader)
//! can pin against the same literal strings the signer used.
//!
//! The signer / runtime contract is that every file in a pack
//! lives at exactly one of these top-level paths (or under
//! [`SCAM_PHRASES_DIR`] / [`HATE_LEXICON_DIR`] / [`REGEX_DIR`]),
//! and every YAML inside a sub-directory uses the `.yaml` suffix.
//! The [`super::verifier`] enforces both invariants before any
//! deserialization is attempted.

/// Path to the per-pack severity rubric inside a compiled skill
/// pack. The YAML payload mirrors
/// [`crate::policy_interpreter::SeverityRubric`] one-for-one.
pub const RUBRIC_PATH: &str = "severity_rubric.yaml";

/// Path to the taxonomy declaration inside a compiled skill
/// pack. The YAML payload mirrors
/// [`super::schema::TaxonomyConfig`] one-for-one.
pub const TAXONOMY_PATH: &str = "taxonomy.yaml";

/// Path to the per-label threshold table inside a compiled
/// skill pack. The YAML payload mirrors
/// [`crate::policy_interpreter::ThresholdsConfig`] one-for-one.
pub const THRESHOLDS_PATH: &str = "thresholds.yaml";

/// Path to the SLM system prompt template inside a compiled
/// skill pack. Stored as raw UTF-8 (no YAML), decoded by the
/// loader straight into a [`String`].
pub const SLM_PROMPT_PATH: &str = "slm_prompt.txt";

/// Top-level sub-directory containing per-language scam phrase
/// lexicons. Each `.yaml` file maps to one
/// [`super::schema::Lexicon`]; the basename (sans `.yaml`) is
/// used as the lookup key on
/// [`super::schema::SkillPack::scam_phrases`].
pub const SCAM_PHRASES_DIR: &str = "scam_phrases";

/// Top-level sub-directory containing per-language hate-speech
/// lexicons. Each `.yaml` file maps to one
/// [`super::schema::Lexicon`]; the basename (sans `.yaml`) is
/// used as the lookup key on
/// [`super::schema::SkillPack::hate_lexicons`].
pub const HATE_LEXICON_DIR: &str = "hate_lexicon";

/// Top-level sub-directory containing named regex sets. Each
/// `.yaml` file maps to one [`super::schema::RegexSet`]; the
/// basename (sans `.yaml`) is used as the lookup key on
/// [`super::schema::SkillPack::regex_sets`].
pub const REGEX_DIR: &str = "regex";

/// Path to the optional known-hash database (perceptual-hash
/// gate) inside a compiled skill pack. The base loader does not
/// decode this file; it is read by the optional WS6D layer.
pub const KNOWN_HASHES_PATH: &str = "known_hashes.bin";

#[cfg(test)]
mod tests {
    use super::*;

    /// Lock the python-side strings against accidental drift. If
    /// the Python compiler ever changes one of these literals the
    /// fixture-generated test pack will fail to deserialize on
    /// the Rust side and this test will start failing in lockstep
    /// — leaving an obvious breadcrumb pointing at the renamed
    /// constant.
    #[test]
    fn canonical_paths_match_python_reference() {
        assert_eq!(RUBRIC_PATH, "severity_rubric.yaml");
        assert_eq!(TAXONOMY_PATH, "taxonomy.yaml");
        assert_eq!(THRESHOLDS_PATH, "thresholds.yaml");
        assert_eq!(SLM_PROMPT_PATH, "slm_prompt.txt");
        assert_eq!(SCAM_PHRASES_DIR, "scam_phrases");
        assert_eq!(HATE_LEXICON_DIR, "hate_lexicon");
        assert_eq!(REGEX_DIR, "regex");
        assert_eq!(KNOWN_HASHES_PATH, "known_hashes.bin");
        // The manifest path lives on the digest module so the
        // crypto layer can recompute the content digest without
        // a circular dep on the skill-pack module.
        assert_eq!(crate::crypto::digest::MANIFEST_PATH, "manifest.json");
    }
}

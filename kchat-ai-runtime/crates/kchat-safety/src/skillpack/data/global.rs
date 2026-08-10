//! Global baseline data — taxonomy, severity rubric, privacy contract, vision
//! baselines, schemas, and transliteration map.
//!
//! These files are embedded from `files/global/`.

/// `files/global/taxonomy.yaml` — 17-category risk taxonomy (IDs 0-16).
pub const TAXONOMY_YAML: &str = include_str!("files/global/taxonomy.yaml");

/// `files/global/baseline.yaml` — global baseline skill definition.
pub const BASELINE_YAML: &str = include_str!("files/global/baseline.yaml");

/// `files/global/severity.yaml` — 0-5 severity rubric.
pub const SEVERITY_YAML: &str = include_str!("files/global/severity.yaml");

/// `files/global/privacy_contract.yaml` — 8 non-negotiable privacy rules.
pub const PRIVACY_CONTRACT_YAML: &str =
    include_str!("files/global/privacy_contract.yaml");

/// `files/global/output_schema.json` — encoder output JSON schema.
pub const OUTPUT_SCHEMA_JSON: &str =
    include_str!("files/global/output_schema.json");

/// `files/global/local_signal_schema.json` — encoder input signal schema.
pub const LOCAL_SIGNAL_SCHEMA_JSON: &str =
    include_str!("files/global/local_signal_schema.json");

/// `files/global/vision_baseline.yaml` — vision safety baseline.
pub const VISION_BASELINE_YAML: &str =
    include_str!("files/global/vision_baseline.yaml");

/// `files/global/vision_image_exemplars.yaml` — vision image exemplars.
pub const VISION_IMAGE_EXEMPLARS_YAML: &str =
    include_str!("files/global/vision_image_exemplars.yaml");

/// `files/global/vision_prototypes_source.yaml` — vision prototype sources.
pub const VISION_PROTOTYPES_YAML: &str =
    include_str!("files/global/vision_prototypes_source.yaml");

/// `files/global/transliteration/translit_core_v1.json` — transliteration map.
pub const TRANSLIT_CORE_JSON: &str =
    include_str!("files/global/transliteration/translit_core_v1.json");

/// `files/test_suite_template.yaml` — canonical test-suite template with 13 metrics.
pub const TEST_SUITE_TEMPLATE_YAML: &str =
    include_str!("files/test_suite_template.yaml");

/// `files/adversarial_corpus.yaml` — adversarial/obfuscation test corpus (60 cases, 6 techniques).
pub const ADVERSARIAL_CORPUS_YAML: &str =
    include_str!("files/adversarial_corpus.yaml");

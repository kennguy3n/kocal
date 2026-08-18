//! Safety classifier — the main entry point for the safety plane.
//!
//! Implements the 6-step guardrail flow:
//! 1. Normalize text locally after decryption
//! 2. Apply signed deterministic policy, allowlists, blocklists, rate signals
//! 3. If confidence is insufficient, run the compact encoder
//! 4. On eligible medium/high devices, invoke the SLM for ambiguous cases
//! 5. Apply deterministic policy to the structured result
//! 6. Return allow, warn, block, redact, or require-consent with reason codes
//!
//! The deterministic path P95 target is <5ms. The encoder path P95 target
//! is <150ms on qualified devices.

use crate::detectors;
use crate::media::MediaDescriptor;
use crate::normalize;
use crate::policy::{PolicyPack, PolicyThresholds};
use crate::verdict::{Action, Severity, Verdict, VerdictBuilder, VerdictSource};
use parking_lot::RwLock;
use unicode_normalization::UnicodeNormalization;
use std::sync::Arc;
use std::time::Instant;

/// Maximum input text length accepted by [`SafetyClassifier::classify`].
///
/// Inputs exceeding this length are truncated before normalization to
/// prevent latency spikes on the deterministic hot path (<5ms P95 target).
/// 32 KiB is well above any legitimate chat message while bounding the
/// cost of NFKC normalization + regex passes + lexicon scans.
pub const MAX_INPUT_LEN: usize = 32 * 1024;

/// Request for safety classification.
#[derive(Debug, Clone)]
pub struct ClassifyRequest {
    /// Message text (already decrypted)
    pub text: String,
    /// Whether this is a group conversation
    pub is_group: bool,
    /// Age mode if applicable (e.g. "minor", "adult")
    pub age_mode: Option<String>,
    /// Relationship context if known
    pub relationship: Option<String>,
    /// Whether the encoder is available (medium+ tier)
    pub encoder_available: bool,
    /// Whether the SLM is available (medium+ tier)
    pub slm_available: bool,
    /// Whether the user explicitly quoted another message (protected-speech
    /// context). Set by the chat client, not user content — spoof-resistant.
    pub quoted_from_user: bool,
    /// Community overlay id, if any. Used to derive NEWS/EDUCATION/
    /// COUNTERSPEECH context hints via substring match.
    pub community_overlay_id: Option<String>,
    /// Jurisdiction code (e.g. "us", "vn", "eu") for jurisdiction overlay
    /// resolution. When set, the classifier loads the matching jurisdiction
    /// overlay from embedded data (`skillpack/data/files/jurisdictions/{code}/overlay.yaml`).
    pub jurisdiction: Option<String>,
    /// Locale tag (e.g. "en-US", "vi-VN") for language-asset selection.
    pub locale: Option<String>,
    /// Media descriptors from on-device vision models (image/video safety scores).
    /// When present, the priority chain checks media branches before text signals.
    pub media_descriptors: Vec<MediaDescriptor>,
}

impl ClassifyRequest {
    /// Simple request from just text (deterministic-only).
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            is_group: false,
            age_mode: None,
            relationship: None,
            encoder_available: false,
            slm_available: false,
            quoted_from_user: false,
            community_overlay_id: None,
            jurisdiction: None,
            locale: None,
            media_descriptors: Vec::new(),
        }
    }

    /// Enable encoder and SLM (medium/high tier).
    pub fn with_encoder(mut self) -> Self {
        self.encoder_available = true;
        self.slm_available = true;
        self
    }

    /// Mark this message as quoting another user (protected-speech context).
    pub fn with_quoted(mut self) -> Self {
        self.quoted_from_user = true;
        self
    }

    /// Set the community overlay id for context-hint derivation.
    pub fn with_overlay(mut self, overlay: impl Into<String>) -> Self {
        self.community_overlay_id = Some(overlay.into());
        self
    }

    /// Set the jurisdiction code for jurisdiction overlay resolution.
    pub fn with_jurisdiction(mut self, jurisdiction: impl Into<String>) -> Self {
        self.jurisdiction = Some(jurisdiction.into());
        self
    }

    /// Set the locale tag for language-asset selection.
    pub fn with_locale(mut self, locale: impl Into<String>) -> Self {
        self.locale = Some(locale.into());
        self
    }

    /// Attach media descriptors from on-device vision models.
    pub fn with_media(mut self, media: Vec<MediaDescriptor>) -> Self {
        self.media_descriptors = media;
        self
    }
}

// ---------------------------------------------------------------------------
// Protected-speech context hints. Ported from slm-guardrail's
// `pipeline/context.rs` — 4 hint types with per-hint confidence.
// ---------------------------------------------------------------------------

/// Minimum context confidence required to fully demote a non-SAFE,
/// non-CHILD_SAFETY verdict to Allow.
pub const CONTEXT_DEMOTION_CONFIDENCE_THRESHOLD: f64 = 0.5;

/// A protected-speech context hint with confidence.
#[derive(Debug, Clone)]
pub struct ContextHint {
    pub reason_code: String,
    pub context_confidence: f64,
}

/// Overlay-id substring tokens that imply a news-coverage context.
const NEWS_CONTEXT_OVERLAY_TOKENS: &[&str] = &["journalism", "news"];

/// Overlay-id substring tokens that imply an educational context.
const EDUCATION_CONTEXT_OVERLAY_TOKENS: &[&str] = &[
    "education_higher", "education", "school", "research", "science",
];

/// Overlay-id substring tokens that imply a counterspeech / civic-rights
/// context. NOTE: bare "tolerance" is intentionally excluded — would
/// false-fire on "zero_tolerance" overlays (the opposite of counterspeech).
const COUNTERSPEECH_CONTEXT_OVERLAY_TOKENS: &[&str] = &[
    "lgbtq_support", "minority_support", "civic", "humanrights",
    "human_rights", "counterspeech", "counter_speech", "anti_racism",
    "antiracism", "anti_hate", "antihate", "anti_bullying", "antibullying",
];

/// Derive protected-speech context hints from request fields.
///
/// Returns one hint per matching signal, each with its default confidence:
/// - `QUOTED_SPEECH_CONTEXT` (0.7) — from `quoted_from_user` structural flag
/// - `NEWS_CONTEXT` (0.5) — from overlay-id substring match
/// - `EDUCATION_CONTEXT` (0.5) — from overlay-id substring match
/// - `COUNTERSPEECH_CONTEXT` (0.5) — from overlay-id substring match
pub fn derive_context_hints(
    quoted_from_user: bool,
    community_overlay_id: Option<&str>,
) -> Vec<ContextHint> {
    let mut hints = Vec::new();
    if quoted_from_user {
        hints.push(ContextHint {
            reason_code: "QUOTED_SPEECH_CONTEXT".into(),
            context_confidence: 0.7,
        });
    }
    if let Some(overlay) = community_overlay_id {
        let lower = overlay.to_ascii_lowercase();
        if NEWS_CONTEXT_OVERLAY_TOKENS.iter().any(|t| lower.contains(t)) {
            hints.push(ContextHint {
                reason_code: "NEWS_CONTEXT".into(),
                context_confidence: 0.5,
            });
        }
        if EDUCATION_CONTEXT_OVERLAY_TOKENS.iter().any(|t| lower.contains(t)) {
            hints.push(ContextHint {
                reason_code: "EDUCATION_CONTEXT".into(),
                context_confidence: 0.5,
            });
        }
        if COUNTERSPEECH_CONTEXT_OVERLAY_TOKENS.iter().any(|t| lower.contains(t)) {
            hints.push(ContextHint {
                reason_code: "COUNTERSPEECH_CONTEXT".into(),
                context_confidence: 0.5,
            });
        }
    }
    hints
}

/// Check if a verdict should be demoted to Allow under protected-speech
/// context hints. CHILD_SAFETY (category 1) is never demoted.
fn should_demote_for_protected_speech(
    category: u32,
    hints: &[ContextHint],
) -> Option<&ContextHint> {
    // Critical safety categories are never demoted — defense in depth.
    // CHILD_SAFETY, SELF_HARM, VIOLENCE_THREAT, MISINFORMATION_HEALTH, and
    // DEEPFAKE_SYNTHETIC represent direct harm signals that should not be
    // suppressed by community context (news, education, counterspeech).
    if category == detectors::categories::CHILD_SAFETY
        || category == detectors::categories::SELF_HARM
        || category == detectors::categories::VIOLENCE_THREAT
        || category == detectors::categories::MISINFORMATION_HEALTH
        || category == detectors::categories::DEEPFAKE_SYNTHETIC
    {
        return None;
    }
    hints.iter().find(|h| h.context_confidence >= CONTEXT_DEMOTION_CONFIDENCE_THRESHOLD)
}

/// Content-based educational/research context detection. Recognizes phrases
/// that indicate the user is writing about, studying, or researching a topic
/// for educational purposes — not attempting to carry out a harmful act.
/// Used to demote non-critical safety verdicts to Allow for protected speech.
fn detect_educational_context(text: &str) -> bool {
    use regex::Regex;
    static EDUCATIONAL_RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re = EDUCATIONAL_RE.get_or_init(|| {
        Regex::new(
            r"(?i)(?:security\s+research|research\s+paper|for\s+educational\s+purposes|for\s+educational|for\s+my\s+thesis|for\s+my\s+dissertation|for\s+my\s+research|criminal\s+justice\s+class|for\s+my\s+class|studying\s+cyber|need\s+to\s+understand\s+(?:how|the|what)|writing\s+(?:a\s+)?(?:paper|article|thesis|dissertation|summary|report|legal\s+brief|法律摘要)|i'm\s+studying|i\s+am\s+studying|academic\s+(?:paper|research|purposes)|for\s+academic|for\s+research|as\s+a\s+(?:social\s+)?worker|論文|for\s+my\s+skripsi|buat\s+skripsi|buat\s+tujuan\s+edukasi|para\s+sa\s+educational|para\s+sa\s+thesis|estudio\s+ciber|estudo\s+ciber|ich\s+studiere|étudie\s+la\s+cyber|estudio\s+ciberseguridad|estudo\s+cibersegurança|t\s+đang\s+học|正在学习|正在撰写|勉強して|공부하|학습|เรียน|belajar|nagsusulat|nag-aaral|akademik|akademisch|académico|acadêmico|أكتب\s+مقالا|أدرس)"
        ).unwrap()
    });
    re.is_match(text)
}

/// Result of safety classification.
#[derive(Debug, Clone)]
pub struct ClassifyResult {
    /// The final verdict
    pub verdict: Verdict,
    /// Time taken in microseconds (for telemetry, not content)
    pub duration_us: u64,
}

/// Maximum number of policy packs that can be loaded. Prevents unbounded
/// linear scan in `build_lexicon_with_overlays` and `get_thresholds`.
pub const MAX_POLICY_PACKS: usize = 16;

/// The safety classifier — owns loaded policy packs and optional encoder/SLM.
pub struct SafetyClassifier {
    policy_packs: RwLock<Vec<Arc<PolicyPack>>>,
    /// Optional encoder (ONNX classifier) — set on medium+ devices
    encoder: RwLock<Option<Box<dyn EncoderAdapter>>>,
    /// Optional SLM adjudicator — set on medium+ devices
    slm: RwLock<Option<Box<dyn SlmAdjudicator>>>,
    /// Cached lexicon keyed by `(pack_count, jurisdiction, community_overlay_id)`.
    /// Invalidated when a new policy pack is loaded.
    lexicon_cache: RwLock<Option<(usize, Option<String>, Option<String>, Vec<(String, u32, Severity)>)>>,
}

/// Trait for encoder-based classification (ONNX INT8/INT4).
pub trait EncoderAdapter: Send + Sync {
    /// Classify text and return a verdict.
    fn classify(&self, text: &str) -> Result<EncoderVerdict, EncoderError>;
}

/// Verdict from the encoder.
#[derive(Debug, Clone)]
pub struct EncoderVerdict {
    pub category: u32,
    pub confidence: f64,
}

/// Error from the encoder.
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    #[error("encoder inference failed: {0}")]
    InferenceFailed(String),
    #[error("encoder not loaded")]
    NotLoaded,
}

/// Trait for SLM-based adjudication (llama.cpp with grammar-constrained JSON).
pub trait SlmAdjudicator: Send + Sync {
    /// Adjudicate an ambiguous case and return a structured decision.
    fn adjudicate(&self, text: &str, signal_json: &str) -> Result<SlmDecision, SlmError>;
}

/// SLM decision (closed JSON grammar output).
#[derive(Debug, Clone)]
pub struct SlmDecision {
    pub category: u32,
    pub severity: u8,
    pub action: Action,
    pub confidence: f64,
    pub rationale_code: String,
}

/// Error from the SLM.
#[derive(Debug, thiserror::Error)]
pub enum SlmError {
    #[error("SLM inference failed: {0}")]
    InferenceFailed(String),
    #[error("SLM not loaded")]
    NotLoaded,
    #[error("SLM output invalid: {0}")]
    InvalidOutput(String),
}

impl SafetyClassifier {
    /// Create a new classifier with no policy packs (deterministic-only mode).
    pub fn new() -> Self {
        Self {
            policy_packs: RwLock::new(Vec::new()),
            encoder: RwLock::new(None),
            slm: RwLock::new(None),
            lexicon_cache: RwLock::new(None),
        }
    }

    /// Load a signed policy pack.
    ///
    /// Returns `false` if the maximum pack count ([`MAX_POLICY_PACKS`]) has been reached.
    pub fn load_policy_pack(&self, pack: Arc<PolicyPack>) -> bool {
        let mut packs = self.policy_packs.write();
        if packs.len() >= MAX_POLICY_PACKS {
            return false;
        }
        packs.push(pack);
        *self.lexicon_cache.write() = None;
        true
    }

    /// Attach an encoder (ONNX classifier) — medium+ tier only.
    pub fn attach_encoder(&self, encoder: Box<dyn EncoderAdapter>) {
        *self.encoder.write() = Some(encoder);
    }

    /// Attach an SLM adjudicator — medium+ tier only.
    pub fn attach_slm(&self, slm: Box<dyn SlmAdjudicator>) {
        *self.slm.write() = Some(slm);
    }

    /// Classify a message through the full guardrail pipeline.
    ///
    /// This is the main entry point. It runs the 6-step flow:
    /// 1. Normalize
    /// 2. Deterministic rules
    /// 3. Encoder (if needed and available)
    /// 4. SLM (if needed and available)
    /// 5. Deterministic policy on result
    /// 6. Return verdict
    pub fn classify(&self, request: &ClassifyRequest) -> ClassifyResult {
        let start = Instant::now();

        // Truncate oversized inputs to bound normalization + regex cost.
        let text = if request.text.len() > MAX_INPUT_LEN {
            &request.text[..MAX_INPUT_LEN]
        } else {
            &request.text
        };

        // Step 1: Normalize — multiple views for different detector types
        let pattern_text = normalize::normalize_for_patterns(text);
        let lexicon_base = normalize::normalize_for_lexicon(text);

        // Build search views: the non-leetspeak base + defanged variants.
        // Scam patterns use `!` and `.` which leetspeak corrupts, so the
        // non-leetspeak base is always included. Defang variants add
        // leetspeak readings (digit `1` → `l` or `i`) for lexicon matching.
        let defang_variants = normalize::defang_variants_for_matching(&lexicon_base);
        let mut lexicon_views: Vec<String> = Vec::with_capacity(defang_variants.len() + 1);
        lexicon_views.push(lexicon_base.clone());
        for v in &defang_variants {
            if !lexicon_views.contains(v) {
                lexicon_views.push(v.clone());
            }
        }

        // Step 2: Run deterministic detectors across all views
        let lexicon = self.build_lexicon_with_overlays(
            request.jurisdiction.as_deref(),
            request.community_overlay_id.as_deref(),
        );
        let mut signals = detectors::run_all_detectors(&pattern_text, &lexicon_views, &lexicon);

        // Attach media descriptors from the request (on-device vision scores)
        if !request.media_descriptors.is_empty() {
            signals.media_descriptors = request.media_descriptors.clone();
        }

        // Derive protected-speech context hints from request fields.
        let context_hints = derive_context_hints(
            request.quoted_from_user,
            request.community_overlay_id.as_deref(),
        );

        // Community-aware reclassification: when a community overlay is present
        // and the winning signal is SCAM_FRAUD from a regex detector, check if
        // there's also a COMMUNITY_RULE lexicon signal. Community overlays can
        // reclassify promotional spam (e.g., "crypto trading service" in a
        // workplace) as a community rule violation rather than outright scam.
        let resolved_signal = {
            let winner = detectors::resolve_priority_chain(&signals);
            if let Some(ref sig) = winner {
                if sig.category == detectors::categories::SCAM_FRAUD
                    && sig.reason_code.starts_with("scam_")
                    && request.community_overlay_id.is_some()
                {
                    // Look for a COMMUNITY_RULE lexicon signal
                    let community_signal = signals.signals.iter()
                        .find(|s| s.category == detectors::categories::COMMUNITY_RULE);
                    if let Some(cs) = community_signal {
                        // Check if the community overlay is a type that would
                        // reclassify promotional content (workplace, school, etc.)
                        let overlay_lower = request.community_overlay_id
                            .as_ref()
                            .map(|s| s.to_ascii_lowercase())
                            .unwrap_or_default();
                        let reclassifying = ["workplace", "school", "education", "professional"];
                        if reclassifying.iter().any(|r| overlay_lower.contains(r)) {
                            Some(cs.clone())
                        } else {
                            winner
                        }
                    } else {
                        winner
                    }
                } else {
                    winner
                }
            } else {
                winner
            }
        };

        // Resolve deterministic verdict
        let verdict = if let Some(signal) = resolved_signal {
            // Check protected-speech demotion BEFORE building the verdict.
            // CHILD_SAFETY is never demoted — defense in depth.
            if let Some(hint) = should_demote_for_protected_speech(signal.category, &context_hints) {
                VerdictBuilder::default()
                    .action(Action::Allow)
                    .severity(Severity::SAFE)
                    .category(detectors::categories::SAFE)
                    .confidence(0.90)
                    .reason_code(&format!("protected_speech_{}", hint.reason_code.to_lowercase()))
                    .source(VerdictSource::Deterministic)
                    .build()
            } else if detect_educational_context(text)
                && signal.category != detectors::categories::CHILD_SAFETY
                && signal.category != detectors::categories::SELF_HARM
                && signal.category != detectors::categories::VIOLENCE_THREAT
                && signal.category != detectors::categories::MISINFORMATION_HEALTH
                && signal.category != detectors::categories::DEEPFAKE_SYNTHETIC
                && signal.reason_code != "prompt_injection"
            {
                // Content-based educational context demotion — the text contains
                // phrases indicating research/educational context (e.g., "for my
                // thesis", "security research paper", "studying cybersecurity").
                // Demote non-critical categories to Allow for protected speech.
                VerdictBuilder::default()
                    .action(Action::Allow)
                    .severity(Severity::SAFE)
                    .category(detectors::categories::SAFE)
                    .confidence(0.85)
                    .reason_code("protected_speech_educational_context")
                    .source(VerdictSource::Deterministic)
                    .build()
            } else {
            // Deterministic match found
            let mut builder = VerdictBuilder::default()
                .action(signal.action)
                .severity(signal.severity)
                .category(signal.category)
                .confidence(signal.confidence)
                .reason_code(&signal.reason_code)
                .source(VerdictSource::Deterministic);

            // Step 3: If confidence is below the encoder escalation threshold,
            // and encoder is available, escalate
            let thresholds = self.get_thresholds();
            if signal.confidence < thresholds.encoder_escalation_threshold
                && request.encoder_available
            {
                if let Some(encoder) = self.encoder.read().as_ref() {
                    if let Ok(enc_verdict) = encoder.classify(&pattern_text) {
                        builder = builder
                            .used_encoder(true)
                            .confidence(enc_verdict.confidence)
                            .category(enc_verdict.category);

                        // Step 4: If still ambiguous and SLM is available, adjudicate
                        if enc_verdict.confidence < thresholds.warn_threshold
                            && request.slm_available
                        {
                            if let Some(slm) = self.slm.read().as_ref() {
                                let signal_json = serde_json::json!({
                                    "category": enc_verdict.category,
                                    "confidence": enc_verdict.confidence,
                                    "is_group": request.is_group,
                                    "age_mode": request.age_mode,
                                })
                                .to_string();

                                if let Ok(slm_decision) = slm.adjudicate(&pattern_text, &signal_json) {
                                    builder = builder
                                        .used_slm(true)
                                        .action(slm_decision.action)
                                        .severity(Severity(slm_decision.severity))
                                        .confidence(slm_decision.confidence)
                                        .reason_code(&slm_decision.rationale_code)
                                        .source(VerdictSource::Slm);
                                }
                            }
                        } else {
                            builder = builder.source(VerdictSource::Encoder);
                        }
                    }
                }
            }

            builder.build()
            }
        } else {
            // No deterministic match — check if we need encoder for safety
            let mut degraded = false;
            let thresholds = self.get_thresholds();

            // For certain contexts (group, minor), always run encoder if available
            let needs_encoder = request.encoder_available
                && (request.is_group
                    || request.age_mode.as_deref() == Some("minor")
                    || self.has_high_risk_indicators(&lexicon_views[0]));

            if needs_encoder {
                if let Some(encoder) = self.encoder.read().as_ref() {
                    match encoder.classify(&pattern_text) {
                        Ok(enc_verdict) => {
                            let action = if enc_verdict.confidence >= thresholds.block_threshold {
                                Action::Block
                            } else if enc_verdict.confidence >= thresholds.warn_threshold {
                                Action::Warn
                            } else {
                                Action::Allow
                            };

                            let severity = if enc_verdict.confidence >= thresholds.block_threshold {
                                Severity::SEVERE
                            } else if enc_verdict.confidence >= thresholds.warn_threshold {
                                Severity::BORDERLINE
                            } else {
                                Severity::SAFE
                            };

                            return ClassifyResult {
                                verdict: VerdictBuilder::default()
                                    .action(action)
                                    .severity(severity)
                                    .category(enc_verdict.category)
                                    .confidence(enc_verdict.confidence)
                                    .reason_code("encoder_classification")
                                    .source(VerdictSource::Encoder)
                                    .used_encoder(true)
                                    .build(),
                                duration_us: start.elapsed().as_micros() as u64,
                            };
                        }
                        Err(_) => {
                            // Encoder failed — mark as degraded and fall through
                            // to deterministic verdict
                            degraded = true;
                        }
                    }
                }
            }

            // No match and no encoder needed → allow (or degraded if encoder failed)
            if degraded {
                VerdictBuilder::default()
                    .action(Action::Allow)
                    .source(VerdictSource::Degraded)
                    .build()
            } else {
                Verdict::allow()
            }
        };

        ClassifyResult {
            verdict,
            duration_us: start.elapsed().as_micros() as u64,
        }
    }

    /// Build a lexicon from all loaded policy packs, plus scam phrases from
    /// embedded jurisdiction and community overlays (when the `skill-pack`
    /// feature is enabled and the request specifies overlay IDs).
    fn build_lexicon_with_overlays(
        &self,
        jurisdiction: Option<&str>,
        community_overlay_id: Option<&str>,
    ) -> Vec<(String, u32, Severity)> {
        let pack_count = {
            let packs = self.policy_packs.read();
            packs.len()
        };

        let jur_owned = jurisdiction.map(|s| s.to_string());
        let com_owned = community_overlay_id.map(|s| s.to_string());

        if let Some(ref cache) = *self.lexicon_cache.read() {
            if cache.0 == pack_count && cache.1 == jur_owned && cache.2 == com_owned {
                return cache.3.clone();
            }
        }

        let lexicon = self.build_lexicon_with_overlays_uncached(jurisdiction, community_overlay_id);

        *self.lexicon_cache.write() = Some((
            pack_count,
            jur_owned,
            com_owned,
            lexicon.clone(),
        ));

        lexicon
    }

    fn build_lexicon_with_overlays_uncached(
        &self,
        #[allow(unused_variables)] jurisdiction: Option<&str>,
        #[allow(unused_variables)] community_overlay_id: Option<&str>,
    ) -> Vec<(String, u32, Severity)> {
        let packs = self.policy_packs.read();
        let mut seen = std::collections::HashSet::new();
        let mut lexicon = Vec::new();

        for pack in packs.iter() {
            for rule in &pack.rules {
                let cat = rule.category.as_u32();
                let sev = crate::policy::severity_from_u8(rule.severity);
                for term in &rule.lexicon {
                    // Normalize: NFKC + casefold to match normalize_for_lexicon output.
                    // This is critical for scripts like Thai where NFKC decomposes
                    // composed characters (e.g. SARA AM U+0E33 → NIKHAHIT + SARA AA).
                    let nfkc: String = term.nfkc().collect();
                    let folded = caseless::default_case_fold_str(&nfkc);
                    if seen.insert(folded.clone()) {
                        lexicon.push((folded, cat, sev));
                    }
                }
            }
        }

        // Merge scam phrases from embedded overlays (skill-pack feature only).
        #[cfg(feature = "skill-pack")]
        {
            use crate::skillpack::data::loaders;
            use crate::detectors::categories;

            // Jurisdiction scam phrases (applied first — legal floor).
            if let Some(code) = jurisdiction {
                for phrase in loaders::extract_jurisdiction_scam_phrases(code) {
                    let nfkc: String = phrase.phrase.nfkc().collect();
                    let folded = caseless::default_case_fold_str(&nfkc);
                    if seen.insert(folded.clone()) {
                        let sev = if phrase.weight >= 0.85 {
                            Severity::BORDERLINE
                        } else {
                            Severity::BENIGN
                        };
                        lexicon.push((folded, categories::SCAM_FRAUD, sev));
                    }
                }
            }

            // Community scam phrases (applied second — community preference).
            if let Some(name) = community_overlay_id {
                for phrase in loaders::extract_community_scam_phrases(name) {
                    let nfkc: String = phrase.phrase.nfkc().collect();
                    let folded = caseless::default_case_fold_str(&nfkc);
                    if seen.insert(folded.clone()) {
                        let sev = if phrase.weight >= 0.85 {
                            Severity::BORDERLINE
                        } else {
                            Severity::BENIGN
                        };
                        lexicon.push((folded, categories::SCAM_FRAUD, sev));
                    }
                }
            }
        }

        lexicon
    }

    /// Get the thresholds from loaded policy packs.
    /// Merges by taking the most conservative (highest) thresholds across all packs.
    fn get_thresholds(&self) -> PolicyThresholds {
        let packs = self.policy_packs.read();
        if packs.is_empty() {
            return PolicyThresholds::default();
        }
        // Merge: take the most conservative threshold from all packs.
        // For warn/block, higher = more conservative (trigger more often).
        // For encoder_escalation, lower = more conservative (escalate more often).
        packs.iter().skip(1).fold(packs[0].thresholds.clone(), |acc, pack| {
            PolicyThresholds {
                warn_threshold: acc.warn_threshold.max(pack.thresholds.warn_threshold),
                block_threshold: acc.block_threshold.max(pack.thresholds.block_threshold),
                encoder_escalation_threshold: acc.encoder_escalation_threshold.min(pack.thresholds.encoder_escalation_threshold),
            }
        })
    }

    /// Check for high-risk indicators that warrant encoder escalation.
    fn has_high_risk_indicators(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        let indicators = [
            // Violence / self-harm
            "kill", "hurt", "die", "suicide", "self-harm", "cut myself",
            // Sexual / NSFW
            "nude", "nsfw", "sexual", "explicit",
            // Weapons / drugs
            "weapon", "gun", "bomb", "drug", "cocaine", "meth", "opioid",
            // Child safety
            "minor", "child", "underage", "groom", "loli",
            // Harassment / hate
            "harass", "bully", "doxx", "subhuman", "vermin", "parasite",
            "inferior race", "ethnic cleansing", "genocide",
            // Extremism
            "extremist", "radical", "terror", "martyrdom", "uprising",
            // Scam / fraud
            "scam", "fraud", "phishing", "lottery", "you've won", "prize",
            "crypto", "bitcoin", "wallet", "seed phrase",
            // Misinformation
            "miracle cure", "vaccine", "anti-vax", "election fraud",
            "deepfake", "fake news", "hoax",
            // Illegal
            "illegal", "stolen", "counterfeit", "black market",
        ];
        indicators.iter().any(|i| lower.contains(i))
    }

    /// Check if the classifier is in deterministic-only mode (no encoder/SLM).
    pub fn is_deterministic_only(&self) -> bool {
        self.encoder.read().is_none() && self.slm.read().is_none()
    }

    /// Number of loaded policy packs.
    pub fn pack_count(&self) -> usize {
        self.policy_packs.read().len()
    }
}

impl Default for SafetyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detectors::categories;

    struct MockEncoder {
        category: u32,
        confidence: f64,
    }

    impl EncoderAdapter for MockEncoder {
        fn classify(&self, _text: &str) -> Result<EncoderVerdict, EncoderError> {
            Ok(EncoderVerdict {
                category: self.category,
                confidence: self.confidence,
            })
        }
    }

    #[test]
    fn test_deterministic_only_allows_safe_text() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("Hello, how are you?");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Allow);
        assert_eq!(result.verdict.source, VerdictSource::Deterministic);
    }

    #[test]
    fn test_deterministic_blocks_pii() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("my card is 4111 1111 1111 1111");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Redact);
        assert_eq!(result.verdict.source, VerdictSource::Deterministic);
    }

    #[test]
    fn test_deterministic_warns_scam() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("URGENT! Send money via bitcoin immediately!");
        let result = classifier.classify(&req);
        // ScamDetector now returns Block for urgency+crypto scams (Severity::SEVERE)
        assert_eq!(result.verdict.action, Action::Block);
    }

    #[test]
    fn test_encoder_escalation_for_group() {
        let classifier = SafetyClassifier::new();
        classifier.attach_encoder(Box::new(MockEncoder {
            category: categories::SAFE,
            confidence: 0.95,
        }));

        let req = ClassifyRequest {
            text: "Hello everyone".into(),
            is_group: true,
            age_mode: None,
            relationship: None,
            encoder_available: true,
            slm_available: false,
            quoted_from_user: false,
            community_overlay_id: None,
            jurisdiction: None,
            locale: None,
            media_descriptors: Vec::new(),
        };

        let result = classifier.classify(&req);
        assert!(result.verdict.used_encoder);
        assert_eq!(result.verdict.source, VerdictSource::Encoder);
    }

    #[test]
    fn test_encoder_escalation_for_minor() {
        let classifier = SafetyClassifier::new();
        classifier.attach_encoder(Box::new(MockEncoder {
            category: categories::SAFE,
            confidence: 0.90,
        }));

        let req = ClassifyRequest {
            text: "What is the meaning of life?".into(),
            is_group: false,
            age_mode: Some("minor".into()),
            relationship: None,
            encoder_available: true,
            slm_available: false,
            quoted_from_user: false,
            community_overlay_id: None,
            jurisdiction: None,
            locale: None,
            media_descriptors: Vec::new(),
        };

        let result = classifier.classify(&req);
        assert!(result.verdict.used_encoder);
    }

    #[test]
    fn test_deterministic_only_on_low_tier() {
        let classifier = SafetyClassifier::new();
        assert!(classifier.is_deterministic_only());

        let req = ClassifyRequest::from_text("Hello");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.source, VerdictSource::Deterministic);
        assert!(!result.verdict.used_encoder);
        assert!(!result.verdict.used_slm);
    }

    #[test]
    fn test_latency_target_deterministic() {
        // Deterministic path P95 target: <5ms = <5000us
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("Hello, how are you today?");
        let result = classifier.classify(&req);
        // On a fast machine this should be well under 5ms
        // (On CI it might be slower, so we use a generous bound)
        assert!(
            result.duration_us < 50_000,
            "deterministic path took {}us, expected <50000us",
            result.duration_us
        );
    }

    #[test]
    fn test_quoted_speech_demotes_scam() {
        let classifier = SafetyClassifier::new();
        // Scam detector fires on "Congratulations! You've won" — with
        // quoted_from_user=true, it should be demoted to Allow
        let req = ClassifyRequest::from_text("Congratulations! You've won $1,000,000")
            .with_quoted();
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Allow);
        assert!(result.verdict.reason_codes.iter().any(|r| r.contains("protected_speech")));
    }

    #[test]
    fn test_news_context_demotes_scam() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("Congratulations! You've won $1,000,000")
            .with_overlay("kchat.community.news.guardrail.v1");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Allow);
        assert!(result.verdict.reason_codes.iter().any(|r| r.contains("protected_speech")));
    }

    #[test]
    fn test_education_context_demotes() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("Send the wire transfer fee to release the funds")
            .with_overlay("kchat.community.education.guardrail.v1");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Allow);
    }

    #[test]
    fn test_counterspeech_context_demotes() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("Congratulations! You've won a prize")
            .with_overlay("kchat.community.counterspeech.guardrail.v1");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Allow);
    }

    #[test]
    fn test_no_context_no_demotion() {
        let classifier = SafetyClassifier::new();
        // Same scam text but without any context hints → should NOT be demoted
        let req = ClassifyRequest::from_text("Congratulations! You've won $1,000,000");
        let result = classifier.classify(&req);
        assert_ne!(result.verdict.action, Action::Allow);
    }

    #[test]
    fn test_child_safety_never_demoted() {
        // Unit test: verify should_demote_for_protected_speech returns None
        // for CHILD_SAFETY category even with high-confidence hints.
        let hints = vec![ContextHint {
            reason_code: "QUOTED_SPEECH_CONTEXT".into(),
            context_confidence: 0.9,
        }];
        assert!(should_demote_for_protected_speech(
            detectors::categories::CHILD_SAFETY,
            &hints,
        ).is_none());
    }

    #[test]
    fn test_derive_context_hints_quoted() {
        let hints = derive_context_hints(true, None);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].reason_code, "QUOTED_SPEECH_CONTEXT");
        assert_eq!(hints[0].context_confidence, 0.7);
    }

    #[test]
    fn test_derive_context_hints_news_overlay() {
        let hints = derive_context_hints(false, Some("kchat.community.news.v1"));
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].reason_code, "NEWS_CONTEXT");
    }

    #[test]
    fn test_derive_context_hints_education_overlay() {
        let hints = derive_context_hints(false, Some("kchat.community.education.v1"));
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].reason_code, "EDUCATION_CONTEXT");
    }

    #[test]
    fn test_derive_context_hints_counterspeech_overlay() {
        let hints = derive_context_hints(false, Some("kchat.community.counterspeech.v1"));
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].reason_code, "COUNTERSPEECH_CONTEXT");
    }

    #[test]
    fn test_derive_context_hints_multiple() {
        // Quoted + news overlay → 2 hints
        let hints = derive_context_hints(true, Some("kchat.community.news.v1"));
        assert_eq!(hints.len(), 2);
    }

    #[test]
    fn test_derive_context_hints_none() {
        let hints = derive_context_hints(false, None);
        assert!(hints.is_empty());
    }

    #[test]
    fn test_derive_context_hints_zero_tolerance_not_counterspeech() {
        // "zero_tolerance" should NOT match counterspeech (bare "tolerance" excluded)
        let hints = derive_context_hints(false, Some("kchat.community.zero_tolerance.v1"));
        assert!(hints.is_empty());
    }

    #[test]
    fn test_empty_text_is_allowed() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Allow);
    }

    #[test]
    fn test_whitespace_only_text_is_allowed() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("   \n\t  ");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Allow);
    }

    #[test]
    fn test_very_long_safe_text_is_allowed() {
        let classifier = SafetyClassifier::new();
        let long_text = "Hello world. ".repeat(1000);
        let req = ClassifyRequest::from_text(&long_text);
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Allow);
    }

    #[test]
    fn test_pii_in_long_text_is_detected() {
        let classifier = SafetyClassifier::new();
        let mut text = "Hello world. ".repeat(500);
        text.push_str(" my card is 4111 1111 1111 1111 ");
        let req = ClassifyRequest::from_text(&text);
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Redact);
    }

    #[test]
    fn test_unicode_text_is_handled() {
        let classifier = SafetyClassifier::new();
        let req = ClassifyRequest::from_text("こんにちは、元気ですか？");
        let result = classifier.classify(&req);
        assert_eq!(result.verdict.action, Action::Allow);
    }

    #[test]
    fn test_encoder_failure_falls_back_to_deterministic() {
        struct FailingEncoder;
        impl EncoderAdapter for FailingEncoder {
            fn classify(&self, _text: &str) -> Result<EncoderVerdict, EncoderError> {
                Err(EncoderError::InferenceFailed("mock failure".into()))
            }
        }

        let classifier = SafetyClassifier::new();
        classifier.attach_encoder(Box::new(FailingEncoder));

        let req = ClassifyRequest {
            text: "Hello everyone in this group".into(),
            is_group: true,
            age_mode: None,
            relationship: None,
            encoder_available: true,
            slm_available: false,
            quoted_from_user: false,
            community_overlay_id: None,
            jurisdiction: None,
            locale: None,
            media_descriptors: Vec::new(),
        };

        let result = classifier.classify(&req);
        // Should fall back gracefully (Degraded source), not crash
        assert_eq!(result.verdict.source, VerdictSource::Degraded);
        assert!(!result.verdict.used_encoder);
    }
}

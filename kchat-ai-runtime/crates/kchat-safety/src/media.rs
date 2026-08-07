//! Media descriptors — host-native safety scores for image/video/audio.
//!
//! Ported from slm-guardrail's `pipeline/lexicon.rs` MediaDescriptor system.
//! The host (iOS/Android/desktop) runs on-device vision models (e.g. MobileCLIP-S2)
//! and produces safety scores for each media attachment. These scores flow into
//! the deterministic priority chain as media branch detectors.
//!
//! The pipeline reasons over these descriptors — not raw media bytes.
//! Decoding, OCR, ASR, and image classification all happen in deterministic
//! local detectors before the classifier is invoked.

use serde::{Deserialize, Serialize};

/// A media descriptor with clamped, schema-valid safety scores.
///
/// Produced by `extract_media_descriptors` from `MediaDescriptorInput`.
/// All scores are clamped to `[0.0, 1.0]`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MediaDescriptor {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nsfw_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub violence_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub self_harm_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hate_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub harassment_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drugs_weapons_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extremism_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub child_safety_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deepfake_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub malware_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub face_count: Option<u32>,
}

/// Loose input shape from JSON — accepts arbitrary keys and silently
/// ignores unknown ones (mirrors the Python pipeline's behavior).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MediaDescriptorInput {
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub nsfw_score: Option<f64>,
    #[serde(default)]
    pub violence_score: Option<f64>,
    #[serde(default)]
    pub self_harm_score: Option<f64>,
    #[serde(default)]
    pub hate_score: Option<f64>,
    #[serde(default)]
    pub harassment_score: Option<f64>,
    #[serde(default)]
    pub drugs_weapons_score: Option<f64>,
    #[serde(default)]
    pub extremism_score: Option<f64>,
    #[serde(default)]
    pub child_safety_score: Option<f64>,
    #[serde(default)]
    pub deepfake_score: Option<f64>,
    #[serde(default)]
    pub malware_score: Option<f64>,
    #[serde(default)]
    pub face_count: Option<i64>,
}

/// Clamp a score to `[0.0, 1.0]`.
fn clamp01(v: f64) -> f64 {
    v.clamp(0.0, 1.0)
}

/// Convert raw `MediaDescriptorInput` values into clamped `MediaDescriptor`s.
///
/// Real implementations derive `nsfw_score` / `violence_score` /
/// `face_count` from on-device vision models. The pipeline only needs
/// to guarantee the shape matches the schema.
pub fn extract_media_descriptors(media: &[MediaDescriptorInput]) -> Vec<MediaDescriptor> {
    media
        .iter()
        .map(|m| MediaDescriptor {
            kind: m.kind.clone().unwrap_or_else(|| "image".to_string()),
            nsfw_score: m.nsfw_score.map(clamp01),
            violence_score: m.violence_score.map(clamp01),
            self_harm_score: m.self_harm_score.map(clamp01),
            hate_score: m.hate_score.map(clamp01),
            harassment_score: m.harassment_score.map(clamp01),
            drugs_weapons_score: m.drugs_weapons_score.map(clamp01),
            extremism_score: m.extremism_score.map(clamp01),
            child_safety_score: m.child_safety_score.map(clamp01),
            deepfake_score: m.deepfake_score.map(clamp01),
            malware_score: m.malware_score.map(clamp01),
            face_count: m.face_count.map(|c| c.max(0) as u32),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_clamps_scores() {
        let input = MediaDescriptorInput {
            kind: Some("image".into()),
            nsfw_score: Some(1.5),
            violence_score: Some(-0.3),
            child_safety_score: Some(0.85),
            ..Default::default()
        };
        let descs = extract_media_descriptors(&[input]);
        assert_eq!(descs.len(), 1);
        assert_eq!(descs[0].nsfw_score, Some(1.0));
        assert_eq!(descs[0].violence_score, Some(0.0));
        assert_eq!(descs[0].child_safety_score, Some(0.85));
    }

    #[test]
    fn test_extract_defaults_kind_to_image() {
        let input = MediaDescriptorInput {
            kind: None,
            ..Default::default()
        };
        let descs = extract_media_descriptors(&[input]);
        assert_eq!(descs[0].kind, "image");
    }

    #[test]
    fn test_extract_face_count_clamped() {
        let input = MediaDescriptorInput {
            face_count: Some(-5),
            ..Default::default()
        };
        let descs = extract_media_descriptors(&[input]);
        assert_eq!(descs[0].face_count, Some(0));
    }

    #[test]
    fn test_extract_empty_input() {
        let descs = extract_media_descriptors(&[]);
        assert!(descs.is_empty());
    }

    #[test]
    fn test_media_descriptor_serialization() {
        let desc = MediaDescriptor {
            kind: "video".into(),
            nsfw_score: Some(0.92),
            violence_score: None,
            self_harm_score: None,
            hate_score: None,
            harassment_score: None,
            drugs_weapons_score: None,
            extremism_score: None,
            child_safety_score: Some(0.1),
            deepfake_score: None,
            malware_score: None,
            face_count: Some(3),
        };
        let json = serde_json::to_string(&desc).unwrap();
        let parsed: MediaDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, desc);
    }
}

//! Temporal aggregation of per-frame vision verdicts for video.
//!
//! Ported from slm-guardrail's `encoder/vision/frame_aggregation.rs`.
//! Two reducers:
//!
//! - [`aggregate_frame_verdicts`]: worst-case (highest severity wins)
//! - [`aggregate_frame_verdicts_smoothed`]: temporal smoothing with persistence floor
//!
//! This module is pure (no I/O, no ONNX dependency). The host extracts
//! and classifies frames, then calls one of the reducers.

use std::cmp::Reverse;

use super::vision_adapter::VisionEncoderVerdict;

/// Safety category for frame aggregation (mirrors slm-guardrail's Category enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameCategory {
    Safe,
    ChildSafety,
    PrivateData,
    ScamFraud,
    HateSpeech,
    Violence,
    Nsfw,
    SelfHarm,
    Spam,
    Extremism,
    Harassment,
    DrugsWeapons,
    Deepfake,
    Malware,
}

impl FrameCategory {
    pub fn id(&self) -> u32 {
        match self {
            Self::Safe => 0,
            Self::ChildSafety => 1,
            Self::PrivateData => 2,
            Self::ScamFraud => 3,
            Self::HateSpeech => 4,
            Self::Violence => 5,
            Self::Nsfw => 6,
            Self::SelfHarm => 7,
            Self::Spam => 8,
            Self::Extremism => 9,
            Self::Harassment => 10,
            Self::DrugsWeapons => 11,
            Self::Deepfake => 12,
            Self::Malware => 13,
        }
    }

    pub const COUNT: usize = 14;
}

/// A single classified video frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameVerdict {
    pub index: u32,
    pub timestamp_s: f32,
    pub severity: u8,
    pub category: FrameCategory,
    pub verdict: VisionEncoderVerdict,
}

/// Error returned by aggregation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameAggregationError {
    Empty,
}

impl std::fmt::Display for FrameAggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "frame aggregation: verdicts must be non-empty"),
        }
    }
}

impl std::error::Error for FrameAggregationError {}

/// Worst-case aggregation: highest severity wins; ties break to lower
/// category id; remaining ties break to earliest frame.
pub fn aggregate_frame_verdicts(
    verdicts: &[FrameVerdict],
) -> Result<&FrameVerdict, FrameAggregationError> {
    verdicts
        .iter()
        .min_by_key(|v| (Reverse(v.severity), v.category.id(), v.index))
        .ok_or(FrameAggregationError::Empty)
}

/// Tuning knobs for smoothed aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalSmoothingConfig {
    pub min_persistence_frames: u32,
    pub critical_severity: u8,
}

impl Default for TemporalSmoothingConfig {
    fn default() -> Self {
        Self {
            min_persistence_frames: 2,
            critical_severity: 4,
        }
    }
}

/// Temporal smoothing: majority vote with persistence floor.
/// Critical-severity frames bypass the vote (worst-case escalation).
pub fn aggregate_frame_verdicts_smoothed<'a>(
    verdicts: &'a [FrameVerdict],
    config: &TemporalSmoothingConfig,
) -> Result<&'a FrameVerdict, FrameAggregationError> {
    if verdicts.is_empty() {
        return Err(FrameAggregationError::Empty);
    }

    // 1. Critical-severity override
    if verdicts.iter().any(|v| v.severity >= config.critical_severity) {
        return aggregate_frame_verdicts(verdicts);
    }

    // 2. Tally per-category
    let mut count = [0u32; FrameCategory::COUNT];
    let mut max_severity = [0u8; FrameCategory::COUNT];
    for v in verdicts {
        let i = v.category.id() as usize;
        count[i] += 1;
        if v.severity > max_severity[i] {
            max_severity[i] = v.severity;
        }
    }

    // 3. Pick winning category by majority vote among eligible
    let safe_id = FrameCategory::Safe.id() as usize;
    let mut winner: Option<usize> = None;
    for i in 0..FrameCategory::COUNT {
        if count[i] == 0 {
            continue;
        }
        if i != safe_id && count[i] < config.min_persistence_frames {
            continue;
        }
        let take = match winner {
            None => true,
            Some(w) => {
                (count[i], max_severity[i], Reverse(i)) > (count[w], max_severity[w], Reverse(w))
            }
        };
        if take {
            winner = Some(i);
        }
    }

    // 4. Fallback to worst-case if no eligible category
    let winner = match winner {
        Some(w) => w,
        None => return aggregate_frame_verdicts(verdicts),
    };

    // 5. Return representative frame for winning category
    verdicts
        .iter()
        .filter(|v| v.category.id() as usize == winner)
        .min_by_key(|v| (Reverse(v.severity), v.index))
        .ok_or(FrameAggregationError::Empty)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::MediaDescriptor;

    fn frame(index: u32, severity: u8, category: FrameCategory) -> FrameVerdict {
        FrameVerdict {
            index,
            timestamp_s: index as f32,
            severity,
            category,
            verdict: VisionEncoderVerdict::new(
                MediaDescriptor {
                    kind: "video".into(),
                    nsfw_score: None,
                    violence_score: None,
                    self_harm_score: None,
                    hate_score: None,
                    harassment_score: None,
                    drugs_weapons_score: None,
                    extremism_score: None,
                    child_safety_score: None,
                    deepfake_score: None,
                    malware_score: None,
                    face_count: None,
                },
                vec![],
            ),
        }
    }

    #[test]
    fn empty_input_is_error() {
        let verdicts: Vec<FrameVerdict> = Vec::new();
        assert_eq!(aggregate_frame_verdicts(&verdicts).unwrap_err(), FrameAggregationError::Empty);
    }

    #[test]
    fn single_frame_returned() {
        let verdicts = vec![frame(0, 0, FrameCategory::Safe)];
        let winner = aggregate_frame_verdicts(&verdicts).unwrap();
        assert_eq!(winner.index, 0);
    }

    #[test]
    fn all_safe_stays_safe() {
        let verdicts = vec![
            frame(0, 0, FrameCategory::Safe),
            frame(1, 0, FrameCategory::Safe),
            frame(2, 0, FrameCategory::Safe),
        ];
        let winner = aggregate_frame_verdicts(&verdicts).unwrap();
        assert_eq!(winner.category, FrameCategory::Safe);
    }

    #[test]
    fn single_harmful_frame_escalates() {
        let verdicts = vec![
            frame(0, 0, FrameCategory::Safe),
            frame(1, 0, FrameCategory::Safe),
            frame(2, 2, FrameCategory::Violence),
            frame(3, 0, FrameCategory::Safe),
        ];
        let winner = aggregate_frame_verdicts(&verdicts).unwrap();
        assert_eq!(winner.category, FrameCategory::Violence);
        assert_eq!(winner.index, 2);
    }

    #[test]
    fn highest_severity_wins() {
        let verdicts = vec![
            frame(0, 1, FrameCategory::ChildSafety),
            frame(1, 2, FrameCategory::Violence),
        ];
        let winner = aggregate_frame_verdicts(&verdicts).unwrap();
        assert_eq!(winner.category, FrameCategory::Violence);
    }

    #[test]
    fn ties_break_to_lower_category() {
        let verdicts = vec![
            frame(0, 2, FrameCategory::DrugsWeapons),
            frame(1, 2, FrameCategory::ChildSafety),
        ];
        let winner = aggregate_frame_verdicts(&verdicts).unwrap();
        assert_eq!(winner.category, FrameCategory::ChildSafety);
    }

    #[test]
    fn ties_break_to_earliest_frame() {
        let ordered = vec![
            frame(2, 2, FrameCategory::Violence),
            frame(5, 2, FrameCategory::Violence),
            frame(0, 0, FrameCategory::Safe),
        ];
        let winner = aggregate_frame_verdicts(&ordered).unwrap();
        assert_eq!(winner.index, 2);
    }

    // -- Smoothed tests --

    fn clip(safe: u32, harmful: u32, severity: u8, category: FrameCategory) -> Vec<FrameVerdict> {
        let mut frames = Vec::new();
        let mut index = 0u32;
        for _ in 0..safe {
            frames.push(frame(index, 0, FrameCategory::Safe));
            index += 1;
        }
        for _ in 0..harmful {
            frames.push(frame(index, severity, category));
            index += 1;
        }
        frames
    }

    #[test]
    fn smoothed_benign_majority_demotes_drift() {
        let verdicts = clip(8, 6, 2, FrameCategory::Violence);
        let winner = aggregate_frame_verdicts_smoothed(&verdicts, &TemporalSmoothingConfig::default()).unwrap();
        assert_eq!(winner.category, FrameCategory::Safe);
    }

    #[test]
    fn smoothed_harmful_majority_escalates() {
        let verdicts = clip(4, 25, 2, FrameCategory::Violence);
        let winner = aggregate_frame_verdicts_smoothed(&verdicts, &TemporalSmoothingConfig::default()).unwrap();
        assert_eq!(winner.category, FrameCategory::Violence);
    }

    #[test]
    fn smoothed_single_drift_below_floor_stays_safe() {
        let verdicts = clip(4, 1, 2, FrameCategory::Violence);
        let winner = aggregate_frame_verdicts_smoothed(&verdicts, &TemporalSmoothingConfig::default()).unwrap();
        assert_eq!(winner.category, FrameCategory::Safe);
    }

    #[test]
    fn smoothed_critical_severity_bypasses_vote() {
        let mut verdicts = clip(8, 0, 0, FrameCategory::Safe);
        verdicts.push(frame(8, 5, FrameCategory::ChildSafety));
        let winner = aggregate_frame_verdicts_smoothed(&verdicts, &TemporalSmoothingConfig::default()).unwrap();
        assert_eq!(winner.category, FrameCategory::ChildSafety);
        assert_eq!(winner.severity, 5);
    }
}

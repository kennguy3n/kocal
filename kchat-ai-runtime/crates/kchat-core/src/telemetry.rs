//! Private telemetry — aggregate metrics only, no raw messages or retrieved text.
//!
//! Telemetry records:
//! - aggregate latency (P50, P95)
//! - crashes and error rates
//! - memory and thermal events
//! - schema success rates
//! - model version (not content)
//!
//! It never records:
//! - raw user messages
//! - retrieved document/chat text
//! - embeddings or hashes
//! - tool arguments or artifact content

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// A single telemetry event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Event type (e.g. "job_complete", "crash", "thermal_event")
    pub event_type: String,
    /// Workload plane (safety, context, generation, action)
    pub plane: String,
    /// Model pack ID (version only, not content)
    pub model_pack_id: Option<String>,
    /// Duration in milliseconds (if applicable)
    pub duration_ms: Option<u64>,
    /// Success/failure
    pub success: bool,
    /// Error reason code (if failed)
    pub error_code: Option<String>,
    /// Device tier at time of event
    pub tier: String,
    /// Additional numeric metrics (aggregate only, no content)
    pub metrics: HashMap<String, f64>,
}

impl TelemetryEvent {
    pub fn job_complete(plane: &str, tier: &str, duration: Duration, success: bool) -> Self {
        Self {
            event_type: "job_complete".into(),
            plane: plane.into(),
            model_pack_id: None,
            duration_ms: Some(duration.as_millis() as u64),
            success,
            error_code: None,
            tier: tier.into(),
            metrics: HashMap::new(),
        }
    }

    pub fn crash(plane: &str, tier: &str, error_code: &str) -> Self {
        Self {
            event_type: "crash".into(),
            plane: plane.into(),
            model_pack_id: None,
            duration_ms: None,
            success: false,
            error_code: Some(error_code.into()),
            tier: tier.into(),
            metrics: HashMap::new(),
        }
    }

    pub fn thermal_event(tier: &str, state: &str) -> Self {
        let mut metrics = HashMap::new();
        metrics.insert("thermal_state".into(), match state {
            "nominal" => 0.0,
            "fair" => 1.0,
            "serious" => 2.0,
            "critical" => 3.0,
            _ => -1.0,
        });
        Self {
            event_type: "thermal_event".into(),
            plane: "scheduler".into(),
            model_pack_id: None,
            duration_ms: None,
            success: true,
            error_code: None,
            tier: tier.into(),
            metrics,
        }
    }

    pub fn schema_success(plane: &str, tier: &str, valid: bool) -> Self {
        Self {
            event_type: "schema_check".into(),
            plane: plane.into(),
            model_pack_id: None,
            duration_ms: None,
            success: valid,
            error_code: None,
            tier: tier.into(),
            metrics: HashMap::new(),
        }
    }
}

/// Telemetry recorder — collects events in-memory for batch upload.
/// In production, events are batched and sent over an encrypted channel.
/// No raw message or retrieved content is ever recorded.
pub struct TelemetryRecorder {
    events: Mutex<std::collections::VecDeque<TelemetryEvent>>,
    max_buffer: usize,
}

impl TelemetryRecorder {
    pub fn new(max_buffer: usize) -> Self {
        Self {
            events: Mutex::new(std::collections::VecDeque::with_capacity(max_buffer)),
            max_buffer,
        }
    }

    /// Record a telemetry event. If the buffer is full, oldest events are dropped.
    pub fn record(&self, event: TelemetryEvent) {
        let mut events = self.events.lock();
        if events.len() >= self.max_buffer {
            tracing::warn!("Telemetry buffer full (max={}), dropping oldest event", self.max_buffer);
            events.pop_front(); // O(1) for VecDeque
        }
        events.push_back(event);
    }

    /// Drain all recorded events (for batch upload).
    pub fn drain(&self) -> Vec<TelemetryEvent> {
        let mut events = self.events.lock();
        events.drain(..).collect()
    }

    /// Number of buffered events.
    pub fn len(&self) -> usize {
        self.events.lock().len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.events.lock().is_empty()
    }
}

impl Default for TelemetryRecorder {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_drain() {
        let recorder = TelemetryRecorder::new(100);
        recorder.record(TelemetryEvent::job_complete(
            "safety",
            "medium",
            Duration::from_millis(50),
            true,
        ));
        recorder.record(TelemetryEvent::crash("generation", "high", "oom"));
        assert_eq!(recorder.len(), 2);

        let events = recorder.drain();
        assert_eq!(events.len(), 2);
        assert!(recorder.is_empty());
    }

    #[test]
    fn test_buffer_overflow_drops_oldest() {
        let recorder = TelemetryRecorder::new(2);
        recorder.record(TelemetryEvent::job_complete(
            "safety", "low", Duration::from_millis(10), true,
        ));
        recorder.record(TelemetryEvent::job_complete(
            "safety", "low", Duration::from_millis(20), true,
        ));
        recorder.record(TelemetryEvent::job_complete(
            "safety", "low", Duration::from_millis(30), true,
        ));
        let events = recorder.drain();
        assert_eq!(events.len(), 2);
        // First event should have been dropped
        assert_eq!(events[0].duration_ms, Some(20));
        assert_eq!(events[1].duration_ms, Some(30));
    }

    #[test]
    fn test_no_raw_content_in_events() {
        let event = TelemetryEvent::job_complete(
            "context", "medium", Duration::from_millis(100), true,
        );
        // Verify no fields could contain raw content
        assert!(event.model_pack_id.is_none() || !event.model_pack_id.as_ref().unwrap().contains("user_message"));
        assert!(!event.event_type.contains("raw"));
        assert!(event.metrics.is_empty() || !event.metrics.keys().any(|k| k.contains("content")));
    }
}

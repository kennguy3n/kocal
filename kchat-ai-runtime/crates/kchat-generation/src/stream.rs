//! Token streaming — streaming output with early termination on safety violation.
//!
//! The stream is cancelled if the safety plane flags content mid-generation.
//! This prevents unsafe content from being fully generated and displayed.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;

/// Stable identifier for a generation stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamId(pub Uuid);

impl StreamId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for StreamId {
    fn default() -> Self {
        Self::new()
    }
}

/// Events emitted during token streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum StreamEvent {
    /// A token was generated
    Token { text: String },
    /// Generation completed successfully
    Complete {
        total_tokens: u32,
        duration_ms: u64,
    },
    /// Generation was cancelled (safety violation or user request)
    Cancelled { reason: String },
    /// Generation failed
    Error { message: String },
}

/// Handle to a generation stream — allows cancellation.
pub struct StreamHandle {
    pub id: StreamId,
    cancelled: Arc<AtomicBool>,
    events: Mutex<Vec<StreamEvent>>,
}

impl StreamHandle {
    pub fn new() -> Self {
        Self {
            id: StreamId::new(),
            cancelled: Arc::new(AtomicBool::new(false)),
            events: Mutex::new(Vec::new()),
        }
    }

    /// Cancel the stream (e.g. due to safety violation).
    pub fn cancel(&self, reason: impl Into<String>) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.events.lock().push(StreamEvent::Cancelled {
            reason: reason.into(),
        });
    }

    /// Check if the stream has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Push a token event.
    pub fn push_token(&self, text: impl Into<String>) {
        if !self.is_cancelled() {
            self.events.lock().push(StreamEvent::Token { text: text.into() });
        }
    }

    /// Mark the stream as complete.
    pub fn complete(&self, total_tokens: u32, duration_ms: u64) {
        self.events.lock().push(StreamEvent::Complete {
            total_tokens,
            duration_ms,
        });
    }

    /// Mark the stream as errored.
    pub fn error(&self, message: impl Into<String>) {
        self.events.lock().push(StreamEvent::Error {
            message: message.into(),
        });
    }

    /// Drain all events.
    pub fn drain_events(&self) -> Vec<StreamEvent> {
        std::mem::take(&mut *self.events.lock())
    }

    /// Get the cancellation flag (for the backend to check during generation).
    pub fn cancellation_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancelled)
    }
}

impl Default for StreamHandle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_token_events() {
        let stream = StreamHandle::new();
        stream.push_token("Hello");
        stream.push_token(" world");
        stream.complete(10, 500);

        let events = stream.drain_events();
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::Token { text } if text == "Hello"));
        assert!(matches!(&events[1], StreamEvent::Token { text } if text == " world"));
        assert!(matches!(&events[2], StreamEvent::Complete { .. }));
    }

    #[test]
    fn test_stream_cancellation() {
        let stream = StreamHandle::new();
        stream.push_token("safe text");
        stream.cancel("safety_violation");

        assert!(stream.is_cancelled());

        // Further tokens should not be pushed
        stream.push_token("unsafe text");

        let events = stream.drain_events();
        assert_eq!(events.len(), 2); // token + cancelled
        assert!(matches!(&events[1], StreamEvent::Cancelled { reason } if reason == "safety_violation"));
    }

    #[test]
    fn test_cancellation_flag() {
        let stream = StreamHandle::new();
        let flag = stream.cancellation_flag();

        assert!(!flag.load(Ordering::SeqCst));

        stream.cancel("test");
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn test_stream_error() {
        let stream = StreamHandle::new();
        stream.error("model crashed");

        let events = stream.drain_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::Error { message } if message == "model crashed"));
    }
}

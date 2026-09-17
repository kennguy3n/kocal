//! Token streaming — streaming output with early termination on safety violation.
//!
//! The stream is cancelled if the safety plane flags content mid-generation.
//! This prevents unsafe content from being fully generated and displayed.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;
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
    Complete { total_tokens: u32, duration_ms: u64 },
    /// Generation was cancelled (safety violation or user request)
    Cancelled { reason: String },
    /// Generation failed
    Error { message: String },
}

/// Handle to a generation stream — allows cancellation.
///
/// Events are stored in a pull buffer (`drain_events`) and, when
/// [`Self::forward_to`] has attached a channel, forwarded in real time.
pub struct StreamHandle {
    pub id: StreamId,
    cancelled: Arc<AtomicBool>,
    /// Set once a terminal event (Complete/Cancelled/Error) has been
    /// emitted — guarantees exactly-once termination and drops late events.
    terminated: AtomicBool,
    events: Mutex<Vec<StreamEvent>>,
    /// Live event channels attached via [`Self::forward_to`].
    forward: Mutex<Vec<UnboundedSender<StreamEvent>>>,
}

impl StreamHandle {
    pub fn new() -> Self {
        Self {
            id: StreamId::new(),
            cancelled: Arc::new(AtomicBool::new(false)),
            terminated: AtomicBool::new(false),
            events: Mutex::new(Vec::new()),
            forward: Mutex::new(Vec::new()),
        }
    }

    /// Attach a live channel — every subsequent event is also sent to `tx`
    /// as it happens (best-effort: a closed receiver is dropped).
    /// Multiple subscribers may be attached (e.g. caller + safety monitor).
    pub fn forward_to(&self, tx: UnboundedSender<StreamEvent>) {
        self.forward.lock().push(tx);
    }

    /// Record an event in the pull buffer and forward it live to all
    /// attached subscribers. Closed channels are dropped. After a terminal
    /// event (Complete/Cancelled/Error) all subscribers are detached so
    /// their receivers observe end-of-stream instead of hanging.
    ///
    /// Terminal events are exactly-once: a `cancel()` racing with backend
    /// `complete()`/`error()` emits only the first terminal event. Events
    /// arriving after termination are dropped.
    fn emit(&self, event: StreamEvent) {
        let terminal = !matches!(event, StreamEvent::Token { .. });
        if terminal {
            if self.terminated.swap(true, Ordering::SeqCst) {
                return;
            }
        } else if self.terminated.load(Ordering::SeqCst) {
            return;
        }
        {
            let mut subs = self.forward.lock();
            subs.retain(|tx| tx.send(event.clone()).is_ok());
            if terminal {
                subs.clear();
            }
        }
        self.events.lock().push(event);
    }

    /// Cancel the stream (e.g. due to safety violation).
    pub fn cancel(&self, reason: impl Into<String>) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.emit(StreamEvent::Cancelled {
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
            self.emit(StreamEvent::Token { text: text.into() });
        }
    }

    /// Mark the stream as complete.
    pub fn complete(&self, total_tokens: u32, duration_ms: u64) {
        self.emit(StreamEvent::Complete {
            total_tokens,
            duration_ms,
        });
    }

    /// Mark the stream as errored.
    pub fn error(&self, message: impl Into<String>) {
        self.emit(StreamEvent::Error {
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

    /// Create a handle sharing an external cancellation flag.
    ///
    /// Used for sub-generations (e.g. best-of-n candidates) that must cancel
    /// together with a parent run but must NOT forward their token events to
    /// the parent's subscribers.
    pub fn with_cancellation(flag: Arc<AtomicBool>) -> Self {
        Self {
            id: StreamId::new(),
            cancelled: flag,
            terminated: AtomicBool::new(false),
            events: Mutex::new(Vec::new()),
            forward: Mutex::new(Vec::new()),
        }
    }
}

impl Default for StreamHandle {
    fn default() -> Self {
        Self::new()
    }
}

/// A live streaming generation running on a background thread.
///
/// Backend inference is synchronous CPU/GPU work, so this runs on a dedicated
/// OS thread rather than a tokio task — no async runtime is required to
/// consume the events, only to `await` the final result.
pub struct StreamingGeneration {
    /// Stream handle — cancel mid-generation via [`StreamHandle::cancel`].
    pub handle: Arc<StreamHandle>,
    /// Real-time event stream (tokens, complete, cancelled, error).
    pub events: tokio::sync::mpsc::UnboundedReceiver<StreamEvent>,
    /// Final generation result — resolves when the worker finishes.
    /// `Err(_)` on the outer result means the worker panicked or was dropped.
    pub result:
        oneshot::Receiver<Result<crate::backend::GenerationResult, crate::backend::BackendError>>,
}

/// Spawn a streaming generation on a dedicated thread.
///
/// The returned `StreamingGeneration` delivers `StreamEvent`s in real time
/// (unlike `drain_events`, which requires polling after the call returns).
/// Cancellation flows through `handle.cancel(..)` into the backend's decode
/// loop, which checks the flag every token.
pub fn spawn_stream(
    backend: Arc<dyn crate::backend::BackendAdapter>,
    prompt: String,
    config: crate::backend::GenerationConfig,
) -> StreamingGeneration {
    let (event_tx, events) = tokio::sync::mpsc::unbounded_channel();
    let (result_tx, result) = oneshot::channel();
    let handle = Arc::new(StreamHandle::new());
    handle.forward_to(event_tx);

    let worker_handle = Arc::clone(&handle);
    std::thread::spawn(move || {
        let res = backend.generate_stream(&prompt, &config, &worker_handle);
        if let Err(e) = &res {
            worker_handle.error(e.to_string());
        }
        // If the receiver hung up, drop the result silently.
        let _ = result_tx.send(res);
    });

    StreamingGeneration {
        handle,
        events,
        result,
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
        assert!(
            matches!(&events[1], StreamEvent::Cancelled { reason } if reason == "safety_violation")
        );
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

    #[test]
    fn test_forward_to_live_channel() {
        let stream = StreamHandle::new();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        stream.forward_to(tx);

        stream.push_token("a");
        stream.complete(1, 5);

        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token { .. })));
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Complete { .. })));
        // Pull buffer still holds the events.
        assert_eq!(stream.drain_events().len(), 2);
    }

    #[tokio::test]
    async fn test_spawn_stream() {
        let backend: Arc<dyn crate::backend::BackendAdapter> =
            Arc::new(crate::backends::MockBackend::new());
        backend
            .load(&crate::backend::BackendConfig::for_tier(
                crate::backend::BackendType::LlamaCppCpu,
                "mock",
                "mock",
                kchat_core::tier::DeviceTier::Low,
                "macos",
            ))
            .unwrap();

        let mut gen = spawn_stream(
            backend,
            "hello world".to_string(),
            crate::backend::GenerationConfig::default(),
        );

        let mut tokens = Vec::new();
        while let Some(ev) = gen.events.recv().await {
            match ev {
                StreamEvent::Token { text } => tokens.push(text),
                StreamEvent::Complete { .. } => break,
                other => panic!("unexpected event: {other:?}"),
            }
        }
        assert!(tokens.len() >= 3, "expected per-word token events");

        let result = gen.result.await.expect("worker result").expect("generate");
        assert!(result.text.ends_with("[mock]"));
    }
}

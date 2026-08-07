//! Decision observer interface — WS6E reporting hooks.
//!
//! Mirrors cv-guard's `shared/policy/decision_observer.py`. The host
//! application can register an implementation of [`DecisionObserver`]
//! on the [`super::interpreter::PolicyInterpreter`] to receive
//! structured, content-free callbacks every time a decision is
//! rendered. The callbacks are designed for three production needs:
//!
//! 1. **Telemetry / audit.** Records *which* rule / SLM call
//!    produced *which* decision (category + severity + rationale
//!    code) so the host can attribute UX events to specific policy
//!    paths without ever seeing the underlying pixels or text. This
//!    is the non-PII-leaking version of "explainability" (PROPOSAL
//!    §11).
//! 2. **Anomaly detection.** Records sanitization drops + SLM
//!    invariant violations so the host can surface alerts when an
//!    upstream component starts feeding bad context hints, or when
//!    the SLM model's outputs systematically violate the
//!    deterministic floor / ceiling rules.
//! 3. **User-facing reporting hooks.** The host can map invariant
//!    violations to a user-facing "report this decision"
//!    affordance: the `rationale_code` + the violation reason are
//!    stable identifiers so the host can attach a structured
//!    report to the decision without re-deriving it.
//!
//! Every callback is intentionally *synchronous* and takes only
//! frozen / immutable values (`&` references). Implementations
//! MUST be `Send + Sync` (the interpreter does not serialise
//! callbacks) and SHOULD NOT panic from inside the callback — the
//! interpreter swallows panics to keep the decision path resilient
//! (matching the Python reference's exception-suppression policy),
//! but a panicking observer is still a bug at the observer site.
//! For heavy work, queue an event inside the observer and process
//! it on a background thread.
//!
//! Concrete implementations
//! ------------------------
//!
//! * [`NullDecisionObserver`] — default no-op observer used when
//!   the host doesn't register one.
//! * [`CompositeDecisionObserver`] — fans every event out to a
//!   list of child observers. Useful when the host wants both a
//!   telemetry sink and an audit-log sink.
//! * [`InMemoryDecisionObserver`] — captures every event in
//!   bounded ring buffers. Used by unit tests and by the iOS /
//!   Android debug screens to display the last N decisions
//!   without a real telemetry backend.
//!
//! The iOS / Android ports mirror this interface as a Swift
//! protocol / Kotlin interface; the wire-level event shape is
//! identical so a cross-platform host can ship one telemetry
//! pipeline.

use std::collections::VecDeque;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};

use super::decision::PolicyDecision;
use super::sanitizer::SanitizationEvent;

/// Snapshot of a finalised decision.
///
/// The event captures the **deterministic identifiers** of the
/// decision (category, severity, rationale code, source) plus
/// bookkeeping flags (`used_slm`, `had_signal_sanitization`,
/// `had_invariant_violation`). It deliberately omits the raw
/// signals payload and the SLM prompt to keep the event
/// content-free — the host application can correlate by
/// `media_id` if it needs to join against its own request logs.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionEvent {
    pub media_id: String,
    pub media_type: String,
    pub decision: PolicyDecision,
    pub sanitization_events: Vec<SanitizationEvent>,
    pub invariant_violations: Vec<String>,
    /// WS6B: `true` when the interpreter chose the rule-path
    /// fallback because the SLM rate limiter denied a token. The
    /// `rationale_code` will end in `.rate_limited` and
    /// `decision.used_slm` will be `false`.
    pub rate_limited: bool,
}

impl DecisionEvent {
    pub fn new(
        media_id: impl Into<String>,
        media_type: impl Into<String>,
        decision: PolicyDecision,
    ) -> Self {
        Self {
            media_id: media_id.into(),
            media_type: media_type.into(),
            decision,
            sanitization_events: Vec::new(),
            invariant_violations: Vec::new(),
            rate_limited: false,
        }
    }

    #[must_use]
    pub fn with_sanitization_events(mut self, events: Vec<SanitizationEvent>) -> Self {
        self.sanitization_events = events;
        self
    }

    #[must_use]
    pub fn with_invariant_violations(mut self, violations: Vec<String>) -> Self {
        self.invariant_violations = violations;
        self
    }

    #[must_use]
    pub fn with_rate_limited(mut self, rate_limited: bool) -> Self {
        self.rate_limited = rate_limited;
        self
    }

    pub fn had_signal_sanitization(&self) -> bool {
        !self.sanitization_events.is_empty()
    }

    pub fn had_invariant_violation(&self) -> bool {
        !self.invariant_violations.is_empty()
    }
}

/// Snapshot of an SLM output that violated a deterministic floor.
///
/// The interpreter emits this **before** it falls back to the
/// rule-based decision so the observer sees both the SLM's claim
/// and the deterministic correction. Observers can use this to
/// detect drift in the SLM model (e.g. a fine-tune that
/// systematically under-classifies adult content).
#[derive(Debug, Clone, PartialEq)]
pub struct InvariantViolationEvent {
    pub media_id: String,
    pub media_type: String,
    pub slm_decision: PolicyDecision,
    pub fallback_decision: PolicyDecision,
    pub violations: Vec<String>,
}

impl InvariantViolationEvent {
    pub fn new(
        media_id: impl Into<String>,
        media_type: impl Into<String>,
        slm_decision: PolicyDecision,
        fallback_decision: PolicyDecision,
        violations: Vec<String>,
    ) -> Self {
        Self {
            media_id: media_id.into(),
            media_type: media_type.into(),
            slm_decision,
            fallback_decision,
            violations,
        }
    }
}

/// Errors raised by [`InMemoryDecisionObserver::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionObserverError {
    InvalidCapacity { capacity: usize },
}

impl fmt::Display for DecisionObserverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecisionObserverError::InvalidCapacity { capacity } => {
                write!(f, "capacity must be > 0, got {capacity}")
            }
        }
    }
}

impl std::error::Error for DecisionObserverError {}

/// Abstract decision-observer trait.
///
/// Implementations MUST be `Send + Sync`; the interpreter does not
/// serialise callbacks. The default no-op
/// [`NullDecisionObserver`] is what the interpreter uses when the
/// host doesn't register a real observer.
pub trait DecisionObserver: Send + Sync {
    /// Fired for every finalised decision (rule-path, child-safety
    /// floor, SLM path, and SLM-with-invariant-violation
    /// fallback).
    fn on_decision(&self, event: &DecisionEvent);

    /// Fired before the SLM is invoked when the sanitizer dropped
    /// one or more fields. Always paired with a subsequent
    /// [`DecisionObserver::on_decision`] for the same `media_id`.
    /// Empty `events` is never passed — the interpreter only fires
    /// when at least one drop happened.
    fn on_signals_sanitized(&self, media_id: &str, events: &[SanitizationEvent]);

    /// Fired when the SLM returned a structurally valid but
    /// semantically-out-of-policy decision (e.g. severity 0 when
    /// a label is above its severe threshold). The
    /// [`super::interpreter::PolicyInterpreter`] will then
    /// override with the deterministic fallback and emit a normal
    /// [`DecisionEvent`] whose `invariant_violations` field
    /// records the violations.
    fn on_invariant_violation(&self, event: &InvariantViolationEvent);
}

/// No-op observer. The default when the host doesn't register one.
///
/// Avoids `Option<Arc<dyn DecisionObserver>>` branching in the
/// interpreter hot path — the calls become indirect-but-tail.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullDecisionObserver;

impl DecisionObserver for NullDecisionObserver {
    fn on_decision(&self, _event: &DecisionEvent) {}
    fn on_signals_sanitized(&self, _media_id: &str, _events: &[SanitizationEvent]) {}
    fn on_invariant_violation(&self, _event: &InvariantViolationEvent) {}
}

/// Fans events out to a list of child observers.
///
/// Each child is called in registration order; if a child panics,
/// the panic is caught and the next child is still invoked (the
/// interpreter has the same suppression policy at the top level).
/// The host application is expected to log panics inside the
/// child observer; the composite deliberately stays content-free
/// and does NOT log payloads.
pub struct CompositeDecisionObserver {
    observers: Vec<Arc<dyn DecisionObserver>>,
}

impl CompositeDecisionObserver {
    pub fn new(observers: Vec<Arc<dyn DecisionObserver>>) -> Self {
        Self { observers }
    }

    pub fn observers(&self) -> &[Arc<dyn DecisionObserver>] {
        &self.observers
    }
}

impl fmt::Debug for CompositeDecisionObserver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompositeDecisionObserver")
            .field("observer_count", &self.observers.len())
            .finish()
    }
}

impl DecisionObserver for CompositeDecisionObserver {
    fn on_decision(&self, event: &DecisionEvent) {
        for child in &self.observers {
            // `catch_unwind` suppresses panics inside individual
            // observers so a misbehaving telemetry sink can't take
            // down the dispatch path. `AssertUnwindSafe` is fine
            // here because `DecisionObserver` callbacks must not
            // mutate `&self` invariants in a way that crosses a
            // panic boundary — that's the trait's documented
            // contract.
            let _ = std::panic::catch_unwind(AssertUnwindSafe(|| child.on_decision(event)));
        }
    }

    fn on_signals_sanitized(&self, media_id: &str, events: &[SanitizationEvent]) {
        for child in &self.observers {
            let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
                child.on_signals_sanitized(media_id, events)
            }));
        }
    }

    fn on_invariant_violation(&self, event: &InvariantViolationEvent) {
        for child in &self.observers {
            let _ =
                std::panic::catch_unwind(AssertUnwindSafe(|| child.on_invariant_violation(event)));
        }
    }
}

/// Bounded ring buffers of every event the interpreter emits.
///
/// Used by unit tests (so they can assert on the sequence of
/// observer callbacks without wiring a real telemetry backend) and
/// by the iOS / Android debug screens (so a developer can scroll
/// through the last N decisions on-device). The ring buffers are
/// bounded — the oldest entry is evicted on overflow — so a
/// long-running process never grows unbounded memory.
pub struct InMemoryDecisionObserver {
    capacity: usize,
    state: Mutex<InMemoryState>,
}

#[derive(Debug, Default)]
struct InMemoryState {
    decisions: VecDeque<DecisionEvent>,
    sanitizations: VecDeque<(String, Vec<SanitizationEvent>)>,
    violations: VecDeque<InvariantViolationEvent>,
}

impl InMemoryDecisionObserver {
    /// Default ring-buffer capacity. Matches the Python reference's
    /// `DEFAULT_CAPACITY = 200`.
    pub const DEFAULT_CAPACITY: usize = 200;

    pub fn new(capacity: usize) -> Result<Self, DecisionObserverError> {
        if capacity == 0 {
            return Err(DecisionObserverError::InvalidCapacity { capacity });
        }
        Ok(Self {
            capacity,
            state: Mutex::new(InMemoryState::default()),
        })
    }

    pub fn with_default_capacity() -> Self {
        // `Self::DEFAULT_CAPACITY` is a compile-time non-zero
        // constant so `unwrap` is unreachable.
        Self::new(Self::DEFAULT_CAPACITY).expect("DEFAULT_CAPACITY is non-zero")
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn decisions(&self) -> Vec<DecisionEvent> {
        self.lock_state().decisions.iter().cloned().collect()
    }

    pub fn sanitizations(&self) -> Vec<(String, Vec<SanitizationEvent>)> {
        self.lock_state().sanitizations.iter().cloned().collect()
    }

    pub fn invariant_violations(&self) -> Vec<InvariantViolationEvent> {
        self.lock_state().violations.iter().cloned().collect()
    }

    pub fn clear(&self) {
        let mut state = self.lock_state();
        state.decisions.clear();
        state.sanitizations.clear();
        state.violations.clear();
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, InMemoryState> {
        match self.state.lock() {
            Ok(g) => g,
            // Lock poisoning here just means an earlier panicking
            // observer call entered the critical section; the
            // payloads themselves are pure data so we can resume.
            Err(p) => p.into_inner(),
        }
    }

    fn push_with_eviction<T>(deque: &mut VecDeque<T>, item: T, capacity: usize) {
        if deque.len() >= capacity {
            // Evict the oldest entry first. `VecDeque::pop_front`
            // is O(1).
            deque.pop_front();
        }
        deque.push_back(item);
    }
}

impl Default for InMemoryDecisionObserver {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

impl fmt::Debug for InMemoryDecisionObserver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock_state();
        f.debug_struct("InMemoryDecisionObserver")
            .field("capacity", &self.capacity)
            .field("decisions_len", &state.decisions.len())
            .field("sanitizations_len", &state.sanitizations.len())
            .field("violations_len", &state.violations.len())
            .finish()
    }
}

impl DecisionObserver for InMemoryDecisionObserver {
    fn on_decision(&self, event: &DecisionEvent) {
        let mut state = self.lock_state();
        Self::push_with_eviction(&mut state.decisions, event.clone(), self.capacity);
    }

    fn on_signals_sanitized(&self, media_id: &str, events: &[SanitizationEvent]) {
        let mut state = self.lock_state();
        Self::push_with_eviction(
            &mut state.sanitizations,
            (media_id.to_string(), events.to_vec()),
            self.capacity,
        );
    }

    fn on_invariant_violation(&self, event: &InvariantViolationEvent) {
        let mut state = self.lock_state();
        Self::push_with_eviction(&mut state.violations, event.clone(), self.capacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy_interpreter::decision::{DecisionSource, UXAction};
    use crate::policy_interpreter::sanitizer::SanitizationReason;

    fn make_decision(category: &str, severity: u8, rationale: &str) -> PolicyDecision {
        PolicyDecision::new(
            category.to_string(),
            severity,
            UXAction::BlurTap,
            rationale.to_string(),
        )
        .unwrap()
        .with_source(DecisionSource::Rule)
    }

    fn make_sanitization_event(field: &str) -> SanitizationEvent {
        SanitizationEvent {
            field: field.to_string(),
            reason: SanitizationReason::UnknownKey,
        }
    }

    #[test]
    fn null_observer_swallows_every_event() {
        let obs = NullDecisionObserver;
        let event = DecisionEvent::new("m1", "image", make_decision("scam", 3, "scam.high"));
        obs.on_decision(&event);
        obs.on_signals_sanitized("m1", &[make_sanitization_event("k")]);
        obs.on_invariant_violation(&InvariantViolationEvent::new(
            "m1",
            "image",
            event.decision.clone(),
            event.decision.clone(),
            vec!["floor.severity".to_string()],
        ));
        // Reaching here without panicking is the assertion.
    }

    #[test]
    fn in_memory_observer_captures_events_in_order() {
        let obs = InMemoryDecisionObserver::with_default_capacity();
        let d1 = make_decision("scam", 3, "scam.high");
        let d2 = make_decision("benign", 1, "benign.low");
        obs.on_decision(&DecisionEvent::new("m1", "image", d1.clone()));
        obs.on_decision(&DecisionEvent::new("m2", "image", d2.clone()));

        let stored = obs.decisions();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].media_id, "m1");
        assert_eq!(stored[0].decision.category, "scam");
        assert_eq!(stored[1].media_id, "m2");
        assert_eq!(stored[1].decision.category, "benign");
    }

    #[test]
    fn in_memory_observer_evicts_oldest_when_capacity_reached() {
        let obs = InMemoryDecisionObserver::new(3).unwrap();
        for i in 0..5 {
            obs.on_decision(&DecisionEvent::new(
                format!("m{i}"),
                "image",
                make_decision("scam", 3, "scam.high"),
            ));
        }
        let stored = obs.decisions();
        assert_eq!(stored.len(), 3);
        // First two events evicted: only m2/m3/m4 remain.
        assert_eq!(stored[0].media_id, "m2");
        assert_eq!(stored[1].media_id, "m3");
        assert_eq!(stored[2].media_id, "m4");
    }

    #[test]
    fn in_memory_observer_rejects_zero_capacity() {
        let err = InMemoryDecisionObserver::new(0).unwrap_err();
        assert!(matches!(
            err,
            DecisionObserverError::InvalidCapacity { capacity: 0 }
        ));
    }

    #[test]
    fn in_memory_observer_clear_resets_all_ring_buffers() {
        let obs = InMemoryDecisionObserver::with_default_capacity();
        obs.on_decision(&DecisionEvent::new(
            "m",
            "image",
            make_decision("scam", 3, "scam.high"),
        ));
        obs.on_signals_sanitized("m", &[make_sanitization_event("k")]);
        let d = make_decision("scam", 3, "scam.high");
        obs.on_invariant_violation(&InvariantViolationEvent::new(
            "m",
            "image",
            d.clone(),
            d,
            vec!["floor.severity".to_string()],
        ));
        assert!(!obs.decisions().is_empty());
        assert!(!obs.sanitizations().is_empty());
        assert!(!obs.invariant_violations().is_empty());

        obs.clear();
        assert!(obs.decisions().is_empty());
        assert!(obs.sanitizations().is_empty());
        assert!(obs.invariant_violations().is_empty());
    }

    #[test]
    fn composite_observer_fans_out_to_all_children() {
        let a = Arc::new(InMemoryDecisionObserver::with_default_capacity());
        let b = Arc::new(InMemoryDecisionObserver::with_default_capacity());
        let composite = CompositeDecisionObserver::new(vec![
            a.clone() as Arc<dyn DecisionObserver>,
            b.clone() as Arc<dyn DecisionObserver>,
        ]);
        let event = DecisionEvent::new("m1", "image", make_decision("scam", 3, "scam.high"));
        composite.on_decision(&event);

        assert_eq!(a.decisions().len(), 1);
        assert_eq!(b.decisions().len(), 1);
        assert_eq!(a.decisions()[0].media_id, "m1");
        assert_eq!(b.decisions()[0].media_id, "m1");
    }

    #[test]
    fn composite_observer_continues_after_child_panic() {
        struct PanickingObserver;
        impl DecisionObserver for PanickingObserver {
            fn on_decision(&self, _event: &DecisionEvent) {
                panic!("intentional panic from test observer");
            }
            fn on_signals_sanitized(&self, _media_id: &str, _events: &[SanitizationEvent]) {}
            fn on_invariant_violation(&self, _event: &InvariantViolationEvent) {}
        }

        let panicker = Arc::new(PanickingObserver) as Arc<dyn DecisionObserver>;
        let downstream = Arc::new(InMemoryDecisionObserver::with_default_capacity());

        let composite = CompositeDecisionObserver::new(vec![
            panicker,
            downstream.clone() as Arc<dyn DecisionObserver>,
        ]);

        let event = DecisionEvent::new("m", "image", make_decision("scam", 3, "scam.high"));
        // Suppress the panic backtrace noise from the test output.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        composite.on_decision(&event);
        std::panic::set_hook(prev_hook);

        // Downstream observer still received the event despite the
        // upstream panic.
        assert_eq!(downstream.decisions().len(), 1);
        assert_eq!(downstream.decisions()[0].media_id, "m");
    }

    #[test]
    fn decision_event_helper_flags_match_population() {
        let event = DecisionEvent::new("m", "image", make_decision("scam", 3, "scam.high"));
        assert!(!event.had_signal_sanitization());
        assert!(!event.had_invariant_violation());

        let with_drops = event
            .clone()
            .with_sanitization_events(vec![make_sanitization_event("k")]);
        assert!(with_drops.had_signal_sanitization());

        let with_violations = event.with_invariant_violations(vec!["floor.severity".to_string()]);
        assert!(with_violations.had_invariant_violation());
    }

    #[test]
    fn observer_is_send_and_sync_for_dyn_dispatch() {
        // Compile-time assertion: every Observer impl must be
        // safely shareable across threads.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NullDecisionObserver>();
        assert_send_sync::<CompositeDecisionObserver>();
        assert_send_sync::<InMemoryDecisionObserver>();
        assert_send_sync::<Box<dyn DecisionObserver>>();
        assert_send_sync::<Arc<dyn DecisionObserver>>();
    }
}

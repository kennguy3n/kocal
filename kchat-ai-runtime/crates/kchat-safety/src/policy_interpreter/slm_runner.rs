//! SLM runner trait and a deterministic mock implementation.
//!
//! Mirrors cv-guard's `shared/policy/slm_runner.py`. The runner is
//! the abstraction the [`super::interpreter::PolicyInterpreter`]
//! uses to consult the small language model in the ambiguous
//! trigger-but-not-severe path (PROPOSAL §4 step 4).
//!
//! Implementations MUST return a [`PolicyDecision`] directly — the
//! tokenization / decoding contract belongs entirely to the runner,
//! not the interpreter. Production hosts plug in a llama.cpp
//! Ternary-Bonsai-1.7B Q2_0 runner with GBNF grammar-constrained
//! decoding; the
//! unit tests + the hermetic CI build use [`MockSlmRunner`] which
//! returns canned decisions from a lookup table keyed by the
//! `test_scenario` context hint.
//!
//! The runner is deliberately decoupled from the prompt format:
//! the interpreter renders the prompt + the signal-JSON payload
//! and hands both to [`SlmRunner::decide`]. A future runner that
//! wants to skip the prompt entirely (e.g. a JSON-mode chat
//! completion API) is free to ignore the `prompt` argument and
//! consume only `signal_json`.

use std::collections::BTreeMap;
use std::fmt;

use parking_lot::Mutex;

use serde_json::Value;

use super::decision::{DecisionSource, PolicyDecision, UXAction};

/// Abstract interface: feed a rendered prompt + the signal JSON
/// payload, get a [`PolicyDecision`] back.
///
/// Implementations MUST be deterministic given a fixed seed and
/// fixed inputs so the parity test suite (and any future
/// regression replay) can lock down their outputs. Production
/// callers that want sampling control should extend the interface
/// rather than relax this contract.
///
/// The trait is `Send + Sync` so a single runner instance can be
/// shared across the dispatch threads the host uses. Production
/// runners should hold any thread-shared state (model weights,
/// llama.cpp context) behind their own locking; the interpreter
/// does NOT serialise calls to `decide`.
pub trait SlmRunner: Send + Sync {
    /// Invoke the SLM.
    ///
    /// * `prompt` — the rendered prompt (skill-pack prompt body +
    ///   the BEGIN/END UNTRUSTED SIGNALS marker block). UTF-8.
    /// * `signal_json` — the same `Value::Object` the interpreter
    ///   serialised into the prompt. Passed alongside the rendered
    ///   prompt so a runner that wants structured data without
    ///   re-parsing the prompt can read it directly. The
    ///   `MockSlmRunner` in particular routes off
    ///   `signal_json["context_hints"]["test_scenario"]`.
    fn decide(&self, prompt: &str, signal_json: &Value) -> PolicyDecision;
}

/// Test runner. Returns decisions from a lookup table keyed by
/// "scenario tags" the test inserts into the signal JSON via
/// `context_hints["test_scenario"]`. Falls back to a deterministic
/// default decision if no scenario matches.
///
/// The runner is internally `Mutex`-protected so the same instance
/// can be passed to multi-threaded tests that exercise the
/// interpreter without `&mut`. The `calls()` snapshot lets a test
/// assert that the SLM was (or was NOT) invoked — important for
/// the fast-path tests where the interpreter must short-circuit
/// before the SLM is consulted.
pub struct MockSlmRunner {
    decisions: BTreeMap<String, PolicyDecision>,
    default: Option<PolicyDecision>,
    calls: Mutex<Vec<Value>>,
}

impl MockSlmRunner {
    /// Build a runner that returns `default` when no scenario tag
    /// matches the lookup table. Pass `None` to fall back to the
    /// hardcoded conservative default (severity 2, blur-and-tap)
    /// matching cv-guard's `slm_runner.py::MockSLMRunner`.
    pub fn new(
        decisions: BTreeMap<String, PolicyDecision>,
        default: Option<PolicyDecision>,
    ) -> Self {
        Self {
            decisions,
            default,
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Convenience: build an empty runner that always returns the
    /// hardcoded conservative default. Matches
    /// `MockSLMRunner()` in Python.
    pub fn with_default_only() -> Self {
        Self::new(BTreeMap::new(), None)
    }

    /// Every signal-JSON passed to [`SlmRunner::decide`] in call
    /// order. Useful for asserting that the fast-path *didn't*
    /// invoke the SLM (`runner.calls().is_empty()`).
    ///
    /// Returns an owned clone so the caller can iterate without
    /// holding the lock.
    pub fn calls(&self) -> Vec<Value> {
        self.calls.lock().clone()
    }

    /// Number of times [`SlmRunner::decide`] has been invoked.
    /// Cheaper than `calls().len()` because it doesn't clone the
    /// underlying `Vec`.
    pub fn call_count(&self) -> usize {
        self.calls.lock().len()
    }

    /// Reset the call log. Useful between phases of a test that
    /// reuse the same runner.
    pub fn clear_calls(&self) {
        self.calls.lock().clear();
    }
}

impl Default for MockSlmRunner {
    fn default() -> Self {
        Self::with_default_only()
    }
}

impl fmt::Debug for MockSlmRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print scenarios + a call count rather than the full
        // payload list; the payloads can contain user-controllable
        // strings and we don't want a stray `dbg!(runner)` to dump
        // them into logs.
        f.debug_struct("MockSlmRunner")
            .field("scenarios", &self.decisions.keys().collect::<Vec<_>>())
            .field("has_default", &self.default.is_some())
            .field("call_count", &self.call_count())
            .finish()
    }
}

impl SlmRunner for MockSlmRunner {
    fn decide(&self, _prompt: &str, signal_json: &Value) -> PolicyDecision {
        // Record the call before any branching so even the fall-
        // through cases are observable.
        self.calls.lock().push(signal_json.clone());

        let scenario = signal_json
            .get("context_hints")
            .and_then(|h| h.as_object())
            .and_then(|h| h.get("test_scenario"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if let Some(d) = self.decisions.get(scenario) {
            return d.clone();
        }
        if let Some(d) = &self.default {
            return d.clone();
        }

        // Conservative fallback — medium severity with a
        // content-free rationale code so tests that forget to wire
        // a default still produce deterministic output. Matches
        // `slm_runner.py::MockSLMRunner.decide` exactly:
        // `(category="benign", severity=2, ux_action=blur_tap,
        // rationale_code="mock.default", allow_reveal=True,
        // allow_forward=False, used_slm=True, source="slm")`.
        PolicyDecision::new(
            "benign".to_string(),
            2,
            UXAction::BlurTap,
            "mock.default".to_string(),
        )
        .expect("hardcoded mock default decision must be valid")
        .with_allow_reveal(true)
        .with_allow_forward(false)
        .with_used_slm(true)
        .with_source(DecisionSource::Slm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_decision(category: &str, severity: u8, rationale: &str) -> PolicyDecision {
        PolicyDecision::new(
            category.to_string(),
            severity,
            UXAction::BlurTap,
            rationale.to_string(),
        )
        .unwrap()
        .with_allow_reveal(true)
        .with_allow_forward(false)
        .with_used_slm(true)
        .with_source(DecisionSource::Slm)
    }

    #[test]
    fn mock_runner_returns_scenario_decision_when_tag_matches() {
        let mut decisions = BTreeMap::new();
        decisions.insert(
            "scam_high".to_string(),
            make_decision("scam", 4, "scam.detected"),
        );
        let runner = MockSlmRunner::new(decisions, None);
        let payload = json!({"context_hints": {"test_scenario": "scam_high"}});

        let out = runner.decide("prompt", &payload);

        assert_eq!(out.category, "scam");
        assert_eq!(out.severity, 4);
        assert_eq!(out.rationale_code, "scam.detected");
        assert_eq!(runner.call_count(), 1);
    }

    #[test]
    fn mock_runner_uses_user_supplied_default_when_no_tag_matches() {
        let default = make_decision("benign", 1, "default.low");
        let runner = MockSlmRunner::new(BTreeMap::new(), Some(default));
        let payload = json!({"context_hints": {"test_scenario": "unknown"}});

        let out = runner.decide("prompt", &payload);

        assert_eq!(out.category, "benign");
        assert_eq!(out.severity, 1);
        assert_eq!(out.rationale_code, "default.low");
    }

    #[test]
    fn mock_runner_falls_back_to_hardcoded_default_when_nothing_provided() {
        let runner = MockSlmRunner::with_default_only();
        let payload = json!({"context_hints": {}});

        let out = runner.decide("prompt", &payload);

        // Matches the hardcoded default from
        // cv-guard's `slm_runner.py::MockSLMRunner.decide` exactly.
        assert_eq!(out.category, "benign");
        assert_eq!(out.severity, 2);
        assert_eq!(out.rationale_code, "mock.default");
        assert!(matches!(out.ux_action, UXAction::BlurTap));
        assert!(out.allow_reveal);
        assert!(!out.allow_forward);
        assert!(out.used_slm);
        assert!(matches!(out.source, DecisionSource::Slm));
    }

    #[test]
    fn mock_runner_records_calls_in_order() {
        let runner = MockSlmRunner::with_default_only();
        runner.decide("p1", &json!({"context_hints": {"test_scenario": "a"}}));
        runner.decide("p2", &json!({"context_hints": {"test_scenario": "b"}}));

        let calls = runner.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[0]["context_hints"]["test_scenario"],
            Value::String("a".to_string())
        );
        assert_eq!(
            calls[1]["context_hints"]["test_scenario"],
            Value::String("b".to_string())
        );
    }

    #[test]
    fn mock_runner_handles_missing_context_hints_block() {
        // Payload with no context_hints at all — must NOT panic,
        // must fall through to default.
        let runner = MockSlmRunner::with_default_only();
        let out = runner.decide("p", &json!({}));
        assert_eq!(out.rationale_code, "mock.default");
    }

    #[test]
    fn mock_runner_handles_non_string_test_scenario() {
        // Defense-in-depth: a malformed payload shouldn't crash.
        let runner = MockSlmRunner::with_default_only();
        let out = runner.decide("p", &json!({"context_hints": {"test_scenario": 123}}));
        assert_eq!(out.rationale_code, "mock.default");
    }

    #[test]
    fn clear_calls_resets_log_without_dropping_runner() {
        let runner = MockSlmRunner::with_default_only();
        runner.decide("p", &json!({}));
        runner.decide("p", &json!({}));
        assert_eq!(runner.call_count(), 2);
        runner.clear_calls();
        assert_eq!(runner.call_count(), 0);
        // Runner still usable after clear.
        runner.decide("p", &json!({}));
        assert_eq!(runner.call_count(), 1);
    }

    #[test]
    fn debug_format_does_not_leak_payload_contents() {
        let runner = MockSlmRunner::with_default_only();
        runner.decide(
            "p",
            &json!({"context_hints": {"test_scenario": "should_not_appear"}}),
        );
        let dbg = format!("{:?}", runner);
        // Only metadata fields appear.
        assert!(dbg.contains("MockSlmRunner"));
        assert!(dbg.contains("scenarios"));
        assert!(dbg.contains("has_default"));
        assert!(dbg.contains("call_count: 1"));
        // The actual payload string is NOT in the Debug output.
        assert!(!dbg.contains("should_not_appear"));
    }
}

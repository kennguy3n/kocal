//! Policy interpreter — foundation pieces.
//!
//! This module ports the deterministic / non-SLM parts of
//! `cv-guard/shared/policy/` into Rust:
//!
//! * [`input`] — [`PolicyInput`](input::PolicyInput) +
//!   [`OCRSignals`](input::OCRSignals) +
//!   [`MediaType`](input::MediaType): the closed-shape signal JSON
//!   the interpreter accepts.
//! * [`decision`] — [`PolicyDecision`](decision::PolicyDecision) +
//!   [`UXAction`](decision::UXAction) +
//!   [`DecisionSource`](decision::DecisionSource): the closed-shape
//!   decision JSON the interpreter returns.
//! * [`severity`] — [`SeverityRubric`](severity::SeverityRubric) +
//!   [`SeverityLevel`](severity::SeverityLevel) +
//!   [`SeverityMapper`](severity::SeverityMapper) +
//!   [`UXDisposition`](severity::UXDisposition): pure-function
//!   severity → (UX action, allow_reveal, allow_forward) mapper.
//! * [`sanitizer`] — closed-set allow-list for `context_hints` +
//!   `pii_categories_matched`, plus [`SanitizationEvent`](sanitizer::SanitizationEvent)
//!   for observer telemetry. This is the first line of defense
//!   against prompt-injection in the SIGNALS blob fed to the SLM.
//!
//! The SLM-coupled pieces (runner trait, observer, rate limiter,
//! interpreter itself) live alongside the SLM runner crate.
//!
//! No item in this module depends on a feature gate — every
//! deployment that uses the orchestrator needs the policy
//! decision / input shapes for the renderer contract.

pub mod decision;
pub mod decision_observer;
pub mod device_state;
pub mod input;
pub mod interpreter;
pub mod rate_limiter;
pub mod sanitizer;
pub mod severity;
pub mod slm_runner;
pub mod thresholds;

pub use decision::{
    DecisionSource, PolicyDecision, PolicyDecisionError, UXAction, UXActionParseError,
};
pub use decision_observer::{
    CompositeDecisionObserver, DecisionEvent, DecisionObserver, DecisionObserverError,
    InMemoryDecisionObserver, InvariantViolationEvent, NullDecisionObserver,
};
pub use device_state::{DeviceState, GatingPlan, ThermalState, LOW_BATTERY_THRESHOLD};
pub use input::{MediaType, OCRSignals, PolicyInput, PolicyInputError, VisionScores};
pub use interpreter::{
    check_slm_invariants, find_hits, InterpreterError, LabelHit, PolicyInterpreter,
    CHILD_SAFETY_FLOOR, CHILD_SAFETY_LABEL,
};
pub use rate_limiter::{
    round4, MockMonotonicClock, MonotonicClock, RateLimitDecision, RateLimiterError,
    SlmRateLimiter, SystemMonotonicClock,
};
pub use sanitizer::{
    sanitize_context_hints, sanitize_pii_categories, SanitizationEvent, SanitizationReason,
    ALLOWED_CONTEXT_HINT_KEYS, ALLOWED_PII_CATEGORIES, MAX_SIGNALS_JSON_CHARS, MAX_URL_LENGTH,
};
pub use severity::{
    default_rubric, SeverityLevel, SeverityMapper, SeverityRubric, SeverityRubricError,
    UXDisposition,
};
pub use slm_runner::{MockSlmRunner, SlmRunner};
pub use thresholds::{ThresholdEntry, ThresholdsConfig, ThresholdsError};

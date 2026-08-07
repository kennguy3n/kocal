//! Battery / thermal / foreground-aware gating.
//!
//! On mass-consumer mobile deployments the heavy stages of the
//! safety pipeline — the ONNX text/vision encoder (a ~55-137 MB
//! resident ORT session) and the llama.cpp SLM (a ~442 MiB GGUF
//! mmap + multi-millisecond decode) — dominate CPU, memory, and
//! battery. When the device is under battery / thermal pressure
//! or the host app is backgrounded, running those stages is both
//! wasteful and user-hostile (it drains battery and heats the
//! handset for a chat the user is not even looking at).
//!
//! [`DeviceState`] is the closed-shape snapshot the host passes
//! at call time; [`DeviceState::gating_plan`] turns it into a
//! [`GatingPlan`] of *advisory* booleans the runtime applies:
//!
//! | Condition                              | Effect                                          |
//! |----------------------------------------|-------------------------------------------------|
//! | `battery_level < 0.15 && !is_charging` | skip SLM **and** skip ONNX encoder (deterministic-only) |
//! | `thermal_state >= Serious`             | skip SLM, reduce ONNX intra-threads to 1        |
//! | `!is_foreground`                       | skip SLM, debounce ONNX to at most 1 call / second |
//!
//! The plan is the *union* of every triggered rule — e.g. a
//! backgrounded handset at 10 % on battery both skips the SLM
//! and drops to deterministic-only. The deterministic detector
//! chain (URL / PII / scam / lexicon) always runs: it is
//! microsecond-cheap and carries the child-safety floor, so
//! safety coverage never drops to zero regardless of device
//! state.
//!
//! This type is feature-gate-free — every deployment that wires
//! the orchestrator can describe its device state — but the
//! runtime only *acts* on the plan where the corresponding heavy
//! stage is compiled in.

/// Coarse thermal pressure bucket, mirroring the OS-native
/// thermal APIs the host reads it from: Apple's
/// `ProcessInfo.thermalState` (`nominal` / `fair` / `serious` /
/// `critical`) and Android's `PowerManager.getCurrentThermalStatus`
/// (`NONE`/`LIGHT`/`MODERATE` → … → `SHUTDOWN`). The host maps its
/// platform value onto the nearest bucket.
///
/// `Ord` is derived with the variants in ascending-severity order
/// so the gating rule can be written as `thermal_state >=
/// ThermalState::Serious`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum ThermalState {
    /// No thermal pressure — full pipeline allowed.
    #[default]
    Nominal,
    /// Mild warming; still safe to run everything.
    Fair,
    /// Sustained load is causing real heat — back off the SLM and
    /// pin the encoder to a single core.
    Serious,
    /// Thermal throttling imminent / active — same back-off as
    /// `Serious` (the runtime treats `>= Serious` uniformly).
    Critical,
}

/// Host-supplied device snapshot consulted at classify / decide
/// time to gate the heavy pipeline stages. All fields are coarse
/// and privacy-safe — no precise telemetry, just enough to decide
/// whether the expensive models are worth running right now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceState {
    /// Battery charge in `[0.0, 1.0]`, or `None` when the host
    /// cannot read it (e.g. a desktop on AC with no battery). A
    /// `None` battery never triggers the low-battery rule — the
    /// runtime treats "unknown" as "not constrained".
    pub battery_level: Option<f32>,
    /// Whether the device is currently charging. A charging
    /// device is never low-battery-gated even at a low level.
    pub is_charging: bool,
    /// Coarse thermal pressure bucket.
    pub thermal_state: ThermalState,
    /// Whether the host app is in the foreground. Backgrounded
    /// apps skip the SLM and debounce the encoder — the user is
    /// not looking at the chat, so latency is irrelevant and
    /// battery is precious.
    pub is_foreground: bool,
}

impl Default for DeviceState {
    /// The unconstrained default: full charge intent (unknown
    /// battery, not charging), no thermal pressure, foreground.
    /// Produces an all-`false` [`GatingPlan`] — i.e. "run
    /// everything", preserving pre-gating behaviour for callers
    /// that pass no device state.
    fn default() -> Self {
        Self {
            battery_level: None,
            is_charging: false,
            thermal_state: ThermalState::Nominal,
            is_foreground: true,
        }
    }
}

/// Battery level below which (when not charging) the device is
/// considered critically low and drops to deterministic-only.
pub const LOW_BATTERY_THRESHOLD: f32 = 0.15;

/// Advisory set of back-off decisions derived from a
/// [`DeviceState`]. Every field defaults to `false` (= "run the
/// stage"); the runtime applies each where the corresponding
/// stage is compiled in and actionable at call time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GatingPlan {
    /// Skip the SLM decode entirely and stay on the rule-based
    /// fast path. Wired into [`PolicyInput::allow_slm`] at the
    /// `policy_decide` boundary.
    ///
    /// [`PolicyInput::allow_slm`]: crate::policy_interpreter::PolicyInput::allow_slm
    pub skip_slm: bool,
    /// Skip the ONNX encoder and classify through the
    /// deterministic detector chain only. Wired into
    /// `classify_text` / `classify_image_bytes` by routing to the
    /// deterministic-only classifier instead of the encoder.
    pub skip_onnx_encoder: bool,
    /// Advisory hint that any *future* encoder attach should pin
    /// intra-op threads to 1. A live ORT session's thread pool is
    /// fixed at build time and cannot be re-sized per call, so the
    /// runtime cannot act on this for an already-attached encoder;
    /// it is surfaced so a host that rebuilds the encoder under
    /// sustained thermal pressure can pass `intra_threads = 1`.
    pub reduce_onnx_threads: bool,
    /// Advisory hint that the host should debounce encoder calls
    /// to at most one per second while backgrounded. Enforcement
    /// lives at the SDK debounce surface (the runtime cannot
    /// rate-limit calls it never receives), so this is exposed for
    /// the host / debounce layer to honour.
    pub debounce_onnx: bool,
}

impl GatingPlan {
    /// `true` when no back-off is requested — the common
    /// foreground / charged / cool case. Lets the runtime skip the
    /// gating branches entirely on the hot path.
    pub fn is_unconstrained(&self) -> bool {
        *self == GatingPlan::default()
    }
}

impl DeviceState {
    /// Derive the [`GatingPlan`] for this snapshot. The result is
    /// the union of every triggered rule (see the module docs).
    pub fn gating_plan(&self) -> GatingPlan {
        let mut plan = GatingPlan::default();

        // Rule 1 — critically low battery while discharging:
        // deterministic-only (skip both heavy stages). An unknown
        // (`None`) battery is treated as unconstrained.
        let low_battery = matches!(self.battery_level, Some(level) if level < LOW_BATTERY_THRESHOLD)
            && !self.is_charging;
        if low_battery {
            plan.skip_slm = true;
            plan.skip_onnx_encoder = true;
        }

        // Rule 2 — sustained thermal pressure: skip the SLM and
        // ask future encoder attaches to single-thread.
        if self.thermal_state >= ThermalState::Serious {
            plan.skip_slm = true;
            plan.reduce_onnx_threads = true;
        }

        // Rule 3 — backgrounded: skip the SLM and debounce the
        // encoder. The user is not looking at the chat.
        if !self.is_foreground {
            plan.skip_slm = true;
            plan.debounce_onnx = true;
        }

        plan
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_device_state_is_unconstrained() {
        let plan = DeviceState::default().gating_plan();
        assert!(plan.is_unconstrained());
        assert!(!plan.skip_slm);
        assert!(!plan.skip_onnx_encoder);
        assert!(!plan.reduce_onnx_threads);
        assert!(!plan.debounce_onnx);
    }

    #[test]
    fn low_battery_discharging_drops_to_deterministic_only() {
        let state = DeviceState {
            battery_level: Some(0.10),
            is_charging: false,
            ..DeviceState::default()
        };
        let plan = state.gating_plan();
        assert!(plan.skip_slm);
        assert!(plan.skip_onnx_encoder);
        // Not thermally / background constrained.
        assert!(!plan.reduce_onnx_threads);
        assert!(!plan.debounce_onnx);
    }

    #[test]
    fn low_battery_but_charging_is_unconstrained() {
        let state = DeviceState {
            battery_level: Some(0.05),
            is_charging: true,
            ..DeviceState::default()
        };
        assert!(state.gating_plan().is_unconstrained());
    }

    #[test]
    fn unknown_battery_never_triggers_low_battery_rule() {
        let state = DeviceState {
            battery_level: None,
            is_charging: false,
            ..DeviceState::default()
        };
        assert!(state.gating_plan().is_unconstrained());
    }

    #[test]
    fn boundary_battery_level_is_not_low() {
        // Exactly at the threshold is NOT below it.
        let state = DeviceState {
            battery_level: Some(LOW_BATTERY_THRESHOLD),
            is_charging: false,
            ..DeviceState::default()
        };
        assert!(state.gating_plan().is_unconstrained());
    }

    #[test]
    fn serious_thermal_skips_slm_and_reduces_threads() {
        let state = DeviceState {
            thermal_state: ThermalState::Serious,
            ..DeviceState::default()
        };
        let plan = state.gating_plan();
        assert!(plan.skip_slm);
        assert!(plan.reduce_onnx_threads);
        // Encoder still runs (just single-threaded on re-attach).
        assert!(!plan.skip_onnx_encoder);
        assert!(!plan.debounce_onnx);
    }

    #[test]
    fn critical_thermal_behaves_like_serious() {
        let serious = DeviceState {
            thermal_state: ThermalState::Serious,
            ..DeviceState::default()
        }
        .gating_plan();
        let critical = DeviceState {
            thermal_state: ThermalState::Critical,
            ..DeviceState::default()
        }
        .gating_plan();
        assert_eq!(serious, critical);
    }

    #[test]
    fn fair_thermal_is_unconstrained() {
        let state = DeviceState {
            thermal_state: ThermalState::Fair,
            ..DeviceState::default()
        };
        assert!(state.gating_plan().is_unconstrained());
    }

    #[test]
    fn background_skips_slm_and_debounces() {
        let state = DeviceState {
            is_foreground: false,
            ..DeviceState::default()
        };
        let plan = state.gating_plan();
        assert!(plan.skip_slm);
        assert!(plan.debounce_onnx);
        assert!(!plan.skip_onnx_encoder);
        assert!(!plan.reduce_onnx_threads);
    }

    #[test]
    fn rules_compose_as_union() {
        // Backgrounded handset at 8 % on battery: every rule that
        // can fire, fires.
        let state = DeviceState {
            battery_level: Some(0.08),
            is_charging: false,
            thermal_state: ThermalState::Critical,
            is_foreground: false,
        };
        let plan = state.gating_plan();
        assert!(plan.skip_slm);
        assert!(plan.skip_onnx_encoder);
        assert!(plan.reduce_onnx_threads);
        assert!(plan.debounce_onnx);
    }

    #[test]
    fn thermal_state_orders_by_ascending_severity() {
        assert!(ThermalState::Nominal < ThermalState::Fair);
        assert!(ThermalState::Fair < ThermalState::Serious);
        assert!(ThermalState::Serious < ThermalState::Critical);
    }
}

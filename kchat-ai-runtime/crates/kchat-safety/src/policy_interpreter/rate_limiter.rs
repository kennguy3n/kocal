//! Adaptive token-bucket rate limiter for the SLM dispatch path.
//!
//! Mirrors cv-guard's `shared/policy/slm_rate_limiter.py`
//! byte-for-byte: the same Lamport / Cisco-style token-bucket
//! algorithm with lazy on-demand refill against a monotonic clock,
//! the same half-away-from-zero rounding at 4 decimal places, and
//! the same `(allowed, tokens_remaining, retry_after_seconds)`
//! decision tuple.
//!
//! Why it exists
//! -------------
//!
//! The SLM is the most expensive step in the pipeline (10-100 ms
//! wall-clock vs <1 ms for the rule path). An adversary who can
//! spam triggered classifier scores at the device — for example
//! by forwarding a large gallery into a single chat thread faster
//! than a human ever would — can:
//!
//! * drain device battery while the SLM is consulted on every
//!   frame,
//! * amplify any latent prompt-injection vector (every additional
//!   SLM call is another roll of the invariant-check dice),
//! * starve other on-device workloads while the inference thread
//!   is saturated.
//!
//! This module installs a per-host token-bucket rate limiter on
//! the SLM path. When the bucket is empty the interpreter falls
//! back to the conservative rule-path severity (`2`) with a
//! rationale code that ends in `.rate_limited`. The iOS / Android
//! mirrors implement the same algorithm byte-for-byte.
//!
//! Cross-platform parity
//! ---------------------
//!
//! The numeric output (`tokens_remaining`, `retry_after_seconds`)
//! must be **byte-identical** between Python, Rust, Swift, and
//! Kotlin so a single audit log can correlate decisions across
//! platforms. The `_round4` helper here matches Python's
//! `math.floor(x * 10_000 + 0.5) / 10_000` exactly, which is what
//! the JVM's `Math.round` and Swift's `Double.rounded()` produce
//! for non-negative inputs (the only inputs the limiter ever
//! generates).

use std::fmt;
use std::time::Instant;

use parking_lot::Mutex;

/// Outcome of a single [`SlmRateLimiter::try_acquire`] call.
///
/// `tokens_remaining` and `retry_after_seconds` are rounded to
/// 4 decimal places via [`round4`] so the cross-platform mirrors
/// produce byte-identical numbers for the same inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateLimitDecision {
    /// `true` if the token was consumed and the caller may proceed
    /// with the SLM call. `false` if the caller MUST fall back to
    /// the rule path.
    pub allowed: bool,
    /// Snapshot of the bucket immediately *after* the decision.
    /// For an allowed call this is one less than the pre-call
    /// value; for a denied call this is unchanged. Rounded to 4
    /// decimal places.
    pub tokens_remaining: f64,
    /// On a denied call, the floor (in seconds) of the time the
    /// caller would have to wait before one full new token has
    /// accumulated. `0.0` on an allowed call. Rounded to 4 decimal
    /// places.
    pub retry_after_seconds: f64,
}

/// Errors raised by the rate-limiter builder.
#[derive(Debug, Clone, PartialEq)]
pub enum RateLimiterError {
    /// `capacity` must be `>= 1`.
    InvalidCapacity { capacity: u32 },
    /// `refill_per_second` must be `> 0.0` and finite.
    InvalidRefillRate { refill_per_second: f64 },
}

impl fmt::Display for RateLimiterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RateLimiterError::InvalidCapacity { capacity } => {
                write!(f, "capacity must be >= 1, got {capacity}")
            }
            RateLimiterError::InvalidRefillRate { refill_per_second } => {
                write!(
                    f,
                    "refill_per_second must be > 0 and finite, got {refill_per_second}"
                )
            }
        }
    }
}

impl std::error::Error for RateLimiterError {}

/// Monotonic-clock abstraction: returns elapsed seconds since some
/// fixed reference point. The reference is irrelevant — only
/// **differences** matter to the limiter. The default
/// [`SystemMonotonicClock`] uses [`std::time::Instant`].
///
/// Tests inject a [`MockMonotonicClock`] to make the limiter fully
/// deterministic without sleeping.
pub trait MonotonicClock: Send + Sync {
    /// Return the current monotonic time in seconds (`f64`).
    /// MUST be non-decreasing across calls — moving backwards
    /// would invalidate the lazy-refill invariant.
    fn now_seconds(&self) -> f64;
}

/// Production clock backed by [`std::time::Instant`].
///
/// `Instant` is guaranteed to be monotonic on every Rust target
/// (it is not affected by NTP / wall-clock adjustments), which is
/// what the limiter needs.
pub struct SystemMonotonicClock {
    origin: Instant,
}

impl SystemMonotonicClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemMonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SystemMonotonicClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystemMonotonicClock").finish()
    }
}

impl MonotonicClock for SystemMonotonicClock {
    fn now_seconds(&self) -> f64 {
        // `Duration::as_secs_f64` is the cross-platform way to get
        // floating-point seconds without intermediate `u128` casts.
        self.origin.elapsed().as_secs_f64()
    }
}

/// Deterministic test clock. The clock returns whatever value the
/// test most recently set via [`MockMonotonicClock::set`] (or `0.0` by
/// default). The same pattern as the Python `clock` callable
/// injection.
///
/// Both [`MockMonotonicClock::set`] and [`MockMonotonicClock::advance`]
/// enforce monotonicity — a [`set`](Self::set) call that would move
/// the clock backwards, or an [`advance`](Self::advance) call with a
/// negative delta, panics. This mirrors the Python parity oracle's
/// `_MockMonotonicClock` (`tools/gen_policy_interpreter_runtime_fixtures.py`)
/// which raises `ValueError` on the same conditions. The type name
/// carries `Monotonic` for a reason; allowing backwards moves would
/// let a test silently violate the contract the production
/// `SystemMonotonicClock` cannot violate, masking bugs.
pub struct MockMonotonicClock {
    now: Mutex<f64>,
}

impl MockMonotonicClock {
    pub fn new() -> Self {
        Self {
            now: Mutex::new(0.0),
        }
    }

    pub fn at(initial_seconds: f64) -> Self {
        Self {
            now: Mutex::new(initial_seconds),
        }
    }

    /// Set the clock to `value` seconds. Panics if `value` would move
    /// the clock backwards from its current position. Mirrors the
    /// Python parity oracle's `_MockMonotonicClock.set` which raises
    /// `ValueError` on the same condition.
    pub fn set(&self, value: f64) {
        let mut guard = self.now.lock();
        assert!(
            value >= *guard,
            "MockMonotonicClock must be monotonically non-decreasing \
             (attempted to set t={value} when t was already {now})",
            now = *guard,
        );
        *guard = value;
    }

    /// Advance the clock by `delta_seconds`. Panics if
    /// `delta_seconds` is negative. Mirrors the Python parity oracle's
    /// `_MockMonotonicClock.advance` which raises `ValueError` on the
    /// same condition.
    pub fn advance(&self, delta_seconds: f64) {
        assert!(
            delta_seconds >= 0.0,
            "MockMonotonicClock::advance requires non-negative dt, got {delta_seconds}",
        );
        let mut guard = self.now.lock();
        *guard += delta_seconds;
    }
}

impl Default for MockMonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MockMonotonicClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let snapshot = *self.now.lock();
        f.debug_struct("MockMonotonicClock")
            .field("now", &snapshot)
            .finish()
    }
}

impl MonotonicClock for MockMonotonicClock {
    fn now_seconds(&self) -> f64 {
        *self.now.lock()
    }
}

/// Per-host token-bucket rate limiter for SLM invocations.
///
/// See the module docstring for the algorithm. The surface here
/// is deliberately minimal so the three platform ports can be
/// audited side-by-side.
///
/// # Concurrency
///
/// All state mutations are guarded by a single [`Mutex`]; the
/// lock is held for the duration of the refill arithmetic only
/// (a few floating-point ops) so contention is bounded even
/// under parallel callers.
pub struct SlmRateLimiter {
    capacity: u32,
    refill_per_second: f64,
    clock: Box<dyn MonotonicClock>,
    state: Mutex<BucketState>,
}

#[derive(Debug)]
struct BucketState {
    tokens: f64,
    last_refill: f64,
}

impl SlmRateLimiter {
    /// Build a new limiter using the production
    /// [`SystemMonotonicClock`].
    ///
    /// * `capacity` — maximum tokens the bucket can hold; also the
    ///   burst size. Must be `>= 1`.
    /// * `refill_per_second` — tokens added per second of
    ///   monotonic wall-clock. Must be finite and `> 0`.
    pub fn new(capacity: u32, refill_per_second: f64) -> Result<Self, RateLimiterError> {
        Self::with_clock(
            capacity,
            refill_per_second,
            Box::new(SystemMonotonicClock::new()),
        )
    }

    /// Build a limiter with a custom clock (tests inject a
    /// [`MockMonotonicClock`]).
    pub fn with_clock(
        capacity: u32,
        refill_per_second: f64,
        clock: Box<dyn MonotonicClock>,
    ) -> Result<Self, RateLimiterError> {
        if capacity < 1 {
            return Err(RateLimiterError::InvalidCapacity { capacity });
        }
        if !refill_per_second.is_finite() || refill_per_second <= 0.0 {
            return Err(RateLimiterError::InvalidRefillRate { refill_per_second });
        }
        let now = clock.now_seconds();
        Ok(Self {
            capacity,
            refill_per_second,
            clock,
            state: Mutex::new(BucketState {
                tokens: capacity as f64,
                last_refill: now,
            }),
        })
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn refill_per_second(&self) -> f64 {
        self.refill_per_second
    }

    /// Return the current token count without consuming.
    ///
    /// Refills the bucket against the clock so a long idle period
    /// followed by [`SlmRateLimiter::snapshot_tokens`] shows the
    /// correct value. Primarily used by tests; production code
    /// should call [`SlmRateLimiter::try_acquire`] directly.
    pub fn snapshot_tokens(&self) -> f64 {
        let mut guard = self.lock_state();
        self.refill_locked(&mut guard);
        guard.tokens
    }

    /// Attempt to consume one token.
    ///
    /// Returns [`RateLimitDecision`]; the interpreter consults the
    /// `allowed` flag to decide between the SLM path and the rule
    /// fallback.
    pub fn try_acquire(&self) -> RateLimitDecision {
        let mut guard = self.lock_state();
        self.refill_locked(&mut guard);
        if guard.tokens >= 1.0 {
            // Floating-point subtraction is exact for values well
            // below 2**52, which the bucket never exceeds in
            // practice (capacity is a small int).
            guard.tokens -= 1.0;
            return RateLimitDecision {
                allowed: true,
                tokens_remaining: round4(guard.tokens),
                retry_after_seconds: 0.0,
            };
        }
        // Denied: compute how long until one full token has
        // accumulated. `refill_per_second > 0` is enforced in
        // `new` so the division is safe.
        let shortfall = 1.0 - guard.tokens;
        let wait = shortfall / self.refill_per_second;
        RateLimitDecision {
            allowed: false,
            tokens_remaining: round4(guard.tokens),
            retry_after_seconds: round4(wait),
        }
    }

    fn lock_state(&self) -> parking_lot::MutexGuard<'_, BucketState> {
        self.state.lock()
    }

    fn refill_locked(&self, state: &mut BucketState) {
        let now = self.clock.now_seconds();
        let delta = now - state.last_refill;
        if delta <= 0.0 {
            // Monotonic clock pinned, or the test clock didn't
            // advance. Move the watermark forward to today's `now`
            // either way — the same behaviour as the Python
            // reference, which catches the `delta <= 0.0` branch
            // before doing any arithmetic.
            state.last_refill = now;
            return;
        }
        let added = delta * self.refill_per_second;
        let capped = (state.tokens + added).min(self.capacity as f64);
        state.tokens = capped;
        state.last_refill = now;
    }
}

impl fmt::Debug for SlmRateLimiter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = self.state.lock();
        let snapshot = format!("tokens={:.4}, last_refill={:.4}", g.tokens, g.last_refill);
        f.debug_struct("SlmRateLimiter")
            .field("capacity", &self.capacity)
            .field("refill_per_second", &self.refill_per_second)
            .field("state", &snapshot)
            .finish()
    }
}

/// Round to 4 decimal places using half-up (half-away-from-zero
/// for non-negative inputs).
///
/// The cross-platform contract is that the three ports — Python,
/// Swift, Kotlin — and Rust produce **byte-identical**
/// `tokens_remaining` / `retry_after_seconds` for the same input.
///
/// Python's reference uses `math.floor(x * 10_000 + 0.5) / 10_000`
/// which matches:
/// * Swift's `Double.rounded()` (half-away-from-zero) on
///   non-negative inputs
/// * Kotlin's `Math.round` (half-up to nearest integer)
///
/// All three values rounded here are non-negative (token counts +
/// wait times), so half-up and half-away-from-zero behave
/// identically. The symmetric branch is provided for the future
/// where a caller might pass a signed value.
pub fn round4(x: f64) -> f64 {
    if x >= 0.0 {
        (x * 10_000.0 + 0.5).floor() / 10_000.0
    } else {
        -((-x * 10_000.0 + 0.5).floor()) / 10_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "values not within eps: {a} vs {b} (eps={eps})",
        );
    }

    #[test]
    fn mock_monotonic_clock_advance_accumulates() {
        let clock = MockMonotonicClock::new();
        approx_eq(clock.now_seconds(), 0.0, 1e-12);
        clock.advance(1.5);
        approx_eq(clock.now_seconds(), 1.5, 1e-12);
        clock.advance(2.25);
        approx_eq(clock.now_seconds(), 3.75, 1e-12);
    }

    #[test]
    fn mock_monotonic_clock_advance_zero_is_a_no_op() {
        // Equal-time is on the non-strict side of "non-decreasing", so
        // dt = 0.0 is allowed. This mirrors the Python parity oracle's
        // `t < self._now` (strict less-than) rejection rule.
        let clock = MockMonotonicClock::at(5.0);
        clock.advance(0.0);
        approx_eq(clock.now_seconds(), 5.0, 1e-12);
    }

    #[test]
    #[should_panic(expected = "MockMonotonicClock::advance requires non-negative dt")]
    fn mock_monotonic_clock_advance_rejects_negative_delta() {
        let clock = MockMonotonicClock::at(10.0);
        clock.advance(-0.001);
    }

    #[test]
    fn mock_monotonic_clock_set_to_equal_time_is_allowed() {
        // Same monotonic semantics as `advance(0.0)` — `t = self._now`
        // is non-decreasing, so allowed. Pinning the boundary because
        // any future tightening to strict greater-than would silently
        // diverge from the Python parity oracle.
        let clock = MockMonotonicClock::at(7.5);
        clock.set(7.5);
        approx_eq(clock.now_seconds(), 7.5, 1e-12);
    }

    #[test]
    fn mock_monotonic_clock_set_advances_forward() {
        let clock = MockMonotonicClock::at(2.0);
        clock.set(5.0);
        approx_eq(clock.now_seconds(), 5.0, 1e-12);
    }

    #[test]
    #[should_panic(expected = "MockMonotonicClock must be monotonically non-decreasing")]
    fn mock_monotonic_clock_set_rejects_backwards_move() {
        let clock = MockMonotonicClock::at(10.0);
        clock.set(9.999);
    }

    #[test]
    fn builder_rejects_zero_capacity() {
        let err = SlmRateLimiter::new(0, 1.0).unwrap_err();
        assert!(matches!(
            err,
            RateLimiterError::InvalidCapacity { capacity: 0 }
        ));
    }

    #[test]
    fn builder_rejects_zero_refill() {
        let err = SlmRateLimiter::new(1, 0.0).unwrap_err();
        match err {
            RateLimiterError::InvalidRefillRate { refill_per_second } => {
                assert_eq!(refill_per_second, 0.0);
            }
            _ => panic!("expected InvalidRefillRate"),
        }
    }

    #[test]
    fn builder_rejects_negative_refill() {
        let err = SlmRateLimiter::new(1, -1.0).unwrap_err();
        assert!(matches!(err, RateLimiterError::InvalidRefillRate { .. }));
    }

    #[test]
    fn builder_rejects_nonfinite_refill() {
        let nan = SlmRateLimiter::new(1, f64::NAN).unwrap_err();
        let inf = SlmRateLimiter::new(1, f64::INFINITY).unwrap_err();
        assert!(matches!(nan, RateLimiterError::InvalidRefillRate { .. }));
        assert!(matches!(inf, RateLimiterError::InvalidRefillRate { .. }));
    }

    #[test]
    fn fresh_bucket_is_full() {
        let clock = Box::new(MockMonotonicClock::new());
        let limiter = SlmRateLimiter::with_clock(5, 1.0, clock).unwrap();
        approx_eq(limiter.snapshot_tokens(), 5.0, 1e-9);
    }

    #[test]
    fn try_acquire_consumes_one_token_when_available() {
        let clock = Box::new(MockMonotonicClock::new());
        let limiter = SlmRateLimiter::with_clock(3, 1.0, clock).unwrap();
        let d = limiter.try_acquire();
        assert!(d.allowed);
        assert_eq!(d.tokens_remaining, 2.0);
        assert_eq!(d.retry_after_seconds, 0.0);
    }

    #[test]
    fn try_acquire_denies_when_bucket_empty() {
        let clock = Box::new(MockMonotonicClock::new());
        let limiter = SlmRateLimiter::with_clock(2, 1.0, clock).unwrap();
        let _ = limiter.try_acquire();
        let _ = limiter.try_acquire();
        let d = limiter.try_acquire();
        assert!(!d.allowed);
        assert_eq!(d.tokens_remaining, 0.0);
        // 1.0 token needed at 1.0/s -> 1.0s wait.
        assert_eq!(d.retry_after_seconds, 1.0);
    }

    #[test]
    fn refill_arithmetic_matches_python_reference() {
        // Mirror cv-guard's
        // `tests/shared/test_slm_rate_limiter.py::test_lazy_refill`.
        //
        // Bucket capacity=10, refill=2.0/s. Drain to 4 tokens
        // (consume 6), then advance the clock 1.5s. Expected
        // tokens after refill: 4 + (2.0 * 1.5) = 7.0.
        let clock = std::sync::Arc::new(MockMonotonicClock::at(0.0));
        let clock_handle = clock.clone();
        struct ClockProxy(std::sync::Arc<MockMonotonicClock>);
        impl MonotonicClock for ClockProxy {
            fn now_seconds(&self) -> f64 {
                self.0.now_seconds()
            }
        }
        let limiter = SlmRateLimiter::with_clock(10, 2.0, Box::new(ClockProxy(clock))).unwrap();
        for _ in 0..6 {
            assert!(limiter.try_acquire().allowed);
        }
        approx_eq(limiter.snapshot_tokens(), 4.0, 1e-9);
        clock_handle.advance(1.5);
        approx_eq(limiter.snapshot_tokens(), 7.0, 1e-9);
    }

    #[test]
    fn refill_is_capped_at_capacity() {
        let clock = std::sync::Arc::new(MockMonotonicClock::at(0.0));
        struct ClockProxy(std::sync::Arc<MockMonotonicClock>);
        impl MonotonicClock for ClockProxy {
            fn now_seconds(&self) -> f64 {
                self.0.now_seconds()
            }
        }
        let limiter =
            SlmRateLimiter::with_clock(3, 5.0, Box::new(ClockProxy(clock.clone()))).unwrap();
        let _ = limiter.try_acquire();
        let _ = limiter.try_acquire();
        approx_eq(limiter.snapshot_tokens(), 1.0, 1e-9);
        // Advance 10s -> would add 50 tokens but cap is 3.
        clock.advance(10.0);
        approx_eq(limiter.snapshot_tokens(), 3.0, 1e-9);
    }

    #[test]
    fn retry_after_uses_partial_token_state() {
        // Capacity=1, refill=4.0/s. Drain, advance 0.1s (gives 0.4
        // tokens), expect denied with retry_after = (1.0 - 0.4) /
        // 4.0 = 0.15s.
        let clock = std::sync::Arc::new(MockMonotonicClock::at(0.0));
        struct ClockProxy(std::sync::Arc<MockMonotonicClock>);
        impl MonotonicClock for ClockProxy {
            fn now_seconds(&self) -> f64 {
                self.0.now_seconds()
            }
        }
        let limiter =
            SlmRateLimiter::with_clock(1, 4.0, Box::new(ClockProxy(clock.clone()))).unwrap();
        assert!(limiter.try_acquire().allowed);
        clock.advance(0.1);
        let d = limiter.try_acquire();
        assert!(!d.allowed);
        // 0.4 tokens accumulated, rounded to 4 decimals.
        assert_eq!(d.tokens_remaining, 0.4);
        assert_eq!(d.retry_after_seconds, 0.15);
    }

    #[test]
    fn round4_matches_python_reference_half_up() {
        // Boundary cases lifted from cv-guard's test_round4 cases.
        assert_eq!(round4(0.0), 0.0);
        assert_eq!(round4(0.00004), 0.0);
        assert_eq!(round4(0.00005), 0.0001); // half-up, not banker's
        assert_eq!(round4(0.00006), 0.0001);
        assert_eq!(round4(0.12345), 0.1235);
        assert_eq!(round4(0.123449), 0.1234);
        assert_eq!(round4(1.0), 1.0);
        assert_eq!(round4(1.99995), 2.0);
        // Negative branch: symmetric (half-away-from-zero).
        assert_eq!(round4(-0.00005), -0.0001);
        assert_eq!(round4(-1.99995), -2.0);
    }

    #[test]
    fn clock_pinned_at_zero_delta_is_safe() {
        // If the test clock doesn't advance between two refills,
        // the limiter should NOT crash on division-by-zero or NaN.
        let clock = std::sync::Arc::new(MockMonotonicClock::at(0.0));
        struct ClockProxy(std::sync::Arc<MockMonotonicClock>);
        impl MonotonicClock for ClockProxy {
            fn now_seconds(&self) -> f64 {
                self.0.now_seconds()
            }
        }
        let limiter = SlmRateLimiter::with_clock(2, 1.0, Box::new(ClockProxy(clock))).unwrap();
        let a = limiter.try_acquire();
        let b = limiter.try_acquire();
        let c = limiter.try_acquire(); // bucket empty
        assert!(a.allowed && b.allowed && !c.allowed);
        // Same clock value: tokens_remaining stable.
        assert_eq!(c.tokens_remaining, 0.0);
    }

    #[test]
    fn debug_format_is_concise_and_does_not_panic() {
        let clock = Box::new(MockMonotonicClock::new());
        let limiter = SlmRateLimiter::with_clock(5, 1.0, clock).unwrap();
        let dbg = format!("{limiter:?}");
        assert!(dbg.contains("SlmRateLimiter"));
        assert!(dbg.contains("capacity: 5"));
        assert!(dbg.contains("refill_per_second: 1"));
    }
}

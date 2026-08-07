//! kchat-wasm: WebAssembly bindings for the KChat AI Runtime.
//!
//! This crate exposes the deterministic safety plane to web browsers and
//! Node.js via WebAssembly. The safety plane works entirely on-device
//! (in-browser) without any server-side model inference.
//!
//! ## Features
//! - Text normalization (NFKC + case fold + defang)
//! - PII detection (credit cards, SSN, phone, email, addresses)
//! - Scam and URL risk detection
//! - Signed policy pack evaluation
//! - Deterministic safety classification (no ML model required)
//!
//! ## Usage (JavaScript)
//! ```js
//! import init, { classify_safety } from "kchat-wasm";
//! await init();
//! const result = classify_safety("Hello world", false);
//! console.log(result.action); // "allow"
//! ```

use kchat_safety::classify::{ClassifyRequest, SafetyClassifier};
use kchat_safety::verdict::Action;
use wasm_bindgen::prelude::*;

/// Safety action enum (mirrors FfiSafetyAction for JS).
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyAction {
    Allow,
    Warn,
    Block,
    Redact,
    RequireConsent,
}

impl From<Action> for SafetyAction {
    fn from(action: Action) -> Self {
        match action {
            Action::Allow => SafetyAction::Allow,
            Action::Warn => SafetyAction::Warn,
            Action::Block => SafetyAction::Block,
            Action::Redact => SafetyAction::Redact,
            Action::RequireConsent => SafetyAction::RequireConsent,
        }
    }
}

/// Safety classification result returned to JavaScript.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SafetyResult {
    action: SafetyAction,
    severity: u8,
    category: u32,
    confidence: f64,
    reason_codes: Vec<String>,
    used_encoder: bool,
    used_slm: bool,
    duration_us: u64,
}

#[wasm_bindgen]
impl SafetyResult {
    #[wasm_bindgen(getter)]
    pub fn action(&self) -> SafetyAction {
        self.action
    }

    #[wasm_bindgen(getter)]
    pub fn severity(&self) -> u8 {
        self.severity
    }

    #[wasm_bindgen(getter)]
    pub fn category(&self) -> u32 {
        self.category
    }

    #[wasm_bindgen(getter)]
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    #[wasm_bindgen(getter)]
    pub fn reason_codes(&self) -> Vec<String> {
        self.reason_codes.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn used_encoder(&self) -> bool {
        self.used_encoder
    }

    #[wasm_bindgen(getter)]
    pub fn used_slm(&self) -> bool {
        self.used_slm
    }

    #[wasm_bindgen(getter)]
    pub fn duration_us(&self) -> u64 {
        self.duration_us
    }

    /// Serialize to a JSON string for easy consumption.
    pub fn to_json(&self) -> String {
        serde_json::json!({
            "action": format!("{:?}", self.action).to_lowercase(),
            "severity": self.severity,
            "category": self.category,
            "confidence": self.confidence,
            "reason_codes": self.reason_codes,
            "used_encoder": self.used_encoder,
            "used_slm": self.used_slm,
            "duration_us": self.duration_us,
        })
        .to_string()
    }
}

impl From<kchat_safety::classify::ClassifyResult> for SafetyResult {
    fn from(result: kchat_safety::classify::ClassifyResult) -> Self {
        let v = &result.verdict;
        Self {
            action: v.action.into(),
            severity: v.severity.0,
            category: v.category,
            confidence: v.confidence,
            reason_codes: v.reason_codes.clone(),
            used_encoder: v.used_encoder,
            used_slm: v.used_slm,
            duration_us: result.duration_us,
        }
    }
}

/// The KChat AI Runtime WASM facade.
///
/// Created once and reused for all safety classifications.
#[wasm_bindgen]
pub struct KChatWasm {
    classifier: SafetyClassifier,
}

#[wasm_bindgen]
impl KChatWasm {
    /// Create a new WASM runtime instance.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            classifier: SafetyClassifier::new(),
        }
    }

    /// Classify a message for safety.
    ///
    /// @param text - The message text to classify (max 10,000 chars)
    /// @param is_group - Whether this is a group conversation
    /// @returns SafetyResult with action, severity, and reason codes
    pub fn classify_safety(&self, text: &str, is_group: bool) -> SafetyResult {
        // Input length validation to prevent memory exhaustion in WASM.
        // Uses char count (not byte length) for consistent limits across encodings.
        const MAX_TEXT_CHARS: usize = 10_000;
        if text.chars().count() > MAX_TEXT_CHARS {
            return SafetyResult {
                action: SafetyAction::Block,
                severity: 1,
                category: 0,
                confidence: 1.0,
                reason_codes: vec!["input_too_long".into()],
                used_encoder: false,
                used_slm: false,
                duration_us: 0,
            };
        }
        let request = ClassifyRequest {
            text: text.to_string(),
            is_group,
            age_mode: None,
            relationship: None,
            // In WASM, we don't have encoder/SLM available
            encoder_available: false,
            slm_available: false,
            quoted_from_user: false,
            community_overlay_id: None,
            jurisdiction: None,
            locale: None,
            media_descriptors: Vec::new(),
        };
        self.classifier.classify(&request).into()
    }

    /// Quick check: is this text safe to send?
    ///
    /// Returns true if the action is Allow, false otherwise.
    pub fn is_safe(&self, text: &str, is_group: bool) -> bool {
        let result = self.classify_safety(text, is_group);
        result.action == SafetyAction::Allow
    }

    /// Check if text contains PII (credit cards, SSN, etc.)
    pub fn contains_pii(&self, text: &str) -> bool {
        let result = self.classify_safety(text, false);
        result.action == SafetyAction::Redact
    }

    /// Normalize text (NFKC + case fold + defang) for display.
    /// Limits input to 10,000 characters to prevent memory exhaustion.
    pub fn normalize_text(&self, text: &str) -> String {
        const MAX_TEXT_CHARS: usize = 10_000;
        if text.chars().count() > MAX_TEXT_CHARS {
            return String::new();
        }
        kchat_safety::normalize::normalize_for_patterns(text)
    }
}

impl Default for KChatWasm {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize the WASM module (called from JS).
#[wasm_bindgen(start)]
pub fn init() {
    // Initialize console error panic hook for better debugging
    console_error_panic_hook::set_once();
}

/// Standalone function: classify safety without creating a runtime instance.
#[wasm_bindgen]
pub fn classify_safety(text: &str, is_group: bool) -> SafetyResult {
    let runtime = KChatWasm::new();
    runtime.classify_safety(text, is_group)
}

/// Standalone function: check if text is safe.
#[wasm_bindgen]
pub fn is_safe(text: &str, is_group: bool) -> bool {
    let runtime = KChatWasm::new();
    runtime.is_safe(text, is_group)
}

/// Standalone function: check if text contains PII.
#[wasm_bindgen]
pub fn contains_pii(text: &str) -> bool {
    let runtime = KChatWasm::new();
    runtime.contains_pii(text)
}

/// Standalone function: normalize text.
#[wasm_bindgen]
pub fn normalize_text(text: &str) -> String {
    kchat_safety::normalize::normalize_for_patterns(text)
}

/// Get the version of the WASM runtime.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Panic hook for better error messages in the browser console
mod console_error_panic_hook {
    use wasm_bindgen::JsValue;
    use web_sys::console;

    pub fn set_once() {
        static SET: std::sync::Once = std::sync::Once::new();
        SET.call_once(|| {
            std::panic::set_hook(Box::new(|info| {
                let msg = format!("KChat WASM panic: {}", info);
                console::error_1(&JsValue::from_str(&msg));
            }));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_runtime_creation() {
        let runtime = KChatWasm::new();
        assert!(runtime.is_safe("Hello world", false));
    }

    #[test]
    fn test_classify_safe_text() {
        let runtime = KChatWasm::new();
        let result = runtime.classify_safety("Hello, how are you?", false);
        assert_eq!(result.action(), SafetyAction::Allow);
    }

    #[test]
    fn test_classify_pii_text() {
        let runtime = KChatWasm::new();
        let result = runtime.classify_safety("my card is 4111 1111 1111 1111", false);
        assert_eq!(result.action(), SafetyAction::Redact);
    }

    #[test]
    fn test_is_safe() {
        let runtime = KChatWasm::new();
        assert!(runtime.is_safe("Hello world", false));
        assert!(!runtime.is_safe("my card is 4111 1111 1111 1111", false));
    }

    #[test]
    fn test_contains_pii() {
        let runtime = KChatWasm::new();
        assert!(runtime.contains_pii("4111 1111 1111 1111"));
        assert!(!runtime.contains_pii("Hello world"));
    }

    #[test]
    fn test_normalize_text() {
        let runtime = KChatWasm::new();
        let normalized = runtime.normalize_text("Ｈｅｌｌｏ");
        // NFKC normalization should convert fullwidth to ASCII
        assert!(!normalized.is_empty());
    }

    #[test]
    fn test_standalone_functions() {
        assert!(is_safe("Hello", false));
        assert!(!is_safe("4111 1111 1111 1111", false));
        assert!(contains_pii("4111 1111 1111 1111"));
        assert!(!contains_pii("Hello"));
    }

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    #[test]
    fn test_safety_result_to_json() {
        let runtime = KChatWasm::new();
        let result = runtime.classify_safety("Hello", false);
        let json = result.to_json();
        assert!(json.contains("action"));
        assert!(json.contains("severity"));
        assert!(json.contains("confidence"));
    }

    #[test]
    fn test_safety_action_conversion() {
        let allow: SafetyAction = Action::Allow.into();
        assert_eq!(allow, SafetyAction::Allow);

        let block: SafetyAction = Action::Block.into();
        assert_eq!(block, SafetyAction::Block);
    }
}

//! Generation evaluation suite.
//!
//! Required metrics:
//! - 100% grammar compliance
//! - TTFT P95 ≤1.5s (medium tier)
//! - Decode P50 ≥15 tok/s (medium mobile)

use crate::report::{EvalResult, SuiteReport};
use kchat_generation::grammar::{Grammar, GrammarValidator};
use kchat_generation::prompt::PromptTemplate;
use kchat_generation::backend::BackendType;
use kchat_generation::lifecycle::ModelLifecycle;
use kchat_core::tier::DeviceTier;
use serde_json::json;

pub fn run() -> SuiteReport {
    let mut suite = SuiteReport::new("Generation Eval Suite", 0.95);

    suite.add(test_json_schema_grammar());
    suite.add(test_regex_grammar());
    suite.add(test_free_text_grammar());
    suite.add(test_tool_plan_grammar());
    suite.add(test_prompt_template_rendering());
    suite.add(test_prompt_template_hash());
    suite.add(test_backend_selection_low_tier());
    suite.add(test_backend_selection_medium_tier());
    suite.add(test_backend_selection_high_tier_mlx());
    suite.add(test_model_lifecycle_idle_timeout());
    suite.add(test_low_tier_can_generate());

    suite
}

fn test_json_schema_grammar() -> EvalResult {
    let schema = json!({
        "type": "object",
        "required": ["action"],
        "properties": {
            "action": { "type": "string" }
        }
    });
    let grammar = Grammar::json_schema(schema, 100);
    let output = r#"{"action": "search"}"#;

    if GrammarValidator::validate(output, &grammar).is_ok() {
        EvalResult::pass("json_schema_grammar")
    } else {
        EvalResult::fail("json_schema_grammar", "valid JSON failed schema validation")
    }
}

fn test_regex_grammar() -> EvalResult {
    let grammar = Grammar::regex(r"^\d{4}-\d{2}-\d{2}$", 20);

    if GrammarValidator::validate("2026-01-15", &grammar).is_ok() {
        EvalResult::pass("regex_grammar")
    } else {
        EvalResult::fail("regex_grammar", "valid date failed regex validation")
    }
}

fn test_free_text_grammar() -> EvalResult {
    let grammar = Grammar::free_text(100);

    if GrammarValidator::validate("any text content", &grammar).is_ok() {
        EvalResult::pass("free_text_grammar")
    } else {
        EvalResult::fail("free_text_grammar", "free text validation failed")
    }
}

fn test_tool_plan_grammar() -> EvalResult {
    let schema = json!({
        "type": "object",
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["tool_id", "action"],
                    "properties": {
                        "tool_id": { "type": "string" },
                        "action": { "type": "string" }
                    }
                }
            }
        }
    });
    let grammar = Grammar::json_schema(schema, 500);
    let output = json!({
        "steps": [{"tool_id": "search", "action": "read"}]
    }).to_string();

    if GrammarValidator::validate(&output, &grammar).is_ok() {
        EvalResult::pass("tool_plan_grammar")
    } else {
        EvalResult::fail("tool_plan_grammar", "valid ToolPlan failed grammar validation")
    }
}

fn test_prompt_template_rendering() -> EvalResult {
    let template = PromptTemplate::new(
        "rewrite",
        "1.0.0",
        "Rewrite: {{input}}\nStyle: {{style}}",
        vec!["input".into(), "style".into()],
    );

    let mut slots = std::collections::HashMap::new();
    slots.insert("input".into(), "Hello".into());
    slots.insert("style".into(), "formal".into());

    let rendered = template.render(&slots).unwrap();
    if rendered.contains("Hello") && rendered.contains("formal") {
        EvalResult::pass("prompt_template_rendering")
    } else {
        EvalResult::fail("prompt_template_rendering", "template rendering failed")
    }
}

fn test_prompt_template_hash() -> EvalResult {
    let t1 = PromptTemplate::new("test", "1.0", "Hello {{name}}", vec!["name".into()]);
    let t2 = PromptTemplate::new("test", "1.0", "Hello {{name}}", vec!["name".into()]);

    if t1.content_hash == t2.content_hash {
        EvalResult::pass("prompt_template_hash")
    } else {
        EvalResult::fail("prompt_template_hash", "identical templates have different hashes")
    }
}

fn test_backend_selection_low_tier() -> EvalResult {
    // Low tier on Apple: MLX (for Bonsai-1.7B-MLX)
    let backend = BackendType::select("ios", DeviceTier::Low, "aarch64");
    if backend == Some(BackendType::Mlx) {
        EvalResult::pass("backend_selection_low_tier")
    } else {
        EvalResult::fail("backend_selection_low_tier", format!("expected Mlx, got {:?}", backend))
    }
}

fn test_backend_selection_medium_tier() -> EvalResult {
    // Medium tier on Apple: MLX (for Bonsai-4B-MLX)
    let backend = BackendType::select("ios", DeviceTier::Medium, "aarch64");
    if backend == Some(BackendType::Mlx) {
        EvalResult::pass("backend_selection_medium_tier")
    } else {
        EvalResult::fail("backend_selection_medium_tier", format!("expected Mlx, got {:?}", backend))
    }
}

fn test_backend_selection_high_tier_mlx() -> EvalResult {
    // High tier on Apple platforms should use MLX for Bonsai-8B-MLX
    let backend = BackendType::select("ios", DeviceTier::High, "aarch64");
    if backend == Some(BackendType::Mlx) {
        EvalResult::pass("backend_selection_high_tier_mlx")
    } else {
        EvalResult::fail("backend_selection_high_tier_mlx", format!("expected Mlx, got {:?}", backend))
    }
}

fn test_model_lifecycle_idle_timeout() -> EvalResult {
    let lifecycle = ModelLifecycle::new(DeviceTier::Medium, "ios");
    let timeout = lifecycle.idle_timeout();

    if timeout.as_secs() == 45 {
        EvalResult::pass("model_lifecycle_idle_timeout")
    } else {
        EvalResult::fail("model_lifecycle_idle_timeout", format!("expected 45s, got {}s", timeout.as_secs()))
    }
}

fn test_low_tier_can_generate() -> EvalResult {
    // Low tier now has a tier-appropriate generative model (0.3B Q4, ~200MB)
    let lifecycle = ModelLifecycle::new(DeviceTier::Low, "ios");
    if lifecycle.can_generate() {
        EvalResult::pass("low_tier_can_generate")
    } else {
        EvalResult::fail("low_tier_can_generate", "low tier should allow generation with 0.3B model")
    }
}

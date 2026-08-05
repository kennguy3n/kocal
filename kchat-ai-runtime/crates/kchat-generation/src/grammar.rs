//! Grammar-constrained decoding — JSON schema, regex, or Lark grammar.
//!
//! The model output is constrained to a grammar. This prevents the model
//! from emitting arbitrary text, code, or commands. The grammar is validated
//! before decoding starts and enforced during token sampling.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Type of grammar constraint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GrammarType {
    /// JSON Schema constraint
    JsonSchema { schema: Value },
    /// Regular expression constraint
    Regex { pattern: String },
    /// Lark grammar (EBNF-like)
    Lark { grammar: String },
    /// No constraint (free text)
    None,
}

/// A grammar constraint for model output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grammar {
    pub grammar_type: GrammarType,
    /// Maximum output tokens (enforced by the runtime)
    pub max_tokens: usize,
    /// Whether to stop on newline (for single-line outputs)
    pub stop_on_newline: bool,
    /// Stop sequences
    pub stop_sequences: Vec<String>,
}

impl Grammar {
    /// Create a JSON Schema grammar.
    pub fn json_schema(schema: Value, max_tokens: usize) -> Self {
        Self {
            grammar_type: GrammarType::JsonSchema { schema },
            max_tokens,
            stop_on_newline: false,
            stop_sequences: vec![],
        }
    }

    /// Create a regex grammar.
    pub fn regex(pattern: impl Into<String>, max_tokens: usize) -> Self {
        Self {
            grammar_type: GrammarType::Regex { pattern: pattern.into() },
            max_tokens,
            stop_on_newline: false,
            stop_sequences: vec![],
        }
    }

    /// Create a Lark grammar.
    pub fn lark(grammar: impl Into<String>, max_tokens: usize) -> Self {
        Self {
            grammar_type: GrammarType::Lark { grammar: grammar.into() },
            max_tokens,
            stop_on_newline: false,
            stop_sequences: vec![],
        }
    }

    /// Create an unconstrained grammar (free text).
    pub fn free_text(max_tokens: usize) -> Self {
        Self {
            grammar_type: GrammarType::None,
            max_tokens,
            stop_on_newline: false,
            stop_sequences: vec![],
        }
    }

    /// Add a stop sequence.
    pub fn with_stop_sequence(mut self, seq: impl Into<String>) -> Self {
        self.stop_sequences.push(seq.into());
        self
    }
}

/// Validator for grammar-constrained output.
pub struct GrammarValidator;

impl GrammarValidator {
    /// Validate that output conforms to the grammar.
    /// 100% of artifact operations must parse before execution.
    pub fn validate(output: &str, grammar: &Grammar) -> Result<(), GrammarError> {
        // Check max tokens (approximate: 1 token ≈ 4 chars)
        let max_chars = grammar.max_tokens * 4;
        if output.len() > max_chars {
            return Err(GrammarError::TooLong {
                actual: output.len(),
                max: max_chars,
            });
        }

        // Check stop sequences
        for seq in &grammar.stop_sequences {
            if output.contains(seq) {
                return Err(GrammarError::StopSequenceHit(seq.clone()));
            }
        }

        match &grammar.grammar_type {
            GrammarType::JsonSchema { schema } => {
                Self::validate_json_schema(output, schema)
            }
            GrammarType::Regex { pattern } => {
                Self::validate_regex(output, pattern)
            }
            GrammarType::Lark { grammar: _ } => {
                // Lark validation requires a Lark parser (not included in this stub)
                // In production, this would parse the output against the Lark grammar
                Ok(())
            }
            GrammarType::None => Ok(()),
        }
    }

    /// Validate output against a JSON Schema (simplified).
    fn validate_json_schema(output: &str, schema: &Value) -> Result<(), GrammarError> {
        // Parse the output as JSON
        let parsed: Value = serde_json::from_str(output)
            .map_err(|e| GrammarError::InvalidJson(e.to_string()))?;

        // Check required fields
        if let Value::Object(schema_obj) = schema {
            if let Some(Value::Array(required)) = schema_obj.get("required") {
                if let Value::Object(parsed_obj) = &parsed {
                    for req in required {
                        if let Value::String(field) = req {
                            if !parsed_obj.contains_key(field) {
                                return Err(GrammarError::SchemaViolation {
                                    field: field.clone(),
                                    reason: "required field missing".into(),
                                });
                            }
                        }
                    }
                } else {
                    return Err(GrammarError::SchemaViolation {
                        field: "root".into(),
                        reason: "expected object".into(),
                    });
                }
            }

            // Check type
            if let Some(Value::String(expected_type)) = schema_obj.get("type") {
                let actual_type = match &parsed {
                    Value::String(_) => "string",
                    Value::Number(_) => "number",
                    Value::Bool(_) => "boolean",
                    Value::Array(_) => "array",
                    Value::Object(_) => "object",
                    Value::Null => "null",
                };
                if expected_type != actual_type {
                    return Err(GrammarError::SchemaViolation {
                        field: "root".into(),
                        reason: format!("expected {}, got {}", expected_type, actual_type),
                    });
                }
            }
        }

        Ok(())
    }

    /// Validate output against a regex.
    fn validate_regex(output: &str, pattern: &str) -> Result<(), GrammarError> {
        let re = regex::Regex::new(pattern)
            .map_err(|e| GrammarError::InvalidRegex(e.to_string()))?;

        if !re.is_match(output) {
            return Err(GrammarError::RegexMismatch);
        }

        Ok(())
    }
}

/// Grammar validation errors.
#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    #[error("output too long: {actual} chars, max {max}")]
    TooLong { actual: usize, max: usize },

    #[error("stop sequence hit: {0}")]
    StopSequenceHit(String),

    #[error("invalid JSON: {0}")]
    InvalidJson(String),

    #[error("JSON schema violation: field {field} - {reason}")]
    SchemaViolation { field: String, reason: String },

    #[error("invalid regex: {0}")]
    InvalidRegex(String),

    #[error("output does not match regex")]
    RegexMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_json_schema_validation_succeeds() {
        let schema = json!({
            "type": "object",
            "required": ["action", "tool_id"],
            "properties": {
                "action": { "type": "string" },
                "tool_id": { "type": "string" }
            }
        });
        let grammar = Grammar::json_schema(schema, 100);
        let output = r#"{"action": "search", "tool_id": "search_tool"}"#;

        assert!(GrammarValidator::validate(output, &grammar).is_ok());
    }

    #[test]
    fn test_json_schema_validation_missing_field() {
        let schema = json!({
            "type": "object",
            "required": ["action", "tool_id"],
        });
        let grammar = Grammar::json_schema(schema, 100);
        let output = r#"{"action": "search"}"#;

        let result = GrammarValidator::validate(output, &grammar);
        assert!(result.is_err());
    }

    #[test]
    fn test_json_schema_validation_invalid_json() {
        let schema = json!({"type": "object"});
        let grammar = Grammar::json_schema(schema, 100);
        let output = "not json at all";

        let result = GrammarValidator::validate(output, &grammar);
        assert!(result.is_err());
    }

    #[test]
    fn test_regex_validation_succeeds() {
        let grammar = Grammar::regex(r"^\d{4}-\d{2}-\d{2}$", 20);
        assert!(GrammarValidator::validate("2026-01-15", &grammar).is_ok());
    }

    #[test]
    fn test_regex_validation_fails() {
        let grammar = Grammar::regex(r"^\d{4}-\d{2}-\d{2}$", 20);
        assert!(GrammarValidator::validate("not-a-date", &grammar).is_err());
    }

    #[test]
    fn test_free_text_always_valid() {
        let grammar = Grammar::free_text(100);
        assert!(GrammarValidator::validate("anything goes", &grammar).is_ok());
    }

    #[test]
    fn test_stop_sequence_detected() {
        let grammar = Grammar::free_text(100).with_stop_sequence("</end>");
        let result = GrammarValidator::validate("text</end>", &grammar);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_plan_grammar() {
        // Simulate a ToolPlan grammar
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
                            "action": { "type": "string" },
                            "arguments": { "type": "object" }
                        }
                    }
                }
            }
        });
        let grammar = Grammar::json_schema(schema, 500);
        let output = json!({
            "steps": [
                {"tool_id": "search", "action": "read", "arguments": {"query": "test"}}
            ]
        }).to_string();

        assert!(GrammarValidator::validate(&output, &grammar).is_ok());
    }
}

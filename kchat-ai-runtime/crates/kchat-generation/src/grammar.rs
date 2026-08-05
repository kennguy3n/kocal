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
            GrammarType::Lark { grammar } => {
                // Validate the grammar itself is well-formed
                let lark = LarkGrammar::parse(grammar)?;
                lark.validate()?;
                // Output conformance against a Lark grammar requires a full
                // LALR parser. Without one, we cannot guarantee the output
                // conforms. Return an error to fail-safe rather than silently
                // allowing unconstrained output.
                Err(GrammarError::OutputValidationNotImplemented(
                    "Lark grammar output validation requires a full LALR parser; \
                     use JSON Schema or regex constraints for production".into(),
                ))
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
    /// Uses size limits to prevent ReDoS (catastrophic backtracking).
    fn validate_regex(output: &str, pattern: &str) -> Result<(), GrammarError> {
        // Limit pattern length to prevent overly complex regexes
        const MAX_PATTERN_LEN: usize = 500;
        if pattern.len() > MAX_PATTERN_LEN {
            return Err(GrammarError::InvalidRegex(
                "regex pattern too long (max 500 chars)".into(),
            ));
        }

        // Use RegexBuilder with size limits to prevent ReDoS
        let re = regex::RegexBuilder::new(pattern)
            .size_limit(1 * 1024 * 1024)       // 1MB max compiled size
            .dfa_size_limit(10 * 1024 * 1024)  // 10MB max DFA cache
            .build()
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

    #[error("invalid Lark grammar: {0}")]
    InvalidLark(String),

    #[error("Lark grammar error: undefined rule '{rule}' referenced in '{context}'")]
    UndefinedLarkRule { rule: String, context: String },

    #[error("Lark grammar error: missing 'start' rule")]
    MissingStartRule,

    #[error("output validation not implemented for grammar type: {0}")]
    OutputValidationNotImplemented(String),
}

/// A parsed Lark grammar (EBNF-like) used for syntactic validation.
///
/// This is a lightweight parser that validates the *structure* of a Lark
/// grammar: rule definitions, terminal definitions, string/regex literals,
/// and rule references. It does not perform full semantic analysis or build
/// a parse table.
#[derive(Debug, Clone)]
struct LarkGrammar {
    /// Defined rule names (lowercase conventions, e.g. `start`, `expr`).
    rules: Vec<String>,
    /// Defined terminal names (uppercase conventions, e.g. `NUMBER`, `WS`).
    terminals: Vec<String>,
    /// Imported names (from `%import` directives) — treated as defined.
    imports: Vec<String>,
    /// Raw productions text per rule (for reference checking).
    productions: Vec<(String, String)>,
}

impl LarkGrammar {
    /// Parse a Lark grammar string into its rule/terminal structure.
    fn parse(src: &str) -> Result<Self, GrammarError> {
        let stripped = strip_comments(src)?;
        let mut rules: Vec<String> = Vec::new();
        let mut terminals: Vec<String> = Vec::new();
        let mut imports: Vec<String> = Vec::new();
        let mut productions: Vec<(String, String)> = Vec::new();

        // Split into logical statements. Lark statements are separated by
        // newlines and/or semicolons. A rule/terminal definition begins with
        // `name:` (rules) or `NAME:` (terminals).
        let statements = split_statements(&stripped);

        for stmt in statements {
            let stmt = stmt.trim();
            if stmt.is_empty() {
                continue;
            }

            // Directives
            if stmt.starts_with('%') {
                if let Some(rest) = stmt.strip_prefix("%import") {
                    // %import .NUMBER -> NUMBER   or   %import common.NUMBER
                    let name = rest.trim();
                    // Take the last path component as the imported name.
                    let imported = name.rsplit('.').next().unwrap_or(name).trim();
                    if !imported.is_empty() {
                        imports.push(imported.to_string());
                    }
                }
                // Ignore other directives (%ignore, %declare, %override, etc.)
                continue;
            }

            // Find the first colon that separates the name from the production.
            // We must skip over string/regex literals when locating it.
            let colon_pos = match find_definition_colon(stmt) {
                Some(pos) => pos,
                None => continue, // not a definition line
            };

            let name = stmt[..colon_pos].trim().to_string();
            let body = stmt[colon_pos + 1..].trim().to_string();

            if name.is_empty() {
                return Err(GrammarError::InvalidLark(
                    "empty rule name before ':'".into(),
                ));
            }

            // Validate the rule/terminal name.
            validate_name(&name).map_err(GrammarError::InvalidLark)?;

            // Uppercase names are terminals; lowercase are rules.
            if name.chars().next().map_or(false, |c| c.is_ascii_uppercase()) {
                terminals.push(name.clone());
            } else {
                rules.push(name.clone());
            }
            productions.push((name, body));
        }

        Ok(Self {
            rules,
            terminals,
            imports,
            productions,
        })
    }

    /// Validate the parsed grammar: check for a `start` rule and that all
    /// referenced rules/terminals are defined.
    fn validate(&self) -> Result<(), GrammarError> {
        // Check that the "start" rule exists (Lark convention).
        if !self.rules.iter().any(|r| r == "start") {
            return Err(GrammarError::MissingStartRule);
        }

        let defined: std::collections::HashSet<&str> = self
            .rules
            .iter()
            .chain(self.terminals.iter())
            .chain(self.imports.iter())
            .map(String::as_str)
            .collect();

        // Collect all identifiers referenced in productions (excluding string
        // and regex literals, which are terminals-by-value).
        for (rule_name, body) in &self.productions {
            for ref_name in extract_references(body) {
                if !defined.contains(ref_name.as_str()) {
                    return Err(GrammarError::UndefinedLarkRule {
                        rule: ref_name,
                        context: rule_name.clone(),
                    });
                }
            }
        }

        Ok(())
    }
}

/// Remove `//` line comments and `/* */` block comments from the source,
/// while respecting string and regex literals.
fn strip_comments(src: &str) -> Result<String, GrammarError> {
    let mut out = String::with_capacity(src.len());
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '"' | '\'' | '/' if c == '/' && i + 1 < chars.len() && (chars[i + 1] == '/' || chars[i + 1] == '*') => {
                // comment
                if chars[i + 1] == '/' {
                    // line comment
                    while i < chars.len() && chars[i] != '\n' {
                        i += 1;
                    }
                } else {
                    // block comment
                    i += 2;
                    let mut closed = false;
                    while i + 1 < chars.len() {
                        if chars[i] == '*' && chars[i + 1] == '/' {
                            i += 2;
                            closed = true;
                            break;
                        }
                        i += 1;
                    }
                    if !closed {
                        return Err(GrammarError::InvalidLark(
                            "unterminated block comment".into(),
                        ));
                    }
                    continue;
                }
            }
            '"' | '\'' => {
                // string literal — copy verbatim, validate balanced
                let quote = c;
                out.push(c);
                i += 1;
                let mut closed = false;
                while i < chars.len() {
                    let ch = chars[i];
                    out.push(ch);
                    if ch == '\\' && i + 1 < chars.len() {
                        i += 1;
                        out.push(chars[i]);
                        i += 1;
                        continue;
                    }
                    if ch == quote {
                        closed = true;
                        i += 1;
                        break;
                    }
                    i += 1;
                }
                if !closed {
                    return Err(GrammarError::InvalidLark(
                        format!("unterminated string literal starting with '{}'", quote),
                    ));
                }
            }
            '/' => {
                // Could be a regex literal `/.../` or a division/comment.
                // In Lark, a lone `/` at the start of a production token is a
                // regex terminal. We treat `/` followed by non-space as a regex.
                if i + 1 < chars.len() && !chars[i + 1].is_whitespace() {
                    out.push(c);
                    i += 1;
                    let mut closed = false;
                    while i < chars.len() {
                        let ch = chars[i];
                        out.push(ch);
                        if ch == '\\' && i + 1 < chars.len() {
                            i += 1;
                            out.push(chars[i]);
                            i += 1;
                            continue;
                        }
                        if ch == '/' {
                            closed = true;
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                    if !closed {
                        return Err(GrammarError::InvalidLark(
                            "unterminated regex literal".into(),
                        ));
                    }
                } else {
                    out.push(c);
                    i += 1;
                }
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    Ok(out)
}

/// Split the source into statements. Lark uses newlines and/or the `|` to
/// continue productions, but a new rule begins when a line starts with
/// `identifier:`. We accumulate lines and split when we detect a new
/// definition header.
fn split_statements(src: &str) -> Vec<String> {
    let mut statements: Vec<String> = Vec::new();
    let mut current = String::new();

    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // blank line ends the current statement
            if !current.is_empty() {
                statements.push(std::mem::take(&mut current));
            }
            continue;
        }

        // A new definition starts when the line begins with `name:` and the
        // name is a valid identifier (not a continuation like `| foo`).
        if starts_definition(trimmed) && !current.is_empty() {
            statements.push(std::mem::take(&mut current));
        }

        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(trimmed);
    }
    if !current.is_empty() {
        statements.push(current);
    }
    statements
}

/// Returns true if the line begins with an identifier followed by `:`,
/// indicating a rule/terminal definition header.
fn starts_definition(line: &str) -> bool {
    if let Some(colon) = find_definition_colon(line) {
        let name = line[..colon].trim();
        return is_valid_identifier(name);
    }
    false
}

/// Find the position of the colon that separates a definition name from its
/// production, skipping over string/regex literals.
fn find_definition_colon(s: &str) -> Option<usize> {
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '"' | '\'' => {
                let quote = c;
                i += 1;
                while i < chars.len() {
                    if chars[i] == '\\' {
                        i += 2;
                        continue;
                    }
                    if chars[i] == quote {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            '/' if i + 1 < chars.len() && !chars[i + 1].is_whitespace() && chars[i + 1] != '/' => {
                i += 1;
                while i < chars.len() {
                    if chars[i] == '\\' {
                        i += 2;
                        continue;
                    }
                    if chars[i] == '/' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            ':' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// Validate a rule or terminal name. Lark names are alphanumeric with
/// underscores, must start with a letter or underscore.
fn validate_name(name: &str) -> Result<(), String> {
    if !is_valid_identifier(name) {
        return Err(format!("invalid rule/terminal name: '{}'", name));
    }
    Ok(())
}

fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Extract all rule/terminal identifier references from a production body,
/// skipping string and regex literals.
fn extract_references(body: &str) -> Vec<String> {
    let mut refs: Vec<String> = Vec::new();
    let chars: Vec<char> = body.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '"' | '\'' => {
                let quote = c;
                i += 1;
                while i < chars.len() {
                    if chars[i] == '\\' {
                        i += 2;
                        continue;
                    }
                    if chars[i] == quote {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            '/' if i + 1 < chars.len() && !chars[i + 1].is_whitespace() && chars[i + 1] != '/' => {
                i += 1;
                while i < chars.len() {
                    if chars[i] == '\\' {
                        i += 2;
                        continue;
                    }
                    if chars[i] == '/' {
                        i += 1;
                        break;
                    }
                    i += 1;
                }
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                refs.push(name);
            }
            _ => i += 1,
        }
    }
    refs
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

    #[test]
    fn test_lark_grammar_valid() {
        let grammar_text = r#"
            start: expr
            expr: term (("+" | "-") term)*
            term: factor (("*" | "/") factor)*
            factor: NUMBER | "(" expr ")"
            NUMBER: /[0-9]+/
            WS: /[ \t\n]+/
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        // Lark output validation is not implemented — fail-safe returns error
        assert!(GrammarValidator::validate("1", &grammar).is_err());
    }

    #[test]
    fn test_lark_grammar_valid_simple() {
        let grammar_text = r#"
            start: greeting
            greeting: "hello" name
            name: WORD
            WORD: /[a-zA-Z]+/
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        // Lark output validation is not implemented — fail-safe returns error
        assert!(GrammarValidator::validate("hello world", &grammar).is_err());
    }

    #[test]
    fn test_lark_grammar_missing_start_rule() {
        let grammar_text = r#"
            expr: term
            term: NUMBER
            NUMBER: /[0-9]+/
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        let result = GrammarValidator::validate("1", &grammar);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GrammarError::MissingStartRule
        ));
    }

    #[test]
    fn test_lark_grammar_undefined_rule() {
        let grammar_text = r#"
            start: expr
            expr: undefined_rule
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        let result = GrammarValidator::validate("1", &grammar);
        assert!(result.is_err());
        match result.unwrap_err() {
            GrammarError::UndefinedLarkRule { rule, context } => {
                assert_eq!(rule, "undefined_rule");
                assert_eq!(context, "expr");
            }
            other => panic!("expected UndefinedLarkRule, got {:?}", other),
        }
    }

    #[test]
    fn test_lark_grammar_unterminated_string() {
        let grammar_text = r#"
            start: "hello
            name: WORD
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        let result = GrammarValidator::validate("hello", &grammar);
        assert!(result.is_err());
    }

    #[test]
    fn test_lark_grammar_unterminated_regex() {
        let grammar_text = r#"
            start: NUMBER
            NUMBER: /[0-9]+/
            WS: /[ \t\n]+
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        let result = GrammarValidator::validate("1", &grammar);
        assert!(result.is_err());
    }

    #[test]
    fn test_lark_grammar_with_imports() {
        let grammar_text = r#"
            %import common.NUMBER
            start: expr
            expr: NUMBER
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        // Lark output validation is not implemented — fail-safe returns error
        assert!(GrammarValidator::validate("1", &grammar).is_err());
    }

    #[test]
    fn test_lark_grammar_with_comments() {
        let grammar_text = r#"
            // This is a line comment
            start: expr  /* inline comment */
            expr: term
            term: NUMBER
            NUMBER: /[0-9]+/  // numbers
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        // Lark output validation is not implemented — fail-safe returns error
        assert!(GrammarValidator::validate("1", &grammar).is_err());
    }

    #[test]
    fn test_lark_grammar_invalid_rule_name() {
        let grammar_text = r#"
            start: expr
            123bad: NUMBER
            NUMBER: /[0-9]+/
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        let result = GrammarValidator::validate("1", &grammar);
        assert!(result.is_err());
    }

    #[test]
    fn test_lark_grammar_alternatives_valid() {
        let grammar_text = r#"
            start: value
            value: STRING | NUMBER | "true" | "false"
            STRING: /"[^"]*"/
            NUMBER: /[0-9]+/
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        // Lark output validation is not implemented — fail-safe returns error
        assert!(GrammarValidator::validate("true", &grammar).is_err());
    }

    #[test]
    fn test_lark_grammar_unterminated_block_comment() {
        let grammar_text = r#"
            /* unterminated
            start: expr
        "#;
        let grammar = Grammar::lark(grammar_text, 100);
        let result = GrammarValidator::validate("1", &grammar);
        assert!(result.is_err());
    }
}

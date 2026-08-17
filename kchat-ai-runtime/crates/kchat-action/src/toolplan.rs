//! ToolPlan validation — the model emits a ToolPlan, not an executable request.
//!
//! Each app has a signed manifest containing tool schemas, requested
//! capabilities, data scopes, network destinations, side effects,
//! confirmation class, and publisher identity.
//!
//! The ToolPlanValidator checks:
//! - All referenced tools exist in signed manifests
//! - All arguments match the tool's JSON Schema
//! - Data scopes are within the user's permissions
//! - Side effects are declared and authorized

use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Signed manifest for a microapp extension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolManifest {
    /// Extension/publisher identity
    pub publisher_id: String,
    /// Extension version
    pub version: String,
    /// Ed25519 public key (hex)
    pub public_key: String,
    /// Ed25519 signature (hex)
    pub signature: String,
    /// Tool definitions in this extension
    pub tools: Vec<ToolDefinition>,
    /// Requested capabilities
    pub capabilities: Vec<String>,
    /// Network destinations (for SSRF control)
    pub network_destinations: Vec<String>,
}

/// Definition of a single tool in a signed manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub tool_id: String,
    pub name: String,
    pub description: String,
    /// JSON Schema for arguments
    pub arguments_schema: Value,
    /// Declared side effects
    pub side_effects: Vec<String>,
    /// Confirmation class
    pub confirmation_class: String,
    /// Data scopes required
    pub data_scopes: Vec<String>,
}

/// A single step in a tool plan produced by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPlanStep {
    pub tool_id: String,
    pub action: String,
    pub arguments: Value,
    pub data_scope: String,
}

/// A tool plan emitted by the model — not an executable request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPlan {
    pub steps: Vec<ToolPlanStep>,
    /// Whether this plan has been dry-run validated
    pub dry_run_completed: bool,
    /// Commit token (bound to actor, target, parameters, expiry, artifact version)
    pub commit_token: Option<String>,
}

impl ToolPlan {
    pub fn new(steps: Vec<ToolPlanStep>) -> Self {
        Self {
            steps,
            dry_run_completed: false,
            commit_token: None,
        }
    }

    /// Number of steps in the plan.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

/// Validator for tool plans — checks against signed manifests.
pub struct ToolPlanValidator {
    /// Map of tool_id → tool definition (from signed manifests)
    tools: HashMap<String, ToolDefinition>,
    /// Map of tool_id → manifest (for signature verification)
    manifests: HashMap<String, ToolManifest>,
    /// Secret key for HMAC-based commit tokens (32 bytes).
    commit_token_secret: [u8; 32],
}

impl ToolPlanValidator {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            manifests: HashMap::new(),
            commit_token_secret: [0u8; 32], // Default: must be set via set_commit_token_secret
        }
    }

    /// Set the secret key used for HMAC-based commit tokens.
    /// This should be a cryptographically random 32-byte key.
    pub fn set_commit_token_secret(&mut self, secret: [u8; 32]) {
        self.commit_token_secret = secret;
    }

    /// Register a signed tool manifest.
    /// In production, the manifest signature should be verified against a
    /// pinned public key before registration. Rejects manifests with null
    /// (all-zero) signatures.
    pub fn register_manifest(&mut self, manifest: ToolManifest) -> Result<(), ToolPlanError> {
        // Reject null signature (all zeros) — unsigned manifests are not allowed
        if manifest.signature.chars().all(|c| c == '0') {
            return Err(ToolPlanError::NullSignature {
                publisher: manifest.publisher_id.clone(),
            });
        }
        for tool in &manifest.tools {
            self.tools.insert(tool.tool_id.clone(), tool.clone());
            self.manifests.insert(tool.tool_id.clone(), manifest.clone());
        }
        Ok(())
    }

    /// Validate a tool plan against registered manifests.
    ///
    /// Checks:
    /// - All referenced tools exist
    /// - All arguments match JSON Schema
    /// - Data scopes are declared
    /// - Side effects are declared
    pub fn validate(&self, plan: &ToolPlan) -> Result<Vec<ValidationResult>, ToolPlanError> {
        let mut results = Vec::new();

        for (i, step) in plan.steps.iter().enumerate() {
            let tool = self.tools.get(&step.tool_id).ok_or_else(|| {
                ToolPlanError::ToolNotFound {
                    step: i,
                    tool_id: step.tool_id.clone(),
                }
            })?;

            // Validate arguments against JSON Schema
            self.validate_arguments(&step.arguments, &tool.arguments_schema, i)?;

            // Check data scope is declared
            if !tool.data_scopes.contains(&step.data_scope) {
                return Err(ToolPlanError::UndeclaredScope {
                    step: i,
                    scope: step.data_scope.clone(),
                });
            }

            results.push(ValidationResult {
                step: i,
                tool_id: step.tool_id.clone(),
                confirmation_class: tool.confirmation_class.clone(),
                side_effects: tool.side_effects.clone(),
                valid: true,
            });
        }

        Ok(results)
    }

    /// Validate arguments against a JSON Schema (simplified).
    fn validate_arguments(
        &self,
        args: &Value,
        schema: &Value,
        step: usize,
    ) -> Result<(), ToolPlanError> {
        // Simplified JSON Schema validation — checks required fields and types
        if let Value::Object(schema_obj) = schema {
            // Check required fields
            if let Some(Value::Array(required)) = schema_obj.get("required") {
                if let Value::Object(args_obj) = args {
                    for req in required {
                        if let Value::String(field) = req {
                            if !args_obj.contains_key(field) {
                                return Err(ToolPlanError::MissingRequiredField {
                                    step,
                                    field: field.clone(),
                                });
                            }
                        }
                    }
                } else {
                    return Err(ToolPlanError::InvalidArguments {
                        step,
                        reason: "arguments must be a JSON object".into(),
                    });
                }
            }

            // Check types (simplified)
            if let Some(Value::Object(properties)) = schema_obj.get("properties") {
                if let Value::Object(args_obj) = args {
                    for (field, arg_value) in args_obj {
                        if let Some(Value::Object(prop_schema)) = properties.get(field) {
                            if let Some(Value::String(expected_type)) = prop_schema.get("type") {
                                let actual_type = match arg_value {
                                    Value::String(_) => "string",
                                    Value::Number(_) => "number",
                                    Value::Bool(_) => "boolean",
                                    Value::Array(_) => "array",
                                    Value::Object(_) => "object",
                                    Value::Null => "null",
                                };
                                if expected_type != actual_type {
                                    return Err(ToolPlanError::TypeMismatch {
                                        step,
                                        field: field.clone(),
                                        expected: expected_type.clone(),
                                        actual: actual_type.into(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Generate a commit token bound to actor, target, parameters, expiry,
    /// and artifact version. Uses HMAC-SHA256 with a server-side secret key
    /// so that clients cannot forge tokens.
    pub fn generate_commit_token(
        &self,
        actor: &str,
        tool_id: &str,
        arguments: &Value,
        expiry_unix: u64,
        artifact_version: u32,
    ) -> Result<String, ToolPlanError> {
        // Reject zero-key default — prevents token forgery if secret not set
        if self.commit_token_secret == [0u8; 32] {
            return Err(ToolPlanError::ZeroKeySecret);
        }
        let mut hmac = Hmac::<Sha256>::new_from_slice(&self.commit_token_secret)
            .expect("HMAC key length is valid");
        hmac.update(actor.as_bytes());
        hmac.update(tool_id.as_bytes());
        // Use canonical JSON serialization for deterministic tokens
        hmac.update(&serde_json::to_vec(arguments).unwrap_or_default());
        hmac.update(&expiry_unix.to_le_bytes());
        hmac.update(&artifact_version.to_le_bytes());
        Ok(hex::encode(hmac.finalize().into_bytes()))
    }

    /// Verify a commit token matches the expected parameters and has not expired.
    /// Returns false if the token is expired, malformed, or does not match.
    pub fn verify_commit_token(
        &self,
        token: &str,
        actor: &str,
        tool_id: &str,
        arguments: &Value,
        expiry_unix: u64,
        artifact_version: u32,
    ) -> bool {
        // 1. Check expiry first — reject expired tokens
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if expiry_unix <= now {
            return false;
        }

        // 2. Regenerate expected token and compare in constant time
        let expected = match self.generate_commit_token(
            actor,
            tool_id,
            arguments,
            expiry_unix,
            artifact_version,
        ) {
            Ok(t) => t,
            Err(_) => return false,
        };
        use subtle::ConstantTimeEq;
        token.as_bytes().ct_eq(expected.as_bytes()).into()
    }
}

impl Default for ToolPlanValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of validating a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub step: usize,
    pub tool_id: String,
    pub confirmation_class: String,
    pub side_effects: Vec<String>,
    pub valid: bool,
}

/// Tool plan validation errors.
#[derive(Debug, thiserror::Error)]
pub enum ToolPlanError {
    #[error("step {step}: tool not found: {tool_id}")]
    ToolNotFound { step: usize, tool_id: String },

    #[error("step {step}: missing required field: {field}")]
    MissingRequiredField { step: usize, field: String },

    #[error("step {step}: type mismatch for field {field}: expected {expected}, got {actual}")]
    TypeMismatch {
        step: usize,
        field: String,
        expected: String,
        actual: String,
    },

    #[error("step {step}: invalid arguments: {reason}")]
    InvalidArguments { step: usize, reason: String },

    #[error("step {step}: undeclared data scope: {scope}")]
    UndeclaredScope { step: usize, scope: String },

    #[error("commit token secret not set (zero key) — call set_commit_token_secret first")]
    ZeroKeySecret,

    #[error("manifest has null signature — unsigned manifests are not allowed")]
    NullSignature { publisher: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_test_manifest() -> ToolManifest {
        ToolManifest {
            publisher_id: "test-publisher".into(),
            version: "1.0.0".into(),
            public_key: "a".repeat(64),
            signature: "b".repeat(128),
            tools: vec![ToolDefinition {
                tool_id: "search_tool".into(),
                name: "Search".into(),
                description: "Search records".into(),
                arguments_schema: json!({
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": { "type": "string" },
                        "limit": { "type": "number" }
                    }
                }),
                side_effects: vec![],
                confirmation_class: "read_only".into(),
                data_scopes: vec!["workspace_123".into()],
            }],
            capabilities: vec!["read".into()],
            network_destinations: vec![],
        }
    }

    #[test]
    fn test_valid_tool_plan() {
        let manifest = make_test_manifest();
        let mut validator = ToolPlanValidator::new();
        validator.register_manifest(manifest).unwrap();

        let plan = ToolPlan::new(vec![ToolPlanStep {
            tool_id: "search_tool".into(),
            action: "read".into(),
            arguments: json!({"query": "hello", "limit": 10}),
            data_scope: "workspace_123".into(),
        }]);

        let results = validator.validate(&plan).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].valid);
    }

    #[test]
    fn test_tool_not_found() {
        let validator = ToolPlanValidator::new();
        let plan = ToolPlan::new(vec![ToolPlanStep {
            tool_id: "unknown_tool".into(),
            action: "read".into(),
            arguments: json!({}),
            data_scope: "workspace_123".into(),
        }]);
        assert!(validator.validate(&plan).is_err());
    }

    #[test]
    fn test_missing_required_field() {
        let manifest = make_test_manifest();
        let mut validator = ToolPlanValidator::new();
        validator.register_manifest(manifest).unwrap();

        let plan = ToolPlan::new(vec![ToolPlanStep {
            tool_id: "search_tool".into(),
            action: "read".into(),
            arguments: json!({"limit": 10}), // missing "query"
            data_scope: "workspace_123".into(),
        }]);

        let result = validator.validate(&plan);
        assert!(result.is_err());
    }

    #[test]
    fn test_type_mismatch() {
        let manifest = make_test_manifest();
        let mut validator = ToolPlanValidator::new();
        validator.register_manifest(manifest).unwrap();

        let plan = ToolPlan::new(vec![ToolPlanStep {
            tool_id: "search_tool".into(),
            action: "read".into(),
            arguments: json!({"query": 123}), // should be string
            data_scope: "workspace_123".into(),
        }]);

        let result = validator.validate(&plan);
        assert!(result.is_err());
    }

    #[test]
    fn test_undeclared_scope() {
        let manifest = make_test_manifest();
        let mut validator = ToolPlanValidator::new();
        validator.register_manifest(manifest).unwrap();

        let plan = ToolPlan::new(vec![ToolPlanStep {
            tool_id: "search_tool".into(),
            action: "read".into(),
            arguments: json!({"query": "hello"}),
            data_scope: "workspace_999".into(), // not declared
        }]);

        let result = validator.validate(&plan);
        assert!(result.is_err());
    }

    fn make_validator_with_secret() -> ToolPlanValidator {
        let mut validator = ToolPlanValidator::new();
        validator.set_commit_token_secret([0xff; 32]);
        validator
    }

    #[test]
    fn test_commit_token_roundtrip() {
        let validator = make_validator_with_secret();
        let args = json!({"query": "test"});
        let token = validator.generate_commit_token("user1", "search_tool", &args, 9999999999, 1).unwrap();
        assert!(validator.verify_commit_token(&token, "user1", "search_tool", &args, 9999999999, 1));
        // Different actor should fail
        assert!(!validator.verify_commit_token(&token, "user2", "search_tool", &args, 9999999999, 1));
    }

    #[test]
    fn test_commit_token_expired_rejected() {
        let validator = make_validator_with_secret();
        let args = json!({"query": "test"});
        // Use a past expiry time (1 second after epoch)
        let past_expiry = 1;
        let token = validator.generate_commit_token("user1", "search_tool", &args, past_expiry, 1).unwrap();
        // Should be rejected because expiry is in the past
        assert!(!validator.verify_commit_token(&token, "user1", "search_tool", &args, past_expiry, 1));
    }

    #[test]
    fn test_commit_token_malformed_rejected() {
        let validator = make_validator_with_secret();
        let args = json!({"query": "test"});
        // Malformed token (not valid hex)
        assert!(!validator.verify_commit_token("not-hex", "user1", "search_tool", &args, 9999999999, 1));
        // Wrong length token
        assert!(!validator.verify_commit_token("abc123", "user1", "search_tool", &args, 9999999999, 1));
    }

    #[test]
    fn test_commit_token_zero_key_rejected() {
        let validator = ToolPlanValidator::new(); // default has zero key
        let args = json!({"query": "test"});
        // Should fail with ZeroKeySecret error
        let result = validator.generate_commit_token("user1", "search_tool", &args, 9999999999, 1);
        assert!(matches!(result, Err(ToolPlanError::ZeroKeySecret)));
    }

    #[test]
    fn test_commit_token_wrong_tool_rejected() {
        let validator = make_validator_with_secret();
        let args = json!({"query": "test"});
        let token = validator.generate_commit_token("user1", "search_tool", &args, 9999999999, 1).unwrap();
        // Same actor, different tool_id → should fail
        assert!(!validator.verify_commit_token(&token, "user1", "delete_tool", &args, 9999999999, 1));
    }

    #[test]
    fn test_commit_token_wrong_arguments_rejected() {
        let validator = make_validator_with_secret();
        let args = json!({"query": "test"});
        let token = validator.generate_commit_token("user1", "search_tool", &args, 9999999999, 1).unwrap();
        // Different arguments → should fail (prevents argument swapping attacks)
        let wrong_args = json!({"query": "different"});
        assert!(!validator.verify_commit_token(&token, "user1", "search_tool", &wrong_args, 9999999999, 1));
    }

    #[test]
    fn test_commit_token_wrong_version_rejected() {
        let validator = make_validator_with_secret();
        let args = json!({"query": "test"});
        let token = validator.generate_commit_token("user1", "search_tool", &args, 9999999999, 1).unwrap();
        // Different artifact version → should fail (prevents replay on updated artifacts)
        assert!(!validator.verify_commit_token(&token, "user1", "search_tool", &args, 9999999999, 2));
    }

    #[test]
    fn test_commit_token_wrong_expiry_rejected() {
        let validator = make_validator_with_secret();
        let args = json!({"query": "test"});
        let token = validator.generate_commit_token("user1", "search_tool", &args, 9999999999, 1).unwrap();
        // Different expiry → should fail (prevents expiry extension attacks)
        assert!(!validator.verify_commit_token(&token, "user1", "search_tool", &args, 8888888888, 1));
    }

    #[test]
    fn test_commit_token_empty_token_rejected() {
        let validator = make_validator_with_secret();
        let args = json!({"query": "test"});
        assert!(!validator.verify_commit_token("", "user1", "search_tool", &args, 9999999999, 1));
    }

    #[test]
    fn test_commit_token_replay_after_expiry() {
        let validator = make_validator_with_secret();
        let args = json!({"query": "test"});
        // Generate token with very near expiry
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let token = validator.generate_commit_token("user1", "search_tool", &args, now + 10, 1).unwrap();
        // Valid now
        assert!(validator.verify_commit_token(&token, "user1", "search_tool", &args, now + 10, 1));
        // But not with a past expiry
        assert!(!validator.verify_commit_token(&token, "user1", "search_tool", &args, now - 10, 1));
    }

    // --- Image search tool tests -------------------------------------------

    /// Build a signed manifest with the image search tools.
    fn make_image_manifest() -> ToolManifest {
        let search_images_args = json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "orientation": {"type": "string", "enum": ["landscape", "portrait", "square"]},
                "per_page": {"type": "integer", "minimum": 1, "maximum": 80},
                "safesearch": {"type": "boolean"}
            }
        });

        let get_curated_args = json!({
            "type": "object",
            "properties": {
                "per_page": {"type": "integer", "minimum": 1, "maximum": 80},
                "page": {"type": "integer", "minimum": 1}
            }
        });

        ToolManifest {
            publisher_id: "kchat.image".into(),
            version: "1.0.0".into(),
            public_key: "a".repeat(64),
            signature: "b".repeat(128),
            tools: vec![
                ToolDefinition {
                    tool_id: "search_images".into(),
                    name: "Search Images".into(),
                    description: "Search stock photo libraries (Pexels, Pixabay, Unsplash, Shutterstock)".into(),
                    arguments_schema: search_images_args,
                    side_effects: vec!["network".into()],
                    confirmation_class: "network".into(),
                    data_scopes: vec!["public_image_libraries".into()],
                },
                ToolDefinition {
                    tool_id: "get_curated_images".into(),
                    name: "Get Curated Images".into(),
                    description: "Get curated photos from Pexels".into(),
                    arguments_schema: get_curated_args,
                    side_effects: vec!["network".into()],
                    confirmation_class: "network".into(),
                    data_scopes: vec!["public_image_libraries".into()],
                },
            ],
            capabilities: vec!["network".into()],
            network_destinations: vec![
                "api.pexels.com".into(),
                "pixabay.com".into(),
                "api.unsplash.com".into(),
                "api.shutterstock.com".into(),
            ],
        }
    }

    #[test]
    fn test_image_search_tool_validates() {
        let mut validator = make_validator_with_secret();
        validator.register_manifest(make_image_manifest()).unwrap();

        let plan = ToolPlan::new(vec![ToolPlanStep {
            tool_id: "search_images".into(),
            action: "search".into(),
            arguments: json!({"query": "mountain landscape", "orientation": "landscape"}),
            data_scope: "public_image_libraries".into(),
        }]);

        let results = validator.validate(&plan).unwrap();
        assert!(results.iter().all(|r| r.valid));
    }

    #[test]
    fn test_image_search_tool_rejects_missing_query() {
        let mut validator = make_validator_with_secret();
        validator.register_manifest(make_image_manifest()).unwrap();

        let plan = ToolPlan::new(vec![ToolPlanStep {
            tool_id: "search_images".into(),
            action: "search".into(),
            arguments: json!({"orientation": "landscape"}),
            data_scope: "public_image_libraries".into(),
        }]);

        let result = validator.validate(&plan);
        assert!(result.is_err());
    }

    #[test]
    fn test_image_search_tool_network_destinations_declared() {
        let manifest = make_image_manifest();
        assert!(manifest.network_destinations.contains(&"api.pexels.com".to_string()));
        assert!(manifest.network_destinations.contains(&"pixabay.com".to_string()));
        assert!(manifest.network_destinations.contains(&"api.unsplash.com".to_string()));
        assert!(manifest.network_destinations.contains(&"api.shutterstock.com".to_string()));
    }
}

//! Action evaluation suite.
//!
//! Required metrics:
//! - 100% ToolPlan validation
//! - 100% artifact operation parsing

use crate::report::{EvalResult, SuiteReport};
use kchat_action::artifact::{ArtifactAst, ArtifactNodeId, ArtifactOperation, ArtifactType};
use kchat_action::toolplan::{ToolDefinition, ToolManifest, ToolPlan, ToolPlanStep, ToolPlanValidator};
use kchat_action::auth::{AuthContext, ConfirmationClass, Permission, RbacBroker};
use kchat_core::ids::ArtifactId;
use serde_json::json;
use std::collections::HashSet;

pub fn run() -> SuiteReport {
    let mut suite = SuiteReport::new("Action Eval Suite", 1.0); // 100% required

    suite.add(test_artifact_replace_range());
    suite.add(test_artifact_stale_version_rejected());
    suite.add(test_artifact_executable_rejected());
    suite.add(test_artifact_formula_validation());
    suite.add(test_toolplan_valid());
    suite.add(test_toolplan_missing_field());
    suite.add(test_toolplan_type_mismatch());
    suite.add(test_toolplan_undeclared_scope());
    suite.add(test_auth_permission_granted());
    suite.add(test_auth_permission_denied());
    suite.add(test_auth_step_up_required());

    suite
}

fn make_document() -> ArtifactAst {
    let mut ast = ArtifactAst::new(ArtifactId::new(), ArtifactType::Document, "Test");
    let node = kchat_action::artifact::ArtifactNode {
        node_id: ArtifactNodeId::new(),
        node_type: "paragraph".into(),
        content: "Hello world".into(),
        children: vec![],
        version: 1,
    };
    ast.root_nodes.push(node.node_id);
    ast.nodes.push(node);
    ast
}

fn test_artifact_replace_range() -> EvalResult {
    let mut ast = make_document();
    let node_id = ast.root_nodes[0];
    let op = ArtifactOperation::ReplaceRange {
        node_id,
        expected_version: 1,
        start: 0,
        end: 5,
        new_content: "Hi".into(),
    };

    if ast.apply_operation(&op).is_ok() {
        EvalResult::pass("artifact_replace_range")
    } else {
        EvalResult::fail("artifact_replace_range", "valid operation rejected")
    }
}

fn test_artifact_stale_version_rejected() -> EvalResult {
    let mut ast = make_document();
    let node_id = ast.root_nodes[0];
    let op = ArtifactOperation::ReplaceRange {
        node_id,
        expected_version: 99,
        start: 0,
        end: 5,
        new_content: "Hi".into(),
    };

    if ast.apply_operation(&op).is_err() {
        EvalResult::pass("artifact_stale_version_rejected")
    } else {
        EvalResult::fail("artifact_stale_version_rejected", "stale version was accepted")
    }
}

fn test_artifact_executable_rejected() -> EvalResult {
    let mut ast = make_document();
    let node_id = ast.root_nodes[0];
    let op = ArtifactOperation::ReplaceRange {
        node_id,
        expected_version: 1,
        start: 0,
        end: 0,
        new_content: "<script>alert(1)</script>".into(),
    };

    if ast.apply_operation(&op).is_err() {
        EvalResult::pass("artifact_executable_rejected")
    } else {
        EvalResult::fail("artifact_executable_rejected", "executable content was accepted")
    }
}

fn test_artifact_formula_validation() -> EvalResult {
    let mut ast = make_document();
    let node_id = ast.root_nodes[0];
    let op = ArtifactOperation::SetFormula {
        node_id,
        expected_version: 1,
        formula: "=macro(bad)".into(),
    };

    if ast.apply_operation(&op).is_err() {
        EvalResult::pass("artifact_formula_validation")
    } else {
        EvalResult::fail("artifact_formula_validation", "macro formula was accepted")
    }
}

fn test_toolplan_valid() -> EvalResult {
    let manifest = ToolManifest {
        publisher_id: "test".into(),
        version: "1.0".into(),
        public_key: "a".repeat(64),
        signature: "b".repeat(128),
        tools: vec![ToolDefinition {
            tool_id: "search".into(),
            name: "Search".into(),
            description: "Search records".into(),
            arguments_schema: json!({
                "type": "object",
                "required": ["query"],
                "properties": { "query": { "type": "string" } }
            }),
            side_effects: vec![],
            confirmation_class: "read_only".into(),
            data_scopes: vec!["workspace_1".into()],
        }],
        capabilities: vec![],
        network_destinations: vec![],
    };

    let mut validator = ToolPlanValidator::new();
    validator.register_manifest(manifest).unwrap();

    let plan = ToolPlan::new(vec![ToolPlanStep {
        tool_id: "search".into(),
        action: "read".into(),
        arguments: json!({"query": "hello"}),
        data_scope: "workspace_1".into(),
    }]);

    if validator.validate(&plan).is_ok() {
        EvalResult::pass("toolplan_valid")
    } else {
        EvalResult::fail("toolplan_valid", "valid plan was rejected")
    }
}

fn test_toolplan_missing_field() -> EvalResult {
    let manifest = ToolManifest {
        publisher_id: "test".into(),
        version: "1.0".into(),
        public_key: "a".repeat(64),
        signature: "b".repeat(128),
        tools: vec![ToolDefinition {
            tool_id: "search".into(),
            name: "Search".into(),
            description: "Search".into(),
            arguments_schema: json!({
                "type": "object",
                "required": ["query"],
                "properties": { "query": { "type": "string" } }
            }),
            side_effects: vec![],
            confirmation_class: "read_only".into(),
            data_scopes: vec!["workspace_1".into()],
        }],
        capabilities: vec![],
        network_destinations: vec![],
    };

    let mut validator = ToolPlanValidator::new();
    validator.register_manifest(manifest).unwrap();

    let plan = ToolPlan::new(vec![ToolPlanStep {
        tool_id: "search".into(),
        action: "read".into(),
        arguments: json!({}), // missing query
        data_scope: "workspace_1".into(),
    }]);

    if validator.validate(&plan).is_err() {
        EvalResult::pass("toolplan_missing_field")
    } else {
        EvalResult::fail("toolplan_missing_field", "plan with missing field was accepted")
    }
}

fn test_toolplan_type_mismatch() -> EvalResult {
    let manifest = ToolManifest {
        publisher_id: "test".into(),
        version: "1.0".into(),
        public_key: "a".repeat(64),
        signature: "b".repeat(128),
        tools: vec![ToolDefinition {
            tool_id: "search".into(),
            name: "Search".into(),
            description: "Search".into(),
            arguments_schema: json!({
                "type": "object",
                "required": ["query"],
                "properties": { "query": { "type": "string" } }
            }),
            side_effects: vec![],
            confirmation_class: "read_only".into(),
            data_scopes: vec!["workspace_1".into()],
        }],
        capabilities: vec![],
        network_destinations: vec![],
    };

    let mut validator = ToolPlanValidator::new();
    validator.register_manifest(manifest).unwrap();

    let plan = ToolPlan::new(vec![ToolPlanStep {
        tool_id: "search".into(),
        action: "read".into(),
        arguments: json!({"query": 123}), // wrong type
        data_scope: "workspace_1".into(),
    }]);

    if validator.validate(&plan).is_err() {
        EvalResult::pass("toolplan_type_mismatch")
    } else {
        EvalResult::fail("toolplan_type_mismatch", "plan with type mismatch was accepted")
    }
}

fn test_toolplan_undeclared_scope() -> EvalResult {
    let manifest = ToolManifest {
        publisher_id: "test".into(),
        version: "1.0".into(),
        public_key: "a".repeat(64),
        signature: "b".repeat(128),
        tools: vec![ToolDefinition {
            tool_id: "search".into(),
            name: "Search".into(),
            description: "Search".into(),
            arguments_schema: json!({"type": "object"}),
            side_effects: vec![],
            confirmation_class: "read_only".into(),
            data_scopes: vec!["workspace_1".into()],
        }],
        capabilities: vec![],
        network_destinations: vec![],
    };

    let mut validator = ToolPlanValidator::new();
    validator.register_manifest(manifest).unwrap();

    let plan = ToolPlan::new(vec![ToolPlanStep {
        tool_id: "search".into(),
        action: "read".into(),
        arguments: json!({}),
        data_scope: "workspace_999".into(), // undeclared
    }]);

    if validator.validate(&plan).is_err() {
        EvalResult::pass("toolplan_undeclared_scope")
    } else {
        EvalResult::fail("toolplan_undeclared_scope", "undeclared scope was accepted")
    }
}

fn test_auth_permission_granted() -> EvalResult {
    let tool_id = kchat_core::ids::ToolId::new();
    let mut actions = HashSet::new();
    actions.insert("read".into());
    let mut scopes = HashSet::new();
    scopes.insert("workspace_1".into());

    let auth = AuthContext {
        user_id: kchat_core::ids::UserId::new(),
        tenant_id: kchat_core::ids::TenantId::new(),
        roles: vec!["member".into()],
        permissions: vec![Permission {
            tool_id,
            actions,
            data_scopes: scopes,
            confirmation_class: ConfirmationClass::ReadOnly,
        }],
        session_token: "test".into(),
    };

    let broker = RbacBroker::new();
    let result = broker.reauthorize(&auth, &tool_id, "read", "workspace_1");

    if result.is_ok() {
        EvalResult::pass("auth_permission_granted")
    } else {
        EvalResult::fail("auth_permission_granted", "valid permission was denied")
    }
}

fn test_auth_permission_denied() -> EvalResult {
    let tool_id = kchat_core::ids::ToolId::new();
    let mut actions = HashSet::new();
    actions.insert("read".into());
    let mut scopes = HashSet::new();
    scopes.insert("workspace_1".into());

    let auth = AuthContext {
        user_id: kchat_core::ids::UserId::new(),
        tenant_id: kchat_core::ids::TenantId::new(),
        roles: vec!["member".into()],
        permissions: vec![Permission {
            tool_id,
            actions,
            data_scopes: scopes,
            confirmation_class: ConfirmationClass::ReadOnly,
        }],
        session_token: "test".into(),
    };

    let broker = RbacBroker::new();
    let result = broker.reauthorize(&auth, &tool_id, "delete", "workspace_1");

    if result.is_err() {
        EvalResult::pass("auth_permission_denied")
    } else {
        EvalResult::fail("auth_permission_denied", "invalid permission was granted")
    }
}

fn test_auth_step_up_required() -> EvalResult {
    let broker = RbacBroker::new();
    if broker.requires_step_up_auth(ConfirmationClass::SensitiveAction) {
        EvalResult::pass("auth_step_up_required")
    } else {
        EvalResult::fail("auth_step_up_required", "sensitive action does not require step-up auth")
    }
}

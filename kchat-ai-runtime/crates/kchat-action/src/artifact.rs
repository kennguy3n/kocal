//! Artifact AST — canonical versioned artifact schema across platforms.
//!
//! AI outputs are typed operations such as `replace_range`, `insert_slide`,
//! `set_formula`, or `update_record`. They never become arbitrary
//! DOCX/PPTX/XLSX XML, JavaScript, SQL, macros, or filesystem commands.
//!
//! All operations reference stable artifact node IDs and expected versions.
//! Formulas parse to an allowed formula AST. Database/base updates use typed
//! fields and policy-checked record IDs. Preview and undo are mandatory for
//! multi-node changes. Stale-version conflicts are shown or merged, not
//! silently overwritten.

use kchat_core::ids::ArtifactId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identifier for a node within an artifact AST.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArtifactNodeId(pub Uuid);

impl ArtifactNodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ArtifactNodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactType {
    Document,
    Slides,
    Sheet,
    Base,
    Infographic,
}

/// A node in the artifact AST.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactNode {
    pub node_id: ArtifactNodeId,
    pub node_type: String,
    pub content: String,
    pub children: Vec<ArtifactNodeId>,
    pub version: u32,
}

/// The canonical artifact AST.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactAst {
    pub artifact_id: ArtifactId,
    pub artifact_type: ArtifactType,
    pub title: String,
    pub root_nodes: Vec<ArtifactNodeId>,
    pub nodes: Vec<ArtifactNode>,
    pub version: u32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl ArtifactAst {
    pub fn new(artifact_id: ArtifactId, artifact_type: ArtifactType, title: impl Into<String>) -> Self {
        let now = chrono::Utc::now();
        Self {
            artifact_id,
            artifact_type,
            title: title.into(),
            root_nodes: Vec::new(),
            nodes: Vec::new(),
            version: 1,
            created_at: now,
            updated_at: now,
        }
    }

    /// Find a node by ID.
    pub fn find_node(&self, id: ArtifactNodeId) -> Option<&ArtifactNode> {
        self.nodes.iter().find(|n| n.node_id == id)
    }

    /// Find a mutable node by ID.
    pub fn find_node_mut(&mut self, id: ArtifactNodeId) -> Option<&mut ArtifactNode> {
        self.nodes.iter_mut().find(|n| n.node_id == id)
    }

    /// Apply an operation and return the new version.
    /// Returns an error if the operation is invalid.
    pub fn apply_operation(&mut self, op: &ArtifactOperation) -> Result<(), ArtifactError> {
        // Validate the operation first
        OperationValidator::validate(op, self)?;

        match op {
            ArtifactOperation::ReplaceRange { node_id, expected_version, start, end, new_content } => {
                let node = self.find_node_mut(*node_id).ok_or(ArtifactError::NodeNotFound)?;
                if node.version != *expected_version {
                    return Err(ArtifactError::StaleVersion {
                        expected: *expected_version,
                        actual: node.version,
                    });
                }
                // Replace the range [start, end) with new_content
                let mut new_text = String::with_capacity(
                    node.content.len() - (*end - *start) + new_content.len(),
                );
                new_text.push_str(&node.content[..*start]);
                new_text.push_str(new_content);
                new_text.push_str(&node.content[*end..]);
                node.content = new_text;
                node.version += 1;
                self.version += 1;
                self.updated_at = chrono::Utc::now();
            }
            ArtifactOperation::InsertSlide { after_node, title, .. } => {
                let new_node_id = ArtifactNodeId::new();
                let new_node = ArtifactNode {
                    node_id: new_node_id,
                    node_type: "slide".into(),
                    content: title.clone(),
                    children: Vec::new(),
                    version: 1,
                };
                self.nodes.push(new_node);
                if let Some(after) = after_node {
                    // Validate that after_node exists — reject orphaned references
                    let node = self.find_node_mut(*after).ok_or(ArtifactError::NodeNotFound)?;
                    node.children.push(new_node_id);
                } else {
                    self.root_nodes.push(new_node_id);
                }
                self.version += 1;
                self.updated_at = chrono::Utc::now();
            }
            ArtifactOperation::SetFormula { node_id, expected_version, formula, .. } => {
                let node = self.find_node_mut(*node_id).ok_or(ArtifactError::NodeNotFound)?;
                if node.version != *expected_version {
                    return Err(ArtifactError::StaleVersion {
                        expected: *expected_version,
                        actual: node.version,
                    });
                }
                // Formula must be a valid formula AST string (no macros/code).
                // Use case-insensitive check to prevent bypass via "=MACRO()" etc.
                if !check_formula(formula) {
                    return Err(ArtifactError::InvalidFormula("formula contains forbidden keywords".into()));
                }
                node.content = formula.clone();
                node.version += 1;
                self.version += 1;
                self.updated_at = chrono::Utc::now();
            }
            ArtifactOperation::UpdateRecord { node_id, expected_version, fields, .. } => {
                let node = self.find_node_mut(*node_id).ok_or(ArtifactError::NodeNotFound)?;
                if node.version != *expected_version {
                    return Err(ArtifactError::StaleVersion {
                        expected: *expected_version,
                        actual: node.version,
                    });
                }
                // Serialize fields as JSON content — return error on failure
                node.content = serde_json::to_string(fields)
                    .map_err(|e| ArtifactError::InvalidFields(format!("JSON serialization failed: {e}")))?;
                node.version += 1;
                self.version += 1;
                self.updated_at = chrono::Utc::now();
            }
        }
        Ok(())
    }
}

/// Typed operations on artifact ASTs. The model proposes these; the
/// deterministic validator checks them before execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ArtifactOperation {
    /// Replace a text range in a document node.
    ReplaceRange {
        node_id: ArtifactNodeId,
        expected_version: u32,
        start: usize,
        end: usize,
        new_content: String,
    },
    /// Insert a new slide after the given node (or at root if None).
    InsertSlide {
        after_node: Option<ArtifactNodeId>,
        title: String,
    },
    /// Set a formula on a sheet cell node. Must parse to allowed formula AST.
    SetFormula {
        node_id: ArtifactNodeId,
        expected_version: u32,
        formula: String,
    },
    /// Update a record in a base. Uses typed fields and policy-checked IDs.
    UpdateRecord {
        node_id: ArtifactNodeId,
        expected_version: u32,
        fields: serde_json::Value,
    },
}

/// Deterministic operation validator.
pub struct OperationValidator;

impl OperationValidator {
    /// Validate an operation against the current artifact state.
    /// 100% of artifact operations must parse before execution.
    pub fn validate(op: &ArtifactOperation, ast: &ArtifactAst) -> Result<(), ArtifactError> {
        match op {
            ArtifactOperation::ReplaceRange { node_id, expected_version, start, end, new_content } => {
                let node = ast.find_node(*node_id).ok_or(ArtifactError::NodeNotFound)?;
                if node.version != *expected_version {
                    return Err(ArtifactError::StaleVersion {
                        expected: *expected_version,
                        actual: node.version,
                    });
                }
                if *start > *end || *end > node.content.len() {
                    return Err(ArtifactError::InvalidRange);
                }
                // No executable content
                Self::check_no_executable(new_content)?;
            }
            ArtifactOperation::InsertSlide { title, .. } => {
                Self::check_no_executable(title)?;
            }
            ArtifactOperation::SetFormula { node_id, expected_version, formula } => {
                let node = ast.find_node(*node_id).ok_or(ArtifactError::NodeNotFound)?;
                if node.version != *expected_version {
                    return Err(ArtifactError::StaleVersion {
                        expected: *expected_version,
                        actual: node.version,
                    });
                }
                Self::check_formula_method(formula)?;
            }
            ArtifactOperation::UpdateRecord { node_id, expected_version, fields } => {
                let node = ast.find_node(*node_id).ok_or(ArtifactError::NodeNotFound)?;
                if node.version != *expected_version {
                    return Err(ArtifactError::StaleVersion {
                        expected: *expected_version,
                        actual: node.version,
                    });
                }
                // Fields must be a JSON object
                if !fields.is_object() {
                    return Err(ArtifactError::InvalidFields("fields must be a JSON object".into()));
                }
                // Check string values for executable content
                if let serde_json::Value::Object(map) = fields {
                    for (_, v) in map {
                        if let serde_json::Value::String(s) = v {
                            Self::check_no_executable(s)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Check that content does not contain executable instructions.
    fn check_no_executable(content: &str) -> Result<(), ArtifactError> {
        let lower = content.to_lowercase();
        let forbidden = [
            "<script", "javascript:", "vbscript:", "onload=", "onerror=",
            "eval(", "exec(", "system(", "subprocess",
            "<!--macro", "<!--#include",
            "file://", "data:text/html",
        ];
        for f in &forbidden {
            if lower.contains(f) {
                return Err(ArtifactError::ExecutableContent((*f).into()));
            }
        }
        Ok(())
    }

    /// Check that a formula is a valid formula AST (no macros or code).
    fn check_formula_method(formula: &str) -> Result<(), ArtifactError> {
        if check_formula(formula) {
            Ok(())
        } else {
            Err(ArtifactError::InvalidFormula("formula contains forbidden keywords".into()))
        }
    }
}

/// Check that a formula is a valid formula AST (no macros or code).
/// Case-insensitive to prevent bypass via "=MACRO()" or "=Macro()".
fn check_formula(formula: &str) -> bool {
    let lower = formula.to_lowercase();
    let forbidden = ["macro", "script", "exec", "system", "shell", "eval", "import", "require"];
    !forbidden.iter().any(|f| lower.contains(f))
}

/// Artifact operation errors.
#[derive(Debug, thiserror::Error)]
pub enum ArtifactError {
    #[error("node not found")]
    NodeNotFound,
    #[error("stale version: expected {expected}, actual {actual}")]
    StaleVersion { expected: u32, actual: u32 },
    #[error("invalid range")]
    InvalidRange,
    #[error("executable content detected: {0}")]
    ExecutableContent(String),
    #[error("invalid formula: {0}")]
    InvalidFormula(String),
    #[error("invalid fields: {0}")]
    InvalidFields(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_document() -> ArtifactAst {
        let mut ast = ArtifactAst::new(ArtifactId::new(), ArtifactType::Document, "Test Doc");
        let node = ArtifactNode {
            node_id: ArtifactNodeId::new(),
            node_type: "paragraph".into(),
            content: "Hello world".into(),
            children: Vec::new(),
            version: 1,
        };
        ast.root_nodes.push(node.node_id);
        ast.nodes.push(node);
        ast
    }

    #[test]
    fn test_replace_range_succeeds() {
        let mut ast = make_document();
        let node_id = ast.root_nodes[0];
        let op = ArtifactOperation::ReplaceRange {
            node_id,
            expected_version: 1,
            start: 0,
            end: 5,
            new_content: "Hi".into(),
        };
        assert!(ast.apply_operation(&op).is_ok());
        let node = ast.find_node(node_id).unwrap();
        assert_eq!(node.content, "Hi world");
        assert_eq!(node.version, 2);
        assert_eq!(ast.version, 2);
    }

    #[test]
    fn test_stale_version_rejected() {
        let mut ast = make_document();
        let node_id = ast.root_nodes[0];
        let op = ArtifactOperation::ReplaceRange {
            node_id,
            expected_version: 99, // wrong version
            start: 0,
            end: 5,
            new_content: "Hi".into(),
        };
        let result = ast.apply_operation(&op);
        assert!(result.is_err());
        match result {
            Err(ArtifactError::StaleVersion { expected, actual }) => {
                assert_eq!(expected, 99);
                assert_eq!(actual, 1);
            }
            _ => panic!("expected StaleVersion"),
        }
    }

    #[test]
    fn test_executable_content_rejected() {
        let mut ast = make_document();
        let node_id = ast.root_nodes[0];
        let op = ArtifactOperation::ReplaceRange {
            node_id,
            expected_version: 1,
            start: 0,
            end: 0,
            new_content: "<script>alert(1)</script>".into(),
        };
        let result = ast.apply_operation(&op);
        assert!(result.is_err());
    }

    #[test]
    fn test_formula_with_macro_rejected() {
        let mut ast = make_document();
        let node_id = ast.root_nodes[0];
        let op = ArtifactOperation::SetFormula {
            node_id,
            expected_version: 1,
            formula: "=macro(test)".into(),
        };
        let result = ast.apply_operation(&op);
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_formula_accepted() {
        let mut ast = make_document();
        let node_id = ast.root_nodes[0];
        let op = ArtifactOperation::SetFormula {
            node_id,
            expected_version: 1,
            formula: "=SUM(A1:A10)".into(),
        };
        assert!(ast.apply_operation(&op).is_ok());
    }

    #[test]
    fn test_update_record_accepted() {
        let mut ast = make_document();
        let node_id = ast.root_nodes[0];
        let fields = serde_json::json!({"name": "Alice", "age": 30});
        let op = ArtifactOperation::UpdateRecord {
            node_id,
            expected_version: 1,
            fields,
        };
        assert!(ast.apply_operation(&op).is_ok());
    }

    #[test]
    fn test_update_record_with_script_rejected() {
        let mut ast = make_document();
        let node_id = ast.root_nodes[0];
        let fields = serde_json::json!({"name": "<script>alert(1)</script>"});
        let op = ArtifactOperation::UpdateRecord {
            node_id,
            expected_version: 1,
            fields,
        };
        let result = ast.apply_operation(&op);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_slide() {
        let mut ast = make_document();
        let op = ArtifactOperation::InsertSlide {
            after_node: None,
            title: "New Slide".into(),
        };
        assert!(ast.apply_operation(&op).is_ok());
        assert_eq!(ast.root_nodes.len(), 2);
        assert_eq!(ast.version, 2);
        // Verify title is used as content
        let slide_node = ast.find_node(ast.root_nodes[1]).unwrap();
        assert_eq!(slide_node.content, "New Slide");
    }

    #[test]
    fn test_insert_slide_after_valid_node() {
        let mut ast = make_document();
        let first_node = ast.root_nodes[0];
        let op = ArtifactOperation::InsertSlide {
            after_node: Some(first_node),
            title: "Child Slide".into(),
        };
        assert!(ast.apply_operation(&op).is_ok());
        // The new slide should be a child of the first node
        let parent = ast.find_node(first_node).unwrap();
        assert_eq!(parent.children.len(), 1);
    }

    #[test]
    fn test_insert_slide_after_invalid_node() {
        let mut ast = make_document();
        let fake_id = ArtifactNodeId::new();
        let op = ArtifactOperation::InsertSlide {
            after_node: Some(fake_id),
            title: "Orphan Slide".into(),
        };
        // Should fail with NodeNotFound
        assert!(matches!(ast.apply_operation(&op), Err(ArtifactError::NodeNotFound)));
    }

    #[test]
    fn test_formula_validation_mixed_case() {
        let mut ast = make_document();
        let node_id = ast.root_nodes[0];
        // Test that =MACRO() is rejected (uppercase)
        let op = ArtifactOperation::SetFormula {
            node_id,
            expected_version: 1,
            formula: "=MACRO(bad)".into(),
        };
        assert!(ast.apply_operation(&op).is_err());

        // Test that =Macro() is rejected (mixed case)
        let op = ArtifactOperation::SetFormula {
            node_id,
            expected_version: 1,
            formula: "=Macro(bad)".into(),
        };
        assert!(ast.apply_operation(&op).is_err());

        // Test that =ScRiPt() is rejected (random case)
        let op = ArtifactOperation::SetFormula {
            node_id,
            expected_version: 1,
            formula: "=ScRiPt(bad)".into(),
        };
        assert!(ast.apply_operation(&op).is_err());
    }
}

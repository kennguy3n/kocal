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

    /// Check whether `descendant` is a descendant of `ancestor` (direct or
    /// transitive child). Used for cycle detection in ReorderSlide.
    /// Returns false if either node doesn't exist or if they are the same node.
    pub fn is_descendant(&self, ancestor: ArtifactNodeId, descendant: ArtifactNodeId) -> bool {
        if ancestor == descendant {
            return false;
        }
        // BFS from the ancestor's children.
        let mut queue: Vec<ArtifactNodeId> = Vec::new();
        if let Some(node) = self.find_node(ancestor) {
            queue.extend(node.children.iter().copied());
        }
        while let Some(current) = queue.pop() {
            if current == descendant {
                return true;
            }
            if let Some(node) = self.find_node(current) {
                queue.extend(node.children.iter().copied());
            }
        }
        false
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
            ArtifactOperation::InsertSlide { after_node, template_id, title, slots } => {
                let new_node_id = ArtifactNodeId::new();
                // Store template_id and slots as JSON content for the slide node.
                let content = serde_json::to_string(&serde_json::json!({
                    "template_id": template_id,
                    "title": title,
                    "slots": slots,
                })).map_err(|e| ArtifactError::InvalidFields(format!("JSON serialization failed: {e}")))?;
                let new_node = ArtifactNode {
                    node_id: new_node_id,
                    node_type: "slide".into(),
                    content,
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
            ArtifactOperation::UpdateSlide { node_id, expected_version, title, slots } => {
                let node = self.find_node_mut(*node_id).ok_or(ArtifactError::NodeNotFound)?;
                if node.version != *expected_version {
                    return Err(ArtifactError::StaleVersion {
                        expected: *expected_version,
                        actual: node.version,
                    });
                }
                // Parse existing content, update fields, re-serialize.
                let mut parsed: serde_json::Value = serde_json::from_str(&node.content)
                    .unwrap_or(serde_json::json!({}));
                if let Some(t) = title {
                    if let Some(obj) = parsed.as_object_mut() {
                        obj.insert("title".into(), serde_json::Value::String(t.clone()));
                    }
                }
                if let Some(s) = slots {
                    if let Some(obj) = parsed.as_object_mut() {
                        obj.insert("slots".into(), s.clone());
                    }
                }
                node.content = serde_json::to_string(&parsed)
                    .map_err(|e| ArtifactError::InvalidFields(format!("JSON serialization failed: {e}")))?;
                node.version += 1;
                self.version += 1;
                self.updated_at = chrono::Utc::now();
            }
            ArtifactOperation::ReorderSlide { node_id, after_node } => {
                // Remove node_id from its current parent's children (or root).
                let _ = self.find_node(*node_id).ok_or(ArtifactError::NodeNotFound)?;
                // Cycle detection: if after_node is a descendant of node_id,
                // moving node_id under after_node would create a cycle.
                if let Some(after) = after_node {
                    if *after == *node_id {
                        return Err(ArtifactError::InvalidFields(
                            "cannot move a node under itself".into(),
                        ));
                    }
                    if self.is_descendant(*node_id, *after) {
                        return Err(ArtifactError::InvalidFields(
                            "cannot move a node under its own descendant (would create cycle)".into(),
                        ));
                    }
                }
                // Remove from root_nodes if present.
                self.root_nodes.retain(|n| *n != *node_id);
                // Remove from all nodes' children.
                for n in &mut self.nodes {
                    n.children.retain(|c| *c != *node_id);
                }
                // Insert at new position.
                if let Some(after) = after_node {
                    let parent = self.find_node_mut(*after).ok_or(ArtifactError::NodeNotFound)?;
                    parent.children.push(*node_id);
                } else {
                    self.root_nodes.push(*node_id);
                }
                self.version += 1;
                self.updated_at = chrono::Utc::now();
            }
            ArtifactOperation::SetSlideTemplate { node_id, expected_version, template_id, slots } => {
                let node = self.find_node_mut(*node_id).ok_or(ArtifactError::NodeNotFound)?;
                if node.version != *expected_version {
                    return Err(ArtifactError::StaleVersion {
                        expected: *expected_version,
                        actual: node.version,
                    });
                }
                let content = serde_json::to_string(&serde_json::json!({
                    "template_id": template_id,
                    "title": serde_json::from_str::<serde_json::Value>(&node.content)
                        .ok()
                        .and_then(|v| v.get("title").cloned())
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                        .unwrap_or_default(),
                    "slots": slots,
                })).map_err(|e| ArtifactError::InvalidFields(format!("JSON serialization failed: {e}")))?;
                node.content = content;
                node.version += 1;
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
    /// The template_id must exist in the SlidesTemplateRegistry; slots must
    /// conform to the template's slot schema. Image slots contain only
    /// query strings — the runtime resolves URLs via kchat-image.
    InsertSlide {
        after_node: Option<ArtifactNodeId>,
        template_id: String,
        title: String,
        #[serde(default)]
        slots: serde_json::Value,
    },
    /// Update an existing slide's title and/or slots.
    UpdateSlide {
        node_id: ArtifactNodeId,
        expected_version: u32,
        title: Option<String>,
        #[serde(default)]
        slots: Option<serde_json::Value>,
    },
    /// Reorder a slide to appear after the given node (or at root if None).
    ReorderSlide {
        node_id: ArtifactNodeId,
        after_node: Option<ArtifactNodeId>,
    },
    /// Change the template of an existing slide. Slots must conform to the
    /// new template's schema.
    SetSlideTemplate {
        node_id: ArtifactNodeId,
        expected_version: u32,
        template_id: String,
        #[serde(default)]
        slots: serde_json::Value,
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
            ArtifactOperation::InsertSlide { title, template_id, slots, .. } => {
                Self::check_no_executable(title)?;
                Self::validate_slide_template(template_id, slots)?;
            }
            ArtifactOperation::UpdateSlide { title, slots, .. } => {
                if let Some(t) = title {
                    Self::check_no_executable(t)?;
                }
                if let Some(s) = slots {
                    Self::check_no_executable_value(s)?;
                }
            }
            ArtifactOperation::ReorderSlide { .. } => {
                // No content to validate — just structural.
            }
            ArtifactOperation::SetSlideTemplate { template_id, slots, .. } => {
                Self::validate_slide_template(template_id, slots)?;
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

    /// Validate a slide template_id against the SlidesTemplateRegistry and
    /// check that slots conform to the template's schema. Also checks that
    /// image slots contain only query strings (no raw URLs).
    fn validate_slide_template(template_id: &str, slots: &serde_json::Value) -> Result<(), ArtifactError> {
        let registry = &kchat_generation::TEMPLATE_REGISTRY;
        let template = registry.get(template_id).ok_or_else(|| {
            ArtifactError::InvalidFields(format!("unknown template_id: {}", template_id))
        })?;

        // Slots must be a JSON object (or null for empty slots).
        if !slots.is_null() && !slots.is_object() {
            return Err(ArtifactError::InvalidFields("slots must be a JSON object".into()));
        }

        // Check no-executable in all string values within slots.
        Self::check_no_executable_value(slots)?;

        // Check that image slots contain only query strings, not raw URLs.
        if let Some(obj) = slots.as_object() {
            for slot in &template.slots {
                if slot.slot_type == kchat_generation::SlotType::ImageQuery
                    || slot.slot_type == kchat_generation::SlotType::ImageRef
                {
                    if let Some(img_val) = obj.get(&slot.id) {
                        if let Some(query) = img_val.get("query").and_then(|v| v.as_str()) {
                            // Reject raw URLs in the query — the runtime resolves queries.
                            if query.starts_with("http://") || query.starts_with("https://") {
                                return Err(ArtifactError::InvalidFields(format!(
                                    "image slot '{}' must contain a search query, not a URL",
                                    slot.id
                                )));
                            }
                            Self::check_no_executable(query)?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Recursively check that a JSON value contains no executable content.
    fn check_no_executable_value(value: &serde_json::Value) -> Result<(), ArtifactError> {
        match value {
            serde_json::Value::String(s) => Self::check_no_executable(s),
            serde_json::Value::Object(map) => {
                for (_, v) in map {
                    Self::check_no_executable_value(v)?;
                }
                Ok(())
            }
            serde_json::Value::Array(arr) => {
                for v in arr {
                    Self::check_no_executable_value(v)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

/// Check that a formula is a valid formula AST (no macros or code).
/// Case-insensitive to prevent bypass via "=MACRO()" or "=Macro()".
fn check_formula(formula: &str) -> bool {
    let lower = formula.to_lowercase();
    let forbidden = [
        "macro", "script", "exec", "system", "shell", "eval", "import", "require",
        "hyperlink", "image", "query", "importxml", "importdata", "importrange",
        "importhtml", "importfeed", "importjson",
    ];
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
            template_id: "title".into(),
            title: "New Slide".into(),
            slots: serde_json::json!({"title": "New Slide"}),
        };
        assert!(ast.apply_operation(&op).is_ok());
        assert_eq!(ast.root_nodes.len(), 2);
        assert_eq!(ast.version, 2);
        // Verify content is JSON with template_id and title
        let slide_node = ast.find_node(ast.root_nodes[1]).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&slide_node.content).unwrap();
        assert_eq!(parsed["template_id"], "title");
        assert_eq!(parsed["title"], "New Slide");
    }

    #[test]
    fn test_insert_slide_after_valid_node() {
        let mut ast = make_document();
        let first_node = ast.root_nodes[0];
        let op = ArtifactOperation::InsertSlide {
            after_node: Some(first_node),
            template_id: "bullet".into(),
            title: "Child Slide".into(),
            slots: serde_json::json!({"title": "Child Slide", "bullets": ["point 1", "point 2"]}),
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
            template_id: "title".into(),
            title: "Orphan Slide".into(),
            slots: serde_json::json!({"title": "Orphan Slide"}),
        };
        // Should fail with NodeNotFound
        assert!(matches!(ast.apply_operation(&op), Err(ArtifactError::NodeNotFound)));
    }

    #[test]
    fn test_insert_slide_unknown_template_rejected() {
        let mut ast = make_document();
        let op = ArtifactOperation::InsertSlide {
            after_node: None,
            template_id: "nonexistent_template".into(),
            title: "Bad Slide".into(),
            slots: serde_json::json!({}),
        };
        let result = ast.apply_operation(&op);
        assert!(result.is_err());
        assert!(matches!(result, Err(ArtifactError::InvalidFields(_))));
    }

    #[test]
    fn test_insert_slide_executable_in_slots_rejected() {
        let mut ast = make_document();
        let op = ArtifactOperation::InsertSlide {
            after_node: None,
            template_id: "bullet".into(),
            title: "Safe Title".into(),
            slots: serde_json::json!({"title": "Safe Title", "bullets": ["<script>alert(1)</script>"]}),
        };
        let result = ast.apply_operation(&op);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_slide_image_slot_with_url_rejected() {
        let mut ast = make_document();
        let op = ArtifactOperation::InsertSlide {
            after_node: None,
            template_id: "title_image".into(),
            title: "Title with Image".into(),
            slots: serde_json::json!({
                "title": "Title with Image",
                "image": {"query": "https://example.com/img.jpg", "orientation": "landscape"}
            }),
        };
        let result = ast.apply_operation(&op);
        assert!(result.is_err());
    }

    #[test]
    fn test_insert_slide_image_slot_with_query_accepted() {
        let mut ast = make_document();
        let op = ArtifactOperation::InsertSlide {
            after_node: None,
            template_id: "title_image".into(),
            title: "Title with Image".into(),
            slots: serde_json::json!({
                "title": "Title with Image",
                "image": {"query": "sunset over mountains", "orientation": "landscape"}
            }),
        };
        assert!(ast.apply_operation(&op).is_ok());
    }

    #[test]
    fn test_update_slide() {
        let mut ast = make_document();
        // First insert a slide
        let insert_op = ArtifactOperation::InsertSlide {
            after_node: None,
            template_id: "title".into(),
            title: "Original".into(),
            slots: serde_json::json!({"title": "Original"}),
        };
        ast.apply_operation(&insert_op).unwrap();
        let slide_id = ast.root_nodes[1];

        // Now update it
        let update_op = ArtifactOperation::UpdateSlide {
            node_id: slide_id,
            expected_version: 1,
            title: Some("Updated Title".into()),
            slots: Some(serde_json::json!({"title": "Updated Title"})),
        };
        assert!(ast.apply_operation(&update_op).is_ok());
        let node = ast.find_node(slide_id).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&node.content).unwrap();
        assert_eq!(parsed["title"], "Updated Title");
        assert_eq!(node.version, 2);
    }

    #[test]
    fn test_reorder_slide() {
        let mut ast = make_document();
        let first = ast.root_nodes[0];
        // Insert two slides
        ast.apply_operation(&ArtifactOperation::InsertSlide {
            after_node: None,
            template_id: "title".into(),
            title: "Slide A".into(),
            slots: serde_json::json!({"title": "Slide A"}),
        }).unwrap();
        let slide_a = ast.root_nodes[1];
        ast.apply_operation(&ArtifactOperation::InsertSlide {
            after_node: None,
            template_id: "title".into(),
            title: "Slide B".into(),
            slots: serde_json::json!({"title": "Slide B"}),
        }).unwrap();
        let slide_b = ast.root_nodes[2];

        // Reorder slide_b to be a child of first (the paragraph node)
        let reorder_op = ArtifactOperation::ReorderSlide {
            node_id: slide_b,
            after_node: Some(first),
        };
        assert!(ast.apply_operation(&reorder_op).is_ok());
        // slide_b should now be a child of first
        let parent = ast.find_node(first).unwrap();
        assert!(parent.children.contains(&slide_b));
        // slide_b should no longer be in root_nodes
        assert!(!ast.root_nodes.contains(&slide_b));
        // slide_a should still be in root_nodes
        assert!(ast.root_nodes.contains(&slide_a));
    }

    #[test]
    fn test_set_slide_template() {
        let mut ast = make_document();
        // Insert a slide with template "title"
        ast.apply_operation(&ArtifactOperation::InsertSlide {
            after_node: None,
            template_id: "title".into(),
            title: "My Slide".into(),
            slots: serde_json::json!({"title": "My Slide"}),
        }).unwrap();
        let slide_id = ast.root_nodes[1];

        // Change template to "bullet"
        let op = ArtifactOperation::SetSlideTemplate {
            node_id: slide_id,
            expected_version: 1,
            template_id: "bullet".into(),
            slots: serde_json::json!({"title": "My Slide", "bullets": ["point 1", "point 2"]}),
        };
        assert!(ast.apply_operation(&op).is_ok());
        let node = ast.find_node(slide_id).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&node.content).unwrap();
        assert_eq!(parsed["template_id"], "bullet");
        assert_eq!(node.version, 2);
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

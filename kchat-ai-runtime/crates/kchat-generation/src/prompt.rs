//! Prompt templates — versioned and hashed for provenance.
//!
//! Templates are parameterized with typed slots. The template hash is
//! recorded in provenance bundles for reproducibility.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use uuid::Uuid;

/// Stable identifier for a prompt template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemplateId(pub Uuid);

impl TemplateId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for TemplateId {
    fn default() -> Self {
        Self::new()
    }
}

/// A versioned prompt template with typed slots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub id: TemplateId,
    pub name: String,
    pub version: String,
    /// The template string with {{slot}} placeholders
    pub template: String,
    /// Slot names and their descriptions
    pub slots: Vec<String>,
    /// SHA-256 hash of the template content (for provenance)
    pub content_hash: String,
    /// Task capabilities this template supports
    pub task_capabilities: Vec<String>,
    /// Minimum tier required
    pub min_tier: String,
}

impl PromptTemplate {
    /// Create a new template, computing the content hash.
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        template: impl Into<String>,
        slots: Vec<String>,
    ) -> Self {
        let template_str = template.into();
        let hash = Self::compute_hash(&template_str);
        Self {
            id: TemplateId::new(),
            name: name.into(),
            version: version.into(),
            template: template_str,
            slots,
            content_hash: hash,
            task_capabilities: Vec::new(),
            min_tier: "medium".into(),
        }
    }

    /// Compute the SHA-256 hash of the template content.
    pub fn compute_hash(template: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(template.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Render the template with the given slot values.
    pub fn render(&self, slots: &HashMap<String, String>) -> Result<String, TemplateError> {
        let mut result = self.template.clone();

        // Check all required slots are provided
        for slot in &self.slots {
            if !slots.contains_key(slot) {
                return Err(TemplateError::MissingSlot(slot.clone()));
            }
        }

        // Replace {{slot}} placeholders.
        // Slot values are sanitized to prevent template injection: any `{{` or `}}`
        // in values is escaped so they cannot be interpreted as template variables.
        for (key, value) in slots {
            let placeholder = format!("{{{{{}}}}}", key);
            // Escape template syntax in slot values to prevent injection
            let sanitized = value.replace("{{", "<<").replace("}}", ">>");
            result = result.replace(&placeholder, &sanitized);
        }

        Ok(result)
    }
}

/// Registry of prompt templates.
pub struct PromptTemplateRegistry {
    templates: HashMap<TemplateId, PromptTemplate>,
    by_name: HashMap<String, TemplateId>,
}

impl PromptTemplateRegistry {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            by_name: HashMap::new(),
        }
    }

    pub fn register(&mut self, template: PromptTemplate) -> TemplateId {
        let id = template.id;
        self.by_name.insert(template.name.clone(), id);
        self.templates.insert(id, template);
        id
    }

    pub fn get(&self, id: &TemplateId) -> Option<&PromptTemplate> {
        self.templates.get(id)
    }

    pub fn get_by_name(&self, name: &str) -> Option<&PromptTemplate> {
        self.by_name.get(name).and_then(|id| self.templates.get(id))
    }

    /// Render a template by ID with the given slots.
    pub fn render(
        &self,
        id: &TemplateId,
        slots: &HashMap<String, String>,
    ) -> Result<String, TemplateError> {
        let template = self.get(id).ok_or(TemplateError::TemplateNotFound)?;
        template.render(slots)
    }
}

impl Default for PromptTemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Template errors.
#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    #[error("missing slot: {0}")]
    MissingSlot(String),

    #[error("template not found")]
    TemplateNotFound,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_render() {
        let template = PromptTemplate::new(
            "rewrite",
            "1.0.0",
            "Rewrite the following text: {{input}}\nStyle: {{style}}",
            vec!["input".into(), "style".into()],
        );

        let mut slots = HashMap::new();
        slots.insert("input".into(), "Hello world".into());
        slots.insert("style".into(), "formal".into());

        let rendered = template.render(&slots).unwrap();
        assert!(rendered.contains("Hello world"));
        assert!(rendered.contains("formal"));
    }

    #[test]
    fn test_missing_slot_error() {
        let template = PromptTemplate::new(
            "test",
            "1.0.0",
            "{{a}} {{b}}",
            vec!["a".into(), "b".into()],
        );

        let mut slots = HashMap::new();
        slots.insert("a".into(), "hello".into());
        // Missing "b"

        let result = template.render(&slots);
        assert!(result.is_err());
    }

    #[test]
    fn test_content_hash_deterministic() {
        let t1 = PromptTemplate::new("test", "1.0", "Hello {{name}}", vec!["name".into()]);
        let t2 = PromptTemplate::new("test", "1.0", "Hello {{name}}", vec!["name".into()]);
        assert_eq!(t1.content_hash, t2.content_hash);
    }

    #[test]
    fn test_registry() {
        let mut registry = PromptTemplateRegistry::new();
        let template = PromptTemplate::new(
            "summarize",
            "1.0.0",
            "Summarize: {{input}}",
            vec!["input".into()],
        );
        let id = registry.register(template);

        let mut slots = HashMap::new();
        slots.insert("input".into(), "Long text...".into());

        let rendered = registry.render(&id, &slots).unwrap();
        assert!(rendered.contains("Long text..."));
    }
}

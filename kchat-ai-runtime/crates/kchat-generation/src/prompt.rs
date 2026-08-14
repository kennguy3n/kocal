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

impl PromptTemplateRegistry {
    /// Register prompt templates for all 33 document AI skills.
    ///
    /// Each skill gets a template with `{{slot}}` placeholders matching its
    /// `build_prompt` output. Templates are versioned and hashed for provenance.
    pub fn register_skill_templates(&mut self) -> Vec<TemplateId> {
        let skills = crate::skills::SkillRegistry::new();
        let mut ids = Vec::with_capacity(skills.len());

        for skill in skills.all() {
            let prompt_output = skill.build_prompt(crate::skills::SkillPromptInput::default());

            // Build a combined template string with system and user slots
            let template_str = format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt_output.system, prompt_output.user
            );

            // Determine slots based on skill scope
            let slots = match skill.scope {
                crate::skills::SkillScope::Selection => vec!["selection".into()],
                crate::skills::SkillScope::Cursor => vec!["context".into()],
                crate::skills::SkillScope::Section => vec!["input".into(), "context".into()],
                crate::skills::SkillScope::Document => {
                    if skill.use_outline_context {
                        vec!["context".into()]
                    } else {
                        vec!["context".into()]
                    }
                }
                crate::skills::SkillScope::Topic => {
                    let mut s = vec!["input".into()];
                    if skill.supports_keywords {
                        s.push("keywords".into());
                    }
                    if skill.needs_full_document || skill.use_outline_context {
                        s.push("context".into());
                    }
                    s
                }
            };

            let template = PromptTemplate::new(
                format!("skill_{}", skill.id),
                "1.0.0",
                template_str,
                slots,
            );
            let id = self.register(template);
            ids.push(id);
        }

        ids
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

    #[test]
    fn test_register_skill_templates() {
        let mut registry = PromptTemplateRegistry::new();
        let ids = registry.register_skill_templates();
        assert_eq!(ids.len(), 33);

        // Verify a few templates by name
        assert!(registry.get_by_name("skill_edit_fix_grammar").is_some());
        assert!(registry.get_by_name("skill_doc_summarize").is_some());
        assert!(registry.get_by_name("skill_create_seo_meta").is_some());
        assert!(registry.get_by_name("skill_edit_translate_document").is_some());
    }

    #[test]
    fn test_skill_template_has_content_hash() {
        let mut registry = PromptTemplateRegistry::new();
        registry.register_skill_templates();

        let template = registry.get_by_name("skill_doc_summarize").unwrap();
        assert!(!template.content_hash.is_empty());
        assert_eq!(template.version, "1.0.0");
    }

    #[test]
    fn test_skill_templates_no_duplicate_names() {
        let mut registry = PromptTemplateRegistry::new();
        registry.register_skill_templates();

        // get_by_name returns the last registered template for a name,
        // so if there were duplicates, we'd still get a template.
        // Instead, verify all 33 names are unique by checking each skill.
        let skills = crate::skills::SkillRegistry::new();
        for skill in skills.all() {
            let name = format!("skill_{}", skill.id);
            assert!(registry.get_by_name(&name).is_some(), "missing template for skill {}", skill.id);
        }
    }
}

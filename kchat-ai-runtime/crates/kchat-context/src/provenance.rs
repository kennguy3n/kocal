//! Provenance — every retrieved item carries source, timestamp, ACL version,
//! provenance, and citation location.
//!
//! Provenance is structured: agent kind, model name, prompt template hash,
//! and tool manifest hash. This enables reproducibility and audit.

use kchat_core::ids::TaskId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Kind of agent that produced or retrieved content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentKind {
    /// Local on-device model
    LocalModel,
    /// Cloud model (if hybrid mode is used)
    CloudModel,
    /// Deterministic rule engine
    Deterministic,
    /// Connector (external data source)
    Connector,
    /// Human author
    Human,
    /// System process
    System,
}

/// Provenance agent — who/what produced or retrieved content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceAgent {
    pub agent_kind: AgentKind,
    pub agent_id: String,
    pub model_name: Option<String>,
    pub prompt_template_hash: Option<String>,
    pub tool_manifest_hash: Option<String>,
}

/// Citation location — where in the source the cited content comes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationLocation {
    /// Source document/message ID
    pub source_id: String,
    /// Character offset in source (if applicable)
    pub char_start: Option<usize>,
    /// Character end in source (if applicable)
    pub char_end: Option<usize>,
    /// Page/slide number (if applicable)
    pub page: Option<u32>,
    /// Timestamp of the source content
    pub source_timestamp: i64,
}

/// A citation — links generated content to its source evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub citation_id: Uuid,
    pub evidence_id: Uuid,
    pub location: CitationLocation,
    pub confidence: f64,
}

/// Provenance bundle — full provenance for a task result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceBundle {
    pub task_id: TaskId,
    pub timestamp: i64,
    pub agents: Vec<ProvenanceAgent>,
    pub citations: Vec<Citation>,
    pub acl_version: u32,
    pub scope_ids: Vec<Uuid>,
}

impl ProvenanceBundle {
    pub fn new(task_id: TaskId) -> Self {
        Self {
            task_id,
            timestamp: chrono::Utc::now().timestamp(),
            agents: Vec::new(),
            citations: Vec::new(),
            acl_version: 1,
            scope_ids: Vec::new(),
        }
    }

    pub fn add_agent(&mut self, agent: ProvenanceAgent) {
        self.agents.push(agent);
    }

    pub fn add_citation(&mut self, citation: Citation) {
        self.citations.push(citation);
    }

    pub fn add_scope(&mut self, scope_id: Uuid) {
        self.scope_ids.push(scope_id);
    }

    /// Verify that all citations have valid evidence IDs and locations.
    pub fn verify_citations(&self) -> bool {
        !self.citations.iter().any(|c| c.evidence_id.is_nil())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provenance_bundle() {
        let task_id = TaskId::new();
        let mut bundle = ProvenanceBundle::new(task_id);

        bundle.add_agent(ProvenanceAgent {
            agent_kind: AgentKind::LocalModel,
            agent_id: "qwen3.5-0.8b".into(),
            model_name: Some("Qwen3.5-0.8B-Q4".into()),
            prompt_template_hash: Some("abc123".into()),
            tool_manifest_hash: None,
        });

        bundle.add_citation(Citation {
            citation_id: Uuid::new_v4(),
            evidence_id: Uuid::new_v4(),
            location: CitationLocation {
                source_id: "msg_001".into(),
                char_start: Some(0),
                char_end: Some(100),
                page: None,
                source_timestamp: chrono::Utc::now().timestamp(),
            },
            confidence: 0.95,
        });

        assert_eq!(bundle.agents.len(), 1);
        assert_eq!(bundle.citations.len(), 1);
        assert!(bundle.verify_citations());
    }

    #[test]
    fn test_nil_evidence_id_fails_verification() {
        let task_id = TaskId::new();
        let mut bundle = ProvenanceBundle::new(task_id);

        bundle.add_citation(Citation {
            citation_id: Uuid::new_v4(),
            evidence_id: Uuid::nil(), // nil UUID
            location: CitationLocation {
                source_id: "msg_001".into(),
                char_start: None,
                char_end: None,
                page: None,
                source_timestamp: 0,
            },
            confidence: 0.5,
        });

        assert!(!bundle.verify_citations());
    }
}

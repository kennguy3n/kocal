//! Intent router — maps a natural-language request to the best skill,
//! its LoRA adapter family, and grammar constraint.
//!
//! With an embedding provider: embeds the request once, compares against
//! cached per-skill description vectors (cosine similarity). Without one:
//! deterministic keyword/label matching — still useful, truthfully degraded.

use kchat_context::embeddings::{cosine_similarity, EmbeddingManager};
use kchat_generation::skills::{SkillDef, SkillRegistry};
use parking_lot::Mutex;
use std::sync::Arc;

/// A routing suggestion: skill + confidence score in `[0, 1]`.
#[derive(Debug, Clone)]
pub struct RouteSuggestion {
    pub skill_id: String,
    pub score: f64,
    /// How the score was produced — `embedding` or `keyword` (truthful
    /// reporting; callers can show lower confidence for keyword matches).
    pub method: &'static str,
}

type SkillVecs = Vec<(String, Vec<f32>)>;

/// Routes natural-language requests to skills.
pub struct SkillRouter {
    skills: Arc<SkillRegistry>,
    embeddings: Option<Arc<EmbeddingManager>>,
    /// Cached skill-description vectors, computed once per embedder attach.
    skill_vecs: Mutex<Option<SkillVecs>>,
}

impl SkillRouter {
    pub fn new(skills: Arc<SkillRegistry>, embeddings: Option<Arc<EmbeddingManager>>) -> Self {
        Self {
            skills,
            embeddings,
            skill_vecs: Mutex::new(None),
        }
    }

    /// Route a request to the top-k matching skills, best first.
    /// Returns an empty vec when nothing scores above noise.
    pub fn route(&self, request: &str, top_k: usize) -> Vec<RouteSuggestion> {
        let request = request.trim();
        if request.is_empty() || top_k == 0 {
            return Vec::new();
        }
        if let Some(embs) = &self.embeddings {
            if let Ok(qe) = embs.embed_query(request) {
                if let Some(ranked) = self.route_embedding(&qe, top_k) {
                    return ranked;
                }
            }
        }
        self.route_keywords(request, top_k)
    }

    /// Embedding-based routing — skill description vectors are computed
    /// once and cached (embeddings are deterministic per model). Returns
    /// `None` when the index could not be built so the caller falls back
    /// to keyword routing; a transient embedder error must not poison
    /// routing for the router's lifetime.
    fn route_embedding(&self, query_vec: &[f32], top_k: usize) -> Option<Vec<RouteSuggestion>> {
        let vecs = {
            let mut cache = self.skill_vecs.lock();
            if cache.is_none() {
                let embs = self.embeddings.as_ref().expect("checked above");
                let texts: Vec<String> = self.skills.all().iter().map(skill_description).collect();
                let refs: Vec<&str> = texts.iter().map(|t| t.as_str()).collect();
                match embs.embed_passages(&refs) {
                    Ok(v) => {
                        *cache = Some(
                            self.skills
                                .all()
                                .iter()
                                .map(|s| s.id.clone())
                                .zip(v)
                                .collect(),
                        );
                    }
                    Err(e) => {
                        tracing::warn!("skill vector index failed: {}", e);
                        return None;
                    }
                }
            }
            cache.clone().unwrap_or_default()
        };

        let mut scored: Vec<RouteSuggestion> = vecs
            .iter()
            .map(|(id, v)| RouteSuggestion {
                skill_id: id.clone(),
                score: cosine_similarity(query_vec, v) as f64,
                method: "embedding",
            })
            .filter(|s| s.score > 0.25) // below ~0.25 cosine is noise for this model
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        Some(scored)
    }

    /// Keyword fallback — token overlap between the request and the skill's
    /// label + description + id tokens. Deterministic, no model needed.
    fn route_keywords(&self, request: &str, top_k: usize) -> Vec<RouteSuggestion> {
        let req_tokens: std::collections::HashSet<String> = request
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| t.len() >= 3)
            .map(|t| t.to_lowercase())
            .collect();
        if req_tokens.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<RouteSuggestion> = self
            .skills
            .all()
            .iter()
            .map(|s| {
                let skill_text =
                    format!("{} {} {}", s.id.replace('_', " "), s.label, s.description)
                        .to_lowercase();
                let hits = req_tokens
                    .iter()
                    .filter(|t| skill_text.contains(t.as_str()))
                    .count();
                RouteSuggestion {
                    skill_id: s.id.clone(),
                    score: hits as f64 / req_tokens.len() as f64,
                    method: "keyword",
                }
            })
            .filter(|s| s.score > 0.0)
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }

    /// Invalidate cached skill vectors (call after swapping embedder/model).
    pub fn invalidate_cache(&self) {
        *self.skill_vecs.lock() = None;
    }
}

/// Text used to represent a skill for routing: label + description + lora
/// task. Kept short — embedding quality degrades on long text.
fn skill_description(skill: &SkillDef) -> String {
    format!(
        "{}: {} ({})",
        skill.label, skill.description, skill.lora_task
    )
}

/// A route decision combining skill + resolved LoRA family + grammar.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub skill_id: String,
    pub lora_task: String,
    pub grammar_type: String,
    pub score: f64,
    pub method: &'static str,
}

impl SkillRouter {
    /// Full routing decision — resolves the top skill and surfaces the
    /// adapter family and grammar it will use.
    pub fn decide(&self, request: &str) -> Option<RouteDecision> {
        let top = self.route(request, 1).into_iter().next()?;
        let skill = self.skills.get(&top.skill_id)?;
        Some(RouteDecision {
            skill_id: top.skill_id,
            lora_task: skill.lora_task.clone(),
            grammar_type: serde_json::to_string(&skill.grammar_type)
                .unwrap_or_default()
                .trim_matches('"')
                .to_string(),
            score: top.score,
            method: top.method,
        })
    }
}

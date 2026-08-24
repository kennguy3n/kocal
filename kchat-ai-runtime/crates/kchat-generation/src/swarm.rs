//! Swarm inference — coordinate multiple small models (peers) to produce
//! higher-quality output than a single model.
//!
//! Swarm inference is useful for complex tasks where one model's weakness is
//! another's strength. Each peer generates independently, then the swarm
//! computes a consensus score. If consensus is below the threshold, peers
//! see each other's outputs and generate again, up to `max_rounds`.
//!
//! The consensus metric is a simple Jaccard similarity over whitespace-
//! tokenized outputs, averaged across all peer pairs. No external
//! dependencies are required.

use crate::backend::{BackendAdapter, BackendError, GenerationConfig, GenerationResult};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::Instant;

/// Configuration for swarm inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Number of peers in the swarm.
    pub num_peers: usize,
    /// Consensus threshold (0.0-1.0). When the average pairwise similarity
    /// reaches this value, the swarm stops early and returns the best output.
    pub consensus_threshold: f64,
    /// Maximum number of rounds before giving up and returning the best output.
    pub max_rounds: usize,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            num_peers: 3,
            consensus_threshold: 0.67,
            max_rounds: 3,
        }
    }
}

impl SwarmConfig {
    /// Validate config values are within reasonable bounds.
    pub fn validate(&self) -> Result<(), SwarmError> {
        if self.num_peers == 0 {
            return Err(SwarmError::GenerationFailed(
                "num_peers must be > 0".into(),
            ));
        }
        if self.num_peers > 20 {
            return Err(SwarmError::GenerationFailed(
                "num_peers must be <= 20".into(),
            ));
        }
        if self.consensus_threshold < 0.0 || self.consensus_threshold > 1.0 {
            return Err(SwarmError::GenerationFailed(
                "consensus_threshold must be in [0.0, 1.0]".into(),
            ));
        }
        if self.max_rounds == 0 {
            return Err(SwarmError::GenerationFailed(
                "max_rounds must be > 0".into(),
            ));
        }
        if self.max_rounds > 10 {
            return Err(SwarmError::GenerationFailed(
                "max_rounds must be <= 10".into(),
            ));
        }
        Ok(())
    }
}

/// A single peer (model) in the swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peer {
    /// Unique peer identifier.
    pub id: String,
    /// Model name (e.g. "bonsai-1.7b-q1_0").
    pub model_name: String,
    /// Task specialty (e.g. "summarize", "translate").
    pub specialty: String,
}

impl Peer {
    /// Create a new peer.
    pub fn new(
        id: impl Into<String>,
        model_name: impl Into<String>,
        specialty: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            model_name: model_name.into(),
            specialty: specialty.into(),
        }
    }
}

/// Result of a swarm inference run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmResult {
    /// The final (best) output text selected by the swarm.
    pub final_text: String,
    /// Consensus score (average pairwise Jaccard similarity) at termination.
    pub consensus_score: f64,
    /// Number of rounds actually executed.
    pub num_rounds: usize,
    /// Per-peer outputs from the final round.
    pub peer_outputs: Vec<String>,
    /// Total tokens consumed across all peers and rounds.
    pub total_tokens: u64,
}

/// Swarm errors.
#[derive(Debug, thiserror::Error)]
pub enum SwarmError {
    #[error("no peers in swarm")]
    NoPeers,

    #[error("peer generation failed: {0}")]
    GenerationFailed(String),

    #[error("backend error: {0}")]
    Backend(#[from] BackendError),
}

/// A swarm of peers that coordinates multi-model inference.
///
/// The swarm is generic over the backend adapter so it can work with any
/// concrete backend (llama.cpp, ONNX, cloud) or with a `MockPeer` for testing.
pub struct Swarm {
    config: SwarmConfig,
    peers: Vec<Peer>,
}

impl Swarm {
    /// Create a new swarm with the given configuration and no peers.
    /// Clamps config values to safe bounds if they are out of range.
    pub fn new(mut config: SwarmConfig) -> Self {
        // Clamp invalid values to safe defaults rather than silently ignoring
        if config.num_peers == 0 {
            tracing::warn!("SwarmConfig.num_peers was 0, clamping to 1");
            config.num_peers = 1;
        } else if config.num_peers > 20 {
            tracing::warn!("SwarmConfig.num_peers was {}, clamping to 20", config.num_peers);
            config.num_peers = 20;
        }
        if config.consensus_threshold < 0.0 {
            tracing::warn!("SwarmConfig.consensus_threshold was {}, clamping to 0.0", config.consensus_threshold);
            config.consensus_threshold = 0.0;
        } else if config.consensus_threshold > 1.0 {
            tracing::warn!("SwarmConfig.consensus_threshold was {}, clamping to 1.0", config.consensus_threshold);
            config.consensus_threshold = 1.0;
        }
        if config.max_rounds == 0 {
            tracing::warn!("SwarmConfig.max_rounds was 0, clamping to 1");
            config.max_rounds = 1;
        } else if config.max_rounds > 10 {
            tracing::warn!("SwarmConfig.max_rounds was {}, clamping to 10", config.max_rounds);
            config.max_rounds = 10;
        }
        Self {
            config,
            peers: Vec::new(),
        }
    }

    /// Add a peer to the swarm.
    pub fn add_peer(&mut self, peer: Peer) {
        self.peers.push(peer);
    }

    /// Get the number of peers currently in the swarm.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Get a reference to the peers.
    pub fn peers(&self) -> &[Peer] {
        &self.peers
    }

    /// Compute the consensus score for a set of outputs.
    ///
    /// This computes the average pairwise Jaccard similarity over
    /// whitespace-tokenized text. If there are fewer than 2 outputs,
    /// the consensus is 1.0 (trivially agreed).
    pub fn compute_consensus(outputs: &[String]) -> f64 {
        if outputs.len() < 2 {
            return 1.0;
        }

        let tokenized: Vec<HashSet<&str>> = outputs
            .iter()
            .map(|s| s.split_whitespace().collect())
            .collect();

        let mut sum = 0.0;
        let mut count = 0u32;
        for i in 0..tokenized.len() {
            for j in (i + 1)..tokenized.len() {
                sum += jaccard(&tokenized[i], &tokenized[j]);
                count += 1;
            }
        }

        if count == 0 {
            return 1.0;
        }
        sum / count as f64
    }

    /// Run swarm inference.
    ///
    /// Each peer generates independently. If the consensus score is below
    /// the threshold, peers see each other's outputs and generate again.
    /// This repeats up to `max_rounds` times.
    ///
    /// The "best" output is selected as the one with the highest average
    /// similarity to all other peer outputs in the final round.
    pub fn generate(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        backends: &[&dyn BackendAdapter],
    ) -> Result<SwarmResult, SwarmError> {
        if self.peers.is_empty() {
            return Err(SwarmError::NoPeers);
        }
        if backends.len() < self.peers.len() {
            return Err(SwarmError::GenerationFailed(format!(
                "need {} backends, got {}",
                self.peers.len(),
                backends.len()
            )));
        }

        let mut total_tokens: u64 = 0;
        let mut current_outputs: Vec<String> = Vec::new();
        let mut num_rounds = 0;

        for round in 0..self.config.max_rounds {
            num_rounds = round + 1;
            let round_prompt = if round == 0 {
                prompt.to_string()
            } else {
                // Peers see each other's outputs from the previous round.
                let peer_views: Vec<String> = self
                    .peers
                    .iter()
                    .zip(current_outputs.iter())
                    .map(|(peer, output)| format!("[peer {}]: {}", peer.id, output))
                    .collect();
                format!(
                    "{}\n\nOther peers produced:\n{}\n\nPlease refine your answer.",
                    prompt,
                    peer_views.join("\n")
                )
            };

            current_outputs.clear();
            for (i, _peer) in self.peers.iter().enumerate() {
                let result: GenerationResult = backends[i].generate(&round_prompt, config)?;
                total_tokens = total_tokens.saturating_add(result.completion_tokens as u64);
                current_outputs.push(result.text);
            }

            let consensus = Self::compute_consensus(&current_outputs);
            tracing::info!(
                "swarm round {} consensus {:.3} (threshold {:.3})",
                round + 1,
                consensus,
                self.config.consensus_threshold
            );

            if consensus >= self.config.consensus_threshold {
                let best = select_best(&current_outputs);
                return Ok(SwarmResult {
                    final_text: best.clone(),
                    consensus_score: consensus,
                    num_rounds,
                    peer_outputs: current_outputs.clone(),
                    total_tokens,
                });
            }
        }

        // Exhausted max_rounds — return the best output we have.
        let consensus = Self::compute_consensus(&current_outputs);
        let best = select_best(&current_outputs);
        tracing::info!(
            "swarm exhausted max_rounds={} final consensus {:.3}",
            self.config.max_rounds,
            consensus
        );
        Ok(SwarmResult {
            final_text: best.clone(),
            consensus_score: consensus,
            num_rounds,
            peer_outputs: current_outputs.clone(),
            total_tokens,
        })
    }
}

/// Compute Jaccard similarity between two token sets.
/// Empty sets are treated as 0.0 similarity (no meaningful consensus on empty output).
fn jaccard(a: &HashSet<&str>, b: &HashSet<&str>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.len() + b.len() - intersection;
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Select the output with the highest average similarity to all others.
fn select_best(outputs: &[String]) -> String {
    if outputs.is_empty() {
        return String::new();
    }
    if outputs.len() == 1 {
        return outputs[0].clone();
    }

    let tokenized: Vec<HashSet<&str>> = outputs
        .iter()
        .map(|s| s.split_whitespace().collect())
        .collect();

    let mut best_idx = 0;
    let mut best_score = f64::MIN;
    for i in 0..outputs.len() {
        let mut sum = 0.0;
        for j in 0..outputs.len() {
            if i != j {
                sum += jaccard(&tokenized[i], &tokenized[j]);
            }
        }
        let avg = sum / (outputs.len() - 1) as f64;
        if avg > best_score {
            best_score = avg;
            best_idx = i;
        }
    }
    outputs[best_idx].clone()
}

/// A mock peer backend for testing swarm inference without a real model.
///
/// The mock peer returns a deterministic output based on its `id` and an
/// optional "refinement" that nudges it toward a shared answer on later
/// rounds. This lets tests exercise the consensus and multi-round logic.
pub struct MockPeer {
    /// The peer ID this backend represents.
    pub peer_id: String,
    /// The base output text to return on round 1.
    pub base_output: String,
    /// The refined output to return on round 2+ (when peers see each other).
    pub refined_output: String,
    /// Simulated completion tokens per generation.
    pub completion_tokens: u32,
    start: Instant,
}

impl MockPeer {
    /// Create a new mock peer.
    pub fn new(
        peer_id: impl Into<String>,
        base_output: impl Into<String>,
        refined_output: impl Into<String>,
    ) -> Self {
        Self {
            peer_id: peer_id.into(),
            base_output: base_output.into(),
            refined_output: refined_output.into(),
            completion_tokens: 10,
            start: Instant::now(),
        }
    }

    /// Set the simulated completion token count.
    pub fn with_tokens(mut self, tokens: u32) -> Self {
        self.completion_tokens = tokens;
        self
    }
}

impl BackendAdapter for MockPeer {
    fn load(&self, _config: &crate::backend::BackendConfig) -> Result<(), BackendError> {
        Ok(())
    }

    fn unload(&self) -> Result<(), BackendError> {
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        true
    }

    fn generate(
        &self,
        prompt: &str,
        _config: &GenerationConfig,
    ) -> Result<GenerationResult, BackendError> {
        // If the prompt includes "Other peers produced", this is a refinement
        // round — return the refined output.
        let text = if prompt.contains("Other peers produced") {
            self.refined_output.clone()
        } else {
            self.base_output.clone()
        };
        let elapsed = self.start.elapsed().as_millis() as u64;
        Ok(GenerationResult {
            text,
            prompt_tokens: 5,
            completion_tokens: self.completion_tokens,
            ttft_ms: 10,
            total_ms: elapsed,
            tokens_per_second: 100.0,
            backend: format!("mock_peer_{}", self.peer_id),
            grammar_valid: true,
        })
    }

    fn generate_stream(
        &self,
        _prompt: &str,
        _config: &GenerationConfig,
        _stream: &crate::stream::StreamHandle,
    ) -> Result<GenerationResult, BackendError> {
        Err(BackendError::Unavailable(
            "MockPeer does not support streaming".into(),
        ))
    }

    fn backend_type(&self) -> crate::backend::BackendType {
        crate::backend::BackendType::LlamaCppCpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendType;

    // --- Config tests ---

    #[test]
    fn test_swarm_config_defaults() {
        let config = SwarmConfig::default();
        assert_eq!(config.num_peers, 3);
        assert!((config.consensus_threshold - 0.67).abs() < f64::EPSILON);
        assert_eq!(config.max_rounds, 3);
    }

    #[test]
    fn test_swarm_config_custom() {
        let config = SwarmConfig {
            num_peers: 5,
            consensus_threshold: 0.8,
            max_rounds: 10,
        };
        assert_eq!(config.num_peers, 5);
        assert!((config.consensus_threshold - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.max_rounds, 10);
    }

    // --- Peer tests ---

    #[test]
    fn test_peer_creation() {
        let peer = Peer::new("p1", "bonsai-1.7b-q1_0", "summarize");
        assert_eq!(peer.id, "p1");
        assert_eq!(peer.model_name, "bonsai-1.7b-q1_0");
        assert_eq!(peer.specialty, "summarize");
    }

    #[test]
    fn test_swarm_peer_management() {
        let mut swarm = Swarm::new(SwarmConfig::default());
        assert_eq!(swarm.peer_count(), 0);
        assert!(swarm.peers().is_empty());

        swarm.add_peer(Peer::new("p1", "model-a", "summarize"));
        assert_eq!(swarm.peer_count(), 1);

        swarm.add_peer(Peer::new("p2", "model-b", "translate"));
        assert_eq!(swarm.peer_count(), 2);

        let peers = swarm.peers();
        assert_eq!(peers[0].id, "p1");
        assert_eq!(peers[1].id, "p2");
    }

    // --- Consensus computation tests ---

    #[test]
    fn test_compute_consensus_identical() {
        let outputs = vec![
            "the quick brown fox".to_string(),
            "the quick brown fox".to_string(),
            "the quick brown fox".to_string(),
        ];
        let score = Swarm::compute_consensus(&outputs);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_consensus_disjoint() {
        let outputs = vec![
            "alpha beta".to_string(),
            "gamma delta".to_string(),
        ];
        let score = Swarm::compute_consensus(&outputs);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_consensus_partial() {
        // Jaccard: intersection {the, cat} = 2, union {the, cat, dog, sat} = 4 => 0.5
        let outputs = vec![
            "the cat sat".to_string(),
            "the cat dog".to_string(),
        ];
        let score = Swarm::compute_consensus(&outputs);
        assert!((score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_compute_consensus_single_output() {
        let outputs = vec!["only one".to_string()];
        let score = Swarm::compute_consensus(&outputs);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_consensus_empty() {
        let outputs: Vec<String> = vec![];
        let score = Swarm::compute_consensus(&outputs);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    // --- Swarm generate tests ---

    #[test]
    fn test_swarm_empty_errors() {
        let swarm = Swarm::new(SwarmConfig::default());
        let config = GenerationConfig::default();
        let result = swarm.generate("hello", &config, &[]);
        assert!(matches!(result, Err(SwarmError::NoPeers)));
    }

    #[test]
    fn test_swarm_single_peer_consensus() {
        let mut swarm = Swarm::new(SwarmConfig {
            num_peers: 1,
            consensus_threshold: 0.67,
            max_rounds: 3,
        });
        swarm.add_peer(Peer::new("p1", "model-a", "summarize"));

        let peer = MockPeer::new("p1", "the quick brown fox", "the quick brown fox");
        let backends: Vec<&dyn BackendAdapter> = vec![&peer];
        let config = GenerationConfig::default();

        let result = swarm.generate("summarize this", &config, &backends).unwrap();
        // Single peer => consensus 1.0 => terminates in round 1
        assert!((result.consensus_score - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.num_rounds, 1);
        assert_eq!(result.final_text, "the quick brown fox");
        assert_eq!(result.peer_outputs.len(), 1);
        assert!(result.total_tokens > 0);
    }

    #[test]
    fn test_swarm_multi_peer_immediate_consensus() {
        let mut swarm = Swarm::new(SwarmConfig {
            num_peers: 3,
            consensus_threshold: 0.67,
            max_rounds: 3,
        });
        swarm.add_peer(Peer::new("p1", "model-a", "summarize"));
        swarm.add_peer(Peer::new("p2", "model-b", "summarize"));
        swarm.add_peer(Peer::new("p3", "model-c", "summarize"));

        // All peers produce identical output => consensus 1.0 on round 1
        let p1 = MockPeer::new("p1", "the answer is forty two", "the answer is forty two");
        let p2 = MockPeer::new("p2", "the answer is forty two", "the answer is forty two");
        let p3 = MockPeer::new("p3", "the answer is forty two", "the answer is forty two");
        let backends: Vec<&dyn BackendAdapter> = vec![&p1, &p2, &p3];
        let config = GenerationConfig::default();

        let result = swarm.generate("what is the answer", &config, &backends).unwrap();
        assert!((result.consensus_score - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.num_rounds, 1);
        assert_eq!(result.final_text, "the answer is forty two");
        assert_eq!(result.peer_outputs.len(), 3);
    }

    #[test]
    fn test_swarm_multi_peer_reaches_consensus_after_refinement() {
        let mut swarm = Swarm::new(SwarmConfig {
            num_peers: 2,
            consensus_threshold: 0.67,
            max_rounds: 3,
        });
        swarm.add_peer(Peer::new("p1", "model-a", "summarize"));
        swarm.add_peer(Peer::new("p2", "model-b", "summarize"));

        // Round 1: peers disagree. Round 2+: both refine to the same answer.
        let p1 = MockPeer::new("p1", "cats are great pets", "the best pet is a cat");
        let p2 = MockPeer::new("p2", "dogs are loyal friends", "the best pet is a cat");
        let backends: Vec<&dyn BackendAdapter> = vec![&p1, &p2];
        let config = GenerationConfig::default();

        let result = swarm.generate("best pet", &config, &backends).unwrap();
        // After refinement both produce identical text => consensus 1.0
        assert!((result.consensus_score - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.num_rounds, 2);
        assert_eq!(result.final_text, "the best pet is a cat");
    }

    #[test]
    fn test_swarm_exhausts_max_rounds() {
        let mut swarm = Swarm::new(SwarmConfig {
            num_peers: 2,
            consensus_threshold: 0.99, // impossibly high
            max_rounds: 2,
        });
        swarm.add_peer(Peer::new("p1", "model-a", "summarize"));
        swarm.add_peer(Peer::new("p2", "model-b", "summarize"));

        // Peers never converge — refined outputs still differ.
        let p1 = MockPeer::new("p1", "alpha beta gamma", "alpha beta delta");
        let p2 = MockPeer::new("p2", "epsilon zeta eta", "epsilon zeta theta");
        let backends: Vec<&dyn BackendAdapter> = vec![&p1, &p2];
        let config = GenerationConfig::default();

        let result = swarm.generate("test", &config, &backends).unwrap();
        assert_eq!(result.num_rounds, 2);
        // Consensus should be below threshold
        assert!(result.consensus_score < 0.99);
        // Should still return some output
        assert!(!result.final_text.is_empty());
        assert_eq!(result.peer_outputs.len(), 2);
    }

    #[test]
    fn test_swarm_total_tokens_accumulate() {
        let mut swarm = Swarm::new(SwarmConfig {
            num_peers: 2,
            consensus_threshold: 0.99,
            max_rounds: 2,
        });
        swarm.add_peer(Peer::new("p1", "model-a", "summarize"));
        swarm.add_peer(Peer::new("p2", "model-b", "summarize"));

        // Peers never converge — outputs are disjoint in both rounds.
        let p1 = MockPeer::new("p1", "alpha beta", "alpha gamma").with_tokens(15);
        let p2 = MockPeer::new("p2", "delta epsilon", "delta zeta").with_tokens(20);
        let backends: Vec<&dyn BackendAdapter> = vec![&p1, &p2];
        let config = GenerationConfig::default();

        let result = swarm.generate("test", &config, &backends).unwrap();
        // 2 rounds × (15 + 20) = 70 tokens
        assert_eq!(result.num_rounds, 2);
        assert_eq!(result.total_tokens, 70);
    }

    #[test]
    fn test_mock_peer_backend_type() {
        let peer = MockPeer::new("p1", "hello", "hello");
        assert_eq!(peer.backend_type(), BackendType::LlamaCppCpu);
        assert!(peer.is_loaded());
    }

    #[test]
    fn test_jaccard_empty_sets() {
        // Empty sets should return 0.0 (no meaningful consensus on empty output)
        let a: HashSet<&str> = HashSet::new();
        let b: HashSet<&str> = HashSet::new();
        assert!((jaccard(&a, &b) - 0.0).abs() < f64::EPSILON);

        // One empty, one non-empty should also be 0.0
        let c: HashSet<&str> = ["hello"].iter().copied().collect();
        assert!((jaccard(&a, &c) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_select_best_picks_most_representative() {
        let outputs = vec![
            "the cat sat".to_string(),
            "the cat ran".to_string(),
            "zzz qqq xxx".to_string(),
        ];
        // "the cat sat" and "the cat ran" share {the, cat} => Jaccard 0.5
        // "zzz qqq xxx" shares nothing with either => avg 0.0
        // Both "the cat sat" and "the cat ran" have avg 0.25 (0.5 + 0.0 / 2)
        let best = select_best(&outputs);
        assert!(best == "the cat sat" || best == "the cat ran");
    }
}

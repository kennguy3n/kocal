//! Integration tests for the kchat-runtime orchestrator — full pipeline
//! through a MockBackend (no real model needed).

use kchat_context::scope::{ScopeFilter, ScopeId};
use kchat_context::store::{ContextStore, ContextStoreConfig, Evidence, EvidenceId};
use kchat_core::tier::DeviceTier;
use kchat_generation::backend::{
    BackendAdapter, BackendConfig, BackendError, BackendType, GenerationConfig, GenerationResult,
};
use kchat_generation::backends::mock::MockBackend;
use kchat_generation::stream::{StreamEvent, StreamHandle};
use kchat_runtime::{Orchestrator, SkillError, SkillRequest};
use std::sync::Arc;
use uuid::Uuid;

/// Deterministically-blocked phrase (scam detector: urgency + crypto).
const BLOCKED_PHRASE: &str = "URGENT! Send money via bitcoin immediately!";

/// Backend that emits a fixed response regardless of the prompt — used to
/// produce output that the safety monitor must block mid-stream/end.
struct CannedBackend {
    text: String,
}

impl BackendAdapter for CannedBackend {
    fn load(&self, _config: &BackendConfig) -> Result<(), BackendError> {
        Ok(())
    }
    fn unload(&self) -> Result<(), BackendError> {
        Ok(())
    }
    fn is_loaded(&self) -> bool {
        true
    }
    fn backend_type(&self) -> BackendType {
        BackendType::LlamaCppMetal
    }
    fn generate(
        &self,
        _prompt: &str,
        config: &GenerationConfig,
    ) -> Result<GenerationResult, BackendError> {
        Ok(GenerationResult {
            text: self.text.clone(),
            prompt_tokens: 1,
            completion_tokens: config.max_tokens as u32,
            ttft_ms: 1,
            total_ms: 2,
            tokens_per_second: 50.0,
            backend: "canned".into(),
            grammar_valid: true,
        })
    }
    fn generate_stream(
        &self,
        _prompt: &str,
        config: &GenerationConfig,
        stream: &StreamHandle,
    ) -> Result<GenerationResult, BackendError> {
        for word in self.text.split_whitespace() {
            if stream.is_cancelled() {
                return Err(BackendError::GenerationFailed("cancelled".into()));
            }
            stream.push_token(format!("{word} "));
        }
        stream.complete(config.max_tokens as u32, 2);
        self.generate(_prompt, config)
    }
}

fn make_backend() -> Arc<MockBackend> {
    let backend = Arc::new(MockBackend::new());
    backend
        .load(&BackendConfig::for_tier(
            kchat_generation::backend::BackendType::LlamaCppMetal,
            "mock",
            "/dev/null",
            DeviceTier::High,
            "macos",
        ))
        .unwrap();
    backend
}

fn make_store() -> ContextStore {
    ContextStore::open_in_memory(&ContextStoreConfig::for_low_tier("test".into(), [7u8; 32]))
        .unwrap()
}

fn make_evidence(scope_id: ScopeId, content: &str) -> Evidence {
    Evidence {
        id: EvidenceId::new(),
        scope_id,
        content_hash: blake3::hash(content.as_bytes()).to_hex().to_string(),
        encrypted_body: vec![],
        nonce: vec![1u8; 24],
        source_ref: None,
        importance: 5,
        language_tag: Some("en".into()),
        created_at: chrono::Utc::now().timestamp(),
        fts_content: content.into(),
    }
}

fn request(input: &str, context: &str) -> SkillRequest {
    SkillRequest {
        input: input.into(),
        context: context.into(),
        ..Default::default()
    }
}

/// Find a skill ID with the given minimum tier (registry contents are
/// data-driven — don't hardcode tier assumptions in tests).
fn skill_with_tier(orc: &Orchestrator, tier: kchat_generation::skills::SkillTier) -> String {
    orc.skills()
        .all()
        .iter()
        .find(|s| s.min_tier == tier)
        .expect("registry should have a skill at every tier")
        .id
        .clone()
}

#[test]
fn test_run_skill_streams_and_completes() {
    let orc = Orchestrator::builder()
        .backend(make_backend())
        .tier(DeviceTier::High)
        .build();

    let skill = skill_with_tier(&orc, kchat_generation::skills::SkillTier::Low);
    let mut run = orc
        .run_skill(&skill, request("do the thing", "document text"))
        .unwrap();

    let mut saw_token = false;
    let mut saw_complete = false;
    while let Some(ev) = run.events.blocking_recv() {
        match ev {
            StreamEvent::Token { .. } => saw_token = true,
            StreamEvent::Complete { .. } => {
                saw_complete = true;
                break;
            }
            StreamEvent::Error { message } => panic!("stream error: {message}"),
            StreamEvent::Cancelled { reason } => panic!("cancelled: {reason}"),
        }
    }
    assert!(saw_token);
    assert!(saw_complete);

    let outcome = run.outcome.blocking_recv().unwrap().unwrap();
    assert!(!outcome.output_blocked);
}

#[test]
fn test_unknown_skill_errors() {
    let orc = Orchestrator::builder()
        .backend(make_backend())
        .tier(DeviceTier::High)
        .build();
    let err = orc
        .run_skill("nonexistent_skill", request("x", ""))
        .err()
        .expect("should fail");
    assert!(matches!(err, SkillError::UnknownSkill(_)));
}

#[test]
fn test_tier_gate_blocks_high_skill_on_low_device() {
    let orc = Orchestrator::builder()
        .backend(make_backend())
        .tier(DeviceTier::Low)
        .build();
    let skill = skill_with_tier(&orc, kchat_generation::skills::SkillTier::High);
    let err = orc
        .run_skill(&skill, request("x", "doc"))
        .err()
        .expect("should fail");
    assert!(matches!(err, SkillError::TierTooLow { .. }));
}

#[test]
fn test_retrieval_injects_context() {
    let store = Arc::new(make_store());
    let scope = ScopeId::new();
    store
        .insert(&make_evidence(scope, "quarterly revenue grew 12 percent"))
        .unwrap();

    let orc = Orchestrator::builder()
        .backend(make_backend())
        .context_store(store)
        .tier(DeviceTier::High)
        .build();

    let skill = skill_with_tier(&orc, kchat_generation::skills::SkillTier::Low);
    let filter = ScopeFilter {
        allowed_scopes: vec![scope],
        denied_scopes: vec![],
        user_id: Uuid::new_v4(),
        roles: vec![],
    };
    let run = orc
        .run_skill(
            &skill,
            SkillRequest {
                input: "revenue".into(),
                context: String::new(),
                scope_filter: Some(filter),
                ..Default::default()
            },
        )
        .unwrap();

    let outcome = run.outcome.blocking_recv().unwrap().unwrap();
    assert_eq!(outcome.retrieved_passages, 1);
    // Mock echoes the prompt — retrieved text should appear in the output
    assert!(outcome.generation.text.contains("quarterly revenue"));
}

#[test]
fn test_best_of_n_high_tier() {
    let orc = Orchestrator::builder()
        .backend(make_backend())
        .tier(DeviceTier::High)
        .build();
    let skill = skill_with_tier(&orc, kchat_generation::skills::SkillTier::Low);
    let run = orc
        .run_skill(
            &skill,
            SkillRequest {
                best_of_n: 2,
                ..request("x", "doc")
            },
        )
        .unwrap();
    let outcome = run.outcome.blocking_recv().unwrap().unwrap();
    assert!(!outcome.generation.text.is_empty());
}

/// Regression test for the aarch64 LLVM miscompile at dev `opt-level=1`:
/// `best_of_n = 0` on High tier must take the single-stream path (the bug
/// folded away the `n > 1` guard and entered best-of-n with n=0).
#[test]
fn test_high_tier_default_best_of_n_single_stream() {
    let orc = Orchestrator::builder()
        .backend(make_backend())
        .tier(DeviceTier::High)
        .build();
    let skill = skill_with_tier(&orc, kchat_generation::skills::SkillTier::Low);
    let mut run = orc
        .run_skill(&skill, request("do the thing", "doc"))
        .unwrap();

    let mut saw_complete = false;
    while let Some(ev) = run.events.blocking_recv() {
        match ev {
            StreamEvent::Complete { .. } => saw_complete = true,
            StreamEvent::Error { message } => panic!("stream error: {message}"),
            StreamEvent::Cancelled { reason } => panic!("cancelled: {reason}"),
            StreamEvent::Token { .. } => {}
        }
    }
    assert!(saw_complete);
    let outcome = run.outcome.blocking_recv().unwrap().unwrap();
    assert!(!outcome.generation.text.is_empty());
}

/// The withheld tail: fewer than SAFETY_CHECK_EVERY_TOKENS tokens of blocked
/// output must never reach the caller — the terminal classify gates them.
#[test]
fn test_blocked_tail_never_reaches_caller() {
    let orc = Orchestrator::builder()
        .backend(Arc::new(CannedBackend {
            text: BLOCKED_PHRASE.into(),
        }))
        .tier(DeviceTier::High)
        .build();
    let skill = skill_with_tier(&orc, kchat_generation::skills::SkillTier::Low);
    let mut run = orc.run_skill(&skill, request("summarize", "doc")).unwrap();

    let mut saw_cancelled = false;
    while let Some(ev) = run.events.blocking_recv() {
        match ev {
            StreamEvent::Token { text } => {
                assert!(!text.contains("URGENT"), "blocked token leaked: {text}");
            }
            StreamEvent::Cancelled { .. } => saw_cancelled = true,
            StreamEvent::Complete { .. } => panic!("blocked stream completed"),
            StreamEvent::Error { .. } => {}
        }
    }
    assert!(saw_cancelled, "blocked output must end with Cancelled");

    let outcome = run.outcome.blocking_recv().unwrap().unwrap();
    assert!(outcome.output_blocked);
    assert!(
        outcome.generation.text.is_empty(),
        "blocked text leaked in outcome"
    );
}

/// Mid-stream block: clean tokens may stream, but the blocked phrase and
/// everything after it must be cut off with a Cancelled event.
#[test]
fn test_midstream_block_cancels() {
    let clean = "word ".repeat(40);
    let text = format!("{clean}{BLOCKED_PHRASE} more tokens after the payload");
    let orc = Orchestrator::builder()
        .backend(Arc::new(CannedBackend { text }))
        .tier(DeviceTier::High)
        .build();
    let skill = skill_with_tier(&orc, kchat_generation::skills::SkillTier::Low);
    let mut run = orc.run_skill(&skill, request("summarize", "doc")).unwrap();

    let mut streamed = String::new();
    let mut saw_cancelled = false;
    while let Some(ev) = run.events.blocking_recv() {
        match ev {
            StreamEvent::Token { text } => streamed.push_str(&text),
            StreamEvent::Cancelled { .. } => saw_cancelled = true,
            StreamEvent::Complete { .. } => panic!("blocked stream completed"),
            StreamEvent::Error { .. } => {}
        }
    }
    assert!(saw_cancelled);
    assert!(
        !streamed.contains("URGENT"),
        "blocked phrase leaked mid-stream"
    );

    // Race-tolerant: if the backend finished before the cancel landed, the
    // outcome is Ok with output_blocked; if it was mid-stream, Err.
    let outcome = run.outcome.blocking_recv().unwrap();
    match outcome {
        Ok(o) => {
            assert!(o.output_blocked);
            assert!(o.generation.text.is_empty());
        }
        Err(_) => {}
    }
}

#[test]
fn test_router_keyword_fallback() {
    let orc = Orchestrator::builder()
        .backend(make_backend())
        .tier(DeviceTier::High)
        .build();
    let routes = orc.router().route("please summarize this document", 3);
    assert!(!routes.is_empty());
    assert_eq!(routes[0].method, "keyword");
    // A summarize-related skill should rank first
    assert!(routes[0].skill_id.contains("summar"));
}

#[test]
fn test_memory_writeback_persists() {
    let store = Arc::new(make_store());
    let scope = ScopeId::new();

    let orc = Orchestrator::builder()
        .backend(make_backend())
        .context_store(store.clone())
        .memory_scope(scope)
        .tier(DeviceTier::High)
        .build();

    // Buffer enough turns to cross the 4KB flush threshold.
    let big = "x".repeat(2500);
    orc.memory().record_turn("q1", &big);
    let flushed = orc.memory().record_turn("q2", &big);
    assert!(flushed, "second large turn should trigger flush");

    // The flushed log is now searchable in the store.
    let filter = ScopeFilter {
        allowed_scopes: vec![scope],
        denied_scopes: vec![],
        user_id: Uuid::new_v4(),
        roles: vec![],
    };
    let rows = store.list_evidence_in_scopes(&filter, 10).unwrap();
    assert!(!rows.is_empty(), "flushed memory should persist");
}

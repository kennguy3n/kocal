//! Generation pipeline overhead — prompt construction, grammar resolution,
//! and mock-decode streaming. These are the per-request costs the
//! orchestrator pays on top of model inference.
//!
//! Run: `cargo bench -p kchat-generation`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kchat_generation::backend::{BackendAdapter, BackendConfig, BackendType, GenerationConfig};
use kchat_generation::grammar::{Grammar, GrammarValidator};
use kchat_generation::skills::{SkillPromptInput, SkillRegistry, SkillTier};
use kchat_generation::stream::StreamHandle;

fn bench_prompt_build(c: &mut Criterion) {
    let registry = SkillRegistry::new();
    let skills = registry.all();
    c.bench_function("build_prompt_all_skills", |b| {
        b.iter(|| {
            for skill in skills {
                let out = skill.build_prompt(SkillPromptInput {
                    input: black_box("Summarize the quarterly report"),
                    context: black_box("Revenue grew 12% year over year..."),
                    keywords: "",
                    variant_context: "",
                    tier: Some(SkillTier::Medium),
                });
                black_box(out.to_chatml(skill.response_prefix.as_deref()));
            }
        })
    });
}

fn bench_grammar(c: &mut Criterion) {
    let registry = SkillRegistry::new();
    let skills = registry.all();
    let mut group = c.benchmark_group("grammar");

    group.bench_function("for_skill_all", |b| {
        b.iter(|| {
            for skill in skills {
                black_box(Grammar::for_skill(skill));
            }
        })
    });

    // Validation on the hot path — JSON-schema output check.
    let json_skill = skills
        .iter()
        .find(|s| {
            matches!(
                s.grammar_type,
                kchat_generation::skills::SkillGrammarType::JsonSchema
            )
        })
        .expect("registry has JSON-schema skills");
    let grammar = Grammar::for_skill(json_skill).expect("schema grammar");
    let sample = r#"{"title": "Q3 Planning", "items": ["a", "b"]}"#;
    group.bench_function("validate_json_output", |b| {
        b.iter(|| {
            let _ = GrammarValidator::validate(black_box(sample), black_box(&grammar));
        })
    });
    group.finish();
}

fn bench_mock_stream(c: &mut Criterion) {
    // Measures the streaming/cancel-flag plumbing cost without a model —
    // the overhead every real generation pays per token.
    let backend = kchat_generation::backends::mock::MockBackend::new();
    backend
        .load(&BackendConfig::for_tier(
            BackendType::LlamaCppCpu,
            "bench",
            "bench",
            kchat_core::tier::DeviceTier::High,
            "macos",
        ))
        .unwrap();
    let config = GenerationConfig::default();

    c.bench_function("mock_generate_stream", |b| {
        b.iter(|| {
            let handle = StreamHandle::new();
            black_box(backend.generate_stream(black_box("hello world"), &config, &handle)).unwrap();
            handle.drain_events();
        })
    });
}

criterion_group!(
    benches,
    bench_prompt_build,
    bench_grammar,
    bench_mock_stream
);
criterion_main!(benches);

//! Orchestrator overhead — the full skill pipeline (safety pre-check →
//! prompt build → LoRA resolve → stream wrap → monitor → post-check →
//! memory) minus model inference, using the MockBackend.
//!
//! This isolates the cost the runtime adds around generation — the number
//! that must stay in the low single-digit ms for the pipeline to be
//! negligible against real decode latency.
//!
//! Run: `cargo bench -p kchat-runtime`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kchat_core::tier::DeviceTier;
use kchat_generation::backend::{BackendAdapter, BackendConfig, BackendType};
use kchat_generation::backends::mock::MockBackend;
use kchat_runtime::{Orchestrator, SkillRequest};
use std::sync::Arc;

fn make_backend() -> Arc<MockBackend> {
    let backend = Arc::new(MockBackend::new());
    backend
        .load(&BackendConfig::for_tier(
            BackendType::LlamaCppCpu,
            "bench",
            "bench",
            DeviceTier::High,
            "macos",
        ))
        .unwrap();
    backend
}

fn bench_run_skill(c: &mut Criterion) {
    let orc = Orchestrator::builder()
        .backend(make_backend())
        .tier(DeviceTier::High)
        .build();
    let skill = orc.skills().all()[0].id.clone();

    let req = SkillRequest {
        input: "summarize this document".into(),
        context: "The quarterly review covered revenue and hiring.".into(),
        ..Default::default()
    };

    c.bench_function("run_skill_end_to_end_mock", |b| {
        b.iter(|| {
            let mut run = orc.run_skill(black_box(&skill), req.clone()).expect("run");
            while let Some(ev) = run.events.blocking_recv() {
                if !matches!(ev, kchat_generation::stream::StreamEvent::Token { .. }) {
                    break;
                }
            }
            run.outcome
                .blocking_recv()
                .expect("outcome")
                .expect("skill");
        })
    });
}

fn bench_route(c: &mut Criterion) {
    let orc = Orchestrator::builder()
        .backend(make_backend())
        .tier(DeviceTier::High)
        .build();
    c.bench_function("route_keyword", |b| {
        b.iter(|| {
            black_box(orc.router().route(black_box("please summarize this"), 3));
        })
    });
}

criterion_group!(benches, bench_run_skill, bench_route);
criterion_main!(benches);

//! Retrieval latency — FTS + RRF fusion over the encrypted store at
//! on-device corpus scales. Target: single-digit-ms P95 at ≤10k rows.
//!
//! Run: `cargo bench -p kchat-context`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kchat_context::retrieval::{RetrievalTier, Retriever};
use kchat_context::scope::{ScopeFilter, ScopeId};
use kchat_context::store::{ContextStore, ContextStoreConfig, Evidence, EvidenceId};
use uuid::Uuid;

fn evidence(scope: ScopeId, i: usize) -> Evidence {
    let content = format!(
        "Document {i}: quarterly planning notes covering revenue targets, \
         hiring plans for the mobile team, and follow-up tasks from the \
         review meeting. Action item {i}: update the roadmap deck."
    );
    Evidence {
        id: EvidenceId::new(),
        scope_id: scope,
        content_hash: blake3::hash(content.as_bytes()).to_hex().to_string(),
        encrypted_body: vec![],
        nonce: vec![0u8; 24],
        source_ref: Some(format!("bench/{i}")),
        importance: 5,
        language_tag: Some("en".into()),
        created_at: 1_700_000_000 + i as i64,
        fts_content: content,
    }
}

fn filter(scope: ScopeId) -> ScopeFilter {
    ScopeFilter {
        allowed_scopes: vec![scope],
        denied_scopes: vec![],
        user_id: Uuid::new_v4(),
        roles: vec![],
    }
}

fn bench_retrieve(c: &mut Criterion) {
    let store =
        ContextStore::open_in_memory(&ContextStoreConfig::for_low_tier("bench".into(), [7u8; 32]))
            .unwrap();

    let mut group = c.benchmark_group("retrieve_fts");
    for corpus in [100usize, 1_000, 10_000] {
        // Fresh scope per corpus size keeps each bench's index exact.
        let scope = ScopeId::new();
        for i in 0..corpus {
            store.insert(&evidence(scope, i)).unwrap();
        }
        group.bench_with_input(BenchmarkId::from_parameter(corpus), &corpus, |b, _| {
            let f = filter(scope);
            b.iter(|| {
                let r = Retriever::new(&store, RetrievalTier::Low);
                black_box(r.retrieve("quarterly revenue targets", black_box(&f), 10)).unwrap()
            })
        });
    }
    group.finish();
}

fn bench_insert_indexed(c: &mut Criterion) {
    // Write-path cost: encrypted insert (AEAD + FTS update).
    let store =
        ContextStore::open_in_memory(&ContextStoreConfig::for_low_tier("bench".into(), [9u8; 32]))
            .unwrap();
    let scope = ScopeId::new();
    let mut i = 0usize;
    c.bench_function("insert_evidence_encrypted", |b| {
        b.iter(|| {
            i += 1;
            store.insert(black_box(&evidence(scope, i))).unwrap();
        })
    });
}

criterion_group!(benches, bench_retrieve, bench_insert_indexed);
criterion_main!(benches);

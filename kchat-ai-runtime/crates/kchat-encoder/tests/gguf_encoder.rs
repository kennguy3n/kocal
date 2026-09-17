//! Integration test for the in-process GGUF encoder backend.
//!
//! Runs only with `--features gguf-runtime` AND a GGUF model in
//! `manifest/packs/` (any pack works — the test derives the hidden size from
//! the model and writes a synthetic classifier-heads safetensors file, so a
//! decoder backbone exercises the last-token pooling fallback).

#![cfg(feature = "gguf-runtime")]

use kchat_encoder::{GgufEncoderSession, EMBEDDING_DIM, NUM_SAFETY_CATEGORIES};
use std::path::PathBuf;

/// Locate any GGUF model under manifest/packs (packs are subdirectories).
fn model_path() -> Option<PathBuf> {
    let pack_dirs = [
        PathBuf::from("manifest/packs"),
        PathBuf::from("../manifest/packs"),
        PathBuf::from("../../manifest/packs"),
    ];
    for dir in &pack_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let candidates: Vec<PathBuf> = if path.is_dir() {
                    std::fs::read_dir(&path)
                        .map(|sub| sub.flatten().map(|e| e.path()).collect())
                        .unwrap_or_default()
                } else {
                    vec![path]
                };
                for p in candidates {
                    if p.extension().is_some_and(|e| e == "gguf") {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
}

/// Write a minimal safetensors file with the classifier-head keys the
/// session expects, sized to `hidden`.
fn write_heads(path: &std::path::Path, hidden: usize) {
    let mut header = serde_json::Map::<String, serde_json::Value>::new();
    let mut blobs: Vec<u8> = Vec::new();
    macro_rules! push {
        ($name:literal, $shape:expr) => {{
            let shape: Vec<usize> = $shape;
            let count: usize = shape.iter().product();
            let start = blobs.len();
            blobs.extend(std::iter::repeat(0u8).take(count * 4));
            header.insert(
                $name.to_string(),
                serde_json::json!({
                    "dtype": "F32",
                    "shape": shape,
                    "data_offsets": [start, blobs.len()],
                }),
            );
        }};
    }
    let half = hidden / 2;
    push!("safety_head.0.weight", vec![hidden, hidden]);
    push!("safety_head.0.bias", vec![hidden]);
    push!("safety_head.2.weight", vec![NUM_SAFETY_CATEGORIES, hidden]);
    push!("safety_head.2.bias", vec![NUM_SAFETY_CATEGORIES]);
    push!("embedding_head.0.weight", vec![EMBEDDING_DIM, hidden]);
    push!("embedding_head.0.bias", vec![EMBEDDING_DIM]);
    push!("rerank_head.0.weight", vec![half, hidden]);
    push!("rerank_head.0.bias", vec![half]);
    push!("rerank_head.2.weight", vec![1, half]);
    push!("rerank_head.2.bias", vec![1]);

    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + blobs.len());
    out.extend((header_bytes.len() as u64).to_le_bytes());
    out.extend(header_bytes);
    out.extend(blobs);
    std::fs::write(path, out).unwrap();
}

/// Read the model's hidden size via llama-cpp-2 so the heads file matches.
fn model_hidden_size(model: &std::path::Path) -> Option<usize> {
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::LlamaModel;
    let backend = LlamaBackend::init().ok()?;
    let m = LlamaModel::load_from_file(&backend, model, &LlamaModelParams::default()).ok()?;
    Some(m.n_embd() as usize)
}

#[test]
fn test_inprocess_encoder_end_to_end() {
    let model = match model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: no GGUF model in manifest/packs");
            return;
        }
    };
    let hidden = match model_hidden_size(&model) {
        Some(h) if h > 0 => h,
        _ => {
            eprintln!("Skipping: could not load model for n_embd");
            return;
        }
    };

    let dir = std::env::temp_dir().join(format!("kchat-enc-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let heads_path = dir.join("heads.safetensors");
    write_heads(&heads_path, hidden);

    let session = GgufEncoderSession::new(model.to_str().unwrap(), heads_path.to_str().unwrap(), 2)
        .expect("in-process encoder session should load");

    // embed: hidden-state read + head application + L2 norm
    let emb = session.embed("hello world").expect("embed should work");
    assert_eq!(emb.len(), EMBEDDING_DIM);
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01 || emb.iter().all(|&x| x == 0.0));

    // classify: 17-category logits through the safety head
    let verdict = session.classify("test input").expect("classify");
    assert!(verdict.category < NUM_SAFETY_CATEGORIES as u32);

    // rerank: query–doc pair score
    let score = session.rerank("query", "document").expect("rerank");
    assert!((0.0..=1.0).contains(&score));

    std::fs::remove_dir_all(&dir).ok();
}

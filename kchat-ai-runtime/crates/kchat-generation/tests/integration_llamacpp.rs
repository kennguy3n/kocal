//! Integration test for the llama.cpp backend with a real GGUF model.
//!
//! This test is only run when the `llamacpp-metal` (or equivalent) feature is
//! enabled AND a real model file exists at `manifest/packs/`. It verifies:
//! - Model loading
//! - Text generation
//! - Streaming generation
//! - JSON Schema grammar constraint
//! - Cancellation

#![cfg(feature = "llamacpp")]

use kchat_generation::{
    BackendAdapter, BackendConfig, BackendType, GenerationConfig, Grammar, LlamaCppBackend,
    StreamHandle,
};
use kchat_core::tier::DeviceTier;
use serde_json::json;
use std::path::PathBuf;

/// Locate the test model file.
/// Prefers Ternary-Bonsai models, falls back to any available GGUF in manifest/packs/.
fn model_path() -> Option<PathBuf> {
    let pack_dirs = [
        PathBuf::from("manifest/packs"),
        PathBuf::from("../manifest/packs"),
        PathBuf::from("../../manifest/packs"),
    ];

    // Collect all available GGUF files
    let mut gguf_files: Vec<PathBuf> = Vec::new();
    for dir in &pack_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "gguf") {
                    gguf_files.push(path);
                }
            }
        }
        if !gguf_files.is_empty() {
            break;
        }
    }

    // Prefer standard quantization formats (Q4_K_M, Q8_0) that llama.cpp can reliably load.
    // Q2_0 ternary models may not be supported by all llama.cpp versions.
    gguf_files.sort_by_key(|p| {
        let name = p.file_name().map_or(false, |n| n.to_str().map_or(false, |s| s.contains("Q4_K_M") || s.contains("Q8_0")));
        if name { 0 } else { 1 }
    });

    gguf_files.first().map(|p| p.canonicalize().unwrap_or_else(|_| p.clone()))
}

#[test]
fn test_llamacpp_load_and_generate() {
    let model_path = match model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no model file found");
            return;
        }
    };

    let backend = LlamaCppBackend::new();
    let config = BackendConfig::for_tier(
        BackendType::LlamaCppMetal,
        "test-gguf-model",
        model_path.to_str().unwrap(),
        DeviceTier::Low,
        "macos",
    );

    backend.load(&config).expect("model should load");
    assert!(backend.is_loaded());

    let gen_config = GenerationConfig {
        max_tokens: 32,
        temperature: 0.7,
        ..Default::default()
    };

    let result = backend
        .generate("Hello, how are you?", &gen_config)
        .expect("generation should succeed");

    println!(
        "Generated: {:?} ({} tokens, {} ms TTFT, {:.1} tok/s)",
        result.text,
        result.completion_tokens,
        result.ttft_ms,
        result.tokens_per_second
    );

    assert!(result.completion_tokens > 0, "should generate at least 1 token");
    assert!(!result.text.is_empty(), "text should not be empty");
    assert!(result.ttft_ms < 5000, "TTFT should be < 5s");
    assert!(result.tokens_per_second > 5.0, "should be > 5 tok/s");

    backend.unload().expect("unload should succeed");
    assert!(!backend.is_loaded());
}

#[test]
fn test_llamacpp_streaming() {
    let model_path = match model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no model file found");
            return;
        }
    };

    let backend = LlamaCppBackend::new();
    let config = BackendConfig::for_tier(
        BackendType::LlamaCppMetal,
        "test-gguf-model",
        model_path.to_str().unwrap(),
        DeviceTier::Low,
        "macos",
    );

    backend.load(&config).expect("model should load");

    let stream = StreamHandle::new();
    let gen_config = GenerationConfig {
        max_tokens: 16,
        temperature: 0.7,
        ..Default::default()
    };

    let result = backend
        .generate_stream("Tell me a short joke", &gen_config, &stream)
        .expect("streaming generation should succeed");

    let events = stream.drain_events();
    let token_count = events
        .iter()
        .filter(|e| matches!(e, kchat_generation::StreamEvent::Token { .. }))
        .count();

    println!(
        "Streamed {} tokens, {} events, text: {:?}",
        token_count,
        events.len(),
        result.text
    );

    assert!(token_count > 0, "should stream at least 1 token");
    assert!(result.completion_tokens > 0);

    backend.unload().unwrap();
}

#[test]
fn test_llamacpp_json_schema_grammar() {
    let model_path = match model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no model file found");
            return;
        }
    };

    let backend = LlamaCppBackend::new();
    let config = BackendConfig::for_tier(
        BackendType::LlamaCppMetal,
        "test-gguf-model",
        model_path.to_str().unwrap(),
        DeviceTier::Low,
        "macos",
    );

    backend.load(&config).expect("model should load");

    let schema = json!({
        "type": "object",
        "properties": {
            "greeting": { "type": "string" }
        },
        "required": ["greeting"]
    });

    let gen_config = GenerationConfig {
        max_tokens: 64,
        temperature: 0.3,
        grammar: Some(Grammar::json_schema(schema, 256)),
        ..Default::default()
    };

    let result = backend
        .generate(r#"Generate a JSON object with a "greeting" field."#, &gen_config)
        .expect("grammar generation should succeed");

    println!("Grammar output: {:?}", result.text);

    // The grammar-constrained output should be valid JSON.
    // Note: Grammar enforcement requires the `common` feature of llama-cpp-2,
    // which is enabled by `llamacpp-metal`/`llamacpp-vulkan`/`llamacpp-cuda`.
    // When running without those features, the output may not be valid JSON.
    #[cfg(any(feature = "llamacpp-metal", feature = "llamacpp-vulkan", feature = "llamacpp-cuda"))]
    {
        // Try to extract JSON from the output (may be wrapped in markdown)
        let text = result.text.trim();
        let json_text = if text.starts_with("```") {
            // Extract from markdown code block
            text.lines()
                .skip(1) // skip ```json
                .take_while(|l| !l.starts_with("```"))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            text.to_string()
        };

        let parsed: serde_json::Value = serde_json::from_str(json_text.trim())
            .expect("output should be valid JSON: {json_text}");
        assert!(parsed.get("greeting").is_some(), "should have greeting field");
    }

    backend.unload().unwrap();
}

#[test]
fn test_llamacpp_cancellation() {
    let model_path = match model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping test: no model file found");
            return;
        }
    };

    let backend = LlamaCppBackend::new();
    let config = BackendConfig::for_tier(
        BackendType::LlamaCppMetal,
        "test-gguf-model",
        model_path.to_str().unwrap(),
        DeviceTier::Low,
        "macos",
    );

    backend.load(&config).expect("model should load");

    let stream = StreamHandle::new();

    // Cancel immediately before generation starts
    stream.cancel("test_cancellation");

    let gen_config = GenerationConfig {
        max_tokens: 32,
        temperature: 0.7,
        ..Default::default()
    };

    let result = backend
        .generate_stream("Hello world", &gen_config, &stream)
        .expect("cancelled generation should still return Ok");

    // Should generate 0 tokens since it was cancelled before starting
    assert!(
        result.completion_tokens == 0,
        "should generate 0 tokens when cancelled immediately"
    );

    backend.unload().unwrap();
}

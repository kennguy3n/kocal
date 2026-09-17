//! Integration test for the in-process whisper.cpp ASR backend.
//!
//! Runs only with `--features whispercpp` AND a GGML whisper model at
//! `manifest/packs/whisper-*-ggml/`. Transcribes the JFK speech sample and
//! verifies the transcript content — this is a real end-to-end inference.

#![cfg(feature = "whispercpp")]

use kchat_asr::{WhisperCppTranscriber, WhisperTranscriber};
use std::path::PathBuf;

/// Locate a GGML whisper model under manifest/packs.
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
                    let is_ggml = p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("ggml-") && n.ends_with(".bin"));
                    if is_ggml {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
}

#[test]
fn test_whispercpp_jfk_transcription() {
    let model = match model_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: no ggml-*.bin model in manifest/packs");
            return;
        }
    };

    let wav = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jfk.wav");
    let audio = std::fs::read(&wav).expect("read jfk.wav fixture");

    let transcriber = WhisperCppTranscriber::new(&model, 4, Some("en".into()))
        .expect("whisper.cpp context should load");
    let result = transcriber
        .transcribe(&audio, "audio/wav")
        .expect("transcription should succeed");

    println!("Transcript: {:?}", result.text);
    let lower = result.text.to_lowercase();
    assert!(
        lower.contains("fellow americans") || lower.contains("ask not"),
        "expected JFK speech content, got: {:?}",
        result.text
    );
    assert!(!result.segments.is_empty(), "should produce segments");
}

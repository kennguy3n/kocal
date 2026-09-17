//! Real-model smoke eval — the always-on CI gate for the in-process engine.
//!
//! Unlike `--realworld` (llama-server subprocess over HTTP), this exercises
//! the same in-process `LlamaCppBackend` path that mobile uses — persistent
//! KV session, streaming, cancellation.
//!
//! Run: `cargo run -p kchat-task-suite --features smoke -- --smoke`
//!      (add `--features smoke-metal` for GPU on Apple Silicon)
//!
//! Baselines: `eval/kchat-task-suite/baselines/smoke.json` holds the last
//! known-good metrics. `--smoke` compares against it (25% tolerance) and
//! fails on regression; `--smoke --write-baseline` records a new baseline.
//! Exit code is non-zero on failure or regression.

use crate::report::SuiteReport;

#[cfg(feature = "smoke")]
mod imp {
    use crate::report::{EvalResult, SuiteReport};
    use kchat_core::capability::CapabilityProbe;
    use kchat_core::tier::DeviceTier;
    use kchat_generation::backend::{BackendAdapter, BackendConfig, BackendType, GenerationConfig};
    use kchat_generation::stream::StreamHandle;
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    /// Fractional slowdown tolerated vs. baseline before failing.
    const REGRESSION_TOLERANCE: f64 = 0.25;

    const PROMPTS: &[&str] = &[
        "Explain what a hash map is in two sentences.",
        "Write a haiku about compilers.",
        "List three benefits of on-device inference.",
    ];

    #[derive(Debug, Serialize, Deserialize)]
    struct Baseline {
        model: String,
        tier: String,
        cold_ttft_ms: u64,
        ttft_p50_ms: u64,
        decode_p50_tps: f64,
        tokens: u32,
    }

    fn find_model() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("KCHAT_MODEL_PATH") {
            if Path::new(&p).exists() {
                return Some(p.into());
            }
        }
        let pack_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../manifest/packs");
        let mut ggufs: Vec<PathBuf> = Vec::new();
        collect_gguf(&pack_dir, &mut ggufs, 2);
        // Prefer Bonsai packs — the project's own models.
        ggufs.sort_by_key(|p| {
            let n = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            (!n.to_lowercase().contains("bonsai"), n)
        });
        ggufs.into_iter().next()
    }

    fn collect_gguf(dir: &Path, out: &mut Vec<PathBuf>, depth: usize) {
        if depth == 0 {
            return;
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    collect_gguf(&path, out, depth - 1);
                } else if path.extension().map(|e| e == "gguf").unwrap_or(false) {
                    out.push(path);
                }
            }
        }
    }

    fn baseline_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("baselines/smoke.json")
    }

    fn percentile(sorted: &[u64], p: usize) -> u64 {
        if sorted.is_empty() {
            return 0;
        }
        sorted[(sorted.len() * p / 100).min(sorted.len() - 1)]
    }

    pub fn run(write_baseline: bool) -> SuiteReport {
        // required_pass_rate stays 0 while we may only produce skips —
        // once a model is found the gate becomes strict (all must pass).
        let mut suite = SuiteReport::new("smoke (in-process real model)", 0.0);

        let Some(model) = find_model() else {
            suite.add(EvalResult::skip(
                "model_present",
                "no GGUF in manifest/packs and KCHAT_MODEL_PATH unset",
            ));
            return suite;
        };
        suite.required_pass_rate = 1.0;

        let tier = CapabilityProbe::probe()
            .map(|c| super::eval_perdevice_select_tier(&c))
            .unwrap_or(DeviceTier::Low);

        let backend_type = match std::env::consts::OS {
            "macos" | "ios" => BackendType::LlamaCppMetal,
            "android" | "windows" => BackendType::LlamaCppVulkan,
            _ => BackendType::LlamaCppCpu,
        };
        let config = BackendConfig::for_tier(
            backend_type,
            "smoke",
            model.to_string_lossy().as_ref(),
            tier,
            std::env::consts::OS,
        );

        let backend = kchat_generation::backends::llamacpp::LlamaCppBackend::new();
        if let Err(e) = backend.load(&config) {
            suite.add(EvalResult::fail(
                "model_load",
                format!("{}: {e}", model.display()),
            ));
            return suite;
        }
        let mut meta = HashMap::new();
        meta.insert("model".into(), model.display().to_string());
        meta.insert("tier".into(), format!("{tier:?}"));
        suite.add(EvalResult::pass_with_meta("model_load", 0, meta));

        let gen_cfg = GenerationConfig {
            max_tokens: 64,
            temperature: 0.0,
            ..Default::default()
        };

        // Cold run — includes first-token path with an empty KV session.
        let cold_start = Instant::now();
        let cold = backend
            .generate_stream(PROMPTS[0], &gen_cfg, &StreamHandle::new())
            .expect("cold generation");
        let cold_wall = cold_start.elapsed().as_millis() as u64;

        // Warm runs — same prompts exercise the KV-prefix reuse path.
        let mut ttfts = vec![cold.ttft_ms];
        let mut tps = vec![cold.tokens_per_second];
        let mut total_tokens = cold.completion_tokens;
        for p in &PROMPTS[1..] {
            let r = backend
                .generate_stream(p, &gen_cfg, &StreamHandle::new())
                .expect("warm generation");
            ttfts.push(r.ttft_ms);
            tps.push(r.tokens_per_second);
            total_tokens += r.completion_tokens;
        }
        // Prefix-hit run — identical prompt reuses the whole session prefix.
        let warm_hit = backend
            .generate_stream(PROMPTS[0], &gen_cfg, &StreamHandle::new())
            .expect("prefix-hit generation");

        ttfts.sort();
        let ttft_p50 = percentile(&ttfts, 50);
        let mut tps_sorted: Vec<u64> = tps.iter().map(|t| *t as u64).collect();
        tps_sorted.sort();
        let decode_p50 = percentile(&tps_sorted, 50) as f64;

        let mut meta = HashMap::new();
        meta.insert("cold_ttft_ms".into(), cold.ttft_ms.to_string());
        meta.insert("cold_wall_ms".into(), cold_wall.to_string());
        meta.insert("ttft_p50_ms".into(), ttft_p50.to_string());
        meta.insert("warm_prefix_ttft_ms".into(), warm_hit.ttft_ms.to_string());
        meta.insert("decode_p50_tps".into(), format!("{decode_p50:.1}"));
        meta.insert(
            "ttft_target_ms".into(),
            tier.ttft_p95_target_ms().to_string(),
        );
        suite.add(EvalResult::pass_with_meta("generation_metrics", 0, meta));

        // Hard gate: decode throughput vs. the platform's tier minimum.
        let min_tps = match std::env::consts::OS {
            "macos" | "windows" | "linux" => tier.desktop_decode_p50_min(),
            _ => tier.mobile_decode_p50_min(),
        };
        if decode_p50 >= min_tps {
            suite.add(EvalResult::pass_with_meta(
                "decode_target",
                0,
                [("min_tps".into(), format!("{min_tps:.1}"))].into(),
            ));
        } else {
            suite.add(EvalResult::fail_with_meta(
                "decode_target",
                format!("decode {decode_p50:.1} tok/s < tier minimum {min_tps:.1}"),
                0,
                HashMap::new(),
            ));
        }

        // Prefix reuse must not regress TTFT.
        if warm_hit.ttft_ms <= cold.ttft_ms.max(1) {
            suite.add(EvalResult::pass("prefix_reuse_ttft"));
        } else {
            suite.add(EvalResult::fail(
                "prefix_reuse_ttft",
                format!(
                    "prefix-hit TTFT {}ms > cold {}ms",
                    warm_hit.ttft_ms, cold.ttft_ms
                ),
            ));
        }

        // Baseline regression check.
        let path = baseline_path();
        let current = Baseline {
            model: model.display().to_string(),
            tier: format!("{tier:?}"),
            cold_ttft_ms: cold.ttft_ms,
            ttft_p50_ms: ttft_p50,
            decode_p50_tps: decode_p50,
            tokens: total_tokens,
        };
        if write_baseline {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match serde_json::to_string_pretty(&current)
                .map_err(|e| e.to_string())
                .and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string()))
            {
                Ok(()) => suite.add(EvalResult::pass_with_meta(
                    "baseline_write",
                    0,
                    [("path".into(), path.display().to_string())].into(),
                )),
                Err(e) => suite.add(EvalResult::fail("baseline_write", e)),
            }
        } else if let Ok(raw) = std::fs::read_to_string(&path) {
            match serde_json::from_str::<Baseline>(&raw) {
                Ok(base) => {
                    let slower =
                        |now: f64, was: f64| now > was * (1.0 + REGRESSION_TOLERANCE) && was > 0.0;
                    let decode_floor = base.decode_p50_tps * (1.0 - REGRESSION_TOLERANCE);
                    if decode_p50 >= decode_floor
                        && !slower(ttft_p50 as f64, base.ttft_p50_ms as f64)
                        && !slower(cold.ttft_ms as f64, base.cold_ttft_ms as f64)
                    {
                        suite.add(EvalResult::pass_with_meta(
                            "baseline_regression",
                            0,
                            [
                                ("base_decode".into(), format!("{:.1}", base.decode_p50_tps)),
                                ("base_ttft_p50".into(), base.ttft_p50_ms.to_string()),
                            ]
                            .into(),
                        ));
                    } else {
                        suite.add(EvalResult::fail(
                            "baseline_regression",
                            format!(
                                "regression vs baseline: decode {decode_p50:.1} vs {:.1}, ttft_p50 {ttft_p50} vs {}",
                                base.decode_p50_tps, base.ttft_p50_ms
                            ),
                        ));
                    }
                }
                Err(e) => suite.add(EvalResult::fail(
                    "baseline_regression",
                    format!("bad baseline: {e}"),
                )),
            }
        } else {
            println!(
                "  note: no baseline at {} — run --smoke --write-baseline to record one",
                path.display()
            );
        }

        suite
    }
}

/// Fallback when the `smoke` feature (in-process llamacpp) isn't compiled in.
#[cfg(not(feature = "smoke"))]
mod imp {
    use crate::report::{EvalResult, SuiteReport};

    pub fn run(_write_baseline: bool) -> SuiteReport {
        // Skip-only suite — nothing required when the feature is off.
        let mut suite = SuiteReport::new("smoke (in-process real model)", 0.0);
        suite.add(EvalResult::skip(
            "smoke",
            "requires --features smoke (or smoke-metal for GPU)",
        ));
        suite
    }
}

/// Probe-based tier selection shared with eval_perdevice's profile logic.
#[cfg(feature = "smoke")]
fn eval_perdevice_select_tier(
    caps: &kchat_core::capability::DeviceCapabilities,
) -> kchat_core::tier::DeviceTier {
    kchat_core::tier::TierSelection::select(caps).unwrap_or(kchat_core::tier::DeviceTier::Low)
}

pub fn run(write_baseline: bool) -> SuiteReport {
    imp::run(write_baseline)
}

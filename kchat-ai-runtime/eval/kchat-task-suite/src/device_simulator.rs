//! Device simulator — runs each device profile through the full runtime
//! decision tree and prints a detailed simulation report.
//!
//! Shows tier selection, model selection, backend selection, resource budgets,
//! performance targets, and transition behavior (thermal, battery, background)
//! for all 12 device profiles.
//!
//! Usage: cargo run -p kchat-task-suite -- --simulate

use crate::eval_device_profile::{all_profiles, select_model_for_tier, DeviceProfile};
use kchat_core::capability::{AppState, DeviceCapabilities, ThermalState};
use kchat_core::registry::{MinTier, ModelRegistry};
use kchat_core::scheduler::{Scheduler, SchedulerConfig};
use kchat_core::tier::{DeviceTier, TierBudget, TierSelection};
use kchat_generation::BackendType;

/// Run the full device simulation and print results.
pub fn run() {
    let profiles = all_profiles();
    let registry = ModelRegistry::default_registry();

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║          KCHAT AI RUNTIME — DEVICE SIMULATION REPORT                         ║");
    println!("║          12 Profiles × Full Decision Tree                                    ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let mut total_pass = 0usize;
    let mut total_checks = 0usize;

    for (idx, profile) in profiles.iter().enumerate() {
        let (pass, total) = simulate_profile(profile, &registry, idx + 1, profiles.len());
        total_pass += pass;
        total_checks += total;
    }

    // Print summary table
    print_summary_table(&profiles);

    // Print transition matrix
    print_transition_matrix(&profiles);

    // Print scheduler simulation
    print_scheduler_simulation(&profiles);

    // Final summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  SIMULATION COMPLETE: {}/{} checks passed ({:.1}%){}",
        total_pass,
        total_checks,
        (total_pass as f64 / total_checks as f64) * 100.0,
        " ".repeat(80 - 47 - format!("{}/{} checks passed ({:.1}%)", total_pass, total_checks, (total_pass as f64 / total_checks as f64) * 100.0).len())
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

fn simulate_profile(
    profile: &DeviceProfile,
    registry: &ModelRegistry,
    idx: usize,
    total: usize,
) -> (usize, usize) {
    let caps = profile.to_caps();
    let mut pass = 0;
    let mut checks = 0;

    let tier_color = match profile.expected_tier {
        DeviceTier::High => "\x1b[32m", // green
        DeviceTier::Medium => "\x1b[33m", // yellow
        DeviceTier::Low => "\x1b[31m", // red
    };
    let reset = "\x1b[0m";

    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!(
        "│ [{}/{}] {} {}{}",
        idx,
        total,
        profile.name,
        tier_color,
        format!("({:?})", profile.expected_tier).dim()
    );
    println!("├──────────────────────────────────────────────────────────────────────────────┤");

    // Hardware specs
    println!(
        "│  Platform:      {:<12}  CPU: {:<10} ({} cores, {:?} perf)",
        profile.platform,
        profile.cpu_arch,
        profile.cpu_cores,
        profile.performance_cores
    );
    println!(
        "│  Memory:        {:>6} MB physical, {:>6} MB safe allocatable",
        profile.physical_memory_mb,
        profile.safe_allocatable_mb
    );
    println!(
        "│  GPU:           {:<12}  NPU: {:<12}",
        format!("{:?}", profile.gpu_backend),
        format!("{:?}", profile.npu_provider)
    );
    println!(
        "│  Storage:       {:>4} GB free    Battery: {:<12} Charger: {}",
        profile.free_storage_gb,
        match profile.battery_level {
            Some(l) => format!("{}%", l),
            None => "N/A".to_string(),
        },
        if profile.on_charger { "yes" } else { "no" }
    );
    println!(
        "│  Thermal:       {:<12}  App: {:<12}   Network: {}",
        format!("{:?}", profile.thermal_state),
        format!("{:?}", profile.app_state),
        if profile.unmetered_network { "unmetered" } else { "metered" }
    );

    // --- Tier Selection ---
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  TIER SELECTION                                                              │");
    let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
    let tier_ok = tier == profile.expected_tier;
    checks += 1;
    if tier_ok { pass += 1; }

    let tier_str = format!("{:?}", tier);
    let expected_str = format!("{:?}", profile.expected_tier);
    let tier_label = format!("{}{}{}", tier_color, tier_str, reset);
    let status = if tier_ok { "✓" } else { "✗" };
    println!(
        "│    {} Selected tier: {:<8}  Expected: {:<8}",
        status,
        tier_label,
        format!("{}{}{}", tier_color, expected_str, reset)
    );

    // Show memory threshold reasoning
    let safe_mb = caps.safe_allocatable_memory / (1024 * 1024);
    let threshold_info = match profile.platform {
        "ios" | "android" => {
            if safe_mb >= 6000 {
                format!("safe_mb={} ≥ 6000 → High", safe_mb)
            } else if safe_mb >= 3500 {
                format!("safe_mb={} ≥ 3500 → Medium", safe_mb)
            } else {
                format!("safe_mb={} < 3500 → Low", safe_mb)
            }
        }
        "macos" | "windows" => {
            if safe_mb >= 20000 {
                format!("safe_mb={} ≥ 20000 → High", safe_mb)
            } else if safe_mb >= 10000 {
                format!("safe_mb={} ≥ 10000 → Medium", safe_mb)
            } else {
                format!("safe_mb={} < 10000 → Low", safe_mb)
            }
        }
        _ => format!("unknown platform → Low"),
    };
    println!("│      └─ {}", threshold_info);

    // --- Model Selection ---
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  MODEL SELECTION                                                             │");
    let model = select_model_for_tier(tier);
    let model_ok = model == profile.expected_model_pack;
    checks += 1;
    if model_ok { pass += 1; }

    let model_str = model.unwrap_or("none (deterministic-only)");
    let status = if model_ok { "✓" } else { "✗" };
    println!(
        "│    {} Generative model: {}",
        status,
        model_str
    );

    if let Some(pack_id) = model {
        if let Some(entry) = registry.find(pack_id) {
            println!(
                "│      └─ pack: {} v{}, size: {} MB, quant: {}, min_tier: {:?}",
                entry.pack_id,
                entry.version,
                entry.size_bytes / (1024 * 1024),
                entry.quantization,
                entry.min_tier
            );
            println!(
                "│      └─ tasks: [{}]  languages: [{}]",
                entry.task_capabilities.join(", "),
                entry.languages.join(", ")
            );
        }
    } else {
        println!("│      └─ No generative model — safety pipeline runs deterministically");
    }

    // Also show available non-generative models for this tier
    let min_tier = match tier {
        DeviceTier::Low => MinTier::Low,
        DeviceTier::Medium => MinTier::Medium,
        DeviceTier::High => MinTier::High,
    };
    let embeddings = registry.find_for_task("embed", min_tier);
    let safety = registry.find_for_task("safety", min_tier);
    if !embeddings.is_empty() || !safety.is_empty() {
        let mut available: Vec<String> = Vec::new();
        for e in &embeddings {
            available.push(format!("{} ({}MB)", e.pack_id, e.size_bytes / (1024 * 1024)));
        }
        for s in &safety {
            available.push(format!("{} ({}MB)", s.pack_id, s.size_bytes / (1024 * 1024)));
        }
        println!("│      └─ Non-generative packs available: {}", available.join(", "));
    }

    // --- Backend Selection ---
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  BACKEND SELECTION                                                           │");
    let backend = BackendType::select(&caps.platform, tier);
    let backend_ok = backend.map(|b| b.as_str()) == profile.expected_backend;
    checks += 1;
    if backend_ok { pass += 1; }

    let backend_str = backend.map(|b| b.as_str().to_string()).unwrap_or("none".into());
    let status = if backend_ok { "✓" } else { "✗" };
    println!(
        "│    {} Backend: {}",
        status,
        backend_str
    );
    if backend.is_some() {
        println!("│      └─ GPU acceleration: {:?} on {}", profile.gpu_backend, profile.platform);
    } else {
        println!("│      └─ No generative backend (Low tier or no GPU)");
    }

    // --- Resource Budget ---
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  RESOURCE BUDGET                                                             │");
    let budget = TierBudget::for_tier(tier, profile.platform);
    let budget_ok = budget.context_cap == tier.context_cap()
        && budget.output_token_range == tier.output_cap()
        && budget.max_memory_bytes == tier.peak_memory_budget(profile.platform)
        && budget.max_perf_cores == tier.max_perf_cores();
    checks += 1;
    if budget_ok { pass += 1; }

    let status = if budget_ok { "✓" } else { "✗" };
    println!(
        "│    {} Context cap:    {:>6} tokens",
        status,
        budget.context_cap
    );
    println!(
        "│      Output range:    {}-{} tokens",
        budget.output_token_range.0,
        budget.output_token_range.1
    );
    println!(
        "│      Max memory:      {:>6} MB ({:.1} GB)",
        budget.max_memory_bytes / (1024 * 1024),
        budget.max_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "│      Max perf cores:  {}",
        budget.max_perf_cores
    );
    println!(
        "│      Idle unload:     {}s",
        budget.idle_unload_secs
    );

    // --- Performance Targets ---
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  PERFORMANCE TARGETS                                                         │");
    let is_mobile = profile.platform == "ios" || profile.platform == "android";
    let ttft = tier.ttft_p95_target_ms();
    let decode = if is_mobile {
        tier.mobile_decode_p50_min()
    } else {
        tier.desktop_decode_p50_min()
    };
    println!(
        "│    TTFT P95 target:     {:>5} ms",
        ttft
    );
    println!(
        "│    Decode P50 min:      {:>5.1} tok/s ({})",
        decode,
        if is_mobile { "mobile" } else { "desktop" }
    );

    // --- Memory Budget Analysis ---
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  MEMORY BUDGET ANALYSIS                                                      │");
    let safe_ai = caps.safe_ai_budget();
    let peak = tier.peak_memory_budget(profile.platform);
    let headroom = safe_ai as i64 - peak as i64;
    let headroom_pct = if safe_ai > 0 {
        (headroom as f64 / safe_ai as f64) * 100.0
    } else {
        0.0
    };

    let mem_ok = peak <= safe_ai;
    checks += 1;
    if mem_ok { pass += 1; }

    let status = if mem_ok { "✓" } else { "✗" };
    println!(
        "│    {} Safe AI budget:    {:>6} MB ({:.1} GB)",
        status,
        safe_ai / (1024 * 1024),
        safe_ai as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!(
        "│      Peak budget:      {:>6} MB ({:.1} GB)",
        peak / (1024 * 1024),
        peak as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    let headroom_color = if headroom > 0 { "\x1b[32m" } else { "\x1b[31m" };
    println!(
        "│      Headroom:         {}{:>6} MB ({:.1}%){}",
        headroom_color,
        headroom / (1024 * 1024),
        headroom_pct,
        reset
    );

    // Model fit check
    if model.is_some() {
        let model_size = 500 * 1024 * 1024; // Q4 model ~500MB
        let fits = model_size <= peak;
        checks += 1;
        if fits { pass += 1; }
        let status = if fits { "✓" } else { "✗" };
        println!(
            "│    {} Model fits:        {} MB / {} MB budget",
            status,
            model_size / (1024 * 1024),
            peak / (1024 * 1024)
        );
    }

    // --- Transition Simulation ---
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  TRANSITION SIMULATION                                                       │");

    // Thermal transitions
    let serious_tier = TierSelection::select(&profile.with_thermal(ThermalState::Serious))
        .unwrap_or(DeviceTier::Low);
    let critical_tier = TierSelection::select(&profile.with_thermal(ThermalState::Critical))
        .unwrap_or(DeviceTier::Low);
    let fair_tier = TierSelection::select(&profile.with_thermal(ThermalState::Fair))
        .unwrap_or(DeviceTier::Low);

    let thermal_ok = serious_tier == TierSelection::apply_thermal_downgrade_public(tier, ThermalState::Serious)
        && critical_tier == DeviceTier::Low
        && fair_tier == tier;
    checks += 1;
    if thermal_ok { pass += 1; }

    let status = if thermal_ok { "✓" } else { "✗" };
    println!("│    {} Thermal transitions:", status);
    println!(
        "│      Nominal → {:?}  |  Fair → {:?}  |  Serious → {:?}  |  Critical → {:?}",
        tier, fair_tier, serious_tier, critical_tier
    );

    // Battery transitions
    let low_battery = TierSelection::re_evaluate(tier, &profile.with_battery(10, false))
        .unwrap_or(DeviceTier::Low);
    let low_charging = TierSelection::re_evaluate(tier, &profile.with_battery(10, true))
        .unwrap_or(DeviceTier::Low);
    let full_battery = TierSelection::re_evaluate(tier, &profile.with_battery(90, false))
        .unwrap_or(DeviceTier::Low);

    let battery_ok = low_battery != tier || tier == DeviceTier::Low;
    checks += 1;
    if battery_ok { pass += 1; }

    let status = if battery_ok { "✓" } else { "✗" };
    println!("│    {} Battery transitions:", status);
    println!(
        "│      10% no charger → {:?}  |  10% charging → {:?}  |  90% → {:?}",
        low_battery, low_charging, full_battery
    );

    // Background transitions (mobile only)
    if profile.platform == "ios" || profile.platform == "android" {
        let bg_tier = TierSelection::re_evaluate(tier, &profile.with_app_state(AppState::Background))
            .unwrap_or(DeviceTier::Low);
        let fg_tier = TierSelection::re_evaluate(tier, &profile.with_app_state(AppState::Foreground))
            .unwrap_or(DeviceTier::Low);

        let bg_ok = bg_tier == DeviceTier::Low && fg_tier == tier;
        checks += 1;
        if bg_ok { pass += 1; }

        let status = if bg_ok { "✓" } else { "✗" };
        println!("│    {} Background transitions:", status);
        println!(
            "│      Foreground → {:?}  |  Background → {:?} (generative blocked)",
            fg_tier, bg_tier
        );
    }

    // --- Scheduler Simulation ---
    println!("├──────────────────────────────────────────────────────────────────────────────┤");
    println!("│  SCHEDULER SIMULATION                                                        │");
    let scheduler = Scheduler::new(SchedulerConfig::default(), tier);
    let requires_gen = tier != DeviceTier::Low;
    let job_result = scheduler.request_job(&caps, requires_gen, peak);

    let sched_ok = job_result.is_ok() || (!requires_gen && job_result.is_err());
    checks += 1;
    if sched_ok { pass += 1; }

    let status = if sched_ok { "✓" } else { "✗" };
    match &job_result {
        Ok(budget) => {
            println!(
                "│    {} Job admitted: tier={:?}, context={}, output={}-{}",
                status,
                budget.tier,
                budget.context_cap,
                budget.output_token_range.0,
                budget.output_token_range.1
            );
        }
        Err(e) => {
            println!(
                "│    {} Job rejected: {}",
                status,
                e
            );
        }
    }

    // Test concurrent job rejection
    if job_result.is_ok() {
        let job2 = scheduler.request_job(&caps, requires_gen, peak);
        let concurrent_ok = job2.is_err();
        checks += 1;
        if concurrent_ok { pass += 1; }
        let status = if concurrent_ok { "✓" } else { "✗" };
        println!(
            "│    {} Concurrent job rejected: {}",
            status,
            if concurrent_ok { "yes (max_concurrent=1)" } else { "no (BUG!)" }
        );
        scheduler.complete_job();
    }

    // Test kill switch
    let scheduler2 = Scheduler::new(SchedulerConfig::default(), tier);
    scheduler2.activate_kill_switch();
    let kill_result = scheduler2.request_job(&caps, requires_gen, peak);
    let kill_ok = kill_result.is_err();
    checks += 1;
    if kill_ok { pass += 1; }
    let status = if kill_ok { "✓" } else { "✗" };
    println!(
        "│    {} Kill switch blocks jobs: {}",
        status,
        if kill_ok { "yes" } else { "no (BUG!)" }
    );

    println!("└──────────────────────────────────────────────────────────────────────────────┘\n");

    (pass, checks)
}

fn print_summary_table(profiles: &[DeviceProfile]) {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY TABLE — All 12 Device Profiles                                      ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:<38} {:<8} {:<10} {:<14} {:<10} {:<8} ║",
        "Device", "Tier", "Backend", "Model", "Ctx", "Mem(MB)");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");

    for p in profiles {
        let caps = p.to_caps();
        let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
        let backend = BackendType::select(&caps.platform, tier)
            .map(|b| b.as_str().to_string())
            .unwrap_or("none".into());
        let model = select_model_for_tier(tier).unwrap_or("—");
        let budget = TierBudget::for_tier(tier, p.platform);

        let tier_str = format!("{:?}", tier);
        let name = if p.name.len() > 36 {
            format!("{}...", &p.name[..33])
        } else {
            p.name.to_string()
        };

        println!("║ {:<38} {:<8} {:<10} {:<14} {:<10} {:>8} ║",
            name,
            tier_str,
            backend,
            model,
            budget.context_cap,
            budget.max_memory_bytes / (1024 * 1024),
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

fn print_transition_matrix(profiles: &[DeviceProfile]) {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  TRANSITION MATRIX — Tier under Stress Conditions                            ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:<28} {:<8} {:<8} {:<8} {:<8} {:<10} {:<10} ║",
        "Device", "Nominal", "Fair", "Serious", "Critical", "LowBatt", "Bgnd(mob)");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");

    for p in profiles {
        let caps = p.to_caps();
        let nominal = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
        let fair = TierSelection::select(&p.with_thermal(ThermalState::Fair)).unwrap_or(DeviceTier::Low);
        let serious = TierSelection::select(&p.with_thermal(ThermalState::Serious)).unwrap_or(DeviceTier::Low);
        let critical = TierSelection::select(&p.with_thermal(ThermalState::Critical)).unwrap_or(DeviceTier::Low);
        let low_batt = TierSelection::re_evaluate(nominal, &p.with_battery(10, false)).unwrap_or(DeviceTier::Low);
        let bgnd = if p.platform == "ios" || p.platform == "android" {
            TierSelection::re_evaluate(nominal, &p.with_app_state(AppState::Background)).unwrap_or(DeviceTier::Low)
        } else {
            nominal // N/A for desktop
        };

        let name = if p.name.len() > 26 {
            format!("{}...", &p.name[..23])
        } else {
            p.name.to_string()
        };

        let bgnd_str = if p.platform == "ios" || p.platform == "android" {
            format!("{:?}", bgnd)
        } else {
            "N/A".to_string()
        };

        println!("║ {:<28} {:<8} {:<8} {:<8} {:<8} {:<10} {:<10} ║",
            name,
            format!("{:?}", nominal),
            format!("{:?}", fair),
            format!("{:?}", serious),
            format!("{:?}", critical),
            format!("{:?}", low_batt),
            bgnd_str,
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

fn print_scheduler_simulation(profiles: &[DeviceProfile]) {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║  SCHEDULER SIMULATION — Job Admission per Profile                            ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ {:<28} {:<8} {:<10} {:<10} {:<10} {:<14} ║",
        "Device", "Tier", "Gen Job", "Det Job", "Kill Sw", "Concurrent");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");

    for p in profiles {
        let caps = p.to_caps();
        let tier = TierSelection::select(&caps).unwrap_or(DeviceTier::Low);
        let peak = tier.peak_memory_budget(p.platform);
        let requires_gen = tier != DeviceTier::Low;

        // Test generative job
        let sched1 = Scheduler::new(SchedulerConfig::default(), tier);
        let gen_result = sched1.request_job(&caps, true, peak);
        let gen_str = if gen_result.is_ok() { "admit" } else { "reject" };

        // Test deterministic job (non-generative)
        let sched2 = Scheduler::new(SchedulerConfig::default(), tier);
        let det_result = sched2.request_job(&caps, false, peak / 4);
        let det_str = if det_result.is_ok() { "admit" } else { "reject" };

        // Test kill switch
        let sched3 = Scheduler::new(SchedulerConfig::default(), tier);
        sched3.activate_kill_switch();
        let kill_result = sched3.request_job(&caps, requires_gen, peak);
        let kill_str = if kill_result.is_err() { "blocked" } else { "LEAK!" };

        // Test concurrent
        let sched4 = Scheduler::new(SchedulerConfig::default(), tier);
        let _ = sched4.request_job(&caps, requires_gen, peak);
        let conc_result = sched4.request_job(&caps, requires_gen, peak);
        let conc_str = if conc_result.is_err() { "rejected" } else { "LEAK!" };

        let name = if p.name.len() > 26 {
            format!("{}...", &p.name[..23])
        } else {
            p.name.to_string()
        };

        println!("║ {:<28} {:<8} {:<10} {:<10} {:<10} {:<14} ║",
            name,
            format!("{:?}", tier),
            gen_str,
            det_str,
            kill_str,
            conc_str,
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
}

// Helper trait extension for testing
trait ThermalDowngradeExt {
    fn apply_thermal_downgrade_public(tier: DeviceTier, thermal: ThermalState) -> DeviceTier;
}

impl ThermalDowngradeExt for TierSelection {
    fn apply_thermal_downgrade_public(tier: DeviceTier, thermal: ThermalState) -> DeviceTier {
        match thermal {
            ThermalState::Nominal | ThermalState::Fair => tier,
            ThermalState::Serious => match tier {
                DeviceTier::High => DeviceTier::Medium,
                DeviceTier::Medium => DeviceTier::Low,
                DeviceTier::Low => DeviceTier::Low,
            },
            ThermalState::Critical => DeviceTier::Low,
        }
    }
}

// Color helper
trait ColorExt {
    fn dim(&self) -> String;
}

impl ColorExt for String {
    fn dim(&self) -> String {
        format!("\x1b[2m{}\x1b[0m", self)
    }
}

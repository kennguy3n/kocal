//! Integration evaluation suite — end-to-end flow tests.
//!
//! Tests the full pipeline: safety → context → generation → action.

use crate::report::{EvalResult, SuiteReport};
use kchat_bindings::{FfiSafetyAction, KChatAiRuntime};
use kchat_core::tier::{DeviceTier, TierSelection};
use kchat_core::capability::{AppState, DeviceCapabilities, GpuBackend, NpuProvider, ThermalState};

pub fn run() -> SuiteReport {
    let mut suite = SuiteReport::new("Integration Eval Suite", 0.90);

    suite.add(test_end_to_end_safety());
    suite.add(test_tier_selection_high());
    suite.add(test_tier_selection_low());
    suite.add(test_tier_thermal_downgrade());
    suite.add(test_tier_policy_cap());
    suite.add(test_manifest_null_digest_rejected());

    suite
}

fn test_end_to_end_safety() -> EvalResult {
    let runtime = KChatAiRuntime::new("ios");

    // Safe text
    let safe = runtime.classify_safety("Hello, how are you?", false);
    if safe.action != FfiSafetyAction::Allow {
        return EvalResult::fail("end_to_end_safety", "safe text was not allowed");
    }

    // PII text
    let pii = runtime.classify_safety("my card is 4111 1111 1111 1111", false);
    if pii.action != FfiSafetyAction::Redact {
        return EvalResult::fail("end_to_end_safety", "PII was not redacted");
    }

    EvalResult::pass("end_to_end_safety")
}

fn test_tier_selection_high() -> EvalResult {
    let caps = DeviceCapabilities {
        platform: "ios".into(),
        physical_memory: 8 * 1024 * 1024 * 1024,
        safe_allocatable_memory: 6 * 1024 * 1024 * 1024,
        cpu_arch: "aarch64".into(),
        cpu_cores: 8,
        performance_cores: Some(4),
        isa_features: vec!["neon".into()],
        gpu_backend: GpuBackend::Metal,
        npu_provider: NpuProvider::AppleNe,
        free_storage: 10 * 1024 * 1024 * 1024,
        battery_level: Some(80),
        on_charger: true,
        thermal_state: ThermalState::Nominal,
        app_state: AppState::Foreground,
        unmetered_network: true,
    };

    let tier = TierSelection::select(&caps).unwrap();
    if tier == DeviceTier::High {
        EvalResult::pass("tier_selection_high")
    } else {
        EvalResult::fail("tier_selection_high", format!("expected High, got {:?}", tier))
    }
}

fn test_tier_selection_low() -> EvalResult {
    let caps = DeviceCapabilities {
        platform: "ios".into(),
        physical_memory: 3 * 1024 * 1024 * 1024,
        safe_allocatable_memory: 2 * 1024 * 1024 * 1024,
        cpu_arch: "aarch64".into(),
        cpu_cores: 4,
        performance_cores: None,
        isa_features: vec!["neon".into()],
        gpu_backend: GpuBackend::None,
        npu_provider: NpuProvider::None,
        free_storage: 2 * 1024 * 1024 * 1024,
        battery_level: Some(80),
        on_charger: true,
        thermal_state: ThermalState::Nominal,
        app_state: AppState::Foreground,
        unmetered_network: true,
    };

    let tier = TierSelection::select(&caps).unwrap();
    if tier == DeviceTier::Low {
        EvalResult::pass("tier_selection_low")
    } else {
        EvalResult::fail("tier_selection_low", format!("expected Low, got {:?}", tier))
    }
}

fn test_tier_thermal_downgrade() -> EvalResult {
    let caps = DeviceCapabilities {
        platform: "ios".into(),
        physical_memory: 8 * 1024 * 1024 * 1024,
        safe_allocatable_memory: 6 * 1024 * 1024 * 1024,
        cpu_arch: "aarch64".into(),
        cpu_cores: 8,
        performance_cores: Some(4),
        isa_features: vec!["neon".into()],
        gpu_backend: GpuBackend::Metal,
        npu_provider: NpuProvider::AppleNe,
        free_storage: 10 * 1024 * 1024 * 1024,
        battery_level: Some(80),
        on_charger: true,
        thermal_state: ThermalState::Critical,
        app_state: AppState::Foreground,
        unmetered_network: true,
    };

    let tier = TierSelection::select(&caps).unwrap();
    if tier == DeviceTier::Low {
        EvalResult::pass("tier_thermal_downgrade")
    } else {
        EvalResult::fail("tier_thermal_downgrade", format!("expected Low (critical thermal), got {:?}", tier))
    }
}

fn test_tier_policy_cap() -> EvalResult {
    // Policy cap should lower tier but never elevate
    let capped = TierSelection::apply_policy_cap(DeviceTier::High, Some(DeviceTier::Low));
    if capped == DeviceTier::Low {
        EvalResult::pass("tier_policy_cap")
    } else {
        EvalResult::fail("tier_policy_cap", format!("expected Low, got {:?}", capped))
    }
}

fn test_manifest_null_digest_rejected() -> EvalResult {
    use kchat_core::manifest::{ManifestSignature, ModelPackManifest, PackChunk, PackType, SignedManifest};
    use ed25519_dalek::{SigningKey, Signer};
    use rand::rngs::OsRng;

    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    let pk_hex = hex::encode(verifying_key.to_bytes());

    let pack = ModelPackManifest {
        pack_id: "test".into(),
        version: "1.0".into(),
        pack_type: PackType::GenerativeModel,
        content_sha256: "0".repeat(64), // NULL digest — release blocker
        chunks: vec![PackChunk {
            index: 0,
            sha256: "b".repeat(64),
            size_bytes: 1024,
        }],
        total_size_bytes: 1024,
        source_repo: "test".into(),
        source_revision: "abc".into(),
        quantization_recipe: "Q4".into(),
        build_env: "test".into(),
        license: "Apache-2.0".into(),
        product_use_approved: true,
        runtime_abi: "v1".into(),
        required_backends: vec!["cpu".into()],
        min_app_version: "1.0".into(),
        min_os_version: "17.0".into(),
        task_capabilities: vec![],
        eligible_tiers: vec![],
        peak_working_set_bytes: 1024,
        eval_suite_version: "v1".into(),
        eval_results_digest: "c".repeat(64),
        rollout_cohort: "internal".into(),
        expires_at: "2027-01-01T00:00:00Z".into(),
        kill_switch: false,
        rollback_target: None,
    };

    let mut manifest = SignedManifest {
        schema_version: 1,
        environment: "production".into(),
        manifest_id: "test".into(),
        generated_at: "2026-01-01T00:00:00Z".into(),
        packs: vec![pack],
        signature: ManifestSignature {
            public_key: pk_hex.clone(),
            signature: "0".repeat(128),
        },
    };

    // Sign
    let message = manifest.canonical_message().unwrap();
    let sig = signing_key.sign(&message);
    manifest.signature.signature = hex::encode(sig.to_bytes());

    let result = manifest.verify(&pk_hex);
    if result.is_err() {
        EvalResult::pass("manifest_null_digest_rejected")
    } else {
        EvalResult::fail("manifest_null_digest_rejected", "null digest was accepted")
    }
}

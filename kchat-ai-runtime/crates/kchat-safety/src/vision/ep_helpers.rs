//! EP selection helpers for the vision MobileCLIP session.
//!
//! Mirrors the EP-selection logic in `kchat-encoder/src/session.rs`,
//! using [`kchat_core::ep`] to pick the platform-appropriate
//! execution provider (CoreML on Apple, DirectML on Windows,
//! NNAPI on Android) with automatic CPU fallback.

/// Build the ort execution-provider dispatch list for the current host
/// using [`kchat_core::ep`] selection.
///
/// Returns a `Vec<ExecutionProviderDispatch>` suitable for
/// [`ort::session::SessionBuilder::with_execution_providers`]. The
/// list is ordered most-preferred-first; ort's runtime handles
/// silent fallback to CPU if an accelerator EP fails to register.
pub(crate) fn build_ort_eps_for_host() -> Vec<ort::execution_providers::ExecutionProviderDispatch> {
    use kchat_core::ep::{EpDeviceCapabilities, EpFallbackChain, Platform};

    // Detect host platform from cfg.
    let (os, arch) = {
        #[cfg(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios")))]
        {
            (Platform::MacOs, kchat_core::ep::Arch::Aarch64)
        }
        #[cfg(all(not(all(target_arch = "aarch64", any(target_os = "macos", target_os = "ios"))), target_os = "macos"))]
        {
            (Platform::MacOs, kchat_core::ep::Arch::X86_64)
        }
        #[cfg(target_os = "ios")]
        {
            (Platform::Ios, kchat_core::ep::Arch::Aarch64)
        }
        #[cfg(target_os = "android")]
        {
            (Platform::Android, kchat_core::ep::Arch::Aarch64)
        }
        #[cfg(target_os = "windows")]
        {
            (
                Platform::Windows,
                if cfg!(target_arch = "aarch64") {
                    kchat_core::ep::Arch::Aarch64
                } else {
                    kchat_core::ep::Arch::X86_64
                },
            )
        }
        #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "openbsd"))]
        {
            (
                Platform::Linux,
                if cfg!(target_arch = "aarch64") {
                    kchat_core::ep::Arch::Aarch64
                } else {
                    kchat_core::ep::Arch::X86_64
                },
            )
        }
        #[cfg(not(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "android",
            target_os = "windows",
            target_os = "linux",
            target_os = "freebsd",
            target_os = "openbsd",
        )))]
        {
            (Platform::Unknown, kchat_core::ep::Arch::Other)
        }
    };

    // Build capabilities — assume accelerator is present on Apple Silicon
    // and Windows with GPU. Production code should probe via
    // `kchat_core::capability::CapabilityProbe::probe()` and bridge through
    // `EpDeviceCapabilities::from_full_caps`.
    let caps = match os {
        Platform::MacOs | Platform::Ios => EpDeviceCapabilities::apple_silicon_mac(),
        Platform::Android => EpDeviceCapabilities::android_with_npu(),
        Platform::Windows => EpDeviceCapabilities::windows_with_gpu("auto"),
        _ => EpDeviceCapabilities::cpu_only(os, arch),
    };

    let chain = EpFallbackChain::for_platform(os, &caps);
    chain
        .as_slice()
        .iter()
        .filter_map(|ep| ep_to_ort_dispatch(*ep))
        .collect()
}

/// Map a [`kchat_core::ep::ExecutionProvider`] to the corresponding
/// ort execution-provider dispatch object.
fn ep_to_ort_dispatch(
    ep: kchat_core::ep::ExecutionProvider,
) -> Option<ort::execution_providers::ExecutionProviderDispatch> {
    use ort::execution_providers::{
        CPUExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
        NNAPIExecutionProvider,
    };

    match ep {
        kchat_core::ep::ExecutionProvider::CoreMl => {
            Some(CoreMLExecutionProvider::default().build())
        }
        kchat_core::ep::ExecutionProvider::Nnapi => {
            Some(NNAPIExecutionProvider::default().build())
        }
        kchat_core::ep::ExecutionProvider::DirectMl => {
            Some(DirectMLExecutionProvider::default().build())
        }
        kchat_core::ep::ExecutionProvider::MetalPerformanceShaders => None,
        kchat_core::ep::ExecutionProvider::Cpu => {
            Some(CPUExecutionProvider::default().build())
        }
    }
}

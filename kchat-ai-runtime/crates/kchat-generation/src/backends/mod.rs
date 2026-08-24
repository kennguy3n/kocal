//! Backend adapters — concrete implementations of [`BackendAdapter`].
//!
//! This module provides the actual inference backends used by the generative
//! plane. The primary backend is [`llamacpp`], which wraps the `llama-cpp-2`
//! crate for in-process llama.cpp inference with Metal/Vulkan/CUDA/CPU
//! acceleration. A [`mock`] backend is always available for testing on
//! platforms without a real model.

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(feature = "mlx")]
pub mod mlx;

pub mod mock;

#[cfg(feature = "llamacpp")]
pub use llamacpp::LlamaCppBackend;

#[cfg(feature = "mlx")]
pub use mlx::MlxBackend;

pub use mock::MockBackend;

use crate::backend::{BackendAdapter, BackendType};
use kchat_core::tier::DeviceTier;

/// Select the best available backend for the given platform and tier.
///
/// All tiers now have generative models available (tier-appropriate sizes).
/// On non-llamacpp builds, always returns the mock backend (for testing).
pub fn select_backend(platform: &str, tier: DeviceTier, cpu_arch: &str) -> Option<Box<dyn BackendAdapter>> {
    let backend_type = BackendType::select(platform, tier, cpu_arch)?;

    // MLX backend takes priority when the mlx feature is enabled and the
    // platform selects MLX (Apple Silicon macOS/iOS).
    #[cfg(feature = "mlx")]
    {
        if backend_type == BackendType::Mlx {
            return Some(Box::new(MlxBackend::new()));
        }
    }

    #[cfg(feature = "llamacpp")]
    {
        let _ = backend_type; // selected inside LlamaCppBackend
        return Some(Box::new(LlamaCppBackend::new()));
    }

    #[cfg(not(feature = "llamacpp"))]
    {
        let _ = backend_type;
        return Some(Box::new(MockBackend::new()));
    }
}

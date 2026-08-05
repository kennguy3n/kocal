//! kchat-core: Shared runtime primitives for KChat on-device AI.
//!
//! This crate provides the foundational types and services used by all four
//! workload planes (safety, context, generation, action):
//!
//! - **Capability probing** and device tier selection
//! - **Signed manifest** management for model packs and runtime binaries
//! - **Scheduler** for memory pressure, thermal throttling, and battery budgets
//! - **Private telemetry** with no raw message or retrieved content
//! - **Shared error and ID types**

pub mod capability;
pub mod error;
pub mod ids;
pub mod manifest;
pub mod scheduler;
pub mod telemetry;
pub mod tier;

pub use capability::{CapabilityProbe, DeviceCapabilities, GpuBackend, NpuProvider};
pub use error::{CoreError, Result};
pub use ids::{ArtifactId, ModelPackId, PolicyPackId, TaskId, TenantId, ToolId, UserId};
pub use manifest::{
    ManifestSignature, ModelPackManifest, PackChunk, PackType, RuntimeManifest,
    SignedManifest,
};
pub use scheduler::{Scheduler, SchedulerConfig, SchedulerState};
pub use telemetry::{TelemetryEvent, TelemetryRecorder};
pub use tier::{DeviceTier, TierBudget, TierSelection};

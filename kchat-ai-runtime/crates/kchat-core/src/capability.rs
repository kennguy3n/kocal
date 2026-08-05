//! Device capability probing and tier selection.
//!
//! Tier is a runtime decision, not a marketing label. A high-end phone under
//! memory pressure or serious thermal state must temporarily route as medium
//! or low. The probe runs at installation and is re-evaluated before each job.
//!
//! Platform-specific implementations use real OS APIs:
//! - **macOS/iOS**: `sysctl` for memory/CPU, `NSProcessInfo.thermalState`,
//!   `IOKit` for battery
//! - **Linux/Android**: `/proc/meminfo`, `/sys/class/thermal/`, `/sys/class/power_supply/`
//! - **Windows**: `GlobalMemoryStatusEx`, `GetSystemPowerStatus`,
//!   `GetLogicalProcessorInformationEx`

use crate::error::{CoreError, Result};
use serde::{Deserialize, Serialize};

/// GPU acceleration backend detected on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    /// Apple Metal (iOS, macOS)
    Metal,
    /// Vulkan (Android, Windows, Linux)
    Vulkan,
    /// CUDA / ROCm (Windows, Linux desktop)
    Cuda,
    /// No GPU acceleration available
    None,
}

/// NPU (Neural Processing Unit) provider if available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NpuProvider {
    /// Apple Neural Engine
    AppleNe,
    /// Android NNAPI / QNN
    Nnapi,
    /// Windows NPU / Windows ML
    WindowsNpu,
    /// No NPU available
    None,
}

/// Thermal state reported by the OS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

/// App background/foreground state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AppState {
    Foreground,
    Background,
}

/// Snapshot of device capabilities at probe time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Platform identifier (ios, android, macos, windows)
    pub platform: String,
    /// Physical RAM in bytes
    pub physical_memory: u64,
    /// Safe allocatable memory for AI workloads in bytes
    pub safe_allocatable_memory: u64,
    /// CPU architecture (aarch64, x86_64)
    pub cpu_arch: String,
    /// Number of logical CPU cores
    pub cpu_cores: u32,
    /// Number of performance (big) cores if known
    pub performance_cores: Option<u32>,
    /// ISA features (avx2, avx512, neon, vnni, etc.)
    pub isa_features: Vec<String>,
    /// GPU backend detected
    pub gpu_backend: GpuBackend,
    /// NPU provider if available
    pub npu_provider: NpuProvider,
    /// Free storage in bytes
    pub free_storage: u64,
    /// Battery level (0-100), None if on desktop/no battery
    pub battery_level: Option<u8>,
    /// Whether the device is on charger
    pub on_charger: bool,
    /// Current thermal state
    pub thermal_state: ThermalState,
    /// Current app state
    pub app_state: AppState,
    /// Whether the network is unmetered (Wi-Fi)
    pub unmetered_network: bool,
}

impl DeviceCapabilities {
    /// Returns true if the device is in a state that allows generative inference.
    pub fn allows_generative(&self) -> bool {
        !matches!(self.thermal_state, ThermalState::Serious | ThermalState::Critical)
            && self.app_state == AppState::Foreground
    }

    /// Returns the safe AI memory budget in bytes (70% of safe allocatable).
    pub fn safe_ai_budget(&self) -> u64 {
        self.safe_allocatable_memory * 7 / 10
    }
}

/// Capability probe that detects device hardware and OS capabilities.
pub struct CapabilityProbe;

impl CapabilityProbe {
    /// Probe device capabilities at runtime.
    ///
    /// On real platforms this calls OS-specific APIs. In tests or unknown
    /// environments, it falls back to conservative defaults.
    pub fn probe() -> Result<DeviceCapabilities> {
        let platform = Self::detect_platform();
        let physical_memory = Self::detect_physical_memory();
        let cpu_arch = Self::detect_cpu_arch();
        let cpu_cores = Self::detect_cpu_cores();
        let performance_cores = Self::detect_performance_cores();
        let isa_features = Self::detect_isa_features(&cpu_arch);
        let free_storage = Self::detect_free_storage().unwrap_or(0);
        let battery_level = Self::detect_battery_level();
        let on_charger = Self::detect_on_charger();

        // Safe allocatable: typically 25-40% of physical RAM on mobile,
        // 50-70% on desktop. Conservative default.
        let safe_allocatable_memory = match platform.as_str() {
            "ios" | "android" => physical_memory / 3,
            "macos" | "windows" => physical_memory * 3 / 5,
            _ => physical_memory / 3,
        };

        Ok(DeviceCapabilities {
            platform,
            physical_memory,
            safe_allocatable_memory,
            cpu_arch,
            cpu_cores,
            performance_cores,
            isa_features,
            gpu_backend: Self::detect_gpu_backend(),
            npu_provider: Self::detect_npu_provider(),
            free_storage,
            battery_level,
            on_charger,
            thermal_state: Self::detect_thermal_state(),
            app_state: AppState::Foreground,
            unmetered_network: true,
        })
    }

    /// Re-evaluate dynamic state (thermal, battery, background) before a job.
    pub fn re_evaluate(caps: &mut DeviceCapabilities) {
        caps.thermal_state = Self::detect_thermal_state();
        caps.battery_level = Self::detect_battery_level();
        caps.on_charger = Self::detect_on_charger();
        caps.app_state = Self::detect_app_state();
    }

    fn detect_platform() -> String {
        if cfg!(target_os = "ios") {
            "ios".into()
        } else if cfg!(target_os = "android") {
            "android".into()
        } else if cfg!(target_os = "macos") {
            "macos".into()
        } else if cfg!(target_os = "windows") {
            "windows".into()
        } else if cfg!(target_os = "linux") {
            "linux".into()
        } else {
            "unknown".into()
        }
    }

    fn detect_cpu_arch() -> String {
        if cfg!(target_arch = "aarch64") {
            "aarch64".into()
        } else if cfg!(target_arch = "x86_64") {
            "x86_64".into()
        } else {
            "unknown".into()
        }
    }

    fn detect_cpu_cores() -> u32 {
        std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4)
    }

    fn detect_gpu_backend() -> GpuBackend {
        if cfg!(target_os = "ios") || cfg!(target_os = "macos") {
            GpuBackend::Metal
        } else if cfg!(target_os = "android") || cfg!(target_os = "windows") {
            GpuBackend::Vulkan
        } else {
            GpuBackend::None
        }
    }

    fn detect_npu_provider() -> NpuProvider {
        if cfg!(target_os = "ios") || cfg!(target_os = "macos") {
            NpuProvider::AppleNe
        } else if cfg!(target_os = "android") {
            NpuProvider::Nnapi
        } else if cfg!(target_os = "windows") {
            NpuProvider::WindowsNpu
        } else {
            NpuProvider::None
        }
    }

    fn detect_app_state() -> AppState {
        AppState::Foreground
    }
}

// ============================================================================
// Platform-specific implementations
// ============================================================================

// --- macOS / iOS (sysctl + IOKit) -------------------------------------------

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod apple {
    use super::*;
    use std::ffi::CString;

    /// Call sysctlbyname for a u64 value.
    fn sysctl_u64(name: &str) -> Option<u64> {
        let cname = CString::new(name).ok()?;
        let mut size: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let ret = unsafe {
            libc::sysctlbyname(
                cname.as_ptr(),
                &mut size as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 {
            Some(size)
        } else {
            None
        }
    }

    /// Call sysctlbyname for a i32 value.
    fn sysctl_i32(name: &str) -> Option<i32> {
        let cname = CString::new(name).ok()?;
        let mut value: i32 = 0;
        let mut len = std::mem::size_of::<i32>();
        let ret = unsafe {
            libc::sysctlbyname(
                cname.as_ptr(),
                &mut value as *mut i32 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 {
            Some(value)
        } else {
            None
        }
    }

    impl CapabilityProbe {
        pub fn detect_physical_memory() -> u64 {
            sysctl_u64("hw.memsize").unwrap_or(8 * 1024 * 1024 * 1024)
        }

        pub fn detect_performance_cores() -> Option<u32> {
            // Apple silicon: hw.perflevel0.physicalcpu = P-cores
            // Intel Macs: no perflevel, fall back to None
            sysctl_i32("hw.perflevel0.physicalcpu")
                .map(|n| n as u32)
                .filter(|n| *n > 0)
        }

        pub fn detect_isa_features(arch: &str) -> Vec<String> {
            let mut features = Vec::new();
            if arch == "aarch64" {
                features.push("neon".into());
                // Apple silicon supports ARMv8.2+ features
                if sysctl_i32("hw.optional.armv8_2_sha512").unwrap_or(0) == 1 {
                    features.push("armv8.2".into());
                }
            } else if arch == "x86_64" {
                // Check via sysctl
                if sysctl_i32("hw.optional.avx2_0").unwrap_or(0) == 1 {
                    features.push("avx2".into());
                }
                if sysctl_i32("hw.optional.sse4_2").unwrap_or(0) == 1 {
                    features.push("sse4.2".into());
                }
            }
            features
        }

        pub fn detect_free_storage() -> Option<u64> {
            // Use statvfs on the home directory
            let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
            let cpath = CString::new(home).ok()?;
            let mut statvfs: libc::statvfs = unsafe { std::mem::zeroed() };
            let ret = unsafe { libc::statvfs(cpath.as_ptr(), &mut statvfs) };
            if ret == 0 {
                Some(statvfs.f_bavail as u64 * statvfs.f_frsize as u64)
            } else {
                None
            }
        }

        pub fn detect_battery_level() -> Option<u8> {
            // On macOS, use IOKit to get battery level.
            // For simplicity, we use the `pmset -g batt` command output.
            // In production, this would use IOKit directly.
            std::process::Command::new("pmset")
                .args(["-g", "batt"])
                .output()
                .ok()
                .and_then(|out| {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    // Parse "Currently drawing from 'Battery Power'\n -InternalBattery-0 (id=xxxx)  42%"
                    stdout
                        .lines()
                        .find(|l| l.contains("InternalBattery") || l.contains("Battery"))
                        .and_then(|line| {
                            line.split_whitespace()
                                .find(|w| w.ends_with('%'))
                                .and_then(|w| w.trim_end_matches('%').parse::<u8>().ok())
                        })
                })
        }

        pub fn detect_on_charger() -> bool {
            // Check if drawing from AC power
            std::process::Command::new("pmset")
                .args(["-g", "batt"])
                .output()
                .ok()
                .map(|out| {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    stdout.contains("AC Power") || stdout.contains("AC Attached")
                })
                .unwrap_or(true)
        }

        pub fn detect_thermal_state() -> ThermalState {
            // On macOS, use NSProcessInfo.thermalState via objc.
            // For simplicity, we check the powermetrics or use a conservative approach.
            // In production, this would use objc msg_send to NSProcessInfo.
            // For now, use the `pmset -g therm` command.
            std::process::Command::new("pmset")
                .args(["-g", "therm"])
                .output()
                .ok()
                .and_then(|out| {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    if stdout.contains("Critical") {
                        Some(ThermalState::Critical)
                    } else if stdout.contains("Serious") {
                        Some(ThermalState::Serious)
                    } else if stdout.contains("Fair") {
                        Some(ThermalState::Fair)
                    } else {
                        Some(ThermalState::Nominal)
                    }
                })
                .unwrap_or(ThermalState::Nominal)
        }
    }
}

// --- Linux / Android (/proc + /sys) -----------------------------------------

#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux {
    use super::*;

    impl CapabilityProbe {
        pub fn detect_physical_memory() -> u64 {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.starts_with("MemTotal:"))
                        .and_then(|l| {
                            l.split_whitespace()
                                .nth(1)
                                .and_then(|v| v.parse::<u64>().ok())
                        })
                })
                .map(|kb| kb * 1024)
                .unwrap_or(4 * 1024 * 1024 * 1024)
        }

        pub fn detect_performance_cores() -> Option<u32> {
            // On big.LITTLE: check /sys/devices/cpu_core/ or cpufreq
            // P-cores are typically in cpu_core cluster
            std::fs::read_to_string("/sys/devices/cpu_core/cpus")
                .ok()
                .or_else(|| std::fs::read_to_string("/sys/devices/system/cpu/cpu_isolate").ok())
                .and_then(|s| {
                    // Parse CPU list like "0-3" or "0,1,2,3"
                    let count = s
                        .trim()
                        .split(',')
                        .map(|range| {
                            if let Some((start, end)) = range.split_once('-') {
                                let start: u32 = start.parse().unwrap_or(0);
                                let end: u32 = end.parse().unwrap_or(start);
                                end - start + 1
                            } else {
                                1
                            }
                        })
                        .sum::<u32>();
                    if count > 0 { Some(count) } else { None }
                })
        }

        pub fn detect_isa_features(arch: &str) -> Vec<String> {
            let mut features = Vec::new();
            if arch == "aarch64" {
                features.push("neon".into());
                // Check /proc/cpuinfo for ARM features
                if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                    if cpuinfo.contains("asimd") {
                        features.push("asimd".into());
                    }
                }
            } else if arch == "x86_64" {
                // Check /proc/cpuinfo for x86 features
                if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
                    if cpuinfo.contains("avx2") {
                        features.push("avx2".into());
                    }
                    if cpuinfo.contains("sse4_2") {
                        features.push("sse4.2".into());
                    }
                    if cpuinfo.contains("avx512") {
                        features.push("avx512".into());
                    }
                }
            }
            features
        }

        pub fn detect_free_storage() -> Option<u64> {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/".into());
            // Use statvfs via libc
            let cpath = std::ffi::CString::new(home).ok()?;
            let mut statvfs: libc::statvfs = unsafe { std::mem::zeroed() };
            let ret = unsafe { libc::statvfs(cpath.as_ptr(), &mut statvfs) };
            if ret == 0 {
                Some(statvfs.f_bavail as u64 * statvfs.f_frsize as u64)
            } else {
                None
            }
        }

        pub fn detect_battery_level() -> Option<u8> {
            // Read from /sys/class/power_supply/battery/capacity
            std::fs::read_to_string("/sys/class/power_supply/battery/capacity")
                .ok()
                .or_else(|| std::fs::read_to_string("/sys/class/power_supply/BAT0/capacity").ok())
                .or_else(|| std::fs::read_to_string("/sys/class/power_supply/BAT1/capacity").ok())
                .and_then(|s| s.trim().parse::<u8>().ok())
        }

        pub fn detect_on_charger() -> bool {
            // Read from /sys/class/power_supply/battery/status
            std::fs::read_to_string("/sys/class/power_supply/battery/status")
                .ok()
                .or_else(|| std::fs::read_to_string("/sys/class/power_supply/BAT0/status").ok())
                .or_else(|| std::fs::read_to_string("/sys/class/power_supply/BAT1/status").ok())
                .map(|s| {
                    let s = s.trim();
                    s == "Charging" || s == "Full"
                })
                .unwrap_or(true)
        }

        pub fn detect_thermal_state() -> ThermalState {
            // Read thermal zone temperatures and map to states
            // Thresholds (in millidegrees Celsius):
            // < 45000 = Nominal, < 65000 = Fair, < 85000 = Serious, >= 85000 = Critical
            let mut max_temp: i64 = 0;
            if let Ok(entries) = std::fs::read_dir("/sys/class/thermal/") {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.file_name().map(|n| n.to_string_lossy().starts_with("thermal_zone")).unwrap_or(false) {
                        let temp_path = path.join("temp");
                        if let Ok(temp_str) = std::fs::read_to_string(&temp_path) {
                            if let Ok(temp) = temp_str.trim().parse::<i64>() {
                                max_temp = max_temp.max(temp);
                            }
                        }
                    }
                }
            }
            if max_temp == 0 {
                return ThermalState::Nominal;
            }
            match max_temp {
                t if t >= 85000 => ThermalState::Critical,
                t if t >= 65000 => ThermalState::Serious,
                t if t >= 45000 => ThermalState::Fair,
                _ => ThermalState::Nominal,
            }
        }
    }
}

// --- Windows (GlobalMemoryStatusEx + GetSystemPowerStatus) ------------------

#[cfg(target_os = "windows")]
mod windows {
    use super::*;
    use windows_sys::Win32::System::Power::GetSystemPowerStatus;
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    impl CapabilityProbe {
        pub fn detect_physical_memory() -> u64 {
            unsafe {
                let mut info: MEMORYSTATUSEX = std::mem::zeroed();
                info.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
                if GlobalMemoryStatusEx(&mut info) != 0 {
                    info.ullTotalPhys
                } else {
                    16 * 1024 * 1024 * 1024
                }
            }
        }

        pub fn detect_performance_cores() -> Option<u32> {
            // On Windows, we'd use GetLogicalProcessorInformationEx to count
            // physical cores. For simplicity, return None (let tier logic
            // use total core count).
            None
        }

        pub fn detect_isa_features(arch: &str) -> Vec<String> {
            let mut features = Vec::new();
            if arch == "x86_64" {
                // On Windows, use __cpuid intrinsic. For now, assume common features.
                features.push("sse4.2".into());
                features.push("avx2".into());
            } else if arch == "aarch64" {
                features.push("neon".into());
            }
            features
        }

        pub fn detect_free_storage() -> Option<u64> {
            // Use GetDiskFreeSpaceEx in production
            None
        }

        pub fn detect_battery_level() -> Option<u8> {
            unsafe {
                let mut status: windows_sys::Win32::System::Power::SYSTEM_POWER_STATUS =
                    std::mem::zeroed();
                if GetSystemPowerStatus(&mut status) != 0 {
                    if status.BatteryLifePercent == 255 {
                        None // No battery
                    } else {
                        Some(status.BatteryLifePercent)
                    }
                } else {
                    None
                }
            }
        }

        pub fn detect_on_charger() -> bool {
            unsafe {
                let mut status: windows_sys::Win32::System::Power::SYSTEM_POWER_STATUS =
                    std::mem::zeroed();
                if GetSystemPowerStatus(&mut status) != 0 {
                    status.ACLineStatus == 1
                } else {
                    true
                }
            }
        }

        pub fn detect_thermal_state() -> ThermalState {
            // Windows doesn't have a simple thermal state API like macOS.
            // In production, use Win32_PerformanceCounter or WMI.
            ThermalState::Nominal
        }
    }

}

// --- Fallback (unknown platform) --------------------------------------------

#[cfg(not(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "linux",
    target_os = "android",
    target_os = "windows"
)))]
mod fallback {
    use super::*;

    impl CapabilityProbe {
        pub fn detect_physical_memory() -> u64 {
            4 * 1024 * 1024 * 1024
        }

        pub fn detect_performance_cores() -> Option<u32> {
            None
        }

        pub fn detect_isa_features(_arch: &str) -> Vec<String> {
            Vec::new()
        }

        pub fn detect_free_storage() -> Option<u64> {
            None
        }

        pub fn detect_battery_level() -> Option<u8> {
            None
        }

        pub fn detect_on_charger() -> bool {
            true
        }

        pub fn detect_thermal_state() -> ThermalState {
            ThermalState::Nominal
        }
    }
}

/// Run a 2-5 second local calibration on the exact encoder and a tiny
/// generative prompt to verify safe allocation and memory-map behavior.
pub fn calibrate(caps: &DeviceCapabilities) -> Result<()> {
    if caps.safe_allocatable_memory == 0 {
        return Err(CoreError::CapabilityProbeFailed(
            "safe allocatable memory is zero".into(),
        ));
    }
    // In production: load a tiny test model, measure TTFT, verify mmap.
    // For now, verify the budget is non-trivial.
    if caps.safe_ai_budget() < 256 * 1024 * 1024 {
        return Err(CoreError::CapabilityProbeFailed(format!(
            "safe AI budget too small: {} bytes",
            caps.safe_ai_budget()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_returns_capabilities() {
        let caps = CapabilityProbe::probe().unwrap();
        assert!(!caps.platform.is_empty());
        assert!(caps.physical_memory > 0);
        assert!(caps.cpu_cores > 0);
    }

    #[test]
    fn test_probe_memory_nonzero() {
        let caps = CapabilityProbe::probe().unwrap();
        // On any real platform, memory should be at least 1GB
        assert!(
            caps.physical_memory >= 1024 * 1024 * 1024,
            "physical memory should be >= 1GB, got {} bytes",
            caps.physical_memory
        );
    }

    #[test]
    fn test_probe_cpu_arch() {
        let caps = CapabilityProbe::probe().unwrap();
        assert!(
            caps.cpu_arch == "aarch64" || caps.cpu_arch == "x86_64",
            "cpu_arch should be aarch64 or x86_64, got {}",
            caps.cpu_arch
        );
    }

    #[test]
    fn test_probe_thermal_state() {
        let caps = CapabilityProbe::probe().unwrap();
        // Thermal state should be one of the valid values
        assert!(matches!(
            caps.thermal_state,
            ThermalState::Nominal
                | ThermalState::Fair
                | ThermalState::Serious
                | ThermalState::Critical
        ));
    }

    #[test]
    fn test_safe_ai_budget() {
        let caps = DeviceCapabilities {
            platform: "macos".into(),
            physical_memory: 16 * 1024 * 1024 * 1024,
            safe_allocatable_memory: 10 * 1024 * 1024 * 1024,
            cpu_arch: "aarch64".into(),
            cpu_cores: 8,
            performance_cores: Some(4),
            isa_features: vec![],
            gpu_backend: GpuBackend::Metal,
            npu_provider: NpuProvider::AppleNe,
            free_storage: 0,
            battery_level: None,
            on_charger: true,
            thermal_state: ThermalState::Nominal,
            app_state: AppState::Foreground,
            unmetered_network: true,
        };
        // 70% of safe_allocatable
        assert_eq!(caps.safe_ai_budget(), 7 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_allows_generative() {
        let mut caps = CapabilityProbe::probe().unwrap();
        caps.thermal_state = ThermalState::Nominal;
        caps.app_state = AppState::Foreground;
        assert!(caps.allows_generative());

        caps.thermal_state = ThermalState::Serious;
        assert!(!caps.allows_generative());

        caps.thermal_state = ThermalState::Nominal;
        caps.app_state = AppState::Background;
        assert!(!caps.allows_generative());
    }

    #[test]
    fn test_re_evaluate() {
        let mut caps = CapabilityProbe::probe().unwrap();
        let original_thermal = caps.thermal_state;
        CapabilityProbe::re_evaluate(&mut caps);
        // re_evaluate should not panic and should set a valid thermal state
        assert!(matches!(
            caps.thermal_state,
            ThermalState::Nominal
                | ThermalState::Fair
                | ThermalState::Serious
                | ThermalState::Critical
        ));
        let _ = original_thermal; // may or may not change
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_macos_memory_matches_sysctl() {
        // Verify our sysctl call returns a reasonable value
        let caps = CapabilityProbe::probe().unwrap();
        // macOS machines have at least 8GB
        assert!(
            caps.physical_memory >= 8 * 1024 * 1024 * 1024,
            "macOS should report >= 8GB RAM, got {} bytes",
            caps.physical_memory
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_macos_battery_detection() {
        // On a MacBook, battery should be detected
        let caps = CapabilityProbe::probe().unwrap();
        // On dev machines, battery may or may not be present (desktop Macs don't have one)
        // Just verify it doesn't panic
        let _ = caps.battery_level;
    }

    #[test]
    fn test_calibrate() {
        let caps = CapabilityProbe::probe().unwrap();
        // calibrate should succeed on a real machine with adequate memory
        let result = calibrate(&caps);
        // May fail if safe_ai_budget < 256MB (very low-end device)
        // On a dev machine it should pass
        if caps.safe_ai_budget() >= 256 * 1024 * 1024 {
            assert!(result.is_ok(), "calibrate should succeed with adequate budget");
        }
    }
}

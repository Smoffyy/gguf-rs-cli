//! Device discovery and selection.
//!
//! Backends are optional at compile time and fallible at run time, so selection is a
//! sequence of attempts that ends at the CPU. A GPU that fails to initialize downgrades
//! with a message rather than aborting the run: an engine that refuses to start because a
//! driver is missing is less useful than a slower one that starts.

use std::str::FromStr;

use gguf_core::{Backend, DeviceInfo, DeviceKind, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSelector {
    Auto,
    Cpu,
    Cuda(Option<u32>),
    Vulkan(Option<u32>),
}

impl FromStr for DeviceSelector {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        let (kind, index) = match s.split_once(':') {
            Some((k, i)) => (
                k.to_string(),
                Some(i.parse::<u32>().map_err(|_| {
                    Error::NoDevice(format!("{i:?} is not a device index"))
                })?),
            ),
            None => (s, None),
        };
        Ok(match kind.as_str() {
            "auto" | "" => Self::Auto,
            "cpu" => Self::Cpu,
            "cuda" | "nvidia" => Self::Cuda(index),
            "vulkan" | "vk" => Self::Vulkan(index),
            other => {
                return Err(Error::NoDevice(format!(
                    "unknown device {other:?}; expected auto, cpu, cuda[:N] or vulkan[:N]"
                )))
            }
        })
    }
}

impl std::fmt::Display for DeviceSelector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda(Some(i)) => write!(f, "cuda:{i}"),
            Self::Cuda(None) => f.write_str("cuda"),
            Self::Vulkan(Some(i)) => write!(f, "vulkan:{i}"),
            Self::Vulkan(None) => f.write_str("vulkan"),
        }
    }
}

/// Every device the process can currently reach, CPU first.
pub fn enumerate() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    if let Ok(cpu) = gguf_backend_cpu::CpuBackend::new() {
        out.push(cpu.info().clone());
    }
    #[cfg(feature = "cuda")]
    out.extend(gguf_backend_cuda::enumerate());
    #[cfg(feature = "vulkan")]
    out.extend(gguf_backend_vk::enumerate());
    out
}

/// Open a backend, falling back toward the CPU and reporting why.
pub fn open(sel: DeviceSelector) -> Result<(Box<dyn Backend>, Vec<String>)> {
    let mut notes = Vec::new();

    let try_cuda = |notes: &mut Vec<String>, index: u32| -> Option<Box<dyn Backend>> {
        #[cfg(feature = "cuda")]
        {
            match gguf_backend_cuda::CudaBackend::new(index) {
                Ok(b) => return Some(Box::new(b) as Box<dyn Backend>),
                Err(e) => notes.push(format!("cuda unavailable: {e}")),
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = index;
            notes.push("cuda support was not compiled in".to_string());
        }
        None
    };

    let try_vulkan = |notes: &mut Vec<String>, index: u32| -> Option<Box<dyn Backend>> {
        #[cfg(feature = "vulkan")]
        {
            match gguf_backend_vk::VulkanBackend::new(index) {
                Ok(b) => return Some(Box::new(b) as Box<dyn Backend>),
                Err(e) => notes.push(format!("vulkan unavailable: {e}")),
            }
        }
        #[cfg(not(feature = "vulkan"))]
        {
            let _ = index;
            notes.push("vulkan support was not compiled in".to_string());
        }
        None
    };

    let backend: Box<dyn Backend> = match sel {
        DeviceSelector::Cpu => Box::new(gguf_backend_cpu::CpuBackend::new()?),
        DeviceSelector::Cuda(i) => match try_cuda(&mut notes, i.unwrap_or(0)) {
            Some(b) => b,
            None => {
                return Err(Error::NoDevice(format!(
                    "cuda was requested explicitly but is not usable: {}",
                    notes.join("; ")
                )))
            }
        },
        DeviceSelector::Vulkan(i) => match try_vulkan(&mut notes, i.unwrap_or(0)) {
            Some(b) => b,
            None => {
                return Err(Error::NoDevice(format!(
                    "vulkan was requested explicitly but is not usable: {}",
                    notes.join("; ")
                )))
            }
        },
        // Preference order: CUDA is the fastest path on the hardware it supports, Vulkan
        // covers everything else with a GPU, and the CPU always works.
        DeviceSelector::Auto => try_cuda(&mut notes, 0)
            .or_else(|| try_vulkan(&mut notes, 0))
            .map(Ok)
            .unwrap_or_else(|| gguf_backend_cpu::CpuBackend::new().map(|b| Box::new(b) as Box<dyn Backend>))?,
    };

    if backend.info().kind == DeviceKind::Cpu && sel == DeviceSelector::Auto && !notes.is_empty() {
        notes.push("falling back to the CPU".to_string());
    }
    Ok((backend, notes))
}

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Vulkan,
}

impl DeviceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
        }
    }
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What a device can do that changes which kernel variant we pick.
#[derive(Debug, Clone, Copy, Default)]
pub struct Caps {
    /// 8-bit integer dot product with 32-bit accumulate (`__dp4a`, `dotPacked4x8AccSatEXT`,
    /// `vpdpbusd`). Gates the quantized-activation matmul path.
    pub int8_dot: bool,
    /// Native f16 arithmetic, not just f16 storage.
    pub f16_math: bool,
    /// Tensor cores / cooperative matrix, used for batched prefill GEMM.
    pub matrix_cores: bool,
    pub max_threads_per_block: u32,
    pub shared_mem_per_block: u32,
    /// Warp / subgroup / SIMD width.
    pub warp_size: u32,
    pub multiprocessors: u32,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    pub index: u32,
    pub name: String,
    pub total_memory: u64,
    pub free_memory: u64,
    pub caps: Caps,
}

impl DeviceInfo {
    pub fn id(&self) -> String {
        format!("{}:{}", self.kind, self.index)
    }
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<10} {:<40} {:>6} MiB",
            self.id(),
            self.name,
            self.total_memory / 1048576
        )
    }
}

/// Opaque handle to a backend-owned activation/scratch buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u32);

/// Opaque handle to a backend-owned weight, stored in its native quantized layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemKind {
    /// Device-local, not host mappable. The default for activations and weights.
    Device,
    /// Host-visible staging memory written by the CPU and read by the device.
    Upload,
    /// Host-visible memory the device writes and the CPU reads back.
    Download,
}

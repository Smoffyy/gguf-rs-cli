use std::collections::HashMap;
use ash::{vk, Entry, Device};

mod context;
mod buffers;
mod dispatch;

const SPV_Q4_0:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_0_gemv.spv"));
const SPV_Q4_1:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_1_gemv.spv"));
const SPV_Q4K:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4k_gemv.spv"));
const SPV_Q3K:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q3k_gemv.spv"));
const SPV_Q5K:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q5k_gemv.spv"));
const SPV_Q6K:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q6k_gemv.spv"));
const SPV_Q8_0:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q8_0_gemv.spv"));
const SPV_F32:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_gemv.spv"));
const SPV_RMSNORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.spv"));
const SPV_ROPE:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv"));
const SPV_KVWRITE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kv_write.spv"));
const SPV_ATTN:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv"));
const SPV_SWIGLU:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv"));
const SPV_ADD:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add.spv"));
const SPV_ADD_RN:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add_rmsnorm.spv"));
const SPV_QK_NORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qk_norm.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shader {
    Q4_0, Q4_1, Q4K, Q3K, Q5K, Q6K, Q8_0, F32, RmsNorm, Rope, KvWrite, Attn, SwiGlu, Add, AddRmsNorm, QkNorm,
}

pub struct GpuTensor {
    pub buf:    vk::Buffer,
    pub _mem:   vk::DeviceMemory,
    pub rows:   u32,
    pub bpr:    u32,
    pub shader: Shader,
    pub row_start: u32,
}

pub struct ActBuf {
    pub buf:  vk::Buffer,
    pub mem:  vk::DeviceMemory,
    pub size: u64,
}

pub struct VkCtx {
    _entry:      Entry,
    pub device:  Device,
    queue:       vk::Queue,
    pipes:       HashMap<Shader, (vk::Pipeline, vk::PipelineLayout)>,
    dsl3:        vk::DescriptorSetLayout,
    dsl4:        vk::DescriptorSetLayout,
    dsl5:        vk::DescriptorSetLayout,
    desc_pool:   vk::DescriptorPool,
    cmd_pool:    vk::CommandPool,
    cmd_buf:     vk::CommandBuffer,
    fence:       vk::Fence,
    pub recording: bool,
    pub max_buf: u64,
    pub dev_idx: u32,
    pub host_idx: u32,
    pub device_name: String,
    staging_buf: vk::Buffer,
    _staging_mem: vk::DeviceMemory,
    staging_ptr: *mut u8,
    staging_size: u64,
    ts_pool:     vk::QueryPool,
    ts_period:   f32,
    ts_count:    u32,
    pub debug_gpu: bool,
    ka_cmd:      vk::CommandBuffer,
    ka_fence:    vk::Fence,
    ka_desc_pool: vk::DescriptorPool,
    ka_buf:      Option<(vk::Buffer, vk::DeviceMemory)>,
    ka_active:   bool,
}

//! Compute pipelines, one per kernel and per quantization where the shader is specialized.
//!
//! The matmul, get_rows and MoE matmul shaders are written once and compiled once; the
//! quantization is a SPIR-V specialization constant, so pipeline creation produces a
//! variant with every branch but one folded away. That is what keeps one readable shader
//! from costing fifteen copies of the same file.

use std::collections::HashMap;

use ash::vk;
use gguf_core::{backend_err, DType, Result};

use crate::device::Context;

/// Descriptor sets recycled per submission. Large enough for a full forward pass of a deep
/// model; the backend flushes and resets if a pass ever exceeds it.
pub const MAX_SETS: u32 = 8192;

macro_rules! spv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

const SPV_MATMUL: &[u8] = spv!("matmul");
const SPV_GET_ROWS: &[u8] = spv!("get_rows");
const SPV_NORM: &[u8] = spv!("norm");
const SPV_ROPE: &[u8] = spv!("rope");
const SPV_KV_WRITE: &[u8] = spv!("kv_write");
const SPV_ATTENTION: &[u8] = spv!("attention");
const SPV_ATTN_COMBINE: &[u8] = spv!("attn_combine");
const SPV_ELEMENTWISE: &[u8] = spv!("elementwise");
const SPV_MOE_ROUTE: &[u8] = spv!("moe_route");
const SPV_MOE_MM: &[u8] = spv!("moe_mm");
const SPV_MOE_MISC: &[u8] = spv!("moe_misc");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kernel {
    MatMul(DType),
    GetRows(DType),
    MoeMm(DType),
    Norm,
    Rope,
    /// Specialized on the cache's storage format: false = f32, true = f16.
    KvWrite(bool),
    Attention(bool),
    AttnCombine,
    Elementwise,
    MoeRoute,
    MoeMisc,
}

/// Quantizations with a Vulkan shader path.
pub const TYPES: &[DType] = &[
    DType::F32,
    DType::F16,
    DType::BF16,
    DType::Q4_0,
    DType::Q4_1,
    DType::Q5_0,
    DType::Q5_1,
    DType::Q8_0,
    DType::Q2K,
    DType::Q3K,
    DType::Q4K,
    DType::Q5K,
    DType::Q6K,
    DType::Iq4Nl,
    DType::Iq4Xs,
    DType::Mxfp4,
    DType::Tq1_0,
    DType::Tq2_0,
];

pub fn supports(dt: DType) -> bool {
    TYPES.contains(&dt)
}

struct Entry {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
}

pub struct Pipelines {
    entries: HashMap<Kernel, Entry>,
    /// One layout per binding count, shared by every pipeline that needs that many.
    set_layouts: HashMap<u32, vk::DescriptorSetLayout>,
    pipeline_layouts: HashMap<u32, vk::PipelineLayout>,
}

impl Pipelines {
    pub fn build(ctx: &Context) -> Result<Self> {
        let mut p = Self {
            entries: HashMap::new(),
            set_layouts: HashMap::new(),
            pipeline_layouts: HashMap::new(),
        };

        // (kernel, spirv, binding count)
        let fixed: &[(Kernel, &[u8], u32)] = &[
            (Kernel::Norm, SPV_NORM, 4),
            (Kernel::Rope, SPV_ROPE, 3),
            (Kernel::AttnCombine, SPV_ATTN_COMBINE, 3),
            (Kernel::Elementwise, SPV_ELEMENTWISE, 3),
            (Kernel::MoeRoute, SPV_MOE_ROUTE, 4),
            (Kernel::MoeMisc, SPV_MOE_MISC, 3),
        ];
        for (kernel, spv, bindings) in fixed {
            let e = p.create(ctx, spv, *bindings, None)?;
            p.entries.insert(*kernel, e);
        }

        // The cache format is a specialization constant, so each kernel that touches the
        // cache exists in an f32 and an f16 variant with the other branch folded away.
        for (f16, code) in [(false, 0), (true, 1)] {
            let kw = p.create(ctx, SPV_KV_WRITE, 4, Some(code))?;
            p.entries.insert(Kernel::KvWrite(f16), kw);
            let at = p.create(ctx, SPV_ATTENTION, 4, Some(code))?;
            p.entries.insert(Kernel::Attention(f16), at);
        }

        for dt in TYPES {
            let ty = *dt as u32 as i32;
            let mm = p.create(ctx, SPV_MATMUL, 4, Some(ty))?;
            p.entries.insert(Kernel::MatMul(*dt), mm);
            let gr = p.create(ctx, SPV_GET_ROWS, 3, Some(ty))?;
            p.entries.insert(Kernel::GetRows(*dt), gr);
            let moe = p.create(ctx, SPV_MOE_MM, 4, Some(ty))?;
            p.entries.insert(Kernel::MoeMm(*dt), moe);
        }
        Ok(p)
    }

    fn set_layout(&mut self, ctx: &Context, bindings: u32) -> Result<vk::DescriptorSetLayout> {
        if let Some(l) = self.set_layouts.get(&bindings) {
            return Ok(*l);
        }
        let binds: Vec<vk::DescriptorSetLayoutBinding> = (0..bindings)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let layout = unsafe {
            ctx.device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
                    None,
                )
                .map_err(|e| backend_err("vulkan", format!("creating a set layout: {e}")))?
        };
        self.set_layouts.insert(bindings, layout);
        Ok(layout)
    }

    fn pipeline_layout(&mut self, ctx: &Context, bindings: u32) -> Result<vk::PipelineLayout> {
        if let Some(l) = self.pipeline_layouts.get(&bindings) {
            return Ok(*l);
        }
        let set_layout = self.set_layout(ctx, bindings)?;
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(crate::PUSH_MAX as u32)];
        let layouts = [set_layout];
        let layout = unsafe {
            ctx.device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&ranges),
                    None,
                )
                .map_err(|e| backend_err("vulkan", format!("creating a pipeline layout: {e}")))?
        };
        self.pipeline_layouts.insert(bindings, layout);
        Ok(layout)
    }

    fn create(&mut self, ctx: &Context, spv: &[u8], bindings: u32, quant: Option<i32>) -> Result<Entry> {
        let set_layout = self.set_layout(ctx, bindings)?;
        let layout = self.pipeline_layout(ctx, bindings)?;

        let code: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let module = unsafe {
            ctx.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .map_err(|e| backend_err("vulkan", format!("creating a shader module: {e}")))?
        };

        let name = std::ffi::CString::new("main").unwrap();
        let entries = [vk::SpecializationMapEntry {
            constant_id: 0,
            offset: 0,
            size: std::mem::size_of::<i32>(),
        }];
        let data = quant.unwrap_or(0).to_ne_bytes();
        let spec = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&data);

        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&name);
        if quant.is_some() {
            stage = stage.specialization_info(&spec);
        }

        let create = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
        let pipeline = unsafe {
            ctx.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create], None)
                .map_err(|(_, e)| backend_err("vulkan", format!("creating a compute pipeline: {e}")))?[0]
        };
        unsafe { ctx.device.destroy_shader_module(module, None) };

        Ok(Entry { pipeline, layout, set_layout })
    }

    pub fn get(&self, kernel: Kernel) -> Result<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout)> {
        let e = self
            .entries
            .get(&kernel)
            .ok_or_else(|| backend_err("vulkan", format!("no pipeline for {kernel:?}")))?;
        Ok((e.pipeline, e.layout, e.set_layout))
    }

    pub fn destroy(&self, ctx: &Context) {
        unsafe {
            for e in self.entries.values() {
                ctx.device.destroy_pipeline(e.pipeline, None);
            }
            for l in self.pipeline_layouts.values() {
                ctx.device.destroy_pipeline_layout(*l, None);
            }
            for l in self.set_layouts.values() {
                ctx.device.destroy_descriptor_set_layout(*l, None);
            }
        }
    }
}

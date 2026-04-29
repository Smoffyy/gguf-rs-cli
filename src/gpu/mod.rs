use std::collections::HashMap;
use wgpu::util::DeviceExt;
use crate::tensor::dequant::QuantTensor;
use crate::gguf::types::GgmlType;

/// Which GPU shader handles this tensor type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuShader { Q4_0, Q4K, Q6K, Q8_0, F32 }

pub struct GpuTensor {
    pub buf:    wgpu::Buffer,
    pub rows:   u32,
    /// Meaning depends on shader: blocks/row for block-quants, cols for F32
    pub bpr:    u32,
    pub shader: GpuShader,
}

pub struct GpuCtx {
    pub device:  wgpu::Device,
    pub queue:   wgpu::Queue,
    bgl:         wgpu::BindGroupLayout,
    pipes:       HashMap<GpuShader, wgpu::ComputePipeline>,
    /// Actual hardware max buffer size (read from adapter.limits())
    pub max_buf: u64,
}

impl GpuCtx {
    pub fn init() -> Option<Self> { pollster::block_on(Self::init_async()) }

    async fn init_async() -> Option<Self> {
        // Try backends in preference order; pick the first real GPU found
        for &backend in &[
            wgpu::Backends::DX12,
            wgpu::Backends::VULKAN,
            wgpu::Backends::METAL,
            wgpu::Backends::GL,
        ] {
            let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: backend, ..Default::default()
            });
            let Some(adapter) = inst.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference:       wgpu::PowerPreference::HighPerformance,
                compatible_surface:     None,
                force_fallback_adapter: false,
            }).await else { continue };

            // Skip software / CPU-only renderers (e.g. warp, llvmpipe)
            if adapter.get_info().device_type == wgpu::DeviceType::Cpu { continue; }

            let info    = adapter.get_info();
            let hw      = adapter.limits();
            // Use the smaller of the two relevant limits for safety
            let max_buf = hw.max_buffer_size
                           .min(hw.max_storage_buffer_binding_size as u64);

            eprintln!("GPU: {} ({:?}) — max buffer {:.0} MB",
                info.name, info.backend, max_buf as f64 / 1_000_000.0);

            let Ok((device, queue)) = adapter.request_device(&wgpu::DeviceDescriptor {
                label:              None,
                required_features:  wgpu::Features::empty(),
                required_limits:    wgpu::Limits {
                    max_buffer_size:                max_buf,
                    max_storage_buffer_binding_size: max_buf.min(u32::MAX as u64) as u32,
                    ..wgpu::Limits::default()
                },
            }, None).await else { continue };

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label:  None,
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true  }),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true  }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                    bgl_entry(3, wgpu::BufferBindingType::Uniform),
                ],
            });

            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
            });

            // Build one compute pipeline per shader entry point
            let mut pipes = HashMap::new();
            for (stype, entry) in [
                (GpuShader::Q4_0, "q4_0_gemv"),
                (GpuShader::Q4K,  "q4k_gemv"),
                (GpuShader::Q6K,  "q6k_gemv"),
                (GpuShader::Q8_0, "q8_0_gemv"),
                (GpuShader::F32,  "f32_gemv"),
            ] {
                pipes.insert(stype, make_pipe(&device, &pl, &shader, entry));
            }

            return Some(Self { device, queue, bgl, pipes, max_buf });
        }
        eprintln!("No suitable GPU found — using CPU.");
        None
    }

    /// Try to upload a weight tensor to GPU.
    /// Returns None if the type is unsupported or the tensor is too large.
    pub fn upload(&self, wt: &QuantTensor) -> Option<GpuTensor> {
        // Pack the tensor into a GPU-friendly format, get shader type and bpr
        let (packed, shader, bpr) = match wt.typ {
            GgmlType::Q4_0 => {
                let p = wt.pack_q4_0_for_gpu();
                (p, GpuShader::Q4_0, (wt.cols / 32) as u32)
            }
            GgmlType::Q4K => {
                // Only valid if cols is a multiple of 256
                if wt.cols % 256 != 0 { return None; }
                let p = wt.pack_q4k_for_gpu();
                (p, GpuShader::Q4K, (wt.cols / 256) as u32)
            }
            GgmlType::Q6K => {
                if wt.cols % 256 != 0 { return None; }
                let p = wt.pack_q6k_for_gpu();
                (p, GpuShader::Q6K, (wt.cols / 256) as u32)
            }
            GgmlType::Q8_0 => {
                if wt.cols % 32 != 0 { return None; }
                let p = wt.pack_q8_0_for_gpu();
                (p, GpuShader::Q8_0, (wt.cols / 32) as u32)
            }
            // Q2_K, Q3_K, Q5_K, Q5_0, Q4_1, etc: fall back to CPU rayon
            _ => return None,
        };

        let byte_size = packed.len() as u64 * 4;
        if byte_size > self.max_buf {
            return None;  // too large for GPU buffer; CPU handles it
        }

        let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: bytemuck::cast_slice(&packed),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        Some(GpuTensor { buf, rows: wt.rows as u32, bpr, shader })
    }

    /// Execute a GEMV on a pre-uploaded tensor: result = wt @ vec_in
    pub fn dispatch(&self, wt: &GpuTensor, vec_in: &[f32]) -> Vec<f32> {
        let in_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: bytemuck::cast_slice(vec_in),
            usage:    wgpu::BufferUsages::STORAGE,
        });
        let out_bytes = wt.rows as u64 * 4;
        let out_buf   = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: out_bytes, mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let params: [u32; 2] = [wt.rows, wt.bpr];
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: bytemuck::cast_slice(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wt.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: param_buf.as_entire_binding() },
            ],
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: out_bytes, mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        let mut enc = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(
                &wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(self.pipes.get(&wt.shader).unwrap());
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wt.rows.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data   = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn make_pipe(device: &wgpu::Device, layout: &wgpu::PipelineLayout,
             shader: &wgpu::ShaderModule, entry: &str) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(layout), module: shader, entry_point: entry,
        compilation_options: Default::default(),
    })
}
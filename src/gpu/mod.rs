use wgpu::util::DeviceExt;
use crate::tensor::dequant::QuantTensor;
use crate::gguf::types::GgmlType;

pub struct GpuTensor {
    pub buf: wgpu::Buffer,
    pub rows: u32,
    pub bpr:  u32,   // blocks-per-row for Q4_0, or cols for F32
    pub is_q4: bool,
}

pub struct GpuCtx {
    pub device:   wgpu::Device,
    pub queue:    wgpu::Queue,
    q4_pipe:  wgpu::ComputePipeline,
    f32_pipe: wgpu::ComputePipeline,
    bgl:      wgpu::BindGroupLayout,
}

impl GpuCtx {
    pub fn init() -> Option<Self> {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Option<Self> {
        let inst = wgpu::Instance::default();
        let adapter = inst.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await?;

        let info = adapter.get_info();
        eprintln!("GPU: {} ({:?})", info.name, info.backend);

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            }, None
        ).await.ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                bgl_entry(3, wgpu::BufferBindingType::Uniform),
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });

        let q4_pipe = make_pipe(&device, &pl, &shader, "q4_gemv");
        let f32_pipe = make_pipe(&device, &pl, &shader, "f32_gemv");

        Some(Self { device, queue, q4_pipe, f32_pipe, bgl })
    }

    pub fn upload(&self, wt: &QuantTensor) -> GpuTensor {
        match wt.typ {
            GgmlType::Q4_0 => {
                let packed = wt.pack_q4_0_for_gpu();
                let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&packed),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                GpuTensor { buf, rows: wt.rows as u32, bpr: (wt.cols/32) as u32, is_q4: true }
            }
            _ => {
                // Dequantize to f32 for GPU
                let data = crate::tensor::dequant::dequantize(wt.typ, &wt.data, wt.rows*wt.cols).unwrap();
                let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                GpuTensor { buf, rows: wt.rows as u32, bpr: wt.cols as u32, is_q4: false }
            }
        }
    }

    pub fn matvec(&self, wt: &QuantTensor, vec_in: &[f32]) -> Vec<f32> {
        // Upload this tensor on-the-fly (simple but not cached)
        // For production you'd cache GpuTensors, but this works
        let gt = self.upload(wt);
        self.dispatch(&gt, vec_in)
    }

    fn dispatch(&self, wt: &GpuTensor, vec_in: &[f32]) -> Vec<f32> {
        let in_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(vec_in),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_bytes = wt.rows as u64 * 4;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: out_bytes, mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params: [u32; 2] = [wt.rows, wt.bpr];
        let param_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding:0, resource: wt.buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding:1, resource: in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding:2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding:3, resource: param_buf.as_entire_binding() },
            ],
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: out_bytes, mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            pass.set_pipeline(if wt.is_q4 { &self.q4_pipe } else { &self.f32_pipe });
            pass.set_bind_group(0, &bg, &[]);
            let wgs = wt.rows.div_ceil(64);
            pass.dispatch_workgroups(wgs, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_bytes);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
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

fn make_pipe(device: &wgpu::Device, layout: &wgpu::PipelineLayout, shader: &wgpu::ShaderModule, entry: &str) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(layout), module: shader,
        entry_point: entry,
        compilation_options: Default::default(),    })
}

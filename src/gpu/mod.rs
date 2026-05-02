use std::collections::HashMap;
use ash::{vk, Entry, Device};
use crate::tensor::dequant::QuantTensor;
use crate::gguf::types::GgmlType;

const SPV_Q4_0:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4_0_gemv.spv"));
const SPV_Q4K:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q4k_gemv.spv"));
const SPV_Q6K:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q6k_gemv.spv"));
const SPV_Q8_0:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/q8_0_gemv.spv"));
const SPV_F32:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/f32_gemv.spv"));
const SPV_RMSNORM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.spv"));
const SPV_ROPE:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.spv"));
const SPV_KVWRITE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kv_write.spv"));
const SPV_ATTN:    &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/attention.spv"));
const SPV_SWIGLU:  &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/swiglu.spv"));
const SPV_ADD:     &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/add.spv"));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Shader { Q4_0, Q4K, Q6K, Q8_0, F32, RmsNorm, Rope, KvWrite, Attn, SwiGlu, Add }

pub struct GpuTensor {
    pub buf:    vk::Buffer,
    pub _mem:   vk::DeviceMemory,
    pub rows:   u32,
    pub bpr:    u32,
    pub shader: Shader,
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
}

impl VkCtx {
    pub fn init() -> anyhow::Result<Self> {
        unsafe {
            let entry    = Entry::load()?;
            let app_info = vk::ApplicationInfo {
                api_version: vk::make_api_version(0, 1, 1, 0),
                ..Default::default()
            };
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info), None)?;

            let phys_devs = instance.enumerate_physical_devices()?;
            let phys_dev  = phys_devs.iter().copied()
                .max_by_key(|&pd| match instance.get_physical_device_properties(pd).device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU   => 3,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                    vk::PhysicalDeviceType::VIRTUAL_GPU    => 1,
                    _ => 0,
                })
                .ok_or_else(|| anyhow::anyhow!("No Vulkan device"))?;

            let props   = instance.get_physical_device_properties(phys_dev);
            let name    = std::ffi::CStr::from_ptr(props.device_name.as_ptr()).to_string_lossy();
            let max_buf = props.limits.max_storage_buffer_range as u64;
            eprintln!("GPU: {} | max SSBO: {} MB", name, max_buf / 1_000_000);

            let qfams     = instance.get_physical_device_queue_family_properties(phys_dev);
            let queue_fam = qfams.iter().enumerate()
                .find(|(_, f)| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow::anyhow!("No compute queue"))?;

            let prio = [1.0f32];
            let qci  = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_fam).queue_priorities(&prio)];
            let device = instance.create_device(phys_dev,
                &vk::DeviceCreateInfo::default().queue_create_infos(&qci), None)?;
            let queue  = device.get_device_queue(queue_fam, 0);

            let mp       = instance.get_physical_device_memory_properties(phys_dev);
            let dev_idx  = find_mem(&mp, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let host_idx = find_mem(&mp,
                vk::MemoryPropertyFlags::HOST_VISIBLE|vk::MemoryPropertyFlags::HOST_COHERENT)?;

            let dsl3 = make_dsl(&device, &[ssbo(0), ssbo(1), ssbo(2)])?;
            let dsl4 = make_dsl(&device, &[ssbo(0), ssbo(1), ssbo(2), ssbo(3)])?;
            let dsl5 = make_dsl(&device, &[ssbo(0), ssbo(1), ssbo(2), ssbo(3), ssbo(4)])?;

            let pc_range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(32)];

            let mut pipes = HashMap::new();
            // (shader, spv, which DSL)
            for (shader, spv, dsl) in [
                (Shader::Q4_0,    SPV_Q4_0,    dsl3),
                (Shader::Q4K,     SPV_Q4K,     dsl3),
                (Shader::Q6K,     SPV_Q6K,     dsl3),
                (Shader::Q8_0,    SPV_Q8_0,    dsl3),
                (Shader::F32,     SPV_F32,     dsl3),
                (Shader::RmsNorm, SPV_RMSNORM, dsl3),
                (Shader::Rope,    SPV_ROPE,    dsl3),
                (Shader::KvWrite, SPV_KVWRITE, dsl4),
                (Shader::Attn,    SPV_ATTN,    dsl5),
                (Shader::SwiGlu,  SPV_SWIGLU,  dsl3),
                (Shader::Add,     SPV_ADD,     dsl3),
            ] {
                let layout = device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[dsl]).push_constant_ranges(&pc_range), None)?;
                let module = make_module(&device, spv)?;
                let name   = std::ffi::CString::new("main").unwrap();
                let stage  = vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&name);
                let pipe   = device.create_compute_pipelines(vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
                    None).map_err(|(_, e)| e)?[0];
                device.destroy_shader_module(module, None);
                pipes.insert(shader, (pipe, layout));
            }

            let pool_sz   = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 10000 }];
            let desc_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(2000).pool_sizes(&pool_sz),
                None)?;

            let cmd_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_fam), None)?;
            let cmd_buf  = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1))?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            Ok(Self {
                _entry: entry, device, queue, pipes, dsl3, dsl4, dsl5,
                desc_pool, cmd_pool, cmd_buf, fence, recording: false,
                max_buf, dev_idx, host_idx,
            })
        }
    }

    pub fn upload(&mut self, wt: &QuantTensor) -> Option<GpuTensor> {
        let (packed, shader, bpr) = match wt.typ {
            GgmlType::Q4_0                  => (wt.pack_q4_0_for_gpu(), Shader::Q4_0, (wt.cols/32) as u32),
            GgmlType::Q4K if wt.cols%256==0 => (wt.pack_q4k_for_gpu(), Shader::Q4K,  (wt.cols/256) as u32),
            GgmlType::Q6K if wt.cols%256==0 => (wt.pack_q6k_for_gpu(), Shader::Q6K,  (wt.cols/256) as u32),
            GgmlType::Q8_0 if wt.cols%32==0 => (wt.pack_q8_0_for_gpu(),Shader::Q8_0, (wt.cols/32) as u32),
            _ => return None,
        };
        let size = packed.len() as u64 * 4;
        if size > self.max_buf { return None; }
        unsafe {
            let (buf, mem) = self.upload_bytes(size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                bytemuck::cast_slice(&packed)).ok()?;
            Some(GpuTensor { buf, _mem: mem, rows: wt.rows as u32, bpr, shader })
        }
    }

    pub fn alloc_act(&mut self, size: u64) -> anyhow::Result<ActBuf> {
        let (buf, mem) = self.alloc_raw(size,
            vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST,
            self.dev_idx)?;
        Ok(ActBuf { buf, mem, size })
    }

    pub fn alloc_readback(&mut self, size: u64) -> anyhow::Result<ActBuf> {
        let (buf, mem) = self.alloc_raw(size, vk::BufferUsageFlags::TRANSFER_DST, self.host_idx)?;
        Ok(ActBuf { buf, mem, size })
    }

    /// Upload CPU float slice into a GPU activation buffer.
    pub fn write_act(&mut self, act: &ActBuf, data: &[f32]) {
        unsafe {
            let size = data.len() as u64 * 4;
            let (stg, sm) = self.alloc_raw(
                size, vk::BufferUsageFlags::TRANSFER_SRC, self.host_idx).unwrap();
            let ptr = self.device.map_memory(sm, 0, size, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, size as usize);
            self.device.unmap_memory(sm);
            let cb = self.one_shot_begin();
            self.device.cmd_copy_buffer(cb, stg, act.buf,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
            self.one_shot_end(cb);
            self.device.destroy_buffer(stg, None);
            self.device.free_memory(sm, None);
        }
    }

    /// Read logits back to CPU after GPU forward pass.
    pub fn read_logits(&self, logits_buf: &ActBuf, rb: &ActBuf) -> Vec<f32> {
        unsafe {
            let cb = self.one_shot_begin();
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.device.cmd_pipeline_barrier(cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[barrier], &[], &[]);
            self.device.cmd_copy_buffer(cb, logits_buf.buf, rb.buf,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: logits_buf.size }]);
            self.one_shot_end(cb);
            let ptr = self.device.map_memory(rb.mem, 0, logits_buf.size,
                vk::MemoryMapFlags::empty()).unwrap();
            let mut out = vec![0f32; (logits_buf.size / 4) as usize];
            std::ptr::copy_nonoverlapping(
                ptr as *const u8, out.as_mut_ptr() as *mut u8, logits_buf.size as usize);
            self.device.unmap_memory(rb.mem);
            out
        }
    }

    // ── Command encoding ─────────────────────────────────────────────────────

    pub fn begin(&mut self) {
        unsafe {
            self.device.reset_command_buffer(
                self.cmd_buf, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.begin_command_buffer(self.cmd_buf,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            self.recording = true;
        }
    }

    pub fn submit(&mut self) {
        unsafe {
            self.device.end_command_buffer(self.cmd_buf).unwrap();
            self.recording = false;
            self.device.queue_submit(self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.cmd_buf])],
                self.fence).unwrap();
            self.device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            self.device.reset_fences(&[self.fence]).unwrap();
        }
    }

    /// Compute-to-compute pipeline barrier.
    pub fn barrier(&self) {
        unsafe {
            let b = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            self.device.cmd_pipeline_barrier(self.cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[b], &[], &[]);
        }
    }

    pub fn cmd_rmsnorm(&mut self, x: &ActBuf, w: &ActBuf, out: &ActBuf, n: u32, eps: f32) {
        let pc: [u32; 2] = [n, eps.to_bits()];
        let ds = self.ds3(x.buf, w.buf, out.buf);
        self.enc3(Shader::RmsNorm, ds, &pc, 1, 1, 1);
    }

    pub fn cmd_gemv(&mut self, wt: &GpuTensor, inp: &ActBuf, out: &ActBuf) {
        let pc: [u32; 2] = [wt.rows, wt.bpr];
        let ds = self.ds3(wt.buf, inp.buf, out.buf);
        self.enc3(wt.shader, ds, &pc, wt.rows.div_ceil(64), 1, 1);
    }

    pub fn cmd_add(&mut self, a: &ActBuf, b: &ActBuf, n: u32) {
        let pc: [u32; 1] = [n];
        let ds = self.ds3(a.buf, b.buf, a.buf);
        self.enc3(Shader::Add, ds, bytemuck::cast_slice(&pc), n.div_ceil(64), 1, 1);
    }

    pub fn cmd_rope(&mut self, q: &ActBuf, k: &ActBuf,
                    n_heads: u32, n_kv_heads: u32, head_dim: u32, pos: u32, freq: f32) {
        let pc: [u32; 5] = [n_heads, n_kv_heads, head_dim, pos, freq.to_bits()];
        let ds = self.ds3(q.buf, k.buf, q.buf);
        self.enc3(Shader::Rope, ds, bytemuck::cast_slice(&pc), n_heads + n_kv_heads, 1, 1);
    }

    pub fn cmd_kv_write(&mut self, k: &ActBuf, v: &ActBuf, kc: &ActBuf, vc: &ActBuf,
                         pos: u32, n_kv_heads: u32, head_dim: u32) {
        let kvd = n_kv_heads * head_dim;
        let pc: [u32; 4] = [pos, n_kv_heads, head_dim, 0];
        let ds = self.ds4(k.buf, v.buf, kc.buf, vc.buf);
        self.enc4(Shader::KvWrite, ds, bytemuck::cast_slice(&pc), kvd.div_ceil(64), 1, 1);
    }

    pub fn cmd_attention(&mut self, q: &ActBuf, kc: &ActBuf, vc: &ActBuf,
                          ao: &ActBuf, scores: &ActBuf,
                          n_heads: u32, n_kv_heads: u32, head_dim: u32,
                          seq_len: u32, n_ctx: u32) {
        let pc: [u32; 5] = [n_heads, n_kv_heads, head_dim, seq_len, n_ctx];
        let ds = self.ds5(q.buf, kc.buf, vc.buf, ao.buf, scores.buf);
        self.enc5(Shader::Attn, ds, bytemuck::cast_slice(&pc), n_heads, 1, 1);
    }

    pub fn cmd_swiglu(&mut self, gate: &ActBuf, up: &ActBuf, n: u32) {
        let pc: [u32; 1] = [n];
        let ds = self.ds3(gate.buf, up.buf, gate.buf);
        self.enc3(Shader::SwiGlu, ds, bytemuck::cast_slice(&pc), n.div_ceil(64), 1, 1);
    }

    // ── Descriptor set helpers ────────────────────────────────────────────────

    fn ds3(&mut self, b0: vk::Buffer, b1: vk::Buffer, b2: vk::Buffer) -> vk::DescriptorSet {
        unsafe {
            let ds = alloc_ds(&self.device, self.desc_pool, self.dsl3);
            upd3(&self.device, ds, b0, b1, b2);
            ds
        }
    }
    fn ds4(&mut self, b0: vk::Buffer, b1: vk::Buffer,
           b2: vk::Buffer, b3: vk::Buffer) -> vk::DescriptorSet {
        unsafe {
            let ds = alloc_ds(&self.device, self.desc_pool, self.dsl4);
            upd4(&self.device, ds, b0, b1, b2, b3);
            ds
        }
    }
    fn ds5(&mut self, b0: vk::Buffer, b1: vk::Buffer, b2: vk::Buffer,
           b3: vk::Buffer, b4: vk::Buffer) -> vk::DescriptorSet {
        unsafe {
            let ds = alloc_ds(&self.device, self.desc_pool, self.dsl5);
            let bufs = [b0, b1, b2, b3, b4];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs.iter()
                .map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE)])
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = (0..5u32)
                .map(|i| vk::WriteDescriptorSet::default()
                    .dst_set(ds).dst_binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[i as usize]))
                .collect();
            self.device.update_descriptor_sets(&writes, &[]);
            ds
        }
    }

    fn enc3(&self, shader: Shader, ds: vk::DescriptorSet, pc: &[u32], x: u32, y: u32, z: u32) {
        unsafe { self.enc(shader, ds, pc, x, y, z); }
    }
    fn enc4(&self, shader: Shader, ds: vk::DescriptorSet, pc: &[u32], x: u32, y: u32, z: u32) {
        unsafe { self.enc(shader, ds, pc, x, y, z); }
    }
    fn enc5(&self, shader: Shader, ds: vk::DescriptorSet, pc: &[u32], x: u32, y: u32, z: u32) {
        unsafe { self.enc(shader, ds, pc, x, y, z); }
    }
    unsafe fn enc(&self, shader: Shader, ds: vk::DescriptorSet,
                  pc: &[u32], x: u32, y: u32, z: u32) {
        let (pipe, layout) = self.pipes[&shader];
        self.device.cmd_bind_pipeline(self.cmd_buf, vk::PipelineBindPoint::COMPUTE, pipe);
        self.device.cmd_bind_descriptor_sets(
            self.cmd_buf, vk::PipelineBindPoint::COMPUTE, layout, 0, &[ds], &[]);
        self.device.cmd_push_constants(self.cmd_buf, layout,
            vk::ShaderStageFlags::COMPUTE, 0,
            std::slice::from_raw_parts(pc.as_ptr() as *const u8, pc.len() * 4));
        self.device.cmd_dispatch(self.cmd_buf, x, y, z);
    }

    // ── Memory helpers ────────────────────────────────────────────────────────

    unsafe fn one_shot_begin(&self) -> vk::CommandBuffer {
        let cb = self.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default().command_pool(self.cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1)).unwrap()[0];
        self.device.begin_command_buffer(cb,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
        cb
    }
    unsafe fn one_shot_end(&self, cb: vk::CommandBuffer) {
        self.device.end_command_buffer(cb).unwrap();
        self.device.queue_submit(self.queue,
            &[vk::SubmitInfo::default().command_buffers(&[cb])], self.fence).unwrap();
        self.device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
        self.device.reset_fences(&[self.fence]).unwrap();
        self.device.free_command_buffers(self.cmd_pool, &[cb]);
    }

    fn upload_bytes(&mut self, size: u64, usage: vk::BufferUsageFlags,
                    data: &[u8]) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        unsafe {
            let (stg, sm) = self.alloc_raw(
                size, vk::BufferUsageFlags::TRANSFER_SRC, self.host_idx)?;
            let ptr = self.device.map_memory(sm, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, size as usize);
            self.device.unmap_memory(sm);
            let (buf, bm) = self.alloc_raw(
                size, usage | vk::BufferUsageFlags::TRANSFER_DST, self.dev_idx)?;
            let cb = self.one_shot_begin();
            self.device.cmd_copy_buffer(cb, stg, buf,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
            self.one_shot_end(cb);
            self.device.destroy_buffer(stg, None);
            self.device.free_memory(sm, None);
            Ok((buf, bm))
        }
    }

    pub fn alloc_raw(&self, size: u64, usage: vk::BufferUsageFlags,
                     mt: u32) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
        unsafe {
            let buf = self.device.create_buffer(
                &vk::BufferCreateInfo::default().size(size).usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
            let reqs = self.device.get_buffer_memory_requirements(buf);
            let mem  = self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size).memory_type_index(mt), None)?;
            self.device.bind_buffer_memory(buf, mem, 0)?;
            Ok((buf, mem))
        }
    }
}

fn find_mem(p: &vk::PhysicalDeviceMemoryProperties,
            f: vk::MemoryPropertyFlags) -> anyhow::Result<u32> {
    (0..p.memory_type_count)
        .find(|&i| p.memory_types[i as usize].property_flags.contains(f))
        .ok_or_else(|| anyhow::anyhow!("No memory type {:?}", f))
}

fn make_dsl(device: &Device,
            bindings: &[vk::DescriptorSetLayoutBinding]) -> anyhow::Result<vk::DescriptorSetLayout> {
    unsafe {
        Ok(device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings), None)?)
    }
}

fn ssbo(b: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(b).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)
}

unsafe fn alloc_ds(device: &Device, pool: vk::DescriptorPool,
                   layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
    device.allocate_descriptor_sets(
        &vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool).set_layouts(&[layout])).unwrap()[0]
}

unsafe fn upd3(device: &Device, ds: vk::DescriptorSet,
               b0: vk::Buffer, b1: vk::Buffer, b2: vk::Buffer) {
    let i = |b| [vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE)];
    let (i0, i1, i2) = (i(b0), i(b1), i(b2));
    device.update_descriptor_sets(&[
        wr(ds, 0, &i0), wr(ds, 1, &i1), wr(ds, 2, &i2),
    ], &[]);
}

unsafe fn upd4(device: &Device, ds: vk::DescriptorSet,
               b0: vk::Buffer, b1: vk::Buffer, b2: vk::Buffer, b3: vk::Buffer) {
    let i = |b| [vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(vk::WHOLE_SIZE)];
    let (i0, i1, i2, i3) = (i(b0), i(b1), i(b2), i(b3));
    device.update_descriptor_sets(&[
        wr(ds, 0, &i0), wr(ds, 1, &i1), wr(ds, 2, &i2), wr(ds, 3, &i3),
    ], &[]);
}

fn wr<'a>(ds: vk::DescriptorSet, b: u32,
          info: &'a [vk::DescriptorBufferInfo]) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(ds).dst_binding(b)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(info)
}

unsafe fn make_module(device: &Device, spv: &[u8]) -> anyhow::Result<vk::ShaderModule> {
    let (p, a, s) = spv.align_to::<u32>();
    assert!(p.is_empty() && s.is_empty());
    Ok(device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(a), None)?)
}
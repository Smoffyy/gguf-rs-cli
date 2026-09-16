use ash::{vk, Device};
use super::*;

impl VkCtx {
    pub fn begin(&mut self) { self.begin_for_sets(4096); }

    pub fn begin_for_sets(&mut self, n_sets: u32) {
        unsafe {
            if self.ka_active {
                self.device.wait_for_fences(&[self.ka_fence], true, u64::MAX).unwrap();
                self.device.reset_fences(&[self.ka_fence]).unwrap();
                self.ka_active = false;
            }
            if self.desc_pool != vk::DescriptorPool::null() {
                self.device.destroy_descriptor_pool(self.desc_pool, None);
                self.desc_pool = vk::DescriptorPool::null();
            }
            let pool_sz = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: n_sets * 5,
            }];
            self.desc_pool = self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(n_sets)
                    .pool_sizes(&pool_sz),
                None).unwrap();

            self.device.reset_command_buffer(
                self.cmd_buf, vk::CommandBufferResetFlags::empty()).unwrap();
            self.device.begin_command_buffer(self.cmd_buf,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
            if self.debug_gpu {
                self.device.cmd_reset_query_pool(self.cmd_buf, self.ts_pool, 0, 128);
            }
            self.ts_count = 0;
            self.recording = true;
        }
    }

    pub fn timestamp(&mut self) {
        if !self.debug_gpu { return; }
        if self.ts_count < 128 {
            unsafe {
                self.device.cmd_write_timestamp(
                    self.cmd_buf, vk::PipelineStageFlags::COMPUTE_SHADER,
                    self.ts_pool, self.ts_count);
            }
            self.ts_count += 1;
        }
    }

    pub fn print_timestamps(&self) {
        if !self.debug_gpu || self.ts_count < 2 { return; }
        let mut data = vec![0u64; self.ts_count as usize];
        unsafe {
            self.device.get_query_pool_results(
                self.ts_pool, 0,
                &mut data, vk::QueryResultFlags::TYPE_64).ok();
        }
        let ns = self.ts_period;
        for i in 1..self.ts_count as usize {
            let dt = (data[i].wrapping_sub(data[i-1])) as f64 * ns as f64 / 1_000_000.0;
            eprint!("[ts{}-{}: {:.2}ms] ", i-1, i, dt);
        }
        let total = (data[self.ts_count as usize - 1].wrapping_sub(data[0])) as f64
            * ns as f64 / 1_000_000.0;
        eprintln!("[total GPU: {:.2}ms]", total);
    }

    pub fn cmd_upload_act(&mut self, act: &ActBuf, data: &[f32]) {
        unsafe {
            let size = (data.len() * 4) as u64;
            debug_assert!(size <= self.staging_size);
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8, self.staging_ptr, size as usize);
            self.device.cmd_copy_buffer(self.cmd_buf, self.staging_buf, act.buf,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size }]);
            let b = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(self.cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[b], &[], &[]);
        }
    }

    pub fn cmd_copy_to_act(&self, act: &ActBuf, src: vk::Buffer, src_offset: u64) {
        unsafe {
            self.device.cmd_copy_buffer(self.cmd_buf, src, act.buf,
                &[vk::BufferCopy { src_offset, dst_offset: 0, size: act.size }]);
            let b = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            self.device.cmd_pipeline_barrier(self.cmd_buf,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[b], &[], &[]);
        }
    }

    pub fn read_logits(&self, _logits_buf: &ActBuf, rb: &ActBuf) -> Vec<f32> {
        unsafe {
            let ptr = self.device.map_memory(rb.mem, 0, rb.size,
                vk::MemoryMapFlags::empty()).unwrap();
            let mut out = vec![0f32; (rb.size / 4) as usize];
            std::ptr::copy_nonoverlapping(
                ptr as *const u8, out.as_mut_ptr() as *mut u8, rb.size as usize);
            self.device.unmap_memory(rb.mem);
            out
        }
    }

    pub fn submit_with_readback(&mut self, logits_buf: &ActBuf, rb: &ActBuf) {
        unsafe {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.device.cmd_pipeline_barrier(self.cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[barrier], &[], &[]);
            self.device.cmd_copy_buffer(self.cmd_buf, logits_buf.buf, rb.buf,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: logits_buf.size }]);
            self.device.end_command_buffer(self.cmd_buf).unwrap();
            self.recording = false;
            self.device.queue_submit(self.queue,
                &[vk::SubmitInfo::default().command_buffers(&[self.cmd_buf])],
                self.fence).unwrap();

            if let Some((ka_buf, _)) = self.ka_buf {
                self.device.reset_command_buffer(
                    self.ka_cmd, vk::CommandBufferResetFlags::empty()).unwrap();
                self.device.begin_command_buffer(self.ka_cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();
                let (pipe, layout) = self.pipes[&Shader::Add];
                self.device.reset_descriptor_pool(
                    self.ka_desc_pool, vk::DescriptorPoolResetFlags::empty()).unwrap();
                let ds = alloc_ds(&self.device, self.ka_desc_pool, self.dsl3);
                let i = |b| [vk::DescriptorBufferInfo::default()
                    .buffer(b).offset(0).range(vk::WHOLE_SIZE)];
                let (i0, i1, i2) = (i(ka_buf), i(ka_buf), i(ka_buf));
                self.device.update_descriptor_sets(&[
                    wr(ds, 0, &i0), wr(ds, 1, &i1), wr(ds, 2, &i2),
                ], &[]);
                let pc: [u32; 1] = [4096];
                self.device.cmd_bind_pipeline(
                    self.ka_cmd, vk::PipelineBindPoint::COMPUTE, pipe);
                self.device.cmd_bind_descriptor_sets(
                    self.ka_cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[ds], &[]);
                self.device.cmd_push_constants(self.ka_cmd, layout,
                    vk::ShaderStageFlags::COMPUTE, 0,
                    std::slice::from_raw_parts(pc.as_ptr() as *const u8, 4));
                for _ in 0..128 {
                    self.device.cmd_dispatch(self.ka_cmd, 64, 1, 1);
                }
                self.device.end_command_buffer(self.ka_cmd).unwrap();
                self.device.queue_submit(self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.ka_cmd])],
                    self.ka_fence).unwrap();
                self.ka_active = true;
            }

            self.device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            self.device.reset_fences(&[self.fence]).unwrap();
        }
    }

    pub fn submit_no_readback(&mut self) {
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

    pub fn cmd_qk_norm(&mut self, x: &ActBuf, w: &ActBuf, n_heads: u32, head_dim: u32, eps: f32) {
        let pc: [u32; 2] = [head_dim, eps.to_bits()];
        let ds = self.ds3(x.buf, w.buf, x.buf);
        self.enc3(Shader::QkNorm, ds, &pc, n_heads, 1, 1);
    }

    pub fn cmd_gemv(&mut self, wt: &GpuTensor, inp: &ActBuf, out: &ActBuf) {
        let pc: [u32; 3] = [wt.rows, wt.bpr, wt.row_start];
        let ds = self.ds3(wt.buf, inp.buf, out.buf);
        self.enc3(wt.shader, ds, &pc, wt.rows, 1, 1);
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
        let total = (n_heads + n_kv_heads) * (head_dim / 2);
        self.enc3(Shader::Rope, ds, bytemuck::cast_slice(&pc), total.div_ceil(64), 1, 1);
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
                          seq_len: u32, n_ctx: u32, softcap: f32) {
        let pc: [u32; 6] = [n_heads, n_kv_heads, head_dim, seq_len, n_ctx, softcap.to_bits()];
        let ds = self.ds5(q.buf, kc.buf, vc.buf, ao.buf, scores.buf);
        self.enc5(Shader::Attn, ds, bytemuck::cast_slice(&pc), n_heads, 1, 1);
    }

    pub fn cmd_swiglu(&mut self, gate: &ActBuf, up: &ActBuf, n: u32, use_gelu: bool) {
        let pc: [u32; 2] = [n, use_gelu as u32];
        let ds = self.ds3(gate.buf, up.buf, gate.buf);
        self.enc3(Shader::SwiGlu, ds, bytemuck::cast_slice(&pc), n.div_ceil(64), 1, 1);
    }

    #[allow(dead_code)]
    pub fn cmd_add_rmsnorm(&mut self, res: &ActBuf, add_buf: &ActBuf,
                           w: &ActBuf, out: &ActBuf, n: u32, eps: f32) {
        let pc: [u32; 2] = [n, eps.to_bits()];
        let ds = self.ds4(res.buf, add_buf.buf, w.buf, out.buf);
        self.enc4(Shader::AddRmsNorm, ds, &pc, 1, 1, 1);
    }

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

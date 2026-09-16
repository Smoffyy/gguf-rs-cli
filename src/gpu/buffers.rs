use ash::vk;
use crate::tensor::dequant::QuantTensor;
use crate::gguf::types::GgmlType;
use super::*;

impl VkCtx {
    pub fn upload(&mut self, wt: &QuantTensor) -> Option<GpuTensor> {
        let (packed, shader, bpr) = match wt.typ {
            GgmlType::Q4_0                  => (wt.pack_q4_0_for_gpu(), Shader::Q4_0, (wt.cols/32) as u32),
            GgmlType::Q4_1                  => (wt.pack_q4_1_for_gpu(), Shader::Q4_1, (wt.cols/32) as u32),
            GgmlType::Q4K if wt.cols%256==0 => (wt.pack_q4k_for_gpu(), Shader::Q4K,  (wt.cols/256) as u32),
            GgmlType::Q3K if wt.cols%256==0 => (wt.pack_q3k_for_gpu(), Shader::Q3K,  (wt.cols/256) as u32),
            GgmlType::Q5K if wt.cols%256==0 => (wt.pack_q5k_for_gpu(), Shader::Q5K,  (wt.cols/256) as u32),
            GgmlType::Q6K if wt.cols%256==0 => (wt.pack_q6k_for_gpu(), Shader::Q6K,  (wt.cols/256) as u32),
            GgmlType::Q8_0 if wt.cols%32==0 => (wt.pack_q8_0_for_gpu(),Shader::Q8_0, (wt.cols/32) as u32),
            _ => return None,
        };
        let size = packed.len() as u64 * 4;
        if size > self.max_buf { return None; }
        let (buf, mem) = self.upload_bytes(size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            bytemuck::cast_slice(&packed)).ok()?;
        Some(GpuTensor { buf, _mem: mem, rows: wt.rows as u32, bpr, shader, row_start: 0 })
    }

    pub fn upload_any(&mut self, wt: &QuantTensor) -> Option<GpuTensor> {
        self.upload(wt).or_else(|| {
            let f32d = wt.to_f32();
            self.upload_f32_weight(&f32d, wt.rows as u32, wt.cols as u32)
        })
    }

    pub fn upload_f32_weight(&mut self, data: &[f32], rows: u32, cols: u32) -> Option<GpuTensor> {
        let size = data.len() as u64 * 4;
        let (buf, mem) = self.upload_bytes(size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            bytemuck::cast_slice(data)).ok()?;
        Some(GpuTensor { buf, _mem: mem, rows, bpr: cols, shader: Shader::F32, row_start: 0 })
    }

    pub fn upload_f32_host_visible(&mut self, data: &[f32]) -> (vk::Buffer, vk::DeviceMemory) {
        unsafe {
            let size = data.len() as u64 * 4;
            let (buf, mem) = self.alloc_raw(
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                self.host_idx,
            ).unwrap();
            let ptr = self.device.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr as *mut u8, size as usize);
            self.device.unmap_memory(mem);
            (buf, mem)
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

    pub fn free_buffer(&self, buf: vk::Buffer, mem: vk::DeviceMemory) {
        unsafe {
            self.device.destroy_buffer(buf, None);
            self.device.free_memory(mem, None);
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
}

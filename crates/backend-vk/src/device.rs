//! Vulkan instance, device, memory and command submission.

use ash::{vk, Device, Entry, Instance};
use gguf_core::{backend_err, Caps, DeviceInfo, DeviceKind, Error, Result};

use crate::Buf;

pub struct Context {
    _entry: Entry,
    pub instance: Instance,
    pub device: Device,
    pub queue: vk::Queue,
    pub info: DeviceInfo,
    device_mem: u32,
    host_mem: u32,
    /// Staging buffer for host transfers, grown on demand.
    staging: std::cell::RefCell<Option<Buf>>,
    transfer_pool: vk::CommandPool,
}

fn score(kind: vk::PhysicalDeviceType) -> u32 {
    match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => 4,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 1,
        _ => 0,
    }
}

fn device_info(instance: &Instance, pd: vk::PhysicalDevice, index: u32) -> DeviceInfo {
    let props = unsafe { instance.get_physical_device_properties(pd) };
    let mem = unsafe { instance.get_physical_device_memory_properties(pd) };
    let name = unsafe { std::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    // Device-local heaps are the ones that matter for "will the model fit".
    let total: u64 = (0..mem.memory_heap_count as usize)
        .filter(|i| mem.memory_heaps[*i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|i| mem.memory_heaps[i].size)
        .sum();
    DeviceInfo {
        kind: DeviceKind::Vulkan,
        index,
        name,
        total_memory: total,
        free_memory: total,
        caps: Caps {
            // The portable shader path is f32; integer dot needs an extension this backend
            // deliberately does not require.
            int8_dot: false,
            f16_math: false,
            matrix_cores: false,
            max_threads_per_block: props.limits.max_compute_work_group_invocations,
            shared_mem_per_block: props.limits.max_compute_shared_memory_size,
            warp_size: 32,
            multiprocessors: 0,
        },
    }
}

/// Every Vulkan device this process can see.
pub fn enumerate() -> Vec<DeviceInfo> {
    let Ok(entry) = (unsafe { Entry::load() }) else { return Vec::new() };
    let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 0, 0));
    let Ok(instance) =
        (unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) })
    else {
        return Vec::new();
    };
    let devices = unsafe { instance.enumerate_physical_devices() }.unwrap_or_default();
    let mut out: Vec<DeviceInfo> = devices
        .iter()
        .enumerate()
        .map(|(i, pd)| device_info(&instance, *pd, i as u32))
        .collect();
    // Report them in the same preference order selection uses, so index 0 is what `auto`
    // would have picked.
    out.sort_by_key(|d| std::cmp::Reverse(d.total_memory));
    unsafe { instance.destroy_instance(None) };
    for (i, d) in out.iter_mut().enumerate() {
        d.index = i as u32;
    }
    out
}

impl Context {
    pub fn new(index: u32) -> Result<Self> {
        unsafe {
            let entry = Entry::load().map_err(|e| {
                Error::NoDevice(format!(
                    "no Vulkan loader found ({e}); install a GPU driver with Vulkan support"
                ))
            })?;
            let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 0, 0));
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
                .map_err(|e| backend_err("vulkan", format!("creating an instance: {e}")))?;

            let mut devices = instance
                .enumerate_physical_devices()
                .map_err(|e| backend_err("vulkan", format!("enumerating devices: {e}")))?;
            if devices.is_empty() {
                return Err(Error::NoDevice("Vulkan reports no devices".into()));
            }
            // Prefer discrete, then by memory, so index 0 is the best device present.
            devices.sort_by_key(|pd| {
                let props = instance.get_physical_device_properties(*pd);
                let mem = instance.get_physical_device_memory_properties(*pd);
                let total: u64 = (0..mem.memory_heap_count as usize)
                    .filter(|i| mem.memory_heaps[*i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
                    .map(|i| mem.memory_heaps[i].size)
                    .sum();
                std::cmp::Reverse((score(props.device_type), total))
            });
            let physical = *devices.get(index as usize).ok_or_else(|| {
                Error::NoDevice(format!(
                    "vulkan:{index} was requested but only {} device(s) are present",
                    devices.len()
                ))
            })?;

            let families = instance.get_physical_device_queue_family_properties(physical);
            let queue_family = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| Error::NoDevice("this Vulkan device has no compute queue".into()))?
                as u32;

            let priorities = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&priorities)];
            let device = instance
                .create_device(physical, &vk::DeviceCreateInfo::default().queue_create_infos(&qci), None)
                .map_err(|e| backend_err("vulkan", format!("creating a logical device: {e}")))?;
            let queue = device.get_device_queue(queue_family, 0);

            let mem_props = instance.get_physical_device_memory_properties(physical);
            let device_mem = find_memory(&mem_props, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
            let host_mem = find_memory(
                &mem_props,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;

            let transfer_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                        .queue_family_index(queue_family),
                    None,
                )
                .map_err(|e| backend_err("vulkan", format!("creating a command pool: {e}")))?;

            let info = device_info(&instance, physical, index);
            Ok(Self {
                _entry: entry,
                instance,
                device,
                queue,
                info,
                device_mem,
                host_mem,
                staging: std::cell::RefCell::new(None),
                transfer_pool,
            })
        }
    }

    fn alloc_raw(&self, bytes: u64, usage: vk::BufferUsageFlags, mem_type: u32) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        unsafe {
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(bytes.max(16))
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .map_err(|e| backend_err("vulkan", format!("creating a buffer: {e}")))?;
            let reqs = self.device.get_buffer_memory_requirements(buffer);
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(mem_type),
                    None,
                )
                .map_err(|_| Error::OutOfMemory {
                    device: self.info.id(),
                    requested: reqs.size,
                    available: self.info.total_memory,
                })?;
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| backend_err("vulkan", format!("binding buffer memory: {e}")))?;
            Ok((buffer, memory))
        }
    }

    pub fn alloc_device(&self, bytes: u64) -> Result<Buf> {
        let (buffer, memory) = self.alloc_raw(
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            self.device_mem,
        )?;
        Ok(Buf { buffer, memory, bytes: bytes.max(16), mapped: std::ptr::null_mut() })
    }

    /// Host-visible and permanently mapped, for the small per-pass inputs.
    pub fn alloc_host(&self, bytes: u64) -> Result<Buf> {
        let (buffer, memory) = self.alloc_raw(
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            self.host_mem,
        )?;
        let mapped = unsafe {
            self.device
                .map_memory(memory, 0, bytes.max(16), vk::MemoryMapFlags::empty())
                .map_err(|e| backend_err("vulkan", format!("mapping host memory: {e}")))? as *mut u8
        };
        Ok(Buf { buffer, memory, bytes: bytes.max(16), mapped })
    }

    pub fn free_buf(&self, b: &Buf) {
        if b.buffer == vk::Buffer::null() {
            return;
        }
        unsafe {
            if !b.mapped.is_null() {
                self.device.unmap_memory(b.memory);
            }
            self.device.destroy_buffer(b.buffer, None);
            self.device.free_memory(b.memory, None);
        }
    }

    pub fn zero(&self, b: &Buf) -> Result<()> {
        let cb = self.one_shot_begin()?;
        unsafe {
            self.device.cmd_fill_buffer(cb, b.buffer, 0, b.bytes, 0);
        }
        self.one_shot_end(cb)
    }

    fn staging_buf(&self, bytes: u64) -> Result<vk::Buffer> {
        let mut slot = self.staging.borrow_mut();
        let need = bytes.next_power_of_two().max(1 << 20);
        if slot.as_ref().map_or(true, |s| s.bytes < bytes) {
            if let Some(old) = slot.take() {
                self.free_buf(&old);
            }
            *slot = Some(self.alloc_host(need)?);
        }
        Ok(slot.as_ref().unwrap().buffer)
    }

    pub fn upload(&self, dst: vk::Buffer, offset: u64, data: &[u8]) -> Result<()> {
        // Large weights are copied in chunks so the staging buffer does not have to match
        // the size of the largest tensor in the model.
        const CHUNK: usize = 64 << 20;
        let mut written = 0usize;
        while written < data.len() {
            let n = CHUNK.min(data.len() - written);
            let staging = self.staging_buf(n as u64)?;
            {
                let slot = self.staging.borrow();
                let s = slot.as_ref().unwrap();
                // SAFETY: host-coherent and mapped for its lifetime, and no device work is
                // in flight against it because every caller syncs first.
                unsafe {
                    std::ptr::copy_nonoverlapping(data[written..].as_ptr(), s.mapped, n);
                }
            }
            let cb = self.one_shot_begin()?;
            unsafe {
                self.device.cmd_copy_buffer(
                    cb,
                    staging,
                    dst,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: offset + written as u64,
                        size: n as u64,
                    }],
                );
            }
            self.one_shot_end(cb)?;
            written += n;
        }
        Ok(())
    }

    pub fn download(&self, src: vk::Buffer, offset: u64, out: &mut [u8]) -> Result<()> {
        let staging = self.staging_buf(out.len() as u64)?;
        let cb = self.one_shot_begin()?;
        unsafe {
            self.device.cmd_copy_buffer(
                cb,
                src,
                staging,
                &[vk::BufferCopy { src_offset: offset, dst_offset: 0, size: out.len() as u64 }],
            );
        }
        self.one_shot_end(cb)?;
        let slot = self.staging.borrow();
        let s = slot.as_ref().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(s.mapped, out.as_mut_ptr(), out.len());
        }
        Ok(())
    }

    fn one_shot_begin(&self) -> Result<vk::CommandBuffer> {
        unsafe {
            let cb = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.transfer_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| backend_err("vulkan", format!("allocating a command buffer: {e}")))?[0];
            self.device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| backend_err("vulkan", format!("beginning a command buffer: {e}")))?;
            Ok(cb)
        }
    }

    fn one_shot_end(&self, cb: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.device
                .end_command_buffer(cb)
                .map_err(|e| backend_err("vulkan", format!("ending a command buffer: {e}")))?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[submit], vk::Fence::null())
                .map_err(|e| backend_err("vulkan", format!("submitting: {e}")))?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|e| backend_err("vulkan", format!("waiting for the queue: {e}")))?;
            self.device.free_command_buffers(self.transfer_pool, &[cb]);
        }
        Ok(())
    }

    pub fn create_recording_state(&self) -> Result<(vk::CommandBuffer, vk::Fence, vk::DescriptorPool)> {
        unsafe {
            let cb = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.transfer_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(|e| backend_err("vulkan", format!("allocating a command buffer: {e}")))?[0];
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| backend_err("vulkan", format!("creating a fence: {e}")))?;
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: crate::pipeline::MAX_SETS * 6,
            }];
            let pool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(crate::pipeline::MAX_SETS)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(|e| backend_err("vulkan", format!("creating a descriptor pool: {e}")))?;
            Ok((cb, fence, pool))
        }
    }

    pub fn destroy_recording_state(&self, cb: vk::CommandBuffer, fence: vk::Fence, pool: vk::DescriptorPool) {
        unsafe {
            self.device.free_command_buffers(self.transfer_pool, &[cb]);
            self.device.destroy_fence(fence, None);
            self.device.destroy_descriptor_pool(pool, None);
        }
    }

    pub fn allocate_set(&self, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        unsafe {
            let layouts = [layout];
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(&layouts);
            Ok(self
                .device
                .allocate_descriptor_sets(&info)
                .map_err(|e| backend_err("vulkan", format!("allocating a descriptor set: {e}")))?[0])
        }
    }

    pub fn write_set(&self, set: vk::DescriptorSet, buffers: &[vk::Buffer]) {
        let infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|b| vk::DescriptorBufferInfo {
                buffer: *b,
                offset: 0,
                range: vk::WHOLE_SIZE,
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn begin_command_buffer(&self, cb: vk::CommandBuffer) -> Result<()> {
        unsafe {
            self.device
                .reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())
                .map_err(|e| backend_err("vulkan", format!("resetting a command buffer: {e}")))?;
            self.device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| backend_err("vulkan", format!("beginning a command buffer: {e}")))
        }
    }

    /// A full shader-write to shader-read barrier.
    ///
    /// SAFETY: must be called while `cb` is recording.
    pub unsafe fn barrier(&self, cb: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ);
        self.device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }

    pub fn end_and_submit(&self, cb: vk::CommandBuffer, fence: vk::Fence) -> Result<()> {
        unsafe {
            self.device
                .end_command_buffer(cb)
                .map_err(|e| backend_err("vulkan", format!("ending a command buffer: {e}")))?;
            self.device
                .reset_fences(&[fence])
                .map_err(|e| backend_err("vulkan", format!("resetting a fence: {e}")))?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            self.device
                .queue_submit(self.queue, &[submit], fence)
                .map_err(|e| backend_err("vulkan", format!("submitting: {e}")))
        }
    }

    pub fn wait(&self, fence: vk::Fence) -> Result<()> {
        unsafe {
            // The caller guarantees work was submitted with this fence; waiting on one that
            // never was would block forever.
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| backend_err("vulkan", format!("waiting for the device: {e}")))
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(s) = self.staging.borrow_mut().take() {
                self.free_buf(&s);
            }
            self.device.destroy_command_pool(self.transfer_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn find_memory(props: &vk::PhysicalDeviceMemoryProperties, flags: vk::MemoryPropertyFlags) -> Result<u32> {
    (0..props.memory_type_count)
        .find(|i| props.memory_types[*i as usize].property_flags.contains(flags))
        .ok_or_else(|| backend_err("vulkan", format!("no memory type with {flags:?}")))
}

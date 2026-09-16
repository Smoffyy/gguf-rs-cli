use ash::{vk, Entry, Device};
use super::*;

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
            for (shader, spv, dsl) in [
                (Shader::Q4_0,    SPV_Q4_0,    dsl3),
                (Shader::Q4_1,    SPV_Q4_1,    dsl3),
                (Shader::Q4K,     SPV_Q4K,     dsl3),
                (Shader::Q3K,     SPV_Q3K,     dsl3),
                (Shader::Q5K,     SPV_Q5K,     dsl3),
                (Shader::Q6K,     SPV_Q6K,     dsl3),
                (Shader::Q8_0,    SPV_Q8_0,    dsl3),
                (Shader::F32,     SPV_F32,     dsl3),
                (Shader::RmsNorm, SPV_RMSNORM, dsl3),
                (Shader::Rope,    SPV_ROPE,    dsl3),
                (Shader::KvWrite, SPV_KVWRITE, dsl4),
                (Shader::Attn,    SPV_ATTN,    dsl5),
                (Shader::SwiGlu,  SPV_SWIGLU,  dsl3),
                (Shader::Add,     SPV_ADD,     dsl3),
                (Shader::AddRmsNorm, SPV_ADD_RN, dsl4),
                (Shader::QkNorm,     SPV_QK_NORM, dsl3),
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
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(2000)
                    .pool_sizes(&pool_sz),
                None)?;

            let cmd_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_fam), None)?;
            let cmd_bufs = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(2))?;
            let cmd_buf = cmd_bufs[0];
            let ka_cmd  = cmd_bufs[1];
            let fence    = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let ka_fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;

            let ka_size = 16 * 1024u64;
            let ka_buf_obj = device.create_buffer(
                &vk::BufferCreateInfo::default().size(ka_size)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
            let ka_reqs = device.get_buffer_memory_requirements(ka_buf_obj);
            let ka_mem = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(ka_reqs.size).memory_type_index(dev_idx), None)?;
            device.bind_buffer_memory(ka_buf_obj, ka_mem, 0)?;

            let ka_pool_sz = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 }];
            let ka_desc_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(1)
                    .pool_sizes(&ka_pool_sz),
                None)?;

            let staging_size = 256 * 1024u64;
            let (staging_buf, _staging_mem) = {
                let buf = device.create_buffer(
                    &vk::BufferCreateInfo::default().size(staging_size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE), None)?;
                let reqs = device.get_buffer_memory_requirements(buf);
                let mem = device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size).memory_type_index(host_idx), None)?;
                device.bind_buffer_memory(buf, mem, 0)?;
                (buf, mem)
            };
            let staging_ptr = device.map_memory(
                _staging_mem, 0, staging_size, vk::MemoryMapFlags::empty())? as *mut u8;

            let ts_period = props.limits.timestamp_period;
            let ts_pool = device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(128), None)?;

            Ok(Self {
                _entry: entry, device, queue, pipes, dsl3, dsl4, dsl5,
                desc_pool, cmd_pool, cmd_buf, fence, recording: false,
                max_buf, dev_idx, host_idx, device_name: name.into_owned(),
                staging_buf, _staging_mem, staging_ptr, staging_size,
                ts_pool, ts_period, ts_count: 0,
                debug_gpu: false,
                ka_cmd, ka_fence, ka_desc_pool, ka_buf: Some((ka_buf_obj, ka_mem)), ka_active: false,
            })
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

unsafe fn make_module(device: &Device, spv: &[u8]) -> anyhow::Result<vk::ShaderModule> {
    let (p, a, s) = spv.align_to::<u32>();
    assert!(p.is_empty() && s.is_empty());
    Ok(device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(a), None)?)
}

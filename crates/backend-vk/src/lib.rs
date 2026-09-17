//! Vulkan backend.
//!
//! This is the portable GPU path: AMD, Intel, NVIDIA, Apple through MoltenVK, and mobile,
//! from one set of SPIR-V shaders targeting Vulkan 1.0 with no optional extensions. It is
//! what makes "runs on the GPU" true on hardware CUDA will never reach.
//!
//! Weights live in device memory in their GGUF block layout, exactly as on the CUDA side,
//! and the shaders decode blocks into registers as they read them.

mod device;
mod pipeline;

use std::collections::HashMap;

use ash::vk;
use gguf_core::{
    backend_err, Activation, AttnCfg, Backend, BinOp, BufferId, DType, DeviceInfo, Error, GluCfg,
    KvView, MatMulCfg, MemKind, MoeCfg, MoeWeights, NormCfg, NormKind, QuantView, Result, RopeCfg,
    RopeKind, WeightId,
};

pub use device::enumerate;
use pipeline::{Kernel, Pipelines};

/// Bytes a push-constant block may occupy. Vulkan guarantees 128.
const PUSH_MAX: usize = 128;

struct Buf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    bytes: u64,
    /// Non-null when the allocation is host-visible and permanently mapped.
    mapped: *mut u8,
}

struct VkWeight {
    buf: Buf,
    dtype: DType,
    rows: usize,
    cols: usize,
}

impl VkWeight {
    fn row_bytes(&self) -> usize {
        self.dtype.row_bytes(self.cols)
    }
}

pub struct VulkanBackend {
    ctx: device::Context,
    pipes: Pipelines,
    info: DeviceInfo,

    buffers: Vec<Option<Buf>>,
    free_buffers: Vec<u32>,
    weights: Vec<Option<VkWeight>>,
    free_weights: Vec<u32>,

    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    recording: bool,
    /// Whether the fence has work to wait on. Waiting on a fence nothing was ever
    /// submitted with blocks forever, and `sync` is called on paths that recorded nothing.
    pending: bool,
    /// Descriptor sets are allocated per dispatch and the whole pool is reset on submit.
    desc_pool: vk::DescriptorPool,
    sets_used: u32,

    /// A valid buffer to bind where a shader declares an input the graph did not supply.
    /// Vulkan has no notion of an unbound descriptor without an optional extension.
    null_buf: Buf,
    /// Small host-visible buffers written directly by the CPU: token ids and positions.
    tokens_buf: Buf,
    positions_buf: Buf,
    /// Device-local scratch for the MoE intermediates.
    moe_scratch: Vec<Buf>,
}

fn act_code(a: Activation) -> u32 {
    match a {
        Activation::Silu => 0,
        Activation::Gelu => 1,
        Activation::GeluTanh => 2,
        Activation::GeluQuick => 3,
        Activation::Relu => 4,
        Activation::Relu2 => 5,
        Activation::Swish => 6,
    }
}

/// Pack a push-constant struct into the byte block the API expects.
fn push<T: Copy>(v: &T) -> [u8; PUSH_MAX] {
    let mut out = [0u8; PUSH_MAX];
    let size = std::mem::size_of::<T>();
    assert!(size <= PUSH_MAX, "push constants exceed the guaranteed 128 bytes");
    // SAFETY: T is Copy and plain-old-data; the destination is large enough.
    unsafe {
        std::ptr::copy_nonoverlapping(v as *const T as *const u8, out.as_mut_ptr(), size);
    }
    out
}

impl VulkanBackend {
    pub fn new(index: u32) -> Result<Self> {
        let ctx = device::Context::new(index)?;
        let info = ctx.info.clone();
        let pipes = Pipelines::build(&ctx)?;

        let (cmd, fence, desc_pool) = ctx.create_recording_state()?;

        let mut be = Self {
            null_buf: Buf { buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(), bytes: 0, mapped: std::ptr::null_mut() },
            tokens_buf: Buf { buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(), bytes: 0, mapped: std::ptr::null_mut() },
            positions_buf: Buf { buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(), bytes: 0, mapped: std::ptr::null_mut() },
            moe_scratch: Vec::new(),
            ctx,
            pipes,
            info,
            buffers: Vec::new(),
            free_buffers: Vec::new(),
            weights: Vec::new(),
            free_weights: Vec::new(),
            cmd,
            fence,
            recording: false,
            pending: false,
            desc_pool,
            sets_used: 0,
        };
        be.null_buf = be.ctx.alloc_device(1024)?;
        be.tokens_buf = be.ctx.alloc_host(4096)?;
        be.positions_buf = be.ctx.alloc_host(4096)?;
        Ok(be)
    }

    fn buf(&self, id: BufferId) -> Result<&Buf> {
        self.buffers
            .get(id.0 as usize)
            .and_then(|b| b.as_ref())
            .ok_or_else(|| backend_err("vulkan", format!("buffer {} is not allocated", id.0)))
    }

    fn handle(&self, id: BufferId) -> Result<vk::Buffer> {
        Ok(self.buf(id)?.buffer)
    }

    fn opt_handle(&self, id: Option<BufferId>) -> Result<vk::Buffer> {
        match id {
            Some(b) => self.handle(b),
            None => Ok(self.null_buf.buffer),
        }
    }

    fn weight(&self, id: WeightId) -> Result<&VkWeight> {
        self.weights
            .get(id.0 as usize)
            .and_then(|w| w.as_ref())
            .ok_or_else(|| backend_err("vulkan", format!("weight {} is not uploaded", id.0)))
    }

    fn begin(&mut self) -> Result<()> {
        if self.recording {
            return Ok(());
        }
        self.ctx.begin_command_buffer(self.cmd)?;
        self.recording = true;
        Ok(())
    }

    /// Record a dispatch, inserting a full barrier first so the previous op's writes are
    /// visible.
    ///
    /// A conservative barrier between every op rather than a dependency graph: the ops in a
    /// transformer layer are almost entirely sequential anyway, so tracking finer
    /// dependencies would buy little and is a rich source of races.
    fn dispatch(
        &mut self,
        kernel: Kernel,
        buffers: &[vk::Buffer],
        pc: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        self.begin()?;
        let (pipeline, layout, set_layout) = self.pipes.get(kernel)?;

        if self.sets_used >= pipeline::MAX_SETS {
            // The pool is a fixed size; flushing is cheaper than growing it unboundedly.
            self.submit()?;
            self.sync()?;
            self.begin()?;
        }
        let set = self.ctx.allocate_set(self.desc_pool, set_layout)?;
        self.sets_used += 1;
        self.ctx.write_set(set, buffers);

        unsafe {
            self.ctx.barrier(self.cmd);
            self.ctx.device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.ctx.device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            self.ctx.device.cmd_push_constants(
                self.cmd,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc,
            );
            self.ctx.device.cmd_dispatch(self.cmd, groups.0.max(1), groups.1.max(1), groups.2.max(1));
        }
        Ok(())
    }

    /// Grow a device-local MoE scratch slot to at least `bytes`.
    fn moe_buf(&mut self, slot: usize, bytes: u64) -> Result<vk::Buffer> {
        while self.moe_scratch.len() <= slot {
            self.moe_scratch.push(self.ctx.alloc_device(256)?);
        }
        if self.moe_scratch[slot].bytes < bytes {
            let old = std::mem::replace(&mut self.moe_scratch[slot], self.ctx.alloc_device(bytes.next_power_of_two())?);
            self.ctx.free_buf(&old);
        }
        Ok(self.moe_scratch[slot].buffer)
    }

    /// Write a small slice into a permanently-mapped host-visible buffer.
    ///
    /// Safe without synchronisation because every forward pass ends in a readback, which
    /// waits for the device, so nothing from a previous pass can still be reading it.
    fn write_host(buf: &mut Buf, ctx: &device::Context, data: &[u8]) -> Result<()> {
        if buf.bytes < data.len() as u64 {
            let old = std::mem::replace(buf, ctx.alloc_host((data.len() as u64).next_power_of_two())?);
            ctx.free_buf(&old);
        }
        // SAFETY: the allocation is host-coherent and mapped for its whole lifetime.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buf.mapped, data.len());
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatmulPc {
    n_in: u32,
    n_out: u32,
    n_tokens: u32,
    row_bytes: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GetRowsPc {
    n_rows: u32,
    cols: u32,
    row_bytes: u32,
    scale: f32,
    n_tokens: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormPc {
    dim: u32,
    kind: u32,
    eps: f32,
    scale: f32,
    has_weight: u32,
    has_bias: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopePc {
    n_heads: u32,
    head_dim: u32,
    n_rot: u32,
    n_tokens: u32,
    kind: u32,
    has_ff: u32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    corr_lo: f32,
    corr_hi: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct KvWritePc {
    dim: u32,
    stride: u32,
    base: u32,
    start_pos: u32,
    n_tokens: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnPc {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    n_tokens: u32,
    kv_len: u32,
    start_pos: u32,
    stride: u32,
    base: u32,
    scale: f32,
    softcap: f32,
    window: u32,
    n_splits: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnCombinePc {
    n_heads: u32,
    head_dim: u32,
    n_tokens: u32,
    n_splits: u32,
    has_sinks: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ElementwisePc {
    n: u32,
    bn: u32,
    op: u32,
    act: u32,
    limit: f32,
    alpha: f32,
    value: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MoeRoutePc {
    n_expert: u32,
    n_used: u32,
    n_tokens: u32,
    gate_func: u32,
    norm_topk: u32,
    has_bias: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MoeMmPc {
    n_in: u32,
    n_out: u32,
    n_used: u32,
    n_tokens: u32,
    row_bytes: u32,
    expert_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MoeMiscPc {
    n: u32,
    dim: u32,
    n_used: u32,
    n_tokens: u32,
    op: u32,
    act: u32,
    has_gate: u32,
    alpha: f32,
}

const WG: u32 = 256;

fn groups(n: u32, size: u32) -> (u32, u32, u32) {
    ((n + size - 1) / size, 1, 1)
}

impl Backend for VulkanBackend {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn alloc(&mut self, bytes: u64, _kind: MemKind) -> Result<BufferId> {
        let buf = self.ctx.alloc_device(bytes.max(16))?;
        self.ctx.zero(&buf)?;
        if let Some(idx) = self.free_buffers.pop() {
            self.buffers[idx as usize] = Some(buf);
            Ok(BufferId(idx))
        } else {
            self.buffers.push(Some(buf));
            Ok(BufferId(self.buffers.len() as u32 - 1))
        }
    }

    fn free(&mut self, id: BufferId) {
        if let Some(slot) = self.buffers.get_mut(id.0 as usize) {
            if let Some(b) = slot.take() {
                self.ctx.free_buf(&b);
                self.free_buffers.push(id.0);
            }
        }
    }

    fn write(&mut self, dst: BufferId, offset_bytes: u64, src: &[u8]) -> Result<()> {
        self.sync()?;
        let buffer = self.handle(dst)?;
        self.ctx.upload(buffer, offset_bytes, src)
    }

    fn read(&mut self, src: BufferId, offset_bytes: u64, dst: &mut [u8]) -> Result<()> {
        self.sync()?;
        let buffer = self.handle(src)?;
        self.ctx.download(buffer, offset_bytes, dst)
    }

    fn upload_weight(&mut self, view: &QuantView<'_>) -> Result<WeightId> {
        view.dtype.check_supported()?;
        if !pipeline::supports(view.dtype) {
            return Err(Error::Unsupported(format!(
                "{} has no Vulkan shader; run this model on --device cpu",
                view.dtype.name()
            )));
        }
        let buf = self.ctx.alloc_device(view.data.len() as u64)?;
        self.ctx.upload(buf.buffer, 0, view.data)?;
        let w = VkWeight { buf, dtype: view.dtype, rows: view.rows, cols: view.cols };
        if let Some(idx) = self.free_weights.pop() {
            self.weights[idx as usize] = Some(w);
            Ok(WeightId(idx))
        } else {
            self.weights.push(Some(w));
            Ok(WeightId(self.weights.len() as u32 - 1))
        }
    }

    fn free_weight(&mut self, id: WeightId) {
        if let Some(slot) = self.weights.get_mut(id.0 as usize) {
            if let Some(w) = slot.take() {
                self.ctx.free_buf(&w.buf);
                self.free_weights.push(id.0);
            }
        }
    }

    fn get_rows(&mut self, dst: BufferId, weight: WeightId, tokens: &[u32], scale: f32) -> Result<()> {
        let (wbuf, dtype, rows, cols, row_bytes) = {
            let w = self.weight(weight)?;
            (w.buf.buffer, w.dtype, w.rows, w.cols, w.row_bytes())
        };
        let raw: Vec<u8> = tokens.iter().flat_map(|t| (*t as i32).to_ne_bytes()).collect();
        let mut tb = std::mem::replace(&mut self.tokens_buf, Buf { buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(), bytes: 0, mapped: std::ptr::null_mut() });
        Self::write_host(&mut tb, &self.ctx, &raw)?;
        let tokens_handle = tb.buffer;
        self.tokens_buf = tb;

        let d = self.handle(dst)?;
        let pc = push(&GetRowsPc {
            n_rows: rows as u32,
            cols: cols as u32,
            row_bytes: row_bytes as u32,
            scale,
            n_tokens: tokens.len() as u32,
        });
        self.dispatch(
            Kernel::GetRows(dtype),
            &[wbuf, tokens_handle, d],
            &pc,
            (tokens.len() as u32, 1, 1),
        )
    }

    fn norm(&mut self, dst: BufferId, src: BufferId, weight: Option<BufferId>, cfg: NormCfg) -> Result<()> {
        let (d, s) = (self.handle(dst)?, self.handle(src)?);
        let w = self.opt_handle(weight)?;
        let b = self.opt_handle(cfg.bias)?;
        let pc = push(&NormPc {
            dim: cfg.dim,
            kind: if cfg.kind == NormKind::Layer { 1 } else { 0 },
            eps: cfg.eps,
            scale: cfg.scale,
            has_weight: weight.is_some() as u32,
            has_bias: cfg.bias.is_some() as u32,
        });
        self.dispatch(Kernel::Norm, &[s, d, w, b], &pc, (cfg.n_tokens.max(1), 1, 1))
    }

    fn matmul(&mut self, dst: BufferId, weight: WeightId, src: BufferId, cfg: MatMulCfg) -> Result<()> {
        let (wbuf, dtype, row_bytes) = {
            let w = self.weight(weight)?;
            (w.buf.buffer, w.dtype, w.row_bytes())
        };
        let (d, s) = (self.handle(dst)?, self.handle(src)?);
        let bias = self.opt_handle(cfg.bias)?;
        let pc = push(&MatmulPc {
            n_in: cfg.n_in,
            n_out: cfg.n_out,
            n_tokens: cfg.n_tokens,
            row_bytes: row_bytes as u32,
            has_bias: cfg.bias.is_some() as u32,
        });
        self.dispatch(Kernel::MatMul(dtype), &[wbuf, s, d, bias], &pc, (cfg.n_out, 1, 1))
    }

    fn matmul_f32(&mut self, dst: BufferId, weight: BufferId, src: BufferId, cfg: MatMulCfg) -> Result<()> {
        let (d, w, s) = (self.handle(dst)?, self.handle(weight)?, self.handle(src)?);
        let bias = self.opt_handle(cfg.bias)?;
        let pc = push(&MatmulPc {
            n_in: cfg.n_in,
            n_out: cfg.n_out,
            n_tokens: cfg.n_tokens,
            row_bytes: cfg.n_in * 4,
            has_bias: cfg.bias.is_some() as u32,
        });
        // A dense f32 buffer is just an F32-typed weight as far as the shader is concerned.
        self.dispatch(Kernel::MatMul(DType::F32), &[w, s, d, bias], &pc, (cfg.n_out, 1, 1))
    }

    fn rope(&mut self, q: BufferId, k: Option<BufferId>, positions: &[i32], cfg: RopeCfg) -> Result<()> {
        if cfg.kind == RopeKind::None || cfg.n_rot == 0 {
            return Ok(());
        }
        let raw: Vec<u8> = positions.iter().flat_map(|p| p.to_ne_bytes()).collect();
        let mut pb = std::mem::replace(&mut self.positions_buf, Buf { buffer: vk::Buffer::null(), memory: vk::DeviceMemory::null(), bytes: 0, mapped: std::ptr::null_mut() });
        Self::write_host(&mut pb, &self.ctx, &raw)?;
        let pos_handle = pb.buffer;
        self.positions_buf = pb;

        let ff = self.opt_handle(cfg.freq_factors)?;
        let (corr_lo, corr_hi) = yarn_corr_dims(&cfg);
        // For a text sequence M-RoPE has one position axis, which makes it NeoX exactly.
        let kind = if cfg.kind == RopeKind::Norm { 0 } else { 1 };

        for (buf, heads) in [(Some(q), cfg.n_heads), (k, cfg.n_kv_heads)] {
            let Some(buf) = buf else { continue };
            let x = self.handle(buf)?;
            let pc = push(&RopePc {
                n_heads: heads,
                head_dim: cfg.head_dim,
                n_rot: cfg.n_rot,
                n_tokens: cfg.n_tokens,
                kind,
                has_ff: cfg.freq_factors.is_some() as u32,
                freq_base: cfg.freq_base,
                freq_scale: cfg.freq_scale,
                ext_factor: cfg.ext_factor,
                attn_factor: cfg.attn_factor,
                corr_lo,
                corr_hi,
            });
            let total = cfg.n_tokens * heads * (cfg.n_rot / 2);
            self.dispatch(Kernel::Rope, &[x, pos_handle, ff], &pc, groups(total, WG))?;
        }
        Ok(())
    }

    fn kv_write(
        &mut self,
        kv: &KvView,
        k: BufferId,
        v: BufferId,
        start_pos: u32,
        n_tokens: u32,
        dim: u32,
    ) -> Result<()> {
        let (kp, vp) = (self.handle(k)?, self.handle(v)?);
        let (kc, vc) = (self.handle(kv.k)?, self.handle(kv.v)?);
        let pc = push(&KvWritePc {
            dim,
            stride: kv.stride,
            base: kv.base,
            start_pos,
            n_tokens,
        });
        // f16 packs two values per word, so half as many invocations cover the same data.
        let per = if kv.is_f16() { 2 } else { 1 };
        self.dispatch(
            Kernel::KvWrite(kv.is_f16()),
            &[kp, vp, kc, vc],
            &pc,
            groups((dim * n_tokens) / per, WG),
        )
    }

    fn attention(&mut self, dst: BufferId, q: BufferId, kv: &KvView, cfg: AttnCfg) -> Result<()> {
        if cfg.head_dim > 256 {
            return Err(Error::Unsupported(format!(
                "head_dim {} exceeds the 256 this backend's attention shader stages in shared                  memory",
                cfg.head_dim
            )));
        }
        let hb = cfg.n_heads * cfg.n_tokens;
        // Enough splits to keep the device busy, but never so few keys per split that the
        // combine pass costs more than it saves.
        let n_splits = (2048u32 / hb.max(1)).clamp(1, 32).min((cfg.kv_len / 64).max(1)).max(1);

        let partial_bytes = (hb as u64) * n_splits as u64 * (cfg.head_dim as u64 + 2) * 4;
        let partial = self.moe_buf(7, partial_bytes)?;

        let (d, qh) = (self.handle(dst)?, self.handle(q)?);
        let (kc, vc) = (self.handle(kv.k)?, self.handle(kv.v)?);
        let sinks = self.opt_handle(cfg.sinks)?;

        let pc = push(&AttnPc {
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            n_tokens: cfg.n_tokens,
            kv_len: cfg.kv_len,
            start_pos: cfg.start_pos,
            stride: kv.stride,
            base: kv.base,
            scale: cfg.scale,
            softcap: cfg.softcap,
            window: cfg.window,
            n_splits,
        });
        self.dispatch(Kernel::Attention(kv.is_f16()), &[qh, kc, vc, partial], &pc, (hb, n_splits, 1))?;

        let pc = push(&AttnCombinePc {
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim,
            n_tokens: cfg.n_tokens,
            n_splits,
            has_sinks: cfg.sinks.is_some() as u32,
        });
        self.dispatch(Kernel::AttnCombine, &[partial, d, sinks], &pc, (hb, 1, 1))
    }

    fn glu(&mut self, dst: BufferId, gate: BufferId, up: BufferId, cfg: GluCfg) -> Result<()> {
        let n = cfg.dim * cfg.n_tokens;
        let (d, g, u) = (self.handle(dst)?, self.handle(gate)?, self.handle(up)?);
        let pc = push(&ElementwisePc {
            n,
            bn: n.max(1),
            op: 3,
            act: act_code(cfg.act),
            limit: cfg.limit,
            alpha: cfg.alpha,
            value: 0.0,
        });
        self.dispatch(Kernel::Elementwise, &[d, g, u], &pc, groups(n, WG))
    }

    fn activate(&mut self, buf: BufferId, cfg: GluCfg) -> Result<()> {
        let n = cfg.dim * cfg.n_tokens;
        let b = self.handle(buf)?;
        let null = self.null_buf.buffer;
        let pc = push(&ElementwisePc {
            n,
            bn: n.max(1),
            op: 4,
            act: act_code(cfg.act),
            limit: cfg.limit,
            alpha: cfg.alpha,
            value: 0.0,
        });
        self.dispatch(Kernel::Elementwise, &[b, null, null], &pc, groups(n, WG))
    }

    fn moe(&mut self, dst: BufferId, src: BufferId, w: &MoeWeights, cfg: MoeCfg) -> Result<()> {
        self.moe_impl(dst, src, w, cfg)
    }

    fn binary(&mut self, dst: BufferId, a: BufferId, b: BufferId, op: BinOp, n: u32) -> Result<()> {
        let bn = (self.buf(b)?.bytes / 4) as u32;
        let (d, x, y) = (self.handle(dst)?, self.handle(a)?, self.handle(b)?);
        let pc = push(&ElementwisePc {
            n,
            bn: bn.min(n).max(1),
            op: match op {
                BinOp::Add => 0,
                BinOp::Mul => 1,
                BinOp::Sub => 2,
            },
            act: 0,
            limit: 0.0,
            alpha: 1.0,
            value: 0.0,
        });
        self.dispatch(Kernel::Elementwise, &[d, x, y], &pc, groups(n, WG))
    }

    fn scale(&mut self, buf: BufferId, factor: f32, n: u32) -> Result<()> {
        let b = self.handle(buf)?;
        let null = self.null_buf.buffer;
        let pc = push(&ElementwisePc { n, bn: n.max(1), op: 5, act: 0, limit: 0.0, alpha: 1.0, value: factor });
        self.dispatch(Kernel::Elementwise, &[b, null, null], &pc, groups(n, WG))
    }

    fn softcap(&mut self, buf: BufferId, cap: f32, n: u32) -> Result<()> {
        if cap <= 0.0 {
            return Ok(());
        }
        let b = self.handle(buf)?;
        let null = self.null_buf.buffer;
        let pc = push(&ElementwisePc { n, bn: n.max(1), op: 6, act: 0, limit: 0.0, alpha: 1.0, value: cap });
        self.dispatch(Kernel::Elementwise, &[b, null, null], &pc, groups(n, WG))
    }

    fn copy(&mut self, dst: BufferId, dst_off: u64, src: BufferId, src_off: u64, bytes: u64) -> Result<()> {
        let (d, s) = (self.handle(dst)?, self.handle(src)?);
        self.begin()?;
        unsafe {
            self.ctx.barrier(self.cmd);
            self.ctx.device.cmd_copy_buffer(
                self.cmd,
                s,
                d,
                &[vk::BufferCopy { src_offset: src_off, dst_offset: dst_off, size: bytes }],
            );
        }
        Ok(())
    }

    fn fill(&mut self, buf: BufferId, value: f32, n: u32) -> Result<()> {
        let b = self.handle(buf)?;
        let null = self.null_buf.buffer;
        let pc = push(&ElementwisePc { n, bn: n.max(1), op: 7, act: 0, limit: 0.0, alpha: 1.0, value });
        self.dispatch(Kernel::Elementwise, &[b, null, null], &pc, groups(n, WG))
    }

    fn submit(&mut self) -> Result<()> {
        if !self.recording {
            return Ok(());
        }
        self.ctx.end_and_submit(self.cmd, self.fence)?;
        self.recording = false;
        self.pending = true;
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.submit()?;
        if !self.pending {
            return Ok(());
        }
        self.ctx.wait(self.fence)?;
        self.pending = false;
        // Descriptor sets are only valid until the work using them completes, so the pool
        // is recycled here rather than per dispatch.
        unsafe {
            self.ctx
                .device
                .reset_descriptor_pool(self.desc_pool, vk::DescriptorPoolResetFlags::empty())
                .map_err(|e| backend_err("vulkan", format!("resetting the descriptor pool: {e}")))?;
        }
        self.sets_used = 0;
        Ok(())
    }
}

impl VulkanBackend {
    fn moe_impl(&mut self, dst: BufferId, src: BufferId, w: &MoeWeights, cfg: MoeCfg) -> Result<()> {
        let n_tokens = cfg.n_tokens;
        let n_used = cfg.n_expert_used;
        let slots = n_tokens * n_used;

        let logits = self.moe_buf(0, (cfg.n_expert * n_tokens * 4) as u64)?;
        let sel_idx = self.moe_buf(1, (slots * 4) as u64)?;
        let sel_w = self.moe_buf(2, (slots * 4) as u64)?;
        let slot_act = self.moe_buf(3, (cfg.n_embd * slots * 4) as u64)?;
        let gate_buf = self.moe_buf(4, (cfg.n_ff * slots * 4) as u64)?;
        let up_buf = self.moe_buf(5, (cfg.n_ff * slots * 4) as u64)?;
        let out_buf = self.moe_buf(6, (cfg.n_embd * slots * 4) as u64)?;

        // Router projection.
        {
            let (gbuf, gdtype, growbytes) = {
                let gw = self.weight(w.gate_inp)?;
                (gw.buf.buffer, gw.dtype, gw.row_bytes())
            };
            let s = self.handle(src)?;
            let null = self.null_buf.buffer;
            let pc = push(&MatmulPc {
                n_in: cfg.n_embd,
                n_out: cfg.n_expert,
                n_tokens,
                row_bytes: growbytes as u32,
                has_bias: 0,
            });
            self.dispatch(Kernel::MatMul(gdtype), &[gbuf, s, logits, null], &pc, (cfg.n_expert, 1, 1))?;
        }

        // Top-k selection.
        {
            let pb = self.opt_handle(w.exp_probs_b)?;
            let pc = push(&MoeRoutePc {
                n_expert: cfg.n_expert,
                n_used,
                n_tokens,
                gate_func: if cfg.gate_func == gguf_core::GateFunc::Softmax { 0 } else { 1 },
                norm_topk: cfg.norm_topk as u32,
                has_bias: w.exp_probs_b.is_some() as u32,
                scale: cfg.scale,
            });
            self.dispatch(Kernel::MoeRoute, &[logits, pb, sel_idx, sel_w], &pc, groups(n_tokens, 64))?;
        }

        // Replicate each token's activation into its expert slots.
        {
            let s = self.handle(src)?;
            let null = self.null_buf.buffer;
            let n = cfg.n_embd * slots;
            let pc = push(&MoeMiscPc {
                n,
                dim: cfg.n_embd,
                n_used,
                n_tokens,
                op: 0,
                act: 0,
                has_gate: 0,
                alpha: 1.0,
            });
            self.dispatch(Kernel::MoeMisc, &[slot_act, s, null], &pc, groups(n, WG))?;
        }

        // Expert projections.
        if let Some(gw) = w.gate_exps {
            self.moe_matmul(gw, slot_act, gate_buf, cfg.n_ff, cfg.n_embd, sel_idx, n_used, n_tokens)?;
        }
        self.moe_matmul(w.up_exps, slot_act, up_buf, cfg.n_ff, cfg.n_embd, sel_idx, n_used, n_tokens)?;

        {
            let n = cfg.n_ff * slots;
            let null = self.null_buf.buffer;
            let pc = push(&MoeMiscPc {
                n,
                dim: cfg.n_ff,
                n_used,
                n_tokens,
                op: 1,
                act: act_code(cfg.act),
                has_gate: w.gate_exps.is_some() as u32,
                alpha: 1.0,
            });
            self.dispatch(Kernel::MoeMisc, &[gate_buf, up_buf, null], &pc, groups(n, WG))?;
        }

        self.moe_matmul(w.down_exps, gate_buf, out_buf, cfg.n_embd, cfg.n_ff, sel_idx, n_used, n_tokens)?;

        // Weighted sum back into one row per token.
        let d = self.handle(dst)?;
        let n = cfg.n_embd * n_tokens;
        let pc = push(&MoeMiscPc {
            n,
            dim: cfg.n_embd,
            n_used,
            n_tokens,
            op: 2,
            act: 0,
            has_gate: 0,
            alpha: 1.0,
        });
        self.dispatch(Kernel::MoeMisc, &[d, out_buf, sel_w], &pc, groups(n, WG))
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_matmul(
        &mut self,
        weight: WeightId,
        act: vk::Buffer,
        dst: vk::Buffer,
        n_out: u32,
        n_in: u32,
        sel_idx: vk::Buffer,
        n_used: u32,
        n_tokens: u32,
    ) -> Result<()> {
        let (wbuf, dtype, row_bytes, rows) = {
            let w = self.weight(weight)?;
            (w.buf.buffer, w.dtype, w.row_bytes(), w.rows)
        };
        debug_assert!(rows % n_out as usize == 0, "expert tensor rows are not a multiple of n_out");
        let pc = push(&MoeMmPc {
            n_in,
            n_out,
            n_used,
            n_tokens,
            row_bytes: row_bytes as u32,
            expert_stride: n_out * row_bytes as u32,
        });
        self.dispatch(
            Kernel::MoeMm(dtype),
            &[wbuf, act, dst, sel_idx],
            &pc,
            (n_out, n_used * n_tokens, 1),
        )
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        let _ = self.sync();
        for b in self.buffers.iter().flatten() {
            self.ctx.free_buf(b);
        }
        for w in self.weights.iter().flatten() {
            self.ctx.free_buf(&w.buf);
        }
        for b in &self.moe_scratch {
            self.ctx.free_buf(b);
        }
        self.ctx.free_buf(&self.null_buf);
        self.ctx.free_buf(&self.tokens_buf);
        self.ctx.free_buf(&self.positions_buf);
        self.pipes.destroy(&self.ctx);
        self.ctx.destroy_recording_state(self.cmd, self.fence, self.desc_pool);
    }
}

// SAFETY: a backend is owned by one thread; the engine pins a model to one worker.
unsafe impl Send for VulkanBackend {}

/// YaRN's correction range in rotary-pair units, matching the other backends exactly so the
/// three agree on positions.
fn yarn_corr_dims(cfg: &RopeCfg) -> (f32, f32) {
    if cfg.ext_factor == 0.0 || cfg.orig_ctx == 0 {
        return (0.0, 0.0);
    }
    let dim = |rot: f32| -> f32 {
        cfg.n_rot as f32 * (cfg.orig_ctx as f32 / (rot * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * cfg.freq_base.ln())
    };
    (
        dim(cfg.beta_fast).floor().max(0.0),
        dim(cfg.beta_slow).ceil().min(cfg.n_rot as f32 - 1.0),
    )
}

const _: () = {
    let _ = std::mem::size_of::<HashMap<u32, u32>>();
};

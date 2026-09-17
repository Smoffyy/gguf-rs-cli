//! CPU backend.
//!
//! This is the backend that always exists. It is also the correctness reference the GPU
//! backends are compared against, so it favours being obviously right over being clever:
//! the parallelism is coarse (rayon over output rows), and every op is written once in
//! plain Rust rather than once per instruction set.
//!
//! Speed still comes from the same place it does on the GPU: activations are quantized to
//! Q8_1 and matmuls run as integer dot products against the weight's native block layout,
//! so a Q4_K weight is never expanded to f32 anywhere.

mod attention;
mod ops;

use std::any::Any;
use std::sync::Arc;

use gguf_core::{
    backend_err, AttnCfg, Backend, BinOp, BufferId, Caps, DType, DeviceInfo,
    DeviceKind, Error, GluCfg, KvView, MatMulCfg, MemKind, MoeCfg, MoeWeights, NormCfg, QuantView,
    Result, RopeCfg, WeightId,
};
use gguf_quant::{check_dot_supported, q8_1_bytes, quantize_q8_1_batch};
use rayon::prelude::*;

/// A weight left in place inside the memory-mapped file.
///
/// The pointer is only valid while the mapping lives, which is what [`CpuBackend::retain`]
/// guarantees: the runtime hands over an `Arc` of the mapping before uploading anything,
/// and the backend holds it until it is dropped.
struct Weight {
    ptr: *const u8,
    len: usize,
    dtype: DType,
    rows: usize,
    cols: usize,
}

// SAFETY: the pointer addresses an immutable, shared memory mapping kept alive by
// `keepalive`. Nothing in this backend ever writes through it.
unsafe impl Send for Weight {}
unsafe impl Sync for Weight {}

impl Weight {
    fn bytes(&self) -> &[u8] {
        // SAFETY: see the Send impl above; the range was validated when the view was built.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn row_bytes(&self) -> usize {
        self.dtype.row_bytes(self.cols)
    }

    fn row(&self, r: usize) -> &[u8] {
        let rb = self.row_bytes();
        &self.bytes()[r * rb..(r + 1) * rb]
    }
}

/// A destination pointer handed to rayon tasks that write disjoint index sets.
///
/// Matmul parallelises over output *rows*, not tokens: a decode step has exactly one
/// token, so splitting by token leaves every core but one idle. Row `r` of token `t` lives
/// at `t * n_out + r`, so a task that owns a row writes a strided, disjoint set of slots.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);

// SAFETY: each task writes only the slots for the rows it owns, and rows are partitioned.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    /// SAFETY: `i` must be within the destination and owned by the calling task.
    unsafe fn set(self, i: usize, v: f32) {
        *self.0.add(i) = v;
    }
}

pub struct CpuBackend {
    info: DeviceInfo,
    buffers: Vec<Option<Box<[f32]>>>,
    free_buffers: Vec<u32>,
    weights: Vec<Option<Weight>>,
    free_weights: Vec<u32>,
    /// Staging area for quantized activations, reused across matmuls.
    quant_scratch: Vec<u8>,
    keepalive: Vec<Arc<dyn Any + Send + Sync>>,
}

impl CpuBackend {
    pub fn new() -> Result<Self> {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);
        Ok(Self {
            info: DeviceInfo {
                kind: DeviceKind::Cpu,
                index: 0,
                name: cpu_name(),
                total_memory: total_ram(),
                free_memory: total_ram(),
                caps: Caps {
                    int8_dot: true,
                    f16_math: false,
                    matrix_cores: false,
                    max_threads_per_block: threads,
                    shared_mem_per_block: 0,
                    warp_size: 1,
                    multiprocessors: threads,
                },
            },
            buffers: Vec::new(),
            free_buffers: Vec::new(),
            weights: Vec::new(),
            free_weights: Vec::new(),
            quant_scratch: Vec::new(),
            keepalive: Vec::new(),
        })
    }

    /// Set the global rayon pool size. Zero means one thread per logical core.
    pub fn set_threads(n: usize) {
        if n > 0 {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
        }
    }

    fn buf(&self, id: BufferId) -> Result<&[f32]> {
        self.buffers
            .get(id.0 as usize)
            .and_then(|b| b.as_deref())
            .ok_or_else(|| backend_err("cpu", format!("buffer {} is not allocated", id.0)))
    }

    /// Mutable view of a buffer.
    ///
    /// Callers may hold this alongside immutable views of *other* buffers; elementwise ops
    /// are also allowed to alias `dst` with an input because they touch matching indices.
    /// `matmul` and friends must not alias, and assert as much.
    fn buf_mut(&mut self, id: BufferId) -> Result<&mut [f32]> {
        let slot = self
            .buffers
            .get_mut(id.0 as usize)
            .and_then(|b| b.as_deref_mut())
            .ok_or_else(|| backend_err("cpu", format!("buffer {} is not allocated", id.0)))?;
        Ok(slot)
    }

    /// Raw parts for an op that needs a mutable destination and immutable sources at once.
    ///
    /// SAFETY contract for callers: `dst` must be distinct from every source unless the op
    /// is elementwise at matching indices.
    fn split(&mut self, dst: BufferId, srcs: &[BufferId]) -> Result<(*mut f32, usize, Vec<(*const f32, usize)>)> {
        let (dp, dl) = {
            let d = self.buf(dst)?;
            (d.as_ptr() as *mut f32, d.len())
        };
        let mut out = Vec::with_capacity(srcs.len());
        for s in srcs {
            let b = self.buf(*s)?;
            out.push((b.as_ptr(), b.len()));
        }
        Ok((dp, dl, out))
    }

    fn weight(&self, id: WeightId) -> Result<&Weight> {
        self.weights
            .get(id.0 as usize)
            .and_then(|w| w.as_ref())
            .ok_or_else(|| backend_err("cpu", format!("weight {} is not uploaded", id.0)))
    }

    /// Quantize `n_tokens` rows of `dim` activations into the reusable scratch area.
    fn stage_activation(&mut self, src: BufferId, dim: usize, n_tokens: usize) -> Result<()> {
        let row = q8_1_bytes(dim);
        let need = row * n_tokens;
        if self.quant_scratch.len() < need {
            self.quant_scratch.resize(need, 0);
        }
        let src = self.buf(src)?;
        if src.len() < dim * n_tokens {
            return Err(Error::Shape(format!(
                "activation holds {} values, need {}",
                src.len(),
                dim * n_tokens
            )));
        }
        // SAFETY: disjoint — `quant_scratch` is never a buffer.
        let scratch = unsafe {
            std::slice::from_raw_parts_mut(self.quant_scratch.as_ptr() as *mut u8, need)
        };
        quantize_q8_1_batch(src, dim, n_tokens, scratch);
        Ok(())
    }
}

impl Backend for CpuBackend {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn retain(&mut self, owner: Arc<dyn Any + Send + Sync>) {
        self.keepalive.push(owner);
    }

    fn alloc(&mut self, bytes: u64, _kind: MemKind) -> Result<BufferId> {
        let n = (bytes as usize).div_ceil(4);
        let data = vec![0f32; n].into_boxed_slice();
        if let Some(idx) = self.free_buffers.pop() {
            self.buffers[idx as usize] = Some(data);
            Ok(BufferId(idx))
        } else {
            self.buffers.push(Some(data));
            Ok(BufferId(self.buffers.len() as u32 - 1))
        }
    }

    fn free(&mut self, id: BufferId) {
        if let Some(slot) = self.buffers.get_mut(id.0 as usize) {
            if slot.take().is_some() {
                self.free_buffers.push(id.0);
            }
        }
    }

    fn write(&mut self, dst: BufferId, offset_bytes: u64, src: &[u8]) -> Result<()> {
        let d = self.buf_mut(dst)?;
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(d);
        let off = offset_bytes as usize;
        if off + src.len() > bytes.len() {
            return Err(Error::Shape(format!(
                "write of {} bytes at {off} exceeds buffer of {}",
                src.len(),
                bytes.len()
            )));
        }
        bytes[off..off + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn read(&mut self, src: BufferId, offset_bytes: u64, dst: &mut [u8]) -> Result<()> {
        let s = self.buf(src)?;
        let bytes: &[u8] = bytemuck::cast_slice(s);
        let off = offset_bytes as usize;
        if off + dst.len() > bytes.len() {
            return Err(Error::Shape(format!(
                "read of {} bytes at {off} exceeds buffer of {}",
                dst.len(),
                bytes.len()
            )));
        }
        dst.copy_from_slice(&bytes[off..off + dst.len()]);
        Ok(())
    }

    fn upload_weight(&mut self, view: &QuantView<'_>) -> Result<WeightId> {
        check_dot_supported(view.dtype)?;
        let w = Weight {
            ptr: view.data.as_ptr(),
            len: view.data.len(),
            dtype: view.dtype,
            rows: view.rows,
            cols: view.cols,
        };
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
            if slot.take().is_some() {
                self.free_weights.push(id.0);
            }
        }
    }

    fn get_rows(&mut self, dst: BufferId, weight: WeightId, tokens: &[u32], scale: f32) -> Result<()> {
        let w = self.weight(weight)?;
        let cols = w.cols;
        let dtype = w.dtype;
        // The rows borrow the mapping, not `self`, so they outlive the mutable borrow of
        // the destination buffer taken next.
        let rows: Vec<&'static [u8]> = tokens
            .iter()
            .map(|&t| {
                let t = (t as usize).min(w.rows.saturating_sub(1));
                let r = w.row(t);
                // SAFETY: `r` points into the retained memory map, which outlives `self`.
                unsafe { std::slice::from_raw_parts(r.as_ptr(), r.len()) }
            })
            .collect();
        let d = self.buf_mut(dst)?;
        d[..cols * tokens.len()]
            .par_chunks_mut(cols)
            .zip(rows.into_par_iter())
            .try_for_each(|(out, src)| -> Result<()> {
                gguf_quant::dequantize(dtype, src, out, cols)?;
                if scale != 1.0 {
                    for v in out.iter_mut() {
                        *v *= scale;
                    }
                }
                Ok(())
            })
    }

    fn norm(&mut self, dst: BufferId, src: BufferId, weight: Option<BufferId>, cfg: NormCfg) -> Result<()> {
        let w = match weight {
            Some(id) => Some(self.buf(id)?.to_vec()),
            None => None,
        };
        let bias = match cfg.bias {
            Some(id) => Some(self.buf(id)?.to_vec()),
            None => None,
        };
        let (dp, _, srcs) = self.split(dst, &[src])?;
        let dim = cfg.dim as usize;
        let n = cfg.n_tokens as usize;
        // SAFETY: dst and src are allowed to alias here; the op reads and writes the same
        // token slice before moving on.
        let d = unsafe { std::slice::from_raw_parts_mut(dp, dim * n) };
        let s = unsafe { std::slice::from_raw_parts(srcs[0].0, dim * n) };
        ops::norm(d, s, w.as_deref(), bias.as_deref(), cfg);
        Ok(())
    }

    fn matmul(&mut self, dst: BufferId, weight: WeightId, src: BufferId, cfg: MatMulCfg) -> Result<()> {
        let (dtype, rows, cols) = {
            let w = self.weight(weight)?;
            (w.dtype, w.rows, w.cols)
        };
        if cols != cfg.n_in as usize {
            return Err(Error::Shape(format!(
                "matmul expects {} input columns, weight has {cols}",
                cfg.n_in
            )));
        }
        if rows < cfg.n_out as usize {
            return Err(Error::Shape(format!(
                "matmul wants {} output rows, weight has {rows}",
                cfg.n_out
            )));
        }
        if gguf_quant::is_float(dtype) {
            return self.matmul_float_weight(dst, weight, src, cfg);
        }

        let n_tokens = cfg.n_tokens as usize;
        let n_out = cfg.n_out as usize;
        self.stage_activation(src, cols, n_tokens)?;

        let bias = match cfg.bias {
            Some(id) => Some(self.buf(id)?.to_vec()),
            None => None,
        };
        let w = self.weight(weight)?;
        let wbytes = w.bytes();
        let row_bytes = w.row_bytes();
        let arow = q8_1_bytes(cols);
        // SAFETY: the scratch is disjoint from every buffer, and `dst` is asserted below
        // to be distinct from `src` by the caller contract for matmul.
        let scratch = unsafe {
            std::slice::from_raw_parts(self.quant_scratch.as_ptr(), arow * n_tokens)
        };
        let out = OutPtr(self.buf(dst)?.as_ptr() as *mut f32);
        let bias = bias.as_deref();

        // Loading the weight row once and reusing it across every token in the batch is
        // what turns prefill from a stack of independent GEMVs into a real GEMM: the
        // weights are the bandwidth cost, and this pays it once per row instead of once
        // per row per token.
        (0..n_out).into_par_iter().with_min_len(4).for_each(|r| {
            let wr = &wbytes[r * row_bytes..(r + 1) * row_bytes];
            let b = bias.map(|b| b[r]).unwrap_or(0.0);
            for t in 0..n_tokens {
                let a = &scratch[t * arow..(t + 1) * arow];
                let v = gguf_quant::vec_dot_q8_1(dtype, cols, wr, a) + b;
                // SAFETY: this task owns row `r`, so slot `t * n_out + r` is its own.
                unsafe { out.set(t * n_out + r, v) };
            }
        });
        Ok(())
    }

    fn matmul_f32(&mut self, dst: BufferId, weight: BufferId, src: BufferId, cfg: MatMulCfg) -> Result<()> {
        let n_in = cfg.n_in as usize;
        let n_out = cfg.n_out as usize;
        let n_tokens = cfg.n_tokens as usize;
        let bias = match cfg.bias {
            Some(id) => Some(self.buf(id)?.to_vec()),
            None => None,
        };
        let (dp, _, srcs) = self.split(dst, &[weight, src])?;
        let w = unsafe { std::slice::from_raw_parts(srcs[0].0, n_out * n_in) };
        let s = unsafe { std::slice::from_raw_parts(srcs[1].0, n_in * n_tokens) };
        let out = OutPtr(dp);
        let bias = bias.as_deref();
        (0..n_out).into_par_iter().with_min_len(4).for_each(|r| {
            let wr = &w[r * n_in..(r + 1) * n_in];
            let b = bias.map(|b| b[r]).unwrap_or(0.0);
            for t in 0..n_tokens {
                let a = &s[t * n_in..(t + 1) * n_in];
                let v: f32 = wr.iter().zip(a).map(|(x, y)| x * y).sum::<f32>() + b;
                // SAFETY: this task owns row `r`.
                unsafe { out.set(t * n_out + r, v) };
            }
        });
        Ok(())
    }

    fn rope(&mut self, q: BufferId, k: Option<BufferId>, positions: &[i32], cfg: RopeCfg) -> Result<()> {
        let factors = match cfg.freq_factors {
            Some(id) => Some(self.buf(id)?.to_vec()),
            None => None,
        };
        {
            let qb = self.buf_mut(q)?;
            ops::rope(qb, positions, cfg.n_heads, factors.as_deref(), &cfg);
        }
        if let Some(k) = k {
            let kb = self.buf_mut(k)?;
            ops::rope(kb, positions, cfg.n_kv_heads, factors.as_deref(), &cfg);
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
        let f16_store = kv.is_f16();
        for (src, dst) in [(k, kv.k), (v, kv.v)] {
            let (dp, dlen, srcs) = self.split(dst, &[src])?;
            let s = unsafe { std::slice::from_raw_parts(srcs[0].0, (dim * n_tokens) as usize) };
            let stride = kv.stride as usize;
            let dim = dim as usize;
            for t in 0..n_tokens as usize {
                let slot = (kv.base + start_pos) as usize + t;
                let off = slot * stride;
                let row = &s[t * dim..(t + 1) * dim];
                if f16_store {
                    // Two cached values share one f32 slot of the backing allocation.
                    let cap = dlen * 2;
                    if off + dim > cap {
                        return Err(Error::Shape(format!(
                            "kv slot {slot} is past the end of the cache"
                        )));
                    }
                    let d = unsafe { std::slice::from_raw_parts_mut(dp as *mut u16, cap) };
                    for (i, val) in row.iter().enumerate() {
                        d[off + i] = half::f16::from_f32(*val).to_bits();
                    }
                } else {
                    if off + dim > dlen {
                        return Err(Error::Shape(format!(
                            "kv slot {slot} is past the end of the cache"
                        )));
                    }
                    let d = unsafe { std::slice::from_raw_parts_mut(dp, dlen) };
                    d[off..off + dim].copy_from_slice(row);
                }
            }
        }
        Ok(())
    }

    fn attention(&mut self, dst: BufferId, q: BufferId, kv: &KvView, cfg: AttnCfg) -> Result<()> {
        let sinks = match cfg.sinks {
            Some(id) => Some(self.buf(id)?.to_vec()),
            None => None,
        };
        let (dp, dlen, srcs) = self.split(dst, &[q, kv.k, kv.v])?;
        let d = unsafe { std::slice::from_raw_parts_mut(dp, dlen) };
        let qb = unsafe { std::slice::from_raw_parts(srcs[0].0, srcs[0].1) };
        let view = |p: (*const f32, usize)| -> attention::KvSlice<'_> {
            if kv.is_f16() {
                attention::KvSlice::F16(unsafe {
                    std::slice::from_raw_parts(p.0 as *const u16, p.1 * 2)
                })
            } else {
                attention::KvSlice::F32(unsafe { std::slice::from_raw_parts(p.0, p.1) })
            }
        };
        attention::flash_attention(d, qb, view(srcs[1]), view(srcs[2]), kv, &cfg, sinks.as_deref());
        Ok(())
    }

    fn glu(&mut self, dst: BufferId, gate: BufferId, up: BufferId, cfg: GluCfg) -> Result<()> {
        let (dp, _, srcs) = self.split(dst, &[gate, up])?;
        let n = (cfg.dim * cfg.n_tokens) as usize;
        let d = unsafe { std::slice::from_raw_parts_mut(dp, n) };
        let g = unsafe { std::slice::from_raw_parts(srcs[0].0, n) };
        let u = unsafe { std::slice::from_raw_parts(srcs[1].0, n) };
        ops::glu(d, g, u, cfg);
        Ok(())
    }

    fn activate(&mut self, buf: BufferId, cfg: GluCfg) -> Result<()> {
        let n = (cfg.dim * cfg.n_tokens) as usize;
        let b = self.buf_mut(buf)?;
        let n = n.min(b.len());
        ops::apply_act_inplace(&mut b[..n], cfg.act, cfg.alpha);
        Ok(())
    }

    fn moe(&mut self, dst: BufferId, src: BufferId, w: &MoeWeights, cfg: MoeCfg) -> Result<()> {
        self.moe_impl(dst, src, w, cfg)
    }

    fn binary(&mut self, dst: BufferId, a: BufferId, b: BufferId, op: BinOp, n: u32) -> Result<()> {
        let (dp, _, srcs) = self.split(dst, &[a, b])?;
        let n = n as usize;
        let d = unsafe { std::slice::from_raw_parts_mut(dp, n) };
        let x = unsafe { std::slice::from_raw_parts(srcs[0].0, n) };
        let y = unsafe { std::slice::from_raw_parts(srcs[1].0, n.min(srcs[1].1)) };
        // A shorter `b` is broadcast, which is how per-channel bias vectors are applied to
        // a multi-token activation without materialising a copy per token.
        let bn = y.len();
        match op {
            BinOp::Add => {
                for i in 0..n {
                    d[i] = x[i] + y[i % bn];
                }
            }
            BinOp::Mul => {
                for i in 0..n {
                    d[i] = x[i] * y[i % bn];
                }
            }
            BinOp::Sub => {
                for i in 0..n {
                    d[i] = x[i] - y[i % bn];
                }
            }
        }
        Ok(())
    }

    fn scale(&mut self, buf: BufferId, factor: f32, n: u32) -> Result<()> {
        let b = self.buf_mut(buf)?;
        let n = (n as usize).min(b.len());
        for v in b[..n].iter_mut() {
            *v *= factor;
        }
        Ok(())
    }

    fn softcap(&mut self, buf: BufferId, cap: f32, n: u32) -> Result<()> {
        if cap <= 0.0 {
            return Ok(());
        }
        let b = self.buf_mut(buf)?;
        let n = (n as usize).min(b.len());
        for v in b[..n].iter_mut() {
            *v = cap * (*v / cap).tanh();
        }
        Ok(())
    }

    fn copy(&mut self, dst: BufferId, dst_off: u64, src: BufferId, src_off: u64, bytes: u64) -> Result<()> {
        let n = bytes as usize / 4;
        let so = src_off as usize / 4;
        let d_off = dst_off as usize / 4;
        let (dp, dlen, srcs) = self.split(dst, &[src])?;
        if d_off + n > dlen || so + n > srcs[0].1 {
            return Err(Error::Shape("copy range is out of bounds".into()));
        }
        let d = unsafe { std::slice::from_raw_parts_mut(dp.add(d_off), n) };
        let s = unsafe { std::slice::from_raw_parts(srcs[0].0.add(so), n) };
        d.copy_from_slice(s);
        Ok(())
    }

    fn fill(&mut self, buf: BufferId, value: f32, n: u32) -> Result<()> {
        let b = self.buf_mut(buf)?;
        let n = (n as usize).min(b.len());
        b[..n].fill(value);
        Ok(())
    }

    fn submit(&mut self) -> Result<()> {
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }
}

impl CpuBackend {
    fn matmul_float_weight(
        &mut self,
        dst: BufferId,
        weight: WeightId,
        src: BufferId,
        cfg: MatMulCfg,
    ) -> Result<()> {
        let w = self.weight(weight)?;
        let (dtype, cols, row_bytes) = (w.dtype, w.cols, w.row_bytes());
        let wbytes = w.bytes();
        let bias = match cfg.bias {
            Some(id) => Some(self.buf(id)?.to_vec()),
            None => None,
        };
        let n_out = cfg.n_out as usize;
        let n_tokens = cfg.n_tokens as usize;
        let s = self.buf(src)?;
        let sptr = s.as_ptr();
        let out = OutPtr(self.buf(dst)?.as_ptr() as *mut f32);
        let act = unsafe { std::slice::from_raw_parts(sptr, cols * n_tokens) };
        let bias = bias.as_deref();
        (0..n_out).into_par_iter().with_min_len(4).for_each(|r| {
            let wr = &wbytes[r * row_bytes..(r + 1) * row_bytes];
            let b = bias.map(|b| b[r]).unwrap_or(0.0);
            for t in 0..n_tokens {
                let a = &act[t * cols..(t + 1) * cols];
                let v = gguf_quant::vec_dot_f32(dtype, cols, wr, a) + b;
                // SAFETY: this task owns row `r`.
                unsafe { out.set(t * n_out + r, v) };
            }
        });
        Ok(())
    }

    fn moe_impl(&mut self, dst: BufferId, src: BufferId, w: &MoeWeights, cfg: MoeCfg) -> Result<()> {
        let n_embd = cfg.n_embd as usize;
        let n_ff = cfg.n_ff as usize;
        let n_tokens = cfg.n_tokens as usize;
        let n_expert = cfg.n_expert as usize;
        let n_used = cfg.n_expert_used as usize;

        // Routing runs in f32: it is tiny next to the expert matmuls, and top-k selection
        // is sensitive enough to rounding that quantizing it would change which experts
        // fire for near-ties.
        let router_logits = {
            let gate = self.weight(w.gate_inp)?;
            let (gd, gcols, grb) = (gate.dtype, gate.cols, gate.row_bytes());
            let gbytes = gate.bytes();
            let s = self.buf(src)?;
            let mut out = vec![0f32; n_expert * n_tokens];
            let staged = if gguf_quant::is_float(gd) {
                None
            } else {
                let mut buf = vec![0u8; q8_1_bytes(gcols) * n_tokens];
                quantize_q8_1_batch(&s[..gcols * n_tokens], gcols, n_tokens, &mut buf);
                Some(buf)
            };
            for t in 0..n_tokens {
                for e in 0..n_expert {
                    let wr = &gbytes[e * grb..(e + 1) * grb];
                    out[t * n_expert + e] = match &staged {
                        Some(b) => {
                            let arow = q8_1_bytes(gcols);
                            gguf_quant::vec_dot_q8_1(gd, gcols, wr, &b[t * arow..(t + 1) * arow])
                        }
                        None => gguf_quant::vec_dot_f32(gd, gcols, wr, &s[t * gcols..(t + 1) * gcols]),
                    };
                }
            }
            out
        };

        let probs_bias = match w.exp_probs_b {
            Some(id) => Some(self.buf(id)?.to_vec()),
            None => None,
        };

        let src_vals = self.buf(src)?.to_vec();
        let mut out = vec![0f32; n_embd * n_tokens];

        for t in 0..n_tokens {
            let logits = &router_logits[t * n_expert..(t + 1) * n_expert];
            let mut probs = logits.to_vec();
            match cfg.gate_func {
                gguf_core::GateFunc::Softmax => ops::softmax(&mut probs),
                gguf_core::GateFunc::Sigmoid => {
                    for p in probs.iter_mut() {
                        *p = 1.0 / (1.0 + (-*p).exp());
                    }
                }
            }
            // The bias steers selection only; the weight applied to the expert output is
            // the unbiased probability, matching DeepSeek-V3's formulation.
            let mut ranked: Vec<usize> = (0..n_expert).collect();
            if let Some(b) = &probs_bias {
                ranked.sort_unstable_by(|&x, &y| {
                    (probs[y] + b[y]).total_cmp(&(probs[x] + b[x]))
                });
            } else {
                ranked.sort_unstable_by(|&x, &y| probs[y].total_cmp(&probs[x]));
            }
            ranked.truncate(n_used);

            let mut weights: Vec<f32> = ranked.iter().map(|&e| probs[e]).collect();
            if cfg.norm_topk {
                let sum: f32 = weights.iter().sum::<f32>().max(1e-20);
                for v in weights.iter_mut() {
                    *v /= sum;
                }
            }
            for v in weights.iter_mut() {
                *v *= cfg.scale;
            }

            let x = &src_vals[t * n_embd..(t + 1) * n_embd];
            let mut xq = vec![0u8; q8_1_bytes(n_embd)];
            gguf_quant::quantize_q8_1(x, &mut xq);

            let acc = &mut out[t * n_embd..(t + 1) * n_embd];
            for (slot, &e) in ranked.iter().enumerate() {
                let gate = match w.gate_exps {
                    Some(id) => Some(self.expert_matvec(id, e, n_ff, n_embd, &xq)?),
                    None => None,
                };
                let up = self.expert_matvec(w.up_exps, e, n_ff, n_embd, &xq)?;
                let mut ff = match gate {
                    Some(mut g) => {
                        ops::apply_glu_inplace(&mut g, &up, cfg.act, 0.0, 1.0);
                        g
                    }
                    None => {
                        let mut u = up;
                        ops::apply_act_inplace(&mut u, cfg.act, 1.0);
                        u
                    }
                };
                let mut ffq = vec![0u8; q8_1_bytes(n_ff)];
                gguf_quant::quantize_q8_1(&ff, &mut ffq);
                ff = self.expert_matvec(w.down_exps, e, n_embd, n_ff, &ffq)?;
                let wgt = weights[slot];
                for (o, v) in acc.iter_mut().zip(&ff) {
                    *o += wgt * v;
                }
            }
        }

        let d = self.buf_mut(dst)?;
        d[..out.len()].copy_from_slice(&out);
        Ok(())
    }

    /// One expert's slice of a stacked `[n_expert, rows, cols]` tensor times a staged
    /// activation.
    fn expert_matvec(
        &self,
        id: WeightId,
        expert: usize,
        rows: usize,
        cols: usize,
        aq: &[u8],
    ) -> Result<Vec<f32>> {
        let w = self.weight(id)?;
        let rb = w.row_bytes();
        let base = expert * rows * rb;
        let bytes = w.bytes();
        if base + rows * rb > bytes.len() {
            return Err(Error::Shape(format!(
                "expert {expert} is past the end of a {} row tensor",
                w.rows
            )));
        }
        let dtype = w.dtype;
        Ok((0..rows)
            .into_par_iter()
            .map(|r| {
                let wr = &bytes[base + r * rb..base + (r + 1) * rb];
                gguf_quant::vec_dot_q8_1(dtype, cols, wr, aq)
            })
            .collect())
    }
}

fn cpu_name() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let mut features = Vec::new();
        if std::arch::is_x86_feature_detected!("avx512f") {
            features.push("AVX-512");
        } else if std::arch::is_x86_feature_detected!("avx2") {
            features.push("AVX2");
        }
        if std::arch::is_x86_feature_detected!("avx512vnni") || std::arch::is_x86_feature_detected!("avxvnni") {
            features.push("VNNI");
        }
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        if features.is_empty() {
            format!("x86-64 ({threads} threads)")
        } else {
            format!("x86-64 {} ({threads} threads)", features.join("+"))
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        format!("{} ({threads} threads)", std::env::consts::ARCH)
    }
}

fn total_ram() -> u64 {
    // Without a platform dependency this is a floor, not a measurement. It is only used
    // for reporting and for the "will this model fit" warning, both of which degrade
    // gracefully when it is wrong.
    16 << 30
}

const _: fn() = || {
    fn assert_backend<T: Backend>() {}
    assert_backend::<CpuBackend>();
};

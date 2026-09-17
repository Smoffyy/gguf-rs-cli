//! CUDA backend, built on the driver API with kernels pre-compiled to PTX.
//!
//! NVIDIA's Rust CUDA tracks (cuda-oxide and cutile) are Linux-only and need a recent
//! toolkit, so they cannot carry a cross-platform engine today. This takes the route that
//! works everywhere NVIDIA hardware does: kernels in CUDA C compiled to PTX at build time,
//! the driver library opened at run time, and no link-time CUDA dependency at all.

mod driver;

use std::collections::HashMap;
use std::ffi::c_void;

use driver::{
    CUcontext, CUdevice, CUdeviceptr, CUfunction, CUmodule, CUstream, Driver, Launch,
    ATTR_COMPUTE_CAPABILITY_MAJOR, ATTR_COMPUTE_CAPABILITY_MINOR, ATTR_MAX_SHARED_PER_BLOCK,
    ATTR_MAX_THREADS_PER_BLOCK, ATTR_MULTIPROCESSOR_COUNT, ATTR_WARP_SIZE,
};
use gguf_core::{
    backend_err, Activation, AttnCfg, Backend, BinOp, BufferId, Caps, DType, DeviceInfo,
    DeviceKind, Error, GluCfg, KvView, MatMulCfg, MemKind, MoeCfg, MoeWeights, NormCfg, NormKind,
    QuantView, Result, RopeCfg, RopeKind, WeightId,
};
use gguf_quant::Q8_1_BLOCK;

/// The kernels, compiled by `build.rs`. Empty when nvcc was unavailable.
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/ops.ptx"));

const WARP: u32 = 32;
/// Warps per matmul block: each warp owns one output row.
const ROWS_PER_BLOCK: u32 = 4;

pub fn kernels_available() -> bool {
    !PTX.trim().is_empty()
}

struct Alloc {
    ptr: CUdeviceptr,
    bytes: u64,
}

struct Weight {
    ptr: CUdeviceptr,
    dtype: DType,
    rows: usize,
    cols: usize,
    bytes: u64,
}

impl Weight {
    fn row_bytes(&self) -> usize {
        self.dtype.row_bytes(self.cols)
    }
}

/// Device-side staging that grows to the largest request and is then reused.
#[derive(Default)]
struct Scratch {
    ptr: CUdeviceptr,
    bytes: u64,
}

pub struct CudaBackend {
    drv: Driver,
    ctx: CUcontext,
    stream: CUstream,
    module: CUmodule,
    funcs: HashMap<String, CUfunction>,
    info: DeviceInfo,
    buffers: Vec<Option<Alloc>>,
    free_buffers: Vec<u32>,
    weights: Vec<Option<Weight>>,
    free_weights: Vec<u32>,
    /// Quantized activations for the current matmul.
    act: Scratch,
    /// Token indices for get_rows, and MoE routing tables.
    aux: Scratch,
    moe_a: Scratch,
    moe_b: Scratch,
    moe_c: Scratch,
    /// Per-split attention partials.
    attn: Scratch,
    total_allocated: u64,
}

/// Every device this process can see, for `gguf-rs devices`.
pub fn enumerate() -> Vec<DeviceInfo> {
    let Ok(drv) = Driver::open() else { return Vec::new() };
    let Ok(n) = drv.device_count() else { return Vec::new() };
    (0..n)
        .filter_map(|i| {
            let mut dev: CUdevice = 0;
            drv.check(unsafe { (drv.cuDeviceGet)(&mut dev, i as i32) }, "cuDeviceGet").ok()?;
            device_info(&drv, dev, i).ok()
        })
        .collect()
}

fn device_info(drv: &Driver, dev: CUdevice, index: u32) -> Result<DeviceInfo> {
    let major = drv.attribute(dev, ATTR_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = drv.attribute(dev, ATTR_COMPUTE_CAPABILITY_MINOR)?;
    let total = drv.total_memory(dev)?;
    Ok(DeviceInfo {
        kind: DeviceKind::Cuda,
        index,
        name: format!("{} (sm_{major}{minor})", drv.device_name(dev)?),
        total_memory: total,
        free_memory: total,
        caps: Caps {
            // dp4a landed with Pascal; anything this backend targets has it.
            int8_dot: major > 6 || (major == 6 && minor >= 1),
            f16_math: major >= 6,
            matrix_cores: major >= 7,
            max_threads_per_block: drv.attribute(dev, ATTR_MAX_THREADS_PER_BLOCK)? as u32,
            shared_mem_per_block: drv.attribute(dev, ATTR_MAX_SHARED_PER_BLOCK)? as u32,
            warp_size: drv.attribute(dev, ATTR_WARP_SIZE)? as u32,
            multiprocessors: drv.attribute(dev, ATTR_MULTIPROCESSOR_COUNT)? as u32,
        },
    })
}

/// Kernel-name suffix for a quantization, or `None` if this backend has no kernel for it.
fn suffix(dt: DType) -> Option<&'static str> {
    Some(match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::Q4_0 => "q4_0",
        DType::Q4_1 => "q4_1",
        DType::Q5_0 => "q5_0",
        DType::Q5_1 => "q5_1",
        DType::Q8_0 => "q8_0",
        DType::Q2K => "q2_k",
        DType::Q3K => "q3_k",
        DType::Q4K => "q4_k",
        DType::Q5K => "q5_k",
        DType::Q6K => "q6_k",
        DType::Iq4Nl => "iq4_nl",
        DType::Iq4Xs => "iq4_xs",
        DType::Mxfp4 => "mxfp4",
        DType::Tq1_0 => "tq1_0",
        DType::Tq2_0 => "tq2_0",
        _ => return None,
    })
}

fn is_float(dt: DType) -> bool {
    matches!(dt, DType::F32 | DType::F16 | DType::BF16)
}

fn act_code(a: Activation) -> i32 {
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

fn grid_1d(n: u32, block: u32) -> (u32, u32, u32) {
    ((n + block - 1) / block, 1, 1)
}

impl CudaBackend {
    pub fn new(index: u32) -> Result<Self> {
        if !kernels_available() {
            return Err(Error::NoDevice(
                "this build has no CUDA kernels: nvcc was not found when it was compiled. \
                 Install the CUDA toolkit and rebuild, or use --device vulkan"
                    .into(),
            ));
        }
        let drv = Driver::open()?;
        let count = drv.device_count()?;
        if count == 0 {
            return Err(Error::NoDevice("the NVIDIA driver reports no devices".into()));
        }
        if index >= count {
            return Err(Error::NoDevice(format!(
                "cuda:{index} was requested but only {count} device(s) are present"
            )));
        }

        let mut dev: CUdevice = 0;
        drv.check(unsafe { (drv.cuDeviceGet)(&mut dev, index as i32) }, "cuDeviceGet")?;
        let info = device_info(&drv, dev, index)?;

        let mut ctx: CUcontext = std::ptr::null_mut();
        drv.check(unsafe { (drv.cuCtxCreate_v2)(&mut ctx, 0, dev) }, "creating a context")?;
        drv.check(unsafe { (drv.cuCtxSetCurrent)(ctx) }, "cuCtxSetCurrent")?;

        let mut stream: CUstream = std::ptr::null_mut();
        drv.check(unsafe { (drv.cuStreamCreate)(&mut stream, 0) }, "creating a stream")?;

        let ptx = std::ffi::CString::new(PTX).map_err(|_| {
            backend_err("cuda", "the compiled PTX contains a NUL byte")
        })?;
        let mut module: CUmodule = std::ptr::null_mut();
        drv.check(
            unsafe { (drv.cuModuleLoadData)(&mut module, ptx.as_ptr() as *const c_void) },
            "loading the kernel module (the driver may be older than the toolkit that built it)",
        )?;

        let mut be = Self {
            drv,
            ctx,
            stream,
            module,
            funcs: HashMap::new(),
            info,
            buffers: Vec::new(),
            free_buffers: Vec::new(),
            weights: Vec::new(),
            free_weights: Vec::new(),
            act: Scratch::default(),
            aux: Scratch::default(),
            moe_a: Scratch::default(),
            moe_b: Scratch::default(),
            moe_c: Scratch::default(),
            attn: Scratch::default(),
            total_allocated: 0,
        };
        be.load_functions()?;
        Ok(be)
    }

    fn load_functions(&mut self) -> Result<()> {
        let fixed = [
            "quantize_q8_1",
            "mul_mat_dense",
            "norm_kernel",
            "rope_kernel",
            "kv_write_f32",
            "kv_write_f16",
            "flash_attn_split_f32",
            "flash_attn_split_f16",
            "flash_attn_combine",
            "glu_kernel",
            "activate_kernel",
            "binary_kernel",
            "scale_kernel",
            "softcap_kernel",
            "fill_kernel",
            "moe_route",
            "moe_glu",
            "moe_reduce",
            "moe_scatter",
        ];
        for name in fixed {
            let f = self.drv.function(self.module, name)?;
            self.funcs.insert(name.to_string(), f);
        }
        // Per-quantization kernels: whichever the PTX actually contains.
        for dt in [
            DType::F32, DType::F16, DType::BF16, DType::Q4_0, DType::Q4_1, DType::Q5_0,
            DType::Q5_1, DType::Q8_0, DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K,
            DType::Q6K, DType::Iq4Nl, DType::Iq4Xs, DType::Mxfp4, DType::Tq1_0, DType::Tq2_0,
        ] {
            let Some(sfx) = suffix(dt) else { continue };
            for prefix in ["get_rows", "mul_mat", "moe_mm"] {
                let name = format!("{prefix}_{sfx}");
                if let Ok(f) = self.drv.function(self.module, &name) {
                    self.funcs.insert(name, f);
                }
            }
        }
        Ok(())
    }

    fn func(&self, name: &str) -> Result<CUfunction> {
        self.funcs
            .get(name)
            .copied()
            .ok_or_else(|| backend_err("cuda", format!("kernel {name} is not in this build")))
    }

    fn dptr(&self, id: BufferId) -> Result<CUdeviceptr> {
        self.buffers
            .get(id.0 as usize)
            .and_then(|b| b.as_ref())
            .map(|a| a.ptr)
            .ok_or_else(|| backend_err("cuda", format!("buffer {} is not allocated", id.0)))
    }

    fn opt_dptr(&self, id: Option<BufferId>) -> Result<CUdeviceptr> {
        match id {
            Some(b) => self.dptr(b),
            None => Ok(0),
        }
    }

    fn raw_alloc(&mut self, bytes: u64) -> Result<CUdeviceptr> {
        let mut ptr: CUdeviceptr = 0;
        let bytes = bytes.max(16);
        let code = unsafe { (self.drv.cuMemAlloc_v2)(&mut ptr, bytes as usize) };
        if code != driver::CUDA_SUCCESS {
            let (mut free, mut total) = (0usize, 0usize);
            unsafe { (self.drv.cuMemGetInfo_v2)(&mut free, &mut total) };
            return Err(Error::OutOfMemory {
                device: self.info.id(),
                requested: bytes,
                available: free as u64,
            });
        }
        self.total_allocated += bytes;
        Ok(ptr)
    }

    fn grow(&mut self, which: usize, bytes: u64) -> Result<CUdeviceptr> {
        let current = match which {
            0 => &self.act,
            1 => &self.aux,
            2 => &self.moe_a,
            3 => &self.moe_b,
            4 => &self.moe_c,
            _ => &self.attn,
        };
        if current.bytes >= bytes && current.ptr != 0 {
            return Ok(current.ptr);
        }
        let old = current.ptr;
        if old != 0 {
            unsafe { (self.drv.cuMemFree_v2)(old) };
            let freed = match which {
                0 => self.act.bytes,
                1 => self.aux.bytes,
                2 => self.moe_a.bytes,
                3 => self.moe_b.bytes,
                4 => self.moe_c.bytes,
                _ => self.attn.bytes,
            };
            self.total_allocated = self.total_allocated.saturating_sub(freed);
        }
        // Round up so a slowly growing batch does not reallocate every step.
        let bytes = (bytes.next_power_of_two()).max(4096);
        let ptr = self.raw_alloc(bytes)?;
        let slot = match which {
            0 => &mut self.act,
            1 => &mut self.aux,
            2 => &mut self.moe_a,
            3 => &mut self.moe_b,
            4 => &mut self.moe_c,
            _ => &mut self.attn,
        };
        slot.ptr = ptr;
        slot.bytes = bytes;
        Ok(ptr)
    }

    fn upload_bytes(&mut self, dst: CUdeviceptr, src: &[u8]) -> Result<()> {
        self.drv.check(
            unsafe {
                (self.drv.cuMemcpyHtoDAsync_v2)(
                    dst,
                    src.as_ptr() as *const c_void,
                    src.len(),
                    self.stream,
                )
            },
            "host to device copy",
        )
    }

    /// Quantize `src` to Q8_1 on the device and return the staging pointer.
    fn stage_activation(&mut self, src: BufferId, dim: u32, n_tokens: u32) -> Result<CUdeviceptr> {
        let blocks = (dim as usize).div_ceil(32) as u32;
        let bytes = (blocks as u64) * Q8_1_BLOCK as u64 * n_tokens as u64;
        let dst = self.grow(0, bytes)?;
        let s = self.dptr(src)?;
        let f = self.func("quantize_q8_1")?;
        let total = blocks * n_tokens;
        let mut l = Launch::new()
            .ptr(s)
            .ptr(dst)
            .i32(dim as i32)
            .i32(blocks as i32)
            .i32(n_tokens as i32);
        unsafe {
            l.run(&self.drv, f, grid_1d(total, 256), (256, 1, 1), 0, self.stream, "quantize_q8_1")?;
        }
        Ok(dst)
    }

    fn launch_1d(&self, name: &str, n: u32, build: impl FnOnce(Launch) -> Launch) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let f = self.func(name)?;
        let mut l = build(Launch::new());
        unsafe { l.run(&self.drv, f, grid_1d(n, 256), (256, 1, 1), 0, self.stream, name) }
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        unsafe {
            for s in [&self.act, &self.aux, &self.moe_a, &self.moe_b, &self.moe_c, &self.attn] {
                if s.ptr != 0 {
                    (self.drv.cuMemFree_v2)(s.ptr);
                }
            }
            for b in self.buffers.iter().flatten() {
                (self.drv.cuMemFree_v2)(b.ptr);
            }
            for w in self.weights.iter().flatten() {
                (self.drv.cuMemFree_v2)(w.ptr);
            }
            (self.drv.cuModuleUnload)(self.module);
            (self.drv.cuStreamDestroy_v2)(self.stream);
            (self.drv.cuCtxDestroy_v2)(self.ctx);
        }
    }
}

// SAFETY: the context is only ever used from the thread that owns the backend; the engine
// pins a model to one worker thread.
unsafe impl Send for CudaBackend {}

impl Backend for CudaBackend {
    fn info(&self) -> &DeviceInfo {
        &self.info
    }

    fn alloc(&mut self, bytes: u64, _kind: MemKind) -> Result<BufferId> {
        let ptr = self.raw_alloc(bytes)?;
        unsafe { (self.drv.cuMemsetD8_v2)(ptr, 0, bytes.max(16) as usize) };
        let alloc = Alloc { ptr, bytes: bytes.max(16) };
        if let Some(idx) = self.free_buffers.pop() {
            self.buffers[idx as usize] = Some(alloc);
            Ok(BufferId(idx))
        } else {
            self.buffers.push(Some(alloc));
            Ok(BufferId(self.buffers.len() as u32 - 1))
        }
    }

    fn free(&mut self, id: BufferId) {
        if let Some(slot) = self.buffers.get_mut(id.0 as usize) {
            if let Some(a) = slot.take() {
                unsafe { (self.drv.cuMemFree_v2)(a.ptr) };
                self.total_allocated = self.total_allocated.saturating_sub(a.bytes);
                self.free_buffers.push(id.0);
            }
        }
    }

    fn write(&mut self, dst: BufferId, offset_bytes: u64, src: &[u8]) -> Result<()> {
        let p = self.dptr(dst)? + offset_bytes;
        self.upload_bytes(p, src)
    }

    fn read(&mut self, src: BufferId, offset_bytes: u64, dst: &mut [u8]) -> Result<()> {
        let p = self.dptr(src)? + offset_bytes;
        self.sync()?;
        self.drv.check(
            unsafe {
                (self.drv.cuMemcpyDtoH_v2)(dst.as_mut_ptr() as *mut c_void, p, dst.len())
            },
            "device to host copy",
        )
    }

    fn upload_weight(&mut self, view: &QuantView<'_>) -> Result<WeightId> {
        view.dtype.check_supported()?;
        if suffix(view.dtype).is_none() {
            return Err(Error::Unsupported(format!(
                "{} has no CUDA kernel; run this model on --device cpu",
                view.dtype.name()
            )));
        }
        let bytes = view.data.len() as u64;
        let ptr = self.raw_alloc(bytes)?;
        // Synchronous: the source is a memory map that the caller may drop right after.
        self.drv.check(
            unsafe {
                (self.drv.cuMemcpyHtoD_v2)(ptr, view.data.as_ptr() as *const c_void, view.data.len())
            },
            "uploading a weight",
        )?;
        let w = Weight {
            ptr,
            dtype: view.dtype,
            rows: view.rows,
            cols: view.cols,
            bytes,
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
            if let Some(w) = slot.take() {
                unsafe { (self.drv.cuMemFree_v2)(w.ptr) };
                self.total_allocated = self.total_allocated.saturating_sub(w.bytes);
                self.free_weights.push(id.0);
            }
        }
    }

    fn get_rows(&mut self, dst: BufferId, weight: WeightId, tokens: &[u32], scale: f32) -> Result<()> {
        let (ptr, dtype, rows, cols, row_bytes) = {
            let w = self
                .weights
                .get(weight.0 as usize)
                .and_then(|w| w.as_ref())
                .ok_or_else(|| backend_err("cuda", "weight is not uploaded"))?;
            (w.ptr, w.dtype, w.rows, w.cols, w.row_bytes())
        };
        let idx_bytes = (tokens.len() * 4) as u64;
        let idx = self.grow(1, idx_bytes)?;
        let raw: Vec<u8> = tokens.iter().flat_map(|t| (*t as i32).to_ne_bytes()).collect();
        self.upload_bytes(idx, &raw)?;

        let name = format!("get_rows_{}", suffix(dtype).unwrap());
        let f = self.func(&name)?;
        let d = self.dptr(dst)?;
        let mut l = Launch::new()
            .ptr(ptr)
            .ptr(idx)
            .ptr(d)
            .i32(rows as i32)
            .i32(cols as i32)
            .i32(row_bytes as i32)
            .f32(scale)
            .i32(tokens.len() as i32);
        unsafe {
            l.run(
                &self.drv,
                f,
                (tokens.len() as u32, 1, 1),
                (256, 1, 1),
                0,
                self.stream,
                &name,
            )
        }
    }

    fn norm(&mut self, dst: BufferId, src: BufferId, weight: Option<BufferId>, cfg: NormCfg) -> Result<()> {
        let f = self.func("norm_kernel")?;
        let (d, s) = (self.dptr(dst)?, self.dptr(src)?);
        let w = self.opt_dptr(weight)?;
        let b = self.opt_dptr(cfg.bias)?;
        // One block per row; a power-of-two width keeps the tree reduction simple.
        let threads = cfg.dim.next_power_of_two().clamp(32, 1024);
        let mut l = Launch::new()
            .ptr(s)
            .ptr(d)
            .ptr(w)
            .ptr(b)
            .i32(cfg.dim as i32)
            .i32(if cfg.kind == NormKind::Layer { 1 } else { 0 })
            .f32(cfg.eps)
            .f32(cfg.scale);
        unsafe {
            l.run(
                &self.drv,
                f,
                (cfg.n_tokens.max(1), 1, 1),
                (threads, 1, 1),
                threads * 4,
                self.stream,
                "norm",
            )
        }
    }

    fn matmul(&mut self, dst: BufferId, weight: WeightId, src: BufferId, cfg: MatMulCfg) -> Result<()> {
        let (ptr, dtype, cols, row_bytes) = {
            let w = self
                .weights
                .get(weight.0 as usize)
                .and_then(|w| w.as_ref())
                .ok_or_else(|| backend_err("cuda", "weight is not uploaded"))?;
            (w.ptr, w.dtype, w.cols, w.row_bytes())
        };
        let sfx = suffix(dtype).ok_or_else(|| {
            Error::Unsupported(format!("{} has no CUDA matmul kernel", dtype.name()))
        })?;
        let name = format!("mul_mat_{sfx}");
        let f = self.func(&name)?;
        let d = self.dptr(dst)?;
        let bias = self.opt_dptr(cfg.bias)?;
        let grid = ((cfg.n_out + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
        let block = (WARP, ROWS_PER_BLOCK, 1);

        if is_float(dtype) {
            let s = self.dptr(src)?;
            let mut l = Launch::new()
                .ptr(ptr)
                .ptr(s)
                .ptr(d)
                .ptr(bias)
                .i32(cfg.n_in as i32)
                .i32(cfg.n_out as i32)
                .i32(cfg.n_tokens as i32)
                .i32(row_bytes as i32);
            return unsafe { l.run(&self.drv, f, grid, block, 0, self.stream, &name) };
        }

        let staged = self.stage_activation(src, cfg.n_in, cfg.n_tokens)?;
        let blocks_per_row = (cols.div_ceil(32)) as i32;
        let mut l = Launch::new()
            .ptr(ptr)
            .ptr(staged)
            .ptr(d)
            .ptr(bias)
            .i32(cfg.n_in as i32)
            .i32(cfg.n_out as i32)
            .i32(cfg.n_tokens as i32)
            .i32(row_bytes as i32)
            .i32(blocks_per_row);
        unsafe { l.run(&self.drv, f, grid, block, 0, self.stream, &name) }
    }

    fn matmul_f32(&mut self, dst: BufferId, weight: BufferId, src: BufferId, cfg: MatMulCfg) -> Result<()> {
        let f = self.func("mul_mat_dense")?;
        let (d, w, s) = (self.dptr(dst)?, self.dptr(weight)?, self.dptr(src)?);
        let bias = self.opt_dptr(cfg.bias)?;
        let mut l = Launch::new()
            .ptr(w)
            .ptr(s)
            .ptr(d)
            .ptr(bias)
            .i32(cfg.n_in as i32)
            .i32(cfg.n_out as i32)
            .i32(cfg.n_tokens as i32);
        unsafe {
            l.run(
                &self.drv,
                f,
                ((cfg.n_out + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1),
                (WARP, ROWS_PER_BLOCK, 1),
                0,
                self.stream,
                "mul_mat_dense",
            )
        }
    }

    fn rope(&mut self, q: BufferId, k: Option<BufferId>, positions: &[i32], cfg: RopeCfg) -> Result<()> {
        if cfg.kind == RopeKind::None || cfg.n_rot == 0 {
            return Ok(());
        }
        let pos_bytes = (positions.len() * 4) as u64;
        let pos = self.grow(1, pos_bytes)?;
        let raw: Vec<u8> = positions.iter().flat_map(|p| p.to_ne_bytes()).collect();
        self.upload_bytes(pos, &raw)?;

        let factors = self.opt_dptr(cfg.freq_factors)?;
        // M-RoPE with one position axis is arithmetically NeoX, which is the text case.
        let kind_code = if cfg.kind == RopeKind::Norm { 0 } else { 1 };
        let (corr_lo, corr_hi) = yarn_corr_dims(&cfg);

        let f = self.func("rope_kernel")?;
        for (buf, heads) in [(Some(q), cfg.n_heads), (k, cfg.n_kv_heads)] {
            let Some(buf) = buf else { continue };
            let p = self.dptr(buf)?;
            let total = cfg.n_tokens * heads * (cfg.n_rot / 2);
            let mut l = Launch::new()
                .ptr(p)
                .ptr(pos)
                .ptr(factors)
                .i32(heads as i32)
                .i32(cfg.head_dim as i32)
                .i32(cfg.n_rot as i32)
                .i32(cfg.n_tokens as i32)
                .i32(kind_code)
                .f32(cfg.freq_base)
                .f32(cfg.freq_scale)
                .f32(cfg.ext_factor)
                .f32(cfg.attn_factor)
                .f32(corr_lo)
                .f32(corr_hi);
            unsafe {
                l.run(&self.drv, f, grid_1d(total, 256), (256, 1, 1), 0, self.stream, "rope")?;
            }
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
        let name = if kv.is_f16() { "kv_write_f16" } else { "kv_write_f32" };
        let f = self.func(name)?;
        let (kp, vp) = (self.dptr(k)?, self.dptr(v)?);
        let (kc, vc) = (self.dptr(kv.k)?, self.dptr(kv.v)?);
        let mut l = Launch::new()
            .ptr(kp)
            .ptr(vp)
            .ptr(kc)
            .ptr(vc)
            .i32(dim as i32)
            .i32(kv.stride as i32)
            .i32(kv.base as i32)
            .i32(start_pos as i32)
            .i32(n_tokens as i32);
        unsafe {
            l.run(
                &self.drv,
                f,
                grid_1d(dim * n_tokens, 256),
                (256, 1, 1),
                0,
                self.stream,
                name,
            )
        }
    }

    fn attention(&mut self, dst: BufferId, q: BufferId, kv: &KvView, cfg: AttnCfg) -> Result<()> {
        if cfg.head_dim > 256 {
            return Err(Error::Unsupported(format!(
                "head_dim {} exceeds the 256 this backend stages in shared memory",
                cfg.head_dim
            )));
        }
        let heads_times_tokens = cfg.n_heads * cfg.n_tokens;
        // Enough splits to fill the machine, but never so many that a split covers only a
        // handful of keys and the combine pass dominates.
        let sms = self.info.caps.multiprocessors.max(1);
        let want = (sms * 2).div_ceil(heads_times_tokens.max(1));
        let by_length = (cfg.kv_len / 64).max(1);
        let n_splits = want.clamp(1, 32).min(by_length).max(1);

        let partial_bytes =
            (heads_times_tokens as u64) * n_splits as u64 * (cfg.head_dim as u64 + 2) * 4;
        let partial = self.grow(5, partial_bytes)?;

        let (d, qp) = (self.dptr(dst)?, self.dptr(q)?);
        let (kc, vc) = (self.dptr(kv.k)?, self.dptr(kv.v)?);
        let sinks = self.opt_dptr(cfg.sinks)?;

        let split_name = if kv.is_f16() { "flash_attn_split_f16" } else { "flash_attn_split_f32" };
        let split = self.func(split_name)?;
        let mut l = Launch::new()
            .ptr(qp)
            .ptr(kc)
            .ptr(vc)
            .ptr(partial)
            .i32(cfg.n_heads as i32)
            .i32(cfg.n_kv_heads as i32)
            .i32(cfg.head_dim as i32)
            .i32(cfg.n_tokens as i32)
            .i32(cfg.kv_len as i32)
            .i32(cfg.start_pos as i32)
            .i32(kv.stride as i32)
            .i32(kv.base as i32)
            .f32(cfg.scale)
            .f32(cfg.softcap)
            .i32(cfg.window as i32)
            .i32(n_splits as i32);
        unsafe {
            l.run(
                &self.drv,
                split,
                (heads_times_tokens, n_splits, 1),
                (128, 1, 1),
                0,
                self.stream,
                split_name,
            )?;
        }

        let combine = self.func("flash_attn_combine")?;
        let mut l = Launch::new()
            .ptr(partial)
            .ptr(d)
            .ptr(sinks)
            .i32(cfg.n_heads as i32)
            .i32(cfg.head_dim as i32)
            .i32(cfg.n_tokens as i32)
            .i32(n_splits as i32);
        unsafe {
            l.run(
                &self.drv,
                combine,
                (heads_times_tokens, 1, 1),
                (128, 1, 1),
                0,
                self.stream,
                "flash_attn_combine",
            )
        }
    }

    fn glu(&mut self, dst: BufferId, gate: BufferId, up: BufferId, cfg: GluCfg) -> Result<()> {
        let n = cfg.dim * cfg.n_tokens;
        let (d, g, u) = (self.dptr(dst)?, self.dptr(gate)?, self.dptr(up)?);
        self.launch_1d("glu_kernel", n, |l| {
            l.ptr(d).ptr(g).ptr(u).i32(n as i32).i32(act_code(cfg.act)).f32(cfg.limit).f32(cfg.alpha)
        })
    }

    fn activate(&mut self, buf: BufferId, cfg: GluCfg) -> Result<()> {
        let n = cfg.dim * cfg.n_tokens;
        let b = self.dptr(buf)?;
        self.launch_1d("activate_kernel", n, |l| {
            l.ptr(b).i32(n as i32).i32(act_code(cfg.act)).f32(cfg.alpha)
        })
    }

    fn moe(&mut self, dst: BufferId, src: BufferId, w: &MoeWeights, cfg: MoeCfg) -> Result<()> {
        self.moe_impl(dst, src, w, cfg)
    }

    fn binary(&mut self, dst: BufferId, a: BufferId, b: BufferId, op: BinOp, n: u32) -> Result<()> {
        let (d, x, y) = (self.dptr(dst)?, self.dptr(a)?, self.dptr(b)?);
        let bn = self
            .buffers
            .get(b.0 as usize)
            .and_then(|s| s.as_ref())
            .map(|s| (s.bytes / 4) as u32)
            .unwrap_or(n)
            .min(n)
            .max(1);
        let code = match op {
            BinOp::Add => 0,
            BinOp::Mul => 1,
            BinOp::Sub => 2,
        };
        self.launch_1d("binary_kernel", n, |l| {
            l.ptr(d).ptr(x).ptr(y).i32(n as i32).i32(bn as i32).i32(code)
        })
    }

    fn scale(&mut self, buf: BufferId, factor: f32, n: u32) -> Result<()> {
        let b = self.dptr(buf)?;
        self.launch_1d("scale_kernel", n, |l| l.ptr(b).f32(factor).i32(n as i32))
    }

    fn softcap(&mut self, buf: BufferId, cap: f32, n: u32) -> Result<()> {
        if cap <= 0.0 {
            return Ok(());
        }
        let b = self.dptr(buf)?;
        self.launch_1d("softcap_kernel", n, |l| l.ptr(b).f32(cap).i32(n as i32))
    }

    fn copy(&mut self, dst: BufferId, dst_off: u64, src: BufferId, src_off: u64, bytes: u64) -> Result<()> {
        let d = self.dptr(dst)? + dst_off;
        let s = self.dptr(src)? + src_off;
        self.drv.check(
            unsafe { (self.drv.cuMemcpyDtoDAsync_v2)(d, s, bytes as usize, self.stream) },
            "device to device copy",
        )
    }

    fn fill(&mut self, buf: BufferId, value: f32, n: u32) -> Result<()> {
        let b = self.dptr(buf)?;
        self.launch_1d("fill_kernel", n, |l| l.ptr(b).f32(value).i32(n as i32))
    }

    fn submit(&mut self) -> Result<()> {
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.drv.check(
            unsafe { (self.drv.cuStreamSynchronize)(self.stream) },
            "waiting for the device",
        )
    }
}

impl CudaBackend {
    /// Mixture-of-experts, kept entirely on the device.
    ///
    /// The previous engine dropped MoE layers to the CPU, which meant copying the
    /// activation across PCIe twice per layer - on a 48-layer model that alone dominated
    /// the step. Routing, expert matmuls and the weighted reduction all run here.
    fn moe_impl(&mut self, dst: BufferId, src: BufferId, w: &MoeWeights, cfg: MoeCfg) -> Result<()> {
        let n_tokens = cfg.n_tokens;
        let n_used = cfg.n_expert_used;
        let slots = n_tokens * n_used;

        // Router logits, into the general aux scratch.
        let logits_bytes = (cfg.n_expert * n_tokens * 4) as u64;
        let sel_idx_bytes = (slots * 4) as u64;
        let sel_w_bytes = (slots * 4) as u64;
        let routing = self.grow(1, logits_bytes + sel_idx_bytes + sel_w_bytes + 4096)?;
        let logits = routing;
        let sel_idx = routing + logits_bytes;
        let sel_w = sel_idx + sel_idx_bytes;

        // Router projection.
        {
            let (gptr, gdtype, gcols, growbytes) = {
                let gw = self
                    .weights
                    .get(w.gate_inp.0 as usize)
                    .and_then(|x| x.as_ref())
                    .ok_or_else(|| backend_err("cuda", "MoE router weight is missing"))?;
                (gw.ptr, gw.dtype, gw.cols, gw.row_bytes())
            };
            let sfx = suffix(gdtype).ok_or_else(|| {
                Error::Unsupported(format!("{} has no CUDA kernel", gdtype.name()))
            })?;
            let name = format!("mul_mat_{sfx}");
            let f = self.func(&name)?;
            let grid = ((cfg.n_expert + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
            if is_float(gdtype) {
                let s = self.dptr(src)?;
                let mut l = Launch::new()
                    .ptr(gptr).ptr(s).ptr(logits).ptr(0)
                    .i32(cfg.n_embd as i32).i32(cfg.n_expert as i32)
                    .i32(n_tokens as i32).i32(growbytes as i32);
                unsafe { l.run(&self.drv, f, grid, (WARP, ROWS_PER_BLOCK, 1), 0, self.stream, &name)? };
            } else {
                let staged = self.stage_activation(src, cfg.n_embd, n_tokens)?;
                let mut l = Launch::new()
                    .ptr(gptr).ptr(staged).ptr(logits).ptr(0)
                    .i32(cfg.n_embd as i32).i32(cfg.n_expert as i32)
                    .i32(n_tokens as i32).i32(growbytes as i32)
                    .i32(gcols.div_ceil(32) as i32);
                unsafe { l.run(&self.drv, f, grid, (WARP, ROWS_PER_BLOCK, 1), 0, self.stream, &name)? };
            }
        }

        // Top-k selection.
        {
            let probs_b = self.opt_dptr(w.exp_probs_b)?;
            let f = self.func("moe_route")?;
            let mut l = Launch::new()
                .ptr(logits)
                .ptr(probs_b)
                .ptr(sel_idx)
                .ptr(sel_w)
                .i32(cfg.n_expert as i32)
                .i32(n_used as i32)
                .i32(n_tokens as i32)
                .i32(if cfg.gate_func == gguf_core::GateFunc::Softmax { 0 } else { 1 })
                .i32(cfg.norm_topk as i32)
                .f32(cfg.scale);
            unsafe {
                l.run(&self.drv, f, grid_1d(n_tokens, 64), (64, 1, 1), 0, self.stream, "moe_route")?;
            }
        }

        // Replicate each token's quantized activation into its expert slots.
        let embd_blocks = (cfg.n_embd as usize).div_ceil(32);
        let slot_act = {
            let staged = self.stage_activation(src, cfg.n_embd, n_tokens)?;
            let bytes = (embd_blocks as u64) * Q8_1_BLOCK as u64 * slots as u64;
            let out = self.grow(2, bytes)?;
            let f = self.func("moe_scatter")?;
            let total = (embd_blocks as u32) * slots;
            let mut l = Launch::new()
                .ptr(staged)
                .ptr(out)
                .i32(embd_blocks as i32)
                .i32(n_used as i32)
                .i32(n_tokens as i32);
            unsafe {
                l.run(&self.drv, f, grid_1d(total, 256), (256, 1, 1), 0, self.stream, "moe_scatter")?;
            }
            out
        };

        // gate and up, then the activation, then down.
        let ff_bytes = (cfg.n_ff as u64) * slots as u64 * 4;
        let gate_buf = self.grow(3, ff_bytes * 2)?;
        let up_buf = gate_buf + ff_bytes;

        if let Some(gate_w) = w.gate_exps {
            self.moe_matmul(gate_w, slot_act, gate_buf, cfg.n_ff, cfg.n_embd, sel_idx, slots, n_used, n_tokens)?;
        }
        self.moe_matmul(w.up_exps, slot_act, up_buf, cfg.n_ff, cfg.n_embd, sel_idx, slots, n_used, n_tokens)?;

        {
            let f = self.func("moe_glu")?;
            let n = cfg.n_ff * slots;
            let mut l = Launch::new()
                .ptr(gate_buf)
                .ptr(up_buf)
                .i32(n as i32)
                .i32(act_code(cfg.act))
                .i32(w.gate_exps.is_some() as i32)
                .f32(1.0);
            unsafe {
                l.run(&self.drv, f, grid_1d(n, 256), (256, 1, 1), 0, self.stream, "moe_glu")?;
            }
        }

        // The down projection needs its input quantized; reuse the gate buffer's contents.
        let ff_blocks = (cfg.n_ff as usize).div_ceil(32);
        let down_in = {
            let bytes = (ff_blocks as u64) * Q8_1_BLOCK as u64 * slots as u64;
            let out = self.grow(4, bytes)?;
            let f = self.func("quantize_q8_1")?;
            let total = (ff_blocks as u32) * slots;
            let mut l = Launch::new()
                .ptr(gate_buf)
                .ptr(out)
                .i32(cfg.n_ff as i32)
                .i32(ff_blocks as i32)
                .i32(slots as i32);
            unsafe {
                l.run(&self.drv, f, grid_1d(total, 256), (256, 1, 1), 0, self.stream, "quantize_q8_1")?;
            }
            out
        };

        // Down projection writes per-slot rows, reusing the gate buffer as the destination.
        self.moe_matmul(w.down_exps, down_in, gate_buf, cfg.n_embd, cfg.n_ff, sel_idx, slots, n_used, n_tokens)?;

        // Weighted sum back into one row per token.
        let d = self.dptr(dst)?;
        let f = self.func("moe_reduce")?;
        let n = cfg.n_embd * n_tokens;
        let mut l = Launch::new()
            .ptr(d)
            .ptr(gate_buf)
            .ptr(sel_w)
            .i32(cfg.n_embd as i32)
            .i32(n_used as i32)
            .i32(n_tokens as i32);
        unsafe { l.run(&self.drv, f, grid_1d(n, 256), (256, 1, 1), 0, self.stream, "moe_reduce") }
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_matmul(
        &mut self,
        weight: WeightId,
        act: CUdeviceptr,
        dst: CUdeviceptr,
        n_out: u32,
        n_in: u32,
        sel_idx: CUdeviceptr,
        slots: u32,
        n_used: u32,
        n_tokens: u32,
    ) -> Result<()> {
        let _ = n_in;
        let (ptr, dtype, cols, row_bytes, rows) = {
            let w = self
                .weights
                .get(weight.0 as usize)
                .and_then(|x| x.as_ref())
                .ok_or_else(|| backend_err("cuda", "MoE expert weight is missing"))?;
            (w.ptr, w.dtype, w.cols, w.row_bytes(), w.rows)
        };
        let sfx = suffix(dtype)
            .ok_or_else(|| Error::Unsupported(format!("{} has no CUDA kernel", dtype.name())))?;
        let name = format!("moe_mm_{sfx}");
        let f = self.func(&name)?;
        // Experts are stacked contiguously, so one expert's slice is a plain byte offset.
        debug_assert!(rows % n_out as usize == 0, "expert tensor rows are not a multiple of n_out");
        let stride = (n_out as u64) * row_bytes as u64;
        let mut l = Launch::new()
            .ptr(ptr)
            .ptr(act)
            .ptr(sel_idx)
            .ptr(dst)
            .i32(n_out as i32)
            .i32(n_used as i32)
            .i32(n_tokens as i32)
            .i32(row_bytes as i32)
            .i32(cols.div_ceil(32) as i32)
            .u64(stride);
        unsafe {
            l.run(
                &self.drv,
                f,
                ((n_out + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, slots, 1),
                (WARP, ROWS_PER_BLOCK, 1),
                0,
                self.stream,
                &name,
            )
        }
    }
}

/// YaRN's correction range, in rotary pair units. Mirrors the CPU path exactly so the two
/// backends agree on positions.
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

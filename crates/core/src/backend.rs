use crate::device::{BufferId, DeviceInfo, MemKind, WeightId};
use crate::dtype::{DType, QuantView};
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    Rms,
    Layer,
}

#[derive(Debug, Clone, Copy)]
pub struct NormCfg {
    pub kind: NormKind,
    pub dim: u32,
    pub n_tokens: u32,
    pub eps: f32,
    pub bias: Option<BufferId>,
    /// Extra constant multiplier folded into the norm.
    pub scale: f32,
}

impl NormCfg {
    pub fn rms(dim: u32, n_tokens: u32, eps: f32) -> Self {
        Self {
            kind: NormKind::Rms,
            dim,
            n_tokens,
            eps,
            bias: None,
            scale: 1.0,
        }
    }

    pub fn layer(dim: u32, n_tokens: u32, eps: f32) -> Self {
        Self { kind: NormKind::Layer, ..Self::rms(dim, n_tokens, eps) }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMulCfg {
    /// Rows of the weight, i.e. the output dimension.
    pub n_out: u32,
    /// Columns of the weight, i.e. the input dimension.
    pub n_in: u32,
    pub n_tokens: u32,
    pub bias: Option<BufferId>,
}

impl MatMulCfg {
    pub fn new(n_out: u32, n_in: u32, n_tokens: u32) -> Self {
        Self { n_out, n_in, n_tokens, bias: None }
    }

    pub fn with_bias(mut self, bias: Option<BufferId>) -> Self {
        self.bias = bias;
        self
    }
}

/// Which pairs of head-dimension elements get rotated together.
///
/// `Norm` rotates adjacent pairs (2i, 2i+1); `Neox` rotates (i, i + n_rot/2). The two are
/// not interchangeable: converters permute Q/K weights to match one of them, so the wrong
/// layout yields fluent-looking but wrong output rather than an obvious failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeKind {
    Norm,
    Neox,
    /// Multimodal RoPE: `sections` splits the rotary dims across position axes.
    MRope,
    /// Position-independent.
    None,
}

#[derive(Debug, Clone, Copy)]
pub struct RopeCfg {
    pub kind: RopeKind,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Rotary dimensions; may be smaller than `head_dim` for partial-rotary models.
    pub n_rot: u32,
    pub n_tokens: u32,
    pub freq_base: f32,
    /// Reciprocal of the linear context-extension factor. 1.0 means no scaling.
    pub freq_scale: f32,
    /// YaRN interpolation/extrapolation mix. 0.0 disables YaRN.
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub orig_ctx: u32,
    /// M-RoPE section widths.
    pub sections: [u32; 4],
    /// Per-dimension frequency divisors. This is how Llama-3 and LongRoPE scaling arrive:
    /// both are precomputed into a rope_freqs tensor at load time.
    pub freq_factors: Option<BufferId>,
}

impl RopeCfg {
    pub fn new(kind: RopeKind, n_heads: u32, n_kv_heads: u32, head_dim: u32, freq_base: f32) -> Self {
        Self {
            kind,
            n_heads,
            n_kv_heads,
            head_dim,
            n_rot: head_dim,
            n_tokens: 1,
            freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            orig_ctx: 0,
            sections: [0; 4],
            freq_factors: None,
        }
    }
}

/// A sequence's key/value cache region.
///
/// `dtype` is the cache's storage format, not the activation format. Attention reads and
/// writes f32 either way; f16 storage halves what is by far the largest allocation after
/// the weights themselves, at a precision cost that does not show up in output.
#[derive(Debug, Clone, Copy)]
pub struct KvView {
    pub k: BufferId,
    pub v: BufferId,
    /// Elements between consecutive token slots, i.e. n_kv_heads * head_dim.
    pub stride: u32,
    /// Index of this sequence's first token slot within the cache buffer.
    pub base: u32,
    pub dtype: DType,
}

impl KvView {
    /// Bytes per cached element.
    pub fn elem_size(&self) -> usize {
        self.dtype.type_size()
    }

    pub fn is_f16(&self) -> bool {
        self.dtype == DType::F16
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttnCfg {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub n_tokens: u32,
    /// Valid cached tokens for this sequence, including the ones written this pass.
    pub kv_len: u32,
    /// Absolute position of the first new token, used to build the causal mask.
    pub start_pos: u32,
    pub scale: f32,
    /// Gemma-2 style `cap * tanh(x / cap)` on attention logits. 0.0 disables.
    pub softcap: f32,
    /// Sliding-window width. 0 means attend to the full history.
    pub window: u32,
    /// Per-head learned sink logits appended to the softmax denominator.
    pub sinks: Option<BufferId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    Gelu,
    GeluTanh,
    GeluQuick,
    Relu,
    Relu2,
    Swish,
}

#[derive(Debug, Clone, Copy)]
pub struct GluCfg {
    pub act: Activation,
    pub dim: u32,
    pub n_tokens: u32,
    /// Clamp applied to gate and up before combining. 0.0 disables.
    pub limit: f32,
    /// Swish beta for models whose gate is not plain SiLU.
    pub alpha: f32,
}

impl GluCfg {
    pub fn new(act: Activation, dim: u32, n_tokens: u32) -> Self {
        Self { act, dim, n_tokens, limit: 0.0, alpha: 1.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateFunc {
    Softmax,
    Sigmoid,
}

#[derive(Debug, Clone, Copy)]
pub struct MoeWeights {
    pub gate_inp: WeightId,
    /// Absent for models whose experts have no gate projection.
    pub gate_exps: Option<WeightId>,
    pub up_exps: WeightId,
    pub down_exps: WeightId,
    /// Per-expert routing bias added before top-k selection.
    pub exp_probs_b: Option<BufferId>,
}

#[derive(Debug, Clone, Copy)]
pub struct MoeCfg {
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_tokens: u32,
    pub act: Activation,
    pub gate_func: GateFunc,
    /// Renormalize the selected top-k weights to sum to 1.
    pub norm_topk: bool,
    pub scale: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Mul,
    Sub,
}

/// A device that can execute the transformer op set.
///
/// Ops are recorded; nothing is guaranteed to have run until [`Backend::submit`], and
/// results are not readable until [`Backend::sync`]. The CPU backend executes eagerly and
/// treats both as no-ops.
///
/// Every op is batched over `n_tokens`, with activations laid out `[n_tokens][dim]`
/// contiguously. Adding a backend means implementing this trait, and a missing op is a
/// compile error rather than a silent fallback to something slower or wrong.
pub trait Backend: Send {
    fn info(&self) -> &DeviceInfo;

    /// Keep a shared owner of mapped weight bytes alive for as long as this backend holds
    /// views into them.
    ///
    /// The CPU backend reads weights straight out of the memory map rather than copying
    /// gigabytes it already has; this is what makes that sound. Backends that copy weights
    /// into device memory ignore it.
    fn retain(&mut self, _owner: std::sync::Arc<dyn std::any::Any + Send + Sync>) {}

    fn alloc(&mut self, bytes: u64, kind: MemKind) -> Result<BufferId>;
    fn free(&mut self, id: BufferId);
    fn write(&mut self, dst: BufferId, offset_bytes: u64, src: &[u8]) -> Result<()>;
    fn read(&mut self, src: BufferId, offset_bytes: u64, dst: &mut [u8]) -> Result<()>;

    /// Upload a weight in its native quantized layout. Implementations must not expand the
    /// block encoding: the point is that device memory holds the bytes the file had.
    fn upload_weight(&mut self, view: &QuantView<'_>) -> Result<WeightId>;
    fn free_weight(&mut self, id: WeightId);

    /// `dst[t] = weight[tokens[t]] * scale`
    fn get_rows(&mut self, dst: BufferId, weight: WeightId, tokens: &[u32], scale: f32) -> Result<()>;

    fn norm(&mut self, dst: BufferId, src: BufferId, weight: Option<BufferId>, cfg: NormCfg) -> Result<()>;

    /// `dst = weight @ src`, with `weight` treated as `[n_out, n_in]` row-major.
    fn matmul(&mut self, dst: BufferId, weight: WeightId, src: BufferId, cfg: MatMulCfg) -> Result<()>;

    /// Dense matmul against an f32 buffer rather than an uploaded quantized weight.
    fn matmul_f32(&mut self, dst: BufferId, weight: BufferId, src: BufferId, cfg: MatMulCfg) -> Result<()>;

    fn rope(&mut self, q: BufferId, k: Option<BufferId>, positions: &[i32], cfg: RopeCfg) -> Result<()>;

    /// Scatter this pass's K and V into the cache starting at `start_pos`.
    fn kv_write(
        &mut self,
        kv: &KvView,
        k: BufferId,
        v: BufferId,
        start_pos: u32,
        n_tokens: u32,
        dim: u32,
    ) -> Result<()>;

    fn attention(&mut self, dst: BufferId, q: BufferId, kv: &KvView, cfg: AttnCfg) -> Result<()>;

    /// `dst = act(gate) * up`, fused.
    fn glu(&mut self, dst: BufferId, gate: BufferId, up: BufferId, cfg: GluCfg) -> Result<()>;

    /// `buf = act(buf)`, for the ungated feed-forward blocks that predate SwiGLU.
    fn activate(&mut self, buf: BufferId, cfg: GluCfg) -> Result<()>;

    fn moe(&mut self, dst: BufferId, src: BufferId, w: &MoeWeights, cfg: MoeCfg) -> Result<()>;

    fn binary(&mut self, dst: BufferId, a: BufferId, b: BufferId, op: BinOp, n: u32) -> Result<()>;

    fn scale(&mut self, buf: BufferId, factor: f32, n: u32) -> Result<()>;

    /// `x = cap * tanh(x / cap)`
    fn softcap(&mut self, buf: BufferId, cap: f32, n: u32) -> Result<()>;

    fn copy(&mut self, dst: BufferId, dst_off: u64, src: BufferId, src_off: u64, bytes: u64) -> Result<()>;

    fn fill(&mut self, buf: BufferId, value: f32, n: u32) -> Result<()>;

    fn submit(&mut self) -> Result<()>;
    fn sync(&mut self) -> Result<()>;

    /// Residual add followed by a norm, the shape every transformer block repeats twice:
    /// `residual += delta; dst = norm(residual)`.
    ///
    /// The default composes two ops; backends that can fuse them into a single pass over
    /// the activation override this.
    fn add_norm(
        &mut self,
        dst: BufferId,
        residual: BufferId,
        delta: BufferId,
        weight: Option<BufferId>,
        cfg: NormCfg,
    ) -> Result<()> {
        self.binary(residual, residual, delta, BinOp::Add, cfg.dim * cfg.n_tokens)?;
        self.norm(dst, residual, weight, cfg)
    }

    /// Read a whole f32 buffer back to the host, synchronising first.
    fn read_f32(&mut self, src: BufferId, n: usize) -> Result<Vec<f32>> {
        let mut bytes = vec![0u8; n * 4];
        self.sync()?;
        self.read(src, 0, &mut bytes)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    fn write_f32(&mut self, dst: BufferId, data: &[f32]) -> Result<()> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.write(dst, 0, &bytes)
    }
}

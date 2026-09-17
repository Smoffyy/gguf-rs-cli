//! Binding GGUF tensors to backend handles.
//!
//! Tensor names drifted over the years (`post_attention_norm` vs `attn_post_norm`,
//! `ffn_gate_exps` vs `ffn_gate_exp`), so every lookup goes through a candidate list. A
//! name that matches nothing is either optional, in which case the slot stays `None`, or
//! required, in which case the error names the alternatives that were tried.

use gguf_core::{Backend, BufferId, MemKind, MoeWeights, QuantView, Result, WeightId};
use gguf_format::GgufModel;

use crate::spec::ArchSpec;

pub struct LayerWeights {
    pub attn_norm: Option<BufferId>,
    pub attn_norm_b: Option<BufferId>,
    pub attn_q: WeightId,
    pub attn_k: WeightId,
    pub attn_v: WeightId,
    pub attn_out: WeightId,
    pub q_bias: Option<BufferId>,
    pub k_bias: Option<BufferId>,
    pub v_bias: Option<BufferId>,
    pub out_bias: Option<BufferId>,
    pub q_norm: Option<BufferId>,
    pub k_norm: Option<BufferId>,
    pub post_attn_norm: Option<BufferId>,
    pub ffn_norm: Option<BufferId>,
    pub ffn_norm_b: Option<BufferId>,
    pub ffn_gate: Option<WeightId>,
    pub ffn_up: Option<WeightId>,
    pub ffn_down: Option<WeightId>,
    pub ffn_up_b: Option<BufferId>,
    pub ffn_down_b: Option<BufferId>,
    pub ffn_post_norm: Option<BufferId>,
    pub moe: Option<MoeWeights>,
    /// Shared always-on expert, present in Qwen-MoE and DeepSeek.
    pub shexp: Option<(Option<WeightId>, WeightId, WeightId)>,
    pub sinks: Option<BufferId>,
}

pub struct Weights {
    pub token_embd: WeightId,
    pub output: WeightId,
    pub output_norm: Option<BufferId>,
    pub output_norm_b: Option<BufferId>,
    pub rope_freqs: Option<BufferId>,
    pub layers: Vec<LayerWeights>,
    pub bytes_uploaded: u64,
}

struct Binder<'a> {
    g: &'a GgufModel,
    uploaded: u64,
}

impl<'a> Binder<'a> {
    /// Upload a small tensor as an f32 vector, which is how norm weights and biases are
    /// consumed by every op that takes them.
    fn vector(&mut self, be: &mut dyn Backend, names: &[&str]) -> Result<Option<BufferId>> {
        let Some((_, view)) = self.g.view_any(names) else {
            return Ok(None);
        };
        let n = view.rows * view.cols;
        let data = gguf_quant::dequantize_vec(view.dtype, view.data, n)?;
        let buf = be.alloc((n * 4) as u64, MemKind::Device)?;
        be.write_f32(buf, &data)?;
        Ok(Some(buf))
    }

    fn matrix(&mut self, be: &mut dyn Backend, names: &[&str]) -> Result<Option<WeightId>> {
        let Some((_, view)) = self.g.view_any(names) else {
            return Ok(None);
        };
        self.uploaded += view.data.len() as u64;
        Ok(Some(be.upload_weight(&view)?))
    }

    fn matrix_req(&mut self, be: &mut dyn Backend, names: &[&str]) -> Result<WeightId> {
        let (_, view) = self.g.view_any_req(names)?;
        self.uploaded += view.data.len() as u64;
        be.upload_weight(&view)
    }

    /// Upload a contiguous row range of a tensor as a weight in its own right.
    ///
    /// Fused QKV and fused gate/up tensors are stored as one matrix whose rows are the
    /// concatenation of the parts. Because every block layout is row-aligned, the split is
    /// just a byte range: no dequantization, no copy, no extra memory.
    fn row_slice(
        &mut self,
        be: &mut dyn Backend,
        view: &QuantView<'a>,
        start: usize,
        count: usize,
    ) -> Result<WeightId> {
        let rb = view.row_bytes();
        let sub = QuantView::new(&view.data[start * rb..(start + count) * rb], view.dtype, count, view.cols);
        self.uploaded += sub.data.len() as u64;
        be.upload_weight(&sub)
    }
}

fn blk(i: usize, suffix: &str) -> String {
    format!("blk.{i}.{suffix}")
}

impl Weights {
    pub fn load(g: &GgufModel, spec: &ArchSpec, be: &mut dyn Backend) -> Result<Self> {
        let mut b = Binder { g, uploaded: 0 };

        let token_embd = b.matrix_req(be, &["token_embd.weight"])?;
        let output = if spec.tied_embeddings {
            token_embd
        } else {
            b.matrix_req(be, &["output.weight"])?
        };
        let output_norm = b.vector(be, &["output_norm.weight"])?;
        let output_norm_b = b.vector(be, &["output_norm.bias"])?;
        let rope_freqs = b.vector(be, &["rope_freqs.weight"])?;

        let mut layers = Vec::with_capacity(spec.n_layers);
        for i in 0..spec.n_layers {
            layers.push(Self::layer(&mut b, be, spec, i)?);
        }

        Ok(Self {
            token_embd,
            output,
            output_norm,
            output_norm_b,
            rope_freqs,
            layers,
            bytes_uploaded: b.uploaded,
        })
    }

    fn layer(b: &mut Binder<'_>, be: &mut dyn Backend, spec: &ArchSpec, i: usize) -> Result<LayerWeights> {
        let n_q = spec.n_embd_q();
        let n_kv = spec.n_embd_k();

        // Q/K/V are either three tensors or one fused one.
        let (attn_q, attn_k, attn_v) = match b.g.view_opt(&blk(i, "attn_qkv.weight")) {
            Some(fused) if !b.g.contains(&blk(i, "attn_q.weight")) => (
                b.row_slice(be, &fused, 0, n_q)?,
                b.row_slice(be, &fused, n_q, n_kv)?,
                b.row_slice(be, &fused, n_q + n_kv, n_kv)?,
            ),
            _ => (
                b.matrix_req(be, &[&blk(i, "attn_q.weight")])?,
                b.matrix_req(be, &[&blk(i, "attn_k.weight")])?,
                b.matrix_req(be, &[&blk(i, "attn_v.weight")])?,
            ),
        };

        // Gate and up likewise, for models that fuse them.
        let (ffn_gate, ffn_up) = match b.g.view_opt(&blk(i, "ffn_gate_up.weight")) {
            Some(fused) => {
                let half = fused.rows / 2;
                (
                    Some(b.row_slice(be, &fused, 0, half)?),
                    Some(b.row_slice(be, &fused, half, half)?),
                )
            }
            None => (
                b.matrix(be, &[&blk(i, "ffn_gate.weight")])?,
                b.matrix(be, &[&blk(i, "ffn_up.weight")])?,
            ),
        };

        let moe = match b.matrix(be, &[&blk(i, "ffn_gate_inp.weight")])? {
            Some(gate_inp) => Some(MoeWeights {
                gate_inp,
                gate_exps: b.matrix(be, &[&blk(i, "ffn_gate_exps.weight"), &blk(i, "ffn_gate_exp.weight")])?,
                up_exps: b
                    .matrix(be, &[&blk(i, "ffn_up_exps.weight"), &blk(i, "ffn_up_exp.weight")])?
                    .ok_or_else(|| {
                        gguf_core::Error::MissingTensor(blk(i, "ffn_up_exps.weight"))
                    })?,
                down_exps: b
                    .matrix(be, &[&blk(i, "ffn_down_exps.weight"), &blk(i, "ffn_down_exp.weight")])?
                    .ok_or_else(|| {
                        gguf_core::Error::MissingTensor(blk(i, "ffn_down_exps.weight"))
                    })?,
                exp_probs_b: b.vector(be, &[&blk(i, "exp_probs_b.bias")])?,
            }),
            None => None,
        };

        let shexp = match b.matrix(be, &[&blk(i, "ffn_up_shexp.weight")])? {
            Some(up) => Some((
                b.matrix(be, &[&blk(i, "ffn_gate_shexp.weight")])?,
                up,
                b.matrix(be, &[&blk(i, "ffn_down_shexp.weight")])?.ok_or_else(|| {
                    gguf_core::Error::MissingTensor(blk(i, "ffn_down_shexp.weight"))
                })?,
            )),
            None => None,
        };

        let _ = spec;
        Ok(LayerWeights {
            attn_norm: b.vector(be, &[&blk(i, "attn_norm.weight"), &blk(i, "pre_attention_layernorm.weight")])?,
            attn_norm_b: b.vector(be, &[&blk(i, "attn_norm.bias")])?,
            attn_q,
            attn_k,
            attn_v,
            attn_out: b.matrix_req(be, &[&blk(i, "attn_output.weight"), &blk(i, "attn_out.weight")])?,
            q_bias: b.vector(be, &[&blk(i, "attn_q.bias")])?,
            k_bias: b.vector(be, &[&blk(i, "attn_k.bias")])?,
            v_bias: b.vector(be, &[&blk(i, "attn_v.bias")])?,
            out_bias: b.vector(be, &[&blk(i, "attn_output.bias")])?,
            q_norm: b.vector(be, &[&blk(i, "attn_q_norm.weight")])?,
            k_norm: b.vector(be, &[&blk(i, "attn_k_norm.weight")])?,
            post_attn_norm: b.vector(
                be,
                &[&blk(i, "post_attention_norm.weight"), &blk(i, "attn_post_norm.weight")],
            )?,
            ffn_norm: b.vector(
                be,
                &[
                    &blk(i, "ffn_norm.weight"),
                    &blk(i, "post_attention_layernorm.weight"),
                    &blk(i, "pre_ff_layernorm.weight"),
                ],
            )?,
            ffn_norm_b: b.vector(be, &[&blk(i, "ffn_norm.bias")])?,
            ffn_gate,
            ffn_up,
            ffn_down: b.matrix(be, &[&blk(i, "ffn_down.weight")])?,
            ffn_up_b: b.vector(be, &[&blk(i, "ffn_up.bias")])?,
            ffn_down_b: b.vector(be, &[&blk(i, "ffn_down.bias")])?,
            ffn_post_norm: b.vector(
                be,
                &[&blk(i, "post_ffw_norm.weight"), &blk(i, "ffn_post_norm.weight")],
            )?,
            moe,
            shexp,
            sinks: b.vector(be, &[&blk(i, "attn_sinks.weight"), &blk(i, "attn_sinks")])?,
        })
    }
}

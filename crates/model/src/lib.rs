//! The model layer: what a GGUF says it is, and how to run it.
//!
//! There is one transformer graph here, not one per architecture. Everything a decoder
//! block varies on is a field of [`ArchSpec`], which is inferred from the file, so an
//! architecture this code has never heard of runs as long as its block is shaped like a
//! transformer block. That is the whole point: the model zoo moves faster than any table
//! of names can be maintained.

pub mod spec;
mod weights;

pub use spec::{ArchSpec, BlockStyle, MoeSpec, RopeSpec};
pub use weights::{LayerWeights, Weights};

use gguf_core::{
    AttnCfg, Backend, BinOp, BufferId, Error, GluCfg, KvView, MatMulCfg, MemKind, MoeCfg, NormCfg,
    Result, RopeCfg,
};
use gguf_format::GgufModel;

/// Per-pass activation buffers, sized for the largest batch the model will see.
struct Scratch {
    x: BufferId,
    xn: BufferId,
    xn2: BufferId,
    q: BufferId,
    k: BufferId,
    v: BufferId,
    attn: BufferId,
    proj: BufferId,
    gate: BufferId,
    up: BufferId,
    ff: BufferId,
    last: BufferId,
    logits: BufferId,
    max_tokens: usize,
}

pub struct Model {
    pub spec: ArchSpec,
    pub weights: Weights,
    scratch: Scratch,
}

impl Model {
    /// Upload every tensor and allocate activation space for batches of up to `max_batch`.
    pub fn load(g: &GgufModel, be: &mut dyn Backend, max_batch: usize) -> Result<Self> {
        let spec = ArchSpec::infer(g)?;
        Self::check_supported(g, &spec)?;

        let weights = Weights::load(g, &spec, be)?;
        let scratch = Self::alloc_scratch(be, &spec, max_batch)?;
        Ok(Self { spec, weights, scratch })
    }

    fn check_supported(g: &GgufModel, spec: &ArchSpec) -> Result<()> {
        for dt in g.dtypes() {
            dt.check_supported()?;
        }
        // A recurrent or linear-attention block is a different layer type, not a variation
        // on a transformer block, so no amount of inference makes it run here. Say so with
        // the tensor that gave it away rather than producing confident nonsense.
        if let Some(name) = (0..spec.n_layers.min(8))
            .flat_map(|i| g.block_tensors(i))
            .find(|n| n.contains(".ssm_") || n.contains(".time_mix_") || n.contains(".shortconv"))
        {
            return Err(Error::Unsupported(format!(
                "{} is a hybrid state-space / linear-attention model ({name} is not part of \
                 a transformer block). This engine implements attention-based decoders only",
                spec.arch
            )));
        }
        if g.contains("blk.0.attn_kv_a_mqa.weight") || g.contains("blk.0.attn_q_a.weight") {
            return Err(Error::Unsupported(format!(
                "{} uses multi-head latent attention (MLA), which this engine does not \
                 implement yet; its attention block is shaped differently from a standard \
                 transformer rather than being a variation on one",
                spec.arch
            )));
        }
        if spec.n_heads == 0 || spec.n_layers == 0 {
            return Err(Error::Unsupported(format!(
                "{} declares {} heads and {} layers; this does not look like a decoder model",
                spec.arch, spec.n_heads, spec.n_layers
            )));
        }
        Ok(())
    }

    fn alloc_scratch(be: &mut dyn Backend, s: &ArchSpec, max_batch: usize) -> Result<Scratch> {
        let t = max_batch.max(1);
        let f = |n: usize| (n * 4) as u64;
        let ff_dim = s
            .moe
            .as_ref()
            .map(|m| m.n_ff_exp.max(s.n_ff))
            .unwrap_or(s.n_ff)
            .max(1);
        Ok(Scratch {
            x: be.alloc(f(t * s.n_embd), MemKind::Device)?,
            xn: be.alloc(f(t * s.n_embd), MemKind::Device)?,
            xn2: be.alloc(f(t * s.n_embd), MemKind::Device)?,
            q: be.alloc(f(t * s.n_embd_q()), MemKind::Device)?,
            k: be.alloc(f(t * s.n_embd_k()), MemKind::Device)?,
            v: be.alloc(f(t * s.n_embd_k()), MemKind::Device)?,
            attn: be.alloc(f(t * s.n_embd_q()), MemKind::Device)?,
            proj: be.alloc(f(t * s.n_embd), MemKind::Device)?,
            gate: be.alloc(f(t * ff_dim), MemKind::Device)?,
            up: be.alloc(f(t * ff_dim), MemKind::Device)?,
            ff: be.alloc(f(t * s.n_embd), MemKind::Device)?,
            last: be.alloc(f(s.n_embd), MemKind::Device)?,
            logits: be.alloc(f(s.n_vocab), MemKind::Device)?,
            max_tokens: t,
        })
    }

    pub fn max_batch(&self) -> usize {
        self.scratch.max_tokens
    }

    /// Run `tokens` through the model and return the logits for the final token.
    ///
    /// Only the last position's logits are computed: the output projection is the widest
    /// matmul in the model, and prefill never needs the intermediate rows.
    pub fn forward(
        &mut self,
        be: &mut dyn Backend,
        tokens: &[u32],
        positions: &[i32],
        kv: &[KvView],
        start_pos: u32,
        kv_len: u32,
    ) -> Result<Vec<f32>> {
        let n = tokens.len();
        if n == 0 {
            return Err(Error::Shape("forward called with no tokens".into()));
        }
        if n > self.scratch.max_tokens {
            return Err(Error::Shape(format!(
                "batch of {n} exceeds the {} this model was loaded for",
                self.scratch.max_tokens
            )));
        }
        if kv.len() < self.spec.n_layers {
            return Err(Error::Shape(format!(
                "need {} kv views, got {}",
                self.spec.n_layers,
                kv.len()
            )));
        }

        let s = &self.spec;
        be.get_rows(self.scratch.x, self.weights.token_embd, tokens, s.embd_scale)?;

        for l in 0..s.n_layers {
            self.layer(be, l, positions, &kv[l], start_pos, kv_len, n)?;
        }

        // Pull the final token's hidden state out before the vocabulary projection.
        let off = ((n - 1) * s.n_embd * 4) as u64;
        be.copy(self.scratch.last, 0, self.scratch.x, off, (s.n_embd * 4) as u64)?;

        let norm_cfg = self.norm_cfg(1);
        be.norm(
            self.scratch.last,
            self.scratch.last,
            self.weights.output_norm,
            NormCfg { bias: self.weights.output_norm_b, ..norm_cfg },
        )?;
        be.matmul(
            self.scratch.logits,
            self.weights.output,
            self.scratch.last,
            MatMulCfg::new(s.n_vocab as u32, s.n_embd as u32, 1),
        )?;
        if s.logit_scale != 1.0 {
            be.scale(self.scratch.logits, s.logit_scale, s.n_vocab as u32)?;
        }
        if s.final_softcap > 0.0 {
            be.softcap(self.scratch.logits, s.final_softcap, s.n_vocab as u32)?;
        }
        be.submit()?;
        be.read_f32(self.scratch.logits, s.n_vocab)
    }

    fn norm_cfg(&self, n_tokens: usize) -> NormCfg {
        let s = &self.spec;
        NormCfg {
            kind: s.norm_kind,
            dim: s.n_embd as u32,
            n_tokens: n_tokens as u32,
            eps: s.norm_eps,
            bias: None,
            scale: 1.0,
        }
    }

    fn layer(
        &self,
        be: &mut dyn Backend,
        l: usize,
        positions: &[i32],
        kv: &KvView,
        start_pos: u32,
        kv_len: u32,
        n: usize,
    ) -> Result<()> {
        let s = &self.spec;
        let w = &self.weights.layers[l];
        let sc = &self.scratch;
        let nt = n as u32;

        // The tensor feeding attention: pre-norm styles normalize first, post-norm feeds
        // the raw residual straight in.
        let attn_in = match s.style {
            BlockStyle::PostNorm => sc.x,
            _ => {
                be.norm(
                    sc.xn,
                    sc.x,
                    w.attn_norm,
                    NormCfg { bias: w.attn_norm_b, ..self.norm_cfg(n) },
                )?;
                sc.xn
            }
        };

        be.matmul(
            sc.q,
            w.attn_q,
            attn_in,
            MatMulCfg::new(s.n_embd_q() as u32, s.n_embd as u32, nt).with_bias(w.q_bias),
        )?;
        be.matmul(
            sc.k,
            w.attn_k,
            attn_in,
            MatMulCfg::new(s.n_embd_k() as u32, s.n_embd as u32, nt).with_bias(w.k_bias),
        )?;
        be.matmul(
            sc.v,
            w.attn_v,
            attn_in,
            MatMulCfg::new(s.n_embd_k() as u32, s.n_embd as u32, nt).with_bias(w.v_bias),
        )?;

        // QK-norm normalizes each head independently, so it is the same RMSNorm op with
        // the head as the row rather than the token.
        if let Some(qn) = w.q_norm {
            be.norm(
                sc.q,
                sc.q,
                Some(qn),
                NormCfg {
                    dim: s.head_dim as u32,
                    n_tokens: (n * s.n_heads) as u32,
                    ..self.norm_cfg(n)
                },
            )?;
        }
        if let Some(kn) = w.k_norm {
            be.norm(
                sc.k,
                sc.k,
                Some(kn),
                NormCfg {
                    dim: s.head_dim as u32,
                    n_tokens: (n * s.n_kv_heads) as u32,
                    ..self.norm_cfg(n)
                },
            )?;
        }

        be.rope(sc.q, Some(sc.k), positions, self.rope_cfg(l, n))?;
        be.kv_write(kv, sc.k, sc.v, start_pos, nt, s.n_embd_k() as u32)?;
        be.attention(
            sc.attn,
            sc.q,
            kv,
            AttnCfg {
                n_heads: s.n_heads as u32,
                n_kv_heads: s.n_kv_heads as u32,
                head_dim: s.head_dim as u32,
                n_tokens: nt,
                kv_len,
                start_pos,
                scale: s.attn_scale,
                softcap: s.attn_softcap,
                window: if s.layer_uses_swa(l) { s.swa_window } else { 0 },
                sinks: w.sinks,
            },
        )?;
        be.matmul(
            sc.proj,
            w.attn_out,
            sc.attn,
            MatMulCfg::new(s.n_embd as u32, s.n_embd_q() as u32, nt).with_bias(w.out_bias),
        )?;
        if let Some(pn) = w.post_attn_norm {
            be.norm(sc.proj, sc.proj, Some(pn), self.norm_cfg(n))?;
        }
        if s.residual_scale != 1.0 {
            be.scale(sc.proj, s.residual_scale, (s.n_embd * n) as u32)?;
        }

        match s.style {
            BlockStyle::ParallelResidual => {
                // Attention and FFN both read the single pre-norm and both land on the
                // original residual, so the FFN input is computed before x is updated.
                self.ffn(be, l, attn_in, n)?;
                be.binary(sc.x, sc.x, sc.proj, BinOp::Add, (s.n_embd * n) as u32)?;
                be.binary(sc.x, sc.x, sc.ff, BinOp::Add, (s.n_embd * n) as u32)?;
            }
            _ => {
                be.binary(sc.x, sc.x, sc.proj, BinOp::Add, (s.n_embd * n) as u32)?;
                let ffn_in = match s.style {
                    BlockStyle::PostNorm => sc.x,
                    _ => {
                        be.norm(
                            sc.xn2,
                            sc.x,
                            w.ffn_norm,
                            NormCfg { bias: w.ffn_norm_b, ..self.norm_cfg(n) },
                        )?;
                        sc.xn2
                    }
                };
                self.ffn(be, l, ffn_in, n)?;
                if let Some(pn) = self.weights.layers[l].ffn_post_norm {
                    be.norm(sc.ff, sc.ff, Some(pn), self.norm_cfg(n))?;
                }
                if s.residual_scale != 1.0 {
                    be.scale(sc.ff, s.residual_scale, (s.n_embd * n) as u32)?;
                }
                be.binary(sc.x, sc.x, sc.ff, BinOp::Add, (s.n_embd * n) as u32)?;
            }
        }
        Ok(())
    }

    /// Writes the feed-forward result into `scratch.ff`.
    fn ffn(&self, be: &mut dyn Backend, l: usize, input: BufferId, n: usize) -> Result<()> {
        let s = &self.spec;
        let w = &self.weights.layers[l];
        let sc = &self.scratch;
        let nt = n as u32;

        if let (Some(moe_w), Some(moe)) = (w.moe.as_ref(), s.moe.as_ref()) {
            be.moe(
                sc.ff,
                input,
                moe_w,
                MoeCfg {
                    n_expert: moe.n_expert as u32,
                    n_expert_used: moe.n_expert_used as u32,
                    n_embd: s.n_embd as u32,
                    n_ff: moe.n_ff_exp as u32,
                    n_tokens: nt,
                    act: s.act,
                    gate_func: moe.gate_func,
                    norm_topk: moe.norm_topk,
                    scale: moe.scale,
                },
            )?;
            // A shared expert runs for every token and is summed with the routed result.
            if let Some((sh_gate, sh_up, sh_down)) = w.shexp.as_ref() {
                let n_ff = moe.n_ff_shexp.max(1);
                self.dense_ffn(be, *sh_gate, *sh_up, *sh_down, None, None, input, n, n_ff, sc.proj)?;
                be.binary(sc.ff, sc.ff, sc.proj, BinOp::Add, (s.n_embd * n) as u32)?;
            }
            return Ok(());
        }

        let n_ff = s.n_ff;
        let down = w
            .ffn_down
            .ok_or_else(|| Error::MissingTensor(format!("blk.{l}.ffn_down.weight")))?;
        let up = w
            .ffn_up
            .ok_or_else(|| Error::MissingTensor(format!("blk.{l}.ffn_up.weight")))?;
        self.dense_ffn(be, w.ffn_gate, up, down, w.ffn_up_b, w.ffn_down_b, input, n, n_ff, sc.ff)
    }

    #[allow(clippy::too_many_arguments)]
    fn dense_ffn(
        &self,
        be: &mut dyn Backend,
        gate_w: Option<gguf_core::WeightId>,
        up_w: gguf_core::WeightId,
        down_w: gguf_core::WeightId,
        up_b: Option<BufferId>,
        down_b: Option<BufferId>,
        input: BufferId,
        n: usize,
        n_ff: usize,
        dst: BufferId,
    ) -> Result<()> {
        let s = &self.spec;
        let sc = &self.scratch;
        let nt = n as u32;

        be.matmul(
            sc.up,
            up_w,
            input,
            MatMulCfg::new(n_ff as u32, s.n_embd as u32, nt).with_bias(up_b),
        )?;
        let activated = match gate_w {
            Some(g) => {
                be.matmul(sc.gate, g, input, MatMulCfg::new(n_ff as u32, s.n_embd as u32, nt))?;
                be.glu(
                    sc.gate,
                    sc.gate,
                    sc.up,
                    GluCfg {
                        act: s.act,
                        dim: n_ff as u32,
                        n_tokens: nt,
                        limit: 0.0,
                        alpha: 1.0,
                    },
                )?;
                sc.gate
            }
            None => {
                // Ungated: the activation applies to `up` on its own, with no second
                // branch to multiply against.
                be.activate(sc.up, GluCfg::new(s.act, n_ff as u32, nt))?;
                sc.up
            }
        };
        be.matmul(
            dst,
            down_w,
            activated,
            MatMulCfg::new(s.n_embd as u32, n_ff as u32, nt).with_bias(down_b),
        )
    }

    fn rope_cfg(&self, layer: usize, n_tokens: usize) -> RopeCfg {
        let s = &self.spec;
        let r = &s.rope;
        // Gemma-3 gives its sliding-window layers a different rope base from the global
        // ones, which is the only per-layer rope variation in circulation.
        let base = match (r.local_base, s.layer_uses_swa(layer)) {
            (Some(local), true) => local,
            _ => r.freq_base,
        };
        RopeCfg {
            kind: r.kind,
            n_heads: s.n_heads as u32,
            n_kv_heads: s.n_kv_heads as u32,
            head_dim: s.head_dim as u32,
            n_rot: r.n_rot as u32,
            n_tokens: n_tokens as u32,
            freq_base: base,
            freq_scale: r.freq_scale,
            ext_factor: r.ext_factor,
            attn_factor: r.attn_factor,
            beta_fast: r.beta_fast,
            beta_slow: r.beta_slow,
            orig_ctx: r.orig_ctx as u32,
            sections: r.sections,
            freq_factors: self.weights.rope_freqs,
        }
    }
}

//! Architecture description, derived from the file rather than from a table of names.
//!
//! The engine has to run models that did not exist when it was built. Almost everything a
//! decoder block varies on is visible in the GGUF itself: which tensors a block contains,
//! and which hyperparameter keys are set. So [`ArchSpec::infer`] reads the shape of the
//! file first and consults [`rope_kind_for`] only for the one thing the file genuinely
//! cannot express.

use gguf_core::{Activation, GateFunc, NormKind, RopeKind};
use gguf_format::GgufModel;

/// How a block wires its normalizations and residuals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockStyle {
    /// Norm before each sublayer, residual around it. Llama, Qwen, Gemma, Phi-3.
    PreNorm,
    /// One norm feeding attention and FFN in parallel, both added to the same residual.
    /// Falcon, GPT-NeoX, Phi-2.
    ParallelResidual,
    /// Sublayer first, norm applied to its output before the residual add. OLMo-2.
    PostNorm,
}

#[derive(Debug, Clone)]
pub struct MoeSpec {
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff_exp: usize,
    pub gate_func: GateFunc,
    pub norm_topk: bool,
    pub scale: f32,
    /// Always-on expert evaluated alongside the routed ones (Qwen2-MoE, DeepSeek).
    pub n_ff_shexp: usize,
}

#[derive(Debug, Clone)]
pub struct RopeSpec {
    pub kind: RopeKind,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub n_rot: usize,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub orig_ctx: usize,
    pub sections: [u32; 4],
    /// Layers alternating between local and global rope, keyed by layer index.
    pub local_base: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct ArchSpec {
    pub arch: String,
    pub name: String,

    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_ctx_train: usize,

    pub style: BlockStyle,
    pub norm_kind: NormKind,
    pub norm_eps: f32,

    pub rope: RopeSpec,
    pub act: Activation,
    /// A gated FFN has separate gate and up projections; an ungated one only has up.
    pub gated_ffn: bool,

    pub qkv_bias: bool,
    pub attn_out_bias: bool,
    pub ffn_bias: bool,
    pub qk_norm: bool,
    /// QK-norm weights span a whole head row rather than one head's dims.
    pub post_attn_norm: bool,
    pub post_ffn_norm: bool,
    pub attn_sinks: bool,

    pub embd_scale: f32,
    pub residual_scale: f32,
    pub logit_scale: f32,
    pub attn_scale: f32,
    pub attn_softcap: f32,
    pub final_softcap: f32,

    /// Sliding-window width, and how often a layer uses it. `every == 0` means never;
    /// `every == 1` means every layer; Gemma-2 uses 2 (alternating), Gemma-3 uses 6.
    pub swa_window: u32,
    pub swa_every: u32,

    pub moe: Option<MoeSpec>,
    pub tied_embeddings: bool,
    pub has_rope_freqs: bool,
}

/// Architectures whose Q and K weights were permuted at conversion time for the
/// adjacent-pair rotation. Everything else uses the split-half NeoX rotation, which is the
/// safe default for an architecture this list has never heard of.
const ROPE_NORM_ARCHS: &[&str] = &[
    "llama", "llama4", "deci", "baichuan", "starcoder", "plamo", "orion", "internlm2",
    "minicpm", "xverse", "command-r", "cohere2", "olmo", "arctic", "deepseek", "deepseek2",
    "chatglm", "granite", "granitemoe", "chameleon", "bailingmoe", "arcee", "mistral",
    "ernie4_5", "refact", "bitnet",
];

pub fn rope_kind_for(arch: &str, has_sections: bool) -> RopeKind {
    if has_sections {
        return RopeKind::MRope;
    }
    if ROPE_NORM_ARCHS.contains(&arch) {
        RopeKind::Norm
    } else {
        RopeKind::Neox
    }
}

impl ArchSpec {
    pub fn infer(g: &GgufModel) -> gguf_core::Result<Self> {
        let m = &g.meta;
        let arch = m.arch.clone();

        let n_embd = m.u32_req("embedding_length")? as usize;
        let n_layers = m.u32_req("block_count")? as usize;
        let n_heads = m.u32_req("attention.head_count")? as usize;
        let n_kv_heads = m.u32_or("attention.head_count_kv", n_heads as u32) as usize;
        let head_dim = m
            .u32("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or_else(|| n_embd / n_heads.max(1));
        let n_ff = m.u32_or("feed_forward_length", 0) as usize;

        let n_vocab = m
            .raw("general.vocab_size")
            .and_then(gguf_format::Value::as_u32)
            .or_else(|| {
                m.raw("tokenizer.ggml.tokens")
                    .and_then(gguf_format::Value::as_array)
                    .map(|a| a.len() as u32)
            })
            .or_else(|| g.info("token_embd.weight").map(|t| t.rows() as u32))
            .unwrap_or(32000) as usize;

        // Block wiring, read off which norms block 0 actually carries.
        let has = |suffix: &str| g.contains(&format!("blk.0.{suffix}"));
        let has_any = |sfx: &[&str]| sfx.iter().any(|s| has(s));

        let has_attn_norm = has_any(&["attn_norm.weight", "attn_norm.bias"]);
        let has_ffn_norm = has("ffn_norm.weight");
        let post_attn_norm =
            has_any(&["post_attention_norm.weight", "attn_post_norm.weight"]);
        let post_ffn_norm = has_any(&["post_ffw_norm.weight", "ffn_post_norm.weight"]);

        let style = if !has_attn_norm && post_attn_norm {
            BlockStyle::PostNorm
        } else if !has_ffn_norm && has_attn_norm {
            BlockStyle::ParallelResidual
        } else {
            BlockStyle::PreNorm
        };

        // A LayerNorm block carries bias tensors next to its norm weights; RMSNorm never
        // does. That distinction survives in the file even when the arch is unknown.
        let norm_kind = if has("attn_norm.bias") || g.contains("output_norm.bias") {
            NormKind::Layer
        } else {
            NormKind::Rms
        };
        let norm_eps = m
            .f32("attention.layer_norm_rms_epsilon")
            .or_else(|| m.f32("attention.layer_norm_epsilon"))
            .unwrap_or(1e-5);

        let is_gemma = arch.starts_with("gemma");
        let gated_ffn = has("ffn_gate.weight") || has("ffn_gate_exps.weight");

        let act = if is_gemma || arch.starts_with("phi") && !has("ffn_gate.weight") {
            Activation::Gelu
        } else if arch == "gptoss" {
            Activation::Swish
        } else if !gated_ffn {
            Activation::Gelu
        } else {
            Activation::Silu
        };

        let sections_raw = m.u32_list("rope.dimension_sections");
        let mut sections = [0u32; 4];
        for (i, v) in sections_raw.iter().take(4).enumerate() {
            sections[i] = *v;
        }
        let has_sections = sections_raw.iter().any(|v| *v > 0);

        let rope_scaling_type = m.string("rope.scaling.type").unwrap_or("").to_string();
        let freq_scale = m
            .f32("rope.scaling.factor")
            .or_else(|| m.f32("rope.scale_linear"))
            .map(|f| if f > 0.0 { 1.0 / f } else { 1.0 })
            .unwrap_or(1.0);
        let is_yarn = rope_scaling_type == "yarn";

        let rope = RopeSpec {
            kind: rope_kind_for(&arch, has_sections),
            freq_base: m.f32_or("rope.freq_base", 10000.0),
            // Llama-3 and LongRoPE scaling arrive as a rope_freqs tensor, so their
            // "factor" must not also be applied as a linear scale.
            freq_scale: if matches!(rope_scaling_type.as_str(), "llama3" | "longrope") {
                1.0
            } else {
                freq_scale
            },
            n_rot: m
                .u32("rope.dimension_count")
                .map(|v| v as usize)
                .unwrap_or(head_dim),
            ext_factor: if is_yarn { 1.0 } else { 0.0 },
            attn_factor: m.f32_or("rope.scaling.attn_factor", 1.0),
            beta_fast: m.f32_or("rope.scaling.beta_fast", 32.0),
            beta_slow: m.f32_or("rope.scaling.beta_slow", 1.0),
            orig_ctx: m
                .u32("rope.scaling.original_context_length")
                .map(|v| v as usize)
                .unwrap_or(0),
            sections,
            // Gemma-3 gives its sliding-window layers a much shorter rope base than its
            // global ones. Converters of that era did not write the key, and llama.cpp
            // carries the same default, so inferring it is what makes those files work.
            local_base: m.f32("rope.freq_base_swa").or_else(|| {
                (arch == "gemma3" || arch == "gemma3n").then_some(10000.0)
            }),
        };

        let moe = {
            let n_expert = m.u32_or("expert_count", 0) as usize;
            if n_expert > 0 && has("ffn_gate_inp.weight") {
                Some(MoeSpec {
                    n_expert,
                    n_expert_used: m.u32_or("expert_used_count", 2) as usize,
                    n_ff_exp: m
                        .u32("expert_feed_forward_length")
                        .map(|v| v as usize)
                        .unwrap_or(n_ff),
                    gate_func: if m.u32_or("expert_gating_func", 1) == 2 {
                        GateFunc::Sigmoid
                    } else {
                        GateFunc::Softmax
                    },
                    // Everything since Mixtral renormalizes the selected weights.
                    norm_topk: m
                        .bool("expert_weights_norm")
                        .unwrap_or(true),
                    scale: m.f32_or("expert_weights_scale", 1.0),
                    n_ff_shexp: m
                        .u32("expert_shared_feed_forward_length")
                        .map(|v| v as usize)
                        .unwrap_or(0),
                })
            } else {
                None
            }
        };

        // Gemma scales embeddings by sqrt(n_embd); it is the one model family that folds a
        // constant this large into the embedding rather than the attention scale.
        let embd_scale = m
            .f32("embedding_scale")
            .unwrap_or(if is_gemma { (n_embd as f32).sqrt() } else { 1.0 });

        let attn_scale = m
            .f32("attention.scale")
            .unwrap_or(1.0 / (head_dim as f32).sqrt());

        let swa_window = m.u32_or("attention.sliding_window", 0);
        let swa_every = if swa_window == 0 {
            0
        } else if arch == "gemma2" {
            2
        } else if arch == "gemma3" || arch == "gemma3n" {
            6
        } else {
            m.u32_or("attention.sliding_window_pattern", 1)
        };

        Ok(Self {
            name: m
                .raw("general.name")
                .and_then(gguf_format::Value::as_str)
                .unwrap_or(&arch)
                .to_string(),
            n_vocab,
            n_embd,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            n_ff,
            n_ctx_train: m.u32_or("context_length", 4096) as usize,
            style,
            norm_kind,
            norm_eps,
            rope,
            act,
            gated_ffn,
            qkv_bias: has_any(&["attn_q.bias", "attn_qkv.bias"]),
            attn_out_bias: has("attn_output.bias"),
            ffn_bias: has_any(&["ffn_up.bias", "ffn_down.bias"]),
            qk_norm: has_any(&["attn_q_norm.weight", "attn_k_norm.weight"]),
            post_attn_norm,
            post_ffn_norm,
            attn_sinks: has_any(&["attn_sinks.weight", "attn_sinks"]),
            embd_scale,
            residual_scale: m.f32_or("residual_scale", 1.0),
            logit_scale: m.f32_or("logit_scale", 1.0),
            attn_scale,
            attn_softcap: m.f32_or("attn_logit_softcapping", 0.0),
            final_softcap: m.f32_or("final_logit_softcapping", 0.0),
            swa_window,
            swa_every,
            moe,
            tied_embeddings: !g.contains("output.weight"),
            has_rope_freqs: g.contains("rope_freqs.weight"),
            arch,
        })
    }

    pub fn n_embd_k(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    pub fn n_embd_q(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Whether layer `i` attends only inside the sliding window.
    ///
    /// Gemma-2 alternates, Gemma-3 makes every sixth layer global; both are expressed as
    /// "layer is local unless it is the last of each group of `swa_every`".
    pub fn layer_uses_swa(&self, i: usize) -> bool {
        if self.swa_window == 0 || self.swa_every == 0 {
            return false;
        }
        if self.swa_every == 1 {
            return true;
        }
        (i + 1) % self.swa_every as usize != 0
    }

    pub fn summary(&self) -> String {
        let mut parts = vec![
            format!("{} | {} layers", self.arch, self.n_layers),
            format!("embd {}", self.n_embd),
            format!("heads {}/{} x {}", self.n_heads, self.n_kv_heads, self.head_dim),
        ];
        if let Some(moe) = &self.moe {
            parts.push(format!(
                "MoE {}/{} experts x {}",
                moe.n_expert_used, moe.n_expert, moe.n_ff_exp
            ));
        } else {
            parts.push(format!("ff {}", self.n_ff));
        }
        parts.push(format!("rope {:?} base {}", self.rope.kind, self.rope.freq_base));
        if self.swa_window > 0 {
            parts.push(format!("swa {} every {}", self.swa_window, self.swa_every));
        }
        if self.qk_norm {
            parts.push("qk-norm".into());
        }
        if self.style != BlockStyle::PreNorm {
            parts.push(format!("{:?}", self.style));
        }
        parts.join(" | ")
    }
}

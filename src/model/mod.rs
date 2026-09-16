pub mod config;
mod loader;
mod forward;

use crate::tensor::dequant::QuantTensor;
use crate::gpu::{GpuTensor, ActBuf};
use config::ModelConfig;

pub struct Weights {
    pub token_embd:  QuantTensor,
    pub output_norm: Vec<f32>,
    pub output:      QuantTensor,
    pub attn_norm:   Vec<Vec<f32>>,
    pub ffn_norm:    Vec<Vec<f32>>,
    pub attn_q:      Vec<QuantTensor>,
    pub attn_k:      Vec<QuantTensor>,
    pub attn_v:      Vec<QuantTensor>,
    pub attn_out:    Vec<QuantTensor>,
    pub ffn_gate:    Vec<Option<QuantTensor>>,
    pub ffn_up:      Vec<Option<QuantTensor>>,
    pub ffn_down:    Vec<Option<QuantTensor>>,
    pub ffn_router:    Vec<Option<QuantTensor>>,
    pub ffn_gate_exps:  Vec<Option<QuantTensor>>,
    pub ffn_up_exps:     Vec<Option<QuantTensor>>,
    pub ffn_down_exps:    Vec<Option<QuantTensor>>,
    pub attn_q_bias: Vec<Option<Vec<f32>>>,
    pub attn_k_bias: Vec<Option<Vec<f32>>>,
    pub attn_v_bias: Vec<Option<Vec<f32>>>,
    pub attn_q_f32:  Vec<Option<Vec<f32>>>,
    pub attn_k_f32:  Vec<Option<Vec<f32>>>,
    pub attn_v_f32:  Vec<Option<Vec<f32>>>,
    pub attn_q_norm: Vec<Option<Vec<f32>>>,
    pub attn_k_norm: Vec<Option<Vec<f32>>>,
    pub attn_post_norm: Vec<Option<Vec<f32>>>,
    pub ffn_post_norm:  Vec<Option<Vec<f32>>>,
}

pub struct GpuWeights {
    pub output:   Option<GpuTensor>,
    pub attn_q:   Vec<Option<GpuTensor>>,
    pub attn_k:   Vec<Option<GpuTensor>>,
    pub attn_v:   Vec<Option<GpuTensor>>,
    pub attn_out: Vec<Option<GpuTensor>>,
    pub ffn_gate: Vec<Option<GpuTensor>>,
    pub ffn_up:   Vec<Option<GpuTensor>>,
    pub ffn_down: Vec<Option<GpuTensor>>,
}

pub struct GpuActs {
    pub x:           ActBuf,
    pub xn:          ActBuf,
    pub q:           ActBuf,
    pub k:           ActBuf,
    pub v:           ActBuf,
    pub attn_out:    ActBuf,
    pub proj:        ActBuf,
    pub gate:        ActBuf,
    pub up:          ActBuf,
    pub ff:          ActBuf,
    pub logits:      ActBuf,
    pub logits_rb:   ActBuf,
    pub k_cache:     Vec<Vec<ActBuf>>,
    pub v_cache:     Vec<Vec<ActBuf>>,
    pub scores:      ActBuf,
    pub ctx_len:     usize,
    pub attn_norms:  Vec<ActBuf>,
    pub ffn_norms:   Vec<ActBuf>,
    pub out_norm:    ActBuf,
    pub q_bias:      Vec<Option<ActBuf>>,
    pub k_bias:      Vec<Option<ActBuf>>,
    pub v_bias:      Vec<Option<ActBuf>>,
    pub q_norm:      Vec<Option<ActBuf>>,
    pub k_norm:      Vec<Option<ActBuf>>,
    pub attn_post_norm: Vec<Option<ActBuf>>,
    pub ffn_post_norm:  Vec<Option<ActBuf>>,
    pub x_rb:             ActBuf,
}

pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}
impl KvCache {
    pub fn new(n_layers: usize, n_ctx: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let sz = n_ctx * n_kv_heads * head_dim;
        Self { k: vec![vec![0f32; sz]; n_layers], v: vec![vec![0f32; sz]; n_layers] }
    }
}

pub struct LlamaModel {
    pub config:   ModelConfig,
    pub weights:  Weights,
    pub gpu_w:    Option<GpuWeights>,
    pub gpu_acts: Option<GpuActs>,
}

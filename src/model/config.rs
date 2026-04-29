#![allow(dead_code)]
use anyhow::Context;
use crate::gguf::types::GgufFile;

#[derive(Debug)]
pub struct ModelConfig {
    pub arch:           String,
    pub n_vocab:        usize,
    pub n_embd:         usize,
    pub n_layers:       usize,
    pub n_heads:        usize,
    pub n_kv_heads:     usize,
    pub n_ff:           usize,
    pub n_ctx:          usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps:   f32,
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> anyhow::Result<Self> {
        let arch = gguf.metadata.get("general.architecture")
            .and_then(|v| v.as_str()).unwrap_or("llama").to_string();

        let get_u32 = |key: &str| -> anyhow::Result<u32> {
            let fk = format!("{}.{}", arch, key);
            gguf.metadata.get(&fk).and_then(|v| v.as_u32())
                .with_context(|| format!("Missing: {}", fk))
        };
        let get_f32 = |key: &str, def: f32| -> f32 {
            gguf.metadata.get(&format!("{}.{}", arch, key))
                .and_then(|v| v.as_f32()).unwrap_or(def)
        };

        let n_vocab = gguf.metadata.get("general.vocab_size")
            .and_then(|v| v.as_u32())
            .or_else(|| gguf.metadata.get("tokenizer.ggml.tokens").and_then(|v|v.as_arr()).map(|a|a.len() as u32))
            .unwrap_or(32000) as usize;

        let n_heads    = get_u32("attention.head_count")? as usize;
        let n_kv_heads = get_u32("attention.head_count_kv").unwrap_or(n_heads as u32) as usize;

        Ok(Self {
            n_vocab,
            n_embd:         get_u32("embedding_length")? as usize,
            n_layers:       get_u32("block_count")? as usize,
            n_ff:           get_u32("feed_forward_length")? as usize,
            n_ctx:          get_u32("context_length").unwrap_or(32768) as usize,
            rope_freq_base: get_f32("rope.freq_base", 10000.0),  // Qwen2.5=1000000 stored as F64
            rms_norm_eps:   get_f32("attention.layer_norm_rms_epsilon", 1e-6),
            arch, n_heads, n_kv_heads,
        })
    }

    pub fn head_dim(&self) -> usize { self.n_embd / self.n_heads }
}
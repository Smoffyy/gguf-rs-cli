use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use crate::gguf::{reader, types::GgufFile};
use crate::tensor::storage::TensorStorage;
use crate::model::config::ModelConfig;
use crate::math::{ops, rope};

pub struct Weights {
    pub token_embd:  Vec<f32>,
    pub output_norm: Vec<f32>,
    pub output:      Vec<f32>,
    // Per-layer vectors
    pub attn_norm:   Vec<Vec<f32>>,
    pub ffn_norm:    Vec<Vec<f32>>,
    pub attn_q:      Vec<Vec<f32>>,
    pub attn_k:      Vec<Vec<f32>>,
    pub attn_v:      Vec<Vec<f32>>,
    pub attn_out:    Vec<Vec<f32>>,
    pub ffn_gate:    Vec<Vec<f32>>,
    pub ffn_up:      Vec<Vec<f32>>,
    pub ffn_down:    Vec<Vec<f32>>,
}

pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(n_layers: usize, n_ctx: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let sz = n_ctx * n_kv_heads * head_dim;
        Self {
            k: vec![vec![0f32; sz]; n_layers],
            v: vec![vec![0f32; sz]; n_layers],
        }
    }
}

pub struct LlamaModel {
    pub config:  ModelConfig,
    pub weights: Weights,
}

impl LlamaModel {
    pub fn load(path: &Path) -> Result<(Self, GgufFile)> {
        eprintln!("Parsing GGUF...");
        let f    = std::fs::File::open(path)?;
        let gguf = reader::parse(std::io::BufReader::new(f))?;
        let cfg  = ModelConfig::from_gguf(&gguf)?;
        let stor = TensorStorage::new(path, gguf.data_offset)?;

        eprintln!("Config: {} layers | embd {} | heads {} | kv_heads {} | ff {}",
            cfg.n_layers, cfg.n_embd, cfg.n_heads, cfg.n_kv_heads, cfg.n_ff);

        let tmap: HashMap<&str, _> = gguf.tensors.iter()
            .map(|t| (t.name.as_str(), t)).collect();

        let get = |name: &str| -> Result<Vec<f32>> {
            let info = tmap.get(name)
                .ok_or_else(|| anyhow::anyhow!("Missing tensor: {}", name))?;
            stor.load_f32(info)
        };

        eprintln!("Loading embeddings and norms...");
        let token_embd  = get("token_embd.weight")?;
        let output_norm = get("output_norm.weight")?;
        // Some models tie input/output embeddings
        let output = get("output.weight").or_else(|_| get("token_embd.weight"))?;

        let mut attn_norm = Vec::new(); let mut ffn_norm  = Vec::new();
        let mut attn_q    = Vec::new(); let mut attn_k    = Vec::new();
        let mut attn_v    = Vec::new(); let mut attn_out  = Vec::new();
        let mut ffn_gate  = Vec::new(); let mut ffn_up    = Vec::new();
        let mut ffn_down  = Vec::new();

        for i in 0..cfg.n_layers {
            eprintln!("  Layer {}/{}", i + 1, cfg.n_layers);
            attn_norm.push(get(&format!("blk.{}.attn_norm.weight", i))?);
            ffn_norm .push(get(&format!("blk.{}.ffn_norm.weight",  i))?);
            attn_q   .push(get(&format!("blk.{}.attn_q.weight",    i))?);
            attn_k   .push(get(&format!("blk.{}.attn_k.weight",    i))?);
            attn_v   .push(get(&format!("blk.{}.attn_v.weight",    i))?);
            attn_out .push(get(&format!("blk.{}.attn_output.weight", i))?);
            ffn_gate .push(get(&format!("blk.{}.ffn_gate.weight",  i))?);
            ffn_up   .push(get(&format!("blk.{}.ffn_up.weight",    i))?);
            ffn_down .push(get(&format!("blk.{}.ffn_down.weight",  i))?);
        }

        eprintln!("Model loaded.");
        let model = Self {
            config: cfg,
            weights: Weights { token_embd, output_norm, output,
                attn_norm, ffn_norm, attn_q, attn_k, attn_v, attn_out,
                ffn_gate, ffn_up, ffn_down },
        };
        Ok((model, gguf))
    }

    // Single-token forward pass. Returns logits over vocab.
    pub fn forward(&self, token: usize, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let c   = &self.config;
        let w   = &self.weights;
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;   // kv dimension

        // --- Embedding lookup ---
        let mut x: Vec<f32> = w.token_embd[token * c.n_embd..(token+1) * c.n_embd].to_vec();

        // Reusable scratch buffers
        let mut xn       = vec![0f32; c.n_embd];
        let mut q        = vec![0f32; c.n_heads * hd];
        let mut k        = vec![0f32; kvd];
        let mut v        = vec![0f32; kvd];
        let mut scores   = vec![0f32; c.n_heads * (pos + 1)];
        let mut attn_out = vec![0f32; c.n_embd];
        let mut gate     = vec![0f32; c.n_ff];
        let mut up       = vec![0f32; c.n_ff];
        let mut ff_out   = vec![0f32; c.n_embd];
        let mut proj     = vec![0f32; c.n_embd];

        for l in 0..c.n_layers {
            // --- Attention block ---
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.attn_norm[l], c.rms_norm_eps);

            ops::matmul(&mut q, &w.attn_q[l], &xn, c.n_heads * hd, c.n_embd);
            ops::matmul(&mut k, &w.attn_k[l], &xn, kvd,            c.n_embd);
            ops::matmul(&mut v, &w.attn_v[l], &xn, kvd,            c.n_embd);

            rope::apply_rope(&mut q, &mut k, pos, hd, c.rope_freq_base, c.n_heads, c.n_kv_heads);

            // Store K, V into cache at this position
            let cb = pos * kvd;
            cache.k[l][cb..cb+kvd].copy_from_slice(&k);
            cache.v[l][cb..cb+kvd].copy_from_slice(&v);

            // Multi-head attention: for each head compute Q @ K^T -> softmax -> @ V
            let kv_ratio = c.n_heads / c.n_kv_heads; // for GQA / MQA
            attn_out.fill(0.0);

            for h in 0..c.n_heads {
                let kv_h = h / kv_ratio;
                let q_h  = &q[h * hd..(h+1) * hd];

                // Attention scores: dot(Q_h, K_t) / sqrt(head_dim)
                let sc = &mut scores[h * (pos+1)..(h+1) * (pos+1)];
                let scale = (hd as f32).sqrt();
                for p in 0..=pos {
                    let ko = p * kvd + kv_h * hd;
                    sc[p] = q_h.iter().zip(cache.k[l][ko..ko+hd].iter())
                        .map(|(a,b)| a*b).sum::<f32>() / scale;
                }
                ops::softmax(sc);

                // Weighted sum of V
                let out_h = &mut attn_out[h * hd..(h+1) * hd];
                out_h.fill(0.0);
                for p in 0..=pos {
                    let vo    = p * kvd + kv_h * hd;
                    let score = sc[p];
                    for (o, vi) in out_h.iter_mut().zip(cache.v[l][vo..vo+hd].iter()) {
                        *o += score * vi;
                    }
                }
            }

            // Project attention output back to embedding dim, add residual
            ops::matmul(&mut proj, &w.attn_out[l], &attn_out, c.n_embd, c.n_embd);
            ops::add_into(&mut x, &proj);

            // --- FFN block (SwiGLU) ---
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.ffn_norm[l], c.rms_norm_eps);

            ops::matmul(&mut gate, &w.ffn_gate[l], &xn, c.n_ff, c.n_embd);
            ops::matmul(&mut up,   &w.ffn_up[l],   &xn, c.n_ff, c.n_embd);

            // SiLU(gate) * up
            for i in 0..c.n_ff { gate[i] = ops::silu(gate[i]) * up[i]; }

            ops::matmul(&mut ff_out, &w.ffn_down[l], &gate, c.n_embd, c.n_ff);
            ops::add_into(&mut x, &ff_out);
        }

        // --- Final norm + project to vocab ---
        ops::rmsnorm(&mut x, &w.output_norm, c.rms_norm_eps);
        let mut logits = vec![0f32; c.n_vocab];
        ops::matmul(&mut logits, &w.output, &x, c.n_vocab, c.n_embd);
        logits
    }
}

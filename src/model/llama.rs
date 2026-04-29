use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use crate::gguf::{reader, types::GgufFile};
use crate::tensor::{dequant::QuantTensor, storage::TensorStorage};
use crate::model::config::ModelConfig;
use crate::math::{ops, rope};
use crate::gpu::GpuCtx;

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
    pub ffn_gate:    Vec<QuantTensor>,
    pub ffn_up:      Vec<QuantTensor>,
    pub ffn_down:    Vec<QuantTensor>,
    // Optional QKV bias (Qwen2/Qwen2.5 uses these)
    pub attn_q_bias: Vec<Option<Vec<f32>>>,
    pub attn_k_bias: Vec<Option<Vec<f32>>>,
    pub attn_v_bias: Vec<Option<Vec<f32>>>,
}

pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}
impl KvCache {
    pub fn new(n_layers: usize, n_ctx: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let sz = n_ctx * n_kv_heads * head_dim;
        Self { k: vec![vec![0f32;sz];n_layers], v: vec![vec![0f32;sz];n_layers] }
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

        eprintln!("Config: {} layers | embd {} | heads {}/{} | ff {} | rope_base {}",
            cfg.n_layers, cfg.n_embd, cfg.n_heads, cfg.n_kv_heads, cfg.n_ff, cfg.rope_freq_base);

        let tmap: HashMap<&str,_> = gguf.tensors.iter().map(|t|(t.name.as_str(),t)).collect();

        // Load a weight matrix as QuantTensor (keeps compressed in RAM)
        let get_q = |name: &str| -> Result<QuantTensor> {
            let info = tmap.get(name).ok_or_else(||anyhow::anyhow!("Missing tensor: {}", name))?;
            Ok(QuantTensor::load(stor.get_bytes(info).to_vec(), info.typ, &info.dims))
        };
        // Load a small weight vector as f32 (norms are always F32)
        let get_f = |name: &str| -> Result<Vec<f32>> {
            let info = tmap.get(name).ok_or_else(||anyhow::anyhow!("Missing tensor: {}", name))?;
            crate::tensor::dequant::dequantize(info.typ, stor.get_bytes(info), info.n_elements())
        };
        // Load optional bias
        let get_bias = |name: &str| -> Option<Vec<f32>> {
            tmap.get(name).and_then(|info|{
                crate::tensor::dequant::dequantize(info.typ, stor.get_bytes(info), info.n_elements()).ok()
            })
        };

        eprintln!("Loading weights (stored compressed, ~{}MB RAM)...",
            gguf.tensors.iter().map(|t|t.byte_size()).sum::<usize>()/1_000_000);

        let token_embd  = get_q("token_embd.weight")?;
        let output_norm = get_f("output_norm.weight")?;
        let output      = get_q("output.weight").or_else(|_|get_q("token_embd.weight"))?;

        let mut attn_norm=vec![]; let mut ffn_norm=vec![];
        let mut attn_q=vec![];    let mut attn_k=vec![];
        let mut attn_v=vec![];    let mut attn_out=vec![];
        let mut ffn_gate=vec![];  let mut ffn_up=vec![];
        let mut ffn_down=vec![];
        let mut attn_q_bias=vec![]; let mut attn_k_bias=vec![]; let mut attn_v_bias=vec![];

        for i in 0..cfg.n_layers {
            if i==0||i==cfg.n_layers-1 { eprintln!("  Layer {}/{}", i+1, cfg.n_layers); }
            attn_norm.push(get_f(&format!("blk.{}.attn_norm.weight",i))?);
            ffn_norm .push(get_f(&format!("blk.{}.ffn_norm.weight",i))?);
            attn_q   .push(get_q(&format!("blk.{}.attn_q.weight",i))?);
            attn_k   .push(get_q(&format!("blk.{}.attn_k.weight",i))?);
            attn_v   .push(get_q(&format!("blk.{}.attn_v.weight",i))?);
            attn_out .push(get_q(&format!("blk.{}.attn_output.weight",i))?);
            ffn_gate .push(get_q(&format!("blk.{}.ffn_gate.weight",i))?);
            ffn_up   .push(get_q(&format!("blk.{}.ffn_up.weight",i))?);
            ffn_down .push(get_q(&format!("blk.{}.ffn_down.weight",i))?);
            // Qwen2/2.5 uses QKV bias; other models silently skip
            attn_q_bias.push(get_bias(&format!("blk.{}.attn_q.bias",i)));
            attn_k_bias.push(get_bias(&format!("blk.{}.attn_k.bias",i)));
            attn_v_bias.push(get_bias(&format!("blk.{}.attn_v.bias",i)));
        }
        eprintln!("Weights loaded.");

        Ok((Self {
            config: cfg,
            weights: Weights { token_embd, output_norm, output,
                attn_norm, ffn_norm, attn_q, attn_k, attn_v, attn_out,
                ffn_gate, ffn_up, ffn_down,
                attn_q_bias, attn_k_bias, attn_v_bias },
        }, gguf))
    }

    pub fn forward(&self, token: usize, pos: usize, cache: &mut KvCache, gpu: Option<&GpuCtx>) -> Vec<f32> {
        let c   = &self.config;
        let w   = &self.weights;
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;

        // Embedding lookup
        let mut x = w.token_embd.get_row(token);

        let mut xn     = vec![0f32; c.n_embd];
        let mut q      = vec![0f32; c.n_heads*hd];
        let mut k      = vec![0f32; kvd];
        let mut v      = vec![0f32; kvd];
        let mut scores = vec![0f32; c.n_heads*(pos+1)];
        let mut attn   = vec![0f32; c.n_embd];
        let mut proj   = vec![0f32; c.n_embd];
        let mut gate   = vec![0f32; c.n_ff];
        let mut up     = vec![0f32; c.n_ff];
        let mut ff     = vec![0f32; c.n_embd];

        let matvec = |wt: &QuantTensor, out: &mut Vec<f32>, inp: &[f32]| {
            match gpu {
                Some(g) => { let r=g.matvec(wt,inp); out.copy_from_slice(&r); }
                None    => wt.matvec(out, inp),
            }
        };

        for l in 0..c.n_layers {
            // Attention pre-norm
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.attn_norm[l], c.rms_norm_eps);

            matvec(&w.attn_q[l], &mut q, &xn);
            matvec(&w.attn_k[l], &mut k, &xn);
            matvec(&w.attn_v[l], &mut v, &xn);

            // Apply QKV bias (Qwen2.5 requires this)
            if let Some(ref b) = w.attn_q_bias[l] { ops::add_into(&mut q, b); }
            if let Some(ref b) = w.attn_k_bias[l] { ops::add_into(&mut k, b); }
            if let Some(ref b) = w.attn_v_bias[l] { ops::add_into(&mut v, b); }

            rope::apply_rope(&mut q, &mut k, pos, hd, c.rope_freq_base, c.n_heads, c.n_kv_heads);

            let cb = pos*kvd;
            cache.k[l][cb..cb+kvd].copy_from_slice(&k);
            cache.v[l][cb..cb+kvd].copy_from_slice(&v);

            let kv_ratio = c.n_heads/c.n_kv_heads;
            attn.fill(0.0);

            for h in 0..c.n_heads {
                let kv_h = h/kv_ratio;
                let qh   = &q[h*hd..(h+1)*hd];
                let sc   = &mut scores[h*(pos+1)..(h+1)*(pos+1)];
                let scale = (hd as f32).sqrt();

                for p in 0..=pos {
                    let ko = p*kvd+kv_h*hd;
                    sc[p] = qh.iter().zip(cache.k[l][ko..ko+hd].iter()).map(|(a,b)|a*b).sum::<f32>()/scale;
                }
                ops::softmax(sc);

                let ah = &mut attn[h*hd..(h+1)*hd];
                ah.fill(0.0);
                for p in 0..=pos {
                    let vo = p*kvd+kv_h*hd;
                    let sc_p = sc[p];
                    for (o,vi) in ah.iter_mut().zip(cache.v[l][vo..vo+hd].iter()) { *o+=sc_p*vi; }
                }
            }

            matvec(&w.attn_out[l], &mut proj, &attn);
            ops::add_into(&mut x, &proj);

            // FFN pre-norm + SwiGLU
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.ffn_norm[l], c.rms_norm_eps);

            matvec(&w.ffn_gate[l], &mut gate, &xn);
            matvec(&w.ffn_up[l],   &mut up,   &xn);
            for i in 0..c.n_ff { gate[i]=ops::silu(gate[i])*up[i]; }
            matvec(&w.ffn_down[l], &mut ff, &gate);
            ops::add_into(&mut x, &ff);
        }

        ops::rmsnorm(&mut x, &w.output_norm, c.rms_norm_eps);
        let mut logits = vec![0f32; c.n_vocab];
        matvec(&w.output, &mut logits, &x);
        logits
    }
}
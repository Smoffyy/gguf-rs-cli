use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use crate::gguf::{reader, types::GgufFile};
use crate::tensor::{dequant::QuantTensor, storage::TensorStorage};
use crate::model::config::ModelConfig;
use crate::math::{ops, rope};
use crate::gpu::{GpuCtx, GpuTensor};

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
    // Optional QKV bias — required by Qwen2/2.5, absent in LLaMA etc.
    pub attn_q_bias: Vec<Option<Vec<f32>>>,
    pub attn_k_bias: Vec<Option<Vec<f32>>>,
    pub attn_v_bias: Vec<Option<Vec<f32>>>,
}

/// Pre-uploaded GPU tensors; None means this tensor stays on CPU.
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
    pub config:  ModelConfig,
    pub weights: Weights,
    /// Populated when --gpu is passed; None in CPU-only mode
    pub gpu_w:   Option<GpuWeights>,
}

impl LlamaModel {
    pub fn load(path: &Path, gpu: Option<&GpuCtx>) -> Result<(Self, GgufFile)> {
        eprintln!("Parsing GGUF...");
        let f    = std::fs::File::open(path)?;
        let gguf = reader::parse(std::io::BufReader::new(f))?;
        let cfg  = ModelConfig::from_gguf(&gguf)?;
        let stor = TensorStorage::new(path, gguf.data_offset)?;

        eprintln!("Config: {} layers | embd {} | heads {}/{} | ff {} | rope_base {}",
            cfg.n_layers, cfg.n_embd, cfg.n_heads, cfg.n_kv_heads,
            cfg.n_ff, cfg.rope_freq_base);

        let tmap: HashMap<&str, _> = gguf.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

        // Helper: load a weight matrix as a zero-copy QuantTensor (no data copied)
        let get_q = |name: &str| -> Result<QuantTensor> {
            let info = tmap.get(name)
                .ok_or_else(|| anyhow::anyhow!("Missing tensor: {}", name))?;
            Ok(QuantTensor::new(
                stor.mmap.clone(),
                stor.tensor_offset(info),
                info.byte_size(),
                info.typ,
                &info.dims,
            ))
        };
        // Helper: load a small norm/bias vector fully into f32
        let get_f = |name: &str| -> Result<Vec<f32>> {
            let info = tmap.get(name)
                .ok_or_else(|| anyhow::anyhow!("Missing tensor: {}", name))?;
            crate::tensor::dequant::dequantize(
                info.typ, &stor.mmap[stor.tensor_offset(info)..stor.tensor_offset(info)+info.byte_size()], info.n_elements())
        };
        // Helper: silently skip missing tensors (e.g. QKV bias absent on non-Qwen models)
        let get_bias = |name: &str| -> Option<Vec<f32>> {
            tmap.get(name).and_then(|info| {
                let start = stor.tensor_offset(info);
                crate::tensor::dequant::dequantize(
                    info.typ, &stor.mmap[start..start+info.byte_size()], info.n_elements()).ok()
            })
        };

        let compressed_mb: usize = gguf.tensors.iter().map(|t| t.byte_size()).sum::<usize>() / 1_000_000;
        eprintln!("Loading weights (zero-copy mmap, ~{} MB on disk)...", compressed_mb);

        let token_embd  = get_q("token_embd.weight")?;
        let output_norm = get_f("output_norm.weight")?;
        // Some models tie the output projection to the embedding table
        let output      = get_q("output.weight").or_else(|_| get_q("token_embd.weight"))?;

        let (mut attn_norm, mut ffn_norm)   = (vec![], vec![]);
        let (mut attn_q, mut attn_k, mut attn_v, mut attn_out) = (vec![], vec![], vec![], vec![]);
        let (mut ffn_gate, mut ffn_up, mut ffn_down)           = (vec![], vec![], vec![]);
        let (mut attn_q_bias, mut attn_k_bias, mut attn_v_bias) = (vec![], vec![], vec![]);

        for i in 0..cfg.n_layers {
            attn_norm.push(get_f(&format!("blk.{}.attn_norm.weight", i))?);
            ffn_norm .push(get_f(&format!("blk.{}.ffn_norm.weight",  i))?);
            attn_q   .push(get_q(&format!("blk.{}.attn_q.weight",    i))?);
            attn_k   .push(get_q(&format!("blk.{}.attn_k.weight",    i))?);
            attn_v   .push(get_q(&format!("blk.{}.attn_v.weight",    i))?);
            attn_out .push(get_q(&format!("blk.{}.attn_output.weight", i))?);
            ffn_gate .push(get_q(&format!("blk.{}.ffn_gate.weight",  i))?);
            ffn_up   .push(get_q(&format!("blk.{}.ffn_up.weight",    i))?);
            ffn_down .push(get_q(&format!("blk.{}.ffn_down.weight",  i))?);
            // QKV bias: present in Qwen2/2.5, silently absent elsewhere
            attn_q_bias.push(get_bias(&format!("blk.{}.attn_q.bias", i)));
            attn_k_bias.push(get_bias(&format!("blk.{}.attn_k.bias", i)));
            attn_v_bias.push(get_bias(&format!("blk.{}.attn_v.bias", i)));
        }
        eprintln!("Weights ready (OS pages in on demand).");

        let weights = Weights {
            token_embd, output_norm, output,
            attn_norm, ffn_norm, attn_q, attn_k, attn_v, attn_out,
            ffn_gate, ffn_up, ffn_down,
            attn_q_bias, attn_k_bias, attn_v_bias,
        };

        // Upload only Q4_0 tensors to GPU; K-quant layers stay on CPU
        let gpu_w = gpu.map(|g| {
            eprintln!("Uploading Q4_0 tensors to GPU (K-quant layers use CPU rayon)...");
            let up  = |wt: &QuantTensor| g.upload(wt);
            let n_gpu = |o: &Option<_>| if o.is_some() { 1usize } else { 0 };

            let output   = up(&weights.output);
            let attn_q:   Vec<_> = weights.attn_q.iter().map(up).collect();
            let attn_k:   Vec<_> = weights.attn_k.iter().map(up).collect();
            let attn_v:   Vec<_> = weights.attn_v.iter().map(up).collect();
            let attn_out: Vec<_> = weights.attn_out.iter().map(up).collect();
            let ffn_gate: Vec<_> = weights.ffn_gate.iter().map(up).collect();
            let ffn_up:   Vec<_> = weights.ffn_up.iter().map(up).collect();
            let ffn_down: Vec<_> = weights.ffn_down.iter().map(up).collect();

            let on_gpu = attn_q.iter().map(n_gpu).sum::<usize>()
                + attn_k.iter().map(n_gpu).sum::<usize>()
                + attn_v.iter().map(n_gpu).sum::<usize>()
                + attn_out.iter().map(n_gpu).sum::<usize>()
                + ffn_gate.iter().map(n_gpu).sum::<usize>()
                + ffn_up.iter().map(n_gpu).sum::<usize>()
                + ffn_down.iter().map(n_gpu).sum::<usize>()
                + n_gpu(&output);
            let total = cfg.n_layers * 7 + 1;
            eprintln!("{}/{} tensors on GPU ({} on CPU rayon)", on_gpu, total, total - on_gpu);

            GpuWeights { output, attn_q, attn_k, attn_v, attn_out, ffn_gate, ffn_up, ffn_down }
        });

        Ok((Self { config: cfg, weights, gpu_w }, gguf))
    }

    /// Single-token forward pass. Returns logit vector over the full vocabulary.
    pub fn forward(&self, token: usize, pos: usize, cache: &mut KvCache,
                   gpu: Option<&GpuCtx>) -> Vec<f32> {
        let c   = &self.config;
        let w   = &self.weights;
        let gw  = self.gpu_w.as_ref();
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;

        // Embedding lookup (always CPU — table is not square, no matmul)
        let mut x = w.token_embd.get_row(token);

        let mut xn     = vec![0f32; c.n_embd];
        let mut q      = vec![0f32; c.n_heads * hd];
        let mut k      = vec![0f32; kvd];
        let mut v      = vec![0f32; kvd];
        let mut scores = vec![0f32; c.n_heads * (pos + 1)];
        let mut attn   = vec![0f32; c.n_embd];
        let mut proj   = vec![0f32; c.n_embd];
        let mut gate   = vec![0f32; c.n_ff];
        let mut up_buf = vec![0f32; c.n_ff];
        let mut ff     = vec![0f32; c.n_embd];

        // Dispatch to GPU if a pre-uploaded tensor exists, otherwise CPU rayon
        let mv = |cpu: &QuantTensor, gt: Option<&GpuTensor>, out: &mut Vec<f32>, inp: &[f32]| {
            match (gpu, gt) {
                (Some(g), Some(t)) => { let r = g.dispatch(t, inp); out.copy_from_slice(&r); }
                _                  => cpu.matvec(out, inp),
            }
        };

        for l in 0..c.n_layers {
            // ── Attention ────────────────────────────────────────────────────
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.attn_norm[l], c.rms_norm_eps);

            mv(&w.attn_q[l],   gw.and_then(|g| g.attn_q[l].as_ref()),   &mut q, &xn);
            mv(&w.attn_k[l],   gw.and_then(|g| g.attn_k[l].as_ref()),   &mut k, &xn);
            mv(&w.attn_v[l],   gw.and_then(|g| g.attn_v[l].as_ref()),   &mut v, &xn);

            // Qwen2/2.5 requires QKV bias; other models have None here
            if let Some(ref b) = w.attn_q_bias[l] { ops::add_into(&mut q, b); }
            if let Some(ref b) = w.attn_k_bias[l] { ops::add_into(&mut k, b); }
            if let Some(ref b) = w.attn_v_bias[l] { ops::add_into(&mut v, b); }

            // Rotary positional embeddings
            rope::apply_rope(&mut q, &mut k, pos, hd, c.rope_freq_base, c.n_heads, c.n_kv_heads);

            // Write K/V into the KV cache at this position
            let cb = pos * kvd;
            cache.k[l][cb..cb+kvd].copy_from_slice(&k);
            cache.v[l][cb..cb+kvd].copy_from_slice(&v);

            // Multi-head attention with GQA support
            let kv_ratio = c.n_heads / c.n_kv_heads;
            attn.fill(0.0);
            for h in 0..c.n_heads {
                let kv_h  = h / kv_ratio;
                let qh    = &q[h*hd..(h+1)*hd];
                let sc    = &mut scores[h*(pos+1)..(h+1)*(pos+1)];
                let scale = (hd as f32).sqrt();
                for p in 0..=pos {
                    let ko = p*kvd + kv_h*hd;
                    sc[p]  = qh.iter().zip(cache.k[l][ko..ko+hd].iter())
                                .map(|(a,b)| a*b).sum::<f32>() / scale;
                }
                ops::softmax(sc);
                let ah = &mut attn[h*hd..(h+1)*hd];
                ah.fill(0.0);
                for p in 0..=pos {
                    let vo  = p*kvd + kv_h*hd;
                    let sp  = sc[p];
                    for (o, vi) in ah.iter_mut().zip(cache.v[l][vo..vo+hd].iter()) {
                        *o += sp * vi;
                    }
                }
            }
            mv(&w.attn_out[l], gw.and_then(|g| g.attn_out[l].as_ref()), &mut proj, &attn);
            ops::add_into(&mut x, &proj);

            // ── Feed-forward (SwiGLU) ────────────────────────────────────────
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.ffn_norm[l], c.rms_norm_eps);

            mv(&w.ffn_gate[l], gw.and_then(|g| g.ffn_gate[l].as_ref()), &mut gate,   &xn);
            mv(&w.ffn_up[l],   gw.and_then(|g| g.ffn_up[l].as_ref()),   &mut up_buf, &xn);
            for i in 0..c.n_ff { gate[i] = ops::silu(gate[i]) * up_buf[i]; }
            mv(&w.ffn_down[l], gw.and_then(|g| g.ffn_down[l].as_ref()), &mut ff,     &gate);
            ops::add_into(&mut x, &ff);
        }

        // ── Final norm + vocabulary projection ──────────────────────────────
        ops::rmsnorm(&mut x, &w.output_norm, c.rms_norm_eps);
        let mut logits = vec![0f32; c.n_vocab];
        mv(&w.output, self.gpu_w.as_ref().and_then(|g| g.output.as_ref()), &mut logits, &x);
        logits
    }
}
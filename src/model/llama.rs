use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use crate::gguf::{reader, types::GgufFile};
use crate::tensor::{dequant::QuantTensor, storage::TensorStorage};
use crate::model::config::ModelConfig;
use crate::math::{ops, rope};
use crate::gpu::{VkCtx, GpuTensor, ActBuf};

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
    pub attn_q_bias: Vec<Option<Vec<f32>>>,
    pub attn_k_bias: Vec<Option<Vec<f32>>>,
    pub attn_v_bias: Vec<Option<Vec<f32>>>,
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
    pub k_cache:     Vec<ActBuf>,
    pub v_cache:     Vec<ActBuf>,
    pub scores:      ActBuf,
    pub ctx_len:     usize,
    pub attn_norms:  Vec<ActBuf>,
    pub ffn_norms:   Vec<ActBuf>,
    pub out_norm:    ActBuf,
    pub q_bias:      Vec<Option<ActBuf>>,
    pub k_bias:      Vec<Option<ActBuf>>,
    pub v_bias:      Vec<Option<ActBuf>>,
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

impl LlamaModel {
    pub fn load(path: &Path, ctx_len: usize,
                gpu: Option<&mut VkCtx>) -> Result<(Self, GgufFile)> {
        eprintln!("Parsing GGUF...");
        let f    = std::fs::File::open(path)?;
        let gguf = reader::parse(std::io::BufReader::new(f))?;
        let cfg  = ModelConfig::from_gguf(&gguf)?;
        let stor = TensorStorage::new(path, gguf.data_offset)?;

        eprintln!("Config: {} layers | embd {} | heads {}/{} | ff {} | rope_base {}",
            cfg.n_layers, cfg.n_embd, cfg.n_heads, cfg.n_kv_heads,
            cfg.n_ff, cfg.rope_freq_base);

        let tmap: HashMap<&str, _> = gguf.tensors.iter()
            .map(|t| (t.name.as_str(), t)).collect();

        let get_q = |name: &str| -> Result<QuantTensor> {
            let info = tmap.get(name)
                .ok_or_else(|| anyhow::anyhow!("Missing tensor: {}", name))?;
            Ok(QuantTensor::new(stor.mmap.clone(), stor.tensor_offset(info),
                                info.byte_size(), info.typ, &info.dims))
        };
        let get_f = |name: &str| -> Result<Vec<f32>> {
            let info = tmap.get(name)
                .ok_or_else(|| anyhow::anyhow!("Missing tensor: {}", name))?;
            let s = stor.tensor_offset(info);
            crate::tensor::dequant::dequantize(
                info.typ, &stor.mmap[s..s+info.byte_size()], info.n_elements())
        };
        let get_bias = |name: &str| -> Option<Vec<f32>> {
            tmap.get(name).and_then(|info| {
                let s = stor.tensor_offset(info);
                crate::tensor::dequant::dequantize(
                    info.typ, &stor.mmap[s..s+info.byte_size()], info.n_elements()).ok()
            })
        };

        let mb = gguf.tensors.iter().map(|t| t.byte_size()).sum::<usize>() / 1_000_000;
        eprintln!("Loading weights (~{} MB)...", mb);

        let token_embd  = get_q("token_embd.weight")?;
        let output_norm = get_f("output_norm.weight")?;
        let output      = get_q("output.weight").or_else(|_| get_q("token_embd.weight"))?;

        let (mut an, mut fn_) = (vec![], vec![]);
        let (mut aq, mut ak, mut av, mut ao) = (vec![], vec![], vec![], vec![]);
        let (mut fg, mut fu, mut fd) = (vec![], vec![], vec![]);
        let (mut aqb, mut akb, mut avb) = (vec![], vec![], vec![]);

        for i in 0..cfg.n_layers {
            an.push(get_f(&format!("blk.{}.attn_norm.weight",   i))?);
            fn_.push(get_f(&format!("blk.{}.ffn_norm.weight",   i))?);
            aq.push(get_q(&format!("blk.{}.attn_q.weight",      i))?);
            ak.push(get_q(&format!("blk.{}.attn_k.weight",      i))?);
            av.push(get_q(&format!("blk.{}.attn_v.weight",      i))?);
            ao.push(get_q(&format!("blk.{}.attn_output.weight", i))?);
            fg.push(get_q(&format!("blk.{}.ffn_gate.weight",    i))?);
            fu.push(get_q(&format!("blk.{}.ffn_up.weight",      i))?);
            fd.push(get_q(&format!("blk.{}.ffn_down.weight",    i))?);
            aqb.push(get_bias(&format!("blk.{}.attn_q.bias",    i)));
            akb.push(get_bias(&format!("blk.{}.attn_k.bias",    i)));
            avb.push(get_bias(&format!("blk.{}.attn_v.bias",    i)));
        }
        eprintln!("Weights ready.");

        let weights = Weights {
            token_embd, output_norm, output,
            attn_norm: an, ffn_norm: fn_,
            attn_q: aq, attn_k: ak, attn_v: av, attn_out: ao,
            ffn_gate: fg, ffn_up: fu, ffn_down: fd,
            attn_q_bias: aqb, attn_k_bias: akb, attn_v_bias: avb,
        };

        let (gpu_w, gpu_acts) = if let Some(g) = gpu {
            eprintln!("Uploading weight tensors to GPU...");
            let n_gpu = |o: &Option<GpuTensor>| if o.is_some() { 1usize } else { 0 };
            let output_gt = g.upload(&weights.output);
            let attn_q:   Vec<_> = weights.attn_q.iter().map(|w| g.upload(w)).collect();
            let attn_k:   Vec<_> = weights.attn_k.iter().map(|w| g.upload(w)).collect();
            let attn_v:   Vec<_> = weights.attn_v.iter().map(|w| g.upload(w)).collect();
            let attn_out: Vec<_> = weights.attn_out.iter().map(|w| g.upload(w)).collect();
            let ffn_gate: Vec<_> = weights.ffn_gate.iter().map(|w| g.upload(w)).collect();
            let ffn_up:   Vec<_> = weights.ffn_up.iter().map(|w| g.upload(w)).collect();
            let ffn_down: Vec<_> = weights.ffn_down.iter().map(|w| g.upload(w)).collect();
            let on = [&attn_q,&attn_k,&attn_v,&attn_out,&ffn_gate,&ffn_up,&ffn_down]
                .iter().flat_map(|v|v.iter()).map(n_gpu).sum::<usize>() + n_gpu(&output_gt);
            eprintln!("{}/{} weight tensors on GPU, {} on CPU rayon",
                on, cfg.n_layers*7+1, cfg.n_layers*7+1 - on);

            let gw = GpuWeights {
                output: output_gt, attn_q, attn_k, attn_v, attn_out,
                ffn_gate, ffn_up, ffn_down,
            };

            eprintln!("Allocating GPU activation buffers (ctx={})...", ctx_len);
            let hd      = cfg.head_dim();
            let kvd     = cfg.n_kv_heads * hd;
            let kv_size = (ctx_len * kvd * 4) as u64;

            let mut k_cache = Vec::with_capacity(cfg.n_layers);
            let mut v_cache = Vec::with_capacity(cfg.n_layers);
            for _ in 0..cfg.n_layers {
                k_cache.push(g.alloc_act(kv_size)?);
                v_cache.push(g.alloc_act(kv_size)?);
            }

            let scores = g.alloc_act((cfg.n_heads * ctx_len) as u64 * 4)?;

            let mut attn_norms = Vec::with_capacity(cfg.n_layers);
            let mut ffn_norms  = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                let ab = g.alloc_act(cfg.n_embd as u64 * 4)?;
                g.write_act(&ab, &weights.attn_norm[i]);
                attn_norms.push(ab);
                let ab = g.alloc_act(cfg.n_embd as u64 * 4)?;
                g.write_act(&ab, &weights.ffn_norm[i]);
                ffn_norms.push(ab);
            }
            let out_norm = g.alloc_act(cfg.n_embd as u64 * 4)?;
            g.write_act(&out_norm, &weights.output_norm);

            let mut q_bias_bufs = Vec::with_capacity(cfg.n_layers);
            let mut k_bias_bufs = Vec::with_capacity(cfg.n_layers);
            let mut v_bias_bufs = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                q_bias_bufs.push(if let Some(ref b) = weights.attn_q_bias[i] {
                    let ab = g.alloc_act(b.len() as u64 * 4)?;
                    g.write_act(&ab, b); Some(ab)
                } else { None });
                k_bias_bufs.push(if let Some(ref b) = weights.attn_k_bias[i] {
                    let ab = g.alloc_act(b.len() as u64 * 4)?;
                    g.write_act(&ab, b); Some(ab)
                } else { None });
                v_bias_bufs.push(if let Some(ref b) = weights.attn_v_bias[i] {
                    let ab = g.alloc_act(b.len() as u64 * 4)?;
                    g.write_act(&ab, b); Some(ab)
                } else { None });
            }

            let acts = GpuActs {
                x:        g.alloc_act(cfg.n_embd as u64 * 4)?,
                xn:       g.alloc_act(cfg.n_embd as u64 * 4)?,
                q:        g.alloc_act((cfg.n_heads * hd) as u64 * 4)?,
                k:        g.alloc_act(kvd as u64 * 4)?,
                v:        g.alloc_act(kvd as u64 * 4)?,
                attn_out: g.alloc_act(cfg.n_embd as u64 * 4)?,
                proj:     g.alloc_act(cfg.n_embd as u64 * 4)?,
                gate:     g.alloc_act(cfg.n_ff as u64 * 4)?,
                up:       g.alloc_act(cfg.n_ff as u64 * 4)?,
                ff:       g.alloc_act(cfg.n_embd as u64 * 4)?,
                logits:   g.alloc_act(cfg.n_vocab as u64 * 4)?,
                logits_rb:g.alloc_readback(cfg.n_vocab as u64 * 4)?,
                k_cache, v_cache, scores,
                ctx_len,
                attn_norms, ffn_norms, out_norm,
                q_bias: q_bias_bufs,
                k_bias: k_bias_bufs,
                v_bias: v_bias_bufs,
            };
            eprintln!("GPU buffers ready.");
            (Some(gw), Some(acts))
        } else {
            (None, None)
        };

        Ok((Self { config: cfg, weights, gpu_w, gpu_acts }, gguf))
    }

    pub fn forward_gpu(&self, token: usize, pos: usize, gpu: &mut VkCtx) -> Vec<f32> {
        let c   = &self.config;
        let w   = &self.weights;
        let gw  = self.gpu_w.as_ref().unwrap();
        let ga  = self.gpu_acts.as_ref().unwrap();
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;

        let emb = w.token_embd.get_row(token);
        gpu.write_act(&ga.x, &emb);

        gpu.begin();

        for l in 0..c.n_layers {
            gpu.cmd_rmsnorm(&ga.x, &ga.attn_norms[l], &ga.xn,
                            c.n_embd as u32, c.rms_norm_eps);
            gpu.barrier();

            if let Some(t) = gw.attn_q[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.q); }
            if let Some(t) = gw.attn_k[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.k); }
            if let Some(t) = gw.attn_v[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.v); }
            gpu.barrier();

            if let Some(ref b) = ga.q_bias[l] {
                gpu.cmd_add(&ga.q, b, (c.n_heads * hd) as u32); gpu.barrier(); }
            if let Some(ref b) = ga.k_bias[l] {
                gpu.cmd_add(&ga.k, b, kvd as u32); gpu.barrier(); }
            if let Some(ref b) = ga.v_bias[l] {
                gpu.cmd_add(&ga.v, b, kvd as u32); gpu.barrier(); }

            gpu.cmd_rope(&ga.q, &ga.k,
                         c.n_heads as u32, c.n_kv_heads as u32,
                         hd as u32, pos as u32, c.rope_freq_base);
            gpu.barrier();

            gpu.cmd_kv_write(&ga.k, &ga.v, &ga.k_cache[l], &ga.v_cache[l],
                             pos as u32, c.n_kv_heads as u32, hd as u32);
            gpu.barrier();

            gpu.cmd_attention(&ga.q, &ga.k_cache[l], &ga.v_cache[l],
                              &ga.attn_out, &ga.scores,
                              c.n_heads as u32, c.n_kv_heads as u32,
                              hd as u32, (pos + 1) as u32, ga.ctx_len as u32);
            gpu.barrier();

            if let Some(t) = gw.attn_out[l].as_ref() {
                gpu.cmd_gemv(t, &ga.attn_out, &ga.proj); gpu.barrier(); }

            gpu.cmd_add(&ga.x, &ga.proj, c.n_embd as u32);
            gpu.barrier();

            gpu.cmd_rmsnorm(&ga.x, &ga.ffn_norms[l], &ga.xn,
                            c.n_embd as u32, c.rms_norm_eps);
            gpu.barrier();

            if let Some(t) = gw.ffn_gate[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.gate); }
            if let Some(t) = gw.ffn_up[l].as_ref()   { gpu.cmd_gemv(t, &ga.xn, &ga.up); }
            gpu.barrier();

            gpu.cmd_swiglu(&ga.gate, &ga.up, c.n_ff as u32);
            gpu.barrier();

            if let Some(t) = gw.ffn_down[l].as_ref() {
                gpu.cmd_gemv(t, &ga.gate, &ga.ff); gpu.barrier(); }

            gpu.cmd_add(&ga.x, &ga.ff, c.n_embd as u32);
            gpu.barrier();
        }

        gpu.cmd_rmsnorm(&ga.x, &ga.out_norm, &ga.xn,
                        c.n_embd as u32, c.rms_norm_eps);
        gpu.barrier();
        if let Some(t) = gw.output.as_ref() {
            gpu.cmd_gemv(t, &ga.xn, &ga.logits);
        }

        gpu.submit();
        gpu.read_logits(&ga.logits, &ga.logits_rb)
    }

    pub fn forward_cpu(&self, token: usize, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let c   = &self.config;
        let w   = &self.weights;
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;

        let mut x      = w.token_embd.get_row(token);
        let mut xn     = vec![0f32; c.n_embd];
        let mut q      = vec![0f32; c.n_heads * hd];
        let mut k      = vec![0f32; kvd];
        let mut v      = vec![0f32; kvd];
        let mut scores = vec![0f32; c.n_heads * (pos + 1)];
        let mut attn   = vec![0f32; c.n_embd];
        let mut proj   = vec![0f32; c.n_embd];
        let mut gate   = vec![0f32; c.n_ff];
        let mut up     = vec![0f32; c.n_ff];
        let mut ff     = vec![0f32; c.n_embd];

        for l in 0..c.n_layers {
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.attn_norm[l], c.rms_norm_eps);
            w.attn_q[l].matvec(&mut q, &xn);
            w.attn_k[l].matvec(&mut k, &xn);
            w.attn_v[l].matvec(&mut v, &xn);
            if let Some(ref b) = w.attn_q_bias[l] { ops::add_into(&mut q, b); }
            if let Some(ref b) = w.attn_k_bias[l] { ops::add_into(&mut k, b); }
            if let Some(ref b) = w.attn_v_bias[l] { ops::add_into(&mut v, b); }
            rope::apply_rope(&mut q, &mut k, pos, hd, c.rope_freq_base, c.n_heads, c.n_kv_heads);
            let cb = pos * kvd;
            cache.k[l][cb..cb+kvd].copy_from_slice(&k);
            cache.v[l][cb..cb+kvd].copy_from_slice(&v);
            let kv_ratio = c.n_heads / c.n_kv_heads;
            attn.fill(0.0);
            for h in 0..c.n_heads {
                let kv_h = h / kv_ratio;
                let qh   = &q[h*hd..(h+1)*hd];
                let sc   = &mut scores[h*(pos+1)..(h+1)*(pos+1)];
                let scale= (hd as f32).sqrt();
                for p in 0..=pos {
                    let ko = p*kvd+kv_h*hd;
                    sc[p] = qh.iter().zip(cache.k[l][ko..ko+hd].iter())
                               .map(|(a,b)| a*b).sum::<f32>() / scale;
                }
                ops::softmax(sc);
                let ah = &mut attn[h*hd..(h+1)*hd];
                ah.fill(0.0);
                for p in 0..=pos {
                    let vo = p*kvd+kv_h*hd;
                    let sp = sc[p];
                    for (o,vi) in ah.iter_mut().zip(cache.v[l][vo..vo+hd].iter()) { *o+=sp*vi; }
                }
            }
            w.attn_out[l].matvec(&mut proj, &attn);
            ops::add_into(&mut x, &proj);
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.ffn_norm[l], c.rms_norm_eps);
            w.ffn_gate[l].matvec(&mut gate, &xn);
            w.ffn_up[l].matvec(&mut up, &xn);
            for i in 0..c.n_ff { gate[i] = ops::silu(gate[i]) * up[i]; }
            w.ffn_down[l].matvec(&mut ff, &gate);
            ops::add_into(&mut x, &ff);
        }
        ops::rmsnorm(&mut x, &w.output_norm, c.rms_norm_eps);
        let mut logits = vec![0f32; c.n_vocab];
        w.output.matvec(&mut logits, &x);
        logits
    }
}
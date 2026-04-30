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
            cfg.n_layers, cfg.n_embd, cfg.n_heads, cfg.n_kv_heads, cfg.n_ff, cfg.rope_freq_base);
        let tmap: HashMap<&str,_> = gguf.tensors.iter().map(|t|(t.name.as_str(),t)).collect();
        let get_q = |name: &str| -> Result<QuantTensor> {
            let info = tmap.get(name).ok_or_else(||anyhow::anyhow!("Missing: {}",name))?;
            Ok(QuantTensor::new(stor.mmap.clone(),stor.tensor_offset(info),info.byte_size(),info.typ,&info.dims))
        };
        let get_f = |name: &str| -> Result<Vec<f32>> {
            let info = tmap.get(name).ok_or_else(||anyhow::anyhow!("Missing: {}",name))?;
            let s = stor.tensor_offset(info);
            crate::tensor::dequant::dequantize(info.typ,&stor.mmap[s..s+info.byte_size()],info.n_elements())
        };
        let get_bias = |name: &str| -> Option<Vec<f32>> {
            tmap.get(name).and_then(|info|{let s=stor.tensor_offset(info);
                crate::tensor::dequant::dequantize(info.typ,&stor.mmap[s..s+info.byte_size()],info.n_elements()).ok()})
        };
        let mb = gguf.tensors.iter().map(|t|t.byte_size()).sum::<usize>()/1_000_000;
        eprintln!("Loading weights (~{} MB compressed on disk)...", mb);
        let token_embd  = get_q("token_embd.weight")?;
        let output_norm = get_f("output_norm.weight")?;
        let output      = get_q("output.weight").or_else(|_|get_q("token_embd.weight"))?;
        let (mut an,mut fn_)=(vec![],vec![]);
        let (mut aq,mut ak,mut av,mut ao)=(vec![],vec![],vec![],vec![]);
        let (mut fg,mut fu,mut fd)=(vec![],vec![],vec![]);
        let (mut aqb,mut akb,mut avb)=(vec![],vec![],vec![]);
        for i in 0..cfg.n_layers {
            an.push(get_f(&format!("blk.{}.attn_norm.weight",i))?);
            fn_.push(get_f(&format!("blk.{}.ffn_norm.weight",i))?);
            aq.push(get_q(&format!("blk.{}.attn_q.weight",i))?);
            ak.push(get_q(&format!("blk.{}.attn_k.weight",i))?);
            av.push(get_q(&format!("blk.{}.attn_v.weight",i))?);
            ao.push(get_q(&format!("blk.{}.attn_output.weight",i))?);
            fg.push(get_q(&format!("blk.{}.ffn_gate.weight",i))?);
            fu.push(get_q(&format!("blk.{}.ffn_up.weight",i))?);
            fd.push(get_q(&format!("blk.{}.ffn_down.weight",i))?);
            aqb.push(get_bias(&format!("blk.{}.attn_q.bias",i)));
            akb.push(get_bias(&format!("blk.{}.attn_k.bias",i)));
            avb.push(get_bias(&format!("blk.{}.attn_v.bias",i)));
        }
        eprintln!("Weights ready.");
        let weights = Weights { token_embd, output_norm, output,
            attn_norm:an, ffn_norm:fn_, attn_q:aq, attn_k:ak, attn_v:av, attn_out:ao,
            ffn_gate:fg, ffn_up:fu, ffn_down:fd,
            attn_q_bias:aqb, attn_k_bias:akb, attn_v_bias:avb };
        let gpu_w = gpu.map(|g| {
            eprintln!("Uploading tensors to GPU...");
            let up    = |wt: &QuantTensor| g.upload(wt);
            let n_gpu = |o: &Option<_>| if o.is_some(){1usize}else{0};
            let output   = up(&weights.output);
            let attn_q:   Vec<_> = weights.attn_q.iter().map(up).collect();
            let attn_k:   Vec<_> = weights.attn_k.iter().map(up).collect();
            let attn_v:   Vec<_> = weights.attn_v.iter().map(up).collect();
            let attn_out: Vec<_> = weights.attn_out.iter().map(up).collect();
            let ffn_gate: Vec<_> = weights.ffn_gate.iter().map(up).collect();
            let ffn_up:   Vec<_> = weights.ffn_up.iter().map(up).collect();
            let ffn_down: Vec<_> = weights.ffn_down.iter().map(up).collect();
            let on_gpu = [&attn_q,&attn_k,&attn_v,&attn_out,&ffn_gate,&ffn_up,&ffn_down]
                .iter().flat_map(|v|v.iter()).map(n_gpu).sum::<usize>()+n_gpu(&output);
            let total = cfg.n_layers*7+1;
            eprintln!("{}/{} tensors on GPU, {} on CPU rayon", on_gpu, total, total-on_gpu);
            GpuWeights { output, attn_q, attn_k, attn_v, attn_out, ffn_gate, ffn_up, ffn_down }
        });
        Ok((Self { config: cfg, weights, gpu_w }, gguf))
    }

    pub fn forward(&self, token: usize, pos: usize, cache: &mut KvCache,
                   gpu: Option<&mut GpuCtx>) -> Vec<f32> {
        let c  = &self.config;
        let w  = &self.weights;
        let gw = self.gpu_w.as_ref();
        let hd = c.head_dim();
        let kvd = c.n_kv_heads * hd;
        let mut x      = w.token_embd.get_row(token);
        let mut xn     = vec![0f32; c.n_embd];
        let mut q      = vec![0f32; c.n_heads*hd];
        let mut k      = vec![0f32; kvd];
        let mut v      = vec![0f32; kvd];
        let mut scores = vec![0f32; c.n_heads*(pos+1)];
        let mut attn   = vec![0f32; c.n_embd];
        let mut proj   = vec![0f32; c.n_embd];
        let mut gate   = vec![0f32; c.n_ff];
        let mut up_buf = vec![0f32; c.n_ff];
        let mut ff     = vec![0f32; c.n_embd];
        let gptr: Option<*mut GpuCtx> = gpu.map(|g| g as *mut GpuCtx);

        // Queue a matmul on GPU if available, else run CPU rayon immediately.
        // Returns true if queued (call flush after the group), false if done on CPU.
        let queue = |cpu: &QuantTensor, gt: Option<&GpuTensor>,
                     fallback: &mut Vec<f32>, inp: &[f32]| -> bool {
            match (gptr, gt) {
                (Some(g), Some(t)) => { unsafe { (*g).queue_matmul(t, inp) }; true }
                _                  => { cpu.matvec(fallback, inp); false }
            }
        };

        // Flush all pending GPU ops and scatter results into destination buffers.
        let flush = |dsts: &mut [*mut Vec<f32>], flags: &[bool]| {
            if !flags.iter().any(|&f| f) { return; }
            let results = unsafe { (*gptr.unwrap()).flush() };
            let mut ri = 0;
            for (i, &used) in flags.iter().enumerate() {
                if used { unsafe { (*dsts[i]).copy_from_slice(&results[ri]) }; ri += 1; }
            }
        };

        for l in 0..c.n_layers {
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.attn_norm[l], c.rms_norm_eps);

            // Batch Q+K+V: all share the same input, submit together
            let fq = queue(&w.attn_q[l], gw.and_then(|g|g.attn_q[l].as_ref()),   &mut q, &xn);
            let fk = queue(&w.attn_k[l], gw.and_then(|g|g.attn_k[l].as_ref()),   &mut k, &xn);
            let fv = queue(&w.attn_v[l], gw.and_then(|g|g.attn_v[l].as_ref()),   &mut v, &xn);
            flush(&mut [&mut q as *mut _, &mut k as *mut _, &mut v as *mut _], &[fq,fk,fv]);

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
                let kv_h  = h/kv_ratio;
                let qh    = &q[h*hd..(h+1)*hd];
                let sc    = &mut scores[h*(pos+1)..(h+1)*(pos+1)];
                let scale = (hd as f32).sqrt();
                for p in 0..=pos {
                    let ko = p*kvd+kv_h*hd;
                    sc[p]  = qh.iter().zip(cache.k[l][ko..ko+hd].iter())
                               .map(|(a,b)|a*b).sum::<f32>()/scale;
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

            let fo = queue(&w.attn_out[l], gw.and_then(|g|g.attn_out[l].as_ref()), &mut proj, &attn);
            flush(&mut [&mut proj as *mut _], &[fo]);
            ops::add_into(&mut x, &proj);

            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.ffn_norm[l], c.rms_norm_eps);

            // Batch gate+up: share the same input
            let fg = queue(&w.ffn_gate[l], gw.and_then(|g|g.ffn_gate[l].as_ref()), &mut gate,   &xn);
            let fu = queue(&w.ffn_up[l],   gw.and_then(|g|g.ffn_up[l].as_ref()),   &mut up_buf, &xn);
            flush(&mut [&mut gate as *mut _, &mut up_buf as *mut _], &[fg,fu]);

            for i in 0..c.n_ff { gate[i] = ops::silu(gate[i])*up_buf[i]; }

            let fd = queue(&w.ffn_down[l], gw.and_then(|g|g.ffn_down[l].as_ref()), &mut ff, &gate);
            flush(&mut [&mut ff as *mut _], &[fd]);
            ops::add_into(&mut x, &ff);
        }

        ops::rmsnorm(&mut x, &w.output_norm, c.rms_norm_eps);
        let mut logits = vec![0f32; c.n_vocab];
        let fl = queue(&w.output, self.gpu_w.as_ref().and_then(|g|g.output.as_ref()), &mut logits, &x);
        flush(&mut [&mut logits as *mut _], &[fl]);
        logits
    }
}
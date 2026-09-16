use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use crate::gguf::{reader, types::GgufFile};
use crate::tensor::{dequant::QuantTensor, storage::TensorStorage};
use crate::gpu::{VkCtx, GpuTensor};
use super::*;

impl LlamaModel {
    pub fn load(path: &Path, ctx_len: usize,
                gpu: Option<&mut VkCtx>) -> Result<(Self, GgufFile)> {
        Self::load_with_slots(path, ctx_len, 1, gpu)
    }

    pub fn load_with_slots(path: &Path, ctx_len: usize, n_slots: usize,
                gpu: Option<&mut VkCtx>) -> Result<(Self, GgufFile)> {
        eprintln!("Parsing GGUF...");
        let f    = std::fs::File::open(path)?;
        let gguf = reader::parse(std::io::BufReader::new(f))?;
        let cfg  = ModelConfig::from_gguf(&gguf)?;
        let stor = TensorStorage::new(path, gguf.data_offset)?;

        eprintln!("Config: {} layers | embd {} | heads {}/{} | ff {} | rope_base {}",
            cfg.n_layers, cfg.n_embd, cfg.n_heads, cfg.n_kv_heads,
            cfg.n_ff, cfg.rope_freq_base);
        let mut blk0: Vec<&str> = gguf.tensors.iter()
            .filter(|t| t.name.starts_with("blk.0."))
            .map(|t| t.name.as_str()).collect();
        blk0.sort();
        eprintln!("Tensors in blk.0: {}", blk0.join(", "));

        let tmap: HashMap<&str, _> = gguf.tensors.iter()
            .map(|t| (t.name.as_str(), t)).collect();

        let similar_names = |name: &str| -> String {
            let prefix = name.split('.').take(2).collect::<Vec<_>>().join(".");
            let mut found: Vec<&str> = tmap.keys()
                .filter(|k| k.starts_with(&prefix))
                .copied().take(8).collect();
            found.sort();
            found.join(", ")
        };
        let get_q = |name: &str| -> Result<QuantTensor> {
            let info = tmap.get(name).ok_or_else(|| {
                anyhow::anyhow!("Missing tensor: {}\n  Similar: {}", name, similar_names(name))
            })?;
            Ok(QuantTensor::new(stor.mmap.clone(), stor.tensor_offset(info),
                                info.byte_size(), info.typ, &info.dims))
        };
        let get_f = |name: &str| -> Result<Vec<f32>> {
            let info = tmap.get(name).ok_or_else(|| {
                anyhow::anyhow!("Missing tensor: {}\n  Similar: {}", name, similar_names(name))
            })?;
            let s = stor.tensor_offset(info);
            crate::tensor::dequant::dequantize(
                info.typ, &stor.mmap[s..s+info.byte_size()], info.n_elements())
        };
        let get_f_any = |names: &[&str]| -> Result<Vec<f32>> {
            for name in names {
                if let Some(info) = tmap.get(*name) {
                    let s = stor.tensor_offset(info);
                    return crate::tensor::dequant::dequantize(
                        info.typ, &stor.mmap[s..s+info.byte_size()], info.n_elements());
                }
            }
            let similar = similar_names(names[0]);
            Err(anyhow::anyhow!("Missing tensor (tried: {})\n  Similar: {}",
                names.join(", "), similar))
        };
        let get_q_any = |names: &[&str]| -> Result<QuantTensor> {
            for name in names {
                if let Some(info) = tmap.get(*name) {
                    return Ok(QuantTensor::new(stor.mmap.clone(), stor.tensor_offset(info),
                                               info.byte_size(), info.typ, &info.dims));
                }
            }
            let similar = similar_names(names[0]);
            Err(anyhow::anyhow!("Missing tensor (tried: {})\n  Similar: {}",
                names.join(", "), similar))
        };
        let get_q_opt = |name: &str| -> Option<QuantTensor> {
            tmap.get(name).map(|info| QuantTensor::new(stor.mmap.clone(),
                stor.tensor_offset(info), info.byte_size(), info.typ, &info.dims))
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
        let (mut aq_f32, mut ak_f32, mut av_f32): (Vec<Option<Vec<f32>>>, Vec<Option<Vec<f32>>>, Vec<Option<Vec<f32>>>) = (vec![], vec![], vec![]);
        let (mut aqn, mut akn): (Vec<Option<Vec<f32>>>, Vec<Option<Vec<f32>>>) = (vec![], vec![]);
        let (mut apn, mut fpn): (Vec<Option<Vec<f32>>>, Vec<Option<Vec<f32>>>) = (vec![], vec![]);
        let (mut frt, mut fge, mut fue, mut fde): (Vec<Option<QuantTensor>>, Vec<Option<QuantTensor>>, Vec<Option<QuantTensor>>, Vec<Option<QuantTensor>>) = (vec![], vec![], vec![], vec![]);

        for i in 0..cfg.n_layers {
            an.push(get_f_any(&[
                &format!("blk.{}.attn_norm.weight",             i),
                &format!("blk.{}.pre_attention_layernorm.weight",i),
            ])?);
            fn_.push(get_f_any(&[
                &format!("blk.{}.ffn_norm.weight",               i),
                &format!("blk.{}.post_attention_layernorm.weight",i),
                &format!("blk.{}.post_attention_norm.weight",     i),
                &format!("blk.{}.pre_ff_layernorm.weight",        i),
            ])?);
            let has_fused_qkv = tmap.contains_key(format!("blk.{}.attn_qkv.weight", i).as_str())
                && !tmap.contains_key(format!("blk.{}.attn_q.weight", i).as_str());
            if has_fused_qkv {
                let fused = get_q_any(&[&format!("blk.{}.attn_qkv.weight", i)])?;
                let n_q   = cfg.n_heads    * cfg.head_dim();
                let n_k   = cfg.n_kv_heads * cfg.head_dim();
                let n_v   = cfg.n_kv_heads * cfg.head_dim();
                let cols  = cfg.n_embd;
                let mut q_f32 = Vec::with_capacity(n_q * cols);
                let mut k_f32 = Vec::with_capacity(n_k * cols);
                let mut v_f32 = Vec::with_capacity(n_v * cols);
                for r in 0..n_q           { q_f32.extend_from_slice(&fused.get_row(r)); }
                for r in n_q..n_q+n_k     { k_f32.extend_from_slice(&fused.get_row(r)); }
                for r in n_q+n_k..n_q+n_k+n_v { v_f32.extend_from_slice(&fused.get_row(r)); }
                aq.push(get_q_any(&[&format!("blk.{}.attn_qkv.weight", i)])?);
                ak.push(get_q_any(&[&format!("blk.{}.attn_qkv.weight", i)])?);
                av.push(get_q_any(&[&format!("blk.{}.attn_qkv.weight", i)])?);
                aq_f32.push(Some(q_f32));
                ak_f32.push(Some(k_f32));
                av_f32.push(Some(v_f32));
            } else {
                aq.push(get_q_any(&[
                    &format!("blk.{}.attn_q.weight",          i),
                    &format!("blk.{}.self_attn.q_proj.weight",i),
                ])?);
                ak.push(get_q_any(&[
                    &format!("blk.{}.attn_k.weight",          i),
                    &format!("blk.{}.self_attn.k_proj.weight",i),
                ])?);
                av.push(get_q_any(&[
                    &format!("blk.{}.attn_v.weight",          i),
                    &format!("blk.{}.self_attn.v_proj.weight",i),
                ])?);
                aq_f32.push(None); ak_f32.push(None); av_f32.push(None);
            }
            ao.push(get_q_any(&[
                &format!("blk.{}.attn_output.weight",            i),
                &format!("blk.{}.self_attn.o_proj.weight",       i),
            ])?);
            let router = get_q_opt(&format!("blk.{}.ffn_gate_inp.weight", i));
            if router.is_some() {
                fg.push(None); fu.push(None); fd.push(None);
                frt.push(router);
                fge.push(get_q_opt(&format!("blk.{}.ffn_gate_exps.weight", i)));
                fue.push(get_q_opt(&format!("blk.{}.ffn_up_exps.weight",   i)));
                fde.push(get_q_opt(&format!("blk.{}.ffn_down_exps.weight", i)));
            } else {
                fg.push(Some(get_q_any(&[
                    &format!("blk.{}.ffn_gate.weight",               i),
                    &format!("blk.{}.mlp.gate_proj.weight",          i),
                ])?));
                fu.push(Some(get_q_any(&[
                    &format!("blk.{}.ffn_up.weight",                 i),
                    &format!("blk.{}.mlp.up_proj.weight",            i),
                ])?));
                fd.push(Some(get_q_any(&[
                    &format!("blk.{}.ffn_down.weight",               i),
                    &format!("blk.{}.mlp.down_proj.weight",          i),
                ])?));
                frt.push(None); fge.push(None); fue.push(None); fde.push(None);
            }
            aqb.push(get_bias(&format!("blk.{}.attn_q.bias",    i)));
            akb.push(get_bias(&format!("blk.{}.attn_k.bias",    i)));
            avb.push(get_bias(&format!("blk.{}.attn_v.bias",    i)));
            aqn.push(get_bias(&format!("blk.{}.attn_q_norm.weight", i)));
            akn.push(get_bias(&format!("blk.{}.attn_k_norm.weight", i)));
            apn.push(get_bias(&format!("blk.{}.post_attention_norm.weight", i)));
            fpn.push(get_bias(&format!("blk.{}.post_ffw_norm.weight", i)));
        }
        eprintln!("Weights ready.");

        let weights = Weights {
            token_embd, output_norm, output,
            attn_norm: an, ffn_norm: fn_,
            attn_q: aq, attn_k: ak, attn_v: av, attn_out: ao,
            ffn_gate: fg, ffn_up: fu, ffn_down: fd,
            attn_q_bias: aqb, attn_k_bias: akb, attn_v_bias: avb,
            attn_q_f32: aq_f32, attn_k_f32: ak_f32, attn_v_f32: av_f32,
            attn_q_norm: aqn, attn_k_norm: akn,
            attn_post_norm: apn, ffn_post_norm: fpn,
            ffn_router: frt, ffn_gate_exps: fge, ffn_up_exps: fue, ffn_down_exps: fde,
        };

        let (gpu_w, gpu_acts) = if let Some(g) = gpu {
            eprintln!("Uploading weight tensors to GPU...");
            let n_gpu = |o: &Option<GpuTensor>| if o.is_some() { 1usize } else { 0 };
            let output_gt = g.upload_any(&weights.output);
            let attn_q: Vec<_> = (0..cfg.n_layers).map(|i| {
                if let Some(ref f32d) = weights.attn_q_f32[i] {
                    g.upload_f32_weight(f32d, (cfg.n_heads * cfg.head_dim()) as u32, cfg.n_embd as u32)
                } else { g.upload_any(&weights.attn_q[i]) }
            }).collect();
            let attn_k: Vec<_> = (0..cfg.n_layers).map(|i| {
                if let Some(ref f32d) = weights.attn_k_f32[i] {
                    g.upload_f32_weight(f32d, (cfg.n_kv_heads * cfg.head_dim()) as u32, cfg.n_embd as u32)
                } else { g.upload_any(&weights.attn_k[i]) }
            }).collect();
            let attn_v: Vec<_> = (0..cfg.n_layers).map(|i| {
                if let Some(ref f32d) = weights.attn_v_f32[i] {
                    g.upload_f32_weight(f32d, (cfg.n_kv_heads * cfg.head_dim()) as u32, cfg.n_embd as u32)
                } else { g.upload_any(&weights.attn_v[i]) }
            }).collect();
            let attn_out: Vec<_> = weights.attn_out.iter().map(|w| g.upload_any(w)).collect();
            let ffn_gate: Vec<_> = weights.ffn_gate.iter().map(|w| w.as_ref().and_then(|w| g.upload_any(w))).collect();
            let ffn_up:   Vec<_> = weights.ffn_up.iter().map(|w| w.as_ref().and_then(|w| g.upload_any(w))).collect();
            let ffn_down: Vec<_> = weights.ffn_down.iter().map(|w| w.as_ref().and_then(|w| g.upload_any(w))).collect();
            let on = [&attn_q,&attn_k,&attn_v,&attn_out,&ffn_gate,&ffn_up,&ffn_down]
                .iter().flat_map(|v|v.iter()).map(n_gpu).sum::<usize>() + n_gpu(&output_gt);
            eprintln!("{}/{} weight tensors resident on GPU", on, cfg.n_layers*7+1);

            let gw = GpuWeights {
                output: output_gt, attn_q, attn_k, attn_v, attn_out,
                ffn_gate, ffn_up, ffn_down,
            };

            let n_slots  = n_slots.max(1);
            let slot_ctx = (ctx_len / n_slots).max(1);
            eprintln!("Allocating GPU activation buffers (ctx={} / {} slot(s) = {} each)...",
                ctx_len, n_slots, slot_ctx);
            let hd      = cfg.head_dim();
            let kvd     = cfg.n_kv_heads * hd;
            let kv_size = (slot_ctx * kvd * 4) as u64;

            let mut k_cache = Vec::with_capacity(n_slots);
            let mut v_cache = Vec::with_capacity(n_slots);
            for _ in 0..n_slots {
                let mut kl = Vec::with_capacity(cfg.n_layers);
                let mut vl = Vec::with_capacity(cfg.n_layers);
                for _ in 0..cfg.n_layers {
                    kl.push(g.alloc_act(kv_size)?);
                    vl.push(g.alloc_act(kv_size)?);
                }
                k_cache.push(kl);
                v_cache.push(vl);
            }

            let scores = g.alloc_act((cfg.n_heads * slot_ctx) as u64 * 4)?;

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

            let mut q_norm_bufs = Vec::with_capacity(cfg.n_layers);
            let mut k_norm_bufs = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                q_norm_bufs.push(if let Some(ref w) = weights.attn_q_norm[i] {
                    let ab = g.alloc_act(w.len() as u64 * 4)?;
                    g.write_act(&ab, w); Some(ab)
                } else { None });
                k_norm_bufs.push(if let Some(ref w) = weights.attn_k_norm[i] {
                    let ab = g.alloc_act(w.len() as u64 * 4)?;
                    g.write_act(&ab, w); Some(ab)
                } else { None });
            }

            let mut attn_post_norm_bufs = Vec::with_capacity(cfg.n_layers);
            let mut ffn_post_norm_bufs  = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                attn_post_norm_bufs.push(if let Some(ref w) = weights.attn_post_norm[i] {
                    let ab = g.alloc_act(w.len() as u64 * 4)?;
                    g.write_act(&ab, w); Some(ab)
                } else { None });
                ffn_post_norm_bufs.push(if let Some(ref w) = weights.ffn_post_norm[i] {
                    let ab = g.alloc_act(w.len() as u64 * 4)?;
                    g.write_act(&ab, w); Some(ab)
                } else { None });
            }

            let acts = GpuActs {
                x:        g.alloc_act(cfg.n_embd as u64 * 4)?,
                xn:       g.alloc_act(cfg.n_embd as u64 * 4)?,
                x_rb:     g.alloc_readback(cfg.n_embd as u64 * 4)?,
                q:        g.alloc_act((cfg.n_heads * hd) as u64 * 4)?,
                k:        g.alloc_act(kvd as u64 * 4)?,
                v:        g.alloc_act(kvd as u64 * 4)?,
                attn_out: g.alloc_act((cfg.n_heads * hd) as u64 * 4)?,
                proj:     g.alloc_act(cfg.n_embd as u64 * 4)?,
                gate:     g.alloc_act(cfg.n_ff as u64 * 4)?,
                up:       g.alloc_act(cfg.n_ff as u64 * 4)?,
                ff:       g.alloc_act(cfg.n_embd as u64 * 4)?,
                logits:   g.alloc_act(cfg.n_vocab as u64 * 4)?,
                logits_rb:g.alloc_readback(cfg.n_vocab as u64 * 4)?,
                k_cache, v_cache, scores,
                ctx_len: slot_ctx,
                attn_norms, ffn_norms, out_norm,
                q_bias: q_bias_bufs,
                k_bias: k_bias_bufs,
                v_bias: v_bias_bufs,
                q_norm: q_norm_bufs,
                k_norm: k_norm_bufs,
                attn_post_norm: attn_post_norm_bufs,
                ffn_post_norm:  ffn_post_norm_bufs,
            };
            eprintln!("GPU buffers ready.");
            (Some(gw), Some(acts))
        } else {
            (None, None)
        };

        Ok((Self { config: cfg, weights, gpu_w, gpu_acts }, gguf))
    }
}

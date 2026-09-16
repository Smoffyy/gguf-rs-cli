use crate::math::{ops, rope};
use crate::gpu::VkCtx;
use super::*;

fn f32_matvec(w: &[f32], inp: &[f32], out: &mut [f32], cols: usize) {
    use rayon::prelude::*;
    out.par_iter_mut().enumerate().for_each(|(r, o)| {
        *o = w[r*cols..(r+1)*cols].iter().zip(inp.iter()).map(|(a,b)| a*b).sum();
    });
}

fn apply_final_softcap(logits: &mut [f32], cap: Option<f32>) {
    if let Some(cap) = cap {
        for v in logits.iter_mut() { *v = cap * (*v / cap).tanh(); }
    }
}

impl LlamaModel {
    fn moe_ffn_cpu(&self, l: usize, xn: &[f32], out: &mut [f32]) {
        let c = &self.config;
        let w = &self.weights;
        let router = w.ffn_router[l].as_ref().unwrap();
        let mut router_logits = vec![0f32; c.n_expert];
        router.matvec(&mut router_logits, xn);
        let mut probs = router_logits;
        ops::softmax(&mut probs);

        let mut idx: Vec<usize> = (0..c.n_expert).collect();
        idx.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        idx.truncate(c.n_expert_used);
        let sum: f32 = idx.iter().map(|&i| probs[i]).sum::<f32>().max(1e-12);

        let gate_exps = w.ffn_gate_exps[l].as_ref().unwrap();
        let up_exps   = w.ffn_up_exps[l].as_ref().unwrap();
        let down_exps = w.ffn_down_exps[l].as_ref().unwrap();

        for v in out.iter_mut() { *v = 0.0; }
        let mut gate = vec![0f32; c.n_ff_exp];
        let mut up   = vec![0f32; c.n_ff_exp];
        let mut down = vec![0f32; c.n_embd];
        for &e in &idx {
            let weight = probs[e] / sum;
            gate_exps.expert(e, c.n_ff_exp).matvec(&mut gate, xn);
            up_exps.expert(e, c.n_ff_exp).matvec(&mut up, xn);
            if c.ffn_gelu {
                for i in 0..c.n_ff_exp { gate[i] = ops::gelu(gate[i]) * up[i]; }
            } else {
                for i in 0..c.n_ff_exp { gate[i] = ops::silu(gate[i]) * up[i]; }
            }
            down_exps.expert(e, c.n_embd).matvec(&mut down, &gate);
            for i in 0..c.n_embd { out[i] += weight * down[i]; }
        }
    }
}

impl LlamaModel {
    fn record_attn_gpu(&self, l: usize, pos: usize, gpu: &mut VkCtx) {
        let c   = &self.config;
        let gw  = self.gpu_w.as_ref().unwrap();
        let ga  = self.gpu_acts.as_ref().unwrap();
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;

        gpu.cmd_rmsnorm(&ga.x, &ga.attn_norms[l], &ga.xn, c.n_embd as u32, c.rms_norm_eps);
        gpu.barrier();

        if let Some(t) = gw.attn_q[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.q); }
        if let Some(t) = gw.attn_k[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.k); }
        if let Some(t) = gw.attn_v[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.v); }
        gpu.barrier();

        if let Some(ref b) = ga.q_bias[l] { gpu.cmd_add(&ga.q, b, (c.n_heads * hd) as u32); }
        if let Some(ref b) = ga.k_bias[l] { gpu.cmd_add(&ga.k, b, kvd as u32); }
        if let Some(ref b) = ga.v_bias[l] { gpu.cmd_add(&ga.v, b, kvd as u32); }
        if ga.q_bias[l].is_some() || ga.k_bias[l].is_some() || ga.v_bias[l].is_some() {
            gpu.barrier();
        }

        if let Some(ref w) = ga.q_norm[l] { gpu.cmd_qk_norm(&ga.q, w, c.n_heads as u32, hd as u32, c.rms_norm_eps); }
        if let Some(ref w) = ga.k_norm[l] { gpu.cmd_qk_norm(&ga.k, w, c.n_kv_heads as u32, hd as u32, c.rms_norm_eps); }
        if ga.q_norm[l].is_some() || ga.k_norm[l].is_some() { gpu.barrier(); }

        gpu.cmd_rope(&ga.q, &ga.k, c.n_heads as u32, c.n_kv_heads as u32,
                     hd as u32, pos as u32, c.rope_freq_base);
        gpu.barrier();

        gpu.cmd_kv_write(&ga.k, &ga.v, &ga.k_cache[l], &ga.v_cache[l],
                         pos as u32, c.n_kv_heads as u32, hd as u32);
        gpu.barrier();

        gpu.cmd_attention(&ga.q, &ga.k_cache[l], &ga.v_cache[l],
                          &ga.attn_out, &ga.scores,
                          c.n_heads as u32, c.n_kv_heads as u32,
                          hd as u32, (pos + 1) as u32, ga.ctx_len as u32,
                          c.attn_logit_softcap.unwrap_or(0.0));
        gpu.barrier();

        if let Some(t) = gw.attn_out[l].as_ref() { gpu.cmd_gemv(t, &ga.attn_out, &ga.proj); }
        gpu.barrier();

        if let Some(ref w) = ga.attn_post_norm[l] {
            gpu.cmd_rmsnorm(&ga.proj, w, &ga.proj, c.n_embd as u32, c.rms_norm_eps);
            gpu.barrier();
        }

        gpu.cmd_add(&ga.x, &ga.proj, c.n_embd as u32);
        gpu.barrier();
    }

    fn record_ffn_gpu(&self, l: usize, gpu: &mut VkCtx) {
        let c   = &self.config;
        let gw  = self.gpu_w.as_ref().unwrap();
        let ga  = self.gpu_acts.as_ref().unwrap();

        gpu.cmd_rmsnorm(&ga.x, &ga.ffn_norms[l], &ga.xn, c.n_embd as u32, c.rms_norm_eps);
        gpu.barrier();

        if let Some(t) = gw.ffn_gate[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.gate); }
        if let Some(t) = gw.ffn_up[l].as_ref()   { gpu.cmd_gemv(t, &ga.xn, &ga.up); }
        gpu.barrier();

        gpu.cmd_swiglu(&ga.gate, &ga.up, c.n_ff as u32, c.ffn_gelu);
        gpu.barrier();

        if let Some(t) = gw.ffn_down[l].as_ref() { gpu.cmd_gemv(t, &ga.gate, &ga.ff); }
        gpu.barrier();

        if let Some(ref w) = ga.ffn_post_norm[l] {
            gpu.cmd_rmsnorm(&ga.ff, w, &ga.ff, c.n_embd as u32, c.rms_norm_eps);
            gpu.barrier();
        }

        gpu.cmd_add(&ga.x, &ga.ff, c.n_embd as u32);
        gpu.barrier();
    }

    fn record_layer_gpu(&self, l: usize, pos: usize, gpu: &mut VkCtx) {
        self.record_attn_gpu(l, pos, gpu);
        if self.weights.ffn_router[l].is_some() {
            let ga = self.gpu_acts.as_ref().unwrap();
            let c  = &self.config;
            gpu.submit_with_readback(&ga.x, &ga.x_rb);
            let x = gpu.read_logits(&ga.x, &ga.x_rb);
            let mut xn = x.clone();
            ops::rmsnorm(&mut xn, &self.weights.ffn_norm[l], c.rms_norm_eps);
            let mut ff = vec![0f32; c.n_embd];
            self.moe_ffn_cpu(l, &xn, &mut ff);
            if let Some(ref pn) = self.weights.ffn_post_norm[l] { ops::rmsnorm(&mut ff, pn, c.rms_norm_eps); }
            let mut x_new = x;
            ops::add_into(&mut x_new, &ff);
            gpu.begin_for_sets(1024);
            gpu.cmd_upload_act(&ga.x, &x_new);
        } else {
            self.record_ffn_gpu(l, gpu);
        }
    }

    pub fn forward_gpu_prefill(&self, tokens: &[usize], start_pos: usize, gpu: &mut VkCtx, chunk: usize) -> Vec<f32> {
        if tokens.is_empty() { return vec![0f32; self.config.n_vocab]; }

        let c = &self.config;
        let chunk_sz = chunk.max(1);

        let mut all_embs = Vec::with_capacity(tokens.len() * c.n_embd);
        for &tok in tokens { all_embs.extend_from_slice(&self.weights.token_embd.get_row(tok)); }
        if c.embd_scale != 1.0 { for v in all_embs.iter_mut() { *v *= c.embd_scale; } }

        let (emb_buf, emb_mem) = gpu.upload_f32_host_visible(&all_embs);

        let n = tokens.len();
        let mut chunk_start = 0;
        while chunk_start < n {
            let chunk_end  = (chunk_start + chunk_sz).min(n);
            let last_chunk = chunk_end == n;

            let tokens_in_chunk = (chunk_end - chunk_start) as u32;
            let ds_per_token    = (c.n_layers as u32) * 22 + 4;
            gpu.begin_for_sets(tokens_in_chunk * ds_per_token * 2);
            for i in chunk_start..chunk_end {
                let ga = self.gpu_acts.as_ref().unwrap();
                gpu.cmd_copy_to_act(&ga.x, emb_buf, (i * c.n_embd * 4) as u64);
                for l in 0..c.n_layers {
                    self.record_layer_gpu(l, start_pos + i, gpu);
                }
            }

            if last_chunk {
                let ga = self.gpu_acts.as_ref().unwrap();
                let gw = self.gpu_w.as_ref().unwrap();
                gpu.cmd_rmsnorm(&ga.x, &ga.out_norm, &ga.xn, c.n_embd as u32, c.rms_norm_eps);
                gpu.barrier();
                if let Some(t) = gw.output.as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.logits); }
                gpu.submit_with_readback(&ga.logits, &ga.logits_rb);
            } else {
                gpu.submit_no_readback();
            }
            chunk_start = chunk_end;
        }

        gpu.free_buffer(emb_buf, emb_mem);
        let ga = self.gpu_acts.as_ref().unwrap();
        let mut logits = gpu.read_logits(&ga.logits, &ga.logits_rb);
        apply_final_softcap(&mut logits, c.final_logit_softcap);
        logits
    }

    pub fn forward_gpu(&self, token: usize, pos: usize, gpu: &mut VkCtx) -> Vec<f32> {
        let c  = &self.config;
        let w  = &self.weights;
        let gw = self.gpu_w.as_ref().unwrap();
        let ga = self.gpu_acts.as_ref().unwrap();

        let mut emb = w.token_embd.get_row(token);
        if c.embd_scale != 1.0 { for v in emb.iter_mut() { *v *= c.embd_scale; } }

        gpu.begin();
        gpu.cmd_upload_act(&ga.x, &emb);
        gpu.timestamp();

        for l in 0..c.n_layers {
            self.record_layer_gpu(l, pos, gpu);
        }
        gpu.timestamp();

        gpu.cmd_rmsnorm(&ga.x, &ga.out_norm, &ga.xn, c.n_embd as u32, c.rms_norm_eps);
        gpu.barrier();
        if let Some(t) = gw.output.as_ref() {
            gpu.cmd_gemv(t, &ga.xn, &ga.logits);
        }
        gpu.timestamp();

        gpu.submit_with_readback(&ga.logits, &ga.logits_rb);
        gpu.print_timestamps();
        let mut logits = gpu.read_logits(&ga.logits, &ga.logits_rb);
        apply_final_softcap(&mut logits, c.final_logit_softcap);
        logits
    }

    pub fn forward_cpu(&self, token: usize, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let c   = &self.config;
        let w   = &self.weights;
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;

        let mut x      = w.token_embd.get_row(token);
        if c.embd_scale != 1.0 { for v in x.iter_mut() { *v *= c.embd_scale; } }
        let mut xn     = vec![0f32; c.n_embd];
        let mut q      = vec![0f32; c.n_heads * hd];
        let mut k      = vec![0f32; kvd];
        let mut v      = vec![0f32; kvd];
        let mut attn   = vec![0f32; c.n_heads * hd];
        let mut proj   = vec![0f32; c.n_embd];
        let mut gate   = vec![0f32; c.n_ff];
        let mut up     = vec![0f32; c.n_ff];
        let mut ff     = vec![0f32; c.n_embd];

        for l in 0..c.n_layers {
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.attn_norm[l], c.rms_norm_eps);
            if let Some(ref qw) = w.attn_q_f32[l] { f32_matvec(qw, &xn, &mut q, c.n_embd); }
            else { w.attn_q[l].matvec(&mut q, &xn); }
            if let Some(ref kw) = w.attn_k_f32[l] { f32_matvec(kw, &xn, &mut k, c.n_embd); }
            else { w.attn_k[l].matvec(&mut k, &xn); }
            if let Some(ref vw) = w.attn_v_f32[l] { f32_matvec(vw, &xn, &mut v, c.n_embd); }
            else { w.attn_v[l].matvec(&mut v, &xn); }
            if let Some(ref b) = w.attn_q_bias[l] { ops::add_into(&mut q, b); }
            if let Some(ref b) = w.attn_k_bias[l] { ops::add_into(&mut k, b); }
            if let Some(ref b) = w.attn_v_bias[l] { ops::add_into(&mut v, b); }
            if let Some(ref qn) = w.attn_q_norm[l] {
                for h in 0..c.n_heads { ops::rmsnorm(&mut q[h*hd..(h+1)*hd], qn, c.rms_norm_eps); }
            }
            if let Some(ref kn) = w.attn_k_norm[l] {
                for h in 0..c.n_kv_heads { ops::rmsnorm(&mut k[h*hd..(h+1)*hd], kn, c.rms_norm_eps); }
            }
            rope::apply_rope(&mut q, &mut k, pos, hd, c.rope_freq_base, c.n_heads, c.n_kv_heads);
            let cb = pos * kvd;
            cache.k[l][cb..cb+kvd].copy_from_slice(&k);
            cache.v[l][cb..cb+kvd].copy_from_slice(&v);
            let kv_ratio = c.n_heads / c.n_kv_heads;
            {
                use rayon::prelude::*;
                let scale = (hd as f32).sqrt();
                attn.par_chunks_mut(hd).enumerate().for_each(|(h, ah)| {
                    let kv_h = h / kv_ratio;
                    let qh   = &q[h*hd..(h+1)*hd];
                    let mut m: f32 = f32::NEG_INFINITY;
                    let mut lsum: f32 = 0.0;
                    ah.fill(0.0);
                    for p in 0..=pos {
                        let ko = p*kvd+kv_h*hd;
                        let mut s = qh.iter().zip(cache.k[l][ko..ko+hd].iter())
                                   .map(|(a,b)| a*b).sum::<f32>() / scale;
                        if let Some(cap) = c.attn_logit_softcap { s = cap * (s / cap).tanh(); }
                        let new_m = m.max(s);
                        let alpha = (m - new_m).exp();
                        let p_s   = (s - new_m).exp();
                        let vo = p*kvd+kv_h*hd;
                        for (o, vi) in ah.iter_mut().zip(cache.v[l][vo..vo+hd].iter()) {
                            *o = *o * alpha + p_s * vi;
                        }
                        lsum = lsum * alpha + p_s;
                        m = new_m;
                    }
                    let lsum = lsum.max(1e-12);
                    for o in ah.iter_mut() { *o /= lsum; }
                });
            }
            w.attn_out[l].matvec(&mut proj, &attn);
            if let Some(ref pn) = w.attn_post_norm[l] { ops::rmsnorm(&mut proj, pn, c.rms_norm_eps); }
            ops::add_into(&mut x, &proj);
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.ffn_norm[l], c.rms_norm_eps);
            if w.ffn_router[l].is_some() {
                self.moe_ffn_cpu(l, &xn, &mut ff);
            } else {
                w.ffn_gate[l].as_ref().unwrap().matvec(&mut gate, &xn);
                w.ffn_up[l].as_ref().unwrap().matvec(&mut up, &xn);
                if c.ffn_gelu {
                    for i in 0..c.n_ff { gate[i] = ops::gelu(gate[i]) * up[i]; }
                } else {
                    for i in 0..c.n_ff { gate[i] = ops::silu(gate[i]) * up[i]; }
                }
                w.ffn_down[l].as_ref().unwrap().matvec(&mut ff, &gate);
            }
            if let Some(ref pn) = w.ffn_post_norm[l] { ops::rmsnorm(&mut ff, pn, c.rms_norm_eps); }
            ops::add_into(&mut x, &ff);
        }
        ops::rmsnorm(&mut x, &w.output_norm, c.rms_norm_eps);
        let mut logits = vec![0f32; c.n_vocab];
        w.output.matvec(&mut logits, &x);
        apply_final_softcap(&mut logits, c.final_logit_softcap);
        logits
    }
}

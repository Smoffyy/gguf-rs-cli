use std::io::{self, Write};
use crate::model::{KvCache, LlamaModel};
use crate::tokenizer::bpe::Tokenizer;
use crate::gpu::VkCtx;
use crate::sampler::SampleParams;

pub struct GenerateOpts<'a> {
    pub max_tokens:     usize,
    pub min_tokens:      usize,
    pub ctx_len:          usize,
    pub sample:            SampleParams,
    pub stops:              &'a [u32],
    pub sys_ids:             &'a [u32],
    pub smart_context:        bool,
    pub prefill_batch:          usize,
}

pub fn rebuild_cache(
    model:        &LlamaModel,
    sys_ids:      &[u32],
    history:      &mut Vec<u32>,
    gpu:          &mut Option<VkCtx>,
    cpu_cache:    &mut KvCache,
    prefill_batch: usize,
) -> usize {
    let drop_n = (history.len() / 4).max(1).min(history.len());
    history.drain(..drop_n);

    let t0 = std::time::Instant::now();
    eprintln!("[Context: dropped {} tokens, replaying system({}) + history({})...]",
        drop_n, sys_ids.len(), history.len());

    if gpu.is_none() {
        let c = &model.config;
        for l in 0..c.n_layers { cpu_cache.k[l].fill(0.0); cpu_cache.v[l].fill(0.0); }
    }

    let (mut pos, _) = prefill_split(model, sys_ids, 0, gpu, cpu_cache, prefill_batch);
    for (i, &id) in history.iter().enumerate() {
        let _ = match gpu.as_mut() {
            Some(g) => model.forward_gpu(id as usize, pos + i, g),
            None    => model.forward_cpu(id as usize, pos + i, cpu_cache),
        };
    }
    pos += history.len();
    eprintln!("[Context: rebuilt in {}ms, pos now {}]",
        t0.elapsed().as_millis(), pos);
    pos
}

pub fn prefill_split(
    model:        &LlamaModel,
    ids:          &[u32],
    start:        usize,
    gpu:          &mut Option<VkCtx>,
    cpu_cache:    &mut KvCache,
    prefill_batch: usize,
) -> (usize, Vec<f32>) {
    if ids.is_empty() { return (start, vec![0f32; model.config.n_vocab]); }
    let logits = match gpu.as_mut() {
        Some(g) => {
            let tokens: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
            model.forward_gpu_prefill(&tokens, start, g, prefill_batch)
        }
        None => {
            let mut logits = vec![0f32; model.config.n_vocab];
            for (i, &id) in ids.iter().enumerate() {
                logits = model.forward_cpu(id as usize, start + i, cpu_cache);
            }
            logits
        }
    };
    (start + ids.len(), logits)
}

#[allow(clippy::too_many_arguments)]
pub fn generate_collect(
    model:      &LlamaModel,
    tok:        &Tokenizer,
    pos:        &mut usize,
    last:       &mut Vec<f32>,
    opts:       &GenerateOpts,
    gpu:        &mut Option<VkCtx>,
    cpu_cache:  &mut KvCache,
    recent:     &mut Vec<u32>,
    history:    &mut Vec<u32>,
    mirostat_mu: &mut f32,
) -> Vec<u32> {
    let mut generated: Vec<u32> = Vec::new();

    loop {
        let next = crate::sampler::sample(last, &opts.sample, recent, mirostat_mu);
        if generated.len() >= opts.min_tokens && opts.stops.contains(&(next as u32)) { break; }
        if generated.len() >= opts.max_tokens { break; }

        if *pos >= opts.ctx_len - 1 {
            if !opts.smart_context { break; }
            history.extend_from_slice(&generated);
            generated.clear();
            *pos = rebuild_cache(model, opts.sys_ids, history, gpu, cpu_cache, opts.prefill_batch);
            if let Some(&last_id) = history.last().or(opts.sys_ids.last()) {
                *last = match gpu.as_mut() {
                    Some(g) => model.forward_gpu(last_id as usize, *pos - 1, g),
                    None    => model.forward_cpu(last_id as usize, pos.saturating_sub(1), cpu_cache),
                };
            }
            continue;
        }

        let word = tok.decode(next as u32);
        if !word.is_empty() { print!("{}", word); io::stdout().flush().ok(); }

        recent.push(next as u32);
        if recent.len() > 64 { recent.remove(0); }
        generated.push(next as u32);

        *last = match gpu.as_mut() {
            Some(g) => model.forward_gpu(next, *pos, g),
            None    => model.forward_cpu(next, *pos, cpu_cache),
        };
        *pos += 1;
    }
    generated
}

pub fn extract_default_system(template: Option<&str>) -> Option<String> {
    let tmpl = template?;
    let mut sf = 0;
    while let Some(rel) = tmpl[sf..].find("<|im_start|>system\n") {
        let start = sf + rel + "<|im_start|>system\n".len();
        if let Some(end) = tmpl[start..].find("<|im_end|>") {
            let c = tmpl[start..start+end].trim();
            if !c.is_empty() && !c.contains("{{") && !c.contains("{%") && !c.contains("messages") {
                return Some(c.to_string());
            }
        }
        sf += rel + "<|im_start|>system\n".len();
    }
    for (open, close) in &[
        ("<<SYS>>\n",      "\n<</SYS>>"),
        ("<|system|>\n",   "<|end|>"),
        ("<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>"),
    ] {
        if let Some(start) = tmpl.find(open) {
            let after = &tmpl[start + open.len()..];
            if let Some(end) = after.find(close) {
                let c = after[..end].trim();
                if !c.is_empty() && !c.contains("{{") && !c.contains("messages") {
                    return Some(c.to_string());
                }
            }
        }
    }
    None
}

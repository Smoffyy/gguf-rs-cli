use std::collections::{HashSet, VecDeque};
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use crate::model::{KvCache, LlamaModel};
use crate::tokenizer::bpe::Tokenizer;
use crate::tokenizer::chat::ChatTemplate;
use crate::gpu::VkCtx;
use crate::sampler::{self, SampleParams};
use super::registry::ModelPreset;

pub enum ChatEvent {
    Token(String),
    Done { finish_reason: String, prompt_tokens: usize, completion_tokens: usize },
    Error(String),
}

pub struct ChatRequest {
    pub model_id: String,
    pub messages: Vec<(String, String)>,
    pub max_tokens: usize,
    pub sample: SampleParams,
    pub stop_strings: Vec<String>,
    pub tx: tokio::sync::mpsc::UnboundedSender<ChatEvent>,
}

#[derive(Clone)]
pub struct WorkerHandle {
    tx: Sender<ChatRequest>,
    pub registry: Arc<std::collections::HashMap<String, ModelPreset>>,
}

impl WorkerHandle {
    pub fn submit(&self, req: ChatRequest) {
        if self.tx.send(req).is_err() {
            eprintln!("[server] worker thread is gone");
        }
    }
}

pub fn spawn(registry: std::collections::HashMap<String, ModelPreset>) -> WorkerHandle {
    let registry = Arc::new(registry);
    let (tx, rx) = channel::<ChatRequest>();
    let reg2 = registry.clone();
    thread::spawn(move || worker_loop(reg2, rx));
    WorkerHandle { tx, registry }
}

struct LoadedModel {
    id: String,
    model: LlamaModel,
    tok: Tokenizer,
    tmpl: ChatTemplate,
    gpu: Option<VkCtx>,
    _clock_lock: crate::power::ClockLock,
    ctx_len: usize,
    parallel: usize,
}

struct Slot {
    slot_idx: usize,
    local_pos: usize,
    prompt_remaining: Vec<u32>,
    prompt_tokens: usize,
    last_logits: Vec<f32>,
    recent: Vec<u32>,
    sample: SampleParams,
    mirostat_mu: f32,
    max_tokens: usize,
    generated: usize,
    stops: Vec<u32>,
    stop_strings: Vec<String>,
    output_acc: String,
    cpu_kv: Option<KvCache>,
    finished: bool,
    tx: tokio::sync::mpsc::UnboundedSender<ChatEvent>,
}

fn load_model(preset: &ModelPreset) -> anyhow::Result<LoadedModel> {
    let mut gpu: Option<VkCtx> = if preset.gpu {
        match VkCtx::init() {
            Ok(g) => { eprintln!("[server] GPU ready for '{}'.", preset.id); Some(g) }
            Err(e) => { eprintln!("[server] No GPU for '{}' — using CPU. ({e})", preset.id); None }
        }
    } else { None };

    let clock_lock = match gpu.as_ref() {
        Some(g) => crate::power::ClockLock::engage(&g.device_name),
        None    => crate::power::ClockLock::none(),
    };

    let path = Path::new(&preset.model_path);
    let (model, gguf) = LlamaModel::load_with_slots(path, preset.ctx_len, preset.parallel, gpu.as_mut())?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let tmpl_str = gguf.metadata.get("tokenizer.chat_template")
        .and_then(|v| v.as_str()).map(|s| s.to_string());
    let tmpl = ChatTemplate::detect(&tok, tmpl_str.as_deref());
    let ctx_len = preset.ctx_len.min(model.config.n_ctx);

    Ok(LoadedModel {
        id: preset.id.clone(), model, tok, tmpl, gpu, _clock_lock: clock_lock,
        ctx_len, parallel: preset.parallel,
    })
}

fn worker_loop(registry: Arc<std::collections::HashMap<String, ModelPreset>>, rx: Receiver<ChatRequest>) {
    let mut current: Option<LoadedModel> = None;
    let mut slots: Vec<Slot> = Vec::new();
    let mut pending: VecDeque<ChatRequest> = VecDeque::new();

    if let Some(p) = registry.values().find(|p| p.load_on_startup) {
        eprintln!("[server] Preloading '{}'...", p.id);
        match load_model(p) {
            Ok(lm) => current = Some(lm),
            Err(e) => eprintln!("[server] Failed to preload '{}': {e}", p.id),
        }
    }

    loop {
        if slots.is_empty() && pending.is_empty() {
            match rx.recv() {
                Ok(req) => pending.push_back(req),
                Err(_) => break,
            }
        }
        while let Ok(req) = rx.try_recv() { pending.push_back(req); }

        admit(&registry, &mut current, &mut slots, &mut pending);

        if let Some(lm) = current.as_mut() {
            for slot in slots.iter_mut() {
                step_slot(lm, slot);
            }
        }
        slots.retain(|s| !s.finished);
    }
}

fn admit(
    registry: &Arc<std::collections::HashMap<String, ModelPreset>>,
    current: &mut Option<LoadedModel>,
    slots: &mut Vec<Slot>,
    pending: &mut VecDeque<ChatRequest>,
) {
    loop {
        let Some(front) = pending.front() else { return };
        if !registry.contains_key(&front.model_id) {
            let req = pending.pop_front().unwrap();
            let _ = req.tx.send(ChatEvent::Error(format!("model '{}' not found", req.model_id)));
            continue;
        }

        let need_switch = current.as_ref().map(|lm| lm.id != front.model_id).unwrap_or(true);
        if need_switch {
            if !slots.is_empty() { return; }
            let preset = registry.get(&front.model_id).unwrap().clone();
            eprintln!("[server] Switching to model '{}'...", preset.id);
            *current = None;
            match load_model(&preset) {
                Ok(lm) => { *current = Some(lm); }
                Err(e) => {
                    let req = pending.pop_front().unwrap();
                    let _ = req.tx.send(ChatEvent::Error(format!("failed to load model '{}': {e}", preset.id)));
                    continue;
                }
            }
        }

        let lm = current.as_ref().unwrap();
        if slots.len() >= lm.parallel { return; }

        let req = pending.pop_front().unwrap();
        match make_slot(lm, req, slots) {
            Ok(slot) => slots.push(slot),
            Err(()) => {}
        }
    }
}

fn make_slot(lm: &LoadedModel, req: ChatRequest, active: &[Slot]) -> Result<Slot, ()> {
    let used: HashSet<usize> = active.iter().map(|s| s.slot_idx).collect();
    let Some(slot_idx) = (0..lm.parallel).find(|i| !used.contains(i)) else {
        let _ = req.tx.send(ChatEvent::Error("no free slot".to_string()));
        return Err(());
    };
    let per_slot_ctx = (lm.ctx_len / lm.parallel).max(1);

    let prompt_text = lm.tmpl.render_conversation(&req.messages);
    let add_bos = lm.tmpl.uses_bos() || lm.tok.add_bos_token;
    let ids = lm.tok.encode(&prompt_text, add_bos);
    let prompt_tokens = ids.len();

    if prompt_tokens >= per_slot_ctx {
        let _ = req.tx.send(ChatEvent::Error(format!(
            "prompt too long for slot context: {prompt_tokens} tokens >= {per_slot_ctx} (ctx-size {} / parallel {})",
            lm.ctx_len, lm.parallel)));
        return Err(());
    }

    let mut stops = lm.tmpl.stop_tokens(&lm.tok);
    stops.sort_unstable(); stops.dedup();

    let cpu_kv = if lm.gpu.is_none() {
        Some(KvCache::new(lm.model.config.n_layers, per_slot_ctx,
            lm.model.config.n_kv_heads, lm.model.config.head_dim()))
    } else { None };

    let mirostat_mu = 2.0 * req.sample.mirostat_tau;
    let max_tokens = req.max_tokens.min(per_slot_ctx.saturating_sub(prompt_tokens).saturating_sub(1)).max(1);

    Ok(Slot {
        slot_idx, local_pos: 0,
        prompt_remaining: ids, prompt_tokens,
        last_logits: Vec::new(),
        recent: Vec::with_capacity(64),
        sample: req.sample, mirostat_mu,
        max_tokens, generated: 0,
        stops, stop_strings: req.stop_strings,
        output_acc: String::new(),
        cpu_kv, finished: false,
        tx: req.tx,
    })
}

fn step_slot(lm: &mut LoadedModel, slot: &mut Slot) {
    if slot.finished { return; }
    if slot.tx.is_closed() { slot.finished = true; return; }

    if !slot.prompt_remaining.is_empty() {
        let chunk_n = 32.min(slot.prompt_remaining.len());
        let chunk: Vec<usize> = slot.prompt_remaining.drain(..chunk_n).map(|t| t as usize).collect();
        let pos = slot.local_pos;
        let logits = match lm.gpu.as_mut() {
            Some(g) => lm.model.forward_gpu_prefill_slot(&chunk, pos, slot.slot_idx, g, 32),
            None => {
                let cache = slot.cpu_kv.as_mut().unwrap();
                let mut l = Vec::new();
                for (i, &t) in chunk.iter().enumerate() {
                    l = lm.model.forward_cpu(t, pos + i, cache);
                }
                l
            }
        };
        slot.local_pos += chunk_n;
        slot.last_logits = logits;
        return;
    }

    let next = sampler::sample(&mut slot.last_logits, &slot.sample, &slot.recent, &mut slot.mirostat_mu);

    let stop_hit = slot.stops.contains(&(next as u32));
    let len_hit  = slot.generated >= slot.max_tokens;

    if !stop_hit && !len_hit {
        let word = lm.tok.decode(next as u32);
        slot.output_acc.push_str(&word);
        let text_stop = slot.stop_strings.iter().any(|s| !s.is_empty() && slot.output_acc.ends_with(s.as_str()));
        if !word.is_empty() && !text_stop {
            let _ = slot.tx.send(ChatEvent::Token(word));
        }
        if text_stop {
            let _ = slot.tx.send(ChatEvent::Done {
                finish_reason: "stop".to_string(),
                prompt_tokens: slot.prompt_tokens, completion_tokens: slot.generated,
            });
            slot.finished = true;
            return;
        }
    }

    if stop_hit || len_hit {
        let _ = slot.tx.send(ChatEvent::Done {
            finish_reason: if stop_hit { "stop".to_string() } else { "length".to_string() },
            prompt_tokens: slot.prompt_tokens, completion_tokens: slot.generated,
        });
        slot.finished = true;
        return;
    }

    slot.recent.push(next as u32);
    if slot.recent.len() > 64 { slot.recent.remove(0); }
    slot.generated += 1;

    let pos = slot.local_pos;
    slot.last_logits = match lm.gpu.as_mut() {
        Some(g) => lm.model.forward_gpu_slot(next, pos, slot.slot_idx, g),
        None    => lm.model.forward_cpu(next, pos, slot.cpu_kv.as_mut().unwrap()),
    };
    slot.local_pos += 1;
}

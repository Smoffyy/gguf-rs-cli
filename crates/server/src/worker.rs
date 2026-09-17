//! The model worker: one thread, one resident model, several sequences in flight.
//!
//! A GPU context belongs to the thread that created it, and an `Engine` holds one, so the
//! model lives on a dedicated thread and requests reach it over a channel. Concurrency
//! comes from interleaving: the worker admits up to `parallel` requests, each with its own
//! KV-cache region, and advances every one of them by a token per round. A request that
//! arrives while another is mid-generation starts producing tokens immediately rather than
//! waiting for the first to finish.
//!
//! Only one model is resident at a time. Asking for a different one drains the in-flight
//! requests, drops the current model, and loads the new one - a 12 GB card cannot hold two
//! large models, and silently swapping mid-request would corrupt the ones in flight.

use std::sync::mpsc::{Receiver, Sender, TryRecvError};

use anyhow::{Context, Result};
use gguf_runtime::{Engine, EngineOptions, StopReason};
use gguf_sample::{SampleParams, Sampler};
use gguf_tokenizer::Detokenizer;

use crate::config::{Registry, ResolvedModel};

/// What a request emits as it runs.
#[derive(Debug)]
pub enum Event {
    /// A piece of decoded text.
    Chunk(String),
    /// Generation ended cleanly.
    Done { reason: StopReason, prompt_tokens: usize, generated: usize },
    /// Generation failed.
    Failed(String),
}

/// What to run. A chat job is rendered by the worker with the model's own template,
/// because the template lives with the tokenizer, which lives with the model.
pub enum JobInput {
    Chat(Vec<gguf_tokenizer::ChatMessage>),
    Raw(String),
}

pub struct Job {
    pub model: String,
    pub input: JobInput,
    pub max_tokens: usize,
    pub params: SampleParams,
    pub stop_strings: Vec<String>,
    pub events: Sender<Event>,
}

pub enum Command {
    Run(Box<Job>),
    /// Report which models exist and which is loaded.
    Status(Sender<Status>),
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct Status {
    pub models: Vec<String>,
    pub loaded: Option<String>,
    pub device: Option<String>,
    pub active: usize,
}

struct Slot {
    job: Job,
    sampler: Sampler,
    detok: Detokenizer,
    logits: Vec<f32>,
    pos: usize,
    generated: usize,
    prompt_tokens: usize,
    text: String,
    seq: usize,
}

pub struct Worker {
    registry: Registry,
    engine: Option<Engine>,
    loaded: Option<ResolvedModel>,
    slots: Vec<Slot>,
}

impl Worker {
    pub fn new(registry: Registry) -> Self {
        Self { registry, engine: None, loaded: None, slots: Vec::new() }
    }

    /// Run until the command channel closes.
    pub fn run(mut self, rx: Receiver<Command>) {
        // Preload anything marked for it, so the first request does not pay the load.
        if let Some(name) = self
            .registry
            .models
            .iter()
            .find(|(_, m)| m.preload)
            .map(|(n, _)| n.clone())
        {
            if let Err(e) = self.ensure_loaded(&name) {
                eprintln!("preloading {name} failed: {e:#}");
            }
        }

        loop {
            // Block only when there is nothing to advance.
            let next = if self.slots.is_empty() {
                match rx.recv() {
                    Ok(c) => Some(c),
                    Err(_) => return,
                }
            } else {
                match rx.try_recv() {
                    Ok(c) => Some(c),
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => return,
                }
            };

            if let Some(cmd) = next {
                match cmd {
                    Command::Shutdown => return,
                    Command::Status(reply) => {
                        let _ = reply.send(Status {
                            models: self.registry.names(),
                            loaded: self.loaded.as_ref().map(|m| m.name.clone()),
                            device: self.engine.as_ref().map(|e| e.device().name.clone()),
                            active: self.slots.len(),
                        });
                    }
                    Command::Run(job) => self.admit(*job),
                }
            }

            self.advance();
        }
    }

    fn ensure_loaded(&mut self, name: &str) -> Result<()> {
        if self.loaded.as_ref().is_some_and(|m| m.name == name) && self.engine.is_some() {
            return Ok(());
        }
        let spec = self
            .registry
            .resolve(name)
            .with_context(|| format!("no model named {name:?} in the registry"))?;

        // Drop the old model before loading the new one: two large models will not fit.
        self.engine = None;
        self.loaded = None;

        let opts = EngineOptions {
            device: spec.device.parse()?,
            n_ctx: spec.ctx,
            n_seq: spec.parallel.max(1),
            batch: spec.batch,
            threads: spec.threads,
            kv_type: if spec.kv_type == "f32" {
                gguf_core::DType::F32
            } else {
                gguf_core::DType::F16
            },
            enable_thinking: spec.enable_thinking,
        };
        let engine = Engine::load(&spec.path, &opts)
            .with_context(|| format!("loading {}", spec.path))?;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "loaded {name} on {} | {} | ctx {} x {} sequences | {:.2} GiB weights + {:.2} GiB kv",
            engine.device().name,
            engine.model.spec.summary(),
            engine.n_ctx,
            opts.n_seq,
            gib(engine.weights_bytes),
            gib(engine.kv_bytes)
        );
        self.engine = Some(engine);
        self.loaded = Some(spec);
        Ok(())
    }

    fn admit(&mut self, job: Job) {
        // Switching models requires the in-flight requests to finish first; they are using
        // the weights that are about to be unloaded.
        let needs_switch = self.loaded.as_ref().is_none_or(|m| m.name != job.model);
        if needs_switch && !self.slots.is_empty() {
            self.drain();
        }
        if let Err(e) = self.ensure_loaded(&job.model) {
            let _ = job.events.send(Event::Failed(format!("{e:#}")));
            return;
        }

        let Some(engine) = self.engine.as_mut() else {
            let _ = job.events.send(Event::Failed("no model is loaded".into()));
            return;
        };

        // Every admitted request needs its own cache region.
        let used: Vec<usize> = self.slots.iter().map(|s| s.seq).collect();
        let Some(seq) = (0..engine.n_seq()).find(|s| !used.contains(s)) else {
            let _ = job.events.send(Event::Failed(
                "every sequence slot is busy; raise `parallel` in the registry".into(),
            ));
            return;
        };

        let prompt = match &job.input {
            JobInput::Chat(messages) => engine.render_chat(messages),
            JobInput::Raw(text) => text.clone(),
        };
        let tokens = engine.tokenize_prompt(&prompt);
        if tokens.is_empty() {
            let _ = job.events.send(Event::Failed("the prompt is empty".into()));
            return;
        }
        if tokens.len() >= engine.n_ctx {
            let _ = job.events.send(Event::Failed(format!(
                "the prompt is {} tokens but the context is {}",
                tokens.len(),
                engine.n_ctx
            )));
            return;
        }

        engine.reset(seq);
        let (logits, prompt_tokens) = match engine.prefill(seq, &tokens) {
            Ok(v) => v,
            Err(e) => {
                let _ = job.events.send(Event::Failed(format!("{e}")));
                return;
            }
        };

        let mut sampler = Sampler::new(job.params);
        for &t in tokens.iter().rev().take(job.params.repeat_last_n) {
            sampler.accept(t);
        }

        self.slots.push(Slot {
            pos: tokens.len(),
            sampler,
            detok: Detokenizer::default(),
            logits,
            generated: 0,
            prompt_tokens,
            text: String::new(),
            seq,
            job,
        });
    }

    /// Advance every active request by one token.
    fn advance(&mut self) {
        let Some(engine) = self.engine.as_mut() else {
            return;
        };
        let mut finished = Vec::new();

        for (i, slot) in self.slots.iter_mut().enumerate() {
            let token = slot.sampler.sample(&mut slot.logits);

            if engine.tokenizer.is_eog(token) {
                finished.push((i, StopReason::EndOfGeneration));
                continue;
            }
            slot.sampler.accept(token);

            let piece = slot.detok.push(&engine.tokenizer.token_bytes(token, true));
            if !piece.is_empty() {
                slot.text.push_str(&piece);
                if slot.job.events.send(Event::Chunk(piece)).is_err() {
                    // The client is gone; stop spending the GPU on it.
                    finished.push((i, StopReason::Cancelled));
                    continue;
                }
            }

            if let Some(hit) = slot
                .job
                .stop_strings
                .iter()
                .find(|s| !s.is_empty() && slot.text.ends_with(*s))
            {
                let n = hit.len();
                slot.text.truncate(slot.text.len() - n);
                finished.push((i, StopReason::StopString));
                continue;
            }

            slot.generated += 1;
            if slot.generated >= slot.job.max_tokens {
                finished.push((i, StopReason::MaxTokens));
                continue;
            }
            if slot.pos + 1 >= engine.n_ctx {
                finished.push((i, StopReason::ContextFull));
                continue;
            }

            match engine.decode_step(slot.seq, token, slot.pos) {
                Ok(l) => {
                    slot.logits = l;
                    slot.pos += 1;
                }
                Err(e) => {
                    let _ = slot.job.events.send(Event::Failed(format!("{e}")));
                    finished.push((i, StopReason::Cancelled));
                }
            }
        }

        // Remove back to front so the earlier indices stay valid.
        for (i, reason) in finished.into_iter().rev() {
            let slot = self.slots.remove(i);
            let tail = {
                let mut d = slot.detok;
                d.flush()
            };
            if !tail.is_empty() {
                let _ = slot.job.events.send(Event::Chunk(tail));
            }
            let _ = slot.job.events.send(Event::Done {
                reason,
                prompt_tokens: slot.prompt_tokens,
                generated: slot.generated,
            });
        }
    }

    /// Run every in-flight request to completion.
    fn drain(&mut self) {
        while !self.slots.is_empty() {
            self.advance();
        }
    }
}

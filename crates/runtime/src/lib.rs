//! The engine: load a model onto a device, hold its KV cache, and generate.

pub mod device;
mod kv;

pub use device::{enumerate, DeviceSelector};
pub use kv::KvCache;

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use gguf_core::{Backend, DType, DeviceInfo, Error, Result};
use gguf_format::GgufModel;
use gguf_model::Model;
use gguf_sample::{SampleParams, Sampler};
use gguf_tokenizer::{ChatMessage, Detokenizer, Tokenizer};

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub device: DeviceSelector,
    /// Context length per sequence. Zero means the model's trained length, capped for sanity.
    pub n_ctx: usize,
    /// Independent sequences the cache must hold at once.
    pub n_seq: usize,
    /// Tokens per prefill chunk, which also sizes the activation scratch.
    pub batch: usize,
    pub threads: usize,
    /// KV cache storage. f16 halves the largest allocation after the weights for no
    /// observable quality cost, which is why it is the default.
    pub kv_type: DType,
    /// Pin the chat template's `enable_thinking`. `None` follows the model's own default.
    pub enable_thinking: Option<bool>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            device: DeviceSelector::Auto,
            n_ctx: 0,
            n_seq: 1,
            batch: 256,
            threads: 0,
            kv_type: DType::F16,
            enable_thinking: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub load_seconds: f64,
}

impl Stats {
    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_seconds <= 0.0 {
            0.0
        } else {
            self.prompt_tokens as f64 / self.prefill_seconds
        }
    }

    pub fn decode_tps(&self) -> f64 {
        if self.decode_seconds <= 0.0 {
            0.0
        } else {
            self.generated_tokens as f64 / self.decode_seconds
        }
    }
}

/// How generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndOfGeneration,
    MaxTokens,
    ContextFull,
    StopString,
    Cancelled,
}

pub struct GenerateOptions {
    pub max_tokens: usize,
    pub params: SampleParams,
    pub stop_strings: Vec<String>,
    pub seq: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            params: SampleParams::default(),
            stop_strings: Vec::new(),
            seq: 0,
        }
    }
}

pub struct Engine {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub info: DeviceInfo,
    pub notes: Vec<String>,
    pub n_ctx: usize,
    pub load_seconds: f64,
    pub weights_bytes: u64,
    pub kv_bytes: u64,
    enable_thinking: Option<bool>,
    backend: Box<dyn Backend>,
    kv: KvCache,
    batch: usize,
    /// Tokens already resident in each sequence's cache, for prefix reuse.
    resident: Vec<Vec<u32>>,
}

impl Engine {
    pub fn load(path: impl AsRef<Path>, opts: &EngineOptions) -> Result<Self> {
        let started = Instant::now();
        if opts.threads > 0 {
            gguf_backend_cpu::CpuBackend::set_threads(opts.threads);
        }

        let gguf = Arc::new(GgufModel::open(path)?);
        let (mut backend, notes) = device::open(opts.device)?;
        // The CPU backend reads weights straight out of the mapping; this is what keeps
        // that mapping alive for as long as the backend refers to it.
        backend.retain(gguf.keepalive());

        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let batch = opts.batch.clamp(1, 4096);
        let model = Model::load(&gguf, backend.as_mut(), batch)?;

        let trained = model.spec.n_ctx_train.max(512);
        let n_ctx = if opts.n_ctx == 0 { trained.min(8192) } else { opts.n_ctx };
        let n_seq = opts.n_seq.max(1);

        let kv = KvCache::new(
            backend.as_mut(),
            model.spec.n_layers,
            n_ctx,
            n_seq,
            model.spec.n_embd_k(),
            opts.kv_type,
        )?;

        Ok(Self {
            info: backend.info().clone(),
            weights_bytes: model.weights.bytes_uploaded,
            kv_bytes: kv.bytes(),
            enable_thinking: opts.enable_thinking,
            load_seconds: started.elapsed().as_secs_f64(),
            model,
            tokenizer,
            notes,
            n_ctx,
            backend,
            kv,
            batch,
            resident: vec![Vec::new(); n_seq],
        })
    }

    pub fn device(&self) -> &DeviceInfo {
        &self.info
    }

    /// Format a conversation the way this model expects it.
    pub fn render_chat(&self, messages: &[ChatMessage]) -> String {
        self.tokenizer
            .template
            .render_with(messages, true, self.enable_thinking)
    }

    pub fn tokenize_prompt(&self, text: &str) -> Vec<u32> {
        // A template that emits its own BOS must not get a second one from the tokenizer.
        let add_special = !self.tokenizer.template.template_adds_bos;
        self.tokenizer.encode(text, add_special)
    }

    pub fn reset(&mut self, seq: usize) {
        if let Some(r) = self.resident.get_mut(seq) {
            r.clear();
        }
    }

    /// Feed `tokens` into sequence `seq`, reusing whatever prefix is already cached.
    ///
    /// Returns the logits for the last token and how many tokens actually had to be run.
    ///
    /// Public so a scheduler can interleave several sequences: the blocking
    /// [`Engine::generate`] loop is only one way to drive the model, and a server needs to
    /// advance many requests a token at a time rather than run one to completion.
    pub fn prefill(&mut self, seq: usize, tokens: &[u32]) -> Result<(Vec<f32>, usize)> {
        let resident = self.resident[seq].clone();
        // The cached prefix is only reusable up to the first divergence, and never the
        // whole prompt: the last token must be run to produce logits.
        let mut shared = 0;
        while shared < resident.len()
            && shared + 1 < tokens.len()
            && resident[shared] == tokens[shared]
        {
            shared += 1;
        }

        let views = self.kv.views(seq);
        let mut logits = Vec::new();
        let mut pos = shared;
        let mut computed = 0;
        while pos < tokens.len() {
            let end = (pos + self.batch).min(tokens.len());
            let chunk = &tokens[pos..end];
            let positions: Vec<i32> = (pos..end).map(|p| p as i32).collect();
            logits = self.model.forward(
                self.backend.as_mut(),
                chunk,
                &positions,
                &views,
                pos as u32,
                end as u32,
            )?;
            computed += chunk.len();
            pos = end;
        }
        self.resident[seq] = tokens.to_vec();
        Ok((logits, computed))
    }

    /// Logits for the final position of `tokens`, with no sampling.
    ///
    /// The cache is cleared first so the result depends only on the input, which is what
    /// makes it usable as a backend comparison.
    pub fn logits_for(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.reset(0);
        let (logits, _) = self.prefill(0, tokens)?;
        Ok(logits)
    }

    /// Generate from a prompt, invoking `on_token` for each new token's text.
    ///
    /// `on_token` returning `false` stops generation, which is how a server cancels a
    /// request whose client has gone away.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        opts: &GenerateOptions,
        mut on_token: impl FnMut(&str, u32) -> bool,
    ) -> Result<(String, Stats, StopReason)> {
        let seq = opts.seq.min(self.kv.n_seq() - 1);
        let mut stats = Stats { load_seconds: self.load_seconds, ..Default::default() };

        if prompt.is_empty() {
            return Err(Error::Shape("cannot generate from an empty prompt".into()));
        }
        if prompt.len() >= self.n_ctx {
            return Err(Error::Shape(format!(
                "prompt is {} tokens but the context is {}; raise --ctx or shorten the prompt",
                prompt.len(),
                self.n_ctx
            )));
        }

        let t0 = Instant::now();
        let (mut logits, computed) = self.prefill(seq, prompt)?;
        stats.prefill_seconds = t0.elapsed().as_secs_f64();
        stats.prompt_tokens = computed;

        let mut sampler = Sampler::new(opts.params);
        for &t in prompt.iter().rev().take(opts.params.repeat_last_n) {
            sampler.accept(t);
        }

        let views = self.kv.views(seq);
        let mut out = String::new();
        let mut detok = Detokenizer::default();
        let mut pos = prompt.len();
        let reason;
        let t1 = Instant::now();

        loop {
            let token = sampler.sample(&mut logits);
            if self.tokenizer.is_eog(token) {
                reason = StopReason::EndOfGeneration;
                break;
            }

            sampler.accept(token);
            self.resident[seq].push(token);

            let text = detok.push(&self.tokenizer.token_bytes(token, true));
            if !text.is_empty() {
                out.push_str(&text);
                if !on_token(&text, token) {
                    reason = StopReason::Cancelled;
                    break;
                }
            }

            if let Some(hit) = opts.stop_strings.iter().find(|s| !s.is_empty() && out.ends_with(*s)) {
                out.truncate(out.len() - hit.len());
                reason = StopReason::StopString;
                break;
            }

            stats.generated_tokens += 1;
            if stats.generated_tokens >= opts.max_tokens {
                reason = StopReason::MaxTokens;
                break;
            }
            if pos + 1 >= self.n_ctx {
                reason = StopReason::ContextFull;
                break;
            }

            logits = self.model.forward(
                self.backend.as_mut(),
                &[token],
                &[pos as i32],
                &views,
                pos as u32,
                pos as u32 + 1,
            )?;
            pos += 1;
        }

        out.push_str(&detok.flush());
        stats.decode_seconds = t1.elapsed().as_secs_f64();
        Ok((out, stats, reason))
    }

    /// One decode step for `seq`: run `token` at `pos` and return the next logits.
    pub fn decode_step(&mut self, seq: usize, token: u32, pos: usize) -> Result<Vec<f32>> {
        let views = self.kv.views(seq);
        let logits = self.model.forward(
            self.backend.as_mut(),
            &[token],
            &[pos as i32],
            &views,
            pos as u32,
            pos as u32 + 1,
        )?;
        if let Some(r) = self.resident.get_mut(seq) {
            r.push(token);
        }
        Ok(logits)
    }

    pub fn n_seq(&self) -> usize {
        self.kv.n_seq()
    }

    pub fn kv_type(&self) -> DType {
        self.kv.dtype()
    }

    /// Time a fixed prefill and decode workload, for `gguf-rs bench`.
    pub fn benchmark(&mut self, prompt_len: usize, gen_len: usize) -> Result<Stats> {
        let bos = self.tokenizer.vocab.bos.unwrap_or(1);
        // Vary the tokens so the run cannot be helped by any degenerate-input shortcut.
        let prompt: Vec<u32> = std::iter::once(bos)
            .chain((1..prompt_len).map(|i| ((i * 7919) % self.model.spec.n_vocab.max(2)) as u32))
            .collect();

        self.reset(0);
        let t0 = Instant::now();
        let (mut logits, _) = self.prefill(0, &prompt)?;
        let prefill_seconds = t0.elapsed().as_secs_f64();

        let views = self.kv.views(0);
        let mut sampler = Sampler::new(SampleParams::greedy());
        let mut pos = prompt.len();
        let t1 = Instant::now();
        let mut generated = 0;
        while generated < gen_len && pos + 1 < self.n_ctx {
            let token = sampler.sample(&mut logits);
            logits = self.model.forward(
                self.backend.as_mut(),
                &[token],
                &[pos as i32],
                &views,
                pos as u32,
                pos as u32 + 1,
            )?;
            pos += 1;
            generated += 1;
        }

        Ok(Stats {
            prompt_tokens: prompt.len(),
            generated_tokens: generated,
            prefill_seconds,
            decode_seconds: t1.elapsed().as_secs_f64(),
            load_seconds: self.load_seconds,
        })
    }
}

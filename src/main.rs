mod gguf; mod tensor; mod model; mod math; mod tokenizer; mod sampler; mod gpu;

use std::io::{self, BufRead, Write};
use std::path::Path;
use anyhow::Result;
use clap::Parser;
use model::llama::{KvCache, LlamaModel};
use tokenizer::bpe::Tokenizer;
use tokenizer::chat::ChatTemplate;
use gpu::GpuCtx;

#[derive(Parser)]
#[command(name="gguf-cli", about="Pure Rust local LLM inference — 100% offline")]
struct Args {
    /// Path to .gguf model file
    #[arg(short, long)]
    model: String,
    /// Single prompt (non-interactive). Omit for chat mode.
    #[arg(short, long)]
    prompt: Option<String>,
    /// System prompt
    #[arg(short, long, default_value="You are a helpful assistant.")]
    system: String,
    /// Max tokens to generate per turn
    #[arg(short='n', long, default_value_t=512)]
    max_tokens: usize,
    /// Sampling temperature (0 = greedy)
    #[arg(short='t', long, default_value_t=0.7)]
    temperature: f32,
    /// Context window length (default 8192, capped at model max)
    #[arg(short='c', long, default_value_t=8192)]
    ctx_len: usize,
    /// Enable GPU acceleration (auto-detects DX12/Vulkan/Metal)
    #[arg(long, default_value_t=false)]
    gpu: bool,
    /// Random seed
    #[arg(long, default_value_t=42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    sampler::set_seed(args.seed);

    // Init GPU before loading so weights can upload immediately
    let gpu: Option<GpuCtx> = if args.gpu {
        match GpuCtx::init() {
            Some(g) => { eprintln!("GPU ready."); Some(g) }
            None    => { eprintln!("No GPU available — using CPU."); None }
        }
    } else { None };

    let path = Path::new(&args.model);
    let (model, gguf) = LlamaModel::load(path, gpu.as_ref())?;
    let tok = Tokenizer::from_gguf(&gguf)?;

    // Prefer the chat_template stored in the GGUF itself (Jinja2 string)
    let tmpl_hint = gguf.metadata.get("tokenizer.chat_template")
        .and_then(|v| v.as_str()).map(|s| s.to_string());
    let tmpl = ChatTemplate::detect(&tok, tmpl_hint.as_deref());
    eprintln!("Tokenizer: {:?} | Template: {:?}", tok.tok_model, tmpl);

    let c       = &model.config;
    let ctx_len = args.ctx_len.min(c.n_ctx);
    eprintln!("Context: {} tokens\n", ctx_len);

    let mut cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
    let stops     = tmpl.stop_tokens(&tok);

    match args.prompt {
        // ── Single-shot mode ──────────────────────────────────────────
        Some(ref p) => {
            print!("{}", p); io::stdout().flush()?;
            let ids = tok.encode(p, true);
            let (mut pos, mut logits) = prefill(&model, &mut cache, &ids, 0, gpu.as_ref());
            generate(&model, &tok, &mut cache, &mut pos, &mut logits,
                     args.max_tokens, args.temperature, ctx_len, &stops, gpu.as_ref());
            println!();
        }

        // ── Interactive chat mode ─────────────────────────────────────
        None => {
            eprintln!("Type your message. /quit to exit.");
            let mut pos = inject_system(&model, &mut cache, &tok, &tmpl, &args.system, 0, gpu.as_ref());
            let stdin = io::stdin();

            loop {
                eprint!("\nYou: "); io::stderr().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                let msg = line.trim();
                if msg.is_empty()                   { continue; }
                if msg == "/quit" || msg == "/exit" { break; }

                let turn = tmpl.user_turn(msg);
                let ids  = tok.encode(&turn, false);  // never BOS after first turn
                let (new_pos, mut logits) = prefill(&model, &mut cache, &ids, pos, gpu.as_ref());
                pos = new_pos;

                eprint!("Assistant: "); io::stderr().flush()?;
                generate(&model, &tok, &mut cache, &mut pos, &mut logits,
                         args.max_tokens, args.temperature, ctx_len, &stops, gpu.as_ref());
                println!();

                // Re-inject system prompt after reset so the model stays on-task
                if pos >= ctx_len.saturating_sub(64) {
                    eprintln!("[Context full — resetting]");
                    cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
                    pos   = inject_system(&model, &mut cache, &tok, &tmpl, &args.system, 0, gpu.as_ref());
                }
            }
        }
    }
    Ok(())
}

/// Inject system prompt into cache at `start_pos`, return next position
fn inject_system(model: &LlamaModel, cache: &mut KvCache, tok: &Tokenizer,
                 tmpl: &ChatTemplate, system: &str, start_pos: usize,
                 gpu: Option<&GpuCtx>) -> usize {
    let sys  = tmpl.system_prompt(system);
    if sys.is_empty() { return start_pos; }
    // ChatML (Qwen etc.) uses NO BOS token — template starts with <|im_start|>
    let ids  = tok.encode(&sys, tmpl.uses_bos());
    let (pos, _) = prefill(model, cache, &ids, start_pos, gpu);
    pos
}

fn prefill(model: &LlamaModel, cache: &mut KvCache, ids: &[u32], start: usize,
           gpu: Option<&GpuCtx>) -> (usize, Vec<f32>) {
    let mut logits = vec![0f32; model.config.n_vocab];
    for (i, &id) in ids.iter().enumerate() {
        logits = model.forward(id as usize, start + i, cache, gpu);
    }
    (start + ids.len(), logits)
}

fn generate(model: &LlamaModel, tok: &Tokenizer, cache: &mut KvCache,
            pos: &mut usize, last: &mut Vec<f32>, max: usize, temp: f32,
            ctx: usize, stops: &[u32], gpu: Option<&GpuCtx>) -> usize {
    let mut n = 0;
    loop {
        let next = sampler::sample(last, temp);
        if stops.contains(&(next as u32)) { break; }
        if *pos >= ctx - 1 || n >= max    { break; }
        print!("{}", tok.decode(next as u32)); io::stdout().flush().ok();
        *last = model.forward(next, *pos, cache, gpu);
        *pos += 1; n += 1;
    }
    n
}
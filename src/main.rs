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
    #[arg(short, long)]
    model: String,

    #[arg(short, long)]
    prompt: Option<String>,

    /// System prompt. If omitted, extracted from the model's chat template in the GGUF.
    #[arg(short, long)]
    system: Option<String>,

    /// Max tokens to generate per turn
    #[arg(short='n', long, default_value_t=512)]
    max_tokens: usize,

    /// Sampling temperature. Lower = more focused, 0 = greedy/deterministic.
    #[arg(short='t', long, default_value_t=0.7)]
    temperature: f32,

    /// Top-k: only sample from the k most likely tokens (0 = disabled)
    #[arg(long, default_value_t=40)]
    top_k: usize,

    /// Top-p nucleus sampling: keep tokens until cumulative probability >= p
    #[arg(long, default_value_t=0.9)]
    top_p: f32,

    /// Repetition penalty > 1.0 discourages the model from repeating itself
    #[arg(long, default_value_t=1.1)]
    rep_penalty: f32,

    /// Context window length (capped at model maximum)
    #[arg(short='c', long, default_value_t=8192)]
    ctx_len: usize,

    /// Enable GPU acceleration (auto-detects DX12 / Vulkan / Metal)
    #[arg(long, default_value_t=false)]
    gpu: bool,

    /// Print first 15 token IDs fed to the model (for debugging)
    #[arg(long, default_value_t=false)]
    debug_tokens: bool,

    #[arg(long, default_value_t=42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    sampler::set_seed(args.seed);

    let mut gpu: Option<GpuCtx> = if args.gpu {
        match GpuCtx::init() {
            Some(g) => { eprintln!("GPU ready."); Some(g) }
            None    => { eprintln!("No GPU — using CPU."); None }
        }
    } else { None };

    let path = Path::new(&args.model);
    let (model, gguf) = LlamaModel::load(path, gpu.as_ref())?;
    let tok = Tokenizer::from_gguf(&gguf)?;

    let model_name = gguf.metadata.get("general.name")
        .and_then(|v| v.as_str()).unwrap_or("Unknown");
    let arch = gguf.metadata.get("general.architecture")
        .and_then(|v| v.as_str()).unwrap_or("unknown");
    eprintln!("Model: {} ({})", model_name, arch);
    eprintln!("EOS tokens: {:?}", tok.eos_ids);

    let tmpl_str = gguf.metadata.get("tokenizer.chat_template")
        .and_then(|v| v.as_str()).map(|s| s.to_string());
    let tmpl = ChatTemplate::detect(&tok, tmpl_str.as_deref());

    let system = args.system.unwrap_or_else(|| {
        extract_default_system(tmpl_str.as_deref())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string())
    });

    eprintln!("Tokenizer: {:?} | Template: {:?} | add_bos: {}",
        tok.tok_model, tmpl, tok.add_bos_token);
    eprintln!("System: {}", &system[..system.len().min(80)]);
    eprintln!("Params: temp={} top_k={} top_p={} rep_penalty={}",
        args.temperature, args.top_k, args.top_p, args.rep_penalty);

    let c       = &model.config;
    let ctx_len = args.ctx_len.min(c.n_ctx);
    eprintln!("Context: {} tokens\n", ctx_len);

    let mut cache  = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
    let stops      = tmpl.stop_tokens(&tok);
    // Rolling window of recent tokens for repetition penalty (last 64)
    let mut recent: Vec<u32> = Vec::with_capacity(64);

    match args.prompt {
        Some(ref p) => {
            print!("{}", p); io::stdout().flush()?;
            let ids = tok.encode(p, true);
            let (mut pos, mut logits) = prefill(&model, &mut cache, &ids, 0,
                                                 gpu.as_mut(), args.debug_tokens);
            generate(&model, &tok, &mut cache, &mut pos, &mut logits,
                     args.max_tokens, args.temperature, args.top_k, args.top_p,
                     args.rep_penalty, ctx_len, &stops, gpu.as_mut(), &mut recent);
            println!();
        }
        None => {
            eprintln!("Type your message. /quit to exit.");
            let mut pos = inject_system(&model, &mut cache, &tok, &tmpl,
                                        &system, 0, gpu.as_mut(), args.debug_tokens);
            let stdin = io::stdin();
            loop {
                eprint!("\nYou: "); io::stderr().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                let msg = line.trim();
                if msg.is_empty()                   { continue; }
                if msg == "/quit" || msg == "/exit" { break; }

                let turn = tmpl.user_turn(msg);
                let ids  = tok.encode(&turn, false);
                if args.debug_tokens {
                    eprintln!("[user turn tokens: {:?}]", &ids[..ids.len().min(15)]);
                }
                let (new_pos, mut logits) = prefill(&model, &mut cache, &ids, pos,
                                                     gpu.as_mut(), false);
                pos = new_pos;

                eprint!("Assistant: "); io::stderr().flush()?;
                generate(&model, &tok, &mut cache, &mut pos, &mut logits,
                         args.max_tokens, args.temperature, args.top_k, args.top_p,
                         args.rep_penalty, ctx_len, &stops, gpu.as_mut(), &mut recent);
                println!();

                // Clear recent tokens between turns so penalty doesn't bleed across turns
                recent.clear();

                if pos >= ctx_len.saturating_sub(64) {
                    eprintln!("[Context full — resetting]");
                    cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
                    pos   = inject_system(&model, &mut cache, &tok, &tmpl,
                                         &system, 0, gpu.as_mut(), false);
                }
            }
        }
    }
    Ok(())
}

/// Extract the default system prompt from the Jinja chat template stored in GGUF metadata.
/// Does not hardcode any model-specific strings — just parses the template structure.
/// Works for any model that embeds a system block in their template.
fn extract_default_system(template: Option<&str>) -> Option<String> {
    let tmpl = template?;

    // Find <|im_start|>system\n...<|im_end|> — the system block used in ChatML-style templates.
    // The content between these markers is the default system prompt.
    // We skip occurrences that contain Jinja variables {{ }} since those are
    // conditional branches, not the hardcoded default.
    let sys_open  = "<|im_start|>system\n";
    let sys_close = "<|im_end|>";

    let mut search_from = 0;
    while let Some(rel) = tmpl[search_from..].find(sys_open) {
        let start   = search_from + rel + sys_open.len();
        let rest    = &tmpl[start..];
        if let Some(end) = rest.find(sys_close) {
            let content = rest[..end].trim();
            // Only use this block if it has no Jinja logic — it's a static default
            if !content.is_empty()
               && !content.contains("{{")
               && !content.contains("{%")
               && !content.contains("messages") {
                return Some(content.to_string());
            }
        }
        search_from += rel + sys_open.len();
    }

    // Fallback for templates that don't use <|im_start|> (LLaMA-3, Gemma, Phi etc.)
    // These embed the default system content as a literal string after markers like
    // "<<SYS>>\n", "<|system|>\n", or similar. We find the marker and extract until
    // the matching close tag.
    let fallback_pairs: &[(&str, &str)] = &[
        ("<<SYS>>\n",      "\n<</SYS>>"),
        ("<|system|>\n",   "<|end|>"),
        ("<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>"),
    ];
    for (open, close) in fallback_pairs {
        if let Some(start) = tmpl.find(open) {
            let after = &tmpl[start + open.len()..];
            if let Some(end) = after.find(close) {
                let content = after[..end].trim();
                if !content.is_empty()
                   && !content.contains("{{")
                   && !content.contains("messages") {
                    return Some(content.to_string());
                }
            }
        }
    }

    None
}

fn inject_system(model: &LlamaModel, cache: &mut KvCache, tok: &Tokenizer,
                 tmpl: &ChatTemplate, system: &str, start: usize,
                 gpu: Option<&mut GpuCtx>, debug: bool) -> usize {
    let sys = tmpl.system_prompt(system);
    if sys.is_empty() { return start; }
    let add_bos = tmpl.uses_bos() && tok.add_bos_token;
    let ids = tok.encode(&sys, add_bos);
    if debug {
        eprintln!("[system prompt text: {:?}]", &sys);
        eprintln!("[system token ids: {:?}]", &ids[..ids.len().min(15)]);
    }
    let (pos, _) = prefill(model, cache, &ids, start, gpu, false);
    pos
}

fn prefill(model: &LlamaModel, cache: &mut KvCache, ids: &[u32], start: usize,
           gpu: Option<&mut GpuCtx>, _debug: bool) -> (usize, Vec<f32>) {
    let mut logits = vec![0f32; model.config.n_vocab];
    let gptr = gpu.map(|g| g as *mut GpuCtx);
    for (i, &id) in ids.iter().enumerate() {
        logits = model.forward(id as usize, start + i, cache,
                               unsafe { gptr.map(|p| &mut *p) });
    }
    (start + ids.len(), logits)
}

fn generate(model: &LlamaModel, tok: &Tokenizer, cache: &mut KvCache,
            pos: &mut usize, last: &mut Vec<f32>,
            max: usize, temperature: f32, top_k: usize, top_p: f32,
            rep_penalty: f32, ctx: usize, stops: &[u32],
            gpu: Option<&mut GpuCtx>, recent: &mut Vec<u32>) -> usize {
    let mut n = 0;
    let gptr = gpu.map(|g| g as *mut GpuCtx);
    loop {
        let next = sampler::sample(last, temperature, top_k, top_p,
                                   rep_penalty, recent);
        if stops.contains(&(next as u32)) { break; }
        if *pos >= ctx - 1 || n >= max    { break; }

        let word = tok.decode(next as u32);
        if !word.is_empty() {
            print!("{}", word);
            io::stdout().flush().ok();
        }

        // Track recent tokens for repetition penalty (rolling window of 64)
        recent.push(next as u32);
        if recent.len() > 64 { recent.remove(0); }

        *last = model.forward(next, *pos, cache, unsafe { gptr.map(|p| &mut *p) });
        *pos += 1;
        n    += 1;
    }
    n
}
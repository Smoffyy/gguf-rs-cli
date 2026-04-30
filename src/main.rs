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
    #[arg(short='n', long, default_value_t=512)]
    max_tokens: usize,
    #[arg(short='t', long, default_value_t=0.7)]
    temperature: f32,
    #[arg(long, default_value_t=40)]
    top_k: usize,
    #[arg(long, default_value_t=0.9)]
    top_p: f32,
    #[arg(long, default_value_t=1.1)]
    rep_penalty: f32,
    #[arg(short='c', long, default_value_t=8192)]
    ctx_len: usize,
    #[arg(long, default_value_t=false)]
    gpu: bool,
    /// Smart context: slide window mid-generation so responses always complete.
    /// Default stops cleanly at context limit.
    #[arg(long, default_value_t=false)]
    smart_context: bool,
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

    let model_name = gguf.metadata.get("general.name").and_then(|v| v.as_str()).unwrap_or("Unknown");
    let arch = gguf.metadata.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("unknown");
    eprintln!("Model: {} ({})", model_name, arch);
    eprintln!("EOS tokens: {:?}", tok.eos_ids);

    let tmpl_str = gguf.metadata.get("tokenizer.chat_template")
        .and_then(|v| v.as_str()).map(|s| s.to_string());
    let tmpl = ChatTemplate::detect(&tok, tmpl_str.as_deref());

    let system = args.system.unwrap_or_else(|| {
        extract_default_system(tmpl_str.as_deref())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string())
    });

    eprintln!("Tokenizer: {:?} | Template: {:?} | add_bos: {}", tok.tok_model, tmpl, tok.add_bos_token);
    eprintln!("System: {}", &system[..system.len().min(80)]);
    eprintln!("Params: temp={} top_k={} top_p={} rep_penalty={} smart_context={}",
        args.temperature, args.top_k, args.top_p, args.rep_penalty, args.smart_context);

    let c       = &model.config;
    let ctx_len = args.ctx_len.min(c.n_ctx);
    eprintln!("Context: {} tokens\n", ctx_len);

    let mut cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
    let stops     = tmpl.stop_tokens(&tok);
    let mut recent: Vec<u32> = Vec::with_capacity(64);

    let sys_text = tmpl.system_prompt(&system);
    let sys_ids: Vec<u32> = if sys_text.is_empty() {
        vec![]
    } else {
        tok.encode(&sys_text, tmpl.uses_bos() && tok.add_bos_token)
    };

    match args.prompt {
        Some(ref p) => {
            print!("{}", p); io::stdout().flush()?;
            let ids = tok.encode(p, true);
            let (mut pos, mut logits) = prefill(&model, &mut cache, &ids, 0, gpu.as_mut(), args.debug_tokens);
            let mut dummy: Vec<u32> = Vec::new();
            generate_collect(&model, &tok, &mut cache, &mut pos, &mut logits,
                args.max_tokens, args.temperature, args.top_k, args.top_p,
                args.rep_penalty, ctx_len, &stops, gpu.as_mut(), &mut recent,
                &sys_ids, &mut dummy, args.smart_context);
            println!();
        }
        None => {
            eprintln!("Type your message. /quit to exit.");

            if args.debug_tokens {
                eprintln!("[system prompt text: {:?}]", &sys_text);
                eprintln!("[system token ids: {:?}]", &sys_ids[..sys_ids.len().min(15)]);
            }

            let (mut pos, _) = prefill(&model, &mut cache, &sys_ids, 0, gpu.as_mut(), false);
            let mut history: Vec<u32> = Vec::new();

            let stdin = io::stdin();
            loop {
                eprint!("\nYou: "); io::stderr().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                let msg = line.trim();
                if msg.is_empty()                   { continue; }
                if msg == "/quit" || msg == "/exit" { break; }

                let turn     = tmpl.user_turn(msg);
                let turn_ids = tok.encode(&turn, false);
                if args.debug_tokens {
                    eprintln!("[user turn tokens: {:?}]", &turn_ids[..turn_ids.len().min(15)]);
                }

                // Safe zone scales with context length: 1/4 of ctx_len, minimum 32.
                // Only applies when smart_context is enabled — otherwise use a flat 32.
                let lookahead = if args.smart_context {
                    (ctx_len / 4).max(32).min(args.max_tokens)
                } else {
                    32.min(args.max_tokens)
                };
                if pos + turn_ids.len() + lookahead >= ctx_len {
                    if history.is_empty() {
                        eprintln!("[Context: too small for this turn, clearing conversation]");
                        let kvd = c.n_kv_heads * c.head_dim();
                        for l in 0..c.n_layers {
                            cache.k[l][sys_ids.len() * kvd..].fill(0.0);
                            cache.v[l][sys_ids.len() * kvd..].fill(0.0);
                        }
                        pos = sys_ids.len();
                    } else {
                        pos = rebuild_cache(&model, &mut cache, &sys_ids, &mut history, ctx_len, gpu.as_mut());
                    }
                }

                let (new_pos, mut logits) = prefill(&model, &mut cache, &turn_ids, pos, gpu.as_mut(), false);
                pos = new_pos;
                history.extend_from_slice(&turn_ids);

                eprint!("Assistant: "); io::stderr().flush()?;
                let gen_ids = generate_collect(
                    &model, &tok, &mut cache, &mut pos, &mut logits,
                    args.max_tokens, args.temperature, args.top_k, args.top_p,
                    args.rep_penalty, ctx_len, &stops, gpu.as_mut(), &mut recent,
                    &sys_ids, &mut history, args.smart_context,
                );
                history.extend_from_slice(&gen_ids);
                println!();
                recent.clear();
            }
        }
    }
    Ok(())
}

fn rebuild_cache(
    model:    &LlamaModel,
    cache:    &mut KvCache,
    sys_ids:  &[u32],
    history:  &mut Vec<u32>,
    _ctx_len: usize,
    gpu:      Option<&mut GpuCtx>,
) -> usize {
    let drop_n = (history.len() / 2).max(1).min(history.len());
    history.drain(..drop_n);
    eprintln!("[Context: dropped {} tokens, rebuilding with {} history tokens]", drop_n, history.len());
    let c = &model.config;
    for l in 0..c.n_layers { cache.k[l].fill(0.0); cache.v[l].fill(0.0); }
    let gptr = gpu.map(|g| g as *mut GpuCtx);
    let mut pos = 0;
    for (i, &id) in sys_ids.iter().enumerate() {
        model.forward(id as usize, i, cache, unsafe { gptr.map(|p| &mut *p) });
        pos += 1;
    }
    for (i, &id) in history.iter().enumerate() {
        model.forward(id as usize, pos + i, cache, unsafe { gptr.map(|p| &mut *p) });
    }
    pos += history.len();
    pos
}

fn generate_collect(
    model:         &LlamaModel,
    tok:           &Tokenizer,
    cache:         &mut KvCache,
    pos:           &mut usize,
    last:          &mut Vec<f32>,
    max:           usize,
    temperature:   f32,
    top_k:         usize,
    top_p:         f32,
    rep_penalty:   f32,
    ctx:           usize,
    stops:         &[u32],
    gpu:           Option<&mut GpuCtx>,
    recent:        &mut Vec<u32>,
    sys_ids:       &[u32],
    history:       &mut Vec<u32>,
    smart_context: bool,
) -> Vec<u32> {
    let mut generated: Vec<u32> = Vec::new();
    let gptr = gpu.map(|g| g as *mut GpuCtx);

    loop {
        let next = sampler::sample(last, temperature, top_k, top_p, rep_penalty, recent);
        if stops.contains(&(next as u32)) { break; }
        if generated.len() >= max         { break; }

        if *pos >= ctx - 1 {
            if !smart_context { break; }

            // Smart context: save partial response, slide window, continue
            history.extend_from_slice(&generated);
            generated.clear();

            let c      = &model.config;
            let drop_n = (history.len() / 2).max(1).min(history.len());
            history.drain(..drop_n);
            for l in 0..c.n_layers { cache.k[l].fill(0.0); cache.v[l].fill(0.0); }

            let mut new_pos = 0;
            for (i, &id) in sys_ids.iter().enumerate() {
                model.forward(id as usize, i, cache, unsafe { gptr.map(|p| &mut *p) });
                new_pos += 1;
            }
            for (i, &id) in history.iter().enumerate() {
                model.forward(id as usize, new_pos + i, cache, unsafe { gptr.map(|p| &mut *p) });
            }
            new_pos += history.len();
            *pos = new_pos;

            if let Some(&last_tok) = history.last().or(sys_ids.last()) {
                *last = model.forward(last_tok as usize, pos.saturating_sub(1), cache,
                                      unsafe { gptr.map(|p| &mut *p) });
            }
            continue;
        }

        let word = tok.decode(next as u32);
        if !word.is_empty() { print!("{}", word); io::stdout().flush().ok(); }

        recent.push(next as u32);
        if recent.len() > 64 { recent.remove(0); }
        generated.push(next as u32);
        *last = model.forward(next, *pos, cache, unsafe { gptr.map(|p| &mut *p) });
        *pos += 1;
    }

    generated
}

fn prefill(model: &LlamaModel, cache: &mut KvCache, ids: &[u32], start: usize,
           gpu: Option<&mut GpuCtx>, _debug: bool) -> (usize, Vec<f32>) {
    let mut logits = vec![0f32; model.config.n_vocab];
    let gptr = gpu.map(|g| g as *mut GpuCtx);
    for (i, &id) in ids.iter().enumerate() {
        logits = model.forward(id as usize, start + i, cache, unsafe { gptr.map(|p| &mut *p) });
    }
    (start + ids.len(), logits)
}

fn extract_default_system(template: Option<&str>) -> Option<String> {
    let tmpl = template?;
    let mut search_from = 0;
    while let Some(rel) = tmpl[search_from..].find("<|im_start|>system\n") {
        let start = search_from + rel + "<|im_start|>system\n".len();
        if let Some(end) = tmpl[start..].find("<|im_end|>") {
            let content = tmpl[start..start+end].trim();
            if !content.is_empty() && !content.contains("{{") && !content.contains("{%") && !content.contains("messages") {
                return Some(content.to_string());
            }
        }
        search_from += rel + "<|im_start|>system\n".len();
    }
    for (open, close) in &[
        ("<<SYS>>\n", "\n<</SYS>>"),
        ("<|system|>\n", "<|end|>"),
        ("<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>"),
    ] {
        if let Some(start) = tmpl.find(open) {
            let after = &tmpl[start + open.len()..];
            if let Some(end) = after.find(close) {
                let content = after[..end].trim();
                if !content.is_empty() && !content.contains("{{") && !content.contains("messages") {
                    return Some(content.to_string());
                }
            }
        }
    }
    None
}
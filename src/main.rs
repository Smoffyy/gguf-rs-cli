mod gguf;
mod tensor;
mod model;
mod math;
mod tokenizer;
mod sampler;

use std::io::{self, BufRead, Write};
use std::path::Path;
use anyhow::Result;
use clap::Parser;
use model::llama::{KvCache, LlamaModel};
use tokenizer::bpe::Tokenizer;
use tokenizer::chat::ChatTemplate;

#[derive(Parser)]
#[command(name = "gguf-cli", about = "Pure Rust local LLM inference")]
struct Args {
    #[arg(short, long)]
    model: String,

    /// Single prompt (non-interactive). Omit to enter chat mode.
    #[arg(short, long)]
    prompt: Option<String>,

    /// System prompt for chat mode
    #[arg(short, long, default_value = "You are a helpful assistant.")]
    system: String,

    /// Max tokens to generate per turn
    #[arg(short = 'n', long, default_value_t = 512)]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(short = 't', long, default_value_t = 0.8)]
    temperature: f32,

    /// Context window size (default 8192, capped at model max)
    #[arg(short = 'c', long, default_value_t = 8192)]
    ctx_len: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    sampler::set_seed(args.seed);

    let path = Path::new(&args.model);
    let (model, gguf) = LlamaModel::load(path)?;
    let tok = Tokenizer::from_gguf(&gguf)?;

    eprintln!("Tokenizer: {:?}", if tok.token_to_id.contains_key("<|im_start|>") { "ChatML/Qwen" }
              else if tok.token_to_id.contains_key("<|eot_id|>") { "LLaMA-3" }
              else if tok.token_to_id.contains_key("[INST]") { "LLaMA-2" }
              else { "LLaMA/Sentencepiece" });

    let c = &model.config;
    // Use the smaller of user-requested ctx_len and model's trained max
    let ctx_len = args.ctx_len.min(c.n_ctx);
    eprintln!("Context: {} tokens", ctx_len);

    let mut cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());

    match args.prompt {
        // ── Single-shot mode ──────────────────────────────────────────────
        Some(ref prompt) => {
            print!("{}", prompt);
            io::stdout().flush()?;
            let ids = tok.encode(prompt, true);
            let (mut pos, mut logits) = prefill(&model, &mut cache, &ids, 0);
            generate(&model, &tok, &mut cache, &mut pos, &mut logits,
                     args.max_tokens, args.temperature, ctx_len, &[tok.eos_id]);
            println!();
        }

        // ── Interactive chat mode ─────────────────────────────────────────
        None => {
            let template = ChatTemplate::detect(&tok);
            let stop_ids = template.stop_tokens(&tok);
            eprintln!("Chat template: {:?} | /quit to exit\n", template);

            let mut pos: usize = 0;

            // Inject system prompt once
            let sys_text = template.system_prompt(&args.system);
            if !sys_text.is_empty() {
                let ids = tok.encode(&sys_text, true);
                let (new_pos, _) = prefill(&model, &mut cache, &ids, pos);
                pos = new_pos;
            }

            let stdin = io::stdin();
            loop {
                eprint!("\nYou: ");
                io::stderr().flush()?;

                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                let user_msg = line.trim();
                if user_msg.is_empty() { continue; }
                if user_msg == "/quit" || user_msg == "/exit" { break; }

                let turn = template.user_turn(user_msg);
                let ids  = tok.encode(&turn, pos == 0);
                let (new_pos, mut logits) = prefill(&model, &mut cache, &ids, pos);
                pos = new_pos;

                eprint!("Assistant: ");
                io::stderr().flush()?;

                generate(&model, &tok, &mut cache, &mut pos, &mut logits,
                         args.max_tokens, args.temperature, ctx_len, &stop_ids);
                println!();

                if pos >= ctx_len.saturating_sub(64) {
                    eprintln!("[Context full — resetting]");
                    cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
                    pos = 0;
                }
            }
        }
    }

    Ok(())
}

fn prefill(model: &LlamaModel, cache: &mut KvCache, ids: &[u32], start: usize) -> (usize, Vec<f32>) {
    let mut logits = vec![0f32; model.config.n_vocab];
    for (i, &id) in ids.iter().enumerate() {
        logits = model.forward(id as usize, start + i, cache);
    }
    (start + ids.len(), logits)
}

fn generate(
    model: &LlamaModel, tok: &Tokenizer, cache: &mut KvCache,
    pos: &mut usize, last_logits: &mut Vec<f32>,
    max_tokens: usize, temperature: f32, ctx_len: usize, stop_ids: &[u32],
) -> usize {
    let mut n = 0;
    loop {
        let next = sampler::sample(last_logits, temperature);
        if stop_ids.contains(&(next as u32)) { break; }
        if *pos >= ctx_len - 1 || n >= max_tokens { break; }

        let word = tok.decode(next as u32);
        print!("{}", word);
        io::stdout().flush().ok();

        *last_logits = model.forward(next, *pos, cache);
        *pos += 1;
        n    += 1;
    }
    n
}
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;
use anyhow::Result;
use clap::Parser;
use gguf_rs::model::{KvCache, LlamaModel};
use gguf_rs::tokenizer::bpe::Tokenizer;
use gguf_rs::tokenizer::chat::ChatTemplate;
use gguf_rs::gpu::VkCtx;
use gguf_rs::sampler::SampleParams;
use gguf_rs::chat::{extract_default_system, generate_collect, prefill_split, rebuild_cache, GenerateOpts};

#[derive(Parser)]
#[command(name="gguf-rs", about="Pure Rust local LLM inference engine with Vulkan GPU acceleration")]
struct Args {
    #[arg(short, long)]
    model: String,
    #[arg(short, long)]
    prompt: Option<String>,
    #[arg(short, long)]
    system: Option<String>,
    #[arg(short='n', long, default_value_t=512)]
    max_tokens: usize,
    #[arg(long, default_value_t=0)]
    min_tokens: usize,
    #[arg(short='t', long, default_value_t=0.7)]
    temperature: f32,
    #[arg(long, default_value_t=40)]
    top_k: usize,
    #[arg(long, default_value_t=0.9)]
    top_p: f32,
    #[arg(long, default_value_t=0.0)]
    min_p: f32,
    #[arg(long, default_value_t=1.1)]
    rep_penalty: f32,
    #[arg(long, default_value_t=0.0)]
    presence_penalty: f32,
    #[arg(long, default_value_t=0.0)]
    frequency_penalty: f32,
    #[arg(long, default_value_t=0, help="Mirostat sampling: 0 = off, 1 or 2 = mirostat v2 (overrides top-k/top-p/min-p)")]
    mirostat: u8,
    #[arg(long, default_value_t=5.0)]
    mirostat_tau: f32,
    #[arg(long, default_value_t=0.1)]
    mirostat_eta: f32,
    #[arg(short='c', long, default_value_t=8192)]
    ctx_len: usize,
    #[arg(long, default_value_t=false)]
    gpu: bool,
    #[arg(long, default_value_t=false, help="Skip chat template formatting; feed the prompt directly (raw completion)")]
    raw: bool,
    #[arg(long, default_value_t=false)]
    smart_context: bool,
    #[arg(long, default_value_t=false)]
    stats: bool,
    #[arg(long, default_value_t=false)]
    debug_tokens: bool,
    #[arg(long, default_value_t=false)]
    debug_gpu: bool,
    #[arg(long, default_value_t=42)]
    seed: u64,
    #[arg(long, default_value_t=512)]
    prefill_batch: usize,
    #[arg(long, default_value_t=0, help="CPU worker threads (0 = all logical cores)")]
    n_threads: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    gguf_rs::sampler::set_seed(args.seed);

    if args.n_threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(args.n_threads).build_global().ok();
    }

    let mut gpu: Option<VkCtx> = if args.gpu {
        match VkCtx::init() {
            Ok(mut g) => {
                g.debug_gpu = args.debug_gpu;
                eprintln!("GPU ready.");
                Some(g)
            }
            Err(e) => { eprintln!("No GPU — using CPU. ({e})"); None }
        }
    } else { None };

    let _clock_lock = match gpu.as_ref() {
        Some(g) => gguf_rs::power::ClockLock::engage(&g.device_name),
        None    => gguf_rs::power::ClockLock::none(),
    };

    let path = Path::new(&args.model);
    let (model, gguf) = LlamaModel::load(path, args.ctx_len, gpu.as_mut())?;
    let tok = Tokenizer::from_gguf(&gguf)?;

    let model_name = gguf.metadata.get("general.name").and_then(|v| v.as_str()).unwrap_or("Unknown");
    let arch       = gguf.metadata.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("unknown");
    eprintln!("Model: {} ({})", model_name, arch);
    eprintln!("EOS tokens: {:?}", tok.eos_ids);

    let tmpl_str = gguf.metadata.get("tokenizer.chat_template")
        .and_then(|v| v.as_str()).map(|s| s.to_string());
    let tmpl = ChatTemplate::detect(&tok, tmpl_str.as_deref());

    let system = args.system.unwrap_or_else(|| {
        extract_default_system(tmpl_str.as_deref())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string())
    });

    eprintln!("Tokenizer: {:?} | Template: {:?} | add_bos: {} | raw: {}",
        tok.tok_model, tmpl, tok.add_bos_token, args.raw);
    eprintln!("System: {}", &system[..system.len().min(80)]);
    eprintln!("Params: temp={} top_k={} top_p={} min_p={} rep_penalty={} presence={} frequency={} mirostat={} smart_context={} prefill_batch={}",
        args.temperature, args.top_k, args.top_p, args.min_p, args.rep_penalty,
        args.presence_penalty, args.frequency_penalty, args.mirostat, args.smart_context, args.prefill_batch);

    let sample = SampleParams {
        temperature: args.temperature, top_k: args.top_k, top_p: args.top_p, min_p: args.min_p,
        rep_penalty: args.rep_penalty, presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty, mirostat: args.mirostat,
        mirostat_tau: args.mirostat_tau, mirostat_eta: args.mirostat_eta,
    };
    let mut mirostat_mu = 2.0 * args.mirostat_tau;

    let c       = &model.config;
    let ctx_len = args.ctx_len.min(c.n_ctx);
    eprintln!("Context: {} tokens | Mode: {}\n",
        ctx_len, if gpu.is_some() { "GPU (Vulkan)" } else { "CPU" });

    let mut cpu_cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
    let stops         = if args.raw { tok.eos_ids.clone() } else { tmpl.stop_tokens(&tok) };
    let mut recent: Vec<u32> = Vec::with_capacity(64);

    let sys_text = if args.raw { String::new() } else { tmpl.system_prompt(&system) };
    let sys_ids: Vec<u32> = if sys_text.is_empty() { vec![] }
    else { tok.encode(&sys_text, tmpl.uses_bos() || tok.add_bos_token) };

    match args.prompt {
        Some(ref p) => {
            print!("{}", p); io::stdout().flush()?;
            let ids = tok.encode(p, true);
            let t0  = Instant::now();
            let (mut pos, mut logits) = prefill_split(&model, &ids, 0, &mut gpu, &mut cpu_cache, args.prefill_batch);
            let pm  = t0.elapsed().as_millis();
            let t1  = Instant::now();
            let mut dummy: Vec<u32> = Vec::new();
            let opts = GenerateOpts {
                max_tokens: args.max_tokens, min_tokens: args.min_tokens, ctx_len,
                sample, stops: &stops, sys_ids: &sys_ids,
                smart_context: args.smart_context, prefill_batch: args.prefill_batch,
            };
            let gen = generate_collect(&model, &tok, &mut pos, &mut logits, &opts,
                &mut gpu, &mut cpu_cache, &mut recent, &mut dummy, &mut mirostat_mu);
            println!();
            if args.stats {
                let gs = t1.elapsed().as_secs_f32();
                eprintln!("[Stats] prefill: {} tok in {}ms | generated: {} tok in {:.2}s ({:.1} tok/s)",
                    ids.len(), pm, gen.len(), gs, gen.len() as f32 / gs.max(0.001));
            }
        }
        None => {
            eprintln!("Type your message. /quit to exit.");

            if args.debug_tokens {
                eprintln!("[system prompt: {:?}]", &sys_text);
                eprintln!("[system ids: {:?}]", &sys_ids[..sys_ids.len().min(15)]);
            }

            let pt = Instant::now();
            let (mut pos, _) = prefill_split(&model, &sys_ids, 0, &mut gpu, &mut cpu_cache, args.prefill_batch);
            if args.stats {
                eprintln!("[Stats] system prefill: {} tok in {}ms",
                    sys_ids.len(), pt.elapsed().as_millis());
            }

            let mut history: Vec<u32> = Vec::new();
            let stdin = io::stdin();

            loop {
                eprint!("\nYou: "); io::stderr().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                let msg = line.trim();
                if msg.is_empty()                   { continue; }
                if msg == "/quit" || msg == "/exit" { break; }

                let turn     = if args.raw { msg.to_string() } else { tmpl.user_turn(msg) };
                let turn_ids = tok.encode(&turn, false);
                if args.debug_tokens {
                    eprintln!("[user turn ids: {:?}]", &turn_ids[..turn_ids.len().min(15)]);
                }

                let reserve = if args.smart_context {
                    (ctx_len / 8).max(32).min(args.max_tokens)
                } else {
                    32.min(args.max_tokens)
                };

                if pos + turn_ids.len() + reserve >= ctx_len {
                    if history.is_empty() {
                        eprintln!("[Context: too small, clearing conversation]");
                        let (p, _) = prefill_split(&model, &sys_ids, 0, &mut gpu, &mut cpu_cache, args.prefill_batch);
                        pos = p;
                    } else {
                        pos = rebuild_cache(&model, &sys_ids, &mut history,
                                            &mut gpu, &mut cpu_cache, args.prefill_batch);
                    }
                }

                let pt0 = Instant::now();
                let (new_pos, mut logits) = prefill_split(&model, &turn_ids, pos,
                                                           &mut gpu, &mut cpu_cache, args.prefill_batch);
                let pm = pt0.elapsed().as_millis();
                pos = new_pos;
                history.extend_from_slice(&turn_ids);

                eprint!("Assistant: "); io::stderr().flush()?;
                let gt0 = Instant::now();
                let opts = GenerateOpts {
                    max_tokens: args.max_tokens, min_tokens: args.min_tokens, ctx_len,
                    sample, stops: &stops, sys_ids: &sys_ids,
                    smart_context: args.smart_context, prefill_batch: args.prefill_batch,
                };
                let gen_ids = generate_collect(
                    &model, &tok, &mut pos, &mut logits, &opts,
                    &mut gpu, &mut cpu_cache, &mut recent, &mut history, &mut mirostat_mu,
                );
                history.extend_from_slice(&gen_ids);
                println!();

                if args.stats {
                    let gs = gt0.elapsed().as_secs_f32();
                    eprintln!("[Stats] prefill: {} tok in {}ms | generated: {} tok in {:.2}s ({:.1} tok/s) | ctx: {}/{}",
                        turn_ids.len(), pm,
                        gen_ids.len(), gs, gen_ids.len() as f32 / gs.max(0.001),
                        pos, ctx_len);
                }
                recent.clear();
            }
        }
    }
    Ok(())
}

use std::io::{BufRead, Write};

use anyhow::{Context, Result};
use gguf_runtime::{DeviceSelector, Engine, GenerateOptions, StopReason};
use gguf_tokenizer::ChatMessage;

use crate::{BenchArgs, RunArgs};

fn banner(engine: &Engine) {
    for note in &engine.notes {
        eprintln!("  {note}");
    }
    eprintln!(
        "  device    {} ({})",
        engine.device().name,
        engine.device().id()
    );
    eprintln!("  model     {}", engine.model.spec.summary());
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    eprintln!(
        "  memory    {:.2} GiB weights + {:.2} GiB kv ({}) = {:.2} GiB",
        gib(engine.weights_bytes),
        gib(engine.kv_bytes),
        engine.kv_type(),
        gib(engine.weights_bytes + engine.kv_bytes)
    );
    eprintln!(
        "  context   {} x {} sequence(s) | loaded in {:.1}s",
        engine.n_ctx,
        engine.n_seq(),
        engine.load_seconds
    );
    let t = &engine.tokenizer.template;
    if t.uses_gguf_template() {
        eprintln!("  template  from the model's own chat_template");
    } else if let Some(err) = &t.jinja_error {
        eprintln!("  template  built-in {:?} (model template unusable: {err})", t.builtin);
    } else {
        eprintln!("  template  built-in {:?}", t.builtin);
    }
    eprintln!();
}

pub fn run(args: RunArgs) -> Result<()> {
    let mut engine = Engine::load(&args.load.model, &args.load.options(1)?)
        .with_context(|| format!("loading {}", args.load.model))?;
    banner(&engine);

    let opts = GenerateOptions {
        max_tokens: args.max_tokens,
        params: args.sampler.params(),
        stop_strings: args.stop.clone(),
        seq: 0,
    };

    match &args.prompt {
        Some(prompt) => {
            let text = if args.raw {
                prompt.clone()
            } else {
                let mut messages = Vec::new();
                if let Some(s) = &args.system {
                    messages.push(ChatMessage::new("system", s));
                }
                messages.push(ChatMessage::new("user", prompt));
                engine.render_chat(&messages)
            };
            let tokens = engine.tokenize_prompt(&text);
            let (_, stats, reason) = engine.generate(&tokens, &opts, |chunk, _| {
                print!("{chunk}");
                let _ = std::io::stdout().flush();
                true
            })?;
            println!();
            if args.stats {
                report(&stats, reason);
            }
        }
        None => interactive(&mut engine, &args, &opts)?,
    }
    Ok(())
}

fn interactive(engine: &mut Engine, args: &RunArgs, opts: &GenerateOptions) -> Result<()> {
    eprintln!("Chat ready. Ctrl-C to quit, /reset to clear the conversation.\n");
    let mut history: Vec<ChatMessage> = Vec::new();
    if let Some(s) = &args.system {
        history.push(ChatMessage::new("system", s));
    }

    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/reset" {
            history.retain(|m| m.role == "system");
            engine.reset(0);
            eprintln!("(conversation cleared)");
            continue;
        }
        if line == "/quit" || line == "/exit" {
            return Ok(());
        }

        history.push(ChatMessage::new("user", line));
        let prompt = engine.render_chat(&history);
        let tokens = engine.tokenize_prompt(&prompt);
        let (reply, stats, reason) = engine.generate(&tokens, opts, |chunk, _| {
            print!("{chunk}");
            let _ = std::io::stdout().flush();
            true
        })?;
        println!();
        if args.stats {
            report(&stats, reason);
        }
        history.push(ChatMessage::new("assistant", reply));
    }
}

fn report(stats: &gguf_runtime::Stats, reason: StopReason) {
    eprintln!(
        "\n  {} prompt tokens in {:.2}s ({:.1} tok/s) | {} generated in {:.2}s ({:.1} tok/s) | {:?}",
        stats.prompt_tokens,
        stats.prefill_seconds,
        stats.prefill_tps(),
        stats.generated_tokens,
        stats.decode_seconds,
        stats.decode_tps(),
        reason
    );
}

pub fn bench(args: BenchArgs) -> Result<()> {
    let devices: Vec<String> = match &args.compare {
        Some(list) => list.split(',').map(|s| s.trim().to_string()).collect(),
        None => vec![args.load.device.clone()],
    };

    println!(
        "{:<10} {:<34} {:>12} {:>12}",
        "DEVICE", "NAME", "PREFILL t/s", "DECODE t/s"
    );
    for dev in devices {
        let mut load = args.load.clone();
        load.device = dev.clone();
        // A device that is not present should not abort a comparison run.
        let opts = match load.options(1) {
            Ok(o) => o,
            Err(e) => {
                println!("{dev:<10} {e}");
                continue;
            }
        };
        let mut engine = match Engine::load(&load.model, &opts) {
            Ok(e) => e,
            Err(e) => {
                println!("{dev:<10} unavailable: {e}");
                continue;
            }
        };
        let stats = engine.benchmark(args.prompt_len, args.gen_len)?;
        println!(
            "{:<10} {:<34} {:>12.1} {:>12.1}",
            engine.device().id(),
            truncate(&engine.device().name, 34),
            stats.prefill_tps(),
            stats.decode_tps()
        );
    }
    let _: DeviceSelector = DeviceSelector::Auto;
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n - 1).collect::<String>() + "…"
    }
}

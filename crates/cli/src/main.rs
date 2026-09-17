//! `gguf-rs` command line.

mod check;
mod inspect;
mod run;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use gguf_runtime::DeviceSelector;
use gguf_sample::SampleParams;

#[derive(Parser)]
#[command(
    name = "gguf-rs",
    version,
    about = "Local GGUF inference. Fully offline, CPU and GPU.",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate text, or chat interactively when no prompt is given.
    Run(RunArgs),
    /// Serve an OpenAI-compatible HTTP API.
    Serve(ServeArgs),
    /// Measure prefill and decode throughput.
    Bench(BenchArgs),
    /// Print what a GGUF declares and how this engine reads it.
    Inspect(InspectArgs),
    /// Compare a GPU backend against the CPU on a real model.
    Check(CheckArgs),
    /// List the devices this build can use.
    Devices,
}

#[derive(Args, Clone)]
pub struct LoadArgs {
    /// Path to a .gguf file.
    #[arg(short, long)]
    pub model: String,
    /// auto, cpu, cuda[:N] or vulkan[:N].
    #[arg(short, long, default_value = "auto")]
    pub device: String,
    /// Context length. 0 uses the model's trained length, capped at 8192.
    #[arg(short = 'c', long, default_value_t = 0)]
    pub ctx: usize,
    /// Tokens per prefill chunk.
    #[arg(short = 'b', long, default_value_t = 256)]
    pub batch: usize,
    /// CPU worker threads. 0 uses every logical core.
    #[arg(short = 't', long, default_value_t = 0)]
    pub threads: usize,
    /// KV cache storage: f16 halves it for no observable quality cost.
    #[arg(long, default_value = "f16", value_parser = ["f16", "f32"])]
    pub kv_type: String,
    /// Suppress a reasoning model's thinking block. Without this the model's own default applies.
    #[arg(long)]
    pub no_think: bool,
}

impl LoadArgs {
    pub fn options(&self, n_seq: usize) -> Result<gguf_runtime::EngineOptions> {
        Ok(gguf_runtime::EngineOptions {
            device: self.device.parse::<DeviceSelector>()?,
            n_ctx: self.ctx,
            n_seq,
            batch: self.batch,
            threads: self.threads,
            kv_type: if self.kv_type == "f32" {
                gguf_core::DType::F32
            } else {
                gguf_core::DType::F16
            },
            enable_thinking: if self.no_think { Some(false) } else { None },
        })
    }
}

#[derive(Args, Clone)]
pub struct SamplerArgs {
    #[arg(long, default_value_t = 0.7)]
    pub temp: f32,
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,
    #[arg(long, default_value_t = 0.9)]
    pub top_p: f32,
    #[arg(long, default_value_t = 0.05)]
    pub min_p: f32,
    #[arg(long, default_value_t = 1.0)]
    pub typical_p: f32,
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,
    #[arg(long, default_value_t = 64)]
    pub repeat_last_n: usize,
    #[arg(long, default_value_t = 0.0)]
    pub presence_penalty: f32,
    #[arg(long, default_value_t = 0.0)]
    pub frequency_penalty: f32,
    /// 0 off, 2 enables Mirostat v2 and overrides top-k/top-p/min-p.
    #[arg(long, default_value_t = 0)]
    pub mirostat: u8,
    #[arg(long, default_value_t = 5.0)]
    pub mirostat_tau: f32,
    #[arg(long, default_value_t = 0.1)]
    pub mirostat_eta: f32,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

impl SamplerArgs {
    pub fn params(&self) -> SampleParams {
        SampleParams {
            temperature: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            typical_p: self.typical_p,
            repeat_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            mirostat: self.mirostat,
            mirostat_tau: self.mirostat_tau,
            mirostat_eta: self.mirostat_eta,
            seed: self.seed,
        }
    }
}

#[derive(Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub load: LoadArgs,
    #[command(flatten)]
    pub sampler: SamplerArgs,
    /// One-shot prompt. Omit for an interactive chat.
    #[arg(short, long)]
    pub prompt: Option<String>,
    #[arg(short, long)]
    pub system: Option<String>,
    #[arg(short = 'n', long, default_value_t = 512)]
    pub max_tokens: usize,
    /// Feed the prompt verbatim, with no chat template.
    #[arg(long)]
    pub raw: bool,
    /// Print timing after generation.
    #[arg(long)]
    pub stats: bool,
    /// Extra stop strings, repeatable.
    #[arg(long = "stop")]
    pub stop: Vec<String>,
}

#[derive(Args)]
pub struct ServeArgs {
    /// TOML model registry. Without it, --model serves a single model.
    #[arg(short = 'f', long)]
    pub config: Option<String>,
    #[arg(short, long)]
    pub model: Option<String>,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(short, long, default_value_t = 8080)]
    pub port: u16,
    /// Concurrent sequences per loaded model.
    #[arg(long, default_value_t = 4)]
    pub parallel: usize,
    #[arg(short, long, default_value = "auto")]
    pub device: String,
    #[arg(short = 'c', long, default_value_t = 0)]
    pub ctx: usize,
    #[arg(short = 't', long, default_value_t = 0)]
    pub threads: usize,
}

#[derive(Args)]
pub struct BenchArgs {
    #[command(flatten)]
    pub load: LoadArgs,
    /// Prompt length to time prefill over.
    #[arg(long, default_value_t = 512)]
    pub prompt_len: usize,
    /// Tokens to time decode over.
    #[arg(long, default_value_t = 128)]
    pub gen_len: usize,
    /// Time each listed device in turn, e.g. --compare cpu,cuda,vulkan.
    #[arg(long)]
    pub compare: Option<String>,
}

#[derive(Args)]
pub struct CheckArgs {
    #[arg(short, long)]
    pub model: String,
    /// The backend to compare against the CPU.
    #[arg(short, long, default_value = "cuda")]
    pub device: String,
    #[arg(short, long)]
    pub prompt: Option<String>,
    /// Greedy tokens to generate on each backend.
    #[arg(short = 'n', long, default_value_t = 32)]
    pub tokens: usize,
    #[arg(short = 'c', long, default_value_t = 2048)]
    pub ctx: usize,
    #[arg(short = 'b', long, default_value_t = 256)]
    pub batch: usize,
}

#[derive(Args)]
pub struct InspectArgs {
    #[arg(short, long)]
    pub model: String,
    /// Also list every tensor.
    #[arg(long)]
    pub tensors: bool,
    /// Also dump all metadata keys.
    #[arg(long)]
    pub metadata: bool,
}

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    match Cli::parse().command {
        Command::Run(args) => run::run(args),
        Command::Bench(args) => run::bench(args),
        Command::Inspect(args) => inspect::inspect(args),
        Command::Check(args) => check::check(args),
        Command::Devices => {
            let devices = gguf_runtime::enumerate();
            if devices.is_empty() {
                println!("no usable devices");
                return Ok(());
            }
            println!("{:<10} {:<40} {:>10}", "DEVICE", "NAME", "MEMORY");
            for d in devices {
                println!(
                    "{:<10} {:<40} {:>7} MiB",
                    d.id(),
                    d.name,
                    d.total_memory / 1048576
                );
            }
            Ok(())
        }
        Command::Serve(args) => gguf_server::serve(gguf_server::ServeConfig {
            config: args.config,
            model: args.model,
            host: args.host,
            port: args.port,
            parallel: args.parallel,
            device: args.device,
            ctx: args.ctx,
            threads: args.threads,
        }),
    }
}

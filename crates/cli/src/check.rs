use anyhow::{Context, Result};
use gguf_runtime::{Engine, EngineOptions};

use crate::CheckArgs;

/// Compare a GPU backend against the CPU on real model weights.
///
/// Op-level conformance tests cover each kernel in isolation; this catches what they
/// cannot: an op used at a size or with a parameter combination the tests do not reach, and
/// the accumulation of small differences across every layer. Reporting the argmax and the
/// top-5 overlap rather than a bare error norm is what makes the result actionable, because
/// what matters is whether the two backends would pick the same token.
pub fn check(args: CheckArgs) -> Result<()> {
    let prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| "The capital of France is Paris, and the capital of Japan is".to_string());

    let mut cpu_opts = EngineOptions { n_ctx: args.ctx, batch: args.batch, ..Default::default() };
    cpu_opts.device = "cpu".parse()?;
    let mut cpu = Engine::load(&args.model, &cpu_opts).context("loading on the cpu")?;

    let mut gpu_opts = cpu_opts.clone();
    gpu_opts.device = args.device.parse()?;
    let mut gpu = Engine::load(&args.model, &gpu_opts)
        .with_context(|| format!("loading on {}", args.device))?;

    println!("reference  {}", cpu.device().name);
    println!("candidate  {}", gpu.device().name);
    println!("model      {}", cpu.model.spec.summary());

    let tokens = cpu.tokenize_prompt(&prompt);
    println!("prompt     {} tokens\n", tokens.len());

    // Prefill in one batch, then compare the logits both backends produce for the same
    // final position.
    let a = cpu.logits_for(&tokens)?;
    let b = gpu.logits_for(&tokens)?;

    let mut worst = 0f32;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }

    let top = |v: &[f32], k: usize| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_unstable_by(|&x, &y| v[y].total_cmp(&v[x]));
        idx.truncate(k);
        idx
    };
    let ta = top(&a, 5);
    let tb = top(&b, 5);
    let overlap = ta.iter().filter(|i| tb.contains(i)).count();

    let decode = |ids: &[usize]| -> String {
        ids.iter()
            .map(|i| format!("{:?}", cpu.tokenizer.decode(&[*i as u32], false)))
            .collect::<Vec<_>>()
            .join(" ")
    };

    println!("max |logit difference|   {worst:.3e} at token {at}");
    println!("argmax                   cpu {} / gpu {}  ({})",
        ta[0], tb[0], if ta[0] == tb[0] { "match" } else { "DIFFER" });
    println!("top-5 overlap            {overlap}/5");
    println!("  cpu top-5   {}", decode(&ta));
    println!("  gpu top-5   {}", decode(&tb));

    // A greedy continuation is the end-to-end question: do they write the same text?
    let gen = gguf_runtime::GenerateOptions {
        max_tokens: args.tokens,
        params: gguf_sample::SampleParams::greedy(),
        ..Default::default()
    };
    let (ca, _, _) = cpu.generate(&tokens, &gen, |_, _| true)?;
    let (cb, _, _) = gpu.generate(&tokens, &gen, |_, _| true)?;
    println!("\ngreedy continuation, {} tokens", args.tokens);
    println!("  cpu  {ca:?}");
    println!("  gpu  {cb:?}");
    if ca == cb {
        println!("\nbackends agree");
    } else {
        let common = ca.chars().zip(cb.chars()).take_while(|(x, y)| x == y).count();
        println!("\nbackends DIVERGE after {common} identical characters");
    }
    Ok(())
}

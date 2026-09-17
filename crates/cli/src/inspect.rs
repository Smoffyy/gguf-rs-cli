use std::collections::BTreeMap;

use anyhow::{Context, Result};
use gguf_core::DType;
use gguf_format::GgufModel;
use gguf_model::ArchSpec;
use gguf_tokenizer::Tokenizer;

use crate::InspectArgs;

/// Report what the file says and what the engine concluded from it.
///
/// This exists because architecture support here is inferred rather than tabulated: when a
/// new model behaves oddly, the first question is what the engine decided about it, and
/// that answer should not require a debugger.
pub fn inspect(args: InspectArgs) -> Result<()> {
    let g = GgufModel::open(&args.model).with_context(|| format!("opening {}", args.model))?;

    println!("file");
    println!("  path            {}", g.path.display());
    println!("  gguf version    {}", g.version);
    println!("  tensors         {}", g.tensors.len());
    println!("  metadata keys   {}", g.meta.len());
    println!(
        "  weights         {:.2} GiB, mostly {}",
        g.weights_bytes() as f64 / (1u64 << 30) as f64,
        g.dominant_dtype()
    );

    let mut by_type: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for t in g.tensors.values() {
        let e = by_type.entry(t.dtype.name()).or_default();
        e.0 += 1;
        e.1 += t.byte_size() as u64;
    }
    print!("  quantizations   ");
    println!(
        "{}",
        by_type
            .iter()
            .map(|(name, (n, bytes))| format!("{name} x{n} ({:.0} MiB)", *bytes as f64 / 1048576.0))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let unsupported: Vec<DType> = g.dtypes().into_iter().filter(|d| d.needs_codebook()).collect();
    if !unsupported.is_empty() {
        println!(
            "  UNSUPPORTED     {} (codebook quantizations this engine does not decode)",
            unsupported.iter().map(|d| d.name()).collect::<Vec<_>>().join(", ")
        );
    }

    println!("\narchitecture (inferred)");
    match ArchSpec::infer(&g) {
        Ok(s) => {
            println!("  declared        {}", s.arch);
            println!("  name            {}", s.name);
            println!("  layers          {}", s.n_layers);
            println!("  embedding       {}", s.n_embd);
            println!(
                "  heads           {} q, {} kv, head_dim {}",
                s.n_heads, s.n_kv_heads, s.head_dim
            );
            println!("  vocab           {}", s.n_vocab);
            println!("  trained ctx     {}", s.n_ctx_train);
            println!("  block style     {:?}", s.style);
            println!("  norm            {:?}, eps {}", s.norm_kind, s.norm_eps);
            println!(
                "  rope            {:?}, base {}, n_rot {}, scale {}",
                s.rope.kind, s.rope.freq_base, s.rope.n_rot, s.rope.freq_scale
            );
            if s.rope.sections.iter().any(|v| *v > 0) {
                println!("  rope sections   {:?}", s.rope.sections);
            }
            if s.rope.ext_factor != 0.0 {
                println!(
                    "  yarn            ext {}, attn {}, beta {}/{}, orig ctx {}",
                    s.rope.ext_factor, s.rope.attn_factor, s.rope.beta_fast, s.rope.beta_slow, s.rope.orig_ctx
                );
            }
            println!("  activation      {:?}, gated {}", s.act, s.gated_ffn);
            if let Some(moe) = &s.moe {
                println!(
                    "  moe             {} of {} experts, ff {}, gate {:?}, norm_topk {}",
                    moe.n_expert_used, moe.n_expert, moe.n_ff_exp, moe.gate_func, moe.norm_topk
                );
                if moe.n_ff_shexp > 0 {
                    println!("  shared expert   ff {}", moe.n_ff_shexp);
                }
            } else {
                println!("  feed-forward    {}", s.n_ff);
            }
            if s.swa_window > 0 {
                println!(
                    "  sliding window  {} on {} of every {} layers",
                    s.swa_window,
                    s.swa_every.saturating_sub(1).max(1),
                    s.swa_every
                );
            }
            for (label, on) in [
                ("qk norm", s.qk_norm),
                ("qkv bias", s.qkv_bias),
                ("post-attn norm", s.post_attn_norm),
                ("post-ffn norm", s.post_ffn_norm),
                ("attention sinks", s.attn_sinks),
                ("tied embeddings", s.tied_embeddings),
                ("rope_freqs tensor", s.has_rope_freqs),
            ] {
                if on {
                    println!("  {label:<15} yes");
                }
            }
            for (label, v) in [
                ("embd scale", s.embd_scale),
                ("logit scale", s.logit_scale),
                ("residual scale", s.residual_scale),
                ("attn softcap", s.attn_softcap),
                ("final softcap", s.final_softcap),
            ] {
                if v != 1.0 && v != 0.0 {
                    println!("  {label:<15} {v}");
                }
            }
        }
        Err(e) => println!("  could not infer: {e}"),
    }

    println!("\ntokenizer");
    match Tokenizer::from_gguf(&g) {
        Ok(t) => {
            println!("  kind            {:?}", t.vocab.kind);
            println!("  pre-tokenizer   {}", t.vocab.pre);
            println!("  tokens          {}", t.vocab.len());
            println!("  merges          {}", t.vocab.merge_rank.len());
            println!("  special tokens  {}", t.vocab.specials.len());
            println!(
                "  bos/eos         {:?} / {:?} (add_bos {})",
                t.vocab.bos, t.vocab.eos, t.vocab.add_bos
            );
            if t.template.uses_gguf_template() {
                println!("  chat template   from the file");
            } else if let Some(err) = &t.template.jinja_error {
                println!("  chat template   built-in {:?}; file template rejected: {err}", t.template.builtin);
            } else {
                println!("  chat template   built-in {:?} (file carries none)", t.template.builtin);
            }
            let rendered = t.template.render(
                &[
                    gguf_tokenizer::ChatMessage::new("user", "Hello"),
                ],
                true,
            );
            println!("  rendered sample {:?}", rendered);
            println!("  sample ids      {:?}", t.encode(&rendered, true));
        }
        Err(e) => println!("  unavailable: {e}"),
    }

    if args.metadata {
        println!("\nmetadata");
        for k in g.meta.sorted_keys() {
            if let Some(v) = g.meta.raw(k) {
                println!("  {k} = {}", v.summary());
            }
        }
    }

    if args.tensors {
        println!("\ntensors");
        for name in &g.order {
            if let Some(t) = g.info(name) {
                println!(
                    "  {:<40} {:<8} {:>14} {:>10.2} MiB",
                    name,
                    t.dtype.name(),
                    t.shape_string(),
                    t.byte_size() as f64 / 1048576.0
                );
            }
        }
    }
    Ok(())
}

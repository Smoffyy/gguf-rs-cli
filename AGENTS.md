# gguf-rs — architecture notes

Local GGUF inference. Pure Rust, no network access at build or run time, CPU and GPU.

Read this before changing anything. The layout is deliberate and a change in the wrong
place is usually a change in the wrong crate.

## The one idea

Everything above `crates/core` is written against the `Backend` trait. Everything below it
is a device that implements `Backend`. No code outside a backend crate knows what a GPU is,
and no backend knows what a transformer is.

The trait uses explicit methods rather than a generic op enum, so **adding a backend is a
compile error until every op exists**. That is the point: a missing kernel must not silently
fall back to something slower or wrong.

## Crates, bottom up

| Crate | Owns |
|---|---|
| `core` | `DType`, `Backend`, the op config structs, errors. No dependencies on anything below. |
| `gguf` | Container parsing: header, metadata, tensor directory, mmap. Hands out borrowed quantized byte ranges. |
| `quant` | Every ggml block layout: decode, activation quantization, integer dot products. The only place a block layout is written down for the CPU. |
| `backend-cpu` | Reference implementation. Correctness first; it is what the GPU backends are tested against. |
| `backend-cuda` | CUDA C kernels compiled to PTX at build time, driver API opened at run time. |
| `backend-vk` | GLSL compute shaders to SPIR-V, Vulkan 1.0, no optional extensions. |
| `model` | `ArchSpec` inference and one generic transformer graph. |
| `tokenizer` | SPM/BPE/WPM, pre-tokenizer regexes, and a Jinja subset for chat templates. |
| `sample` | Logit processing and token selection. |
| `runtime` | Device selection, KV cache, the generation loop. |
| `server` | OpenAI-compatible HTTP. |
| `cli` | The `gguf-rs` binary. |

## Things that are easy to get wrong

**Block layouts are law.** A wrong shift or scale in `quant/src/dequant.rs` does not crash;
it produces fluent, confident, wrong text. Any change there must keep
`crates/quant/tests/dot_matches_dequant.rs` passing, which checks the integer dot path
against the decode path for every type.

**RoPE has two incompatible layouts.** `Norm` rotates adjacent pairs, `Neox` rotates split
halves. Converters permute Q/K to match one of them. Which one a model uses is *not*
recoverable from the file, so it comes from `ROPE_NORM_ARCHS` in `model/src/spec.rs`; the
default for an unknown architecture is NeoX, which is correct for everything modern.

**Gemma's norm weights already have their `+1`.** `convert_hf_to_gguf.py` bakes it in. Do
not add it again in the norm op — that was a real bug, and it produced pure noise.

**The KV cache stores f16 by default.** `KvView.dtype` is the *storage* format; attention
reads and writes f32 either way. Any kernel touching the cache has an f32 and an f16 variant
(CUDA templates, Vulkan specialization constants), and the conformance tests cover both,
comparing `kv_write` on the stored bits rather than the decoded values.

**Do not pin `enable_thinking`.** Leaving it undefined is what makes a reasoning model
behave as shipped; setting it false makes Qwen-3's template emit an empty `<think></think>`
and suppress reasoning. It is a user-facing option (`--no-think`), not a default.

**Architecture support is inferred, not tabulated.** `ArchSpec::infer` reads which tensors a
block contains and which metadata keys are set. Prefer extending the inference over adding a
name to a list; a table entry only helps models that already exist.

## Testing

```
cargo test                                          # everything that needs no GPU
cargo test -p gguf-runtime --features cuda,vulkan --test backend_conformance -- --test-threads=1
gguf-rs check -m MODEL --device cuda                # real weights, CPU vs GPU
```

`backend_conformance` is the important one: it runs each op on the CPU and on every GPU
backend with identical inputs and compares by NMSE. Run it serially — each test opens its
own GPU context.

Tolerances there are not arbitrary. Exact ops must match bit for bit. Reductions differ by
f32 reassociation. Matmul is loosest because the backends genuinely disagree on how the
*activation* is represented: CPU and CUDA quantize it to Q8_1 so the dot product runs in
integers, Vulkan keeps it in f32. If a tolerance needs raising, find out why first — a
decode bug moves the NMSE by orders of magnitude, not by a factor of two.

## Deliberate limitations

- **IQ1/IQ2/IQ3** need ggml's built-in codebook grids. They are refused with a clear error
  rather than guessed at.
- **State-space and linear-attention hybrids** (`ssm_*`, `time_mix_*` tensors) are a
  different layer type, not a transformer variation. Refused explicitly.
- **MLA** (DeepSeek-V2/V3 attention) is refused explicitly.
- **No partial GPU offload yet.** A model either fits on the device or runs on the CPU.
  (On Windows the driver will page an oversized model to host memory and it will run, slowly.)
- NVIDIA's `cuda-oxide` and `cutile` Rust tracks are Linux-only; the CUDA backend here is
  CUDA C through the driver API so that one build covers Windows and Linux.

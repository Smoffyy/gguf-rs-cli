<img src="rust-logo.svg" width="72" align="right" alt="">

# gguf-rs

Local LLM inference for GGUF models. Pure Rust, CPU and GPU, **no network access at build
or run time**.

```
gguf-rs run     -m model.gguf -p "hello"   generate, or chat when no prompt is given
gguf-rs serve   -f models.toml             OpenAI-compatible HTTP API
gguf-rs bench   -m model.gguf              prefill and decode throughput
gguf-rs inspect -m model.gguf              what the file declares, and what the engine made of it
gguf-rs check   -m model.gguf              compare a GPU backend against the CPU
gguf-rs devices                            list what this build can run on
```

## Backends

| Device | How | Notes |
|---|---|---|
| CPU | AVX-512/AVX2 detected at run time, rayon over output rows | Always available. The correctness reference. |
| CUDA | CUDA C compiled to PTX at build time, driver API loaded at run time | No link-time CUDA dependency — the binary runs on machines with no CUDA installed. |
| Vulkan | GLSL to SPIR-V, Vulkan 1.0, no optional extensions | AMD, Intel, NVIDIA, Apple via MoltenVK, mobile. |

`--device auto` prefers CUDA, then Vulkan, then the CPU, and says why when it falls back.

### On NVIDIA's Rust CUDA tracks

NVIDIA's `cuda-oxide` (SIMT) and `cutile` (tile) are both **Linux-only** today, and need a
pinned nightly toolchain or CUDA 13.3 respectively. Neither can carry a cross-platform
engine yet. The CUDA backend here writes kernels in CUDA C, compiles them to PTX with
`nvcc` at build time, and opens `nvcuda.dll` / `libcuda.so.1` at run time — so one build
targets Windows and Linux and every GPU from Turing onward, and the driver JITs the PTX to
whatever card is actually present.

If `nvcc` is absent the CUDA backend compiles to a stub that reports itself unavailable;
the build does not fail.

## Model support

Architectures are **inferred from the file**, not looked up in a table. `ArchSpec::infer`
reads which tensors a block contains and which metadata keys are set, so a model the engine
has never seen runs as long as its block is shaped like a transformer block. What gets
detected: MoE routing and shared experts, QK-norm, post-attention and post-FFN norms,
sliding-window attention and its per-layer pattern, attention sinks, fused QKV and gate/up,
partial rotary, tied embeddings, logit softcapping, parallel-residual and post-norm blocks,
YaRN and Llama-3 rope scaling, M-RoPE sections.

Chat formatting comes from the model's own `tokenizer.chat_template`, rendered by a Jinja
subset interpreter, with built-in ChatML/Llama-3/Llama-2/Gemma/Phi-3 formats as a fallback.
`gguf-rs inspect` prints which one is in use.

Reasoning models keep their own default: the template variable `enable_thinking` is left
undefined unless you pass `--no-think`, so Qwen-3 and friends behave as shipped rather than
as this engine assumes.

**Quantizations:** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
Q6_K, IQ4_NL, IQ4_XS, MXFP4, TQ1_0, TQ2_0.

**Not supported**, refused with a clear message rather than guessed at:

- IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S — these depend on large codebook grids built into ggml.
- State-space and linear-attention hybrids (`ssm_*`, `time_mix_*`) — a different layer type.
- Multi-head latent attention (DeepSeek-V2/V3).

## Performance

RTX 3080, 24-core x86-64, Qwen3-1.7B Q4_K_M, 512-token prompt:

| Device | Prefill tok/s | Decode tok/s |
|---|---|---|
| CPU | 30 | 15 |
| Vulkan | 85 | 43 |
| CUDA | 562 | 138 |

Where the speed comes from:

- **Weights keep their GGUF block layout in device memory.** Nothing is expanded to f32, so
  VRAM equals file size and the upload is a memcpy. Kernels decode blocks inline.
- **Quantized activations.** CPU and CUDA stage activations to Q8_1 and run the dot product
  in integers (`__dp4a`, `vpdpbusd`).
- **Coalesced warp-cooperative matmul.** Lane *l* takes the *l*-th word of one block's quant
  region, so a warp reads 128 contiguous bytes per instruction.
- **Split-K flash decoding.** The key/value range is split across blocks so a one-token
  decode step fills the GPU instead of launching one block per head. This was worth 3x.
- **MoE stays on the GPU** — routing, expert matmuls and reduction, with no host round trip.

### Memory

`gguf-rs run` prints the breakdown. Qwen3-8B at 8192 context:

```
memory    4.68 GiB weights + 1.12 GiB kv (F16) = 5.80 GiB
```

The KV cache defaults to f16, which halves it for no observable quality cost — measured on
an RTX 3080, an 8B model at 8k context goes from 7.5 GiB to 6.3 GiB resident. `--kv-type f32`
restores f32 storage if you want output bit-comparable with an f32 reference. Context length
is the lever that matters: the cache grows linearly with it, and with `parallel` in the
server.

## Server

```toml
# models.toml
[defaults]
device   = "auto"
ctx      = 8192
parallel = 4        # concurrent sequences per loaded model

[models.qwen]
path    = "model/path"
preload = true

[models.gemma]
path        = "model/path"
temperature = 0.4
```

```
gguf-rs serve -f models.toml --port 8080
```

`GET /v1/models`, `POST /v1/chat/completions` (streaming and not), `POST /v1/completions`,
`GET /health`. One model is resident at a time; requesting another drains the in-flight
requests and swaps. Concurrent requests get their own KV-cache region and are advanced a
token at a time in round-robin, so a request arriving mid-generation starts producing
immediately.

## Building

```
cargo build --release
```

CUDA needs `nvcc` (found via `CUDA_PATH` or the usual install roots; on Windows the build
also locates an MSVC host compiler, trying each installed version because CUDA rejects ones
newer than itself). Vulkan needs nothing beyond the crate's `shaderc`. Set `GGUF_RS_NO_CUDA`
to skip the CUDA backend. Either GPU backend can be dropped with
`--no-default-features --features vulkan` or similar.

## Correctness

`cargo test -p gguf-runtime --features cuda,vulkan --test backend_conformance -- --test-threads=1`
runs every op on the CPU and on each GPU backend with identical inputs and compares by NMSE.
`gguf-rs check -m MODEL --device cuda` does the same on real weights and reports whether the
two backends pick the same tokens.

The backends are not bit-identical and cannot be: f32 addition is not associative, and CPU
and CUDA quantize activations to 8 bits where Vulkan does not. On a real model the logits
agree to a few parts in a thousand and the top-5 tokens match; greedy decoding can still
diverge on a near-tie.

## License

MIT.

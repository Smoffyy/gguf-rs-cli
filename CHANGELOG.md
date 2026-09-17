# Changelog
All notable changes will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026-09-16
### Added
KV cache stores f16 by default, halving the largest allocation after the weights. Qwen3-8B
at 8192 context drops from 7.5 GiB to 6.3 GiB resident. `--kv-type f32` restores the old
storage. `gguf-rs run` now prints the weights/KV/total breakdown.

`--no-think`, and `enable_thinking` in the server registry.

### Fixed
Chat templates using an extended slice (`messages[::-1]`, which Qwen3 uses to find the last
user turn) were rejected, silently falling back to built-in ChatML. The Jinja parser now
handles a slice step, including negative ones.

`enable_thinking` was pinned to false, which made Qwen3's template emit an empty
`<think></think>` pair and suppressed reasoning. It is now left undefined so the model's own
default applies.

## [3.0.0] - 2026-09-16
### Changed
Complete rewrite. The single crate is now a workspace of twelve, split around one
`Backend` trait: everything above it is device-agnostic, everything below it is a device.
The trait uses explicit methods, so adding a backend fails to compile until every op exists
rather than silently falling back.

### Added
Native CUDA backend. Kernels are CUDA C compiled to PTX at build time and driven through
the driver API loaded at run time, so there is no link-time CUDA dependency and one build
covers Windows and Linux on any GPU from Turing onward. NVIDIA's own Rust CUDA tracks
(cuda-oxide, cutile) are Linux-only and could not be used.

Vulkan backend rewritten against native GGUF block layouts, Vulkan 1.0 with no optional
extensions, one shader per kernel with the quantization as a specialization constant.

Architecture inference replaces the hardcoded graph: `ArchSpec::infer` reads which tensors a
block contains and which metadata keys are set, so unseen architectures run. Detects MoE and
shared experts, QK-norm, post-norms, sliding windows and their per-layer pattern, attention
sinks, fused QKV and gate/up, partial rotary, tied embeddings, softcapping,
parallel-residual and post-norm blocks, YaRN and Llama-3 scaling, M-RoPE sections.

Jinja subset interpreter for `tokenizer.chat_template`, so chat formatting comes from the
model rather than a maintained table of six formats.

Quantization coverage: BF16, IQ4_NL, IQ4_XS, MXFP4, TQ1_0, TQ2_0 alongside the existing set.
The codebook-grid types (IQ1/IQ2/IQ3) are refused with a clear message instead of guessed at,
as are state-space hybrids and MLA.

`gguf-rs inspect`, `check`, `bench` and `devices`; TOML model registry replacing models.ini;
op-level cross-backend conformance tests.

### Fixed
RoPE only implemented the NeoX split-half layout, which is wrong for LLaMA, Mistral and the
rest of that family — they need the NORM adjacent-pair layout. Both are now implemented and
selected per architecture.

Gemma produced noise: the engine applied `x * (1 + w)` to norm weights, but the GGUF
converter already bakes that `+1` in, so it was counted twice.

MoE layers no longer drop to the CPU mid-forward; routing, expert matmuls and reduction all
run on the device.

Matmul parallelised over tokens, which left every core but one idle during decode. Now over
output rows: 6.5x on CPU.

Attention launched one block per head, so a decode step used a fraction of the GPU while
each block walked the history serially. Split-K flash decoding: 3x.

### Performance
RTX 3080, Qwen3-1.7B Q4_K_M, 512-token prompt. Prefill/decode tok/s: CPU 30/15,
Vulkan 85/43, CUDA 562/138.

## [2.1.0] - 2026-09-15
### Added
`gguf-rs-server`: an OpenAI-compatible local HTTP server (`GET /v1/models`, `POST
/v1/chat/completions`, streaming and non-streaming), configured via a `models.ini`
registry in the same shape as llama.cpp's `--models-preset` router file. Just-in-time
model loading, only one model resident at a time (switching drains in-flight requests
on the old model, unloads it, loads the new one), and real concurrent request handling:
each loaded model gets `parallel` independent slots — round-robin scheduled on a
dedicated worker thread, each with its own KV-cache region — instead of a single
request queue.

Required giving each parallel slot genuine memory isolation rather than a shared-buffer
position offset: `GpuActs.k_cache`/`v_cache` became `Vec<Vec<ActBuf>>` (slot × layer),
`LlamaModel::load_with_slots()` allocates one independent KV-cache set per slot sized
`ctx-size / parallel`, and `forward_gpu`/`forward_gpu_prefill` gained slot-aware
variants. An earlier position-offset design was caught in testing: two concurrent
requests produced correct output individually but garbled output when run in parallel,
because the attention read range wasn't scoped to each slot's own window.

Also fixed in the process: BOS-token gating used `template_needs_bos && add_bos_token`
(AND) instead of `template_needs_bos || add_bos_token` (OR), which suppressed BOS for
Gemma even though its GGUF metadata declares `add_bos_token=true` — silently producing
empty completions for any Gemma chat turn that included a system prompt. Affected the
CLI too, not just the new server.

Added `render_conversation()` to `ChatTemplate` for rendering a full OpenAI-style
message history (not just a single system+user turn) per template.

## [2.0.0] - 2026-09-15
### Fixed
Garbled output on every model regardless of quantization. The Q2_K/Q3_K/Q4_K/Q5_K/Q6_K
dequantizers (CPU and GPU shaders) used an invented flat bit-layout instead of the real
interleaved ggml block format, and Q5_0/Q5_1 read the high-bit mask from the wrong bit
positions. All rewritten to match the reference layout.

GPU mode silently skipping any weight tensor whose quant type had no native GPU shader
(Q5_0, Q5_1, Q2_K, etc.) instead of actually computing it — the advertised "CPU rayon
fallback" never existed. `VkCtx::upload_any()` now dequantizes to f32 and uploads that
when no native shader applies, so every tensor is actually computed.

### Added
Automatic NVIDIA GPU clock lock (`src/power.rs`) — locks clocks to the GPU's max boost
clock for the duration of a `--gpu` run via `nvidia-smi -lgc`, restored on exit. Requires
running as Administrator; falls back to driver defaults with a warning otherwise.

Qwen3 support — per-head QK-RMSNorm (`attn_q_norm`/`attn_k_norm`) applied before RoPE,
on both CPU and GPU (new `qk_norm.glsl` shader).

### Added
Flash-Attention-2-style attention on both GPU and CPU: a single streaming pass with
online-softmax rescaling (tiled, shared-memory-staged on GPU) replacing the old 3-pass
score-buffer approach. Verified numerically consistent across head_dim 64/128/256 and
across single-tile and multi-tile (100+ token) sequences.

Mixture-of-experts support (`qwen3moe` and compatible GGUFs): per-token top-k expert
routing with softmax-then-renormalize weighting, matching the standard Qwen3-MoE/Mixtral
convention. Expert weights stay CPU-resident (a 30B-A3B model's experts alone are ~17GB
quantized — far past consumer VRAM) while attention and shared layers still run on GPU
under `--gpu`, round-tripping the residual stream through CPU once per MoE layer.

Gemma 3 support — verified against a real checkpoint; needed no new code since its
differences from Gemma 2 (QK-RMSNorm instead of logit softcapping, tighter sliding
window) were already covered by the existing generic tensor-presence-based loading.

MoE verified end-to-end against a real Qwen3-30B-A3B checkpoint (128 experts, top-8
routing): CPU and GPU-hybrid paths produce identical output under greedy decoding, and
the GPU path correctly uploads only attention/shared weights (193/337 tensors) while
leaving all 144 expert tensors CPU-resident.

New CLI options: `--min-p`, `--presence-penalty`, `--frequency-penalty`, `--mirostat`/
`--mirostat-tau`/`--mirostat-eta`, `--min-tokens`, `--raw` (skip chat template), and
`--n-threads` (CPU worker pool size).

### Fixed
A pre-existing bug surfaced by the Flash Attention rewrite: the per-head attention
output buffer was sized by `n_embd` instead of `n_heads * head_dim`. Harmless under the
old fixed-bounds loop, but caused an out-of-bounds panic on CPU once head_dim no longer
evenly divides n_embd (Gemma 2/3) under the new chunked iteration. Fixed on both CPU and
GPU.

Gemma 2 produced garbled output outright. `head_dim` was always derived as
`n_embd/n_heads`, which is wrong whenever a model's GGUF declares an explicit
`attention.key_length` (Gemma 2 2B: 256 vs. the derived 288) — now read from GGUF when
present. Also missing: embedding scaling by `sqrt(hidden_size)`, the "sandwich" post-
attention/post-FFN RMSNorms, GeGLU (Gemma uses tanh-GELU gating, not SiLU), and
attention/final logit softcapping. All added, architecture-gated off `general.architecture`.

### Changed
Split the two largest files along responsibility lines: `gpu/mod.rs` into
`context.rs` (Vulkan init), `buffers.rs` (allocation/upload), `dispatch.rs` (command
recording); `model/llama.rs` into `mod.rs` (struct defs), `loader.rs` (GGUF → weights),
`forward.rs` (CPU/GPU forward pass). Chat-loop logic (prefill, generation, context
rebuild) moved out of `bin/cli.rs` into a reusable `src/chat.rs` lib module, leaving the
binary as a thin argument-parsing entrypoint.

Updated all dependencies to latest (clap, anyhow, bytemuck, memmap2, rayon, shaderc
0.8→0.10 with the accompanying `Result`-returning API fix in `build.rs`).

Removed all comments across the codebase per a clean-base request.

### Removed
Unused GEMM/batch-prefill GPU shaders and dispatch code (15 shader files, ~400 lines) —
dead since the batched prefill path was never wired up; `forward_gpu`'s per-token layer
loop now shares `record_layer_gpu` with the prefill path instead of duplicating it.

The OpenAI-compatible HTTP server (`src/bin/server.rs`) and its dependencies (axum,
tokio, uuid, tower-http, async-stream, futures, serde*) — deferred until the CLI/engine
is solid; this release is CLI-only.

## [1.1.0] - 2026-05-03
### Added
Add suport for more quantization types.

src/gpu/q4_1_gemv.glsl — Q4_1 shader (4-bit asymmetric with scale + min per 32-weight block)
src/gpu/q3k_gemv.glsl — Q3_K shader (3-bit with high-bit mask, 6-bit scales, 256-weight superblocks)
src/gpu/q5k_gemv.glsl — Q5_K shader (5-bit with high bits, scale + min, 256-weight superblocks)

### Changed
src/tensor/dequant.rs — added pack_q4_1_for_gpu(), pack_q3k_for_gpu(), pack_q5k_for_gpu()
src/gpu/mod.rs (gpu_mod.rs) — added Q4_1, Q3K, Q5K to shader enum, pipeline creation, and upload dispatch
build.rs — added the 3 new shaders to the compile list


## [1.0.0] - 2026-05-03
### Added
First initial release, will contain bugs but is functional!
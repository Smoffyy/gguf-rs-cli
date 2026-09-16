# gguf-rs

A local LLM inference engine, written in Rust from scratch — no PyTorch, no CUDA, no
ggml/llama.cpp bindings. GGUF parsing, dequantization, the transformer forward pass,
and GPU compute are all hand-written. Vulkan is used for GPU acceleration (NVIDIA, AMD,
Intel) with a full CPU fallback.

<p align="center">
  <img src="rust-logo.svg" alt="Rust logo" width="400"/>
</p>

CLI-only right now. An OpenAI-compatible local server existed as a WIP and was removed
to keep this focused — it'll come back once the CLI/engine is solid.

## What it does

- **Every GGUF quantization type dequantizes correctly** — F32/F16, the legacy Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1,
  and the K-quants Q2_K–Q8_K, all bit-exact against the reference ggml block layout.
  On GPU, Q4_0/Q4_1/Q4_K/Q3_K/Q5_K/Q6_K/Q8_0/F32 run on native compute shaders; anything
  else is dequantized to f32 and still runs on GPU rather than being silently skipped.
- **Flash-Attention-2-style attention**, GPU and CPU — a single streaming pass with
  online-softmax rescaling (tiled, shared-memory-staged on GPU) instead of materializing
  a full score buffer and doing separate max/sum/normalize passes over it.
- **Mixture-of-experts** — per-token top-k expert routing, with expert weights kept
  CPU-resident (they're far too large for consumer VRAM even quantized) while attention
  and the shared layers still run on GPU when `--gpu` is set.
- **Architecture-specific correctness**, not just "loads without erroring": explicit
  per-model `head_dim` (not always `n_embd/n_heads`), embedding scaling, QK-RMSNorm,
  Gemma's sandwich (pre+post) norms and GeGLU activation, attention/final logit
  softcapping — each gated off GGUF metadata so nothing changes for models that don't
  use them.
- **Automatic GPU clock lock** on NVIDIA — locks clocks to the GPU's max boost for the
  run via `nvidia-smi -lgc` so the driver doesn't leave throughput on the table between
  dispatches (needs an elevated terminal; degrades gracefully otherwise).
- **Fully customizable sampling** — temperature, top-k, top-p, min-p, repetition/presence/
  frequency penalties, and Mirostat v2, all independently tunable from the CLI.
- **Smart context** — automatically drops old turns and replays system+history when the
  context window fills, instead of just stopping.
- **100% offline** — no network calls, no telemetry.

## Supported architectures

| GGUF `general.architecture` | Covers | Notes |
|---|---|---|
| `llama` | Llama 2/3/3.1, Mistral, CodeLlama | |
| `qwen`, `qwen2` | Qwen 1/1.5/2/2.5 | |
| `qwen3` | Qwen 3 (dense) | QK-RMSNorm |
| `qwen3moe` | Qwen 3 MoE (e.g. 30B-A3B) | Expert routing, CPU-resident experts |
| `gemma`, `gemma2`, `gemma3` | Gemma 1/2/3 | Embedding scale, sandwich norms, GeGLU; Gemma 2 also has logit softcapping |
| `phi3` | Phi-3, Phi-3.5 | |

Not supported: hybrid SSM/attention architectures (interleaved gated-delta-net and
full-attention layers) — that needs a separate, largely new inference path. Gemma's
alternating local/global sliding-window attention also isn't implemented — every layer
attends over the full context, which only diverges from the reference past a few
thousand tokens of conversation.

## Building

Requires the Rust toolchain and the Vulkan SDK (shader compilation happens at build
time via `shaderc`; the compiled binary needs no SDK or Vulkan headers at runtime, only
a GPU driver's Vulkan loader if you use `--gpu`).

```bash
cargo build --release
# binary at target/release/gguf-rs (gguf-rs.exe on Windows)
```

The resulting binary is a standalone native executable — running it on another machine
needs no Rust, no Vulkan SDK, nothing but the exe itself (and, for `--gpu`, a GPU with a
reasonably current driver; it falls back to CPU cleanly if Vulkan isn't available).

## Usage

```bash
# Interactive chat
gguf-rs --model model.gguf --gpu

# One-shot prompt
gguf-rs --model model.gguf --gpu -p "Explain recursion in one sentence."

# Raw completion (no chat template)
gguf-rs --model model.gguf --gpu --raw -p "The mitochondria is"
```

In interactive mode, type at the `You:` prompt and get a reply; the conversation carries
across turns. `/quit`, `/exit`, or Ctrl+C to leave.

### Options

```
Usage: gguf-rs [OPTIONS] --model <MODEL>

  -m, --model <MODEL>          Path to GGUF model file
  -p, --prompt <PROMPT>        Single prompt (non-interactive mode)
  -s, --system <SYSTEM>        Custom system prompt
      --raw                    Skip chat template formatting; raw completion
  -n, --max-tokens <N>         Max tokens to generate [default: 512]
      --min-tokens <N>         Minimum tokens before EOS is honored [default: 0]
  -c, --ctx-len <N>            Context window size [default: 8192]
      --smart-context          Auto-rebuild context when the window fills up

  -t, --temperature <T>        Sampling temperature [default: 0.7]
      --top-k <K>              Top-K sampling [default: 40]
      --top-p <P>              Top-P (nucleus) sampling [default: 0.9]
      --min-p <P>              Min-P sampling (0 = off) [default: 0.0]
      --rep-penalty <P>        Repetition penalty [default: 1.1]
      --presence-penalty <P>   Flat penalty per token already generated [default: 0.0]
      --frequency-penalty <P>  Penalty scaled by occurrence count [default: 0.0]
      --mirostat <0|1|2>       Mirostat v2 (overrides top-k/top-p/min-p) [default: 0]
      --mirostat-tau <T>       Target surprise [default: 5.0]
      --mirostat-eta <E>       Learning rate [default: 0.1]
      --seed <N>               RNG seed [default: 42]

      --gpu                    Enable Vulkan GPU acceleration
      --n-threads <N>          CPU worker threads (0 = all logical cores) [default: 0]
      --prefill-batch <N>      Prompt tokens per GPU submit during prefill [default: 512]

      --stats                  Print throughput statistics
      --debug-tokens           Show tokenization details
      --debug-gpu               Show per-token GPU timing breakdown
  -h, --help                   Print help
```

## Performance

Measured on an NVIDIA RTX 3080:

| Model | Mode | Speed |
|---|---|---|
| Qwen2.5-0.5B-Instruct Q4_K_M | GPU (Vulkan) | ~85 tok/s |
| Qwen2.5-0.5B-Instruct Q4_K_M | CPU (Rayon) | ~15 tok/s |
| Qwen3-1.7B Q4_K_M | GPU (Vulkan) | ~65 tok/s |
| Qwen3-8B Q4_K_M | GPU (Vulkan) | ~22 tok/s |
| Qwen3-30B-A3B (MoE) Q4_K_M | GPU attn + CPU experts | ~4.3 tok/s |
| Qwen3-30B-A3B (MoE) Q4_K_M | CPU only | ~3.3 tok/s |

Speed scales with model size and depends on GPU power state — see the clock-lock note
above; without it the driver's default power management can leave the GPU well below
its boost clock during inference. MoE models are slower per-token than their dense
counterparts of similar active-parameter count, since expert FFN layers round-trip
through CPU each layer.

## Architecture

```
src/
├── bin/
│   └── cli.rs               # Argument parsing, entrypoint
├── chat.rs                   # Chat loop: prefill, generation, context-window rebuild
├── power.rs                  # Automatic NVIDIA clock lock
├── sampler.rs                 # Temperature/top-k/top-p/min-p/penalties/Mirostat
├── gguf/
│   ├── reader.rs             # GGUF file parser
│   └── types.rs              # GGUF value types, GgmlType enum
├── tensor/
│   ├── dequant.rs            # Dequantization for every GGUF quant type + GPU packing + MoE expert slicing
│   └── storage.rs            # Memory-mapped tensor storage
├── model/
│   ├── mod.rs                # LlamaModel / Weights / GpuWeights / GpuActs / KvCache
│   ├── config.rs             # Model config from GGUF metadata (head_dim, MoE, Gemma flags, softcapping)
│   ├── loader.rs             # GGUF → Weights, GPU upload
│   └── forward.rs            # Transformer forward pass (CPU + GPU), MoE routing, Flash Attention
├── math/
│   ├── ops.rs                 # RMSNorm, softmax, SiLU, GELU, vector add
│   └── rope.rs                 # Rotary Position Embeddings
├── tokenizer/
│   ├── bpe.rs                 # BPE tokenizer (Llama + GPT-2 style)
│   └── chat.rs                 # Chat template detection and formatting
├── gpu/
│   ├── mod.rs                # Shader enum, GpuTensor/ActBuf/VkCtx definitions
│   ├── context.rs            # Vulkan instance/device/pipeline setup (VkCtx::init)
│   ├── buffers.rs            # Buffer allocation, weight upload (incl. f32 fallback)
│   ├── dispatch.rs           # Command recording, descriptor sets, submit/readback
│   ├── q4k_gemv.glsl / q3k_gemv.glsl / q5k_gemv.glsl / q6k_gemv.glsl
│   │                          # K-quant matrix-vector multiply shaders
│   ├── q4_0_gemv.glsl / q4_1_gemv.glsl / q8_0_gemv.glsl / f32_gemv.glsl
│   │                          # Legacy-quant / f32 matrix-vector multiply shaders
│   ├── qk_norm.glsl           # Per-head QK-RMSNorm shader
│   ├── rmsnorm.glsl           # RMS normalization shader
│   ├── attention.glsl         # Flash-Attention-2-style tiled attention shader
│   ├── rope.glsl              # RoPE shader
│   ├── kv_write.glsl          # KV cache write shader
│   ├── swiglu.glsl            # SwiGLU / GeGLU activation shader
│   ├── add.glsl               # Vector addition shader
│   └── add_rmsnorm.glsl       # Fused add + RMSNorm shader
└── build.rs                   # GLSL → SPIR-V shader compilation
```

## License

MIT
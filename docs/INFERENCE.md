# Inference Guide

This guide covers how to run local LLM inference with gguf-rs-cli.

## Getting a Model

You need a model in GGUF format. Download from Hugging Face:

1. Go to [huggingface.co/models](https://huggingface.co/models?search=gguf)
2. Search for your desired model + "GGUF"
3. Download the quantized file (recommended: `Q4_K_M` for best quality/speed balance)

### Recommended Models by VRAM

| VRAM | Model Size | Example |
|------|-----------|---------|
| 4 GB | 3B Q4_K_M | Llama-3.2-3B-Instruct |
| 6 GB | 7B Q4_K_M | Qwen2.5-7B-Instruct |
| 8 GB | 7B Q6_K | Qwen2.5-7B-Instruct |
| 12 GB | 7B Q4_K_M + 32K ctx | Any 7B model with large context |
| 16 GB | 13B Q4_K_M | Llama-2-13B-Chat |

### Quantization Types

| Type | Bits/Weight | Quality | GPU Support |
|------|------------|---------|-------------|
| Q4_0 | 4.5 | Good | Yes |
| Q4_K_M | 4.8 | Better | Yes |
| Q6_K | 6.6 | Great | Yes |
| Q8_0 | 8.5 | Excellent | Yes |
| F16 | 16 | Original | CPU only |

## Running Inference

### Basic GPU Inference

```bash
gguf-rs-cli --model path/to/model.gguf --gpu
```

This starts an interactive chat session. Type your messages and press Enter. Type `/quit` to exit.

### CPU-Only Inference

```bash
gguf-rs-cli --model path/to/model.gguf
```

No `--gpu` flag = CPU mode. Slower but works without a GPU.

### Single Prompt (Non-Interactive)

```bash
gguf-rs-cli --model model.gguf --gpu --prompt "Write a haiku about Rust"
```

Output goes to stdout, diagnostics to stderr. Useful for scripting:

```bash
gguf-rs-cli --model model.gguf --gpu --prompt "Summarize this:" < input.txt > output.txt
```

## Tuning Parameters

### Context Length (`--ctx-len`)

Controls how many tokens the model can "see" at once. Larger = more memory.

```bash
# Small context (faster, less memory)
gguf-rs-cli --model model.gguf --gpu --ctx-len 2048

# Large context (more memory, same speed per token)
gguf-rs-cli --model model.gguf --gpu --ctx-len 32000
```

Memory usage: ~`n_layers × n_kv_heads × head_dim × ctx_len × 8` bytes for KV cache.
For Qwen2.5-7B with 32K context: ~230 MB KV cache on GPU.

### Temperature (`--temperature`)

Controls randomness. Lower = more focused, higher = more creative.

```bash
# Precise/factual (good for code, math)
--temperature 0.3

# Balanced (default)
--temperature 0.7

# Creative (good for stories, brainstorming)
--temperature 1.0
```

### Top-K and Top-P

Additional sampling controls:

```bash
# Conservative: only consider top 20 tokens
--top-k 20

# Loose: consider more options
--top-k 100

# Nucleus sampling: consider tokens until 95% probability mass
--top-p 0.95
```

### Repetition Penalty (`--rep-penalty`)

Prevents the model from repeating itself. 1.0 = no penalty, higher = less repetition.

```bash
--rep-penalty 1.1  # default, mild penalty
--rep-penalty 1.3  # stronger, for models that repeat a lot
```

### Max Tokens (`--max-tokens`)

Maximum number of tokens to generate per response.

```bash
--max-tokens 1024  # longer responses
--max-tokens 128   # short responses
```

## Smart Context

When the context window fills up during a long conversation, `--smart-context` automatically rebuilds the KV cache to keep going:

```bash
gguf-rs-cli --model model.gguf --gpu --smart-context --ctx-len 8192
```

What happens when context fills up:
1. The oldest 25% of conversation history is dropped
2. System prompt + remaining history are replayed through the model
3. Generation continues seamlessly

Without `--smart-context`, generation stops when the context window is full.

## Performance Tips

### GPU Performance

1. **Use `--stats`** to see your actual throughput
2. **NVIDIA users**: GPU may run at reduced clock speed (P3 state). For best performance:
   ```bash
   nvidia-smi -lgc 1500,1950  # lock clocks high (requires admin)
   ```
3. **Larger models are proportionally slower** — speed scales with weight data size
4. **Context length doesn't affect per-token speed** (only memory usage)

### Monitoring

```bash
# Show throughput stats
--stats

# Show GPU timing per operation (for debugging)
--debug-gpu

# Show tokenization details
--debug-tokens
```

### Expected Speeds (RTX 3080, 7B Q4_K_M)

| Metric | Speed |
|--------|-------|
| System prefill | ~50 tok/s |
| Generation | 12-20 tok/s |
| Prefill per turn | ~13 tok/s |

## Custom System Prompts

```bash
gguf-rs-cli --model model.gguf --gpu \
  --system "You are a Python expert. Always provide code examples."
```

If no `--system` is specified, the model's built-in system prompt is used (from the GGUF metadata), or "You are a helpful assistant." as fallback.

## Chat Templates

The tool automatically detects the correct chat template from the GGUF file:

- **ChatML**: Qwen 1/1.5/2/2.5
- **Llama3**: Llama 3, Llama 3.1
- **Llama2**: Llama 2, Mistral, CodeLlama
- **Gemma**: Gemma, Gemma 2
- **Phi3**: Phi-3, Phi-3.5

No manual configuration needed — the template is detected from GGUF metadata and vocabulary.

## Troubleshooting

### "No GPU — using CPU"
- Install the Vulkan SDK and ensure your GPU drivers are up to date
- Check that `vulkaninfo` runs successfully

### "ERROR_OUT_OF_POOL_MEMORY" or crash during generation
- This was fixed in v1.0. If you see it, ensure you have the latest build

### Model loads but output is garbage
- Check that you're using an instruction-tuned model (not a base model)
- Try lowering temperature: `--temperature 0.5`
- Ensure the model format is supported (Q4_0, Q4_K_M, Q6_K, Q8_0)

### Very slow on GPU
- Run `nvidia-smi` and check the GPU is in P0/P2 state, not P3
- Lock clocks: `nvidia-smi -lgc 1500,1950`
- Close other GPU-intensive applications

### Context too small
- Increase with `--ctx-len 16384` or higher
- Use `--smart-context` for long conversations
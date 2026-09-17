// CUDA kernels for the gguf-rs transformer op set.
//
// Every kernel here is declared `extern "C"` so it can be found by name through the driver
// API; the host side never links against the CUDA runtime, it opens the driver library at
// startup and loads the PTX this file compiles to.
//
// Activations are f32 and laid out [n_tokens][dim]. Weights stay in their GGUF block
// layout and are decoded inside the kernel, which is what keeps device memory at the size
// of the file rather than the size of the dequantized model.

#include "quants.cuh"

#define WARP 32

// CUDA's INFINITY macro is a double, and narrowing it warns on every use. The bit pattern
// is exact and needs no conversion.
#define NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off, WARP);
    }
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = WARP / 2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off, WARP));
    }
    return v;
}

// =============================================================== activation quantization

// One block per activation row-chunk: 32 values to one Q8_1 block.
extern "C" __global__ void quantize_q8_1(const float *__restrict__ src,
                                         block_q8_1 *__restrict__ dst,
                                         const int dim, const int blocks_per_row,
                                         const int n_tokens) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = blocks_per_row * n_tokens;
    if (idx >= total) return;

    const int t = idx / blocks_per_row;
    const int b = idx % blocks_per_row;
    const int base = b * 32;

    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int i = base + j;
        v[j] = (i < dim) ? src[(size_t)t * dim + i] : 0.0f;
        amax = fmaxf(amax, fabsf(v[j]));
    }
    const float d = amax / 127.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;

    block_q8_1 *o = &dst[idx];
    int sum = 0;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        int q = __float2int_rn(v[j] * id);
        q = min(127, max(-127, q));
        o->qs[j] = (int8_t)q;
        sum += q;
    }
    o->d = __float2half(d);
    o->s = __float2half((float)sum * d);
}

// ================================================================================ matmul

// One warp per output row, looping over the batch.
//
// A row's weight bytes are a few kilobytes, so the per-token passes hit L1 after the first
// and prefill does not re-pay DRAM bandwidth per token.
// Tokens processed together per pass. The weight bytes for a span are read once per token
// in the tile, back to back from the same warp, so every read after the first is an L1 hit.
// That is what makes prefill pay the weight bandwidth once per tile rather than once per
// token, without needing a separate decode-then-dot split per quantization.
#define TOKEN_TILE 8

template <int TYPE>
__device__ void mul_mat_impl(const u8 *__restrict__ w, const block_q8_1 *__restrict__ y,
                             float *__restrict__ dst, const float *__restrict__ bias,
                             const int n_in, const int n_out, const int n_tokens,
                             const int row_bytes, const int blocks_per_row) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const u8 *wrow = w + (size_t)row * row_bytes;
    const float b = bias ? bias[row] : 0.0f;

    const int per_span = has_warp_dot<TYPE>::value ? warp_chunks<TYPE>() : WARP;
    const int n_spans = (blocks_per_row + per_span - 1) / per_span;

    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {
        const int tile = min(TOKEN_TILE, n_tokens - t0);
        float acc[TOKEN_TILE];
        #pragma unroll
        for (int k = 0; k < TOKEN_TILE; ++k) acc[k] = 0.0f;

        if (has_warp_dot<TYPE>::value) {
            for (int s = 0; s < n_spans; ++s) {
                for (int k = 0; k < tile; ++k) {
                    acc[k] += warp_dot<TYPE>(wrow, s, y + (size_t)(t0 + k) * blocks_per_row, lane);
                }
            }
        } else {
            for (int c = lane; c < blocks_per_row; c += WARP) {
                for (int k = 0; k < tile; ++k) {
                    acc[k] += dot32<TYPE>(wrow, c, &y[(size_t)(t0 + k) * blocks_per_row + c]);
                }
            }
        }

        for (int k = 0; k < tile; ++k) {
            const float v = warp_sum(acc[k]);
            if (lane == 0) dst[(size_t)(t0 + k) * n_out + row] = v + b;
        }
    }
}

#define MATMUL_KERNEL(NAME, TYPE)                                                          \
extern "C" __global__ void NAME(const u8 *w, const block_q8_1 *y, float *dst,               \
                                const float *bias, int n_in, int n_out, int n_tokens,       \
                                int row_bytes, int blocks_per_row) {                        \
    mul_mat_impl<TYPE>(w, y, dst, bias, n_in, n_out, n_tokens, row_bytes, blocks_per_row);  \
}

MATMUL_KERNEL(mul_mat_q4_0,  T_Q4_0)
MATMUL_KERNEL(mul_mat_q4_1,  T_Q4_1)
MATMUL_KERNEL(mul_mat_q5_0,  T_Q5_0)
MATMUL_KERNEL(mul_mat_q5_1,  T_Q5_1)
MATMUL_KERNEL(mul_mat_q8_0,  T_Q8_0)
MATMUL_KERNEL(mul_mat_q2_k,  T_Q2_K)
MATMUL_KERNEL(mul_mat_q3_k,  T_Q3_K)
MATMUL_KERNEL(mul_mat_q4_k,  T_Q4_K)
MATMUL_KERNEL(mul_mat_q5_k,  T_Q5_K)
MATMUL_KERNEL(mul_mat_q6_k,  T_Q6_K)
MATMUL_KERNEL(mul_mat_iq4_nl, T_IQ4_NL)
MATMUL_KERNEL(mul_mat_iq4_xs, T_IQ4_XS)
MATMUL_KERNEL(mul_mat_mxfp4, T_MXFP4)
MATMUL_KERNEL(mul_mat_tq1_0, T_TQ1_0)
MATMUL_KERNEL(mul_mat_tq2_0, T_TQ2_0)

// Float weights keep the activation in f32; there is nothing to gain from quantizing it.
template <int TYPE>
__device__ void mul_mat_f_impl(const u8 *__restrict__ w, const float *__restrict__ y,
                               float *__restrict__ dst, const float *__restrict__ bias,
                               const int n_in, const int n_out, const int n_tokens,
                               const int row_bytes) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const u8 *wrow = w + (size_t)row * row_bytes;
    const int chunks = n_in / 32;

    const float b = bias ? bias[row] : 0.0f;
    for (int t0 = 0; t0 < n_tokens; t0 += TOKEN_TILE) {
        const int tile = min(TOKEN_TILE, n_tokens - t0);
        float acc[TOKEN_TILE];
        #pragma unroll
        for (int k = 0; k < TOKEN_TILE; ++k) acc[k] = 0.0f;
        for (int c = lane; c < chunks; c += WARP) {
            float v[32];
            dequant32<TYPE>(wrow, c, v);
            for (int k = 0; k < tile; ++k) {
                const float *yrow = y + (size_t)(t0 + k) * n_in;
                #pragma unroll
                for (int j = 0; j < 32; ++j) acc[k] += v[j] * yrow[c * 32 + j];
            }
        }
        for (int k = 0; k < tile; ++k) {
            const float s = warp_sum(acc[k]);
            if (lane == 0) dst[(size_t)(t0 + k) * n_out + row] = s + b;
        }
    }
}

#define MATMUL_F_KERNEL(NAME, TYPE)                                                        \
extern "C" __global__ void NAME(const u8 *w, const float *y, float *dst,                    \
                                const float *bias, int n_in, int n_out, int n_tokens,       \
                                int row_bytes) {                                            \
    mul_mat_f_impl<TYPE>(w, y, dst, bias, n_in, n_out, n_tokens, row_bytes);                \
}

MATMUL_F_KERNEL(mul_mat_f32,  T_F32)
MATMUL_F_KERNEL(mul_mat_f16,  T_F16)
MATMUL_F_KERNEL(mul_mat_bf16, T_BF16)

// Dense f32 weight buffer times f32 activations.
extern "C" __global__ void mul_mat_dense(const float *__restrict__ w,
                                         const float *__restrict__ y,
                                         float *__restrict__ dst,
                                         const float *__restrict__ bias,
                                         int n_in, int n_out, int n_tokens) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const float *wrow = w + (size_t)row * n_in;
    for (int t = 0; t < n_tokens; ++t) {
        const float *yrow = y + (size_t)t * n_in;
        float acc = 0.0f;
        for (int i = lane; i < n_in; i += WARP) acc += wrow[i] * yrow[i];
        acc = warp_sum(acc);
        if (lane == 0) dst[(size_t)t * n_out + row] = acc + (bias ? bias[row] : 0.0f);
    }
}

// ============================================================================== get_rows

template <int TYPE>
__device__ void get_rows_impl(const u8 *__restrict__ w, const int *__restrict__ tokens,
                              float *__restrict__ dst, const int n_rows, const int cols,
                              const int row_bytes, const float scale, const int n_tokens) {
    const int t = blockIdx.x;
    if (t >= n_tokens) return;
    int r = tokens[t];
    r = min(max(r, 0), n_rows - 1);
    const u8 *wrow = w + (size_t)r * row_bytes;
    const int chunks = cols / 32;
    for (int c = threadIdx.x; c < chunks; c += blockDim.x) {
        float v[32];
        dequant32<TYPE>(wrow, c, v);
        #pragma unroll
        for (int j = 0; j < 32; ++j) dst[(size_t)t * cols + c * 32 + j] = v[j] * scale;
    }
}

#define GET_ROWS_KERNEL(NAME, TYPE)                                                        \
extern "C" __global__ void NAME(const u8 *w, const int *tokens, float *dst,                 \
                                int n_rows, int cols, int row_bytes, float scale,           \
                                int n_tokens) {                                             \
    get_rows_impl<TYPE>(w, tokens, dst, n_rows, cols, row_bytes, scale, n_tokens);          \
}

GET_ROWS_KERNEL(get_rows_f32,   T_F32)
GET_ROWS_KERNEL(get_rows_f16,   T_F16)
GET_ROWS_KERNEL(get_rows_bf16,  T_BF16)
GET_ROWS_KERNEL(get_rows_q4_0,  T_Q4_0)
GET_ROWS_KERNEL(get_rows_q4_1,  T_Q4_1)
GET_ROWS_KERNEL(get_rows_q5_0,  T_Q5_0)
GET_ROWS_KERNEL(get_rows_q5_1,  T_Q5_1)
GET_ROWS_KERNEL(get_rows_q8_0,  T_Q8_0)
GET_ROWS_KERNEL(get_rows_q2_k,  T_Q2_K)
GET_ROWS_KERNEL(get_rows_q3_k,  T_Q3_K)
GET_ROWS_KERNEL(get_rows_q4_k,  T_Q4_K)
GET_ROWS_KERNEL(get_rows_q5_k,  T_Q5_K)
GET_ROWS_KERNEL(get_rows_q6_k,  T_Q6_K)
GET_ROWS_KERNEL(get_rows_iq4_nl, T_IQ4_NL)
GET_ROWS_KERNEL(get_rows_iq4_xs, T_IQ4_XS)
GET_ROWS_KERNEL(get_rows_mxfp4, T_MXFP4)
GET_ROWS_KERNEL(get_rows_tq1_0, T_TQ1_0)
GET_ROWS_KERNEL(get_rows_tq2_0, T_TQ2_0)

// ================================================================================= norms

// One block per row. `kind` 0 is RMSNorm, 1 is LayerNorm.
extern "C" __global__ void norm_kernel(const float *__restrict__ src,
                                       float *__restrict__ dst,
                                       const float *__restrict__ weight,
                                       const float *__restrict__ bias,
                                       int dim, int kind, float eps, float scale) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const float *s = src + (size_t)row * dim;
    float *d = dst + (size_t)row * dim;
    const int tid = threadIdx.x;

    float mean = 0.0f;
    if (kind == 1) {
        float sum = 0.0f;
        for (int i = tid; i < dim; i += blockDim.x) sum += s[i];
        smem[tid] = sum;
        __syncthreads();
        for (int st = blockDim.x / 2; st > 0; st >>= 1) {
            if (tid < st) smem[tid] += smem[tid + st];
            __syncthreads();
        }
        mean = smem[0] / (float)dim;
        __syncthreads();
    }

    float sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float v = s[i] - mean;
        sq += v * v;
    }
    smem[tid] = sq;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (tid < st) smem[tid] += smem[tid + st];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)dim + eps);

    for (int i = tid; i < dim; i += blockDim.x) {
        float v = (s[i] - mean) * inv;
        if (weight) v *= weight[i];
        if (bias) v += bias[i];
        d[i] = v * scale;
    }
}

// ================================================================================== rope

__device__ __forceinline__ float yarn_ramp(float low, float high, int i) {
    const float y = ((float)i - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

// `kind` 0 = adjacent pairs (NORM), 1 = split halves (NeoX and, for text, M-RoPE).
extern "C" __global__ void rope_kernel(float *__restrict__ x, const int *__restrict__ pos,
                                       const float *__restrict__ freq_factors,
                                       int n_heads, int head_dim, int n_rot, int n_tokens,
                                       int kind, float freq_base, float freq_scale,
                                       float ext_factor, float attn_factor,
                                       float corr_lo, float corr_hi) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs = n_rot / 2;
    const int total = n_tokens * n_heads * pairs;
    if (idx >= total) return;

    const int i = idx % pairs;                       // rotary pair index
    const int h = (idx / pairs) % n_heads;
    const int t = idx / (pairs * n_heads);

    const float ff = freq_factors ? freq_factors[i] : 1.0f;
    // One power rather than raising a precomputed step: the same expression the other
    // backends evaluate, and more accurate at the high end of the frequency ladder.
    const float theta_extrap =
        (float)pos[t] * powf(freq_base, -2.0f * (float)i / (float)n_rot) / ff;
    const float theta_interp = freq_scale * theta_extrap;

    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float mix = yarn_ramp(corr_lo, corr_hi, i) * ext_factor;
        theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    float sn, cs;
    sincosf(theta, &sn, &cs);
    sn *= mscale;
    cs *= mscale;

    float *head = x + ((size_t)t * n_heads + h) * head_dim;
    const int a = (kind == 0) ? (i * 2) : i;
    const int b = (kind == 0) ? (i * 2 + 1) : (i + n_rot / 2);
    const float x0 = head[a], x1 = head[b];
    head[a] = x0 * cs - x1 * sn;
    head[b] = x0 * sn + x1 * cs;
}

// ============================================================================== kv cache

// The cache stores f32 or f16. Attention reads and writes f32 either way; f16 storage
// halves the largest allocation after the weights and costs nothing observable in output.
template <int KVT>
__device__ __forceinline__ float kv_load(const void *p, size_t i);

template <>
__device__ __forceinline__ float kv_load<0>(const void *p, size_t i) {
    return ((const float *)p)[i];
}

template <>
__device__ __forceinline__ float kv_load<1>(const void *p, size_t i) {
    return __half2float(((const half *)p)[i]);
}

template <int KVT>
__device__ __forceinline__ void kv_store(void *p, size_t i, float v);

template <>
__device__ __forceinline__ void kv_store<0>(void *p, size_t i, float v) {
    ((float *)p)[i] = v;
}

template <>
__device__ __forceinline__ void kv_store<1>(void *p, size_t i, float v) {
    ((half *)p)[i] = __float2half(v);
}

template <int KVT>
__device__ void kv_write_impl(const float *__restrict__ k, const float *__restrict__ v,
                              void *__restrict__ k_cache, void *__restrict__ v_cache,
                              int dim, int stride, int base, int start_pos, int n_tokens) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim * n_tokens) return;
    const int t = idx / dim;
    const int i = idx % dim;
    const size_t off = (size_t)(base + start_pos + t) * stride + i;
    kv_store<KVT>(k_cache, off, k[(size_t)t * dim + i]);
    kv_store<KVT>(v_cache, off, v[(size_t)t * dim + i]);
}

#define KV_WRITE_KERNEL(NAME, KVT)                                                         extern "C" __global__ void NAME(const float *k, const float *v, void *k_cache,                                              void *v_cache, int dim, int stride, int base,                                               int start_pos, int n_tokens) {                                  kv_write_impl<KVT>(k, v, k_cache, v_cache, dim, stride, base, start_pos, n_tokens);     }

KV_WRITE_KERNEL(kv_write_f32, 0)
KV_WRITE_KERNEL(kv_write_f16, 1)

// ============================================================================= attention

// Flash decoding: the key/value range is split across blocks, each of which reduces its
// own slice with an online softmax, and a second pass combines the partials.
//
// The obvious shape - one block per (token, head), looping the whole history - is what this
// replaces. During decode there is one token, so that shape launches only n_heads blocks
// and leaves most of the GPU idle while each block walks thousands of keys serially with a
// block-wide reduction per key. Splitting the sequence is what turns attention from the
// dominant cost of a decode step back into a rounding error.
//
// Inside a block each warp owns a strided subset of the keys and keeps its own running
// maximum, denominator and weighted value sum, so the inner loop needs warp shuffles only
// and never a barrier.

#define ATTN_WARPS 4
#define ATTN_THREADS (ATTN_WARPS * WARP)
#define ATTN_MAX_HEAD_DIM 256

template <int KVT>
__device__ void flash_attn_split_impl(const float *__restrict__ q,
                                      const void *__restrict__ k_cache,
                                      const void *__restrict__ v_cache,
                                      float *__restrict__ partial,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      int n_tokens, int kv_len, int start_pos,
                                      int stride, int base,
                                      float scale, float softcap, int window,
                                      int n_splits) {
    __shared__ float sq[ATTN_MAX_HEAD_DIM];
    __shared__ float sacc[ATTN_WARPS][ATTN_MAX_HEAD_DIM];
    __shared__ float sm[ATTN_WARPS];
    __shared__ float sl[ATTN_WARPS];

    const int hb = blockIdx.x;
    const int split = blockIdx.y;
    const int h = hb % n_heads;
    const int t = hb / n_heads;
    if (t >= n_tokens) return;

    const int tid = threadIdx.x;
    const int warp = tid / WARP;
    const int lane = tid % WARP;
    const int gqa = n_heads / n_kv_heads;
    const int kv_h = h / gqa;

    const float *qh = q + ((size_t)t * n_heads + h) * head_dim;
    for (int i = tid; i < head_dim; i += ATTN_THREADS) sq[i] = qh[i];
    for (int i = lane; i < head_dim; i += WARP) sacc[warp][i] = 0.0f;
    __syncthreads();

    const int q_pos = start_pos + t;
    const int hi = min(q_pos + 1, kv_len);
    const int lo = window > 0 ? max(0, q_pos + 1 - window) : 0;

    // Contiguous slices rather than a stride, so each block's key reads stay local.
    const int span = (hi - lo + n_splits - 1) / n_splits;
    const int my_lo = lo + split * span;
    const int my_hi = min(hi, my_lo + span);

    float m = NEG_INF;
    float l = 0.0f;

    for (int p = my_lo + warp; p < my_hi; p += ATTN_WARPS) {
        const size_t off = (size_t)(base + p) * stride + (size_t)kv_h * head_dim;
        float s = 0.0f;
        for (int i = lane; i < head_dim; i += WARP) s += sq[i] * kv_load<KVT>(k_cache, off + i);
        s = warp_sum(s) * scale;
        if (softcap > 0.0f) s = softcap * tanhf(s / softcap);

        const float new_m = fmaxf(m, s);
        const float alpha = expf(m - new_m);
        const float p_s = expf(s - new_m);
        for (int i = lane; i < head_dim; i += WARP) {
            sacc[warp][i] = sacc[warp][i] * alpha + p_s * kv_load<KVT>(v_cache, off + i);
        }
        l = l * alpha + p_s;
        m = new_m;
    }

    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
    __syncthreads();

    // Combine the warps' partials the same way the split pass combines blocks.
    if (warp == 0) {
        float gm = NEG_INF;
        for (int w = 0; w < ATTN_WARPS; ++w) gm = fmaxf(gm, sm[w]);
        float gl = 0.0f;
        for (int w = 0; w < ATTN_WARPS; ++w) gl += sl[w] * expf(sm[w] - gm);

        float *out = partial + ((size_t)(hb * n_splits + split)) * (head_dim + 2);
        for (int i = lane; i < head_dim; i += WARP) {
            float acc = 0.0f;
            for (int w = 0; w < ATTN_WARPS; ++w) acc += sacc[w][i] * expf(sm[w] - gm);
            out[i] = acc;
        }
        if (lane == 0) {
            out[head_dim] = gm;
            out[head_dim + 1] = gl;
        }
    }
}

#define ATTN_SPLIT_KERNEL(NAME, KVT)                                                       extern "C" __global__ void NAME(const float *q, const void *k_cache, const void *v_cache,                                   float *partial, int n_heads, int n_kv_heads, int head_dim,                                  int n_tokens, int kv_len, int start_pos, int stride,                                        int base, float scale, float softcap, int window,                                           int n_splits) {                                                 flash_attn_split_impl<KVT>(q, k_cache, v_cache, partial, n_heads, n_kv_heads, head_dim,                                n_tokens, kv_len, start_pos, stride, base, scale, softcap,                                  window, n_splits);                                           }

ATTN_SPLIT_KERNEL(flash_attn_split_f32, 0)
ATTN_SPLIT_KERNEL(flash_attn_split_f16, 1)

extern "C" __global__ void flash_attn_combine(const float *__restrict__ partial,
                                              float *__restrict__ dst,
                                              const float *__restrict__ sinks,
                                              int n_heads, int head_dim, int n_tokens,
                                              int n_splits) {
    const int hb = blockIdx.x;
    const int h = hb % n_heads;
    const int t = hb / n_heads;
    if (t >= n_tokens) return;
    const int tid = threadIdx.x;

    const float *base = partial + (size_t)hb * n_splits * (head_dim + 2);

    float gm = NEG_INF;
    for (int s = 0; s < n_splits; ++s) gm = fmaxf(gm, base[(size_t)s * (head_dim + 2) + head_dim]);
    // A split that covered no keys has m = -inf and l = 0, and drops out of both sums.
    float gl = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const float *p = base + (size_t)s * (head_dim + 2);
        gl += p[head_dim + 1] * expf(p[head_dim] - gm);
    }

    // A learned sink is an extra logit with no value vector: it moves probability mass out
    // of the softmax without contributing to the output.
    if (sinks) {
        const float sink = sinks[h];
        const float new_m = fmaxf(gm, sink);
        gl = gl * expf(gm - new_m) + expf(sink - new_m);
        gm = new_m;
    }

    const float inv = 1.0f / fmaxf(gl, 1e-20f);
    float *out = dst + ((size_t)t * n_heads + h) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_splits; ++s) {
            const float *p = base + (size_t)s * (head_dim + 2);
            acc += p[i] * expf(p[head_dim] - gm);
        }
        out[i] = acc * inv;
    }
}

// ============================================================================ elementwise

__device__ __forceinline__ float act_apply(int act, float x, float alpha) {
    switch (act) {
        case 0: return x / (1.0f + __expf(-x));                        // silu
        case 1:                                                         // gelu (tanh)
        case 2: return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        case 3: return x / (1.0f + __expf(-1.702f * x));               // gelu quick
        case 4: return fmaxf(x, 0.0f);                                 // relu
        case 5: { float r = fmaxf(x, 0.0f); return r * r; }            // relu squared
        default: return x / (1.0f + __expf(-alpha * x));               // swish
    }
}

extern "C" __global__ void glu_kernel(float *__restrict__ dst, const float *__restrict__ gate,
                                      const float *__restrict__ up, int n, int act,
                                      float limit, float alpha) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i], u = up[i];
    if (limit > 0.0f) {
        g = fminf(fmaxf(g, -limit), limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    dst[i] = act_apply(act, g, alpha) * u;
}

extern "C" __global__ void activate_kernel(float *__restrict__ buf, int n, int act, float alpha) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    buf[i] = act_apply(act, buf[i], alpha);
}

// `b` shorter than `n` is broadcast, which is how a per-channel vector applies across a
// multi-token activation without materialising a copy per token.
extern "C" __global__ void binary_kernel(float *__restrict__ dst, const float *__restrict__ a,
                                         const float *__restrict__ b, int n, int bn, int op) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = a[i], y = b[i % bn];
    dst[i] = op == 0 ? x + y : (op == 1 ? x * y : x - y);
}

extern "C" __global__ void scale_kernel(float *__restrict__ buf, float f, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] *= f;
}

extern "C" __global__ void softcap_kernel(float *__restrict__ buf, float cap, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = cap * tanhf(buf[i] / cap);
}

extern "C" __global__ void fill_kernel(float *__restrict__ buf, float v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = v;
}

// ================================================================================== MoE

// Softmax or sigmoid over the router logits, then top-k selection.
//
// Selection runs in f32 on one thread per token: it is negligible next to the expert
// matmuls, and a parallel top-k would change which expert wins a near-tie, which changes
// the output.
extern "C" __global__ void moe_route(const float *__restrict__ logits,
                                     const float *__restrict__ probs_bias,
                                     int *__restrict__ sel_idx,
                                     float *__restrict__ sel_w,
                                     int n_expert, int n_used, int n_tokens,
                                     int gate_func, int norm_topk, float scale) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens) return;

    const float *lg = logits + (size_t)t * n_expert;
    extern __shared__ float sm[];
    float *p = sm + threadIdx.x * 0;  // unused; probabilities are recomputed per candidate

    float maxl = NEG_INF, sum = 0.0f;
    if (gate_func == 0) {
        for (int e = 0; e < n_expert; ++e) maxl = fmaxf(maxl, lg[e]);
        for (int e = 0; e < n_expert; ++e) sum += expf(lg[e] - maxl);
    }
    (void)p;

    float wsum = 0.0f;
    for (int k = 0; k < n_used; ++k) {
        int best = -1;
        float best_key = NEG_INF;
        for (int e = 0; e < n_expert; ++e) {
            bool taken = false;
            for (int j = 0; j < k; ++j) {
                if (sel_idx[(size_t)t * n_used + j] == e) { taken = true; break; }
            }
            if (taken) continue;
            // The routing bias steers selection only; the weight applied to the expert
            // output stays the unbiased probability.
            float key = lg[e] + (probs_bias ? probs_bias[e] : 0.0f);
            if (key > best_key) { best_key = key; best = e; }
        }
        if (best < 0) best = 0;
        float prob = gate_func == 0
            ? expf(lg[best] - maxl) / fmaxf(sum, 1e-20f)
            : 1.0f / (1.0f + expf(-lg[best]));
        sel_idx[(size_t)t * n_used + k] = best;
        sel_w[(size_t)t * n_used + k] = prob;
        wsum += prob;
    }
    for (int k = 0; k < n_used; ++k) {
        float w = sel_w[(size_t)t * n_used + k];
        if (norm_topk) w /= fmaxf(wsum, 1e-20f);
        sel_w[(size_t)t * n_used + k] = w * scale;
    }
}

// Expert matmul: one warp per (slot, output row). `expert_stride` is the byte distance
// between consecutive experts in the stacked tensor.
template <int TYPE>
__device__ void moe_mm_impl(const u8 *__restrict__ w, const block_q8_1 *__restrict__ y,
                            const int *__restrict__ sel_idx, float *__restrict__ dst,
                            int n_out, int n_used, int n_tokens,
                            int row_bytes, int blocks_per_row, size_t expert_stride) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_out) return;
    const int slot = blockIdx.y;                  // token * n_used + k
    const int t = slot / n_used;
    if (t >= n_tokens) return;
    const int lane = threadIdx.x;

    const int e = sel_idx[slot];
    const u8 *wrow = w + (size_t)e * expert_stride + (size_t)row * row_bytes;
    const block_q8_1 *yrow = y + (size_t)slot * blocks_per_row;

    float acc = 0.0f;
    if (has_warp_dot<TYPE>::value) {
        const int per_span = warp_chunks<TYPE>();
        const int n_spans = (blocks_per_row + per_span - 1) / per_span;
        for (int s = 0; s < n_spans; ++s) {
            acc += warp_dot<TYPE>(wrow, s, yrow, lane);
        }
    } else {
        for (int c = lane; c < blocks_per_row; c += WARP) {
            acc += dot32<TYPE>(wrow, c, &yrow[c]);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) dst[(size_t)slot * n_out + row] = acc;
}

#define MOE_MM_KERNEL(NAME, TYPE)                                                          \
extern "C" __global__ void NAME(const u8 *w, const block_q8_1 *y, const int *sel_idx,        \
                                float *dst, int n_out, int n_used, int n_tokens,             \
                                int row_bytes, int blocks_per_row,                           \
                                unsigned long long expert_stride) {                          \
    moe_mm_impl<TYPE>(w, y, sel_idx, dst, n_out, n_used, n_tokens, row_bytes,                \
                      blocks_per_row, (size_t)expert_stride);                                \
}

MOE_MM_KERNEL(moe_mm_q4_0,  T_Q4_0)
MOE_MM_KERNEL(moe_mm_q4_1,  T_Q4_1)
MOE_MM_KERNEL(moe_mm_q5_0,  T_Q5_0)
MOE_MM_KERNEL(moe_mm_q5_1,  T_Q5_1)
MOE_MM_KERNEL(moe_mm_q8_0,  T_Q8_0)
MOE_MM_KERNEL(moe_mm_q2_k,  T_Q2_K)
MOE_MM_KERNEL(moe_mm_q3_k,  T_Q3_K)
MOE_MM_KERNEL(moe_mm_q4_k,  T_Q4_K)
MOE_MM_KERNEL(moe_mm_q5_k,  T_Q5_K)
MOE_MM_KERNEL(moe_mm_q6_k,  T_Q6_K)
MOE_MM_KERNEL(moe_mm_iq4_nl, T_IQ4_NL)
MOE_MM_KERNEL(moe_mm_iq4_xs, T_IQ4_XS)
MOE_MM_KERNEL(moe_mm_mxfp4, T_MXFP4)
MOE_MM_KERNEL(moe_mm_tq1_0, T_TQ1_0)
MOE_MM_KERNEL(moe_mm_tq2_0, T_TQ2_0)

// Fused gate/up activation across every selected expert slot.
extern "C" __global__ void moe_glu(float *__restrict__ gate, const float *__restrict__ up,
                                   int n, int act, int has_gate, float alpha) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    gate[i] = has_gate ? act_apply(act, gate[i], alpha) * up[i]
                       : act_apply(act, up[i], alpha);
}

// Weighted sum of each token's expert outputs back into one activation row.
extern "C" __global__ void moe_reduce(float *__restrict__ dst, const float *__restrict__ src,
                                      const float *__restrict__ sel_w,
                                      int n_embd, int n_used, int n_tokens) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_embd * n_tokens) return;
    const int t = idx / n_embd;
    const int i = idx % n_embd;
    float acc = 0.0f;
    for (int k = 0; k < n_used; ++k) {
        const int slot = t * n_used + k;
        acc += sel_w[slot] * src[(size_t)slot * n_embd + i];
    }
    dst[idx] = acc;
}

// Replicate each token's activation into its expert slots, so the expert matmul reads a
// contiguous per-slot row.
extern "C" __global__ void moe_scatter(const block_q8_1 *__restrict__ src,
                                       block_q8_1 *__restrict__ dst,
                                       int blocks_per_row, int n_used, int n_tokens) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = blocks_per_row * n_used * n_tokens;
    if (idx >= total) return;
    const int b = idx % blocks_per_row;
    const int slot = idx / blocks_per_row;
    const int t = slot / n_used;
    dst[idx] = src[(size_t)t * blocks_per_row + b];
}

// Block decoding for every quantization the engine supports.
//
// Weights sit in device memory in exactly the layout the GGUF file had: no repacking, no
// expansion to f32. That keeps VRAM at the file's size and makes the upload a straight
// memcpy, but it means every kernel that touches a weight has to decode blocks inline.
//
// The uniform shape here is `dot32`: the dot product of one 32-element run of a weight row
// against one Q8_1 activation block. Every quantization implements that one function, and
// every matmul kernel is written once against it. A 256-element K-quant super-block is just
// eight consecutive chunks that happen to share a scale header.

#ifndef GGUF_QUANTS_CUH
#define GGUF_QUANTS_CUH

#include <cuda_fp16.h>

#define QK8_1 32

// Activation block: f16 scale, f16 sum-of-quants-times-scale, 32 signed bytes.
struct block_q8_1 {
    half  d;
    half  s;
    int8_t qs[32];
};

typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;

// Type tags, matching the ggml type ids so host and device agree without a translation.
#define T_F32    0
#define T_F16    1
#define T_Q4_0   2
#define T_Q4_1   3
#define T_Q5_0   6
#define T_Q5_1   7
#define T_Q8_0   8
#define T_Q2_K  10
#define T_Q3_K  11
#define T_Q4_K  12
#define T_Q5_K  13
#define T_Q6_K  14
#define T_IQ4_NL 20
#define T_IQ4_XS 23
#define T_BF16  30
#define T_TQ1_0 34
#define T_TQ2_0 35
#define T_MXFP4 39

__device__ __forceinline__ float load_f16(const u8 *p) {
    half h;
    memcpy(&h, p, sizeof(half));
    return __half2float(h);
}

__device__ __forceinline__ float load_bf16(const u8 *p) {
    u16 raw;
    memcpy(&raw, p, sizeof(u16));
    u32 bits = ((u32) raw) << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

__device__ __forceinline__ float load_f32(const u8 *p) {
    float f;
    memcpy(&f, p, sizeof(float));
    return f;
}

__constant__ int8_t KVALUES_IQ4NL[16] =
    {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__constant__ int8_t KVALUES_MXFP4[16] =
    {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ float e8m0_half(u8 e) {
    u32 bits = e < 2 ? (0x00200000u << e) : ((u32)(e - 1) << 23);
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

// Q4_K / Q5_K pack eight 6-bit scale/min pairs into twelve bytes.
__device__ __forceinline__ void scale_min_k4(int j, const u8 *q, float *d, float *m) {
    if (j < 4) {
        *d = (float)(q[j] & 63);
        *m = (float)(q[j + 4] & 63);
    } else {
        *d = (float)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (float)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Q3_K packs sixteen 6-bit signed scales into twelve bytes.
__device__ __forceinline__ int q3k_scale(const u8 *sc, int k) {
    int lo4, hi2;
    if (k < 4)       { lo4 = sc[k] & 0xF;        hi2 = sc[8 + k] & 3; }
    else if (k < 8)  { lo4 = sc[k] & 0xF;        hi2 = (sc[4 + k] >> 2) & 3; }
    else if (k < 12) { lo4 = (sc[k - 8] >> 4) & 0xF; hi2 = (sc[k] >> 4) & 3; }
    else             { lo4 = (sc[k - 8] >> 4) & 0xF; hi2 = (sc[k - 4] >> 6) & 3; }
    return (int)((int8_t)(lo4 | (hi2 << 4))) - 32;
}

// Four signed 8-bit products accumulated into an int, using the hardware instruction where
// the architecture has it. This is the single biggest lever in a quantized matmul.
__device__ __forceinline__ int dp4(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t *pa = (const int8_t *)&a;
    const int8_t *pb = (const int8_t *)&b;
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
#endif
}

__device__ __forceinline__ int y4(const int8_t *qs, int off) {
    int v;
    memcpy(&v, qs + off, sizeof(int));
    return v;
}

// ---------------------------------------------------------------------------------------
// dot32<TYPE>: dot product of weight-row chunk `c` (32 elements) with activation block `yb`.
//
// `w` is the start of the weight row. Each specialisation locates its own block header.
// ---------------------------------------------------------------------------------------

template <int TYPE>
__device__ __forceinline__ float dot32(const u8 *w, int c, const block_q8_1 *yb);

template <>
__device__ __forceinline__ float dot32<T_Q4_0>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + c * 18;
    const float d = load_f16(b);
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        int q;
        memcpy(&q, b + 2 + j * 4, 4);
        sumi = dp4(q & 0x0F0F0F0F, y4(yb->qs, j * 4), sumi);
        sumi = dp4((q >> 4) & 0x0F0F0F0F, y4(yb->qs, 16 + j * 4), sumi);
    }
    // y = (q - 8) * d, so the offset becomes a constant multiple of the activation's sum.
    return d * (__half2float(yb->d) * (float)sumi) - 8.0f * d * __half2float(yb->s);
}

template <>
__device__ __forceinline__ float dot32<T_Q4_1>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + c * 20;
    const float d = load_f16(b), m = load_f16(b + 2);
    int sumi = 0;
    for (int j = 0; j < 4; ++j) {
        int q;
        memcpy(&q, b + 4 + j * 4, 4);
        sumi = dp4(q & 0x0F0F0F0F, y4(yb->qs, j * 4), sumi);
        sumi = dp4((q >> 4) & 0x0F0F0F0F, y4(yb->qs, 16 + j * 4), sumi);
    }
    return d * __half2float(yb->d) * (float)sumi + m * __half2float(yb->s);
}

template <>
__device__ __forceinline__ float dot32<T_Q5_0>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + c * 22;
    const float d = load_f16(b);
    u32 qh;
    memcpy(&qh, b + 2, 4);
    int sumi = 0;
    for (int j = 0; j < 16; ++j) {
        int lo = (b[6 + j] & 0xF) | (int)(((qh >> j) << 4) & 0x10);
        int hi = (b[6 + j] >> 4)  | (int)((qh >> (j + 12)) & 0x10);
        sumi += lo * yb->qs[j] + hi * yb->qs[j + 16];
    }
    return d * __half2float(yb->d) * (float)sumi - 16.0f * d * __half2float(yb->s);
}

template <>
__device__ __forceinline__ float dot32<T_Q5_1>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + c * 24;
    const float d = load_f16(b), m = load_f16(b + 2);
    u32 qh;
    memcpy(&qh, b + 4, 4);
    int sumi = 0;
    for (int j = 0; j < 16; ++j) {
        int lo = (b[8 + j] & 0xF) | (int)(((qh >> j) << 4) & 0x10);
        int hi = (b[8 + j] >> 4)  | (int)((qh >> (j + 12)) & 0x10);
        sumi += lo * yb->qs[j] + hi * yb->qs[j + 16];
    }
    return d * __half2float(yb->d) * (float)sumi + m * __half2float(yb->s);
}

template <>
__device__ __forceinline__ float dot32<T_Q8_0>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + c * 34;
    const float d = load_f16(b);
    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
        int q;
        memcpy(&q, b + 2 + j * 4, 4);
        sumi = dp4(q, y4(yb->qs, j * 4), sumi);
    }
    return d * __half2float(yb->d) * (float)sumi;
}

template <>
__device__ __forceinline__ float dot32<T_Q2_K>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + (c >> 3) * 84;
    const int sub = c & 7;           // which 32-element chunk of the super-block
    const int h = sub >> 2;          // 128-element half
    const int j = sub & 3;           // 2-bit shift group
    const u8 *sc = b;
    const u8 *qs = b + 16 + h * 32;
    const float d = load_f16(b + 80), dmin = load_f16(b + 82);
    const int shift = j * 2;
    const int is = h * 8 + j * 2;

    // The chunk spans two 16-element scale groups, each with its own minimum, so each
    // keeps its own partial activation sum.
    int si0 = 0, sy0 = 0, si1 = 0, sy1 = 0;
    for (int l = 0; l < 16; ++l) {
        si0 += ((qs[l] >> shift) & 3) * yb->qs[l];
        sy0 += yb->qs[l];
        si1 += ((qs[l + 16] >> shift) & 3) * yb->qs[l + 16];
        sy1 += yb->qs[l + 16];
    }
    const float ad = __half2float(yb->d);
    const u8 s0 = sc[is], s1 = sc[is + 1];
    return ad * (d * (float)(s0 & 0xF) * si0 - dmin * (float)(s0 >> 4) * sy0)
         + ad * (d * (float)(s1 & 0xF) * si1 - dmin * (float)(s1 >> 4) * sy1);
}

template <>
__device__ __forceinline__ float dot32<T_Q3_K>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + (c >> 3) * 110;
    const int sub = c & 7;
    const int h = sub >> 2;
    const int j = sub & 3;
    const u8 *hmask = b;
    const u8 *qs = b + 32 + h * 32;
    const u8 *scales = b + 96;
    const float d = load_f16(b + 108);
    const int shift = j * 2;
    const u8 m = (u8)(1u << (h * 4 + j));
    const int is = h * 8 + j * 2;

    int si0 = 0, si1 = 0;
    for (int l = 0; l < 16; ++l) {
        int hv0 = (hmask[l] & m) ? 0 : 4;
        int hv1 = (hmask[l + 16] & m) ? 0 : 4;
        si0 += (((qs[l] >> shift) & 3) - hv0) * yb->qs[l];
        si1 += (((qs[l + 16] >> shift) & 3) - hv1) * yb->qs[l + 16];
    }
    return __half2float(yb->d) * d
         * ((float)q3k_scale(scales, is) * si0 + (float)q3k_scale(scales, is + 1) * si1);
}

template <>
__device__ __forceinline__ float dot32<T_Q4_K>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + (c >> 3) * 144;
    const int sub = c & 7;
    const float d = load_f16(b), dmin = load_f16(b + 2);
    const u8 *scales = b + 4;
    const u8 *qs = b + 16 + (sub >> 1) * 32;
    const int hi = sub & 1;

    float sc, mn;
    scale_min_k4(sub, scales, &sc, &mn);

    int sumi = 0;
    for (int j = 0; j < 8; ++j) {
        int q;
        memcpy(&q, qs + j * 4, 4);
        int nib = hi ? ((q >> 4) & 0x0F0F0F0F) : (q & 0x0F0F0F0F);
        sumi = dp4(nib, y4(yb->qs, j * 4), sumi);
    }
    return d * sc * __half2float(yb->d) * (float)sumi - dmin * mn * __half2float(yb->s);
}

template <>
__device__ __forceinline__ float dot32<T_Q5_K>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + (c >> 3) * 176;
    const int sub = c & 7;
    const float d = load_f16(b), dmin = load_f16(b + 2);
    const u8 *scales = b + 4;
    const u8 *qh = b + 16;
    const u8 *ql = b + 48 + (sub >> 1) * 32;
    const int g = sub >> 1, hi = sub & 1;
    const u8 bit = (u8)((1u + (u32)hi) << (2 * g));

    float sc, mn;
    scale_min_k4(sub, scales, &sc, &mn);

    int sumi = 0;
    for (int l = 0; l < 32; ++l) {
        int lo = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
        int hv = (qh[l] & bit) ? 16 : 0;
        sumi += (lo + hv) * yb->qs[l];
    }
    return d * sc * __half2float(yb->d) * (float)sumi - dmin * mn * __half2float(yb->s);
}

template <>
__device__ __forceinline__ float dot32<T_Q6_K>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + (c >> 3) * 210;
    const int sub = c & 7;
    const int h = sub >> 2;
    const int g = sub & 3;
    const u8 *ql = b + h * 64 + (g & 1) * 32;
    const u8 *qh = b + 128 + h * 32;
    const u8 *sc = b + 192 + h * 8;
    const float d = load_f16(b + 208);
    const int lo_shift = (g >> 1) * 4;
    const int hi_shift = g * 2;

    // Scales change every 16 elements inside this 32-element chunk.
    int si0 = 0, si1 = 0;
    for (int l = 0; l < 32; ++l) {
        int lo = (ql[l] >> lo_shift) & 0xF;
        int hv = ((qh[l] >> hi_shift) & 3) << 4;
        int q = (lo | hv) - 32;
        if (l < 16) si0 += q * yb->qs[l];
        else        si1 += q * yb->qs[l];
    }
    return __half2float(yb->d) * d
         * ((float)((int8_t)sc[2 * g]) * si0 + (float)((int8_t)sc[2 * g + 1]) * si1);
}

template <>
__device__ __forceinline__ float dot32<T_IQ4_NL>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + c * 18;
    const float d = load_f16(b);
    int sumi = 0;
    for (int j = 0; j < 16; ++j) {
        sumi += KVALUES_IQ4NL[b[2 + j] & 0xF] * yb->qs[j];
        sumi += KVALUES_IQ4NL[b[2 + j] >> 4]  * yb->qs[j + 16];
    }
    return d * __half2float(yb->d) * (float)sumi;
}

template <>
__device__ __forceinline__ float dot32<T_IQ4_XS>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + (c >> 3) * 136;
    const int ib = c & 7;
    const float d = load_f16(b);
    u16 sh;
    memcpy(&sh, b + 2, 2);
    const u8 *sl = b + 4;
    const u8 *qs = b + 8 + ib * 16;
    int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * (float)(ls - 32);
    int sumi = 0;
    for (int j = 0; j < 16; ++j) {
        sumi += KVALUES_IQ4NL[qs[j] & 0xF] * yb->qs[j];
        sumi += KVALUES_IQ4NL[qs[j] >> 4]  * yb->qs[j + 16];
    }
    return dl * __half2float(yb->d) * (float)sumi;
}

template <>
__device__ __forceinline__ float dot32<T_MXFP4>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + c * 17;
    const float d = e8m0_half(b[0]);
    int sumi = 0;
    for (int j = 0; j < 16; ++j) {
        sumi += KVALUES_MXFP4[b[1 + j] & 0xF] * yb->qs[j];
        sumi += KVALUES_MXFP4[b[1 + j] >> 4]  * yb->qs[j + 16];
    }
    return d * __half2float(yb->d) * (float)sumi;
}

template <>
__device__ __forceinline__ float dot32<T_TQ2_0>(const u8 *w, int c, const block_q8_1 *yb) {
    const u8 *b = w + (c >> 3) * 66;
    const int sub = c & 7;
    const int j = (sub >> 2) * 32;   // which half of qs
    const int l = sub & 3;           // which 2-bit shift
    const float d = load_f16(b + 64);
    int sumi = 0;
    for (int m = 0; m < 32; ++m) {
        sumi += (((b[j + m] >> (l * 2)) & 3) - 1) * yb->qs[m];
    }
    return d * __half2float(yb->d) * (float)sumi;
}

template <>
__device__ __forceinline__ float dot32<T_TQ1_0>(const u8 *w, int c, const block_q8_1 *yb) {
    // Five base-3 digits per byte for the first 240 values, four per byte for the last 16.
    const u8 *b = w + (c >> 3) * 54;
    const int sub = c & 7;
    const float d = load_f16(b + 52);
    const int pow3[5] = {1, 3, 9, 27, 81};
    int sumi = 0;
    for (int m = 0; m < 32; ++m) {
        int e = sub * 32 + m;
        int t;
        if (e < 160) {
            int n = e / 32, j = e % 32;
            t = (int)(((u32)((b[j] * pow3[n]) & 0xFF) * 3) >> 8);
        } else if (e < 240) {
            int r = e - 160;
            int n = r / 16, j = r % 16;
            t = (int)(((u32)((b[32 + j] * pow3[n]) & 0xFF) * 3) >> 8);
        } else {
            int r = e - 240;
            int n = r / 4, j = r % 4;
            t = (int)(((u32)((b[48 + j] * pow3[n]) & 0xFF) * 3) >> 8);
        }
        sumi += (t - 1) * yb->qs[m];
    }
    return d * __half2float(yb->d) * (float)sumi;
}

// ---------------------------------------------------------------------------------------
// warp_dot: one super-block (or an equivalent run) consumed by a whole warp at once.
//
// `dot32` above gives each lane its own 32-element chunk, which is correct but reads badly:
// the 32 addresses a warp issues in one instruction are spread across hundreds of bytes, so
// most of every fetched cache sector is thrown away. On an RTX 3080 that lands near a tenth
// of peak bandwidth, and a quantized GEMV is bandwidth-bound - the arithmetic is nearly
// free by comparison.
//
// These variants instead give lane `l` the l-th four-byte word of one block's quant region,
// so a warp reads 128 contiguous bytes per instruction. The arithmetic is identical; only
// the assignment of work to lanes changes.
//
// Types without one keep the general path, which is what the long tail of rarely-used
// quantizations needs.
// ---------------------------------------------------------------------------------------

template <int TYPE>
struct has_warp_dot { static const bool value = false; };
template <> struct has_warp_dot<T_Q4_0> { static const bool value = true; };
template <> struct has_warp_dot<T_Q8_0> { static const bool value = true; };
template <> struct has_warp_dot<T_Q4_K> { static const bool value = true; };
template <> struct has_warp_dot<T_Q5_K> { static const bool value = true; };
template <> struct has_warp_dot<T_Q6_K> { static const bool value = true; };

/// 32-element chunks one warp consumes per call.
template <int TYPE>
__device__ __forceinline__ int warp_chunks() {
    switch (TYPE) {
        case T_Q4_K: case T_Q5_K: case T_Q6_K: return 8;  // one super-block
        case T_Q8_0: return 4;                            // 8 lanes per block
        case T_Q4_0: return 8;                            // 4 lanes per block
    }
    return 1;
}

template <int TYPE>
__device__ __forceinline__ float warp_dot(const u8 *w, int span, const block_q8_1 *y, int lane);

// Q4_K: 144-byte super-block, eight 32-element sub-blocks with 6-bit scale/min pairs.
// Lane l reads qs word l, covering four low nibbles of sub-block 2*(l/8) and four high
// nibbles of sub-block 2*(l/8)+1.
template <>
__device__ __forceinline__ float warp_dot<T_Q4_K>(const u8 *w, int sb, const block_q8_1 *y, int lane) {
    const u8 *blk = w + sb * 144;
    const float d = load_f16(blk);
    const float dmin = load_f16(blk + 2);

    int q;
    memcpy(&q, blk + 16 + lane * 4, 4);

    const int g = lane >> 3;
    const int p = (lane & 7) * 4;
    const block_q8_1 *y0 = &y[sb * 8 + 2 * g];
    const block_q8_1 *y1 = &y[sb * 8 + 2 * g + 1];

    const int s0 = dp4(q & 0x0F0F0F0F, y4(y0->qs, p), 0);
    const int s1 = dp4((q >> 4) & 0x0F0F0F0F, y4(y1->qs, p), 0);

    float sc0, mn0, sc1, mn1;
    scale_min_k4(2 * g, blk + 4, &sc0, &mn0);
    scale_min_k4(2 * g + 1, blk + 4, &sc1, &mn1);

    float acc = d * sc0 * __half2float(y0->d) * (float)s0
              + d * sc1 * __half2float(y1->d) * (float)s1;
    // A sub-block's minimum is one term, not eight: only one of the lanes sharing a
    // sub-block contributes it.
    if ((lane & 7) == 0) {
        acc -= dmin * mn0 * __half2float(y0->s) + dmin * mn1 * __half2float(y1->s);
    }
    return acc;
}

// Q5_K: as Q4_K, with a fifth bit per value held in a separate 32-byte plane.
template <>
__device__ __forceinline__ float warp_dot<T_Q5_K>(const u8 *w, int sb, const block_q8_1 *y, int lane) {
    const u8 *blk = w + sb * 176;
    const float d = load_f16(blk);
    const float dmin = load_f16(blk + 2);
    const u8 *qh = blk + 16;
    const u8 *ql = blk + 48;

    const int g = lane >> 3;
    const int p = (lane & 7) * 4;

    int q;
    memcpy(&q, ql + lane * 4, 4);
    int h;
    memcpy(&h, qh + p, 4);

    const block_q8_1 *y0 = &y[sb * 8 + 2 * g];
    const block_q8_1 *y1 = &y[sb * 8 + 2 * g + 1];

    // The extra bit for sub-block 2g is at bit 2g of each qh byte, and for 2g+1 at 2g+1.
    const int m0 = ((h >> (2 * g)) & 0x01010101) << 4;
    const int m1 = ((h >> (2 * g + 1)) & 0x01010101) << 4;

    const int s0 = dp4((q & 0x0F0F0F0F) | m0, y4(y0->qs, p), 0);
    const int s1 = dp4(((q >> 4) & 0x0F0F0F0F) | m1, y4(y1->qs, p), 0);

    float sc0, mn0, sc1, mn1;
    scale_min_k4(2 * g, blk + 4, &sc0, &mn0);
    scale_min_k4(2 * g + 1, blk + 4, &sc1, &mn1);

    float acc = d * sc0 * __half2float(y0->d) * (float)s0
              + d * sc1 * __half2float(y1->d) * (float)s1;
    if ((lane & 7) == 0) {
        acc -= dmin * mn0 * __half2float(y0->s) + dmin * mn1 * __half2float(y1->s);
    }
    return acc;
}

// Q6_K: 210-byte super-block. Lane l reads ql word l; its four bytes hold the low nibbles
// of one 32-element group and the high nibbles of the group two along, with the top two
// bits of every value coming from the qh plane.
template <>
__device__ __forceinline__ float warp_dot<T_Q6_K>(const u8 *w, int sb, const block_q8_1 *y, int lane) {
    const u8 *blk = w + sb * 210;
    const u8 *ql = blk;
    const u8 *qh = blk + 128;
    const int8_t *sc = (const int8_t *)(blk + 192);
    const float d = load_f16(blk + 208);

    const int h = lane >> 4;             // which 128-element half
    const int low16 = lane & 15;
    const int half_sel = low16 >> 3;     // selects the group pair (0,2) or (1,3)
    const int within = (low16 & 7) * 4;  // element offset inside the group

    int q;
    memcpy(&q, ql + lane * 4, 4);
    int hi;
    memcpy(&hi, qh + h * 32 + within, 4);

    const int gA = half_sel;
    const int gB = half_sel + 2;

    const int hA = ((hi >> (2 * gA)) & 0x03030303) << 4;
    const int hB = ((hi >> (2 * gB)) & 0x03030303) << 4;

    const int vA = (q & 0x0F0F0F0F) | hA;
    const int vB = ((q >> 4) & 0x0F0F0F0F) | hB;

    const block_q8_1 *yA = &y[sb * 8 + h * 4 + gA];
    const block_q8_1 *yB = &y[sb * 8 + h * 4 + gB];
    const int aA = y4(yA->qs, within);
    const int aB = y4(yB->qs, within);

    // Q6_K values carry a -32 offset. Keeping the codes unsigned lets dp4a do the work and
    // folds the offset into a second dot against the activation's own bytes.
    const int sA = dp4(vA, aA, 0);
    const int sB = dp4(vB, aB, 0);
    const int tA = dp4(0x01010101, aA, 0);
    const int tB = dp4(0x01010101, aB, 0);

    const int scA = (int)sc[h * 8 + 2 * gA + (within >= 16 ? 1 : 0)];
    const int scB = (int)sc[h * 8 + 2 * gB + (within >= 16 ? 1 : 0)];

    return d * (float)scA * __half2float(yA->d) * (float)(sA - 32 * tA)
         + d * (float)scB * __half2float(yB->d) * (float)(sB - 32 * tB);
}

// Q8_0: eight lanes to a 32-element block, four blocks to a warp.
template <>
__device__ __forceinline__ float warp_dot<T_Q8_0>(const u8 *w, int span, const block_q8_1 *y, int lane) {
    const int b = span * 4 + (lane >> 3);
    const int p = (lane & 7) * 4;
    const u8 *blk = w + b * 34;
    const float d = load_f16(blk);
    const block_q8_1 *yb = &y[b];
    int q;
    memcpy(&q, blk + 2 + p, 4);
    return d * __half2float(yb->d) * (float)dp4(q, y4(yb->qs, p), 0);
}

// Q4_0: four lanes to a 32-element block, eight blocks to a warp.
template <>
__device__ __forceinline__ float warp_dot<T_Q4_0>(const u8 *w, int span, const block_q8_1 *y, int lane) {
    const int b = span * 8 + (lane >> 2);
    const int p = (lane & 3) * 4;
    const u8 *blk = w + b * 18;
    const float d = load_f16(blk);
    const block_q8_1 *yb = &y[b];
    int q;
    memcpy(&q, blk + 2 + p, 4);
    int sumi = dp4(q & 0x0F0F0F0F, y4(yb->qs, p), 0);
    sumi = dp4((q >> 4) & 0x0F0F0F0F, y4(yb->qs, 16 + p), 0) + sumi;
    float acc = d * __half2float(yb->d) * (float)sumi;
    // y = (q - 8) * d, and that offset belongs to the block, not to each quarter of it.
    if ((lane & 3) == 0) {
        acc -= 8.0f * d * __half2float(yb->s);
    }
    return acc;
}

// ---------------------------------------------------------------------------------------
// dequant32: write 32 decoded floats for chunk `c`. Used by get_rows and by the float
// fallbacks; the matmuls never call it.
// ---------------------------------------------------------------------------------------

template <int TYPE>
__device__ __forceinline__ void dequant32(const u8 *w, int c, float *y);

template <>
__device__ __forceinline__ void dequant32<T_F32>(const u8 *w, int c, float *y) {
    for (int j = 0; j < 32; ++j) y[j] = load_f32(w + (c * 32 + j) * 4);
}

template <>
__device__ __forceinline__ void dequant32<T_F16>(const u8 *w, int c, float *y) {
    for (int j = 0; j < 32; ++j) y[j] = load_f16(w + (c * 32 + j) * 2);
}

template <>
__device__ __forceinline__ void dequant32<T_BF16>(const u8 *w, int c, float *y) {
    for (int j = 0; j < 32; ++j) y[j] = load_bf16(w + (c * 32 + j) * 2);
}

template <>
__device__ __forceinline__ void dequant32<T_Q4_0>(const u8 *w, int c, float *y) {
    const u8 *b = w + c * 18;
    const float d = load_f16(b);
    for (int j = 0; j < 16; ++j) {
        y[j]      = ((float)(b[2 + j] & 0xF) - 8.0f) * d;
        y[j + 16] = ((float)(b[2 + j] >> 4)  - 8.0f) * d;
    }
}

template <>
__device__ __forceinline__ void dequant32<T_Q4_1>(const u8 *w, int c, float *y) {
    const u8 *b = w + c * 20;
    const float d = load_f16(b), m = load_f16(b + 2);
    for (int j = 0; j < 16; ++j) {
        y[j]      = (float)(b[4 + j] & 0xF) * d + m;
        y[j + 16] = (float)(b[4 + j] >> 4)  * d + m;
    }
}

template <>
__device__ __forceinline__ void dequant32<T_Q5_0>(const u8 *w, int c, float *y) {
    const u8 *b = w + c * 22;
    const float d = load_f16(b);
    u32 qh; memcpy(&qh, b + 2, 4);
    for (int j = 0; j < 16; ++j) {
        int lo = ((b[6 + j] & 0xF) | (int)(((qh >> j) << 4) & 0x10)) - 16;
        int hi = ((b[6 + j] >> 4)  | (int)((qh >> (j + 12)) & 0x10)) - 16;
        y[j] = lo * d;  y[j + 16] = hi * d;
    }
}

template <>
__device__ __forceinline__ void dequant32<T_Q5_1>(const u8 *w, int c, float *y) {
    const u8 *b = w + c * 24;
    const float d = load_f16(b), m = load_f16(b + 2);
    u32 qh; memcpy(&qh, b + 4, 4);
    for (int j = 0; j < 16; ++j) {
        int lo = (b[8 + j] & 0xF) | (int)(((qh >> j) << 4) & 0x10);
        int hi = (b[8 + j] >> 4)  | (int)((qh >> (j + 12)) & 0x10);
        y[j] = lo * d + m;  y[j + 16] = hi * d + m;
    }
}

template <>
__device__ __forceinline__ void dequant32<T_Q8_0>(const u8 *w, int c, float *y) {
    const u8 *b = w + c * 34;
    const float d = load_f16(b);
    for (int j = 0; j < 32; ++j) y[j] = (float)((int8_t)b[2 + j]) * d;
}

template <>
__device__ __forceinline__ void dequant32<T_Q2_K>(const u8 *w, int c, float *y) {
    const u8 *b = w + (c >> 3) * 84;
    const int sub = c & 7, h = sub >> 2, j = sub & 3;
    const u8 *qs = b + 16 + h * 32;
    const float d = load_f16(b + 80), dmin = load_f16(b + 82);
    const int shift = j * 2, is = h * 8 + j * 2;
    const u8 s0 = b[is], s1 = b[is + 1];
    for (int l = 0; l < 16; ++l) {
        y[l]      = d * (float)(s0 & 0xF) * (float)((qs[l] >> shift) & 3)
                  - dmin * (float)(s0 >> 4);
        y[l + 16] = d * (float)(s1 & 0xF) * (float)((qs[l + 16] >> shift) & 3)
                  - dmin * (float)(s1 >> 4);
    }
}

template <>
__device__ __forceinline__ void dequant32<T_Q3_K>(const u8 *w, int c, float *y) {
    const u8 *b = w + (c >> 3) * 110;
    const int sub = c & 7, h = sub >> 2, j = sub & 3;
    const u8 *hmask = b;
    const u8 *qs = b + 32 + h * 32;
    const u8 *scales = b + 96;
    const float d = load_f16(b + 108);
    const int shift = j * 2, is = h * 8 + j * 2;
    const u8 m = (u8)(1u << (h * 4 + j));
    const float d0 = d * (float)q3k_scale(scales, is);
    const float d1 = d * (float)q3k_scale(scales, is + 1);
    for (int l = 0; l < 16; ++l) {
        y[l]      = d0 * (float)(((qs[l] >> shift) & 3) - ((hmask[l] & m) ? 0 : 4));
        y[l + 16] = d1 * (float)(((qs[l + 16] >> shift) & 3) - ((hmask[l + 16] & m) ? 0 : 4));
    }
}

template <>
__device__ __forceinline__ void dequant32<T_Q4_K>(const u8 *w, int c, float *y) {
    const u8 *b = w + (c >> 3) * 144;
    const int sub = c & 7;
    const float d = load_f16(b), dmin = load_f16(b + 2);
    const u8 *qs = b + 16 + (sub >> 1) * 32;
    float sc, mn;
    scale_min_k4(sub, b + 4, &sc, &mn);
    const int hi = sub & 1;
    for (int l = 0; l < 32; ++l) {
        int nib = hi ? (qs[l] >> 4) : (qs[l] & 0xF);
        y[l] = d * sc * (float)nib - dmin * mn;
    }
}

template <>
__device__ __forceinline__ void dequant32<T_Q5_K>(const u8 *w, int c, float *y) {
    const u8 *b = w + (c >> 3) * 176;
    const int sub = c & 7, g = sub >> 1, hi = sub & 1;
    const float d = load_f16(b), dmin = load_f16(b + 2);
    const u8 *qh = b + 16;
    const u8 *ql = b + 48 + g * 32;
    float sc, mn;
    scale_min_k4(sub, b + 4, &sc, &mn);
    const u8 bit = (u8)((1u + (u32)hi) << (2 * g));
    for (int l = 0; l < 32; ++l) {
        int lo = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
        int hv = (qh[l] & bit) ? 16 : 0;
        y[l] = d * sc * (float)(lo + hv) - dmin * mn;
    }
}

template <>
__device__ __forceinline__ void dequant32<T_Q6_K>(const u8 *w, int c, float *y) {
    const u8 *b = w + (c >> 3) * 210;
    const int sub = c & 7, h = sub >> 2, g = sub & 3;
    const u8 *ql = b + h * 64 + (g & 1) * 32;
    const u8 *qh = b + 128 + h * 32;
    const u8 *sc = b + 192 + h * 8;
    const float d = load_f16(b + 208);
    const int lo_shift = (g >> 1) * 4, hi_shift = g * 2;
    for (int l = 0; l < 32; ++l) {
        int q = (((ql[l] >> lo_shift) & 0xF) | (((qh[l] >> hi_shift) & 3) << 4)) - 32;
        int s = (l < 16) ? (int)((int8_t)sc[2 * g]) : (int)((int8_t)sc[2 * g + 1]);
        y[l] = d * (float)s * (float)q;
    }
}

template <>
__device__ __forceinline__ void dequant32<T_IQ4_NL>(const u8 *w, int c, float *y) {
    const u8 *b = w + c * 18;
    const float d = load_f16(b);
    for (int j = 0; j < 16; ++j) {
        y[j]      = d * (float)KVALUES_IQ4NL[b[2 + j] & 0xF];
        y[j + 16] = d * (float)KVALUES_IQ4NL[b[2 + j] >> 4];
    }
}

template <>
__device__ __forceinline__ void dequant32<T_IQ4_XS>(const u8 *w, int c, float *y) {
    const u8 *b = w + (c >> 3) * 136;
    const int ib = c & 7;
    const float d = load_f16(b);
    u16 sh; memcpy(&sh, b + 2, 2);
    const u8 *qs = b + 8 + ib * 16;
    int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = d * (float)(ls - 32);
    for (int j = 0; j < 16; ++j) {
        y[j]      = dl * (float)KVALUES_IQ4NL[qs[j] & 0xF];
        y[j + 16] = dl * (float)KVALUES_IQ4NL[qs[j] >> 4];
    }
}

template <>
__device__ __forceinline__ void dequant32<T_MXFP4>(const u8 *w, int c, float *y) {
    const u8 *b = w + c * 17;
    const float d = e8m0_half(b[0]);
    for (int j = 0; j < 16; ++j) {
        y[j]      = d * (float)KVALUES_MXFP4[b[1 + j] & 0xF];
        y[j + 16] = d * (float)KVALUES_MXFP4[b[1 + j] >> 4];
    }
}

template <>
__device__ __forceinline__ void dequant32<T_TQ2_0>(const u8 *w, int c, float *y) {
    const u8 *b = w + (c >> 3) * 66;
    const int sub = c & 7, j = (sub >> 2) * 32, l = sub & 3;
    const float d = load_f16(b + 64);
    for (int m = 0; m < 32; ++m) y[m] = (float)(((b[j + m] >> (l * 2)) & 3) - 1) * d;
}

template <>
__device__ __forceinline__ void dequant32<T_TQ1_0>(const u8 *w, int c, float *y) {
    const u8 *b = w + (c >> 3) * 54;
    const int sub = c & 7;
    const float d = load_f16(b + 52);
    const int pow3[5] = {1, 3, 9, 27, 81};
    for (int m = 0; m < 32; ++m) {
        int e = sub * 32 + m, t;
        if (e < 160)      { int n = e / 32, j = e % 32;       t = (int)(((u32)((b[j]      * pow3[n]) & 0xFF) * 3) >> 8); }
        else if (e < 240) { int r = e - 160, n = r / 16, j = r % 16; t = (int)(((u32)((b[32 + j] * pow3[n]) & 0xFF) * 3) >> 8); }
        else              { int r = e - 240, n = r / 4,  j = r % 4;  t = (int)(((u32)((b[48 + j] * pow3[n]) & 0xFF) * 3) >> 8); }
        y[m] = (float)(t - 1) * d;
    }
}

#endif // GGUF_QUANTS_CUH

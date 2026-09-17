// Shared block decoding for the Vulkan kernels.
//
// This file is prepended to every shader by build.rs rather than #included, which keeps the
// shader compiler free of an include resolver.
//
// Weights are storage buffers of `uint` holding the GGUF bytes verbatim. Vulkan 1.0 has no
// byte-addressable storage, and requiring VK_KHR_8bit_storage would exclude older drivers
// on exactly the hardware this backend exists to reach, so bytes are extracted with shifts.
// Block strides like Q4_0's 18 bytes are not 4-aligned, so every access goes through
// `getb` rather than assuming alignment.

#define T_F32     0
#define T_F16     1
#define T_Q4_0    2
#define T_Q4_1    3
#define T_Q5_0    6
#define T_Q5_1    7
#define T_Q8_0    8
#define T_Q2_K   10
#define T_Q3_K   11
#define T_Q4_K   12
#define T_Q5_K   13
#define T_Q6_K   14
#define T_IQ4_NL 20
#define T_IQ4_XS 23
#define T_BF16   30
#define T_TQ1_0  34
#define T_TQ2_0  35
#define T_MXFP4  39

// Selected at pipeline creation, so the compiler folds away every branch but one.
layout(constant_id = 0) const int QUANT_TYPE = T_Q4_K;

const int KV_IQ4NL[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10,
                                 1, 13, 25, 38, 53, 69, 89, 113);
const int KV_MXFP4[16] = int[16](0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);

uint getb(uint base) {
    return (WEIGHTS[base >> 2u] >> ((base & 3u) * 8u)) & 0xFFu;
}

int getb_signed(uint base) {
    return int(getb(base) << 24u) >> 24;
}

uint getu16(uint base) {
    return getb(base) | (getb(base + 1u) << 8u);
}

uint getu32(uint base) {
    return getb(base) | (getb(base + 1u) << 8u) | (getb(base + 2u) << 16u) | (getb(base + 3u) << 24u);
}

// IEEE half to float. `unpackHalf2x16` is core in GLSL 4.2 and universally available in
// SPIR-V compute, so there is no reason to decode the bits by hand.
float half_at(uint base) {
    return unpackHalf2x16(getu16(base)).x;
}

float bf16_at(uint base) {
    return uintBitsToFloat(getu16(base) << 16u);
}

float f32_at(uint base) {
    return uintBitsToFloat(getu32(base));
}

// E8M0 exponent byte to 2^(e-127)/2.
float e8m0_half(uint e) {
    uint bits = e < 2u ? (0x00200000u << e) : ((e - 1u) << 23u);
    return uintBitsToFloat(bits);
}

// Bytes per block and elements per block, needed to walk a row.
uint type_size(int t) {
    switch (t) {
        case T_F32: return 4u;
        case T_F16: case T_BF16: return 2u;
        case T_Q4_0: case T_IQ4_NL: return 18u;
        case T_Q4_1: return 20u;
        case T_Q5_0: return 22u;
        case T_Q5_1: return 24u;
        case T_Q8_0: return 34u;
        case T_Q2_K: return 84u;
        case T_Q3_K: return 110u;
        case T_Q4_K: return 144u;
        case T_Q5_K: return 176u;
        case T_Q6_K: return 210u;
        case T_IQ4_XS: return 136u;
        case T_TQ1_0: return 54u;
        case T_TQ2_0: return 66u;
        case T_MXFP4: return 17u;
    }
    return 1u;
}

bool is_super(int t) {
    switch (t) {
        case T_Q2_K: case T_Q3_K: case T_Q4_K: case T_Q5_K: case T_Q6_K:
        case T_IQ4_XS: case T_TQ1_0: case T_TQ2_0:
            return true;
    }
    return false;
}

// Byte offset of the block header covering 32-element chunk `c` of a row starting at `row`.
uint block_base(uint row, uint c) {
    if (is_super(QUANT_TYPE)) {
        return row + (c >> 3u) * type_size(QUANT_TYPE);
    }
    return row + c * type_size(QUANT_TYPE);
}

// Q4_K / Q5_K: eight 6-bit scale/min pairs packed into twelve bytes.
void scale_min_k4(uint sc_base, int j, out float d, out float m) {
    if (j < 4) {
        d = float(getb(sc_base + uint(j)) & 63u);
        m = float(getb(sc_base + uint(j) + 4u) & 63u);
    } else {
        uint a = getb(sc_base + uint(j) + 4u);
        uint b = getb(sc_base + uint(j) - 4u);
        uint c = getb(sc_base + uint(j));
        d = float((a & 0xFu) | ((b >> 6u) << 4u));
        m = float((a >> 4u) | ((c >> 6u) << 4u));
    }
}

// Q3_K: sixteen 6-bit signed scales packed into twelve bytes.
int q3k_scale(uint sc_base, int k) {
    uint lo4, hi2;
    if (k < 4)       { lo4 = getb(sc_base + uint(k)) & 0xFu;            hi2 = getb(sc_base + 8u + uint(k)) & 3u; }
    else if (k < 8)  { lo4 = getb(sc_base + uint(k)) & 0xFu;            hi2 = (getb(sc_base + 4u + uint(k)) >> 2u) & 3u; }
    else if (k < 12) { lo4 = (getb(sc_base + uint(k) - 8u) >> 4u) & 0xFu; hi2 = (getb(sc_base + uint(k)) >> 4u) & 3u; }
    else             { lo4 = (getb(sc_base + uint(k) - 8u) >> 4u) & 0xFu; hi2 = (getb(sc_base + uint(k) - 4u) >> 6u) & 3u; }
    return (int((lo4 | (hi2 << 4u)) << 24u) >> 24) - 32;
}

// ---------------------------------------------------------------------------------------
// Decode 32 elements of chunk `c` into `y`. One function per family, dispatched on the
// specialization constant so only the selected branch survives compilation.
// ---------------------------------------------------------------------------------------
void dequant32(uint row, uint c, out float y[32]) {
    uint b = block_base(row, c);
    int t = QUANT_TYPE;

    if (t == T_F32) {
        for (int j = 0; j < 32; ++j) y[j] = f32_at(row + (c * 32u + uint(j)) * 4u);
    } else if (t == T_F16) {
        for (int j = 0; j < 32; ++j) y[j] = half_at(row + (c * 32u + uint(j)) * 2u);
    } else if (t == T_BF16) {
        for (int j = 0; j < 32; ++j) y[j] = bf16_at(row + (c * 32u + uint(j)) * 2u);
    } else if (t == T_Q4_0) {
        float d = half_at(b);
        for (int j = 0; j < 16; ++j) {
            uint q = getb(b + 2u + uint(j));
            y[j]      = (float(q & 0xFu) - 8.0) * d;
            y[j + 16] = (float(q >> 4u)  - 8.0) * d;
        }
    } else if (t == T_Q4_1) {
        float d = half_at(b), m = half_at(b + 2u);
        for (int j = 0; j < 16; ++j) {
            uint q = getb(b + 4u + uint(j));
            y[j]      = float(q & 0xFu) * d + m;
            y[j + 16] = float(q >> 4u)  * d + m;
        }
    } else if (t == T_Q5_0) {
        float d = half_at(b);
        uint qh = getu32(b + 2u);
        for (int j = 0; j < 16; ++j) {
            uint q = getb(b + 6u + uint(j));
            uint hl = ((qh >> uint(j)) << 4u) & 0x10u;
            uint hh = (qh >> (uint(j) + 12u)) & 0x10u;
            y[j]      = float(int((q & 0xFu) | hl) - 16) * d;
            y[j + 16] = float(int((q >> 4u)  | hh) - 16) * d;
        }
    } else if (t == T_Q5_1) {
        float d = half_at(b), m = half_at(b + 2u);
        uint qh = getu32(b + 4u);
        for (int j = 0; j < 16; ++j) {
            uint q = getb(b + 8u + uint(j));
            uint hl = ((qh >> uint(j)) << 4u) & 0x10u;
            uint hh = (qh >> (uint(j) + 12u)) & 0x10u;
            y[j]      = float((q & 0xFu) | hl) * d + m;
            y[j + 16] = float((q >> 4u)  | hh) * d + m;
        }
    } else if (t == T_Q8_0) {
        float d = half_at(b);
        for (int j = 0; j < 32; ++j) y[j] = float(getb_signed(b + 2u + uint(j))) * d;
    } else if (t == T_Q2_K) {
        uint sub = c & 7u, h = sub >> 2u, j4 = sub & 3u;
        uint qs = b + 16u + h * 32u;
        float d = half_at(b + 80u), dmin = half_at(b + 82u);
        uint shift = j4 * 2u;
        uint is = h * 8u + j4 * 2u;
        uint s0 = getb(b + is), s1 = getb(b + is + 1u);
        for (int l = 0; l < 16; ++l) {
            y[l]      = d * float(s0 & 0xFu) * float((getb(qs + uint(l)) >> shift) & 3u)
                      - dmin * float(s0 >> 4u);
            y[l + 16] = d * float(s1 & 0xFu) * float((getb(qs + uint(l) + 16u) >> shift) & 3u)
                      - dmin * float(s1 >> 4u);
        }
    } else if (t == T_Q3_K) {
        uint sub = c & 7u, h = sub >> 2u, j4 = sub & 3u;
        uint hmask = b;
        uint qs = b + 32u + h * 32u;
        uint scales = b + 96u;
        float d = half_at(b + 108u);
        uint shift = j4 * 2u;
        uint m = 1u << (h * 4u + j4);
        int is = int(h * 8u + j4 * 2u);
        float d0 = d * float(q3k_scale(scales, is));
        float d1 = d * float(q3k_scale(scales, is + 1));
        for (int l = 0; l < 16; ++l) {
            int hv0 = (getb(hmask + uint(l)) & m) != 0u ? 0 : 4;
            int hv1 = (getb(hmask + uint(l) + 16u) & m) != 0u ? 0 : 4;
            y[l]      = d0 * float(int((getb(qs + uint(l)) >> shift) & 3u) - hv0);
            y[l + 16] = d1 * float(int((getb(qs + uint(l) + 16u) >> shift) & 3u) - hv1);
        }
    } else if (t == T_Q4_K) {
        uint sub = c & 7u;
        float d = half_at(b), dmin = half_at(b + 2u);
        uint qs = b + 16u + (sub >> 1u) * 32u;
        float sc, mn;
        scale_min_k4(b + 4u, int(sub), sc, mn);
        bool hi = (sub & 1u) != 0u;
        for (int l = 0; l < 32; ++l) {
            uint q = getb(qs + uint(l));
            uint nib = hi ? (q >> 4u) : (q & 0xFu);
            y[l] = d * sc * float(nib) - dmin * mn;
        }
    } else if (t == T_Q5_K) {
        uint sub = c & 7u, g = sub >> 1u;
        bool hi = (sub & 1u) != 0u;
        float d = half_at(b), dmin = half_at(b + 2u);
        uint qh = b + 16u;
        uint ql = b + 48u + g * 32u;
        float sc, mn;
        scale_min_k4(b + 4u, int(sub), sc, mn);
        uint bit = (hi ? 2u : 1u) << (2u * g);
        for (int l = 0; l < 32; ++l) {
            uint q = getb(ql + uint(l));
            uint lo = hi ? (q >> 4u) : (q & 0xFu);
            uint hv = (getb(qh + uint(l)) & bit) != 0u ? 16u : 0u;
            y[l] = d * sc * float(lo + hv) - dmin * mn;
        }
    } else if (t == T_Q6_K) {
        uint sub = c & 7u, h = sub >> 2u, g = sub & 3u;
        uint ql = b + h * 64u + (g & 1u) * 32u;
        uint qh = b + 128u + h * 32u;
        uint sc = b + 192u + h * 8u;
        float d = half_at(b + 208u);
        uint lo_shift = (g >> 1u) * 4u;
        uint hi_shift = g * 2u;
        for (int l = 0; l < 32; ++l) {
            uint lo = (getb(ql + uint(l)) >> lo_shift) & 0xFu;
            uint hv = ((getb(qh + uint(l)) >> hi_shift) & 3u) << 4u;
            int q = int(lo | hv) - 32;
            int s = l < 16 ? getb_signed(sc + 2u * g) : getb_signed(sc + 2u * g + 1u);
            y[l] = d * float(s) * float(q);
        }
    } else if (t == T_IQ4_NL) {
        float d = half_at(b);
        for (int j = 0; j < 16; ++j) {
            uint q = getb(b + 2u + uint(j));
            y[j]      = d * float(KV_IQ4NL[q & 0xFu]);
            y[j + 16] = d * float(KV_IQ4NL[q >> 4u]);
        }
    } else if (t == T_IQ4_XS) {
        uint ib = c & 7u;
        float d = half_at(b);
        uint sh = getu16(b + 2u);
        uint qs = b + 8u + ib * 16u;
        int ls = int((getb(b + 4u + (ib >> 1u)) >> (4u * (ib & 1u))) & 0xFu)
               | int(((sh >> (2u * ib)) & 3u) << 4u);
        float dl = d * float(ls - 32);
        for (int j = 0; j < 16; ++j) {
            uint q = getb(qs + uint(j));
            y[j]      = dl * float(KV_IQ4NL[q & 0xFu]);
            y[j + 16] = dl * float(KV_IQ4NL[q >> 4u]);
        }
    } else if (t == T_MXFP4) {
        float d = e8m0_half(getb(b));
        for (int j = 0; j < 16; ++j) {
            uint q = getb(b + 1u + uint(j));
            y[j]      = d * float(KV_MXFP4[q & 0xFu]);
            y[j + 16] = d * float(KV_MXFP4[q >> 4u]);
        }
    } else if (t == T_TQ2_0) {
        uint sub = c & 7u;
        uint j = (sub >> 2u) * 32u;
        uint l2 = sub & 3u;
        float d = half_at(b + 64u);
        for (int m = 0; m < 32; ++m) {
            y[m] = float(int((getb(b + j + uint(m)) >> (l2 * 2u)) & 3u) - 1) * d;
        }
    } else if (t == T_TQ1_0) {
        uint sub = c & 7u;
        float d = half_at(b + 52u);
        int pow3[5] = int[5](1, 3, 9, 27, 81);
        for (int m = 0; m < 32; ++m) {
            uint e = sub * 32u + uint(m);
            uint src; int n;
            if (e < 160u)      { n = int(e / 32u); src = b + (e % 32u); }
            else if (e < 240u) { uint r = e - 160u; n = int(r / 16u); src = b + 32u + (r % 16u); }
            else               { uint r = e - 240u; n = int(r / 4u);  src = b + 48u + (r % 4u); }
            uint q = (getb(src) * uint(pow3[n])) & 0xFFu;
            y[m] = float(int((q * 3u) >> 8u) - 1) * d;
        }
    } else {
        for (int j = 0; j < 32; ++j) y[j] = 0.0;
    }
}

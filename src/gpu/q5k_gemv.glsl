#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; } pc;

// Q5_K superblock = 44 u32s per 256 weights:
//   [0]      = d (f32)
//   [1]      = dmin (f32)
//   [2..4]   = scales (12 bytes = 3 u32s)
//   [5..12]  = qh (32 bytes = 8 u32s, high bits)
//   [13..44] = qs (128 bytes = 32 u32s, low 4 bits)

shared float sdata[256];

// Extract 6-bit scale and min from the packed 12-byte scales array
void get_scale_min(uint j, uint s0, uint s1, uint s2,
                   out float sc_out, out float mn_out) {
    // 12 bytes packed as 3 u32s, encoding 8 pairs of (scale, min) at 6 bits each
    // Same layout as Q4_K scales
    uint byte_idx = j;
    uint scales_bytes[12];
    scales_bytes[0]  = s0 & 0xFFu; scales_bytes[1]  = (s0 >> 8u) & 0xFFu;
    scales_bytes[2]  = (s0 >> 16u) & 0xFFu; scales_bytes[3]  = (s0 >> 24u) & 0xFFu;
    scales_bytes[4]  = s1 & 0xFFu; scales_bytes[5]  = (s1 >> 8u) & 0xFFu;
    scales_bytes[6]  = (s1 >> 16u) & 0xFFu; scales_bytes[7]  = (s1 >> 24u) & 0xFFu;
    scales_bytes[8]  = s2 & 0xFFu; scales_bytes[9]  = (s2 >> 8u) & 0xFFu;
    scales_bytes[10] = (s2 >> 16u) & 0xFFu; scales_bytes[11] = (s2 >> 24u) & 0xFFu;

    if (j < 4u) {
        sc_out = float(scales_bytes[j] & 63u);
        mn_out = float(scales_bytes[j + 4u] & 63u);
    } else {
        uint off = j - 4u;
        sc_out = float((scales_bytes[off + 4u] >> 6u) | ((scales_bytes[off + 8u] & 0xFu) << 2u));
        mn_out = float((scales_bytes[off + 4u + 4u] >> 6u) | ((scales_bytes[off + 8u] >> 4u) << 2u));
    }
}

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint tid = gl_LocalInvocationID.x;
    float sum = 0.0;

    uint row_base = row * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint blk = (row_base + b) * 44u;
        uint vb = b * 256u;
        float df   = uintBitsToFloat(mat.data[blk]);
        float dmin = uintBitsToFloat(mat.data[blk + 1u]);
        uint s0 = mat.data[blk + 2u];
        uint s1 = mat.data[blk + 3u];
        uint s2 = mat.data[blk + 4u];

        // tid [0..255] maps to weight index
        uint i = tid;
        uint quarter = i / 64u;    // [0..3]
        uint li = i % 64u;         // [0..63]
        uint sub = li / 32u;       // 0 or 1 within quarter
        uint pos = li % 32u;       // [0..31]

        uint is = quarter * 2u + sub;
        float sc_val, mn_val;
        get_scale_min(is, s0, s1, s2, sc_val, mn_val);

        // Get low 4 bits from qs
        uint qs_idx = quarter * 32u + pos;
        uint qs_word = mat.data[blk + 13u + qs_idx / 4u];
        uint qs_byte = (qs_word >> ((qs_idx % 4u) * 8u)) & 0xFFu;
        uint lo = (sub == 0u) ? (qs_byte & 0xFu) : (qs_byte >> 4u);

        // Get high bit from qh
        uint qh_word = mat.data[blk + 5u + pos / 4u];
        uint qh_byte = (qh_word >> ((pos % 4u) * 8u)) & 0xFFu;
        uint hbit = (qh_byte >> (quarter * 2u + sub)) & 1u;

        float val = float(lo | (hbit << 4u));
        sum += (df * sc_val * val - dmin * mn_val) * vin.data[vb + i];
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[row] = sdata[0];
}

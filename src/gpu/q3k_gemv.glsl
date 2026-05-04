#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; } pc;

// Q3_K superblock = 28 u32s per 256 weights:
//   [0]       = d (f32)
//   [1..8]    = hmask (32 bytes = 8 u32s)
//   [9..24]   = qs (64 bytes = 16 u32s, 2-bit values)
//   [25..27]  = scales_packed (12 bytes = 3 u32s, 16x 6-bit scales)

shared float sdata[256];

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint tid = gl_LocalInvocationID.x;
    float sum = 0.0;

    uint row_base = row * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint blk = (row_base + b) * 28u;
        uint vb = b * 256u;
        float d_val = uintBitsToFloat(mat.data[blk]);

        // tid maps to weight index [0..255]
        uint i = tid;
        // Extract 2-bit quant from qs
        uint qs_word = mat.data[blk + 9u + i / 16u];
        uint qs_byte = (qs_word >> ((i % 16u) * 2u)) & 3u;

        // Extract high bit from hmask
        uint hm_word = mat.data[blk + 1u + (i % 32u) / 4u];
        uint hm_byte_in_word = (hm_word >> (((i % 32u) % 4u) * 8u)) & 0xFFu;
        uint hbit = (hm_byte_in_word >> (i / 32u)) & 1u;

        // Combine: 3-bit value = (low2 | high1<<2) - 4
        int q = int(qs_byte | (hbit << 2u)) - 4;

        // Extract 6-bit scale for this sub-block (i/16)
        uint si = i / 16u;
        uint scales_u32_0 = mat.data[blk + 25u];
        uint scales_u32_1 = mat.data[blk + 26u];
        uint scales_u32_2 = mat.data[blk + 27u];
        // Pack 12 bytes into accessible form
        uint sb = si;
        uint scale_byte;
        if (sb < 4u) {
            scale_byte = (scales_u32_0 >> (sb * 8u)) & 0x3Fu;
        } else if (sb < 8u) {
            uint raw_lo = (mat.data[blk + 25u + (sb/4u)] >> ((sb%4u) * 8u)) & 0xFu;
            uint raw_hi = (mat.data[blk + 25u + (sb/4u - 1u)] >> ((sb%4u) * 8u + 6u)) & 0x3u;
            scale_byte = raw_lo | (raw_hi << 4u);
        } else if (sb < 12u) {
            scale_byte = (scales_u32_2 >> ((sb - 8u) * 8u / 2u)) & 0x3Fu;
        } else {
            uint raw_lo = (mat.data[blk + 26u + ((sb-8u)/4u)] >> (((sb-8u)%4u) * 8u + 4u)) & 0xFu;
            uint raw_hi = (mat.data[blk + 25u + ((sb-8u)/4u)] >> (((sb-8u)%4u) * 8u + 6u)) & 0x3u;
            scale_byte = raw_lo | (raw_hi << 4u);
        }
        int sc = int(scale_byte) - 32;

        sum += d_val * float(sc) * float(q) * vin.data[vb + i];
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[row] = sdata[0];
}

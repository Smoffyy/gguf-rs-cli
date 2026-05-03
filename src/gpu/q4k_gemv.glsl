#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; } pc;

// Q4K block = 48 u32s per 256 weights:
//   [0..7]   = 8 pre-scaled d values (f32)
//   [8..15]  = 8 pre-scaled min values (f32)
//   [16..47] = 32 nibble words (128 bytes = 256 nibbles)

shared float sdata[256];

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint tid = gl_LocalInvocationID.x;
    float sum = 0.0;
    uint row_base = row * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint blk = (row_base + b) * 48u;
        uint vb = b * 256u;
        // tid maps to weight index [0..255]
        uint sub = tid >> 6u;       // sub-block [0..3]
        uint p   = tid & 63u;       // position within sub-block
        float sc, mn;
        uint nib;
        if (p < 32u) {
            sc  = uintBitsToFloat(mat.data[blk + sub * 2u]);
            mn  = uintBitsToFloat(mat.data[blk + 8u + sub * 2u]);
            nib = (mat.data[blk + 16u + sub * 8u + (p >> 2u)] >> ((p & 3u) * 8u)) & 0xFu;
        } else {
            sc  = uintBitsToFloat(mat.data[blk + sub * 2u + 1u]);
            mn  = uintBitsToFloat(mat.data[blk + 8u + sub * 2u + 1u]);
            uint lp = p - 32u;
            nib = (mat.data[blk + 16u + sub * 8u + (lp >> 2u)] >> ((lp & 3u) * 8u + 4u)) & 0xFu;
        }
        sum += (sc * float(nib) - mn) * vin.data[vb + tid];
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[row] = sdata[0];
}

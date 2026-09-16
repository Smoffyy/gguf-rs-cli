#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; uint row_start; } pc;


shared float sdata[256];

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint tid = gl_LocalInvocationID.x;
    float sum = 0.0;
    uint row_base = (row + pc.row_start) * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint blk = (row_base + b) * 48u;
        uint vb  = b * 256u;
        uint i   = tid;

        uint p       = i / 64u;
        uint within  = i % 64u;
        bool isLow   = within < 32u;
        uint l       = within % 32u;
        uint j       = p * 2u + (isLow ? 0u : 1u);

        float sc = uintBitsToFloat(mat.data[blk + j]);
        float mn = uintBitsToFloat(mat.data[blk + 8u + j]);

        uint k         = p * 32u + l;
        uint qs_word   = mat.data[blk + 16u + k / 4u];
        uint qs_byte   = (qs_word >> ((k % 4u) * 8u)) & 0xFFu;
        uint nib = isLow ? (qs_byte & 0xFu) : (qs_byte >> 4u);

        sum += (sc * float(nib) - mn) * vin.data[vb + i];
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[row] = sdata[0];
}

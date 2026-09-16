#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; } pc;


shared float sdata[256];

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint tid = gl_LocalInvocationID.x;
    float sum = 0.0;
    uint row_base = row * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint base = (row_base + b) * 64u;
        uint vb   = b * 256u;
        uint i    = tid;

        uint nblock = i / 128u;
        uint local  = i % 128u;
        uint sub    = local / 32u;
        uint l      = local % 32u;

        uint qloff = nblock * 64u;
        uint qhoff = nblock * 32u;
        uint scoff = nblock * 8u;

        uint ql_byte_idx = qloff + l + ((sub == 1u || sub == 3u) ? 32u : 0u);
        uint ql_word = mat.data[base + 16u + ql_byte_idx/4u];
        uint ql_byte = (ql_word >> ((ql_byte_idx%4u)*8u)) & 0xFFu;
        uint nib = (sub == 0u || sub == 1u) ? (ql_byte & 0xFu) : (ql_byte >> 4u);

        uint qh_byte_idx = qhoff + l;
        uint qh_word = mat.data[base + 48u + qh_byte_idx/4u];
        uint qh_byte = (qh_word >> ((qh_byte_idx%4u)*8u)) & 0xFFu;
        uint hi2 = (qh_byte >> (sub*2u)) & 3u;

        uint scale_idx = scoff + (l/16u) + sub*2u;
        float sc = uintBitsToFloat(mat.data[base + scale_idx]);

        int q = int(nib | (hi2 << 4u)) - 32;
        sum += sc * float(q) * vin.data[vb + i];
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[row] = sdata[0];
}

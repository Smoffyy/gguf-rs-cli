#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; } pc;


shared float sdata[256];

uint sc_byte(uint blk, uint k) {
    return (mat.data[blk + 2u + k / 4u] >> ((k % 4u) * 8u)) & 0xFFu;
}

void get_scale_min(uint j, uint blk, out float sc_out, out float mn_out) {
    uint d_val, m_val;
    if (j < 4u) {
        d_val = sc_byte(blk, j) & 63u;
        m_val = sc_byte(blk, j + 4u) & 63u;
    } else {
        uint off = j - 4u;
        d_val = (sc_byte(blk, off + 8u) & 0xFu) | ((sc_byte(blk, off) >> 6u) << 4u);
        m_val = (sc_byte(blk, off + 8u) >> 4u) | ((sc_byte(blk, off + 4u) >> 6u) << 4u);
    }
    sc_out = float(d_val);
    mn_out = float(m_val);
}

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint tid = gl_LocalInvocationID.x;
    float sum = 0.0;
    uint row_base = row * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint blk = (row_base + b) * 45u;
        uint vb  = b * 256u;
        uint i   = tid;
        float df   = uintBitsToFloat(mat.data[blk]);
        float dmin = uintBitsToFloat(mat.data[blk + 1u]);

        uint p      = i / 64u;
        uint within = i % 64u;
        bool isLow  = within < 32u;
        uint l      = within % 32u;
        uint j      = p * 2u + (isLow ? 0u : 1u);

        float sc_val, mn_val;
        get_scale_min(j, blk, sc_val, mn_val);

        uint qs_byte_idx = p * 32u + l;
        uint qs_word = mat.data[blk + 13u + qs_byte_idx / 4u];
        uint qs_byte = (qs_word >> ((qs_byte_idx % 4u) * 8u)) & 0xFFu;
        uint nib = isLow ? (qs_byte & 0xFu) : (qs_byte >> 4u);

        uint qh_byte_idx = l;
        uint qh_word = mat.data[blk + 5u + qh_byte_idx / 4u];
        uint qh_byte = (qh_word >> ((qh_byte_idx % 4u) * 8u)) & 0xFFu;
        uint hbitpos = p * 2u + (isLow ? 0u : 1u);
        uint hbit = (qh_byte >> hbitpos) & 1u;

        float val = float(nib) + float(hbit) * 16.0;
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

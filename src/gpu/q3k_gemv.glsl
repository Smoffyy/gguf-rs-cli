#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; } pc;


shared float sdata[256];

uint sc_byte(uint blk, uint k) {
    return (mat.data[blk + 25u + k/4u] >> ((k%4u)*8u)) & 0xFFu;
}

float decode_scale(uint blk, uint k) {
    uint lo4; uint hi2;
    if (k < 4u) {
        lo4 = sc_byte(blk, k) & 0xFu;
        hi2 = (sc_byte(blk, 8u+k) >> 0u) & 0x3u;
    } else if (k < 8u) {
        lo4 = sc_byte(blk, k) & 0xFu;
        hi2 = (sc_byte(blk, 8u+k-4u) >> 2u) & 0x3u;
    } else if (k < 12u) {
        lo4 = (sc_byte(blk, k-8u) >> 4u) & 0xFu;
        hi2 = (sc_byte(blk, k) >> 4u) & 0x3u;
    } else {
        lo4 = (sc_byte(blk, k-8u) >> 4u) & 0xFu;
        hi2 = (sc_byte(blk, k-4u) >> 6u) & 0x3u;
    }
    return float(int(lo4 | (hi2 << 4u)) - 32);
}

void main() {
    uint row = gl_WorkGroupID.x;
    if (row >= pc.rows) return;
    uint tid = gl_LocalInvocationID.x;
    float sum = 0.0;
    uint row_base = row * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint blk = (row_base + b) * 28u;
        uint vb  = b * 256u;
        float d_val = uintBitsToFloat(mat.data[blk]);
        uint i = tid;

        uint nblock = i / 128u;
        uint shift  = ((i / 32u) % 4u) * 2u;
        uint qs_byte_idx = nblock * 32u + i % 32u;
        uint qs_word = mat.data[blk + 9u + qs_byte_idx/4u];
        uint qs_byte = (qs_word >> ((qs_byte_idx%4u)*8u)) & 0xFFu;
        uint qs_2bit = (qs_byte >> shift) & 3u;

        uint hm_byte_idx = i % 32u;
        uint hm_word = mat.data[blk + 1u + hm_byte_idx/4u];
        uint hm_byte = (hm_word >> ((hm_byte_idx%4u)*8u)) & 0xFFu;
        uint hbit    = (hm_byte >> (i/32u)) & 1u;

        int q = int(qs_2bit | (hbit << 2u)) - 4;
        float sc = decode_scale(blk, i/16u);
        sum += d_val * sc * float(q) * vin.data[vb + i];
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        barrier();
    }
    if (tid == 0u) vout.data[row] = sdata[0];
}

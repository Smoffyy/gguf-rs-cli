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

    // bpr = cols/256, each block = 64 u32s: [16 scales, 32 ql, 16 qh]
    for (uint b = tid; b < pc.bpr; b += 256u) {
        uint base = (row * pc.bpr + b) * 64u;
        uint vb = b * 256u;

        // Two halves of 128 weights each
        for (uint hf = 0u; hf < 2u; hf++) {
            uint ql_off = base + 16u + hf * 16u;
            uint qh_off = base + 48u + hf * 8u;
            uint si     = hf * 8u;

            for (uint l = 0u; l < 32u; l++) {
                uint ql_word = mat.data[ql_off + l/4u];
                uint ql_byte = (ql_word >> ((l%4u)*8u)) & 0xFFu;
                uint qh_word = mat.data[qh_off + l/4u];
                uint qh_byte = (qh_word >> ((l%4u)*8u)) & 0xFFu;

                uint is = l / 16u;
                uint vo = vb + hf * 128u;

                int q1 = int((ql_byte & 0xFu) | ((qh_byte & 3u) << 4u)) - 32;
                float s1 = uintBitsToFloat(mat.data[base + si + is]);
                sum += s1 * float(q1) * vin.data[vo + l];

                int q2 = int(((ql_byte >> 4u) & 0xFu) | (((qh_byte >> 4u) & 3u) << 4u)) - 32;
                float s3 = uintBitsToFloat(mat.data[base + si + is + 4u]);
                sum += s3 * float(q2) * vin.data[vo + l + 64u];
            }

            // Second ql block (offset +32 bytes from hf start)
            for (uint l = 0u; l < 32u; l++) {
                uint ql_word = mat.data[ql_off + 8u + l/4u];
                uint ql_byte = (ql_word >> ((l%4u)*8u)) & 0xFFu;
                uint qh_word = mat.data[qh_off + l/4u];
                uint qh_byte = (qh_word >> ((l%4u)*8u)) & 0xFFu;

                uint is = l / 16u;
                uint vo = vb + hf * 128u;

                int q2 = int((ql_byte & 0xFu) | (((qh_byte >> 2u) & 3u) << 4u)) - 32;
                float s2 = uintBitsToFloat(mat.data[base + si + is + 2u]);
                sum += s2 * float(q2) * vin.data[vo + l + 32u];

                int q4 = int(((ql_byte >> 4u) & 0xFu) | (((qh_byte >> 6u) & 3u) << 4u)) - 32;
                float s4 = uintBitsToFloat(mat.data[base + si + is + 6u]);
                sum += s4 * float(q4) * vin.data[vo + l + 96u];
            }
        }
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[row] = sdata[0];
}
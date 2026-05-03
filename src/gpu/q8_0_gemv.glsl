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

    // bpr = cols/32, each block = 9 u32s: [1 scale, 8 i8-packed]
    for (uint b = tid; b < pc.bpr; b += 256u) {
        uint base = (row * pc.bpr + b) * 9u;
        float sc = uintBitsToFloat(mat.data[base]);
        uint vb = b * 32u;
        for (uint w = 0u; w < 8u; w++) {
            uint pk = mat.data[base+1u+w];
            for (uint i = 0u; i < 4u; i++) {
                int q = int((pk >> (i*8u)) & 0xFFu);
                if (q > 127) q -= 256;
                sum += float(q) * sc * vin.data[vb + w*4u + i];
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

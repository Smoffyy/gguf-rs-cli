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

    for (uint b = tid; b < pc.bpr; b += 256u) {
        uint base = (row * pc.bpr + b) * 5u;
        uint sr = mat.data[base] & 0xFFFFu;
        float scale = uintBitsToFloat((sr & 0x8000u) != 0u
            ? (0xE0000000u | (sr << 13u)) : ((sr + 0x38000u) << 13u));
        uint vb = b * 32u;
        for (uint w = 0u; w < 4u; w++) {
            uint packed = mat.data[base+1u+w];
            for (uint i = 0u; i < 4u; i++) {
                uint byte = (packed >> (i*8u)) & 0xFFu;
                uint j = w*4u+i;
                sum += float(int(byte & 0xFu)-8) * scale * vin.data[vb+j];
                sum += float(int(byte >> 4u)-8)  * scale * vin.data[vb+j+16u];
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
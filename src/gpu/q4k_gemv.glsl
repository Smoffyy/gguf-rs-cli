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

    // Each thread handles a subset of blocks for this row
    for (uint b = tid; b < pc.bpr; b += 256u) {
        uint base = (row * pc.bpr + b) * 48u;
        uint vb = b * 256u;
        for (uint iter = 0u; iter < 4u; iter++) {
            float d1 = uintBitsToFloat(mat.data[base+iter*2u]);
            float d2 = uintBitsToFloat(mat.data[base+iter*2u+1u]);
            float m1 = uintBitsToFloat(mat.data[base+8u+iter*2u]);
            float m2 = uintBitsToFloat(mat.data[base+8u+iter*2u+1u]);
            uint qs = base+16u+iter*8u;
            uint vo = vb+iter*64u;
            for (uint l = 0u; l < 8u; l++) {
                uint pk = mat.data[qs+l];
                for (uint i = 0u; i < 4u; i++) {
                    uint by = (pk>>(i*8u))&0xFFu;
                    uint li = vo+l*4u+i;
                    sum += (d1*float(by&0xFu)-m1)*vin.data[li];
                    sum += (d2*float(by>>4u)-m2)*vin.data[li+32u];
                }
            }
        }
    }

    sdata[tid] = sum;
    barrier();
    // Parallel reduction
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[row] = sdata[0];
}
#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; } pc;
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;
    float sum = 0.0;
    for (uint b = 0u; b < pc.bpr; b++) {
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
    vout.data[row] = sum;
}
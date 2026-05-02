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
    vout.data[row] = sum;
}
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
    for (uint b=0u;b<pc.bpr;b++){
        uint base=(row*pc.bpr+b)*9u;
        float d=uintBitsToFloat(mat.data[base]);
        uint vb=b*32u;
        for (uint l=0u;l<8u;l++){
            uint pk=mat.data[base+1u+l];
            for (uint i=0u;i<4u;i++){
                uint by=(pk>>(i*8u))&0xFFu;
                float q=(by>=128u)?float(int(by)-256):float(by);
                sum+=d*q*vin.data[vb+l*4u+i];
            }
        }
    }
    vout.data[row] = sum;
}
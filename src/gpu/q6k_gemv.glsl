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
        uint base = (row*pc.bpr+b)*64u, vb=b*256u;
        for (uint half=0u;half<2u;half++){
            uint ql=base+16u+half*16u,qh=base+48u+half*8u,yb=vb+half*128u,si=half*8u;
            for (uint l=0u;l<32u;l++){
                uint qll=(mat.data[ql+l/4u]>>(( l   %4u)*8u))&0xFFu;
                uint qlh=(mat.data[ql+(l+32u)/4u]>>(((l+32u)%4u)*8u))&0xFFu;
                uint qhb=(mat.data[qh+l/4u]>>((l%4u)*8u))&0xFFu;
                float q1=float(int((qll&0xFu)|((qhb&3u)   <<4u))-32);
                float q2=float(int((qlh&0xFu)|(((qhb>>2u)&3u)<<4u))-32);
                float q3=float(int((qll>>4u)|(((qhb>>4u)&3u)<<4u))-32);
                float q4=float(int((qlh>>4u)|(((qhb>>6u)&3u)<<4u))-32);
                uint is=l/16u;
                sum+=uintBitsToFloat(mat.data[base+si+is])   *q1*vin.data[yb+l];
                sum+=uintBitsToFloat(mat.data[base+si+is+2u])*q2*vin.data[yb+l+32u];
                sum+=uintBitsToFloat(mat.data[base+si+is+4u])*q3*vin.data[yb+l+64u];
                sum+=uintBitsToFloat(mat.data[base+si+is+6u])*q4*vin.data[yb+l+96u];
            }
        }
    }
    vout.data[row] = sum;
}
#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) readonly buffer Mat { float data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint cols; } pc;
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.rows) return;
    float sum = 0.0;
    for (uint c=0u;c<pc.cols;c++) sum+=mat.data[row*pc.cols+c]*vin.data[c];
    vout.data[row] = sum;
}
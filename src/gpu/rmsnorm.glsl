#version 450
layout(local_size_x=1) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float w[]; };
layout(set=0,binding=2) buffer O { float o[]; };
layout(push_constant) uniform PC { uint n; float eps; } pc;

void main() {
    float ss = 0.0;
    for (uint i = 0u; i < pc.n; i++)
        ss += x[i] * x[i];
    float scale = 1.0 / sqrt(ss / float(pc.n) + pc.eps);
    for (uint i = 0u; i < pc.n; i++)
        o[i] = x[i] * scale * w[i];
}
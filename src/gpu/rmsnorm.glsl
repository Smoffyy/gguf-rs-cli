#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float w[]; };
layout(set=0,binding=2) buffer O { float o[]; };
layout(push_constant) uniform PC { uint n; float eps; } pc;

shared float sdata[256];

void main() {
    uint tid = gl_LocalInvocationID.x;

    float ss = 0.0;
    for (uint i = tid; i < pc.n; i += 256u)
        ss += x[i] * x[i];
    sdata[tid] = ss;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }

    float scale = 1.0 / sqrt(sdata[0] / float(pc.n) + pc.eps);
    barrier();
    for (uint i = tid; i < pc.n; i += 256u)
        o[i] = x[i] * scale * w[i];
}

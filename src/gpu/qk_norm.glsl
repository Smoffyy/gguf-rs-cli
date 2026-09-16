#version 450
layout(local_size_x=128) in;
layout(set=0,binding=0) readonly buffer X { float x[]; };
layout(set=0,binding=1) readonly buffer W { float w[]; };
layout(set=0,binding=2) buffer O { float o[]; };
layout(push_constant) uniform PC { uint head_dim; float eps; } pc;

shared float sdata[128];

void main() {
    uint head = gl_WorkGroupID.x;
    uint tid  = gl_LocalInvocationID.x;
    uint base = head * pc.head_dim;

    float ss = 0.0;
    for (uint i = tid; i < pc.head_dim; i += 128u)
        ss += x[base+i] * x[base+i];
    sdata[tid] = ss;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    float scale = 1.0 / sqrt(sdata[0] / float(pc.head_dim) + pc.eps);
    barrier();
    for (uint i = tid; i < pc.head_dim; i += 128u)
        o[base+i] = x[base+i] * scale * w[i];
}

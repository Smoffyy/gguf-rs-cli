#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) buffer Res { float res[]; };
layout(set=0,binding=1) readonly buffer Add { float add_buf[]; };
layout(set=0,binding=2) readonly buffer W { float w[]; };
layout(set=0,binding=3) buffer O { float o[]; };
layout(push_constant) uniform PC { uint n; float eps; } pc;

shared float sdata[256];

void main() {
    uint tid = gl_LocalInvocationID.x;

    // Phase 1: residual add (res += add_buf) and partial sum of squares
    float ss = 0.0;
    for (uint i = tid; i < pc.n; i += 256u) {
        float v = res[i] + add_buf[i];
        res[i] = v;
        ss += v * v;
    }
    sdata[tid] = ss;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }

    float scale = 1.0 / sqrt(sdata[0] / float(pc.n) + pc.eps);
    barrier();
    for (uint i = tid; i < pc.n; i += 256u)
        o[i] = res[i] * scale * w[i];
}

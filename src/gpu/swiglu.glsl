#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) buffer       G { float g[]; };
layout(set=0,binding=1) readonly buffer U { float u[]; };
layout(push_constant) uniform PC { uint n; uint use_gelu; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    float x = g[i];
    float act;
    if (pc.use_gelu != 0u) {
        act = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    } else {
        act = x / (1.0 + exp(-x));
    }
    g[i] = act * u[i];
}

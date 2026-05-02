#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) buffer       G { float g[]; };  // gate (in-place)
layout(set=0,binding=1) readonly buffer U { float u[]; };  // up
layout(push_constant) uniform PC { uint n; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    float x = g[i];
    g[i] = (x / (1.0 + exp(-x))) * u[i];  // silu(gate) * up
}
#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) buffer       A { float a[]; };
layout(set=0,binding=1) readonly buffer B { float b[]; };
layout(push_constant) uniform PC { uint n; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;
    a[i] += b[i];
}
#version 450
layout(local_size_x=1) in;
layout(set=0,binding=0) buffer Q { float q[]; };
layout(set=0,binding=1) buffer K { float k[]; };
layout(push_constant) uniform PC {
    uint n_heads; uint n_kv_heads; uint head_dim; uint pos;
    float freq_base;
} pc;

void rot(inout float a, inout float b, float c, float s) {
    float tmp = a; a = tmp*c - b*s; b = tmp*s + b*c;
}

void main() {
    uint h    = gl_WorkGroupID.x;
    uint half = pc.head_dim / 2u;
    float pos = float(pc.pos);
    if (h < pc.n_heads) {
        uint base = h * pc.head_dim;
        for (uint i = 0u; i < half; i++) {
            float theta = pos * pow(pc.freq_base, -2.0*float(i)/float(pc.head_dim));
            rot(q[base+i], q[base+i+half], cos(theta), sin(theta));
        }
    } else {
        uint kh = h - pc.n_heads;
        if (kh >= pc.n_kv_heads) return;
        uint base = kh * pc.head_dim;
        for (uint i = 0u; i < half; i++) {
            float theta = pos * pow(pc.freq_base, -2.0*float(i)/float(pc.head_dim));
            rot(k[base+i], k[base+i+half], cos(theta), sin(theta));
        }
    }
}
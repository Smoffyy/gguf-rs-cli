#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) buffer Q { float q[]; };
layout(set=0,binding=1) buffer K { float k[]; };
layout(set=0,binding=2) buffer Dummy { float dummy[]; };
layout(push_constant) uniform PC {
    uint n_heads; uint n_kv_heads; uint head_dim; uint pos; uint freq_bits;
} pc;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total_heads = pc.n_heads + pc.n_kv_heads;
    uint hd2 = pc.head_dim / 2u;
    uint h = idx / hd2;
    uint i = idx % hd2;
    if (h >= total_heads) return;

    float freq = uintBitsToFloat(pc.freq_bits);
    float t = float(pc.pos) * pow(freq, -2.0 * float(i) / float(pc.head_dim));
    float s = sin(t);
    float c = cos(t);

    if (h < pc.n_heads) {
        uint b = h * pc.head_dim + i;
        float x0 = q[b];
        float x1 = q[b + hd2];
        q[b]        = x0 * c - x1 * s;
        q[b + hd2] = x0 * s + x1 * c;
    } else {
        uint kh = h - pc.n_heads;
        uint b = kh * pc.head_dim + i;
        float x0 = k[b];
        float x1 = k[b + hd2];
        k[b]        = x0 * c - x1 * s;
        k[b + hd2] = x0 * s + x1 * c;
    }
}
#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) readonly buffer K  { float k[];  };
layout(set=0,binding=1) readonly buffer V  { float v[];  };
layout(set=0,binding=2) buffer KC { float kc[]; };
layout(set=0,binding=3) buffer VC { float vc[]; };
layout(push_constant) uniform PC { uint pos; uint n_kv_heads; uint head_dim; uint n_ctx; } pc;

void main() {
    uint i   = gl_GlobalInvocationID.x;
    uint kvd = pc.n_kv_heads * pc.head_dim;
    if (i >= kvd) return;
    uint off = pc.pos * kvd + i;
    kc[off] = k[i];
    vc[off] = v[i];
}
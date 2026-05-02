#version 450
layout(local_size_x=1) in;
layout(set=0,binding=0) readonly buffer Q  { float q[];  };
layout(set=0,binding=1) readonly buffer KC { float kc[]; };
layout(set=0,binding=2) readonly buffer VC { float vc[]; };
layout(set=0,binding=3) buffer AO { float ao[]; };
layout(set=0,binding=4) buffer SC { float sc[]; };
layout(push_constant) uniform PC {
    uint n_heads; uint n_kv_heads; uint head_dim; uint seq_len; uint n_ctx;
} pc;

void main() {
    uint h = gl_WorkGroupID.x;
    if (h >= pc.n_heads) return;
    uint kv_ratio = pc.n_heads / pc.n_kv_heads;
    uint kv_h     = h / kv_ratio;
    uint kvd      = pc.n_kv_heads * pc.head_dim;
    float scale   = 1.0 / sqrt(float(pc.head_dim));
    uint q_base   = h * pc.head_dim;
    uint sc_base  = h * pc.n_ctx;

    float max_s = -1e30;
    for (uint p = 0u; p < pc.seq_len; p++) {
        float s = 0.0;
        uint k_base = p * kvd + kv_h * pc.head_dim;
        for (uint d = 0u; d < pc.head_dim; d++)
            s += q[q_base + d] * kc[k_base + d];
        sc[sc_base + p] = s * scale;
        if (sc[sc_base + p] > max_s) max_s = sc[sc_base + p];
    }
    float sum = 0.0;
    for (uint p = 0u; p < pc.seq_len; p++) {
        sc[sc_base + p] = exp(sc[sc_base + p] - max_s);
        sum += sc[sc_base + p];
    }
    for (uint p = 0u; p < pc.seq_len; p++)
        sc[sc_base + p] /= sum;

    uint ao_base = h * pc.head_dim;
    for (uint d = 0u; d < pc.head_dim; d++) {
        float val = 0.0;
        for (uint p = 0u; p < pc.seq_len; p++)
            val += sc[sc_base + p] * vc[p * kvd + kv_h * pc.head_dim + d];
        ao[ao_base + d] = val;
    }
}
#version 450
layout(local_size_x=128) in;
layout(set=0,binding=0) readonly buffer Q  { float q[];  };
layout(set=0,binding=1) readonly buffer KC { float kc[]; };
layout(set=0,binding=2) readonly buffer VC { float vc[]; };
layout(set=0,binding=3) buffer AO { float ao[]; };
layout(set=0,binding=4) buffer SC { float sc[]; };
layout(push_constant) uniform PC {
    uint n_heads; uint n_kv_heads; uint head_dim; uint seq_len; uint n_ctx; float softcap;
} pc;

const uint TILE = 32u;
const uint MAX_HEAD_DIM = 256u;

shared float s_scores[TILE];
shared float s_acc[MAX_HEAD_DIM];
shared float s_m;
shared float s_l;

void main() {
    uint h = gl_WorkGroupID.x;
    if (h >= pc.n_heads) return;
    uint tid = gl_LocalInvocationID.x;
    uint kv_h = h / (pc.n_heads / pc.n_kv_heads);
    uint kvd  = pc.n_kv_heads * pc.head_dim;
    float inv_sqrt = 1.0 / sqrt(float(pc.head_dim));
    uint q_base = h * pc.head_dim;

    for (uint d = tid; d < pc.head_dim; d += 128u) s_acc[d] = 0.0;
    if (tid == 0u) { s_m = -1e30; s_l = 0.0; }
    barrier();

    uint n_tiles = (pc.seq_len + TILE - 1u) / TILE;
    for (uint t = 0u; t < n_tiles; t++) {
        uint tile_start = t * TILE;
        uint tile_len = min(TILE, pc.seq_len - tile_start);

        if (tid < tile_len) {
            uint p = tile_start + tid;
            uint k_base = p * kvd + kv_h * pc.head_dim;
            float s = 0.0;
            for (uint d = 0u; d < pc.head_dim; d++) s += q[q_base + d] * kc[k_base + d];
            s *= inv_sqrt;
            if (pc.softcap > 0.0) s = pc.softcap * tanh(s / pc.softcap);
            s_scores[tid] = s;
        }
        barrier();

        float tile_max = -1e30;
        for (uint i = 0u; i < tile_len; i++) tile_max = max(tile_max, s_scores[i]);
        float new_m = max(s_m, tile_max);
        float alpha = exp(s_m - new_m);

        if (tid < tile_len) s_scores[tid] = exp(s_scores[tid] - new_m);
        barrier();

        float tile_l = 0.0;
        for (uint i = 0u; i < tile_len; i++) tile_l += s_scores[i];

        for (uint d = tid; d < pc.head_dim; d += 128u) {
            float acc = s_acc[d] * alpha;
            for (uint i = 0u; i < tile_len; i++) {
                uint p = tile_start + i;
                uint v_base = p * kvd + kv_h * pc.head_dim;
                acc += s_scores[i] * vc[v_base + d];
            }
            s_acc[d] = acc;
        }
        barrier();

        if (tid == 0u) { s_m = new_m; s_l = s_l * alpha + tile_l; }
        barrier();
    }

    float l = max(s_l, 1e-12);
    for (uint d = tid; d < pc.head_dim; d += 128u) {
        ao[h * pc.head_dim + d] = s_acc[d] / l;
    }
}

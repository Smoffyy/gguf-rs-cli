// Applies RoPE in-place to Q and K tensors
// Each head rotates pairs of values using position-dependent angles
pub fn apply_rope(
    q: &mut [f32], k: &mut [f32],
    pos: usize, head_dim: usize, freq_base: f32,
    n_heads: usize, n_kv_heads: usize,
) {
    let half = head_dim / 2;
    apply_rope_to(q, pos, head_dim, freq_base, n_heads, half);
    apply_rope_to(k, pos, head_dim, freq_base, n_kv_heads, half);
}

fn apply_rope_to(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32, n_heads: usize, half: usize) {
    for h in 0..n_heads {
        for i in 0..half {
            let theta = (pos as f32) * freq_base.powf(-2.0 * i as f32 / head_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let base = h * head_dim + i;
            let x0 = x[base];
            let x1 = x[base + half];
            x[base]        = x0 * cos - x1 * sin;
            x[base + half] = x0 * sin + x1 * cos;
        }
    }
}

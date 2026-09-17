//! Activation quantization.
//!
//! Matrix-vector and matrix-matrix products against quantized weights run far faster as
//! integer dot products than as float ones, but that requires the *activation* to be
//! quantized too. Q8_1 is the universal staging format here: 32-element blocks carrying an
//! f16 scale and an f16 pre-computed sum, the latter being exactly what the asymmetric
//! quantizations (Q4_1, Q4_K, Q5_K, ...) need to fold in their per-block minimum.

use half::f16;

/// Bytes per Q8_1 block: `f16 d, f16 s, i8 qs[32]`.
pub const Q8_1_BLOCK: usize = 36;
pub const Q8_1_SIZE: usize = 32;

/// Bytes needed to hold `n` values as Q8_1. `n` is rounded up to a whole block.
pub const fn q8_1_bytes(n: usize) -> usize {
    n.div_ceil(Q8_1_SIZE) * Q8_1_BLOCK
}

/// Quantize `x` to Q8_1 blocks. A trailing partial block is zero-padded, which is correct
/// because the matching weight block is padded the same way by the file format.
pub fn quantize_q8_1(x: &[f32], out: &mut [u8]) {
    let nb = x.len().div_ceil(Q8_1_SIZE);
    debug_assert!(out.len() >= nb * Q8_1_BLOCK);

    for b in 0..nb {
        let lo = b * Q8_1_SIZE;
        let hi = (lo + Q8_1_SIZE).min(x.len());
        let src = &x[lo..hi];

        let amax = src.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };

        let blk = &mut out[b * Q8_1_BLOCK..(b + 1) * Q8_1_BLOCK];
        let mut sum: i32 = 0;
        for i in 0..Q8_1_SIZE {
            // Ties to even, matching CUDA's `__float2int_rn` and Vulkan's `roundEven`.
            // Two backends that round the same value differently disagree by a whole
            // quantization step, and that disagreement compounds through every layer.
            let q = if i < src.len() {
                (src[i] * id).round_ties_even() as i32
            } else {
                0
            };
            let q = q.clamp(-127, 127) as i8;
            blk[4 + i] = q as u8;
            sum += q as i32;
        }
        blk[0..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        blk[2..4].copy_from_slice(&f16::from_f32(sum as f32 * d).to_le_bytes());
    }
}

/// Quantize a batch of `n_tokens` rows of `dim` values each into contiguous Q8_1 rows.
pub fn quantize_q8_1_batch(x: &[f32], dim: usize, n_tokens: usize, out: &mut [u8]) {
    let row = q8_1_bytes(dim);
    for t in 0..n_tokens {
        quantize_q8_1(&x[t * dim..(t + 1) * dim], &mut out[t * row..(t + 1) * row]);
    }
}

#[inline(always)]
pub fn q8_1_scale(blk: &[u8]) -> f32 {
    f16::from_le_bytes([blk[0], blk[1]]).to_f32()
}

/// The block's `sum(q) * d`, i.e. the sum of the dequantized activations.
#[inline(always)]
pub fn q8_1_sum(blk: &[u8]) -> f32 {
    f16::from_le_bytes([blk[2], blk[3]]).to_f32()
}

#[inline(always)]
pub fn q8_1_quants(blk: &[u8]) -> &[i8] {
    // SAFETY: i8 and u8 have identical layout; the slice length is unchanged.
    unsafe { std::slice::from_raw_parts(blk[4..36].as_ptr() as *const i8, 32) }
}

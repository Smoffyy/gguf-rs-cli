//! Dot products between a quantized weight row and a Q8_1-quantized activation row.
//!
//! Every one of these follows the same algebra: an integer dot of the raw quant codes,
//! scaled by `weight_scale * activation_scale`, plus a correction term built from the
//! activation's block sum whenever the weight encoding is asymmetric (carries a minimum)
//! or offset (Q4_0's -8, Q5_0's -16, ternary's -1).
//!
//! Keeping the correction in closed form is what makes the integer path exact rather than
//! approximate: it is the same value the float path would produce, not a nearby one.

use gguf_core::{DType, Error, Result};

use crate::dequant::{bf16, e8m0_to_f32_half, fp16, q3k_scale, scale_min_k4, KVALUES_IQ4NL, KVALUES_MXFP4};
use crate::quantize::{q8_1_quants, q8_1_scale, q8_1_sum, Q8_1_BLOCK};

/// Dot product of one quantized weight row with a Q8_1 activation row.
///
/// `n` is the row length in elements, `w` the weight bytes, `a` the Q8_1 blocks.
pub fn vec_dot_q8_1(dtype: DType, n: usize, w: &[u8], a: &[u8]) -> f32 {
    match dtype {
        DType::Q4_0 => q4_0(n, w, a),
        DType::Q4_1 => q4_1(n, w, a),
        DType::Q5_0 => q5_0(n, w, a),
        DType::Q5_1 => q5_1(n, w, a),
        DType::Q8_0 => q8_0(n, w, a),
        DType::Q2K => q2_k(n, w, a),
        DType::Q3K => q3_k(n, w, a),
        DType::Q4K => q4_k(n, w, a),
        DType::Q5K => q5_k(n, w, a),
        DType::Q6K => q6_k(n, w, a),
        DType::Iq4Nl => codebook32(n, w, a, &KVALUES_IQ4NL, 2, |b| fp16(b, 0)),
        DType::Iq4Xs => iq4_xs(n, w, a),
        DType::Mxfp4 => codebook32(n, w, a, &KVALUES_MXFP4, 1, |b| e8m0_to_f32_half(b[0])),
        DType::Tq1_0 => tq1_0(n, w, a),
        DType::Tq2_0 => tq2_0(n, w, a),
        // Float weights never take this path; the caller uses `vec_dot_f32` instead.
        _ => 0.0,
    }
}

/// Dot product of a float-typed weight row with a plain f32 activation row.
pub fn vec_dot_f32(dtype: DType, n: usize, w: &[u8], a: &[f32]) -> f32 {
    match dtype {
        DType::F32 => w[..n * 4]
            .chunks_exact(4)
            .zip(a)
            .map(|(b, y)| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * y)
            .sum(),
        DType::F16 => (0..n).map(|i| fp16(w, i * 2) * a[i]).sum(),
        DType::BF16 => (0..n).map(|i| bf16(w, i * 2) * a[i]).sum(),
        _ => 0.0,
    }
}

pub fn is_float(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

/// Rejects a weight type that has no integer dot path, so a caller cannot silently get
/// zeros from `vec_dot_q8_1`.
pub fn check_dot_supported(dtype: DType) -> Result<()> {
    dtype.check_supported()?;
    if is_float(dtype) {
        return Ok(());
    }
    match dtype {
        DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Q8_0
        | DType::Q2K | DType::Q3K | DType::Q4K | DType::Q5K | DType::Q6K
        | DType::Iq4Nl | DType::Iq4Xs | DType::Mxfp4 | DType::Tq1_0 | DType::Tq2_0 => Ok(()),
        other => Err(Error::Unsupported(format!(
            "{} has no matmul path on this backend",
            other.name()
        ))),
    }
}

#[inline(always)]
fn ablk(a: &[u8], i: usize) -> &[u8] {
    &a[i * Q8_1_BLOCK..(i + 1) * Q8_1_BLOCK]
}

fn q4_0(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 32 {
        let blk = &w[b * 18..(b + 1) * 18];
        let d = fp16(blk, 0);
        let ay = ablk(a, b);
        let y = q8_1_quants(ay);
        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += (blk[2 + j] & 0xF) as i32 * y[j] as i32;
            sumi += (blk[2 + j] >> 4) as i32 * y[j + 16] as i32;
        }
        // y = (q - 8) * d  =>  d*da*sumi - 8*d*sum(a)
        acc += d * (q8_1_scale(ay) * sumi as f32) - 8.0 * d * q8_1_sum(ay);
    }
    acc
}

fn q4_1(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 32 {
        let blk = &w[b * 20..(b + 1) * 20];
        let (d, m) = (fp16(blk, 0), fp16(blk, 2));
        let ay = ablk(a, b);
        let y = q8_1_quants(ay);
        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += (blk[4 + j] & 0xF) as i32 * y[j] as i32;
            sumi += (blk[4 + j] >> 4) as i32 * y[j + 16] as i32;
        }
        acc += d * q8_1_scale(ay) * sumi as f32 + m * q8_1_sum(ay);
    }
    acc
}

fn q5_0(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 32 {
        let blk = &w[b * 22..(b + 1) * 22];
        let d = fp16(blk, 0);
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let ay = ablk(a, b);
        let y = q8_1_quants(ay);
        let mut sumi = 0i32;
        for j in 0..16 {
            let hl = ((qh >> j) << 4) as u8 & 0x10;
            let hh = (qh >> (j + 12)) as u8 & 0x10;
            sumi += ((blk[6 + j] & 0xF) | hl) as i32 * y[j] as i32;
            sumi += ((blk[6 + j] >> 4) | hh) as i32 * y[j + 16] as i32;
        }
        acc += d * q8_1_scale(ay) * sumi as f32 - 16.0 * d * q8_1_sum(ay);
    }
    acc
}

fn q5_1(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 32 {
        let blk = &w[b * 24..(b + 1) * 24];
        let (d, m) = (fp16(blk, 0), fp16(blk, 2));
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        let ay = ablk(a, b);
        let y = q8_1_quants(ay);
        let mut sumi = 0i32;
        for j in 0..16 {
            let hl = ((qh >> j) << 4) as u8 & 0x10;
            let hh = (qh >> (j + 12)) as u8 & 0x10;
            sumi += ((blk[8 + j] & 0xF) | hl) as i32 * y[j] as i32;
            sumi += ((blk[8 + j] >> 4) | hh) as i32 * y[j + 16] as i32;
        }
        acc += d * q8_1_scale(ay) * sumi as f32 + m * q8_1_sum(ay);
    }
    acc
}

fn q8_0(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 32 {
        let blk = &w[b * 34..(b + 1) * 34];
        let d = fp16(blk, 0);
        let ay = ablk(a, b);
        let y = q8_1_quants(ay);
        let mut sumi = 0i32;
        for j in 0..32 {
            sumi += (blk[2 + j] as i8) as i32 * y[j] as i32;
        }
        acc += d * q8_1_scale(ay) * sumi as f32;
    }
    acc
}

fn q2_k(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 256 {
        let blk = &w[b * 84..(b + 1) * 84];
        let sc = &blk[0..16];
        let qs = &blk[16..80];
        let d = fp16(blk, 80);
        let dmin = fp16(blk, 82);
        for h in 0..2usize {
            let q = &qs[h * 32..];
            for j in 0..4usize {
                let e0 = h * 128 + j * 32;
                let ay = ablk(a, b * 8 + e0 / 32);
                let y = q8_1_quants(ay);
                let ad = q8_1_scale(ay);
                let shift = (j * 2) as u32;
                let is = h * 8 + j * 2;
                // Two 16-element scale groups share one 32-element activation block, so
                // each keeps its own partial activation sum for the minimum term.
                let (mut si0, mut sy0, mut si1, mut sy1) = (0i32, 0i32, 0i32, 0i32);
                for l in 0..16 {
                    si0 += ((q[l] >> shift) & 3) as i32 * y[l] as i32;
                    sy0 += y[l] as i32;
                    si1 += ((q[l + 16] >> shift) & 3) as i32 * y[l + 16] as i32;
                    sy1 += y[l + 16] as i32;
                }
                let (s0, s1) = (sc[is], sc[is + 1]);
                acc += ad
                    * (d * (s0 & 0xF) as f32 * si0 as f32 - dmin * (s0 >> 4) as f32 * sy0 as f32);
                acc += ad
                    * (d * (s1 & 0xF) as f32 * si1 as f32 - dmin * (s1 >> 4) as f32 * sy1 as f32);
            }
        }
    }
    acc
}

fn q3_k(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 256 {
        let blk = &w[b * 110..(b + 1) * 110];
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let scales = &blk[96..108];
        let d = fp16(blk, 108);
        let sc: [i8; 16] = std::array::from_fn(|k| q3k_scale(scales, k));
        for h in 0..2usize {
            let q = &qs[h * 32..];
            for j in 0..4usize {
                let e0 = h * 128 + j * 32;
                let ay = ablk(a, b * 8 + e0 / 32);
                let y = q8_1_quants(ay);
                let ad = q8_1_scale(ay);
                let shift = (j * 2) as u32;
                let m: u8 = 1u8 << (h * 4 + j);
                let is = h * 8 + j * 2;
                let (mut si0, mut si1) = (0i32, 0i32);
                for l in 0..16 {
                    let hv0 = if hmask[l] & m != 0 { 0 } else { 4 };
                    let hv1 = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                    si0 += (((q[l] >> shift) & 3) as i32 - hv0) * y[l] as i32;
                    si1 += (((q[l + 16] >> shift) & 3) as i32 - hv1) * y[l + 16] as i32;
                }
                acc += ad * d * (sc[is] as f32 * si0 as f32 + sc[is + 1] as f32 * si1 as f32);
            }
        }
    }
    acc
}

fn q4_k(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 256 {
        let blk = &w[b * 144..(b + 1) * 144];
        let d = fp16(blk, 0);
        let dmin = fp16(blk, 2);
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for g in 0..4usize {
            let q = &qs[g * 32..];
            for half in 0..2usize {
                let ay = ablk(a, b * 8 + g * 2 + half);
                let y = q8_1_quants(ay);
                let (sc, mn) = scale_min_k4(g * 2 + half, scales);
                let mut sumi = 0i32;
                for l in 0..32 {
                    let c = if half == 0 { q[l] & 0xF } else { q[l] >> 4 };
                    sumi += c as i32 * y[l] as i32;
                }
                acc += d * sc * q8_1_scale(ay) * sumi as f32 - dmin * mn * q8_1_sum(ay);
            }
        }
    }
    acc
}

fn q5_k(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 256 {
        let blk = &w[b * 176..(b + 1) * 176];
        let d = fp16(blk, 0);
        let dmin = fp16(blk, 2);
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let ql = &blk[48..176];
        for g in 0..4usize {
            let q = &ql[g * 32..];
            for half in 0..2usize {
                let ay = ablk(a, b * 8 + g * 2 + half);
                let y = q8_1_quants(ay);
                let (sc, mn) = scale_min_k4(g * 2 + half, scales);
                let bit: u8 = ((1u32 + half as u32) << (2 * g)) as u8;
                let mut sumi = 0i32;
                for l in 0..32 {
                    let lo = if half == 0 { q[l] & 0xF } else { q[l] >> 4 };
                    let hv = if qh[l] & bit != 0 { 16 } else { 0 };
                    sumi += (lo as i32 + hv) * y[l] as i32;
                }
                acc += d * sc * q8_1_scale(ay) * sumi as f32 - dmin * mn * q8_1_sum(ay);
            }
        }
    }
    acc
}

fn q6_k(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 256 {
        let blk = &w[b * 210..(b + 1) * 210];
        let ql_all = &blk[0..128];
        let qh_all = &blk[128..192];
        let sc_all = &blk[192..208];
        let d = fp16(blk, 208);
        for h in 0..2usize {
            let ql = &ql_all[h * 64..];
            let qh = &qh_all[h * 32..];
            let sc = &sc_all[h * 8..];
            for g in 0..4usize {
                let ay = ablk(a, b * 8 + h * 4 + g);
                let y = q8_1_quants(ay);
                let ad = q8_1_scale(ay);
                let ql_off = (g % 2) * 32;
                let lo_shift = (g / 2) * 4;
                let hi_shift = (g * 2) as u32;
                // Scales change every 16 elements inside this 32-element block.
                let (mut si0, mut si1) = (0i32, 0i32);
                for l in 0..32 {
                    let lo = (ql[ql_off + l] >> lo_shift) & 0xF;
                    let hi = ((qh[l] >> hi_shift) & 3) << 4;
                    let q = (lo | hi) as i32 - 32;
                    if l < 16 {
                        si0 += q * y[l] as i32;
                    } else {
                        si1 += q * y[l] as i32;
                    }
                }
                acc += ad
                    * d
                    * ((sc[2 * g] as i8) as f32 * si0 as f32
                        + (sc[2 * g + 1] as i8) as f32 * si1 as f32);
            }
        }
    }
    acc
}

/// Shared shape for the 32-element codebook types: one f16/E8M0 scale then 16 packed bytes.
fn codebook32(
    n: usize,
    w: &[u8],
    a: &[u8],
    table: &[i8; 16],
    qs_off: usize,
    scale: impl Fn(&[u8]) -> f32,
) -> f32 {
    let ts = qs_off + 16;
    let mut acc = 0f32;
    for b in 0..n / 32 {
        let blk = &w[b * ts..(b + 1) * ts];
        let d = scale(blk);
        let ay = ablk(a, b);
        let y = q8_1_quants(ay);
        let mut sumi = 0i32;
        for j in 0..16 {
            sumi += table[(blk[qs_off + j] & 0xF) as usize] as i32 * y[j] as i32;
            sumi += table[(blk[qs_off + j] >> 4) as usize] as i32 * y[j + 16] as i32;
        }
        acc += d * q8_1_scale(ay) * sumi as f32;
    }
    acc
}

fn iq4_xs(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 256 {
        let blk = &w[b * 136..(b + 1) * 136];
        let d = fp16(blk, 0);
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = &blk[8..136];
        for ib in 0..8usize {
            let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
                | (((scales_h >> (2 * ib)) & 3) as i32) << 4;
            let dl = d * (ls - 32) as f32;
            let q = &qs[ib * 16..];
            let ay = ablk(a, b * 8 + ib);
            let y = q8_1_quants(ay);
            let mut sumi = 0i32;
            for j in 0..16 {
                sumi += KVALUES_IQ4NL[(q[j] & 0xF) as usize] as i32 * y[j] as i32;
                sumi += KVALUES_IQ4NL[(q[j] >> 4) as usize] as i32 * y[j + 16] as i32;
            }
            acc += dl * q8_1_scale(ay) * sumi as f32;
        }
    }
    acc
}

fn tq1_0(n: usize, w: &[u8], a: &[u8]) -> f32 {
    const POW3: [u16; 5] = [1, 3, 9, 27, 81];
    let mut acc = 0f32;
    for b in 0..n / 256 {
        let blk = &w[b * 54..(b + 1) * 54];
        let qs = &blk[0..48];
        let qh = &blk[48..52];
        let d = fp16(blk, 52);

        // Unpack the 256 ternary digits in file order, then dot them blockwise. The
        // packing is base-3 across 5 digits per byte, so there is no cheaper route than
        // materialising the digits first.
        let mut t = [0i8; 256];
        let mut e = 0usize;
        for nn in 0..5usize {
            for j in 0..32usize {
                let q = (qs[j] as u16).wrapping_mul(POW3[nn]) & 0xFF;
                t[e] = (((q * 3) >> 8) as i32 - 1) as i8;
                e += 1;
            }
        }
        for nn in 0..5usize {
            for j in 0..16usize {
                let q = (qs[j + 32] as u16).wrapping_mul(POW3[nn]) & 0xFF;
                t[e] = (((q * 3) >> 8) as i32 - 1) as i8;
                e += 1;
            }
        }
        for nn in 0..4usize {
            for j in 0..4usize {
                let q = (qh[j] as u16).wrapping_mul(POW3[nn]) & 0xFF;
                t[e] = (((q * 3) >> 8) as i32 - 1) as i8;
                e += 1;
            }
        }

        let mut sub = 0f32;
        for ib in 0..8usize {
            let ay = ablk(a, b * 8 + ib);
            let y = q8_1_quants(ay);
            let mut sumi = 0i32;
            for m in 0..32usize {
                sumi += t[ib * 32 + m] as i32 * y[m] as i32;
            }
            sub += q8_1_scale(ay) * sumi as f32;
        }
        acc += d * sub;
    }
    acc
}

fn tq2_0(n: usize, w: &[u8], a: &[u8]) -> f32 {
    let mut acc = 0f32;
    for b in 0..n / 256 {
        let blk = &w[b * 66..(b + 1) * 66];
        let qs = &blk[0..64];
        let d = fp16(blk, 64);
        let mut e = 0usize;
        let mut sub = 0f32;
        for j in (0..64).step_by(32) {
            for l in 0..4usize {
                let ay = ablk(a, b * 8 + e / 32);
                let y = q8_1_quants(ay);
                let ad = q8_1_scale(ay);
                let mut sumi = 0i32;
                for m in 0..32usize {
                    let q = ((qs[j + m] >> (l * 2)) & 3) as i32;
                    sumi += (q - 1) * y[m] as i32;
                }
                sub += ad * sumi as f32;
                e += 32;
            }
        }
        acc += d * sub;
    }
    acc
}

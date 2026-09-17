//! Block decoders for every ggml quantization this engine supports.
//!
//! Each function takes the raw bytes of one or more blocks exactly as they appear in the
//! file and writes `n` f32 values. Layouts follow ggml's `block_*` structs byte for byte;
//! an invented layout produces fluent-sounding but wrong output rather than a crash, so
//! these are covered by round-trip tests against known-good vectors.

use gguf_core::{DType, Error, Result};
use half::f16;

#[inline(always)]
pub(crate) fn fp16(b: &[u8], at: usize) -> f32 {
    f16::from_le_bytes([b[at], b[at + 1]]).to_f32()
}

#[inline(always)]
pub(crate) fn bf16(b: &[u8], at: usize) -> f32 {
    f32::from_bits((u16::from_le_bytes([b[at], b[at + 1]]) as u32) << 16)
}

/// IQ4_NL / IQ4_XS share this 16-entry non-linear codebook. Unlike the IQ1/IQ2/IQ3 grids
/// it is small enough to be part of the format rather than a lookup table.
pub const KVALUES_IQ4NL: [i8; 16] =
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// MXFP4 E2M1 codebook, stored at 2x so the exponent scale carries the halving.
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// E8M0 exponent byte to `2^(e-127) / 2`.
#[inline(always)]
pub(crate) fn e8m0_to_f32_half(e: u8) -> f32 {
    let bits = if e < 2 { 0x0020_0000u32 << e } else { (e as u32 - 1) << 23 };
    f32::from_bits(bits)
}

/// Decode the 6-bit scale at index `k` out of Q4_K/Q5_K's packed 12-byte scale block.
#[inline(always)]
pub(crate) fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    let (d, m) = if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    };
    (d as f32, m as f32)
}

/// Decode the 6-bit signed scale at index `k` out of Q3_K's packed 12-byte scale block.
///
/// ggml assembles these by shuffling three 32-bit words; this is the same shuffle expressed
/// per index so it can be used from a loop without a scratch buffer.
#[inline(always)]
pub(crate) fn q3k_scale(sc: &[u8], k: usize) -> i8 {
    let (lo4, hi2) = match k {
        0..=3 => (sc[k] & 0xF, sc[8 + k] & 0x3),
        4..=7 => (sc[k] & 0xF, (sc[4 + k] >> 2) & 0x3),
        8..=11 => ((sc[k - 8] >> 4) & 0xF, (sc[k] >> 4) & 0x3),
        _ => ((sc[k - 8] >> 4) & 0xF, (sc[k - 4] >> 6) & 0x3),
    };
    ((lo4 | (hi2 << 4)) as i8).wrapping_sub(32)
}

/// Decode `n` elements of `dtype` from `src` into `dst`.
///
/// `n` must be a multiple of the type's block size and `dst.len()` must be at least `n`.
pub fn dequantize(dtype: DType, src: &[u8], dst: &mut [f32], n: usize) -> Result<()> {
    dtype.check_supported()?;
    let bs = dtype.block_size();
    if n % bs != 0 {
        return Err(Error::Shape(format!(
            "{n} elements is not a multiple of the {} block size {bs}",
            dtype.name()
        )));
    }
    let need_src = dtype.row_bytes(n);
    if src.len() < need_src {
        return Err(Error::Shape(format!(
            "{}: need {need_src} bytes for {n} elements, got {}",
            dtype.name(),
            src.len()
        )));
    }
    if dst.len() < n {
        return Err(Error::Shape(format!("destination holds {} of {n} values", dst.len())));
    }

    let ts = dtype.type_size();
    let nb = n / bs;
    match dtype {
        DType::F32 => {
            for (o, c) in dst[..n].iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        DType::F16 => {
            for (o, c) in dst[..n].iter_mut().zip(src.chunks_exact(2)) {
                *o = f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        DType::BF16 => {
            for (o, c) in dst[..n].iter_mut().zip(src.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        DType::F64 => {
            for (o, c) in dst[..n].iter_mut().zip(src.chunks_exact(8)) {
                *o = f64::from_le_bytes(c.try_into().unwrap()) as f32;
            }
        }
        DType::I8 => {
            for (o, c) in dst[..n].iter_mut().zip(src.iter()) {
                *o = *c as i8 as f32;
            }
        }
        DType::I16 => {
            for (o, c) in dst[..n].iter_mut().zip(src.chunks_exact(2)) {
                *o = i16::from_le_bytes([c[0], c[1]]) as f32;
            }
        }
        DType::I32 => {
            for (o, c) in dst[..n].iter_mut().zip(src.chunks_exact(4)) {
                *o = i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32;
            }
        }
        DType::I64 => {
            for (o, c) in dst[..n].iter_mut().zip(src.chunks_exact(8)) {
                *o = i64::from_le_bytes(c.try_into().unwrap()) as f32;
            }
        }
        _ => {
            for b in 0..nb {
                let blk = &src[b * ts..(b + 1) * ts];
                let out = &mut dst[b * bs..(b + 1) * bs];
                match dtype {
                    DType::Q4_0 => q4_0(blk, out),
                    DType::Q4_1 => q4_1(blk, out),
                    DType::Q5_0 => q5_0(blk, out),
                    DType::Q5_1 => q5_1(blk, out),
                    DType::Q8_0 => q8_0(blk, out),
                    DType::Q8_1 => q8_1(blk, out),
                    DType::Q2K => q2_k(blk, out),
                    DType::Q3K => q3_k(blk, out),
                    DType::Q4K => q4_k(blk, out),
                    DType::Q5K => q5_k(blk, out),
                    DType::Q6K => q6_k(blk, out),
                    DType::Q8K => q8_k(blk, out),
                    DType::Iq4Nl => iq4_nl(blk, out),
                    DType::Iq4Xs => iq4_xs(blk, out),
                    DType::Tq1_0 => tq1_0(blk, out),
                    DType::Tq2_0 => tq2_0(blk, out),
                    DType::Mxfp4 => mxfp4(blk, out),
                    other => {
                        return Err(Error::Unsupported(format!("decoding {}", other.name())))
                    }
                }
            }
        }
    }
    Ok(())
}

/// Convenience wrapper allocating the destination.
pub fn dequantize_vec(dtype: DType, src: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; n];
    dequantize(dtype, src, &mut out, n)?;
    Ok(out)
}

fn q4_0(b: &[u8], y: &mut [f32]) {
    let d = fp16(b, 0);
    for j in 0..16 {
        y[j] = ((b[2 + j] & 0xF) as i32 - 8) as f32 * d;
        y[j + 16] = ((b[2 + j] >> 4) as i32 - 8) as f32 * d;
    }
}

fn q4_1(b: &[u8], y: &mut [f32]) {
    let d = fp16(b, 0);
    let m = fp16(b, 2);
    for j in 0..16 {
        y[j] = (b[4 + j] & 0xF) as f32 * d + m;
        y[j + 16] = (b[4 + j] >> 4) as f32 * d + m;
    }
}

fn q5_0(b: &[u8], y: &mut [f32]) {
    let d = fp16(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    for j in 0..16 {
        let hl = ((qh >> j) << 4) as u8 & 0x10;
        let hh = (qh >> (j + 12)) as u8 & 0x10;
        y[j] = (((b[6 + j] & 0xF) | hl) as i32 - 16) as f32 * d;
        y[j + 16] = (((b[6 + j] >> 4) | hh) as i32 - 16) as f32 * d;
    }
}

fn q5_1(b: &[u8], y: &mut [f32]) {
    let d = fp16(b, 0);
    let m = fp16(b, 2);
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    for j in 0..16 {
        let hl = ((qh >> j) << 4) as u8 & 0x10;
        let hh = (qh >> (j + 12)) as u8 & 0x10;
        y[j] = ((b[8 + j] & 0xF) | hl) as f32 * d + m;
        y[j + 16] = ((b[8 + j] >> 4) | hh) as f32 * d + m;
    }
}

fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = fp16(b, 0);
    for i in 0..32 {
        y[i] = (b[2 + i] as i8) as f32 * d;
    }
}

fn q8_1(b: &[u8], y: &mut [f32]) {
    let d = fp16(b, 0);
    for i in 0..32 {
        y[i] = (b[4 + i] as i8) as f32 * d;
    }
}

fn q2_k(b: &[u8], y: &mut [f32]) {
    let sc = &b[0..16];
    let qs = &b[16..80];
    let d = fp16(b, 80);
    let dmin = fp16(b, 82);
    let mut o = 0usize;
    let mut is = 0usize;
    for half in 0..2usize {
        let q = &qs[half * 32..];
        let mut shift = 0u32;
        for _ in 0..4 {
            let s0 = sc[is];
            is += 1;
            let (dl, ml) = (d * (s0 & 0xF) as f32, dmin * (s0 >> 4) as f32);
            for l in 0..16 {
                y[o] = dl * ((q[l] >> shift) & 3) as f32 - ml;
                o += 1;
            }
            let s1 = sc[is];
            is += 1;
            let (dl, ml) = (d * (s1 & 0xF) as f32, dmin * (s1 >> 4) as f32);
            for l in 0..16 {
                y[o] = dl * ((q[l + 16] >> shift) & 3) as f32 - ml;
                o += 1;
            }
            shift += 2;
        }
    }
}

fn q3_k(b: &[u8], y: &mut [f32]) {
    let hmask = &b[0..32];
    let qs = &b[32..96];
    let scales = &b[96..108];
    let d = fp16(b, 108);
    let sc: [i8; 16] = std::array::from_fn(|k| q3k_scale(scales, k));
    let mut o = 0usize;
    let mut m: u8 = 1;
    for half in 0..2usize {
        let q = &qs[half * 32..];
        let mut shift = 0u32;
        for j in 0..4usize {
            let is = half * 8 + j * 2;
            let dl = d * sc[is] as f32;
            for l in 0..16 {
                let hv = if hmask[l] & m != 0 { 0 } else { 4 };
                y[o] = dl * (((q[l] >> shift) & 3) as i32 - hv) as f32;
                o += 1;
            }
            let dl = d * sc[is + 1] as f32;
            for l in 0..16 {
                let hv = if hmask[l + 16] & m != 0 { 0 } else { 4 };
                y[o] = dl * (((q[l + 16] >> shift) & 3) as i32 - hv) as f32;
                o += 1;
            }
            shift += 2;
            m <<= 1;
        }
    }
}

fn q4_k(b: &[u8], y: &mut [f32]) {
    let d = fp16(b, 0);
    let dmin = fp16(b, 2);
    let scales = &b[4..16];
    let qs = &b[16..144];
    let mut o = 0usize;
    let mut is = 0usize;
    for g in 0..4usize {
        let q = &qs[g * 32..];
        let (s1, m1) = scale_min_k4(is, scales);
        let (s2, m2) = scale_min_k4(is + 1, scales);
        let (d1, mm1) = (d * s1, dmin * m1);
        let (d2, mm2) = (d * s2, dmin * m2);
        for l in 0..32 {
            y[o + l] = d1 * (q[l] & 0xF) as f32 - mm1;
        }
        for l in 0..32 {
            y[o + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2;
        }
        o += 64;
        is += 2;
    }
}

fn q5_k(b: &[u8], y: &mut [f32]) {
    let d = fp16(b, 0);
    let dmin = fp16(b, 2);
    let scales = &b[4..16];
    let qh = &b[16..48];
    let ql = &b[48..176];
    let mut o = 0usize;
    let mut is = 0usize;
    let mut u1: u8 = 1;
    let mut u2: u8 = 2;
    for g in 0..4usize {
        let q = &ql[g * 32..];
        let (s1, m1) = scale_min_k4(is, scales);
        let (s2, m2) = scale_min_k4(is + 1, scales);
        let (d1, mm1) = (d * s1, dmin * m1);
        let (d2, mm2) = (d * s2, dmin * m2);
        for l in 0..32 {
            let hv = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            y[o + l] = d1 * ((q[l] & 0xF) as f32 + hv) - mm1;
        }
        for l in 0..32 {
            let hv = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            y[o + 32 + l] = d2 * ((q[l] >> 4) as f32 + hv) - mm2;
        }
        o += 64;
        is += 2;
        u1 = u1.wrapping_shl(2);
        u2 = u2.wrapping_shl(2);
    }
}

fn q6_k(b: &[u8], y: &mut [f32]) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let sc = &b[192..208];
    let d = fp16(b, 208);
    for half in 0..2usize {
        let ql = &ql[half * 64..];
        let qh = &qh[half * 32..];
        let sc = &sc[half * 8..];
        let y = &mut y[half * 128..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
            let q2 = ((ql[l + 32] & 0xF) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
            let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
            let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;
            y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            y[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            y[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            y[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

fn q8_k(b: &[u8], y: &mut [f32]) {
    let d = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    for i in 0..256 {
        y[i] = d * (b[4 + i] as i8) as f32;
    }
}

fn iq4_nl(b: &[u8], y: &mut [f32]) {
    let d = fp16(b, 0);
    for j in 0..16 {
        y[j] = d * KVALUES_IQ4NL[(b[2 + j] & 0xF) as usize] as f32;
        y[j + 16] = d * KVALUES_IQ4NL[(b[2 + j] >> 4) as usize] as f32;
    }
}

fn iq4_xs(b: &[u8], y: &mut [f32]) {
    let d = fp16(b, 0);
    let scales_h = u16::from_le_bytes([b[2], b[3]]);
    let scales_l = &b[4..8];
    let qs = &b[8..136];
    for ib in 0..8usize {
        let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
            | (((scales_h >> (2 * ib)) & 3) as i32) << 4;
        let dl = d * (ls - 32) as f32;
        let q = &qs[ib * 16..];
        let y = &mut y[ib * 32..];
        for j in 0..16 {
            y[j] = dl * KVALUES_IQ4NL[(q[j] & 0xF) as usize] as f32;
            y[j + 16] = dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
        }
    }
}

fn tq1_0(b: &[u8], y: &mut [f32]) {
    const POW3: [u16; 5] = [1, 3, 9, 27, 81];
    let qs = &b[0..48];
    let qh = &b[48..52];
    let d = fp16(b, 52);
    let mut o = 0usize;
    // Five ternary digits packed per byte for the first 160 values.
    for n in 0..5usize {
        for j in 0..32usize {
            let q = (qs[j] as u16).wrapping_mul(POW3[n]) & 0xFF;
            y[o] = ((q * 3) >> 8) as i32 as f32 - 1.0;
            y[o] *= d;
            o += 1;
        }
    }
    for n in 0..5usize {
        for j in 0..16usize {
            let q = (qs[j + 32] as u16).wrapping_mul(POW3[n]) & 0xFF;
            y[o] = ((q * 3) >> 8) as i32 as f32 - 1.0;
            y[o] *= d;
            o += 1;
        }
    }
    // Remaining 16 values use four digits per byte out of qh.
    for n in 0..4usize {
        for j in 0..4usize {
            let q = (qh[j] as u16).wrapping_mul(POW3[n]) & 0xFF;
            y[o] = ((q * 3) >> 8) as i32 as f32 - 1.0;
            y[o] *= d;
            o += 1;
        }
    }
}

fn tq2_0(b: &[u8], y: &mut [f32]) {
    let qs = &b[0..64];
    let d = fp16(b, 64);
    let mut o = 0usize;
    for j in (0..64).step_by(32) {
        for l in 0..4usize {
            for m in 0..32usize {
                let q = ((qs[j + m] >> (l * 2)) & 3) as i32;
                y[o] = (q - 1) as f32 * d;
                o += 1;
            }
        }
    }
}

fn mxfp4(b: &[u8], y: &mut [f32]) {
    let d = e8m0_to_f32_half(b[0]);
    for j in 0..16 {
        y[j] = d * KVALUES_MXFP4[(b[1 + j] & 0xF) as usize] as f32;
        y[j + 16] = d * KVALUES_MXFP4[(b[1 + j] >> 4) as usize] as f32;
    }
}

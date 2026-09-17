//! The integer dot path and the decode path must agree.
//!
//! `dequantize` is the definition of a format; `vec_dot_q8_1` is an optimization of
//! "decode, then dot". Any index, shift or scale mistake in the second shows up here as a
//! large relative error, which is the failure mode that otherwise reaches the user as
//! fluent but wrong text.

use gguf_core::DType;
use gguf_quant::{dequantize_vec, q8_1_bytes, quantize_q8_1, vec_dot_q8_1};

/// Deterministic byte source; the block contents only need to be varied, not meaningful.
fn pseudo_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

fn pseudo_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as i32 as f32 / 8388608.0) - 0.5
        })
        .collect()
}

/// Build block bytes whose f16 scale fields are sane. Random bytes can land on NaN or a
/// huge exponent, which would swamp the comparison without indicating a real bug.
fn blocks_with_scales(dtype: DType, n_blocks: usize, seed: u64) -> Vec<u8> {
    let ts = dtype.type_size();
    let mut raw = pseudo_bytes(ts * n_blocks, seed);
    let scale = half::f16::from_f32(0.0123).to_le_bytes();
    let min = half::f16::from_f32(-0.0071).to_le_bytes();
    for b in 0..n_blocks {
        let blk = &mut raw[b * ts..(b + 1) * ts];
        match dtype {
            DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::Iq4Nl => {
                blk[0..2].copy_from_slice(&scale);
            }
            DType::Q4_1 | DType::Q5_1 => {
                blk[0..2].copy_from_slice(&scale);
                blk[2..4].copy_from_slice(&min);
            }
            DType::Q2K => {
                blk[80..82].copy_from_slice(&scale);
                blk[82..84].copy_from_slice(&min);
            }
            DType::Q3K => blk[108..110].copy_from_slice(&scale),
            DType::Q4K | DType::Q5K => {
                blk[0..2].copy_from_slice(&scale);
                blk[2..4].copy_from_slice(&min);
            }
            DType::Q6K => blk[208..210].copy_from_slice(&scale),
            DType::Iq4Xs => blk[0..2].copy_from_slice(&scale),
            DType::Tq1_0 => blk[52..54].copy_from_slice(&scale),
            DType::Tq2_0 => blk[64..66].copy_from_slice(&scale),
            DType::Mxfp4 => blk[0] = 124, // 2^(124-128) = 1/16
            _ => {}
        }
    }
    raw
}

const TYPES: &[DType] = &[
    DType::Q4_0,
    DType::Q4_1,
    DType::Q5_0,
    DType::Q5_1,
    DType::Q8_0,
    DType::Q2K,
    DType::Q3K,
    DType::Q4K,
    DType::Q5K,
    DType::Q6K,
    DType::Iq4Nl,
    DType::Iq4Xs,
    DType::Tq1_0,
    DType::Tq2_0,
    DType::Mxfp4,
];

#[test]
fn integer_dot_agrees_with_decoded_dot() {
    for &dt in TYPES {
        // 1024 elements is several blocks of every type.
        let n = 1024;
        let n_blocks = n / dt.block_size();
        let w = blocks_with_scales(dt, n_blocks, 0xABCD ^ dt as u64);
        let decoded = dequantize_vec(dt, &w, n).expect("decode");

        let act = pseudo_floats(n, 0x1234);
        let mut aq = vec![0u8; q8_1_bytes(n)];
        quantize_q8_1(&act, &mut aq);

        let exact: f32 = decoded.iter().zip(&act).map(|(x, y)| x * y).sum();
        let fast = vec_dot_q8_1(dt, n, &w, &aq);

        // The only legitimate difference is the activation's own 8-bit rounding.
        let scale = decoded.iter().map(|v| v.abs()).sum::<f32>()
            * act.iter().fold(0f32, |m, v| m.max(v.abs()));
        let tol = 1e-2 * scale.max(1e-3);
        assert!(
            (exact - fast).abs() <= tol,
            "{}: decoded dot {exact}, integer dot {fast} (tolerance {tol})",
            dt.name()
        );
    }
}

#[test]
fn decoding_is_exactly_reproducible() {
    for &dt in TYPES {
        let n = dt.block_size() * 3;
        let w = blocks_with_scales(dt, 3, 0x5EED ^ dt as u64);
        let a = dequantize_vec(dt, &w, n).unwrap();
        let b = dequantize_vec(dt, &w, n).unwrap();
        assert_eq!(a, b, "{} decode is not deterministic", dt.name());
        assert!(
            a.iter().all(|v| v.is_finite()),
            "{} produced a non-finite value",
            dt.name()
        );
    }
}

#[test]
fn block_sizes_match_the_format() {
    // These are the numbers GGUF files are written against; getting one wrong silently
    // misaligns every tensor after it.
    let expect = [
        (DType::Q4_0, 32, 18),
        (DType::Q4_1, 32, 20),
        (DType::Q5_0, 32, 22),
        (DType::Q5_1, 32, 24),
        (DType::Q8_0, 32, 34),
        (DType::Q8_1, 32, 36),
        (DType::Q2K, 256, 84),
        (DType::Q3K, 256, 110),
        (DType::Q4K, 256, 144),
        (DType::Q5K, 256, 176),
        (DType::Q6K, 256, 210),
        (DType::Q8K, 256, 292),
        (DType::Iq4Nl, 32, 18),
        (DType::Iq4Xs, 256, 136),
        (DType::Tq1_0, 256, 54),
        (DType::Tq2_0, 256, 66),
        (DType::Mxfp4, 32, 17),
    ];
    for (dt, bs, ts) in expect {
        assert_eq!(dt.block_size(), bs, "{} block size", dt.name());
        assert_eq!(dt.type_size(), ts, "{} type size", dt.name());
    }
}

#[test]
fn codebook_quants_are_refused_rather_than_guessed() {
    for dt in [DType::Iq1S, DType::Iq1M, DType::Iq2Xxs, DType::Iq2Xs, DType::Iq2S, DType::Iq3Xxs, DType::Iq3S] {
        let err = dequantize_vec(dt, &vec![0u8; dt.type_size()], 256).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("codebook"), "{}: unhelpful error {msg}", dt.name());
    }
}

//! Elementwise and per-token kernels.

use gguf_core::{Activation, GluCfg, NormCfg, NormKind, RopeCfg, RopeKind};
use rayon::prelude::*;

pub fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = 1.0 / sum.max(1e-20);
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The tanh approximation, which is what every GGUF-era model was trained with.
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

#[inline(always)]
pub fn gelu_quick(x: f32) -> f32 {
    x / (1.0 + (-1.702 * x).exp())
}

#[inline(always)]
pub fn apply(act: Activation, x: f32, alpha: f32) -> f32 {
    match act {
        Activation::Silu => silu(x),
        Activation::Gelu | Activation::GeluTanh => gelu(x),
        Activation::GeluQuick => gelu_quick(x),
        Activation::Relu => x.max(0.0),
        Activation::Relu2 => {
            let r = x.max(0.0);
            r * r
        }
        Activation::Swish => x / (1.0 + (-alpha * x).exp()),
    }
}

pub fn norm(dst: &mut [f32], src: &[f32], weight: Option<&[f32]>, bias: Option<&[f32]>, cfg: NormCfg) {
    let dim = cfg.dim as usize;
    let n = cfg.n_tokens as usize;
    dst[..dim * n]
        .par_chunks_mut(dim)
        .zip(src[..dim * n].par_chunks(dim))
        .for_each(|(o, s)| {
            match cfg.kind {
                NormKind::Rms => {
                    let ms = s.iter().map(|v| v * v).sum::<f32>() / dim as f32;
                    let inv = 1.0 / (ms + cfg.eps).sqrt();
                    for (d, x) in o.iter_mut().zip(s) {
                        *d = x * inv;
                    }
                }
                NormKind::Layer => {
                    let mean = s.iter().sum::<f32>() / dim as f32;
                    let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
                    let inv = 1.0 / (var + cfg.eps).sqrt();
                    for (d, x) in o.iter_mut().zip(s) {
                        *d = (x - mean) * inv;
                    }
                }
            }
            if let Some(w) = weight {
                for (d, wv) in o.iter_mut().zip(w) {
                    *d *= wv;
                }
            }
            if let Some(b) = bias {
                for (d, bv) in o.iter_mut().zip(b) {
                    *d += bv;
                }
            }
            if cfg.scale != 1.0 {
                for d in o.iter_mut() {
                    *d *= cfg.scale;
                }
            }
        });
}

pub fn glu(dst: &mut [f32], gate: &[f32], up: &[f32], cfg: GluCfg) {
    let n = (cfg.dim * cfg.n_tokens) as usize;
    for i in 0..n {
        let (mut g, mut u) = (gate[i], up[i]);
        if cfg.limit > 0.0 {
            g = g.clamp(-cfg.limit, cfg.limit);
            u = u.clamp(-cfg.limit, cfg.limit);
        }
        dst[i] = apply(cfg.act, g, cfg.alpha) * u;
    }
}

pub fn apply_glu_inplace(gate: &mut [f32], up: &[f32], act: Activation, limit: f32, alpha: f32) {
    for (g, u) in gate.iter_mut().zip(up) {
        let mut gv = *g;
        let mut uv = *u;
        if limit > 0.0 {
            gv = gv.clamp(-limit, limit);
            uv = uv.clamp(-limit, limit);
        }
        *g = apply(act, gv, alpha) * uv;
    }
}

pub fn apply_act_inplace(x: &mut [f32], act: Activation, alpha: f32) {
    for v in x.iter_mut() {
        *v = apply(act, *v, alpha);
    }
}

/// YaRN's dimension correction range, in units of rotary dimension pairs.
fn yarn_corr_dims(n_rot: u32, orig_ctx: u32, base: f32, beta_fast: f32, beta_slow: f32) -> (f32, f32) {
    let dim = |rot: f32| -> f32 {
        n_rot as f32 * (orig_ctx as f32 / (rot * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * base.ln())
    };
    let start = dim(beta_fast).floor();
    let end = dim(beta_slow).ceil();
    (start.max(0.0), end.min(n_rot as f32 - 1.0))
}

#[inline(always)]
fn yarn_ramp(low: f32, high: f32, i: usize) -> f32 {
    let y = (i as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// Rotary position embedding, batched over tokens and heads.
///
/// `x` is laid out `[n_tokens][n_heads][head_dim]`. Dimensions at or beyond `n_rot` are
/// left untouched, which is how partial-rotary models keep part of each head unrotated.
pub fn rope(x: &mut [f32], positions: &[i32], n_heads: u32, freq_factors: Option<&[f32]>, cfg: &RopeCfg) {
    if cfg.kind == RopeKind::None {
        return;
    }
    let hd = cfg.head_dim as usize;
    let n_rot = (cfg.n_rot as usize).min(hd);
    let heads = n_heads as usize;
    let n_tokens = cfg.n_tokens as usize;
    let (corr_lo, corr_hi) = if cfg.ext_factor != 0.0 {
        yarn_corr_dims(cfg.n_rot, cfg.orig_ctx, cfg.freq_base, cfg.beta_fast, cfg.beta_slow)
    } else {
        (0.0, 0.0)
    };
    // YaRN's magnitude correction, applied once rather than per dimension.
    let mscale = if cfg.ext_factor != 0.0 {
        cfg.attn_factor * (1.0 + 0.1 * (1.0 / cfg.freq_scale).ln())
    } else {
        cfg.attn_factor
    };

    // M-RoPE with a single position axis is arithmetically identical to NeoX, which is
    // exactly the text-only case. The section widths only start to matter once image or
    // video positions differ per axis.
    let neox_pairing = matches!(cfg.kind, RopeKind::Neox | RopeKind::MRope);

    x[..hd * heads * n_tokens]
        .par_chunks_mut(hd * heads)
        .enumerate()
        .for_each(|(t, tok)| {
            let pos = positions.get(t).copied().unwrap_or(0) as f32;
            for h in 0..heads {
                let head = &mut tok[h * hd..(h + 1) * hd];
                for i in (0..n_rot).step_by(2) {
                    let ff = freq_factors.and_then(|f| f.get(i / 2).copied()).unwrap_or(1.0);
                    // One power, not a chain of two: `base^(-2i/n_rot)` computed directly
                    // is both more accurate than raising a precomputed step to a power and
                    // the identical expression every backend evaluates, which is what keeps
                    // them agreeing on a rotation angle.
                    let te = pos * cfg.freq_base.powf(-2.0 * (i / 2) as f32 / n_rot as f32) / ff;
                    let ti = cfg.freq_scale * te;
                    let theta = if cfg.ext_factor != 0.0 {
                        let mix = yarn_ramp(corr_lo, corr_hi, i / 2) * cfg.ext_factor;
                        ti * (1.0 - mix) + te * mix
                    } else {
                        ti
                    };
                    let (sin, cos) = theta.sin_cos();
                    let (sin, cos) = (sin * mscale, cos * mscale);
                    let (a, b) = if neox_pairing {
                        (i / 2, i / 2 + n_rot / 2)
                    } else {
                        (i, i + 1)
                    };
                    let (x0, x1) = (head[a], head[b]);
                    head[a] = x0 * cos - x1 * sin;
                    head[b] = x0 * sin + x1 * cos;
                }
            }
        });
}

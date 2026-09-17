//! Single-pass attention with online softmax rescaling.
//!
//! Scores are never materialised: each key is folded into the running maximum, denominator
//! and weighted value sum as it is read. That keeps memory flat in sequence length, which
//! matters more on CPU than on GPU because the score array would otherwise blow the cache
//! on long contexts.

use gguf_core::{AttnCfg, KvView};
use half::f16;
use rayon::prelude::*;

/// A cache read as either f32 or f16 storage.
///
/// The discriminant is loop-invariant, so the branch hoists out of the inner product; what
/// matters is that the two storage formats share one code path rather than two copies of
/// the same online-softmax loop.
#[derive(Clone, Copy)]
pub enum KvSlice<'a> {
    F32(&'a [f32]),
    F16(&'a [u16]),
}

impl KvSlice<'_> {
    #[inline(always)]
    fn get(&self, i: usize) -> f32 {
        match self {
            Self::F32(s) => s[i],
            Self::F16(s) => f16::from_bits(s[i]).to_f32(),
        }
    }
}

pub fn flash_attention(
    dst: &mut [f32],
    q: &[f32],
    k_cache: KvSlice<'_>,
    v_cache: KvSlice<'_>,
    kv: &KvView,
    cfg: &AttnCfg,
    sinks: Option<&[f32]>,
) {
    let hd = cfg.head_dim as usize;
    let n_heads = cfg.n_heads as usize;
    let n_kv_heads = cfg.n_kv_heads as usize;
    let n_tokens = cfg.n_tokens as usize;
    let gqa = (n_heads / n_kv_heads).max(1);
    let stride = kv.stride as usize;
    let base = kv.base as usize;

    // One task per (token, head): the outputs are disjoint and each reads the whole cache.
    dst[..n_heads * hd * n_tokens]
        .par_chunks_mut(hd)
        .enumerate()
        .for_each(|(idx, out)| {
            let t = idx / n_heads;
            let h = idx % n_heads;
            let kv_h = h / gqa;
            let q_head = &q[(t * n_heads + h) * hd..(t * n_heads + h) * hd + hd];

            // Absolute position of this query, which sets both the causal cut-off and the
            // left edge of a sliding window.
            let q_pos = cfg.start_pos as usize + t;
            let hi = (q_pos + 1).min(cfg.kv_len as usize);
            let lo = if cfg.window > 0 {
                (q_pos + 1).saturating_sub(cfg.window as usize)
            } else {
                0
            };

            let mut m = f32::NEG_INFINITY;
            let mut l = 0f32;
            out.fill(0.0);

            for p in lo..hi {
                let off = (base + p) * stride + kv_h * hd;
                let mut s = q_head
                    .iter()
                    .enumerate()
                    .map(|(i, a)| a * k_cache.get(off + i))
                    .sum::<f32>()
                    * cfg.scale;
                if cfg.softcap > 0.0 {
                    s = cfg.softcap * (s / cfg.softcap).tanh();
                }

                let new_m = m.max(s);
                let alpha = (m - new_m).exp();
                let p_s = (s - new_m).exp();
                for (i, o) in out.iter_mut().enumerate() {
                    *o = *o * alpha + p_s * v_cache.get(off + i);
                }
                l = l * alpha + p_s;
                m = new_m;
            }

            // A learned sink contributes to the denominator only: it is an extra logit
            // with no value vector, so it pulls probability mass away from real tokens.
            if let Some(sinks) = sinks {
                if let Some(&sink) = sinks.get(h) {
                    let new_m = m.max(sink);
                    let alpha = (m - new_m).exp();
                    if alpha != 1.0 {
                        for o in out.iter_mut() {
                            *o *= alpha;
                        }
                    }
                    l = l * alpha + (sink - new_m).exp();
                }
            }

            let inv = 1.0 / l.max(1e-20);
            for o in out.iter_mut() {
                *o *= inv;
            }
        });
}

//! Logit post-processing and token selection.
//!
//! The pipeline runs in a fixed order: penalties, then temperature, then truncation
//! (top-k, then min-p, then top-p), then the draw. Order matters - applying top-p before
//! top-k, for instance, changes which tokens survive - and this is the order llama.cpp and
//! the OpenAI API both document.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub typical_p: f32,
    /// Divides logits of tokens seen in the recent window.
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// 0 off, 1 or 2 select Mirostat v2.
    pub mirostat: u8,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            typical_p: 1.0,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            seed: 0,
        }
    }
}

impl SampleParams {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, ..Default::default() }
    }
}

/// A small, fast, explicitly-seeded generator.
///
/// Sampling is per-sequence state, not global: two concurrent requests with the same seed
/// must produce the same output regardless of how their token draws interleave, which a
/// shared global generator cannot promise.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // A zero seed would stick at zero; fall back to a fixed nonzero constant so the
        // "unseeded" case is still reproducible.
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

pub struct Sampler {
    pub params: SampleParams,
    rng: Rng,
    mirostat_mu: f32,
    /// Ring of recently emitted tokens, used by the repetition penalties.
    recent: Vec<u32>,
}

impl Sampler {
    pub fn new(params: SampleParams) -> Self {
        Self {
            rng: Rng::new(params.seed),
            mirostat_mu: 2.0 * params.mirostat_tau,
            recent: Vec::new(),
            params,
        }
    }

    pub fn accept(&mut self, token: u32) {
        self.recent.push(token);
        let keep = self.params.repeat_last_n.max(1);
        if self.recent.len() > keep {
            let drop = self.recent.len() - keep;
            self.recent.drain(..drop);
        }
    }

    pub fn reset(&mut self) {
        self.recent.clear();
        self.mirostat_mu = 2.0 * self.params.mirostat_tau;
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        let p = self.params;

        // A non-finite logit would poison the softmax for every other token.
        for v in logits.iter_mut() {
            if !v.is_finite() {
                *v = f32::NEG_INFINITY;
            }
        }

        apply_repetition(logits, &self.recent, p.repeat_penalty);
        apply_presence_frequency(logits, &self.recent, p.presence_penalty, p.frequency_penalty);

        if p.temperature <= 0.0 {
            return argmax(logits);
        }
        let inv_t = 1.0 / p.temperature;
        for v in logits.iter_mut() {
            *v *= inv_t;
        }

        if p.mirostat > 0 {
            return self.mirostat_v2(logits);
        }

        let mut cands = sorted_softmax(logits, p.top_k);
        if p.min_p > 0.0 {
            let floor = p.min_p * cands[0].1;
            let cut = cands.iter().position(|c| c.1 < floor).unwrap_or(cands.len()).max(1);
            cands.truncate(cut);
            renormalize(&mut cands);
        }
        if p.typical_p < 1.0 {
            typical_filter(&mut cands, p.typical_p);
            renormalize(&mut cands);
        }
        if p.top_p < 1.0 {
            let mut cum = 0.0;
            let mut cut = cands.len();
            for (i, c) in cands.iter().enumerate() {
                cum += c.1;
                if cum >= p.top_p {
                    cut = i + 1;
                    break;
                }
            }
            cands.truncate(cut.max(1));
            renormalize(&mut cands);
        }
        draw(&cands, self.rng.f32())
    }

    fn mirostat_v2(&mut self, logits: &[f32]) -> u32 {
        let all = sorted_softmax(logits, 0);
        let mut cut = all.len();
        for (i, (_, prob)) in all.iter().enumerate() {
            if -(prob.max(1e-20).log2()) > self.mirostat_mu {
                cut = i.max(1);
                break;
            }
        }
        let mut cands = all[..cut].to_vec();
        renormalize(&mut cands);
        let chosen = draw(&cands, self.rng.f32());
        let observed = all
            .iter()
            .find(|(id, _)| *id == chosen)
            .map(|(_, p)| *p)
            .unwrap_or(1e-20);
        let surprise = -(observed.max(1e-20).log2());
        self.mirostat_mu -= self.params.mirostat_eta * (surprise - self.params.mirostat_tau);
        chosen
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn apply_repetition(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if penalty == 1.0 || recent.is_empty() {
        return;
    }
    for &t in recent {
        if let Some(l) = logits.get_mut(t as usize) {
            // Dividing a positive logit and multiplying a negative one both move it down,
            // which is what makes the penalty meaningful on either side of zero.
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

fn apply_presence_frequency(logits: &mut [f32], recent: &[u32], presence: f32, frequency: f32) {
    if presence == 0.0 && frequency == 0.0 || recent.is_empty() {
        return;
    }
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in recent {
        *counts.entry(t).or_default() += 1;
    }
    for (tok, n) in counts {
        if let Some(l) = logits.get_mut(tok as usize) {
            *l -= presence + frequency * n as f32;
        }
    }
}

/// Sort by logit, keep at most `top_k`, then softmax just that prefix.
///
/// Softmaxing after truncation rather than before saves a pass over the whole vocabulary,
/// and is equivalent because the truncation is by rank.
fn sorted_softmax(logits: &[f32], top_k: usize) -> Vec<(u32, f32)> {
    let mut pairs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, *v))
        .collect();

    let k = if top_k > 0 { top_k.min(pairs.len()) } else { pairs.len() };
    if k < pairs.len() {
        pairs.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        pairs.truncate(k);
    }
    pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let max = pairs.first().map(|p| p.1).unwrap_or(0.0);
    let mut sum = 0.0f32;
    for p in pairs.iter_mut() {
        p.1 = (p.1 - max).exp();
        sum += p.1;
    }
    let inv = 1.0 / sum.max(1e-20);
    for p in pairs.iter_mut() {
        p.1 *= inv;
    }
    pairs
}

fn renormalize(cands: &mut [(u32, f32)]) {
    let sum: f32 = cands.iter().map(|c| c.1).sum();
    let inv = 1.0 / sum.max(1e-20);
    for c in cands.iter_mut() {
        c.1 *= inv;
    }
}

/// Locally-typical sampling: keep the tokens whose surprise is closest to the
/// distribution's entropy, rather than simply the most likely ones.
fn typical_filter(cands: &mut Vec<(u32, f32)>, typical_p: f32) {
    let entropy: f32 = -cands.iter().map(|c| c.1 * c.1.max(1e-20).ln()).sum::<f32>();
    let mut scored: Vec<(usize, f32)> = cands
        .iter()
        .enumerate()
        .map(|(i, c)| (i, (-c.1.max(1e-20).ln() - entropy).abs()))
        .collect();
    scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let mut cum = 0.0;
    let mut keep = Vec::new();
    for (i, _) in scored {
        cum += cands[i].1;
        keep.push(i);
        if cum >= typical_p {
            break;
        }
    }
    keep.sort_unstable();
    *cands = keep.into_iter().map(|i| cands[i]).collect();
}

fn draw(cands: &[(u32, f32)], r: f32) -> u32 {
    let mut cum = 0.0;
    for (id, p) in cands {
        cum += p;
        if r < cum {
            return *id;
        }
    }
    cands.last().map(|c| c.0).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_the_maximum() {
        let mut s = Sampler::new(SampleParams::greedy());
        let mut logits = vec![0.1, 5.0, 0.2, 4.9];
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn same_seed_gives_the_same_stream() {
        let params = SampleParams { seed: 7, ..Default::default() };
        let run = || {
            let mut s = Sampler::new(params);
            (0..16)
                .map(|_| {
                    let mut l: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
                    let t = s.sample(&mut l);
                    s.accept(t);
                    t
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn top_k_of_one_is_deterministic() {
        let params = SampleParams { top_k: 1, temperature: 1.0, seed: 3, ..Default::default() };
        let mut s = Sampler::new(params);
        let mut logits = vec![1.0, 9.0, 2.0];
        assert_eq!(s.sample(&mut logits), 1);
    }

    #[test]
    fn repetition_penalty_pushes_a_repeated_token_down() {
        let params = SampleParams {
            temperature: 0.0,
            repeat_penalty: 2.0,
            repeat_last_n: 8,
            ..Default::default()
        };
        let mut s = Sampler::new(params);
        s.accept(1);
        let mut logits = vec![4.0, 5.0, 0.0];
        // Token 1 starts ahead but is halved, so token 0 wins.
        assert_eq!(s.sample(&mut logits), 0);
    }

    #[test]
    fn non_finite_logits_do_not_poison_the_draw() {
        let mut s = Sampler::new(SampleParams { temperature: 1.0, seed: 1, ..Default::default() });
        let mut logits = vec![f32::NAN, 2.0, f32::INFINITY, 1.0];
        let t = s.sample(&mut logits);
        assert!(t == 1 || t == 3, "picked a non-finite logit: {t}");
    }
}

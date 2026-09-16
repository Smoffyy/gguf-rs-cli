use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

static SEED: AtomicU64 = AtomicU64::new(12345);

pub fn set_seed(s: u64) { SEED.store(s, Ordering::Relaxed); }

fn rand_f32() -> f32 {
    let s = SEED.fetch_update(Ordering::Relaxed, Ordering::Relaxed,
        |v| Some(v.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)))
        .unwrap();
    ((s >> 33) as f32) / (u32::MAX as f32)
}

#[derive(Clone, Copy, Debug)]
pub struct SampleParams {
    pub temperature:       f32,
    pub top_k:              usize,
    pub top_p:               f32,
    pub min_p:                f32,
    pub rep_penalty:           f32,
    pub presence_penalty:       f32,
    pub frequency_penalty:       f32,
    pub mirostat:                  u8,
    pub mirostat_tau:                f32,
    pub mirostat_eta:                 f32,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.7, top_k: 40, top_p: 0.9, min_p: 0.0, rep_penalty: 1.1,
            presence_penalty: 0.0, frequency_penalty: 0.0,
            mirostat: 0, mirostat_tau: 5.0, mirostat_eta: 0.1,
        }
    }
}

fn greedy(logits: &[f32]) -> usize {
    logits.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i).unwrap_or(0)
}

fn apply_rep_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if penalty == 1.0 { return; }
    for &tok in recent {
        if let Some(l) = logits.get_mut(tok as usize) {
            if *l > 0.0 { *l /= penalty; } else { *l *= penalty; }
        }
    }
}

fn apply_presence_freq_penalty(logits: &mut [f32], recent: &[u32], presence: f32, frequency: f32) {
    if presence == 0.0 && frequency == 0.0 { return; }
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in recent { *counts.entry(t).or_insert(0) += 1; }
    for (tok, count) in counts {
        if let Some(l) = logits.get_mut(tok as usize) {
            *l -= presence + frequency * count as f32;
        }
    }
}

fn softmax_sorted(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    let max = pairs[0].1;
    let mut sum = 0.0f32;
    for p in pairs.iter_mut() { p.1 = (p.1 - max).exp(); sum += p.1; }
    for p in pairs.iter_mut() { p.1 /= sum; }
    pairs
}

fn sample_from(pairs: &[(usize, f32)]) -> usize {
    let r = rand_f32();
    let mut cum = 0.0f32;
    for (idx, prob) in pairs {
        cum += prob;
        if r < cum { return *idx; }
    }
    pairs.last().map(|(i, _)| *i).unwrap_or(0)
}

fn sample_mirostat_v2(logits: &[f32], tau: f32, eta: f32, mu: &mut f32) -> usize {
    let pairs = softmax_sorted(logits);
    let mut cut = pairs.len();
    for (i, (_, p)) in pairs.iter().enumerate() {
        if -(p.max(1e-12).log2()) > *mu { cut = i.max(1); break; }
    }
    let mut candidates = pairs[..cut].to_vec();
    let s: f32 = candidates.iter().map(|p| p.1).sum();
    for p in candidates.iter_mut() { p.1 /= s; }
    let chosen = sample_from(&candidates);
    let chosen_p = pairs.iter().find(|(i, _)| *i == chosen).map(|(_, p)| *p).unwrap_or(1e-12);
    let surprise = -(chosen_p.max(1e-12).log2());
    *mu -= eta * (surprise - tau);
    chosen
}

pub fn sample(logits: &mut Vec<f32>, params: &SampleParams, recent: &[u32], mirostat_mu: &mut f32) -> usize {
    for v in logits.iter_mut() {
        if !v.is_finite() { *v = -1e9; }
    }

    apply_rep_penalty(logits, recent, params.rep_penalty);
    apply_presence_freq_penalty(logits, recent, params.presence_penalty, params.frequency_penalty);

    if params.temperature <= 0.0 { return greedy(logits); }

    for v in logits.iter_mut() { *v /= params.temperature; }

    if params.mirostat == 1 || params.mirostat == 2 {
        return sample_mirostat_v2(logits, params.mirostat_tau, params.mirostat_eta, mirostat_mu);
    }

    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    if params.top_k > 0 && params.top_k < pairs.len() {
        pairs.truncate(params.top_k);
    }

    let max = pairs[0].1;
    let mut sum = 0.0f32;
    for p in pairs.iter_mut() { p.1 = (p.1 - max).exp(); sum += p.1; }
    for p in pairs.iter_mut() { p.1 /= sum; }

    if params.min_p > 0.0 {
        let top_prob = pairs[0].1;
        let threshold = params.min_p * top_prob;
        let cut = pairs.iter().position(|p| p.1 < threshold).unwrap_or(pairs.len()).max(1);
        pairs.truncate(cut);
        let s: f32 = pairs.iter().map(|p| p.1).sum();
        for p in pairs.iter_mut() { p.1 /= s; }
    }

    if params.top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut cut = pairs.len();
        for (i, p) in pairs.iter().enumerate() {
            cum += p.1;
            if cum >= params.top_p { cut = i + 1; break; }
        }
        pairs.truncate(cut);
        let s: f32 = pairs.iter().map(|p| p.1).sum();
        for p in pairs.iter_mut() { p.1 /= s; }
    }

    sample_from(&pairs)
}

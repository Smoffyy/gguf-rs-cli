use std::sync::atomic::{AtomicU64, Ordering};

static SEED: AtomicU64 = AtomicU64::new(12345);

pub fn set_seed(s: u64) { SEED.store(s, Ordering::Relaxed); }

fn rand_f32() -> f32 {
    let s = SEED.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
    }).unwrap();
    ((s >> 33) as f32) / (u32::MAX as f32)
}

pub fn sample_greedy(logits: &[f32]) -> usize {
    logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
}

// Apply temperature then sample from the distribution
pub fn sample(logits: &mut Vec<f32>, temperature: f32) -> usize {
    if temperature <= 0.0 { return sample_greedy(logits); }

    for v in logits.iter_mut() { *v /= temperature; }

    // Softmax
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    for v in logits.iter_mut() { *v /= sum; }

    // Multinomial sample
    let r = rand_f32();
    let mut cum = 0.0f32;
    for (i, &p) in logits.iter().enumerate() {
        cum += p;
        if r < cum { return i; }
    }
    logits.len() - 1
}

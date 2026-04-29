use rayon::prelude::*;

// Normalize x by RMS, then scale by weight vector
pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    for (xi, wi) in x.iter_mut().zip(weight.iter()) {
        *xi = *xi / rms * wi;
    }
}

// Matrix-vector product: out(m) = A(m,k) @ b(k)
// Parallelized over output rows via rayon
pub fn matmul(out: &mut [f32], a: &[f32], b: &[f32], _m: usize, k: usize) {
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        *o = a[i*k..(i+1)*k].iter().zip(b.iter()).map(|(x,y)| x*y).sum();
    });
}

// In-place softmax
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() { *v = (*v - max).exp(); sum += *v; }
    for v in x.iter_mut() { *v /= sum; }
}

// SiLU: x * sigmoid(x)
pub fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

pub fn add_into(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b.iter()) { *x += y; }
}
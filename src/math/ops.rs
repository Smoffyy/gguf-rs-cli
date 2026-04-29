pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    let rms = (x.iter().map(|v|v*v).sum::<f32>()/x.len() as f32+eps).sqrt();
    for (xi,wi) in x.iter_mut().zip(weight.iter()) { *xi=*xi/rms*wi; }
}
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s=0f32;
    for v in x.iter_mut() { *v=(*v-max).exp(); s+=*v; }
    for v in x.iter_mut() { *v/=s; }
}
pub fn silu(x: f32) -> f32 { x/(1.0+(-x).exp()) }
pub fn add_into(a: &mut [f32], b: &[f32]) {
    for (x,y) in a.iter_mut().zip(b.iter()) { *x+=y; }
}
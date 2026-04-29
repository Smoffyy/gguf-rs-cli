use std::sync::atomic::{AtomicU64,Ordering};
static SEED:AtomicU64=AtomicU64::new(12345);
pub fn set_seed(s:u64){SEED.store(s,Ordering::Relaxed);}
fn rand_f32()->f32{
    let s=SEED.fetch_update(Ordering::Relaxed,Ordering::Relaxed,|v|Some(v.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))).unwrap();
    ((s>>33) as f32)/(u32::MAX as f32)
}
pub fn sample(logits:&mut Vec<f32>,temp:f32)->usize{
    if temp<=0.0{return logits.iter().enumerate().max_by(|a,b|a.1.partial_cmp(b.1).unwrap()).map(|(i,_)|i).unwrap_or(0);}
    for v in logits.iter_mut(){*v/=temp;}
    let max=logits.iter().cloned().fold(f32::NEG_INFINITY,f32::max);
    let mut s=0f32;
    for v in logits.iter_mut(){*v=(*v-max).exp();s+=*v;}
    for v in logits.iter_mut(){*v/=s;}
    let r=rand_f32(); let mut c=0f32;
    for(i,&p) in logits.iter().enumerate(){c+=p;if r<c{return i;}}
    logits.len()-1
}
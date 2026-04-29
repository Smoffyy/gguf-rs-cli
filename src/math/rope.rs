pub fn apply_rope(q:&mut[f32],k:&mut[f32],pos:usize,hd:usize,freq:f32,nh:usize,nkv:usize){
    let half=hd/2;
    rope_heads(q,pos,hd,freq,nh,half);
    rope_heads(k,pos,hd,freq,nkv,half);
}
fn rope_heads(x:&mut[f32],pos:usize,hd:usize,freq:f32,n:usize,half:usize){
    for h in 0..n {
        for i in 0..half {
            let t=(pos as f32)*freq.powf(-2.0*i as f32/hd as f32);
            let(s,c)=t.sin_cos();
            let b=h*hd+i;
            let x0=x[b]; let x1=x[b+half];
            x[b]=x0*c-x1*s; x[b+half]=x0*s+x1*c;
        }
    }
}
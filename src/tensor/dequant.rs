use half::f16;
use std::sync::Arc;
use memmap2::Mmap;
use crate::gguf::types::GgmlType;
use rayon::prelude::*;

pub fn dequantize(typ: GgmlType, data: &[u8], n: usize) -> anyhow::Result<Vec<f32>> {
    Ok(match typ {
        GgmlType::F32  => data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect(),
        GgmlType::F16  => data.chunks_exact(2).map(|b| f16::from_le_bytes([b[0],b[1]]).to_f32()).collect(),
        GgmlType::Q4_0 => dq40(data, n),
        GgmlType::Q4_1 => dq41(data, n),
        GgmlType::Q5_0 => dq50(data, n),
        GgmlType::Q5_1 => dq51(data, n),
        GgmlType::Q8_0 => dq80(data, n),
        GgmlType::Q8_1 => dq81(data, n),
        GgmlType::Q2K  => dq2k(data, n),
        GgmlType::Q3K  => dq3k(data, n),
        GgmlType::Q4K  => dq4k(data, n),
        GgmlType::Q5K  => dq5k(data, n),
        GgmlType::Q6K  => dq6k(data, n),
        GgmlType::Q8K  => dq8k(data, n),
    })
}


fn dq40(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(18).enumerate() {
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for j in 0..16 {
            o[b*32+j]    = ((blk[2+j] & 0xF) as i32 - 8) as f32 * d;
            o[b*32+j+16] = ((blk[2+j] >>  4) as i32 - 8) as f32 * d;
        }
    }
    o
}
fn dq41(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(20).enumerate() {
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let m = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        for j in 0..16 {
            o[b*32+j]    = (blk[4+j] & 0xF) as f32 * d + m;
            o[b*32+j+16] = (blk[4+j] >>  4) as f32 * d + m;
        }
    }
    o
}
fn dq50(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(22).enumerate() {
        let d  = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        for j in 0..16 {
            let lo = ((blk[6+j] & 0xF) as i32 | (((qh >> j)      & 1) as i32) * 16) - 16;
            let hi = ((blk[6+j] >>  4) as i32 | (((qh >> (j+16)) & 1) as i32) * 16) - 16;
            o[b*32+j]    = lo as f32 * d;
            o[b*32+j+16] = hi as f32 * d;
        }
    }
    o
}
fn dq51(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(24).enumerate() {
        let d  = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let m  = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        for j in 0..16 {
            let lo = (blk[8+j] & 0xF) as f32 + ((qh >> j)      & 1) as f32 * 16.0;
            let hi = (blk[8+j] >>  4) as f32 + ((qh >> (j+16)) & 1) as f32 * 16.0;
            o[b*32+j]    = lo * d + m;
            o[b*32+j+16] = hi * d + m;
        }
    }
    o
}
fn dq80(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(34).enumerate() {
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for i in 0..32 { o[b*32+i] = (blk[2+i] as i8) as f32 * d; }
    }
    o
}
fn dq81(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(36).enumerate() {
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for i in 0..32 { o[b*32+i] = (blk[4+i] as i8) as f32 * d; }
    }
    o
}


fn dq2k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(84).enumerate() {
        let sc   = &blk[0..16];
        let qs   = &blk[16..80];
        let d    = f16::from_le_bytes([blk[80], blk[81]]).to_f32();
        let dmin = f16::from_le_bytes([blk[82], blk[83]]).to_f32();
        let mut y  = b*256;
        let mut is = 0usize;
        for nblock in 0..2usize {
            let qoff = nblock*32;
            let mut shift = 0u32;
            for _ in 0..4 {
                let s0 = sc[is]; is += 1;
                let (dl0, ml0) = (d*(s0&0xF) as f32, dmin*(s0>>4) as f32);
                for l in 0..16 {
                    let qv = ((qs[qoff+l] >> shift) & 3) as f32;
                    o[y] = dl0*qv - ml0; y += 1;
                }
                let s1 = sc[is]; is += 1;
                let (dl1, ml1) = (d*(s1&0xF) as f32, dmin*(s1>>4) as f32);
                for l in 0..16 {
                    let qv = ((qs[qoff+l+16] >> shift) & 3) as f32;
                    o[y] = dl1*qv - ml1; y += 1;
                }
                shift += 2;
            }
        }
    }
    o
}

fn decode_q3k_scale(sc: &[u8], k: usize) -> i8 {
    let (lo4, hi2) = if k < 4 {
        (sc[k] & 0xF,          (sc[8+k]   >> 0) & 0x3)
    } else if k < 8 {
        (sc[k] & 0xF,          (sc[8+k-4] >> 2) & 0x3)
    } else if k < 12 {
        ((sc[k-8] >> 4) & 0xF, (sc[k]     >> 4) & 0x3)
    } else {
        ((sc[k-8] >> 4) & 0xF, (sc[k-4]   >> 6) & 0x3)
    };
    ((lo4 | (hi2 << 4)) as i8).wrapping_sub(32)
}

fn dq3k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(110).enumerate() {
        let hmask  = &blk[0..32];
        let qs     = &blk[32..96];
        let scales = &blk[96..108];
        let d      = f16::from_le_bytes([blk[108], blk[109]]).to_f32();
        let sc: [i8; 16] = std::array::from_fn(|k| decode_q3k_scale(scales, k));
        let mut y = b*256;
        let mut m: u8 = 1;
        for nblock in 0..2usize {
            let qoff = nblock*32;
            let mut shift = 0u32;
            for j in 0..4usize {
                let is0 = nblock*8 + j*2;
                let dl0 = d * sc[is0] as f32;
                for l in 0..16 {
                    let qv = ((qs[qoff+l] >> shift) & 3) as i32;
                    let hv = if hmask[l] & m != 0 { 0 } else { 4 };
                    o[y] = dl0 * (qv - hv) as f32; y += 1;
                }
                let dl1 = d * sc[is0+1] as f32;
                for l in 0..16 {
                    let qv = ((qs[qoff+l+16] >> shift) & 3) as i32;
                    let hv = if hmask[l+16] & m != 0 { 0 } else { 4 };
                    o[y] = dl1 * (qv - hv) as f32; y += 1;
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
    o
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    let (d, m) = if j < 4 {
        (q[j] & 63, q[j+4] & 63)
    } else {
        ((q[j+4] & 0xF) | ((q[j-4] >> 6) << 4),
         (q[j+4] >>  4) | ((q[j]   >> 6) << 4))
    };
    (d as f32, m as f32)
}

fn dq4k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(144).enumerate() {
        let df     = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin   = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let scales = &blk[4..16];
        let qs     = &blk[16..144];
        let mut y    = b*256;
        let mut qoff = 0usize;
        let mut is   = 0usize;
        for _ in 0..4 {
            let (sc1, mn1) = get_scale_min_k4(is, scales);
            let (d1, m1)   = (df*sc1, dmin*mn1);
            let (sc2, mn2) = get_scale_min_k4(is+1, scales);
            let (d2, m2)   = (df*sc2, dmin*mn2);
            for l in 0..32 { o[y+l]    = d1 * (qs[qoff+l] & 0xF) as f32 - m1; }
            for l in 0..32 { o[y+32+l] = d2 * (qs[qoff+l] >>  4) as f32 - m2; }
            y += 64; qoff += 32; is += 2;
        }
    }
    o
}

fn dq5k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(176).enumerate() {
        let df     = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin   = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let scales = &blk[4..16];
        let qh     = &blk[16..48];
        let ql     = &blk[48..176];
        let mut y     = b*256;
        let mut qloff  = 0usize;
        let mut is    = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..4 {
            let (sc1, mn1) = get_scale_min_k4(is, scales);
            let (d1, m1)   = (df*sc1, dmin*mn1);
            let (sc2, mn2) = get_scale_min_k4(is+1, scales);
            let (d2, m2)   = (df*sc2, dmin*mn2);
            for l in 0..32 {
                let hv = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                o[y+l] = d1 * ((ql[qloff+l] & 0xF) as f32 + hv) - m1;
            }
            for l in 0..32 {
                let hv = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                o[y+32+l] = d2 * ((ql[qloff+l] >> 4) as f32 + hv) - m2;
            }
            y += 64; qloff += 32; is += 2;
            u1 = u1.wrapping_shl(2); u2 = u2.wrapping_shl(2);
        }
    }
    o
}

fn dq6k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(210).enumerate() {
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let d  = f16::from_le_bytes([blk[208], blk[209]]).to_f32();
        let mut y = b*256;
        for nblock in 0..2usize {
            let qloff = nblock*64;
            let qhoff = nblock*32;
            let scoff = nblock*8;
            for l in 0..32 {
                let is = l/16;
                let q1 = ((ql[qloff+l]    & 0xF) as i32 | (((qh[qhoff+l] >> 0) & 3) as i32) << 4) - 32;
                let q2 = ((ql[qloff+l+32] & 0xF) as i32 | (((qh[qhoff+l] >> 2) & 3) as i32) << 4) - 32;
                let q3 = ((ql[qloff+l]    >> 4)  as i32 | (((qh[qhoff+l] >> 4) & 3) as i32) << 4) - 32;
                let q4 = ((ql[qloff+l+32] >> 4)  as i32 | (((qh[qhoff+l] >> 6) & 3) as i32) << 4) - 32;
                o[y+l]    = d * (sc[scoff+is]   as i8) as f32 * q1 as f32;
                o[y+l+32] = d * (sc[scoff+is+2] as i8) as f32 * q2 as f32;
                o[y+l+64] = d * (sc[scoff+is+4] as i8) as f32 * q3 as f32;
                o[y+l+96] = d * (sc[scoff+is+6] as i8) as f32 * q4 as f32;
            }
            y += 128;
        }
    }
    o
}

fn dq8k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for (b, blk) in data.chunks_exact(292).enumerate() {
        let d = f32::from_le_bytes([blk[0],blk[1],blk[2],blk[3]]);
        for i in 0..256 { o[b*256+i] = d * (blk[4+i] as i8) as f32; }
    }
    o
}


pub struct QuantTensor {
    mmap:   Arc<Mmap>,
    offset: usize,
    len:    usize,
    pub typ:  GgmlType,
    pub rows: usize,
    pub cols: usize,
}

impl QuantTensor {
    pub fn new(mmap: Arc<Mmap>, offset: usize, len: usize,
               typ: GgmlType, dims: &[u64]) -> Self {
        let cols = dims[0] as usize;
        let rows = if dims.len() > 1 { dims[1] as usize } else { 1 };
        Self { mmap, offset, len, typ, rows, cols }
    }

    #[inline] pub fn data(&self) -> &[u8] { &self.mmap[self.offset..self.offset+self.len] }

    pub fn to_f32(&self) -> Vec<f32> {
        dequantize(self.typ, self.data(), self.rows*self.cols).unwrap()
    }

    pub fn expert(&self, e: usize, rows_per_expert: usize) -> QuantTensor {
        let expert_bytes = self.typ.byte_size(self.cols) * rows_per_expert;
        QuantTensor {
            mmap: self.mmap.clone(),
            offset: self.offset + e * expert_bytes,
            len: expert_bytes,
            typ: self.typ,
            rows: rows_per_expert,
            cols: self.cols,
        }
    }

    pub fn get_row(&self, r: usize) -> Vec<f32> {
        let rb = self.typ.byte_size(self.cols);
        let d  = self.data();
        dequantize(self.typ, &d[r*rb..(r+1)*rb], self.cols).unwrap()
    }

    pub fn row_dot(&self, r: usize, b: &[f32]) -> f32 {
        let rb = self.typ.byte_size(self.cols);
        let d  = self.data();
        let d  = &d[r*rb..(r+1)*rb];
        match self.typ {
            GgmlType::F32 => (0..self.cols)
                .map(|i| f32::from_le_bytes([d[i*4],d[i*4+1],d[i*4+2],d[i*4+3]])*b[i]).sum(),
            GgmlType::F16 => (0..self.cols)
                .map(|i| f16::from_le_bytes([d[i*2],d[i*2+1]]).to_f32()*b[i]).sum(),
            GgmlType::Q4_0 => (0..self.cols/32).map(|blk| {
                let d = &d[blk*18..]; let sc = f16::from_le_bytes([d[0],d[1]]).to_f32();
                (0..16).map(|j|
                      ((d[2+j]&0xF) as i32-8) as f32*sc*b[blk*32+j]
                    + ((d[2+j]>> 4) as i32-8) as f32*sc*b[blk*32+j+16]).sum::<f32>()
            }).sum(),
            GgmlType::Q8_0 => (0..self.cols/32).map(|blk| {
                let d = &d[blk*34..]; let sc = f16::from_le_bytes([d[0],d[1]]).to_f32();
                (0..32).map(|i| (d[2+i] as i8) as f32*sc*b[blk*32+i]).sum::<f32>()
            }).sum(),
            _ => { let v = dequantize(self.typ, d, self.cols).unwrap();
                   v.iter().zip(b).map(|(x,y)| x*y).sum() }
        }
    }

    pub fn matvec(&self, out: &mut [f32], b: &[f32]) {
        out.par_iter_mut().enumerate().for_each(|(i, o)| *o = self.row_dot(i, b));
    }


    pub fn pack_q4_0_for_gpu(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.rows*self.cols/32*5);
        for blk in self.data().chunks_exact(18) {
            v.push(f16::from_le_bytes([blk[0],blk[1]]).to_f32().to_bits());
            for i in 0..4 { v.push(u32::from_le_bytes([blk[2+i*4],blk[3+i*4],blk[4+i*4],blk[5+i*4]])); }
        }
        v
    }
    pub fn pack_q4_1_for_gpu(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.rows*self.cols/32*6);
        for blk in self.data().chunks_exact(20) {
            v.push(f16::from_le_bytes([blk[0],blk[1]]).to_f32().to_bits());
            v.push(f16::from_le_bytes([blk[2],blk[3]]).to_f32().to_bits());
            for i in 0..4 { v.push(u32::from_le_bytes([blk[4+i*4],blk[5+i*4],blk[6+i*4],blk[7+i*4]])); }
        }
        v
    }
    pub fn pack_q8_0_for_gpu(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.rows*self.cols/32*9);
        for blk in self.data().chunks_exact(34) {
            v.push(f16::from_le_bytes([blk[0],blk[1]]).to_f32().to_bits());
            for i in 0..8 { v.push(u32::from_le_bytes([blk[2+i*4],blk[3+i*4],blk[4+i*4],blk[5+i*4]])); }
        }
        v
    }
    pub fn pack_q3k_for_gpu(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.rows*self.cols/256*28);
        for blk in self.data().chunks_exact(110) {
            v.push(f16::from_le_bytes([blk[108],blk[109]]).to_f32().to_bits());
            for i in 0..8  { v.push(u32::from_le_bytes([blk[i*4],blk[i*4+1],blk[i*4+2],blk[i*4+3]])); }
            for i in 0..16 { v.push(u32::from_le_bytes([blk[32+i*4],blk[33+i*4],blk[34+i*4],blk[35+i*4]])); }
            for i in 0..3  { v.push(u32::from_le_bytes([blk[96+i*4],blk[97+i*4],blk[98+i*4],blk[99+i*4]])); }
        }
        v
    }
    pub fn pack_q4k_for_gpu(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.rows*self.cols/256*48);
        for blk in self.data().chunks_exact(144) {
            let df   = f16::from_le_bytes([blk[0],blk[1]]).to_f32();
            let dmin = f16::from_le_bytes([blk[2],blk[3]]).to_f32();
            let sc   = &blk[4..16]; let qs = &blk[16..144];
            for i in 0..8 { let (s,_)=get_scale_min_k4(i,sc); v.push((df*s).to_bits()); }
            for i in 0..8 { let (_,m)=get_scale_min_k4(i,sc); v.push((dmin*m).to_bits()); }
            for i in 0..32 { v.push(u32::from_le_bytes([qs[i*4],qs[i*4+1],qs[i*4+2],qs[i*4+3]])); }
        }
        v
    }
    pub fn pack_q5k_for_gpu(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.rows*self.cols/256*45);
        for blk in self.data().chunks_exact(176) {
            v.push(f16::from_le_bytes([blk[0],blk[1]]).to_f32().to_bits());
            v.push(f16::from_le_bytes([blk[2],blk[3]]).to_f32().to_bits());
            for i in 0..3  { v.push(u32::from_le_bytes([blk[4+i*4],blk[5+i*4],blk[6+i*4],blk[7+i*4]])); }
            for i in 0..8  { v.push(u32::from_le_bytes([blk[16+i*4],blk[17+i*4],blk[18+i*4],blk[19+i*4]])); }
            for i in 0..32 { v.push(u32::from_le_bytes([blk[48+i*4],blk[49+i*4],blk[50+i*4],blk[51+i*4]])); }
        }
        v
    }
    pub fn pack_q6k_for_gpu(&self) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.rows*self.cols/256*64);
        for blk in self.data().chunks_exact(210) {
            let ql=&blk[0..128]; let qh=&blk[128..192]; let sc=&blk[192..208];
            let df = f16::from_le_bytes([blk[208],blk[209]]).to_f32();
            for i in 0..16 { v.push((df*(sc[i] as i8) as f32).to_bits()); }
            for i in 0..32 { v.push(u32::from_le_bytes([ql[i*4],ql[i*4+1],ql[i*4+2],ql[i*4+3]])); }
            for i in 0..16 { v.push(u32::from_le_bytes([qh[i*4],qh[i*4+1],qh[i*4+2],qh[i*4+3]])); }
        }
        v
    }
}

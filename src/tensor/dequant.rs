use half::f16;
use std::sync::Arc;
use memmap2::Mmap;
use crate::gguf::types::GgmlType;
use rayon::prelude::*;

pub fn dequantize(typ: GgmlType, data: &[u8], n: usize) -> anyhow::Result<Vec<f32>> {
    Ok(match typ {
        GgmlType::F32  => (0..n).map(|i| f32::from_le_bytes([data[i*4],data[i*4+1],data[i*4+2],data[i*4+3]])).collect(),
        GgmlType::F16  => (0..n).map(|i| f16::from_le_bytes([data[i*2],data[i*2+1]]).to_f32()).collect(),
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

// Q4_0: 18 bytes/block of 32 weights.
// Layout: [f16 scale (2 bytes)] [16 nibble bytes]
// Lower nibble of byte j  -> weight j      (range 0..15)
// Upper nibble of byte j  -> weight j + 16 (range 16..31)
fn dq40(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/32 {
        let d  = &data[b*18..];
        let sc = f16::from_le_bytes([d[0], d[1]]).to_f32();
        for j in 0..16 {
            o[b*32 + j]      = ((d[2+j] & 0xf) as i32 - 8) as f32 * sc;
            o[b*32 + j + 16] = ((d[2+j] >>  4) as i32 - 8) as f32 * sc;
        }
    }
    o
}
fn dq41(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/32 {
        let d  = &data[b*20..];
        let sc = f16::from_le_bytes([d[0], d[1]]).to_f32();
        let mn = f16::from_le_bytes([d[2], d[3]]).to_f32();
        for j in 0..16 {
            o[b*32 + j]      = (d[4+j] & 0xf) as f32 * sc + mn;
            o[b*32 + j + 16] = (d[4+j] >>  4) as f32 * sc + mn;
        }
    }
    o
}
fn dq50(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/32 {
        let d  = &data[b*22..];
        let sc = f16::from_le_bytes([d[0], d[1]]).to_f32();
        let qh = u32::from_le_bytes([d[2], d[3], d[4], d[5]]);
        for j in 0..16 {
            let lo = (d[6+j] & 0xf) as i32 | (((qh >> (j*2))   & 1) as i32 * 16) - 16;
            let hi = (d[6+j] >>  4) as i32 | (((qh >> (j*2+1)) & 1) as i32 * 16) - 16;
            o[b*32 + j]      = lo as f32 * sc;
            o[b*32 + j + 16] = hi as f32 * sc;
        }
    }
    o
}
fn dq51(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/32 {
        let d  = &data[b*24..];
        let sc = f16::from_le_bytes([d[0], d[1]]).to_f32();
        let mn = f16::from_le_bytes([d[2], d[3]]).to_f32();
        let qh = u32::from_le_bytes([d[4], d[5], d[6], d[7]]);
        for j in 0..16 {
            let lo = (d[8+j] & 0xf) as f32 + ((qh >> (j*2))   & 1) as f32 * 16.0;
            let hi = (d[8+j] >>  4) as f32 + ((qh >> (j*2+1)) & 1) as f32 * 16.0;
            o[b*32 + j]      = lo * sc + mn;
            o[b*32 + j + 16] = hi * sc + mn;
        }
    }
    o
}
fn dq80(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/32 {
        let d  = &data[b*34..];
        let sc = f16::from_le_bytes([d[0], d[1]]).to_f32();
        for i in 0..32 { o[b*32+i] = (d[2+i] as i8) as f32 * sc; }
    }
    o
}
fn dq81(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/32 {
        let d  = &data[b*36..];
        let sc = f32::from_le_bytes([d[0], d[1], d[2], d[3]]);
        for i in 0..32 { o[b*32+i] = (d[8+i] as i8) as f32 * sc; }
    }
    o
}
fn dq2k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/256 {
        let d    = &data[b*84..];
        let sc_d = &d[0..16]; let qs = &d[16..80];
        let d_f  = f16::from_le_bytes([d[80], d[81]]).to_f32();
        let dmin = f16::from_le_bytes([d[82], d[83]]).to_f32();
        for i in 0..256 {
            let sc = (sc_d[i/16] & 0xF) as f32;
            let mn = (sc_d[i/16] >>  4) as f32;
            let q  = ((qs[i/4] >> (2*(i%4))) & 3) as f32;
            o[b*256+i] = d_f * sc * q - dmin * mn;
        }
    }
    o
}
fn dq3k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/256 {
        let d      = &data[b*110..];
        let hmask  = &d[0..32]; let qs = &d[32..96]; let scales = &d[96..108];
        let d_f    = f16::from_le_bytes([d[108], d[109]]).to_f32();
        let mut sc = [0i8; 16];
        for j in 0..4 {
            sc[j]    = ((scales[j]   & 0x3F) as i8) - 32;
            sc[j+4]  = (((scales[j+4] & 0xF) | (scales[j]   >> 4) << 4) as i8) - 32;
            sc[j+8]  = ((scales[j+8] & 0x3F) as i8) - 32;
            sc[j+12] = (((scales[j+8] >> 4)  | (scales[j+4] >> 4) << 4) as i8) - 32;
        }
        for i in 0..256 {
            let hbit = (hmask[i%32] >> (i/32)) & 1;
            let q    = ((qs[i/4] >> (2*(i%4))) & 3) as i32 | (hbit as i32 * 4) - 4;
            o[b*256+i] = d_f * sc[i/16] as f32 * q as f32;
        }
    }
    o
}
fn get_scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    let (d, m) = if j < 4 {
        (q[j] & 63, q[j+4] & 63)
    } else {
        ((q[j+4] & 0xF) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4))
    };
    (d as f32, m as f32)
}
fn dq4k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/256 {
        let d      = &data[b*144..];
        let df     = f16::from_le_bytes([d[0], d[1]]).to_f32();
        let dmin   = f16::from_le_bytes([d[2], d[3]]).to_f32();
        let scales = &d[4..16]; let qs = &d[16..144];
        let (mut is, mut qoff, mut y) = (0, 0, b*256);
        for _ in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is,   scales);
            let (sc2, m2) = get_scale_min_k4(is+1, scales);
            let (d1, mb1) = (df*sc1, dmin*m1);
            let (d2, mb2) = (df*sc2, dmin*m2);
            for l in 0..32 {
                o[y+l]    = d1 * (qs[qoff+l] & 0xF) as f32 - mb1;
                o[y+l+32] = d2 * (qs[qoff+l] >>  4) as f32 - mb2;
            }
            is += 2; qoff += 32; y += 64;
        }
    }
    o
}
fn dq5k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/256 {
        let d      = &data[b*176..];
        let df     = f16::from_le_bytes([d[0], d[1]]).to_f32();
        let dmin   = f16::from_le_bytes([d[2], d[3]]).to_f32();
        let scales = &d[4..16]; let qh = &d[16..48]; let qs = &d[48..176];
        let (mut is, mut qoff, mut qhoff, mut y) = (0, 0, 0, b*256);
        for j in 0..4usize {
            let hm1 = 1u8 << (j*2); let hm2 = 1u8 << (j*2+1);
            let (sc1, m1) = get_scale_min_k4(is,   scales);
            let (sc2, m2) = get_scale_min_k4(is+1, scales);
            let (d1, mb1) = (df*sc1, dmin*m1);
            let (d2, mb2) = (df*sc2, dmin*m2);
            for l in 0..32 {
                let v1 = (qs[qoff+l] & 0xF) as f32 + if qh[qhoff+l] & hm1 != 0 { 16.0 } else { 0.0 };
                let v2 = (qs[qoff+l] >>  4) as f32 + if qh[qhoff+l] & hm2 != 0 { 16.0 } else { 0.0 };
                o[y+l]    = d1*v1 - mb1;
                o[y+l+32] = d2*v2 - mb2;
            }
            is += 2; qoff += 32; y += 64;
            if j % 2 == 1 { qhoff += 32; }
        }
    }
    o
}
fn dq6k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/256 {
        let d  = &data[b*210..];
        let ql = &d[0..128]; let qh = &d[128..192]; let sc = &d[192..208];
        let df = f16::from_le_bytes([d[208], d[209]]).to_f32();
        let (mut y, mut qlo, mut qhi, mut si) = (b*256, 0, 0, 0);
        for _ in 0..2 {
            for l in 0..32 {
                let q1 = (((ql[qlo+l]    & 0xF) as i32) | (((qh[qhi+l] >> 0) & 3) as i32 * 16)) - 32;
                let q2 = (((ql[qlo+l+32] & 0xF) as i32) | (((qh[qhi+l] >> 2) & 3) as i32 * 16)) - 32;
                let q3 = (((ql[qlo+l]    >>  4) as i32) | (((qh[qhi+l] >> 4) & 3) as i32 * 16)) - 32;
                let q4 = (((ql[qlo+l+32] >>  4) as i32) | (((qh[qhi+l] >> 6) & 3) as i32 * 16)) - 32;
                let is = l / 16;
                o[y+l]    = df * (sc[si+is]   as i8 as f32) * q1 as f32;
                o[y+l+32] = df * (sc[si+is+2] as i8 as f32) * q2 as f32;
                o[y+l+64] = df * (sc[si+is+4] as i8 as f32) * q3 as f32;
                o[y+l+96] = df * (sc[si+is+6] as i8 as f32) * q4 as f32;
            }
            y += 128; qlo += 64; qhi += 32; si += 8;
        }
    }
    o
}
fn dq8k(data: &[u8], n: usize) -> Vec<f32> {
    let mut o = vec![0f32; n];
    for b in 0..n/256 {
        let d  = &data[b*292..];
        let df = f32::from_le_bytes([d[0], d[1], d[2], d[3]]);
        for i in 0..256 { o[b*256+i] = df * (d[4+i] as i8) as f32; }
    }
    o
}

// ── QuantTensor ───────────────────────────────────────────────────────────────
// Zero-copy: holds Arc<Mmap> + byte range instead of a Vec<u8> copy.
// This keeps RAM usage equal to the compressed model size on disk.

// Helper used by pack_q4k_for_gpu
fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    let (d, m) = if j < 4 {
        (q[j] & 63, q[j+4] & 63)
    } else {
        ((q[j+4] & 0xF) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4))
    };
    (d as f32, m as f32)
}

pub struct QuantTensor {
    mmap:   Arc<Mmap>,   // shared reference to the memory-mapped file
    offset: usize,        // byte start within mmap
    len:    usize,        // byte count for this tensor
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

    /// Raw bytes for this tensor (zero-copy view into mmap)
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.mmap[self.offset..self.offset + self.len]
    }

    /// Dequantize a single row to f32 (used for embedding table lookups)
    pub fn get_row(&self, r: usize) -> Vec<f32> {
        let rb = self.typ.byte_size(self.cols);
        let d  = self.data();
        dequantize(self.typ, &d[r*rb..(r+1)*rb], self.cols).unwrap()
    }

    /// Dot product of row r with vector b — no allocation for common types
    pub fn row_dot(&self, r: usize, b: &[f32]) -> f32 {
        let rb = self.typ.byte_size(self.cols);
        let d  = self.data();
        let d  = &d[r*rb..(r+1)*rb];
        match self.typ {
            GgmlType::F32 => (0..self.cols)
                .map(|i| f32::from_le_bytes([d[i*4],d[i*4+1],d[i*4+2],d[i*4+3]]) * b[i])
                .sum(),
            GgmlType::F16 => (0..self.cols)
                .map(|i| f16::from_le_bytes([d[i*2],d[i*2+1]]).to_f32() * b[i])
                .sum(),
            // Q4_0: corrected nibble order — lower nibbles first, then upper
            GgmlType::Q4_0 => {
                (0..self.cols/32).map(|blk| {
                    let d  = &d[blk*18..];
                    let sc = f16::from_le_bytes([d[0], d[1]]).to_f32();
                    (0..16).map(|j| {
                        ((d[2+j] & 0xf) as i32 - 8) as f32 * sc * b[blk*32 + j]
                      + ((d[2+j] >>  4) as i32 - 8) as f32 * sc * b[blk*32 + j + 16]
                    }).sum::<f32>()
                }).sum()
            }
            GgmlType::Q8_0 => {
                (0..self.cols/32).map(|blk| {
                    let d  = &d[blk*34..];
                    let sc = f16::from_le_bytes([d[0], d[1]]).to_f32();
                    (0..32).map(|i| (d[2+i] as i8) as f32 * sc * b[blk*32+i]).sum::<f32>()
                }).sum()
            }
            // All other types: full dequant then dot (K-quants etc.)
            _ => {
                let v = dequantize(self.typ, d, self.cols).unwrap();
                v.iter().zip(b).map(|(x, y)| x * y).sum()
            }
        }
    }

    /// Parallel matrix-vector multiply: out[i] = row_dot(i, b)
    pub fn matvec(&self, out: &mut [f32], b: &[f32]) {
        out.par_iter_mut().enumerate().for_each(|(i, o)| *o = self.row_dot(i, b));
    }

    /// Pack Q4_0 blocks into GPU-friendly [scale_bits, nibble_u32 x4] layout.
    /// Only valid when typ == Q4_0.
    pub fn pack_q4_0_for_gpu(&self) -> Vec<u32> {
        let nb = self.rows * self.cols / 32;
        let d  = self.data();
        let mut v = Vec::with_capacity(nb * 5);
        for b in 0..nb {
            let blk = &d[b*18..b*18+18];
            let sc  = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            v.push(sc.to_bits());
            for i in 0..4 {
                v.push(u32::from_le_bytes([blk[2+i*4], blk[3+i*4], blk[4+i*4], blk[5+i*4]]));
            }
        }
        v
    }

    /// Pack Q4_K blocks for GPU (48 u32s/block).
    /// Layout: [scale*df x8, min*dmin x8, nibbles x32]
    pub fn pack_q4k_for_gpu(&self) -> Vec<u32> {
        let nb   = self.rows * self.cols / 256;
        let data = self.data();
        let mut v = Vec::with_capacity(nb * 48);
        for b in 0..nb {
            let blk    = &data[b*144..(b+1)*144];
            let df     = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            let dmin   = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
            let scales = &blk[4..16];
            let qs     = &blk[16..144];
            for i in 0..4 {
                let (s0, _) = scale_min_k4(i*2,     scales);
                let (s1, _) = scale_min_k4(i*2 + 1, scales);
                v.push((df * s0).to_bits());
                v.push((df * s1).to_bits());
            }
            for i in 0..4 {
                let (_, m0) = scale_min_k4(i*2,     scales);
                let (_, m1) = scale_min_k4(i*2 + 1, scales);
                v.push((dmin * m0).to_bits());
                v.push((dmin * m1).to_bits());
            }
            for i in 0..32 {
                v.push(u32::from_le_bytes([qs[i*4], qs[i*4+1], qs[i*4+2], qs[i*4+3]]));
            }
        }
        v
    }

    /// Pack Q6_K blocks for GPU (64 u32s/block).
    /// Layout: [scale*df x16 as f32, ql x32 u32s, qh x16 u32s]
    pub fn pack_q6k_for_gpu(&self) -> Vec<u32> {
        let nb   = self.rows * self.cols / 256;
        let data = self.data();
        let mut v = Vec::with_capacity(nb * 64);
        for b in 0..nb {
            let blk = &data[b*210..(b+1)*210];
            let ql  = &blk[0..128];
            let qh  = &blk[128..192];
            let sc  = &blk[192..208];
            let df  = half::f16::from_le_bytes([blk[208], blk[209]]).to_f32();
            for i in 0..16 {
                v.push((df * (sc[i] as i8) as f32).to_bits());
            }
            for i in 0..32 {
                v.push(u32::from_le_bytes([ql[i*4], ql[i*4+1], ql[i*4+2], ql[i*4+3]]));
            }
            for i in 0..16 {
                v.push(u32::from_le_bytes([qh[i*4], qh[i*4+1], qh[i*4+2], qh[i*4+3]]));
            }
        }
        v
    }

    /// Pack Q8_0 blocks for GPU (9 u32s/block).
    /// Layout: [d_f32_bits, i8_values x8 u32s]
    pub fn pack_q8_0_for_gpu(&self) -> Vec<u32> {
        let nb   = self.rows * self.cols / 32;
        let data = self.data();
        let mut v = Vec::with_capacity(nb * 9);
        for b in 0..nb {
            let d  = &data[b*34..(b+1)*34];
            let df = half::f16::from_le_bytes([d[0], d[1]]).to_f32();
            v.push(df.to_bits());
            for i in 0..8 {
                v.push(u32::from_le_bytes([d[2+i*4], d[3+i*4], d[4+i*4], d[5+i*4]]));
            }
        }
        v
    }
}
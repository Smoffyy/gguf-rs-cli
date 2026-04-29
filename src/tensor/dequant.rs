use half::f16;
use crate::gguf::types::GgmlType;

// ── Block-level dot products (no intermediate Vec allocation) ───────────────

pub fn dot_q4_0(data: &[u8], b: &[f32], n: usize) -> f32 {
    let mut sum = 0f32;
    for blk in 0..n/32 {
        let d = &data[blk*18..];
        let scale = f16::from_le_bytes([d[0],d[1]]).to_f32();
        for i in 0..16 {
            sum += ((d[2+i]&0xf) as i32-8) as f32 * scale * b[blk*32+i*2];
            sum += ((d[2+i]>>4)  as i32-8) as f32 * scale * b[blk*32+i*2+1];
        }
    }
    sum
}

pub fn dot_q8_0(data: &[u8], b: &[f32], n: usize) -> f32 {
    let mut sum = 0f32;
    for blk in 0..n/32 {
        let d = &data[blk*34..];
        let scale = f16::from_le_bytes([d[0],d[1]]).to_f32();
        for i in 0..32 { sum += (d[2+i] as i8) as f32 * scale * b[blk*32+i]; }
    }
    sum
}

pub fn dot_f16(data: &[u8], b: &[f32], n: usize) -> f32 {
    (0..n).map(|i| f16::from_le_bytes([data[i*2],data[i*2+1]]).to_f32()*b[i]).sum()
}

pub fn dot_f32_raw(data: &[u8], b: &[f32], n: usize) -> f32 {
    (0..n).map(|i| f32::from_le_bytes([data[i*4],data[i*4+1],data[i*4+2],data[i*4+3]])*b[i]).sum()
}

// ── Full dequantization (used for embedding rows) ───────────────────────────

pub fn dequantize(typ: GgmlType, data: &[u8], n: usize) -> anyhow::Result<Vec<f32>> {
    Ok(match typ {
        GgmlType::F32 => (0..n).map(|i|f32::from_le_bytes([data[i*4],data[i*4+1],data[i*4+2],data[i*4+3]])).collect(),
        GgmlType::F16 => (0..n).map(|i|f16::from_le_bytes([data[i*2],data[i*2+1]]).to_f32()).collect(),
        GgmlType::Q4_0 => {
            let mut out=vec![0f32;n];
            for blk in 0..n/32 {
                let d=&data[blk*18..];
                let sc=f16::from_le_bytes([d[0],d[1]]).to_f32();
                for i in 0..16 {
                    out[blk*32+i*2]   = ((d[2+i]&0xf) as i32-8) as f32*sc;
                    out[blk*32+i*2+1] = ((d[2+i]>>4)  as i32-8) as f32*sc;
                }
            }
            out
        }
        GgmlType::Q8_0 => {
            let mut out=vec![0f32;n];
            for blk in 0..n/32 {
                let d=&data[blk*34..];
                let sc=f16::from_le_bytes([d[0],d[1]]).to_f32();
                for i in 0..32 { out[blk*32+i]=(d[2+i] as i8) as f32*sc; }
            }
            out
        }
        _ => anyhow::bail!("Unsupported quant for dequant: {:?}. Use F16/F32/Q4_0/Q8_0.", typ),
    })
}

// ── QuantTensor: lazy row-by-row dequantization (keeps weights compressed) ──

pub struct QuantTensor {
    pub data: Vec<u8>,
    pub typ:  GgmlType,
    pub rows: usize,
    pub cols: usize,
}

impl QuantTensor {
    pub fn load(data: Vec<u8>, typ: GgmlType, dims: &[u64]) -> Self {
        // GGUF dims are innermost-first: dims[0]=cols, dims[1]=rows
        let cols = dims[0] as usize;
        let rows = if dims.len() > 1 { dims[1] as usize } else { 1 };
        Self { data, typ, rows, cols }
    }

    // Dot product of row `r` with vector `b` — no allocation
    pub fn row_dot(&self, r: usize, b: &[f32]) -> f32 {
        let rb = self.typ.byte_size(self.cols);
        let d  = &self.data[r*rb..(r+1)*rb];
        match self.typ {
            GgmlType::F32  => dot_f32_raw(d, b, self.cols),
            GgmlType::F16  => dot_f16(d, b, self.cols),
            GgmlType::Q4_0 => dot_q4_0(d, b, self.cols),
            GgmlType::Q8_0 => dot_q8_0(d, b, self.cols),
            _ => { let v=dequantize(self.typ,d,self.cols).unwrap(); v.iter().zip(b).map(|(x,y)|x*y).sum() }
        }
    }

    // Dequantize a full row (for embedding table lookups)
    pub fn get_row(&self, r: usize) -> Vec<f32> {
        let rb = self.typ.byte_size(self.cols);
        dequantize(self.typ, &self.data[r*rb..(r+1)*rb], self.cols).unwrap()
    }

    // Parallel matrix-vector multiply: out = self @ b
    pub fn matvec(&self, out: &mut [f32], b: &[f32]) {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(i,o)| *o = self.row_dot(i,b));
    }

    // Pack Q4_0 data for GPU upload (5 u32s per block)
    pub fn pack_q4_0_for_gpu(&self) -> Vec<u32> {
        let n_blocks = self.rows * self.cols / 32;
        let mut out = Vec::with_capacity(n_blocks * 5);
        for b in 0..n_blocks {
            let d = &self.data[b*18..b*18+18];
            let scale = half::f16::from_le_bytes([d[0],d[1]]).to_f32();
            out.push(scale.to_bits());
            for i in 0..4 { out.push(u32::from_le_bytes([d[2+i*4],d[3+i*4],d[4+i*4],d[5+i*4]])); }
        }
        out
    }
}
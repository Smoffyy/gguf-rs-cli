use half::f16;
use crate::gguf::types::GgmlType;

// Q4_0: 18 bytes/block, 32 weights
// Layout: [f16 scale | 16 bytes of packed nibbles]
// weight = (nibble - 8) * scale
pub fn dequant_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    let n_blocks = n / 32;
    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let blk = &data[b * 18..];
        let scale = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for i in 0..16 {
            let byte = blk[2 + i];
            out[b * 32 + i * 2]     = ((byte & 0x0f) as i32 - 8) as f32 * scale;
            out[b * 32 + i * 2 + 1] = ((byte >> 4)   as i32 - 8) as f32 * scale;
        }
    }
    out
}

// Q8_0: 34 bytes/block, 32 weights
// Layout: [f16 scale | 32 x i8 values]
// weight = q * scale
pub fn dequant_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let n_blocks = n / 32;
    let mut out = vec![0f32; n];
    for b in 0..n_blocks {
        let blk = &data[b * 34..];
        let scale = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        for i in 0..32 {
            out[b * 32 + i] = (blk[2 + i] as i8) as f32 * scale;
        }
    }
    out
}

pub fn dequant_f16(data: &[u8], n: usize) -> Vec<f32> {
    (0..n).map(|i| f16::from_le_bytes([data[i*2], data[i*2+1]]).to_f32()).collect()
}

pub fn dequant_f32(data: &[u8], n: usize) -> Vec<f32> {
    (0..n).map(|i| f32::from_le_bytes([data[i*4], data[i*4+1], data[i*4+2], data[i*4+3]])).collect()
}

pub fn dequantize(typ: GgmlType, data: &[u8], n: usize) -> anyhow::Result<Vec<f32>> {
    Ok(match typ {
        GgmlType::F32  => dequant_f32(data, n),
        GgmlType::F16  => dequant_f16(data, n),
        GgmlType::Q4_0 => dequant_q4_0(data, n),
        GgmlType::Q8_0 => dequant_q8_0(data, n),
        _ => anyhow::bail!("Unsupported quant type for dequantization: {:?}. Use F16/F32/Q4_0/Q8_0 models.", typ),
    })
}

use std::io::{Read, Seek};
use anyhow::{bail, Context};
use super::types::*;

fn read_u8(r: &mut impl Read) -> anyhow::Result<u8> {
    let mut b = [0u8; 1]; r.read_exact(&mut b)?; Ok(b[0])
}
fn read_u16(r: &mut impl Read) -> anyhow::Result<u16> {
    let mut b = [0u8; 2]; r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b))
}
fn read_u32(r: &mut impl Read) -> anyhow::Result<u32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b))
}
fn read_i32(r: &mut impl Read) -> anyhow::Result<i32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(i32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> anyhow::Result<u64> {
    let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b))
}
fn read_i64(r: &mut impl Read) -> anyhow::Result<i64> {
    let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(i64::from_le_bytes(b))
}
fn read_f32(r: &mut impl Read) -> anyhow::Result<f32> {
    let mut b = [0u8; 4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b))
}
fn read_f64(r: &mut impl Read) -> anyhow::Result<f64> {
    let mut b = [0u8; 8]; r.read_exact(&mut b)?; Ok(f64::from_le_bytes(b))
}

fn read_string(r: &mut impl Read) -> anyhow::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_value(r: &mut impl Read, type_id: u32) -> anyhow::Result<GgufValue> {
    Ok(match type_id {
        0  => GgufValue::U8(read_u8(r)?),
        1  => GgufValue::I8(read_u8(r)? as i8),
        2  => GgufValue::U16(read_u16(r)?),
        3  => GgufValue::I16(read_u16(r)? as i16),
        4  => GgufValue::U32(read_u32(r)?),
        5  => GgufValue::I32(read_i32(r)?),
        6  => GgufValue::F32(read_f32(r)?),
        7  => GgufValue::Bool(read_u8(r)? != 0),
        8  => GgufValue::String(read_string(r)?),
        9  => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(count);
            for _ in 0..count { arr.push(read_value(r, elem_type)?); }
            GgufValue::Array(arr)
        },
        10 => GgufValue::U64(read_u64(r)?),
        11 => GgufValue::I64(read_i64(r)?),
        12 => GgufValue::F64(read_f64(r)?),
        _  => bail!("Unknown GGUF value type: {}", type_id),
    })
}

pub fn parse<R: Read + Seek>(mut r: R) -> anyhow::Result<GgufFile> {
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != b"GGUF" { bail!("Not a GGUF file"); }

    let version = read_u32(&mut r)?;
    if version < 2 || version > 3 { bail!("Unsupported GGUF version: {}", version); }

    let n_tensors = read_u64(&mut r)? as usize;
    let n_kv      = read_u64(&mut r)? as usize;

    // Read all key-value metadata pairs
    let mut metadata = std::collections::HashMap::new();
    for _ in 0..n_kv {
        let key     = read_string(&mut r)?;
        let type_id = read_u32(&mut r)?;
        let value   = read_value(&mut r, type_id)
            .with_context(|| format!("Reading key '{}'", key))?;
        metadata.insert(key, value);
    }

    // Read tensor descriptors
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name   = read_string(&mut r)?;
        let n_dims = read_u32(&mut r)? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims { dims.push(read_u64(&mut r)?); }
        let typ    = GgmlType::from_u32(read_u32(&mut r)?)?;
        let offset = read_u64(&mut r)?;
        tensors.push(TensorInfo { name, dims, typ, offset });
    }

    // Align to general.alignment (default 32)
    let alignment = metadata.get("general.alignment")
        .and_then(|v| v.as_u32())
        .unwrap_or(32) as u64;
    let pos = r.stream_position()?;
    let data_offset = (pos + alignment - 1) / alignment * alignment;

    Ok(GgufFile { metadata, tensors, data_offset })
}

use std::io::{Read, Seek};
use anyhow::{bail, Context};
use super::types::*;

fn ru8(r: &mut impl Read) -> anyhow::Result<u8>  { let mut b=[0u8;1]; r.read_exact(&mut b)?; Ok(b[0]) }
fn ru16(r:&mut impl Read)->anyhow::Result<u16>   { let mut b=[0u8;2]; r.read_exact(&mut b)?; Ok(u16::from_le_bytes(b)) }
fn ru32(r:&mut impl Read)->anyhow::Result<u32>   { let mut b=[0u8;4]; r.read_exact(&mut b)?; Ok(u32::from_le_bytes(b)) }
fn ri32(r:&mut impl Read)->anyhow::Result<i32>   { let mut b=[0u8;4]; r.read_exact(&mut b)?; Ok(i32::from_le_bytes(b)) }
fn ru64(r:&mut impl Read)->anyhow::Result<u64>   { let mut b=[0u8;8]; r.read_exact(&mut b)?; Ok(u64::from_le_bytes(b)) }
fn ri64(r:&mut impl Read)->anyhow::Result<i64>   { let mut b=[0u8;8]; r.read_exact(&mut b)?; Ok(i64::from_le_bytes(b)) }
fn rf32(r:&mut impl Read)->anyhow::Result<f32>   { let mut b=[0u8;4]; r.read_exact(&mut b)?; Ok(f32::from_le_bytes(b)) }
fn rf64(r:&mut impl Read)->anyhow::Result<f64>   { let mut b=[0u8;8]; r.read_exact(&mut b)?; Ok(f64::from_le_bytes(b)) }

fn rstr(r: &mut impl Read) -> anyhow::Result<String> {
    let len = ru64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn rval(r: &mut impl Read, tid: u32) -> anyhow::Result<GgufValue> {
    Ok(match tid {
        0  => GgufValue::U8(ru8(r)?),
        1  => GgufValue::I8(ru8(r)? as i8),
        2  => GgufValue::U16(ru16(r)?),
        3  => GgufValue::I16(ru16(r)? as i16),
        4  => GgufValue::U32(ru32(r)?),
        5  => GgufValue::I32(ri32(r)?),
        6  => GgufValue::F32(rf32(r)?),
        7  => GgufValue::Bool(ru8(r)? != 0),
        8  => GgufValue::String(rstr(r)?),
        9  => {
            let et = ru32(r)?; let n = ru64(r)? as usize;
            let mut arr = Vec::with_capacity(n);
            for _ in 0..n { arr.push(rval(r, et)?); }
            GgufValue::Array(arr)
        }
        10 => GgufValue::U64(ru64(r)?),
        11 => GgufValue::I64(ri64(r)?),
        12 => GgufValue::F64(rf64(r)?),
        _  => bail!("Unknown GGUF value type: {}", tid),
    })
}

pub fn parse<R: Read+Seek>(mut r: R) -> anyhow::Result<GgufFile> {
    let mut magic=[0u8;4]; r.read_exact(&mut magic)?;
    if &magic != b"GGUF" { bail!("Not a GGUF file"); }
    let ver = ru32(&mut r)?;
    if ver<2||ver>3 { bail!("Unsupported GGUF version: {}", ver); }

    let n_tensors = ru64(&mut r)? as usize;
    let n_kv      = ru64(&mut r)? as usize;

    let mut metadata = std::collections::HashMap::new();
    for _ in 0..n_kv {
        let key = rstr(&mut r)?;
        let tid = ru32(&mut r)?;
        let val = rval(&mut r, tid).with_context(||format!("key '{}'",key))?;
        metadata.insert(key, val);
    }

    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name   = rstr(&mut r)?;
        let ndims  = ru32(&mut r)? as usize;
        let mut dims = Vec::with_capacity(ndims);
        for _ in 0..ndims { dims.push(ru64(&mut r)?); }
        let typ    = GgmlType::from_u32(ru32(&mut r)?)?;
        let offset = ru64(&mut r)?;
        tensors.push(TensorInfo { name, dims, typ, offset });
    }

    let alignment = metadata.get("general.alignment")
        .and_then(|v|v.as_u32()).unwrap_or(32) as u64;
    let pos = r.stream_position()?;
    let data_offset = (pos+alignment-1)/alignment*alignment;

    Ok(GgufFile { metadata, tensors, data_offset })
}
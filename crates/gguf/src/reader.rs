use std::collections::HashMap;
use std::io::Read;

use gguf_core::{DType, Error, Result};

use crate::value::Value;
use crate::TensorInfo;

pub struct Parsed {
    pub metadata: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub version: u32,
}

/// Counts bytes consumed so the tensor-data alignment can be computed without a `Seek`
/// bound, which lets this run over any reader.
struct Counting<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Counting<R> {
    fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }

    fn exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.inner.read_exact(buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.exact(&mut b)?;
        Ok(b[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        if n > 1 << 30 {
            return Err(Error::Format(format!("string length {n} is implausible")));
        }
        let mut b = vec![0u8; n];
        self.exact(&mut b)?;
        Ok(String::from_utf8_lossy(&b).into_owned())
    }
}

const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

fn read_value<R: Read>(r: &mut Counting<R>, type_id: u32, depth: u32) -> Result<Value> {
    if depth > 4 {
        return Err(Error::Format("metadata array nested too deeply".into()));
    }
    Ok(match type_id {
        T_U8 => Value::U8(r.u8()?),
        T_I8 => Value::I8(r.u8()? as i8),
        T_U16 => Value::U16(r.u16()?),
        T_I16 => Value::I16(r.u16()? as i16),
        T_U32 => Value::U32(r.u32()?),
        T_I32 => Value::I32(r.u32()? as i32),
        T_F32 => Value::F32(r.f32()?),
        T_BOOL => Value::Bool(r.u8()? != 0),
        T_STRING => Value::String(r.string()?),
        T_U64 => Value::U64(r.u64()?),
        T_I64 => Value::I64(r.u64()? as i64),
        T_F64 => Value::F64(r.f64()?),
        T_ARRAY => {
            let elem = r.u32()?;
            let n = r.u64()? as usize;
            if n > 1 << 28 {
                return Err(Error::Format(format!("array length {n} is implausible")));
            }
            let mut out = Vec::with_capacity(n.min(1 << 20));
            for _ in 0..n {
                out.push(read_value(r, elem, depth + 1)?);
            }
            Value::Array(out)
        }
        other => return Err(Error::Format(format!("unknown metadata value type {other}"))),
    })
}

pub fn parse<R: Read>(reader: R) -> Result<Parsed> {
    let mut r = Counting::new(reader);

    let mut magic = [0u8; 4];
    r.exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(Error::Format(format!(
            "not a GGUF file (magic {:02x?}; a .bin or .pth checkpoint needs converting first)",
            magic
        )));
    }

    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(Error::Format(format!(
            "GGUF version {version} is not supported (this engine reads v2 and v3)"
        )));
    }

    let n_tensors = r.u64()? as usize;
    let n_kv = r.u64()? as usize;

    let mut metadata = HashMap::with_capacity(n_kv);
    for _ in 0..n_kv {
        let key = r.string()?;
        let type_id = r.u32()?;
        let value = read_value(&mut r, type_id, 0)
            .map_err(|e| Error::Format(format!("metadata key {key:?}: {e}")))?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        if n_dims > 4 {
            return Err(Error::Format(format!(
                "tensor {name} declares {n_dims} dimensions; ggml allows at most 4"
            )));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let dtype = DType::from_u32(r.u32()?)
            .map_err(|e| Error::Format(format!("tensor {name}: {e}")))?;
        let offset = r.u64()?;
        tensors.push(TensorInfo { name, dims, dtype, offset });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(Value::as_u64)
        .unwrap_or(32)
        .max(1);
    let data_offset = r.pos.div_ceil(alignment) * alignment;

    Ok(Parsed { metadata, tensors, data_offset, version })
}

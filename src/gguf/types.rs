#![allow(dead_code)]
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8), I8(i8), U16(u16), I16(i16),
    U32(u32), I32(i32), F32(f32),
    U64(u64), I64(i64), F64(f64),
    Bool(bool), String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U8(v)  => Some(*v as u32),
            Self::U16(v) => Some(*v as u32),
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            Self::U64(v) => Some(*v as u32),
            Self::I64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),  // CRITICAL: Qwen stores rope_freq_base as F64
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(v) = self { Some(v) } else { None }
    }
    pub fn as_arr(&self) -> Option<&[GgufValue]> {
        if let Self::Array(v) = self { Some(v) } else { None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32=0, F16=1, Q4_0=2, Q4_1=3,
    Q5_0=6, Q5_1=7, Q8_0=8, Q8_1=9,
    Q2K=10, Q3K=11, Q4K=12, Q5K=13, Q6K=14, Q8K=15,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> anyhow::Result<Self> {
        Ok(match v {
            0=>Self::F32, 1=>Self::F16,
            2=>Self::Q4_0, 3=>Self::Q4_1,
            6=>Self::Q5_0, 7=>Self::Q5_1,
            8=>Self::Q8_0, 9=>Self::Q8_1,
            10=>Self::Q2K, 11=>Self::Q3K, 12=>Self::Q4K,
            13=>Self::Q5K, 14=>Self::Q6K, 15=>Self::Q8K,
            _ => anyhow::bail!("Unknown ggml type: {}", v),
        })
    }
    pub fn byte_size(&self, n: usize) -> usize {
        match self {
            Self::F32  => n*4,
            Self::F16  => n*2,
            Self::Q4_0 => n/32*18,
            Self::Q4_1 => n/32*20,
            Self::Q8_0 => n/32*34,
            Self::Q8_1 => n/32*36,
            Self::Q2K  => n/256*84,
            Self::Q3K  => n/256*110,
            Self::Q4K  => n/256*144,
            Self::Q5K  => n/256*176,
            Self::Q6K  => n/256*210,
            Self::Q8K  => n/256*292,
            Self::Q5_0 => n/32*22,
            Self::Q5_1 => n/32*24,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub typ: GgmlType,
    pub offset: u64,
}
impl TensorInfo {
    pub fn n_elements(&self) -> usize { self.dims.iter().product::<u64>() as usize }
    pub fn byte_size(&self) -> usize  { self.typ.byte_size(self.n_elements()) }
}

pub struct GgufFile {
    pub metadata: HashMap<String, GgufValue>,
    pub tensors:  Vec<TensorInfo>,
    pub data_offset: u64,
}

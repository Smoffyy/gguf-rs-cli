use std::path::Path;
use memmap2::Mmap;
use crate::gguf::types::TensorInfo;
use super::dequant::dequantize;

pub struct TensorStorage {
    mmap: Mmap,
    pub data_offset: u64,
}

impl TensorStorage {
    pub fn new(path: &Path, data_offset: u64) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        // Safety: we treat the mmap as read-only; file is not modified during lifetime
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap, data_offset })
    }

    pub fn get_bytes(&self, info: &TensorInfo) -> &[u8] {
        let start = (self.data_offset + info.offset) as usize;
        &self.mmap[start..start + info.byte_size()]
    }

    pub fn load_f32(&self, info: &TensorInfo) -> anyhow::Result<Vec<f32>> {
        dequantize(info.typ, self.get_bytes(info), info.n_elements())
    }
}

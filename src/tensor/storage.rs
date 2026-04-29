use std::path::Path;
use memmap2::Mmap;
use crate::gguf::types::TensorInfo;

pub struct TensorStorage { mmap: Mmap, pub data_offset: u64 }
impl TensorStorage {
    pub fn new(path: &Path, data_offset: u64) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap, data_offset })
    }
    pub fn get_bytes(&self, info: &TensorInfo) -> &[u8] {
        let s = (self.data_offset + info.offset) as usize;
        &self.mmap[s..s+info.byte_size()]
    }
}
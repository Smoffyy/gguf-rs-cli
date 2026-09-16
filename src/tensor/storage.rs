use std::path::Path;
use std::sync::Arc;
use memmap2::Mmap;
use crate::gguf::types::TensorInfo;

pub struct TensorStorage {
    pub mmap: Arc<Mmap>,
    pub data_offset: u64,
}

impl TensorStorage {
    pub fn new(path: &Path, data_offset: u64) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap: Arc::new(mmap), data_offset })
    }

    pub fn tensor_offset(&self, info: &TensorInfo) -> usize {
        (self.data_offset + info.offset) as usize
    }
}

use std::path::Path;
use std::sync::Arc;
use memmap2::Mmap;
use crate::gguf::types::TensorInfo;

/// Memory-maps the entire GGUF file.
/// We expose Arc<Mmap> so QuantTensor can hold a zero-copy reference
/// into the file without any heap allocation or data copying.
pub struct TensorStorage {
    pub mmap: Arc<Mmap>,
    pub data_offset: u64,
}

impl TensorStorage {
    pub fn new(path: &Path, data_offset: u64) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        // Safety: we never mutate the file while it is mapped
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { mmap: Arc::new(mmap), data_offset })
    }

    /// Byte offset of a tensor's data within the mmap
    pub fn tensor_offset(&self, info: &TensorInfo) -> usize {
        (self.data_offset + info.offset) as usize
    }
}
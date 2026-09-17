//! GGUF container parsing.
//!
//! The file is memory-mapped and never copied: [`GgufModel::view`] hands out borrowed,
//! still-quantized byte ranges that backends upload verbatim.

mod reader;
mod value;

pub use value::{Metadata, Value};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gguf_core::{DType, Error, QuantView, Result};
use memmap2::Mmap;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml dimension order: `dims[0]` is the fastest-varying axis, i.e. the row length.
    pub dims: Vec<u64>,
    pub dtype: DType,
    /// Offset from the start of the tensor data section.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    pub fn byte_size(&self) -> usize {
        self.dtype.row_bytes(self.n_elements())
    }

    /// Row length, i.e. the input dimension of a weight matrix.
    pub fn cols(&self) -> usize {
        self.dims.first().copied().unwrap_or(1) as usize
    }

    /// Total rows across every trailing axis, so a stacked `[n_expert, rows, cols]` MoE
    /// tensor reports `n_expert * rows`.
    pub fn rows(&self) -> usize {
        self.dims.iter().skip(1).product::<u64>().max(1) as usize
    }

    /// Number of experts for a 3-D stacked tensor, else 1.
    pub fn n_expert(&self) -> usize {
        self.dims.get(2).copied().unwrap_or(1) as usize
    }

    pub fn shape_string(&self) -> String {
        self.dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(" x ")
    }
}

pub struct GgufModel {
    pub path: PathBuf,
    pub meta: Metadata,
    pub tensors: HashMap<String, TensorInfo>,
    /// File order, which is also roughly load order.
    pub order: Vec<String>,
    pub version: u32,
    data_offset: u64,
    mmap: Arc<Mmap>,
}

impl GgufModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let parsed = reader::parse(std::io::BufReader::new(&file))?;

        // SAFETY: the mapping is read-only and lives as long as the Arc. A concurrent
        // truncation of the file by another process would be unsound, which is the same
        // caveat every mmap-based loader carries.
        let mmap = unsafe { Mmap::map(&file)? };
        let file_len = mmap.len() as u64;

        let mut tensors = HashMap::with_capacity(parsed.tensors.len());
        let mut order = Vec::with_capacity(parsed.tensors.len());
        for t in parsed.tensors {
            let end = parsed.data_offset + t.offset + t.byte_size() as u64;
            if end > file_len {
                return Err(Error::Format(format!(
                    "tensor {} runs past end of file ({} > {})",
                    t.name, end, file_len
                )));
            }
            if t.cols() % t.dtype.block_size() != 0 {
                return Err(Error::Format(format!(
                    "tensor {} has row length {} which is not a multiple of the {} block size {}",
                    t.name,
                    t.cols(),
                    t.dtype.name(),
                    t.dtype.block_size()
                )));
            }
            order.push(t.name.clone());
            tensors.insert(t.name.clone(), t);
        }

        Ok(Self {
            path: path.to_path_buf(),
            meta: Metadata::new(parsed.metadata),
            tensors,
            order,
            version: parsed.version,
            data_offset: parsed.data_offset,
            mmap: Arc::new(mmap),
        })
    }

    /// A shared owner of the mapping, for backends that borrow weight bytes rather than
    /// copying them. Handing this to `Backend::retain` is what makes those borrows sound.
    pub fn keepalive(&self) -> Arc<dyn std::any::Any + Send + Sync> {
        self.mmap.clone()
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn view_opt(&self, name: &str) -> Option<QuantView<'_>> {
        let t = self.tensors.get(name)?;
        let start = (self.data_offset + t.offset) as usize;
        Some(QuantView::new(
            &self.mmap[start..start + t.byte_size()],
            t.dtype,
            t.rows(),
            t.cols(),
        ))
    }

    pub fn view(&self, name: &str) -> Result<QuantView<'_>> {
        self.view_opt(name)
            .ok_or_else(|| Error::MissingTensor(format!("{name}{}", self.near_miss(name))))
    }

    /// First of several candidate names that is present, with the name that matched.
    pub fn view_any<'a, 'n>(&'a self, names: &[&'n str]) -> Option<(&'n str, QuantView<'a>)> {
        names.iter().find_map(|n| self.view_opt(n).map(|v| (*n, v)))
    }

    pub fn view_any_req<'a, 'n>(&'a self, names: &[&'n str]) -> Result<(&'n str, QuantView<'a>)> {
        self.view_any(names).ok_or_else(|| {
            Error::MissingTensor(format!(
                "none of [{}]{}",
                names.join(", "),
                self.near_miss(names[0])
            ))
        })
    }

    pub fn contains_any(&self, names: &[&str]) -> bool {
        names.iter().any(|n| self.contains(n))
    }

    /// Total size of all tensor data.
    pub fn weights_bytes(&self) -> u64 {
        self.tensors.values().map(|t| t.byte_size() as u64).sum()
    }

    /// The quantization that dominates the file, used for display.
    pub fn dominant_dtype(&self) -> DType {
        let mut by_type: HashMap<DType, u64> = HashMap::new();
        for t in self.tensors.values() {
            *by_type.entry(t.dtype).or_default() += t.byte_size() as u64;
        }
        by_type
            .into_iter()
            .max_by_key(|(_, b)| *b)
            .map(|(t, _)| t)
            .unwrap_or(DType::F32)
    }

    /// Distinct dtypes present, so the loader can reject a file up front if it contains a
    /// quantization we refuse to decode rather than failing partway through the load.
    pub fn dtypes(&self) -> Vec<DType> {
        let mut v: Vec<DType> = self.tensors.values().map(|t| t.dtype).collect();
        v.sort_unstable_by_key(|d| *d as u32);
        v.dedup();
        v
    }

    /// Tensor names belonging to block `i`, sorted. Used by architecture inference.
    pub fn block_tensors(&self, i: usize) -> Vec<&str> {
        let prefix = format!("blk.{i}.");
        let mut v: Vec<&str> = self
            .tensors
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .map(String::as_str)
            .collect();
        v.sort_unstable();
        v
    }

    /// Suffix of block-0 tensor names, e.g. `attn_q.weight`. The shape of this set is how
    /// the model layer infers capabilities without a per-architecture table.
    pub fn block0_suffixes(&self) -> Vec<String> {
        self.block_tensors(0)
            .iter()
            .filter_map(|n| n.strip_prefix("blk.0.").map(str::to_string))
            .collect()
    }

    fn near_miss(&self, name: &str) -> String {
        let stem: String = name.split('.').take(2).collect::<Vec<_>>().join(".");
        let mut close: Vec<&str> = self
            .tensors
            .keys()
            .filter(|k| k.starts_with(&stem))
            .map(String::as_str)
            .take(8)
            .collect();
        if close.is_empty() {
            return String::new();
        }
        close.sort_unstable();
        format!("\n  present with that prefix: {}", close.join(", "))
    }
}

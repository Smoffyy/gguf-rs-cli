use gguf_core::{Backend, BufferId, DType, KvView, MemKind, Result};

/// Key/value cache for every layer and sequence.
///
/// One buffer per layer holds all sequences side by side; a sequence's region starts at
/// `seq * n_ctx` token slots. Keeping sequences in one allocation rather than one per
/// sequence means a single buffer per layer regardless of parallelism, which matters
/// because GPU allocations are a scarcer resource than GPU bytes.
pub struct KvCache {
    k: Vec<BufferId>,
    v: Vec<BufferId>,
    stride: u32,
    n_ctx: u32,
    n_seq: u32,
    dtype: DType,
}

impl KvCache {
    pub fn new(
        be: &mut dyn Backend,
        n_layers: usize,
        n_ctx: usize,
        n_seq: usize,
        stride: usize,
        dtype: DType,
    ) -> Result<Self> {
        let per_layer = (n_ctx * n_seq * stride * dtype.type_size()) as u64;
        let mut k = Vec::with_capacity(n_layers);
        let mut v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k.push(be.alloc(per_layer, MemKind::Device)?);
            v.push(be.alloc(per_layer, MemKind::Device)?);
        }
        Ok(Self {
            k,
            v,
            stride: stride as u32,
            n_ctx: n_ctx as u32,
            n_seq: n_seq as u32,
            dtype,
        })
    }

    pub fn n_seq(&self) -> usize {
        self.n_seq as usize
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx as usize
    }

    /// Total bytes across every layer.
    pub fn bytes(&self) -> u64 {
        self.k.len() as u64
            * 2
            * (self.n_ctx as u64 * self.n_seq as u64 * self.stride as u64
                * self.dtype.type_size() as u64)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn views(&self, seq: usize) -> Vec<KvView> {
        let base = seq as u32 * self.n_ctx;
        self.k
            .iter()
            .zip(&self.v)
            .map(|(k, v)| KvView {
                k: *k,
                v: *v,
                stride: self.stride,
                base,
                dtype: self.dtype,
            })
            .collect()
    }

    pub fn free(&mut self, be: &mut dyn Backend) {
        for id in self.k.drain(..).chain(self.v.drain(..)) {
            be.free(id);
        }
    }
}

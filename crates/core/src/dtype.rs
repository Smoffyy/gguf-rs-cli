use crate::error::{Error, Result};

/// A ggml tensor element type. Discriminants match `enum ggml_type` exactly, because
/// they are what the GGUF file stores on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    BF16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
}

impl DType {
    pub fn from_u32(v: u32) -> Result<Self> {
        use DType::*;
        Ok(match v {
            0 => F32, 1 => F16, 2 => Q4_0, 3 => Q4_1,
            6 => Q5_0, 7 => Q5_1, 8 => Q8_0, 9 => Q8_1,
            10 => Q2K, 11 => Q3K, 12 => Q4K, 13 => Q5K, 14 => Q6K, 15 => Q8K,
            16 => Iq2Xxs, 17 => Iq2Xs, 18 => Iq3Xxs, 19 => Iq1S, 20 => Iq4Nl,
            21 => Iq3S, 22 => Iq2S, 23 => Iq4Xs,
            24 => I8, 25 => I16, 26 => I32, 27 => I64, 28 => F64,
            29 => Iq1M, 30 => BF16, 34 => Tq1_0, 35 => Tq2_0, 39 => Mxfp4,
            4 | 5 => return Err(Error::Unsupported(
                "ggml type Q4_2/Q4_3 (removed from ggml years ago; requantize the model)".into())),
            31..=33 | 36..=38 => return Err(Error::Unsupported(format!(
                "ggml type {v} (repacked AArch64 variant, removed from ggml; requantize the model)"))),
            _ => return Err(Error::Format(format!("unknown ggml type id {v}"))),
        })
    }

    /// Number of elements encoded by one block of this type.
    pub const fn block_size(self) -> usize {
        use DType::*;
        match self {
            F32 | F16 | BF16 | F64 | I8 | I16 | I32 | I64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 | Iq4Nl | Mxfp4 => 32,
            _ => 256,
        }
    }

    /// Size in bytes of one block of this type.
    pub const fn type_size(self) -> usize {
        use DType::*;
        match self {
            F32 | I32 => 4,
            F16 | BF16 | I16 => 2,
            F64 | I64 => 8,
            I8 => 1,
            Q4_0 => 18,
            Q4_1 => 20,
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34,
            Q8_1 => 36,
            Q2K => 84,
            Q3K => 110,
            Q4K => 144,
            Q5K => 176,
            Q6K => 210,
            Q8K => 292,
            Iq2Xxs => 66,
            Iq2Xs => 74,
            Iq2S => 82,
            Iq3Xxs => 98,
            Iq3S => 110,
            Iq1S => 50,
            Iq1M => 56,
            Iq4Nl => 18,
            Iq4Xs => 136,
            Tq1_0 => 54,
            Tq2_0 => 66,
            Mxfp4 => 17,
        }
    }

    /// Bytes occupied by `n` contiguous elements. `n` must be a multiple of `block_size()`.
    pub const fn row_bytes(self, n: usize) -> usize {
        n / self.block_size() * self.type_size()
    }

    pub const fn is_quantized(self) -> bool {
        !matches!(
            self,
            DType::F32 | DType::F16 | DType::BF16 | DType::F64
                | DType::I8 | DType::I16 | DType::I32 | DType::I64
        )
    }

    /// K-quant and IQ families carry per-super-block scales and need 256-element rows.
    pub const fn is_k_quant(self) -> bool {
        self.block_size() == 256
    }

    /// Types whose decode depends on a large exact codebook table shipped inside ggml.
    /// We refuse these rather than emit plausible-looking garbage.
    pub const fn needs_codebook(self) -> bool {
        matches!(
            self,
            DType::Iq1S | DType::Iq1M | DType::Iq2Xxs | DType::Iq2Xs
                | DType::Iq2S | DType::Iq3Xxs | DType::Iq3S
        )
    }

    pub const fn name(self) -> &'static str {
        use DType::*;
        match self {
            F32 => "F32", F16 => "F16", BF16 => "BF16", F64 => "F64",
            I8 => "I8", I16 => "I16", I32 => "I32", I64 => "I64",
            Q4_0 => "Q4_0", Q4_1 => "Q4_1", Q5_0 => "Q5_0", Q5_1 => "Q5_1",
            Q8_0 => "Q8_0", Q8_1 => "Q8_1",
            Q2K => "Q2_K", Q3K => "Q3_K", Q4K => "Q4_K",
            Q5K => "Q5_K", Q6K => "Q6_K", Q8K => "Q8_K",
            Iq2Xxs => "IQ2_XXS", Iq2Xs => "IQ2_XS", Iq2S => "IQ2_S",
            Iq3Xxs => "IQ3_XXS", Iq3S => "IQ3_S", Iq1S => "IQ1_S", Iq1M => "IQ1_M",
            Iq4Nl => "IQ4_NL", Iq4Xs => "IQ4_XS",
            Tq1_0 => "TQ1_0", Tq2_0 => "TQ2_0", Mxfp4 => "MXFP4",
        }
    }

    /// Fails with an actionable message for types this engine deliberately does not decode.
    pub fn check_supported(self) -> Result<()> {
        if self.needs_codebook() {
            return Err(Error::Unsupported(format!(
                "{} weights: this quantization depends on ggml's built-in codebook grids, which \
                 this engine does not carry. Use a K-quant (Q4_K/Q5_K/Q6_K), IQ4_XS, or Q8_0 \
                 build of the model instead",
                self.name()
            )));
        }
        Ok(())
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A borrowed, still-quantized 2-D weight: raw bytes exactly as they sit in the mmap.
#[derive(Clone, Copy)]
pub struct QuantView<'a> {
    pub data: &'a [u8],
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
}

impl<'a> QuantView<'a> {
    pub fn new(data: &'a [u8], dtype: DType, rows: usize, cols: usize) -> Self {
        Self { data, dtype, rows, cols }
    }

    pub fn row_bytes(&self) -> usize {
        self.dtype.row_bytes(self.cols)
    }

    pub fn row(&self, r: usize) -> &'a [u8] {
        let rb = self.row_bytes();
        &self.data[r * rb..(r + 1) * rb]
    }

    /// One expert's slice out of a stacked `[n_expert, rows, cols]` MoE tensor.
    pub fn expert(&self, e: usize, rows_per_expert: usize) -> QuantView<'a> {
        let bytes = self.row_bytes() * rows_per_expert;
        QuantView {
            data: &self.data[e * bytes..(e + 1) * bytes],
            dtype: self.dtype,
            rows: rows_per_expert,
            cols: self.cols,
        }
    }
}

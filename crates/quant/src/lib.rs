//! ggml block quantization formats.
//!
//! Three concerns live here and nowhere else: how a block decodes to floats
//! ([`dequant`]), how an activation is staged for integer arithmetic ([`quantize`]), and
//! how the two combine into a dot product ([`dot`]). Backends reuse the block layout
//! constants; nothing above this crate needs to know a block layout.

pub mod dequant;
pub mod dot;
pub mod quantize;

pub use dequant::{dequantize, dequantize_vec, KVALUES_IQ4NL, KVALUES_MXFP4};
pub use dot::{check_dot_supported, is_float, vec_dot_f32, vec_dot_q8_1};
pub use quantize::{q8_1_bytes, quantize_q8_1, quantize_q8_1_batch, Q8_1_BLOCK, Q8_1_SIZE};

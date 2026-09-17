//! Device-agnostic foundations for the gguf-rs inference engine.
//!
//! Everything above this crate is written against `Backend`; everything below it is a
//! device that implements `Backend`. Nothing here knows about a specific GPU API.

pub mod backend;
pub mod device;
pub mod dtype;
pub mod error;

pub use backend::{
    Activation, AttnCfg, Backend, BinOp, GateFunc, GluCfg, KvView, MatMulCfg, MoeCfg, MoeWeights,
    NormCfg, NormKind, RopeCfg, RopeKind,
};
pub use device::{BufferId, Caps, DeviceInfo, DeviceKind, MemKind, WeightId};
pub use dtype::{DType, QuantView};
pub use error::{backend_err, Error, Result};

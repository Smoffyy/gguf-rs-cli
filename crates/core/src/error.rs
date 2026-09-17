use std::fmt;

#[derive(Debug)]
pub enum Error {
    Format(String),
    Unsupported(String),
    MissingTensor(String),
    MissingKey(String),
    Backend { device: String, detail: String },
    OutOfMemory { device: String, requested: u64, available: u64 },
    NoDevice(String),
    Shape(String),
    Io(std::io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(m) => write!(f, "malformed GGUF: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
            Self::MissingTensor(n) => write!(f, "missing tensor: {n}"),
            Self::MissingKey(k) => write!(f, "missing metadata key: {k}"),
            Self::Backend { device, detail } => write!(f, "{device} backend: {detail}"),
            Self::OutOfMemory { device, requested, available } => write!(
                f,
                "{device} out of memory: needed {:.1} MiB, {:.1} MiB free",
                *requested as f64 / 1048576.0,
                *available as f64 / 1048576.0
            ),
            Self::NoDevice(m) => write!(f, "no usable device: {m}"),
            Self::Shape(m) => write!(f, "shape error: {m}"),
            Self::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn backend_err(device: &str, detail: impl Into<String>) -> Error {
    Error::Backend { device: device.to_string(), detail: detail.into() }
}

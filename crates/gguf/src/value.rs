use std::collections::HashMap;

/// A GGUF metadata value. Discriminants of the *wire* type ids are handled in `reader`;
/// this is the decoded form.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match self {
            Self::U8(v) => *v as u64,
            Self::I8(v) => *v as u64,
            Self::U16(v) => *v as u64,
            Self::I16(v) => *v as u64,
            Self::U32(v) => *v as u64,
            Self::I32(v) => *v as u64,
            Self::U64(v) => *v,
            Self::I64(v) => *v as u64,
            Self::Bool(v) => *v as u64,
            _ => return None,
        })
    }

    pub fn as_u32(&self) -> Option<u32> {
        self.as_u64().map(|v| v as u32)
    }

    pub fn as_i32(&self) -> Option<i32> {
        Some(match self {
            Self::I8(v) => *v as i32,
            Self::I16(v) => *v as i32,
            Self::I32(v) => *v,
            Self::I64(v) => *v as i32,
            _ => self.as_u64()? as i32,
        })
    }

    pub fn as_f32(&self) -> Option<f32> {
        Some(match self {
            Self::F32(v) => *v,
            Self::F64(v) => *v as f32,
            _ => self.as_u64()? as f32,
        })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            other => other.as_u64().map(|v| v != 0),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Scalars answer as a one-element list, because GGUF writers are inconsistent about
    /// whether things like `eos_token_id` are a scalar or an array.
    pub fn as_u32_list(&self) -> Vec<u32> {
        match self {
            Self::Array(a) => a.iter().filter_map(Value::as_u32).collect(),
            other => other.as_u32().into_iter().collect(),
        }
    }

    pub fn as_f32_list(&self) -> Vec<f32> {
        match self {
            Self::Array(a) => a.iter().filter_map(Value::as_f32).collect(),
            other => other.as_f32().into_iter().collect(),
        }
    }

    /// A short single-line rendering for `inspect`, with long arrays elided.
    pub fn summary(&self) -> String {
        match self {
            Self::String(s) if s.len() > 96 => format!("{:?}... ({} bytes)", &s[..90.min(s.len())], s.len()),
            Self::String(s) => format!("{s:?}"),
            Self::Array(a) => {
                let head: Vec<String> = a.iter().take(6).map(Value::summary).collect();
                if a.len() > 6 {
                    format!("[{}, ... {} items]", head.join(", "), a.len())
                } else {
                    format!("[{}]", head.join(", "))
                }
            }
            Self::Bool(v) => v.to_string(),
            Self::F32(v) => format!("{v}"),
            Self::F64(v) => format!("{v}"),
            other => other.as_u64().map(|v| v.to_string()).unwrap_or_default(),
        }
    }
}

/// Metadata with architecture-aware lookup.
///
/// Nearly every hyperparameter key is prefixed with the model's own architecture name
/// (`qwen3.block_count`, `gemma3.embedding_length`). Resolving that prefix here is what
/// lets an architecture the engine has never seen still load: the keys are found by shape,
/// not by a hardcoded list of known prefixes.
#[derive(Debug, Clone)]
pub struct Metadata {
    map: HashMap<String, Value>,
    pub arch: String,
}

impl Metadata {
    pub fn new(map: HashMap<String, Value>) -> Self {
        let arch = map
            .get("general.architecture")
            .and_then(Value::as_str)
            .unwrap_or("llama")
            .to_string();
        Self { map, arch }
    }

    /// Exact key lookup, no prefixing.
    pub fn raw(&self, key: &str) -> Option<&Value> {
        self.map.get(key)
    }

    /// Lookup of `{arch}.{suffix}`.
    pub fn get(&self, suffix: &str) -> Option<&Value> {
        self.map.get(&format!("{}.{}", self.arch, suffix))
    }

    /// First of several architecture-prefixed suffixes that exists.
    pub fn get_any(&self, suffixes: &[&str]) -> Option<&Value> {
        suffixes.iter().find_map(|s| self.get(s))
    }

    pub fn u32(&self, suffix: &str) -> Option<u32> {
        self.get(suffix).and_then(Value::as_u32)
    }

    pub fn u32_or(&self, suffix: &str, default: u32) -> u32 {
        self.u32(suffix).unwrap_or(default)
    }

    pub fn f32(&self, suffix: &str) -> Option<f32> {
        self.get(suffix).and_then(Value::as_f32)
    }

    pub fn f32_or(&self, suffix: &str, default: f32) -> f32 {
        self.f32(suffix).unwrap_or(default)
    }

    pub fn bool(&self, suffix: &str) -> Option<bool> {
        self.get(suffix).and_then(Value::as_bool)
    }

    pub fn string(&self, suffix: &str) -> Option<&str> {
        self.get(suffix).and_then(Value::as_str)
    }

    pub fn u32_list(&self, suffix: &str) -> Vec<u32> {
        self.get(suffix).map(Value::as_u32_list).unwrap_or_default()
    }

    pub fn f32_list(&self, suffix: &str) -> Vec<f32> {
        self.get(suffix).map(Value::as_f32_list).unwrap_or_default()
    }

    pub fn u32_req(&self, suffix: &str) -> gguf_core::Result<u32> {
        self.u32(suffix)
            .ok_or_else(|| gguf_core::Error::MissingKey(format!("{}.{}", self.arch, suffix)))
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Keys sorted, for deterministic `inspect` output.
    pub fn sorted_keys(&self) -> Vec<&str> {
        let mut k: Vec<&str> = self.map.keys().map(String::as_str).collect();
        k.sort_unstable();
        k
    }
}

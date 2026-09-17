//! Model registry.
//!
//! A TOML file mapping the `model` field of an API request to a file on disk and the
//! settings to load it with. `[defaults]` applies to every entry unless overridden, which
//! is what keeps a registry of a dozen models from repeating the same six keys.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ModelDefaults {
    pub ctx: Option<usize>,
    pub batch: Option<usize>,
    pub parallel: Option<usize>,
    pub threads: Option<usize>,
    pub device: Option<String>,
    pub kv_type: Option<String>,
    /// Pin the chat template's `enable_thinking`. Absent follows the model's own default.
    pub enable_thinking: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub max_tokens: Option<usize>,
    pub system: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    /// Path to the .gguf file.
    pub path: String,
    /// Load this model as soon as the server starts rather than on first request.
    #[serde(default)]
    pub preload: bool,
    #[serde(flatten)]
    pub settings: ModelDefaults,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    #[serde(default)]
    pub defaults: ModelDefaults,
    #[serde(default)]
    pub models: BTreeMap<String, ModelEntry>,
}

/// Merge a model's settings over the registry defaults.
macro_rules! merged {
    ($self:expr, $entry:expr, $field:ident, $fallback:expr) => {
        $entry
            .settings
            .$field
            .clone()
            .or_else(|| $self.defaults.$field.clone())
            .unwrap_or($fallback)
    };
}

impl Registry {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let reg: Self = toml::from_str(&text)
            .with_context(|| format!("parsing {}", path.display()))?;
        if reg.models.is_empty() {
            anyhow::bail!(
                "{} declares no models; add a [models.<name>] section with a path",
                path.display()
            );
        }
        for (name, m) in &reg.models {
            if !Path::new(&m.path).exists() {
                anyhow::bail!("model {name:?} points at {}, which does not exist", m.path);
            }
        }
        Ok(reg)
    }

    /// A registry with a single model, for `serve --model FILE`.
    pub fn single(path: &str, device: &str, ctx: usize, parallel: usize, threads: usize) -> Result<Self> {
        if !Path::new(path).exists() {
            anyhow::bail!("{path} does not exist");
        }
        let name = Path::new(path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "default".to_string());
        let mut models = BTreeMap::new();
        models.insert(
            name,
            ModelEntry {
                path: path.to_string(),
                preload: true,
                settings: ModelDefaults::default(),
            },
        );
        Ok(Self {
            defaults: ModelDefaults {
                ctx: Some(ctx),
                parallel: Some(parallel),
                threads: Some(threads),
                device: Some(device.to_string()),
                ..Default::default()
            },
            models,
        })
    }

    pub fn resolve(&self, name: &str) -> Option<ResolvedModel> {
        let entry = self.models.get(name)?;
        Some(ResolvedModel {
            name: name.to_string(),
            path: entry.path.clone(),
            preload: entry.preload,
            ctx: merged!(self, entry, ctx, 0),
            batch: merged!(self, entry, batch, 256),
            parallel: merged!(self, entry, parallel, 4),
            threads: merged!(self, entry, threads, 0),
            device: merged!(self, entry, device, "auto".to_string()),
            kv_type: merged!(self, entry, kv_type, "f16".to_string()),
            enable_thinking: entry
                .settings
                .enable_thinking
                .or(self.defaults.enable_thinking),
            temperature: merged!(self, entry, temperature, 0.7),
            top_k: merged!(self, entry, top_k, 40),
            top_p: merged!(self, entry, top_p, 0.9),
            min_p: merged!(self, entry, min_p, 0.05),
            repeat_penalty: merged!(self, entry, repeat_penalty, 1.1),
            max_tokens: merged!(self, entry, max_tokens, 512),
            system: entry.settings.system.clone().or_else(|| self.defaults.system.clone()),
        })
    }

    pub fn names(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// The model to use when a request does not name one.
    pub fn default_name(&self) -> Option<String> {
        self.models
            .iter()
            .find(|(_, m)| m.preload)
            .or_else(|| self.models.iter().next())
            .map(|(n, _)| n.clone())
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub name: String,
    pub path: String,
    pub preload: bool,
    pub ctx: usize,
    pub batch: usize,
    pub parallel: usize,
    pub threads: usize,
    pub device: String,
    pub kv_type: String,
    pub enable_thinking: Option<bool>,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub max_tokens: usize,
    pub system: Option<String>,
}

pub const EXAMPLE: &str = r#"# gguf-rs server registry.
#
# [defaults] applies to every model; a [models.<name>] section overrides it. The section
# name is what clients send as "model" in an API request.

[defaults]
device   = "auto"      # auto, cpu, cuda[:N], vulkan[:N]
kv_type  = "f16"       # f16 halves the KV cache; f32 for bit-comparable output
ctx      = 8192
batch    = 256
parallel = 4           # concurrent sequences per loaded model
temperature   = 0.7
top_k         = 40
top_p         = 0.9
min_p         = 0.05
repeat_penalty = 1.1
max_tokens    = 512

[models.qwen]
path    = "D:/models/Qwen3-1.7B-Q4_K_M.gguf"
preload = true

[models.gemma]
path        = "D:/models/gemma-2-2b-it-Q4_K_M.gguf"
ctx         = 4096
temperature = 0.4
"#;

use std::collections::HashMap;
use std::path::Path;
use crate::sampler::SampleParams;
use super::ini::Ini;

#[derive(Clone)]
pub struct ModelPreset {
    pub id: String,
    pub model_path: String,
    pub mmproj: Option<String>,
    pub ctx_len: usize,
    pub gpu: bool,
    pub sample: SampleParams,
    pub seed: u64,
    pub load_on_startup: bool,
    pub parallel: usize,
}

fn get<'a>(m: &'a HashMap<String, String>, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| m.get(*k)).map(|s| s.as_str())
}

fn get_f32(m: &HashMap<String, String>, keys: &[&str], def: f32) -> f32 {
    get(m, keys).and_then(|s| s.parse().ok()).unwrap_or(def)
}

fn get_usize(m: &HashMap<String, String>, keys: &[&str], def: usize) -> usize {
    get(m, keys).and_then(|s| s.parse().ok()).unwrap_or(def)
}

fn get_bool(m: &HashMap<String, String>, keys: &[&str], def: bool) -> bool {
    get(m, keys).map(|s| matches!(s, "true" | "1" | "on" | "yes")).unwrap_or(def)
}

pub fn load_registry(path: &Path) -> anyhow::Result<HashMap<String, ModelPreset>> {
    let text = std::fs::read_to_string(path)?;
    let ini = Ini::parse(&text);
    let mut out = HashMap::new();
    for id in ini.model_ids() {
        let m = ini.resolved(&id).unwrap();
        let Some(model_path) = get(&m, &["model"]) else {
            eprintln!("[server] Skipping [{id}]: no 'model' key");
            continue;
        };
        let ngl = get(&m, &["ngl", "n-gpu-layers", "gpu-layers"]);
        let gpu = ngl.map(|v| v.trim() != "0").unwrap_or(true);
        let preset = ModelPreset {
            id: id.clone(),
            model_path: model_path.to_string(),
            mmproj: get(&m, &["mmproj"]).map(|s| s.to_string()),
            ctx_len: get_usize(&m, &["ctx-size", "ctx_len", "c"], 8192),
            gpu,
            sample: SampleParams {
                temperature:       get_f32(&m, &["temp", "temperature"], 0.7),
                top_k:             get_usize(&m, &["top-k"], 40),
                top_p:             get_f32(&m, &["top-p"], 0.9),
                min_p:             get_f32(&m, &["min-p"], 0.0),
                rep_penalty:       get_f32(&m, &["repeat-penalty", "rep-penalty"], 1.1),
                presence_penalty:  get_f32(&m, &["presence-penalty"], 0.0),
                frequency_penalty: get_f32(&m, &["frequency-penalty"], 0.0),
                mirostat:          get_usize(&m, &["mirostat"], 0) as u8,
                mirostat_tau:      get_f32(&m, &["mirostat-tau"], 5.0),
                mirostat_eta:      get_f32(&m, &["mirostat-eta"], 0.1),
            },
            seed: get_usize(&m, &["seed"], 42) as u64,
            load_on_startup: get_bool(&m, &["load-on-startup"], false),
            parallel: get_usize(&m, &["parallel", "n-parallel", "np"], 4).max(1),
        };
        if preset.mmproj.is_some() {
            eprintln!("[server] [{id}]: mmproj set but vision isn't supported yet — ignoring, text-only");
        }
        out.insert(id, preset);
    }
    Ok(out)
}

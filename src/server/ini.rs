use std::collections::HashMap;

pub struct Ini {
    pub sections: HashMap<String, HashMap<String, String>>,
}

fn strip_comment(line: &str) -> &str {
    let mut in_quotes = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_quotes = !in_quotes,
            ';' | '#' if !in_quotes => return &line[..i],
            _ => {}
        }
    }
    line
}

impl Ini {
    pub fn parse(text: &str) -> Self {
        let mut sections: HashMap<String, HashMap<String, String>> = HashMap::new();
        let mut current = String::from("*");
        for raw in text.lines() {
            let line = strip_comment(raw).trim();
            if line.is_empty() { continue; }
            if let Some(inner) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                current = inner.trim().to_string();
                sections.entry(current.clone()).or_default();
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                let k = k.trim().to_lowercase();
                let v = v.trim().trim_matches('"').to_string();
                sections.entry(current.clone()).or_default().insert(k, v);
            }
        }
        Self { sections }
    }

    pub fn model_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.sections.keys()
            .filter(|k| k.as_str() != "*").cloned().collect();
        ids.sort();
        ids
    }

    pub fn resolved(&self, id: &str) -> Option<HashMap<String, String>> {
        let model = self.sections.get(id)?;
        let mut merged = self.sections.get("*").cloned().unwrap_or_default();
        for (k, v) in model { merged.insert(k.clone(), v.clone()); }
        Some(merged)
    }
}

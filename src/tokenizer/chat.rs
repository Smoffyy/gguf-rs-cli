use crate::tokenizer::bpe::Tokenizer;

#[derive(Debug, Clone, PartialEq)]
pub enum ChatTemplate { ChatML, Llama3, Llama2, Gemma, Phi3, Simple }

impl ChatTemplate {
    pub fn detect(tok: &Tokenizer, chat_template: Option<&str>) -> Self {
        if let Some(tmpl) = chat_template {
            if tmpl.contains("<|im_start|>")        { return Self::ChatML; }
            if tmpl.contains("<|start_header_id|>") { return Self::Llama3; }
            if tmpl.contains("<start_of_turn>")     { return Self::Gemma;  }
            if tmpl.contains("<|user|>")             { return Self::Phi3;   }
            if tmpl.contains("[INST]")               { return Self::Llama2; }
        }
        if tok.token_to_id.contains_key("<|im_start|>")    { return Self::ChatML; }
        if tok.token_to_id.contains_key("<|eot_id|>")      { return Self::Llama3; }
        if tok.token_to_id.contains_key("<start_of_turn>") { return Self::Gemma;  }
        if tok.token_to_id.contains_key("<|user|>")        { return Self::Phi3;   }
        if tok.token_to_id.contains_key("[INST]")          { return Self::Llama2; }
        Self::Simple
    }

    pub fn uses_bos(&self) -> bool {
        matches!(self, Self::Llama2 | Self::Simple)
    }

    pub fn system_prompt(&self, sys: &str) -> String {
        match self {
            Self::ChatML  => format!("<|im_start|>system\n{}<|im_end|>\n", sys),
            Self::Llama3  => format!("<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", sys),
            Self::Llama2  => format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", sys),
            Self::Gemma   => format!("<start_of_turn>user\n[System]: {}\n", sys),
            Self::Phi3    => format!("<|system|>\n{}<|end|>\n", sys),
            Self::Simple  => format!("System: {}\n\n", sys),
        }
    }

    pub fn user_turn(&self, msg: &str) -> String {
        match self {
            Self::ChatML  => format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", msg),
            Self::Llama3  => format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                msg),
            Self::Llama2  => format!("{} [/INST]", msg),
            Self::Gemma   => format!("<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n", msg),
            Self::Phi3    => format!("<|user|>\n{}<|end|>\n<|assistant|>\n", msg),
            Self::Simple  => format!("User: {}\nAssistant:", msg),
        }
    }

    pub fn render_conversation(&self, messages: &[(String, String)]) -> String {
        let supports_system_role = matches!(self, Self::ChatML | Self::Llama3 | Self::Phi3);
        let mut out = String::new();
        let mut pending_system = String::new();
        for (role, content) in messages {
            let role = if role == "developer" { "system" } else { role.as_str() };
            if role == "system" && !supports_system_role {
                if !pending_system.is_empty() { pending_system.push('\n'); }
                pending_system.push_str(content);
                continue;
            }
            let merged;
            let content: &str = if !pending_system.is_empty() && role != "system" {
                merged = format!("[System]: {pending_system}\n\n{content}");
                pending_system.clear();
                &merged
            } else { content };
            match self {
                Self::ChatML => out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n")),
                Self::Llama3 => out.push_str(&format!(
                    "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")),
                Self::Phi3   => out.push_str(&format!("<|{role}|>\n{content}<|end|>\n")),
                Self::Gemma  => {
                    let r = if role == "assistant" { "model" } else { "user" };
                    out.push_str(&format!("<start_of_turn>{r}\n{content}<end_of_turn>\n"));
                }
                Self::Llama2 => {
                    if role == "user" { out.push_str(&format!("[INST] {content} [/INST]")); }
                    else { out.push_str(&format!(" {content} ")); }
                }
                Self::Simple => {
                    let label = if role == "user" { "User" } else if role == "assistant" { "Assistant" } else { "System" };
                    out.push_str(&format!("{label}: {content}\n"));
                }
            }
        }
        out.push_str(match self {
            Self::ChatML => "<|im_start|>assistant\n",
            Self::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
            Self::Gemma  => "<start_of_turn>model\n",
            Self::Phi3   => "<|assistant|>\n",
            Self::Llama2 => "",
            Self::Simple => "Assistant:",
        });
        out
    }

    pub fn stop_tokens(&self, tok: &Tokenizer) -> Vec<u32> {
        let mut stops = tok.eos_ids.clone();
        let extras: &[&str] = match self {
            Self::ChatML  => &["<|im_end|>", "<|endoftext|>"],
            Self::Llama3  => &["<|eot_id|>", "<|end_of_text|>"],
            Self::Gemma   => &["<end_of_turn>"],
            Self::Phi3    => &["<|end|>"],
            _             => &[],
        };
        for &t in extras {
            if let Some(&id) = tok.token_to_id.get(t) { stops.push(id); }
        }
        stops.sort_unstable();
        stops.dedup();
        stops
    }
}

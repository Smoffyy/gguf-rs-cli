use crate::tokenizer::bpe::Tokenizer;

#[derive(Debug, Clone, PartialEq)]
pub enum ChatTemplate {
    ChatML,     // Qwen 1.5/2/2.5/3, InternLM, Yi — uses <|im_start|>/<|im_end|>
    Llama3,     // LLaMA-3.x — uses <|start_header_id|>/<|eot_id|>
    Llama2,     // LLaMA-2, Mistral-v1 — uses [INST]/[/INST]
    Gemma,      // Gemma 1/2/3 — uses <start_of_turn>/<end_of_turn>
    Phi3,       // Phi-3/3.5 — uses <|user|>/<|assistant|>
    Simple,     // Fallback plain-text format
}

impl ChatTemplate {
    /// Auto-detect from GGUF chat_template string (Jinja2) if available,
    /// otherwise fall back to checking special tokens in the vocabulary.
    pub fn detect(tok: &Tokenizer, chat_template: Option<&str>) -> Self {
        // Primary: scan the actual Jinja template stored in the GGUF
        if let Some(tmpl) = chat_template {
            if tmpl.contains("<|im_start|>")          { return Self::ChatML; }
            if tmpl.contains("<|start_header_id|>")   { return Self::Llama3; }
            if tmpl.contains("<start_of_turn>")        { return Self::Gemma; }
            if tmpl.contains("<|user|>") && tmpl.contains("<|assistant|>") { return Self::Phi3; }
            if tmpl.contains("[INST]")                 { return Self::Llama2; }
        }
        // Fallback: detect from vocabulary tokens
        if tok.token_to_id.contains_key("<|im_start|>")          { return Self::ChatML; }
        if tok.token_to_id.contains_key("<|eot_id|>")            { return Self::Llama3; }
        if tok.token_to_id.contains_key("<start_of_turn>")       { return Self::Gemma; }
        if tok.token_to_id.contains_key("<|user|>")              { return Self::Phi3; }
        if tok.token_to_id.contains_key("[INST]")                { return Self::Llama2; }
        Self::Simple
    }

    /// Whether this template needs a BOS token prepended (LLaMA-style).
    /// ChatML/Qwen templates start directly with <|im_start|> — no BOS.
    pub fn uses_bos(&self) -> bool {
        matches!(self, Self::Llama2 | Self::Simple)
    }

    /// Full system prompt block, ready to encode.
    pub fn system_prompt(&self, sys: &str) -> String {
        match self {
            // Qwen2.5 verified format (from official docs):
            // <|im_start|>system\n{content}<|im_end|>\n
            Self::ChatML  => format!("<|im_start|>system\n{}<|im_end|>\n", sys),
            Self::Llama3  => format!(
                "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>", sys),
            Self::Llama2  => format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", sys),
            Self::Gemma   => format!("<start_of_turn>user\n[System]: {}\n", sys),
            Self::Phi3    => format!("<|system|>\n{}<|end|>\n", sys),
            Self::Simple  => format!("System: {}\n\n", sys),
        }
    }

    /// User turn + assistant prompt prefix, ready to encode.
    /// Qwen2.5 verified format:
    /// <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
    pub fn user_turn(&self, msg: &str) -> String {
        match self {
            Self::ChatML  => format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", msg),
            Self::Llama3  => format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
                 <|start_header_id|>assistant<|end_header_id|>\n\n", msg),
            Self::Llama2  => format!("{} [/INST]", msg),
            Self::Gemma   => format!(
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n", msg),
            Self::Phi3    => format!("<|user|>\n{}<|end|>\n<|assistant|>\n", msg),
            Self::Simple  => format!("User: {}\nAssistant:", msg),
        }
    }

    /// Token IDs that signal the end of the assistant's turn.
    pub fn stop_tokens(&self, tok: &Tokenizer) -> Vec<u32> {
        let mut stops = vec![tok.eos_id];
        let extras: &[&str] = match self {
            Self::ChatML  => &["<|im_end|>", "<|endoftext|>"],
            Self::Llama3  => &["<|eot_id|>", "<|end_of_text|>"],
            Self::Gemma   => &["<end_of_turn>"],
            Self::Phi3    => &["<|end|>", "<|endoftext|>"],
            _             => &[],
        };
        for &t in extras {
            if let Some(&id) = tok.token_to_id.get(t) { stops.push(id); }
        }
        stops.dedup();
        stops
    }
}
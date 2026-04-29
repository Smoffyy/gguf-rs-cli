use crate::tokenizer::bpe::Tokenizer;

#[derive(Debug, Clone, PartialEq)]
pub enum ChatTemplate {
    Llama2,   // [INST] ... [/INST]
    Llama3,   // <|start_header_id|>...<|eot_id|>
    ChatML,   // <|im_start|>...<|im_end|>
    Simple,   // "User: ...\nAssistant:"
}

impl ChatTemplate {
    // Detect template from vocab tokens present in the model
    pub fn detect(tok: &Tokenizer) -> Self {
        if tok.token_to_id.contains_key("<|eot_id|>") {
            Self::Llama3
        } else if tok.token_to_id.contains_key("<|im_start|>") {
            Self::ChatML
        } else if tok.token_to_id.contains_key("[INST]") {
            Self::Llama2
        } else {
            Self::Simple
        }
    }

    // Returns the text to prepend before the first user turn (system prompt wrapper)
    pub fn system_prompt(&self, system: &str) -> String {
        match self {
            Self::Llama2  => format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", system),
            Self::Llama3  => format!("<|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>", system),
            Self::ChatML  => format!("<|im_start|>system\n{}<|im_end|>\n", system),
            Self::Simple  => format!("System: {}\n\n", system),
        }
    }

    // Returns the formatted user turn text to append
    pub fn user_turn(&self, msg: &str) -> String {
        match self {
            Self::Llama2 => format!("{} [/INST] ", msg),
            Self::Llama3 => format!(
                "<|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
                msg
            ),
            Self::ChatML => format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", msg
            ),
            Self::Simple => format!("User: {}\nAssistant:", msg),
        }
    }

    // Returns the formatted assistant turn to append after the response
    #[allow(dead_code)]
    pub fn assistant_turn_end(&self) -> &str {
        match self {
            Self::Llama2 => " </s><s>[INST] ",
            Self::Llama3 => "<|eot_id|>",
            Self::ChatML => "<|im_end|>\n",
            Self::Simple => "\n",
        }
    }

    // Token ids that signal end of the assistant's response
    pub fn stop_tokens(&self, tok: &Tokenizer) -> Vec<u32> {
        let mut stops = vec![tok.eos_id];
        let extra: &[&str] = match self {
            Self::Llama3 => &["<|eot_id|>", "<|end_of_text|>"],
            Self::ChatML => &["<|im_end|>"],
            _            => &[],
        };
        for &t in extra {
            if let Some(&id) = tok.token_to_id.get(t) { stops.push(id); }
        }
        stops
    }
}

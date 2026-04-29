use crate::tokenizer::bpe::Tokenizer;

#[derive(Debug,Clone,PartialEq)]
pub enum ChatTemplate { Llama2, Llama3, ChatML, Simple }

impl ChatTemplate {
    pub fn detect(tok: &Tokenizer) -> Self {
        if tok.token_to_id.contains_key("<|eot_id|>")     { Self::Llama3 }
        else if tok.token_to_id.contains_key("<|im_start|>") { Self::ChatML }
        else if tok.token_to_id.contains_key("[INST]")    { Self::Llama2 }
        else { Self::Simple }
    }
    pub fn system_prompt(&self, s: &str) -> String {
        match self {
            Self::Llama2  => format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", s),
            Self::Llama3  => format!("<|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>", s),
            Self::ChatML  => format!("<|im_start|>system\n{}<|im_end|>\n", s),
            Self::Simple  => format!("System: {}\n\n", s),
        }
    }
    pub fn user_turn(&self, msg: &str) -> String {
        match self {
            Self::Llama2 => format!("{} [/INST] ", msg),
            Self::Llama3 => format!("<|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n", msg),
            Self::ChatML => format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", msg),
            Self::Simple => format!("User: {}\nAssistant:", msg),
        }
    }
    pub fn stop_tokens(&self, tok: &Tokenizer) -> Vec<u32> {
        let mut s=vec![tok.eos_id];
        let extras:&[&str]=match self{
            Self::Llama3=>&["<|eot_id|>","<|end_of_text|>"],
            Self::ChatML=>&["<|im_end|>"],
            _=>&[],
        };
        for &t in extras { if let Some(&id)=tok.token_to_id.get(t){s.push(id);} }
        s
    }
}
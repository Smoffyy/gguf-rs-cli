//! Tokenization and chat formatting, both driven by what the GGUF itself declares.

mod encode;
pub mod jinja;
mod pretok;
mod template;
mod vocab;

pub use template::{ChatMessage, ChatTemplate};
pub use vocab::{TokenizerKind, Vocab};

use gguf_core::Result;
use gguf_format::GgufModel;

pub struct Tokenizer {
    pub vocab: Vocab,
    pub template: ChatTemplate,
}

impl Tokenizer {
    pub fn from_gguf(g: &GgufModel) -> Result<Self> {
        let vocab = Vocab::from_gguf(g)?;
        let template = ChatTemplate::from_vocab(&vocab);
        Ok(Self { vocab, template })
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        encode::encode(&self.vocab, text, add_special)
    }

    /// Raw bytes for one token. Multi-byte characters can span several tokens, so callers
    /// accumulate bytes and convert once rather than per token.
    pub fn token_bytes(&self, id: u32, skip_special: bool) -> Vec<u8> {
        encode::decode_token(&self.vocab, id, skip_special)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.token_bytes(id, skip_special));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.vocab.eos
    }

    pub fn is_eog(&self, id: u32) -> bool {
        self.vocab.is_eog(id) || self.template.is_stop(id)
    }
}

/// Accumulates token bytes and emits only complete UTF-8, so a multi-byte character split
/// across tokens never reaches the terminal as a replacement character.
#[derive(Default)]
pub struct Detokenizer {
    pending: Vec<u8>,
}

impl Detokenizer {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let good = e.valid_up_to();
                if good == 0 {
                    // Hold back an incomplete sequence, but never grow without bound: four
                    // bytes is the longest a valid character can be.
                    if self.pending.len() > 4 {
                        let out = String::from_utf8_lossy(&self.pending).into_owned();
                        self.pending.clear();
                        return out;
                    }
                    return String::new();
                }
                let out = String::from_utf8_lossy(&self.pending[..good]).into_owned();
                self.pending.drain(..good);
                out
            }
        }
    }

    pub fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

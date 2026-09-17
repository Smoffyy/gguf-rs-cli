use std::collections::HashMap;

use gguf_core::{Error, Result};
use gguf_format::{GgufModel, Value};

/// Which merge algorithm the vocabulary was built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    /// SentencePiece: merge the adjacent pair with the best score.
    Spm,
    /// Byte-level BPE: pre-tokenize, then merge by merge-list rank.
    Bpe,
    /// WordPiece, used by embedding models.
    Wpm,
}

/// ggml token type codes.
pub const TOKEN_NORMAL: u32 = 1;
pub const TOKEN_UNKNOWN: u32 = 2;
pub const TOKEN_CONTROL: u32 = 3;
pub const TOKEN_USER_DEFINED: u32 = 4;
pub const TOKEN_UNUSED: u32 = 5;
pub const TOKEN_BYTE: u32 = 6;

pub struct Vocab {
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub types: Vec<u32>,
    pub by_text: HashMap<String, u32>,
    pub merge_rank: HashMap<(String, String), u32>,
    pub kind: TokenizerKind,
    /// Value of `tokenizer.ggml.pre`, which selects the pre-tokenizer split rules.
    pub pre: String,

    pub bos: Option<u32>,
    pub eos: Vec<u32>,
    pub pad: Option<u32>,
    pub unk: Option<u32>,
    pub add_bos: bool,
    pub add_eos: bool,
    /// SPM prepends a space to the input so the first word tokenizes like an inner one.
    pub add_space_prefix: bool,
    pub chat_template: Option<String>,

    /// Tokens that must be matched literally before any merge algorithm runs, longest
    /// first, so that `<|im_start|>` never gets split into pieces.
    pub specials: Vec<(String, u32)>,
    byte_decoder: HashMap<char, u8>,
    byte_encoder: [char; 256],
}

impl Vocab {
    pub fn from_gguf(g: &GgufModel) -> Result<Self> {
        let m = &g.meta;
        let raw = |k: &str| m.raw(k);

        let tokens: Vec<String> = raw("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                Error::MissingKey(
                    "tokenizer.ggml.tokens (the file carries no vocabulary; a split GGUF \
                     needs its first shard, and an mmproj file is not a model)"
                        .into(),
                )
            })?
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();

        let scores: Vec<f32> = raw("tokenizer.ggml.scores")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect())
            .unwrap_or_else(|| vec![0.0; tokens.len()]);

        let types: Vec<u32> = raw("tokenizer.ggml.token_type")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|v| v.as_u32().unwrap_or(TOKEN_NORMAL)).collect())
            .unwrap_or_else(|| vec![TOKEN_NORMAL; tokens.len()]);

        let model_name = raw("tokenizer.ggml.model").and_then(Value::as_str).unwrap_or("llama");
        let kind = match model_name {
            "gpt2" => TokenizerKind::Bpe,
            "bert" | "wpm" => TokenizerKind::Wpm,
            _ => TokenizerKind::Spm,
        };

        let mut by_text = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            by_text.entry(t.clone()).or_insert(i as u32);
        }

        let mut merge_rank = HashMap::new();
        if let Some(arr) = raw("tokenizer.ggml.merges").and_then(Value::as_array) {
            for (rank, v) in arr.iter().enumerate() {
                let Some(s) = v.as_str() else { continue };
                // A merge line is "left right"; the pieces themselves never contain a
                // space, so splitting on the first one is unambiguous.
                if let Some(sp) = s.find(' ') {
                    merge_rank.insert((s[..sp].to_string(), s[sp + 1..].to_string()), rank as u32);
                }
            }
        }

        let eos: Vec<u32> = raw("tokenizer.ggml.eos_token_id")
            .map(Value::as_u32_list)
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![2]);

        let byte_encoder = build_byte_encoder();
        let byte_decoder = byte_encoder.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();

        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| {
                let ty = types.get(*i).copied().unwrap_or(TOKEN_NORMAL);
                !t.is_empty()
                    && (ty == TOKEN_CONTROL
                        || ty == TOKEN_USER_DEFINED
                        || (t.starts_with("<|") && t.ends_with("|>"))
                        || (t.starts_with("<") && t.ends_with(">") && t.len() > 2 && !t.starts_with("<0x")))
            })
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        // Longest first so a prefix of another special never wins.
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.0.cmp(&b.0)));

        Ok(Self {
            by_text,
            merge_rank,
            kind,
            pre: raw("tokenizer.ggml.pre")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_string(),
            bos: raw("tokenizer.ggml.bos_token_id").and_then(Value::as_u32),
            eos,
            pad: raw("tokenizer.ggml.padding_token_id").and_then(Value::as_u32),
            unk: raw("tokenizer.ggml.unknown_token_id").and_then(Value::as_u32),
            add_bos: raw("tokenizer.ggml.add_bos_token")
                .and_then(Value::as_bool)
                .unwrap_or(kind == TokenizerKind::Spm),
            add_eos: raw("tokenizer.ggml.add_eos_token").and_then(Value::as_bool).unwrap_or(false),
            add_space_prefix: raw("tokenizer.ggml.add_space_prefix")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            chat_template: raw("tokenizer.chat_template")
                .and_then(Value::as_str)
                .map(str::to_string),
            specials,
            tokens,
            scores,
            types,
            byte_decoder,
            byte_encoder,
        })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn id(&self, text: &str) -> Option<u32> {
        self.by_text.get(text).copied()
    }

    pub fn text(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map(String::as_str).unwrap_or("")
    }

    pub fn token_type(&self, id: u32) -> u32 {
        self.types.get(id as usize).copied().unwrap_or(TOKEN_NORMAL)
    }

    pub fn is_control(&self, id: u32) -> bool {
        matches!(self.token_type(id), TOKEN_CONTROL | TOKEN_UNKNOWN)
    }

    pub fn is_eog(&self, id: u32) -> bool {
        self.eos.contains(&id)
    }

    pub fn byte_to_char(&self, b: u8) -> char {
        self.byte_encoder[b as usize]
    }

    pub fn char_to_byte(&self, c: char) -> Option<u8> {
        self.byte_decoder.get(&c).copied()
    }

    /// The `<0xNN>` fallback id for a raw byte, if the vocabulary has one.
    pub fn byte_fallback(&self, b: u8) -> Option<u32> {
        self.id(&format!("<0x{b:02X}>"))
    }
}

/// GPT-2's byte-to-printable-character map. Bytes that are already printable ASCII or
/// Latin-1 map to themselves; the rest are shifted into the U+0100 block so that every
/// byte has a distinct, non-whitespace character.
fn build_byte_encoder() -> [char; 256] {
    let mut enc = ['\0'; 256];
    let mut mapped = [false; 256];
    for &(lo, hi) in &[(33u32, 126u32), (161, 172), (174, 255)] {
        for b in lo..=hi {
            enc[b as usize] = char::from_u32(b).unwrap();
            mapped[b as usize] = true;
        }
    }
    let mut n = 0u32;
    for b in 0..256usize {
        if !mapped[b] {
            enc[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    enc
}

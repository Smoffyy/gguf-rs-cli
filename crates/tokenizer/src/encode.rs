//! Encoding text to token ids.
//!
//! Both algorithms are the same shape: split the input into symbols, then repeatedly merge
//! the best adjacent pair. "Best" is the highest vocabulary score for SentencePiece and
//! the lowest merge-list rank for BPE. The candidate pairs live in a heap, so a merge
//! costs a log-factor rather than a full rescan of the symbol list — the difference between
//! quadratic and near-linear on a long prompt.

use std::collections::BinaryHeap;

use crate::pretok;
use crate::vocab::{TokenizerKind, Vocab};

/// A doubly-linked symbol run over the input, so merging is a pointer update rather than a
/// vector shift.
#[derive(Clone, Copy)]
struct Symbol {
    prev: i32,
    next: i32,
    start: usize,
    len: usize,
}

#[derive(PartialEq)]
struct Candidate {
    /// Ordering key. SPM maximizes score; BPE minimizes rank, which is stored negated so
    /// one max-heap serves both.
    key: f32,
    left: i32,
    right: i32,
    len: usize,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Ties break toward the earlier position, matching the reference tokenizers.
        self.key
            .total_cmp(&other.key)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub fn encode(vocab: &Vocab, text: &str, add_special: bool) -> Vec<u32> {
    let mut out = Vec::new();
    if add_special && vocab.add_bos {
        if let Some(bos) = vocab.bos {
            out.push(bos);
        }
    }

    for (piece, special) in split_specials(vocab, text) {
        match special {
            Some(id) => out.push(id),
            None => encode_piece(vocab, &piece, &mut out),
        }
    }

    if add_special && vocab.add_eos {
        if let Some(&eos) = vocab.eos.first() {
            out.push(eos);
        }
    }
    out
}

/// Cut the text at every special-token occurrence. Specials are matched literally and never
/// participate in merging, which is what stops a prompt containing `<|im_end|>` from being
/// tokenized as ordinary text.
fn split_specials(vocab: &Vocab, text: &str) -> Vec<(String, Option<u32>)> {
    if vocab.specials.is_empty() {
        return vec![(text.to_string(), None)];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut i = 0;
    let bytes = text.as_bytes();
    'outer: while i < bytes.len() {
        for (tok, id) in &vocab.specials {
            if text[i..].starts_with(tok.as_str()) {
                if !cur.is_empty() {
                    out.push((std::mem::take(&mut cur), None));
                }
                out.push((tok.clone(), Some(*id)));
                i += tok.len();
                continue 'outer;
            }
        }
        // Advance one whole character, not one byte.
        let ch_len = text[i..].chars().next().map(char::len_utf8).unwrap_or(1);
        cur.push_str(&text[i..i + ch_len]);
        i += ch_len;
    }
    if !cur.is_empty() {
        out.push((cur, None));
    }
    out
}

fn encode_piece(vocab: &Vocab, text: &str, out: &mut Vec<u32>) {
    if text.is_empty() {
        return;
    }
    match vocab.kind {
        TokenizerKind::Spm => encode_spm(vocab, text, out),
        TokenizerKind::Bpe => {
            for word in pretok::split(&vocab.pre, text) {
                let staged: String = word.bytes().map(|b| vocab.byte_to_char(b)).collect();
                merge(vocab, &staged, out, false);
            }
        }
        TokenizerKind::Wpm => encode_wpm(vocab, text, out),
    }
}

fn encode_spm(vocab: &Vocab, text: &str, out: &mut Vec<u32>) {
    // SentencePiece works on a space-escaped string, and treats the start of the input as
    // if it followed a space so that "Hello" and " Hello" agree on the first token.
    let mut staged = String::with_capacity(text.len() + 1);
    if vocab.add_space_prefix && !text.starts_with(' ') {
        staged.push('\u{2581}');
    }
    for c in text.chars() {
        if c == ' ' {
            staged.push('\u{2581}');
        } else {
            staged.push(c);
        }
    }
    merge(vocab, &staged, out, true);
}

fn merge(vocab: &Vocab, text: &str, out: &mut Vec<u32>, spm: bool) {
    if text.is_empty() {
        return;
    }

    let mut syms: Vec<Symbol> = Vec::new();
    let mut offset = 0usize;
    for c in text.chars() {
        let len = c.len_utf8();
        let idx = syms.len() as i32;
        syms.push(Symbol { prev: idx - 1, next: idx + 1, start: offset, len });
        offset += len;
    }
    if let Some(last) = syms.last_mut() {
        last.next = -1;
    }

    let mut heap = BinaryHeap::new();
    let push = |heap: &mut BinaryHeap<Candidate>, syms: &[Symbol], left: i32, right: i32| {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (&syms[left as usize], &syms[right as usize]);
        if l.len == 0 || r.len == 0 {
            return;
        }
        let merged = &text[l.start..r.start + r.len];
        let key = if spm {
            match vocab.id(merged) {
                Some(id) => vocab.scores.get(id as usize).copied().unwrap_or(f32::NEG_INFINITY),
                None => return,
            }
        } else {
            let lhs = &text[l.start..l.start + l.len];
            let rhs = &text[r.start..r.start + r.len];
            match vocab.merge_rank.get(&(lhs.to_string(), rhs.to_string())) {
                Some(rank) => -(*rank as f32),
                None => return,
            }
        };
        heap.push(Candidate { key, left, right, len: merged.len() });
    };

    for i in 0..syms.len() as i32 - 1 {
        push(&mut heap, &syms, i, i + 1);
    }

    while let Some(c) = heap.pop() {
        let (li, ri) = (c.left as usize, c.right as usize);
        // A symbol that has already been absorbed has length zero, and a stale candidate
        // no longer spans what it did when it was queued.
        if syms[li].len == 0 || syms[ri].len == 0 {
            continue;
        }
        if syms[li].len + syms[ri].len != c.len {
            continue;
        }
        let right_next = syms[ri].next;
        syms[li].len += syms[ri].len;
        syms[ri].len = 0;
        syms[li].next = right_next;
        if right_next >= 0 {
            syms[right_next as usize].prev = c.left;
        }
        let left_prev = syms[li].prev;
        push(&mut heap, &syms, left_prev, c.left);
        push(&mut heap, &syms, c.left, right_next);
    }

    let mut i = 0i32;
    while i >= 0 && (i as usize) < syms.len() {
        let s = syms[i as usize];
        if s.len > 0 {
            emit(vocab, &text[s.start..s.start + s.len], out, spm);
        }
        i = s.next;
    }
}

fn emit(vocab: &Vocab, piece: &str, out: &mut Vec<u32>, spm: bool) {
    if let Some(id) = vocab.id(piece) {
        out.push(id);
        return;
    }
    // Nothing in the vocabulary covers this run, so fall back byte by byte. SPM vocabs
    // carry `<0xNN>` tokens for exactly this; byte-level BPE vocabs encode bytes as
    // characters already, so a miss there means a genuinely unknown symbol.
    if spm {
        for b in piece.bytes() {
            if let Some(id) = vocab.byte_fallback(b) {
                out.push(id);
            } else if let Some(unk) = vocab.unk {
                out.push(unk);
            }
        }
    } else {
        for c in piece.chars() {
            if let Some(id) = vocab.id(&c.to_string()) {
                out.push(id);
            } else if let Some(unk) = vocab.unk {
                out.push(unk);
            }
        }
    }
}

fn encode_wpm(vocab: &Vocab, text: &str, out: &mut Vec<u32>) {
    for word in text.split_whitespace() {
        let lowered = word.to_lowercase();
        let chars: Vec<char> = lowered.chars().collect();
        let mut start = 0usize;
        let mut pieces = Vec::new();
        let mut ok = true;
        while start < chars.len() {
            let mut end = chars.len();
            let mut matched = None;
            while end > start {
                let sub: String = chars[start..end].iter().collect();
                let candidate = if start == 0 { sub } else { format!("##{sub}") };
                if let Some(id) = vocab.id(&candidate) {
                    matched = Some((id, end));
                    break;
                }
                end -= 1;
            }
            match matched {
                Some((id, e)) => {
                    pieces.push(id);
                    start = e;
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            out.extend(pieces);
        } else if let Some(unk) = vocab.unk {
            out.push(unk);
        }
    }
}

/// Render one token back to raw bytes.
///
/// Bytes rather than a string: SPM byte-fallback tokens and multi-byte characters split
/// across several tokens are not individually valid UTF-8, so assembly happens in the
/// caller.
pub fn decode_token(vocab: &Vocab, id: u32, skip_special: bool) -> Vec<u8> {
    if id as usize >= vocab.len() {
        return Vec::new();
    }
    let ty = vocab.token_type(id);
    if skip_special && (ty == crate::vocab::TOKEN_CONTROL || ty == crate::vocab::TOKEN_UNKNOWN) {
        return Vec::new();
    }
    // A slot the vocabulary marks unused has no text, whatever string is stored in it.
    if ty == crate::vocab::TOKEN_UNUSED {
        return Vec::new();
    }
    let raw = vocab.text(id);
    match vocab.kind {
        TokenizerKind::Spm => {
            // Byte-fallback tokens are `<0xNN>`; the type code says so when the file sets it,
            // and the spelling is the fallback for files that do not.
            if let Some(hex) = raw
                .strip_prefix("<0x")
                .and_then(|s| s.strip_suffix('>'))
                .filter(|_| ty == crate::vocab::TOKEN_BYTE || ty == crate::vocab::TOKEN_NORMAL)
            {
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    return vec![b];
                }
            }
            raw.replace('\u{2581}', " ").into_bytes()
        }
        TokenizerKind::Bpe => {
            let bytes: Vec<u8> = raw.chars().filter_map(|c| vocab.char_to_byte(c)).collect();
            if bytes.len() == raw.chars().count() {
                bytes
            } else {
                raw.as_bytes().to_vec()
            }
        }
        TokenizerKind::Wpm => match raw.strip_prefix("##") {
            Some(rest) => rest.as_bytes().to_vec(),
            None => format!(" {raw}").into_bytes(),
        },
    }
}

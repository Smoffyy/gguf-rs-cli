//! Byte-level BPE pre-tokenization.
//!
//! Before merging, BPE vocabularies split text into words with a fixed regex, and the
//! regex differs per model family. Getting it wrong does not error: it silently produces a
//! different, still-valid-looking token sequence, and the model's output degrades in ways
//! that are easy to mistake for a bad prompt.
//!
//! These are hand-written matchers rather than a regex engine, because the patterns in
//! circulation are a handful of fixed alternations and pulling in a full Unicode regex crate
//! for them would be the larger dependency.

/// The split rules a given `tokenizer.ggml.pre` value selects.
#[derive(Debug, Clone, Copy)]
struct Rules {
    /// Contractions match regardless of case (Llama-3, Qwen-2 and descendants).
    ci_contractions: bool,
    /// Letters may absorb any single preceding non-letter, non-digit, non-newline
    /// character. When false, only a space may be absorbed.
    letters_absorb_any: bool,
    /// Maximum digits per token. Qwen splits digits singly, Llama-3 in threes, GPT-2 takes
    /// a whole run.
    digit_group: usize,
    /// A symbol run may swallow the newlines that follow it.
    symbols_take_newlines: bool,
}

const GPT2: Rules = Rules {
    ci_contractions: false,
    letters_absorb_any: false,
    digit_group: usize::MAX,
    symbols_take_newlines: false,
};

const LLAMA3: Rules = Rules {
    ci_contractions: true,
    letters_absorb_any: true,
    digit_group: 3,
    symbols_take_newlines: true,
};

const QWEN2: Rules = Rules {
    ci_contractions: true,
    letters_absorb_any: true,
    digit_group: 1,
    symbols_take_newlines: true,
};

fn rules_for(pre: &str) -> Rules {
    match pre {
        "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "pixtral" | "seed-coder" => LLAMA3,
        "qwen2" | "deepseek-llm" | "deepseek-coder" | "deepseek-v3" | "jina-v2-code"
        | "smaug-bpe" | "chatglm-bpe" | "minerva-7b" | "hunyuan" | "exaone" | "gpt-4o"
        | "superbpe" | "kimi-k2" | "llama4" | "bailingmoe" | "a.x-4.0" => QWEN2,
        // GPT-2's pattern is the common ancestor and the right default: it is what an
        // unrecognised `pre` value most likely means.
        _ => GPT2,
    }
}

#[inline]
fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

#[inline]
fn is_digit(c: char) -> bool {
    c.is_numeric()
}

#[inline]
fn is_ws(c: char) -> bool {
    c.is_whitespace()
}

const CONTRACTIONS: [&str; 6] = ["'s", "'t", "'re", "'ve", "'m", "'ll"];

/// Split `text` into the pieces BPE merges independently.
pub fn split(pre: &str, text: &str) -> Vec<String> {
    let r = rules_for(pre);
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        if let Some(n) = match_contraction(&chars, i, r) {
            out.push(chars[i..i + n].iter().collect());
            i += n;
            continue;
        }
        if let Some(n) = match_letters(&chars, i, r) {
            out.push(chars[i..i + n].iter().collect());
            i += n;
            continue;
        }
        if let Some(n) = match_digits(&chars, i, r) {
            out.push(chars[i..i + n].iter().collect());
            i += n;
            continue;
        }
        if let Some(n) = match_symbols(&chars, i, r) {
            out.push(chars[i..i + n].iter().collect());
            i += n;
            continue;
        }
        if let Some(n) = match_whitespace(&chars, i, r) {
            out.push(chars[i..i + n].iter().collect());
            i += n;
            continue;
        }
        // Unreachable for well-formed input, but a lone unmatched character must still
        // make progress rather than spin.
        out.push(chars[i].to_string());
        i += 1;
    }
    out
}

fn match_contraction(c: &[char], i: usize, r: Rules) -> Option<usize> {
    if c[i] != '\'' {
        return None;
    }
    let rest: String = c[i..(i + 3).min(c.len())].iter().collect();
    let hay = if r.ci_contractions { rest.to_lowercase() } else { rest };
    let mut best = None;
    for pat in CONTRACTIONS {
        if hay.starts_with(pat) {
            let n = pat.chars().count();
            if best.map_or(true, |b| n > b) {
                best = Some(n);
            }
        }
    }
    // "'d" is in the list too, but only as a two-character match.
    if best.is_none() && hay.starts_with("'d") {
        best = Some(2);
    }
    best
}

fn match_letters(c: &[char], i: usize, r: Rules) -> Option<usize> {
    let mut j = i;
    // Optional single leading character.
    if r.letters_absorb_any {
        if !is_letter(c[j]) && !is_digit(c[j]) && c[j] != '\r' && c[j] != '\n' {
            j += 1;
        }
    } else if c[j] == ' ' {
        j += 1;
    }
    let start_letters = j;
    while j < c.len() && is_letter(c[j]) {
        j += 1;
    }
    if j == start_letters {
        return None;
    }
    Some(j - i)
}

fn match_digits(c: &[char], i: usize, r: Rules) -> Option<usize> {
    let mut j = i;
    let leading_space = r.digit_group == usize::MAX && c[j] == ' ';
    if leading_space {
        j += 1;
    }
    let start = j;
    let limit = if r.digit_group == usize::MAX { usize::MAX } else { r.digit_group };
    while j < c.len() && is_digit(c[j]) && (j - start) < limit {
        j += 1;
    }
    if j == start {
        return None;
    }
    Some(j - i)
}

fn match_symbols(c: &[char], i: usize, r: Rules) -> Option<usize> {
    let mut j = i;
    if c[j] == ' ' && j + 1 < c.len() && !is_ws(c[j + 1]) && !is_letter(c[j + 1]) && !is_digit(c[j + 1]) {
        j += 1;
    }
    let start = j;
    while j < c.len() && !is_ws(c[j]) && !is_letter(c[j]) && !is_digit(c[j]) {
        j += 1;
    }
    if j == start {
        return None;
    }
    if r.symbols_take_newlines {
        while j < c.len() && (c[j] == '\r' || c[j] == '\n') {
            j += 1;
        }
    }
    Some(j - i)
}

fn match_whitespace(c: &[char], i: usize, r: Rules) -> Option<usize> {
    if !is_ws(c[i]) {
        return None;
    }

    // `\s*[\r\n]+`: a whitespace run that ends in newlines is one token.
    if r.symbols_take_newlines {
        let mut j = i;
        while j < c.len() && is_ws(c[j]) && c[j] != '\r' && c[j] != '\n' {
            j += 1;
        }
        if j < c.len() && (c[j] == '\r' || c[j] == '\n') {
            while j < c.len() && (c[j] == '\r' || c[j] == '\n') {
                j += 1;
            }
            return Some(j - i);
        }
    }

    let mut j = i;
    while j < c.len() && is_ws(c[j]) {
        j += 1;
    }
    // `\s+(?!\S)` gives the final space back to whatever follows, so " a" tokenizes with
    // the space attached to the word rather than as a separate run.
    if j < c.len() && j - i > 1 {
        return Some(j - i - 1);
    }
    Some(j - i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_splits_like_the_reference_pattern() {
        assert_eq!(split("default", "Hello world"), vec!["Hello", " world"]);
        assert_eq!(split("default", "it's"), vec!["it", "'s"]);
        assert_eq!(split("default", "a  b"), vec!["a", " ", " b"]);
        assert_eq!(split("default", "x=1234"), vec!["x", "=", "1234"]);
    }

    #[test]
    fn qwen_splits_digits_singly_and_llama3_in_threes() {
        assert_eq!(split("qwen2", "12345"), vec!["1", "2", "3", "4", "5"]);
        assert_eq!(split("llama3", "12345"), vec!["123", "45"]);
    }

    #[test]
    fn newlines_stay_with_their_run() {
        assert_eq!(split("qwen2", "a\n\nb"), vec!["a", "\n\n", "b"]);
    }

    #[test]
    fn every_input_is_covered_exactly_once() {
        for pre in ["default", "llama3", "qwen2"] {
            for s in ["", "  ", "\n", "héllo wörld 42!", "a'B'c", "\t\tx", "]]>", "🙂 ok"] {
                let joined: String = split(pre, s).concat();
                assert_eq!(joined, s, "{pre} lost or duplicated text in {s:?}");
            }
        }
    }
}

//! A Jinja2 subset, enough to render the `tokenizer.chat_template` that ships inside GGUF
//! files.
//!
//! Every model carries its own prompt format, and that format changes faster than any
//! hardcoded table can track. Interpreting the template the file already contains is the
//! difference between supporting the models that existed when this was written and
//! supporting the ones that ship next month.
//!
//! This is deliberately not a general Jinja engine. It implements the constructs that
//! appear in real chat templates - output, conditionals, loops, assignment, filters, tests,
//! string methods and whitespace control - and reports a clear error for anything else so
//! the caller can fall back to a built-in format rather than emit a malformed prompt.

use std::collections::BTreeMap;
use std::fmt::Write as _;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    List(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

impl Value {
    pub fn truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Bool(b) => *b,
            Self::Num(n) => *n != 0.0,
            Self::Str(s) => !s.is_empty(),
            Self::List(l) => !l.is_empty(),
            Self::Map(m) => !m.is_empty(),
        }
    }

    pub fn to_text(&self) -> String {
        match self {
            Self::Null => String::new(),
            Self::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            Self::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            Self::Str(s) => s.clone(),
            Self::List(_) | Self::Map(_) => self.to_json(),
        }
    }

    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out);
        out
    }

    fn write_json(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => {
                let _ = write!(out, "{b}");
            }
            Self::Num(n) => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    let _ = write!(out, "{}", *n as i64);
                } else {
                    let _ = write!(out, "{n}");
                }
            }
            Self::Str(s) => {
                out.push('"');
                for c in s.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\r' => out.push_str("\\r"),
                        '\t' => out.push_str("\\t"),
                        c if (c as u32) < 0x20 => {
                            let _ = write!(out, "\\u{:04x}", c as u32);
                        }
                        c => out.push(c),
                    }
                }
                out.push('"');
            }
            Self::List(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    v.write_json(out);
                }
                out.push(']');
            }
            Self::Map(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    Self::Str(k.clone()).write_json(out);
                    out.push_str(": ");
                    v.write_json(out);
                }
                out.push('}');
            }
        }
    }

    pub fn str(s: impl Into<String>) -> Self {
        Self::Str(s.into())
    }
}

pub type Error = String;
type R<T> = Result<T, Error>;

// ---------------------------------------------------------------------------- expressions

#[derive(Debug, Clone)]
enum Expr {
    Lit(Value),
    Var(String),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>, Vec<(String, Expr)>),
    Filter(Box<Expr>, String, Vec<Expr>),
    Test(Box<Expr>, String, bool, Option<Box<Expr>>),
    Unary(&'static str, Box<Expr>),
    Binary(&'static str, Box<Expr>, Box<Expr>),
    Ternary(Box<Expr>, Box<Expr>, Box<Expr>),
    List(Vec<Expr>),
    Map(Vec<(Expr, Expr)>),
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Name(String),
    Str(String),
    Num(f64),
    Op(String),
    End,
}

struct Lexer {
    toks: Vec<Tok>,
    pos: usize,
}

fn lex(src: &str) -> R<Vec<Tok>> {
    let c: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '"' || ch == '\'' {
            let quote = ch;
            i += 1;
            let mut s = String::new();
            while i < c.len() && c[i] != quote {
                if c[i] == '\\' && i + 1 < c.len() {
                    i += 1;
                    s.push(match c[i] {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '0' => '\0',
                        other => other,
                    });
                } else {
                    s.push(c[i]);
                }
                i += 1;
            }
            if i >= c.len() {
                return Err("unterminated string in expression".into());
            }
            i += 1;
            out.push(Tok::Str(s));
            continue;
        }
        if ch.is_ascii_digit() {
            let start = i;
            while i < c.len() && (c[i].is_ascii_digit() || c[i] == '.') {
                i += 1;
            }
            let text: String = c[start..i].iter().collect();
            out.push(Tok::Num(text.parse().map_err(|_| format!("bad number {text}"))?));
            continue;
        }
        if ch.is_alphabetic() || ch == '_' {
            let start = i;
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            out.push(Tok::Name(c[start..i].iter().collect()));
            continue;
        }
        // Two-character operators first so `==` never lexes as two `=`.
        let two: String = c[i..(i + 2).min(c.len())].iter().collect();
        if ["==", "!=", "<=", ">=", "//", "**"].contains(&two.as_str()) {
            out.push(Tok::Op(two));
            i += 2;
            continue;
        }
        out.push(Tok::Op(ch.to_string()));
        i += 1;
    }
    out.push(Tok::End);
    Ok(out)
}

impl Lexer {
    fn peek(&self) -> &Tok {
        self.toks.get(self.pos).unwrap_or(&Tok::End)
    }

    fn next(&mut self) -> Tok {
        let t = self.toks.get(self.pos).cloned().unwrap_or(Tok::End);
        self.pos += 1;
        t
    }

    fn eat_op(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Tok::Op(o) if o == op) {
            self.pos += 1;
            return true;
        }
        false
    }

    fn eat_name(&mut self, name: &str) -> bool {
        if matches!(self.peek(), Tok::Name(n) if n == name) {
            self.pos += 1;
            return true;
        }
        false
    }

    fn expect_op(&mut self, op: &str) -> R<()> {
        if self.eat_op(op) {
            Ok(())
        } else {
            Err(format!("expected {op:?}, found {:?}", self.peek()))
        }
    }

    fn at_name(&self, name: &str) -> bool {
        matches!(self.peek(), Tok::Name(n) if n == name)
    }
}

fn parse_expr_str(src: &str) -> R<Expr> {
    let mut lx = Lexer { toks: lex(src)?, pos: 0 };
    let e = parse_ternary(&mut lx)?;
    if *lx.peek() != Tok::End {
        return Err(format!("trailing input in expression: {:?}", lx.peek()));
    }
    Ok(e)
}

fn parse_ternary(lx: &mut Lexer) -> R<Expr> {
    let lhs = parse_or(lx)?;
    if lx.eat_name("if") {
        let cond = parse_or(lx)?;
        // `a if c` with no else yields undefined, which renders as empty.
        let other = if lx.eat_name("else") {
            parse_ternary(lx)?
        } else {
            Expr::Lit(Value::Null)
        };
        return Ok(Expr::Ternary(Box::new(cond), Box::new(lhs), Box::new(other)));
    }
    Ok(lhs)
}

fn parse_or(lx: &mut Lexer) -> R<Expr> {
    let mut lhs = parse_and(lx)?;
    while lx.eat_name("or") {
        lhs = Expr::Binary("or", Box::new(lhs), Box::new(parse_and(lx)?));
    }
    Ok(lhs)
}

fn parse_and(lx: &mut Lexer) -> R<Expr> {
    let mut lhs = parse_not(lx)?;
    while lx.eat_name("and") {
        lhs = Expr::Binary("and", Box::new(lhs), Box::new(parse_not(lx)?));
    }
    Ok(lhs)
}

fn parse_not(lx: &mut Lexer) -> R<Expr> {
    if lx.eat_name("not") {
        return Ok(Expr::Unary("not", Box::new(parse_not(lx)?)));
    }
    parse_compare(lx)
}

fn parse_compare(lx: &mut Lexer) -> R<Expr> {
    let lhs = parse_additive(lx)?;
    for op in ["==", "!=", "<=", ">=", "<", ">"] {
        if lx.eat_op(op) {
            let rhs = parse_additive(lx)?;
            let op: &'static str = match op {
                "==" => "==",
                "!=" => "!=",
                "<=" => "<=",
                ">=" => ">=",
                "<" => "<",
                _ => ">",
            };
            return Ok(Expr::Binary(op, Box::new(lhs), Box::new(rhs)));
        }
    }
    if lx.eat_name("in") {
        return Ok(Expr::Binary("in", Box::new(lhs), Box::new(parse_additive(lx)?)));
    }
    if lx.at_name("not") {
        let save = lx.pos;
        lx.pos += 1;
        if lx.eat_name("in") {
            return Ok(Expr::Unary(
                "not",
                Box::new(Expr::Binary("in", Box::new(lhs), Box::new(parse_additive(lx)?))),
            ));
        }
        lx.pos = save;
    }
    if lx.eat_name("is") {
        let negated = lx.eat_name("not");
        let name = match lx.next() {
            Tok::Name(n) => n,
            other => return Err(format!("expected a test name after `is`, found {other:?}")),
        };
        // `is equalto(x)` and `is divisibleby(x)` take an argument.
        let arg = if lx.eat_op("(") {
            let a = parse_ternary(lx)?;
            lx.expect_op(")")?;
            Some(Box::new(a))
        } else {
            None
        };
        return Ok(Expr::Test(Box::new(lhs), name, negated, arg));
    }
    Ok(lhs)
}

fn parse_additive(lx: &mut Lexer) -> R<Expr> {
    let mut lhs = parse_multiplicative(lx)?;
    loop {
        if lx.eat_op("+") {
            lhs = Expr::Binary("+", Box::new(lhs), Box::new(parse_multiplicative(lx)?));
        } else if lx.eat_op("-") {
            lhs = Expr::Binary("-", Box::new(lhs), Box::new(parse_multiplicative(lx)?));
        } else if lx.eat_op("~") {
            lhs = Expr::Binary("~", Box::new(lhs), Box::new(parse_multiplicative(lx)?));
        } else {
            return Ok(lhs);
        }
    }
}

fn parse_multiplicative(lx: &mut Lexer) -> R<Expr> {
    let mut lhs = parse_unary(lx)?;
    loop {
        let op = if lx.eat_op("*") {
            "*"
        } else if lx.eat_op("//") {
            "//"
        } else if lx.eat_op("/") {
            "/"
        } else if lx.eat_op("%") {
            "%"
        } else {
            return Ok(lhs);
        };
        lhs = Expr::Binary(op, Box::new(lhs), Box::new(parse_unary(lx)?));
    }
}

fn parse_unary(lx: &mut Lexer) -> R<Expr> {
    if lx.eat_op("-") {
        return Ok(Expr::Unary("-", Box::new(parse_unary(lx)?)));
    }
    parse_postfix(lx)
}

fn parse_postfix(lx: &mut Lexer) -> R<Expr> {
    let mut e = parse_primary(lx)?;
    loop {
        if lx.eat_op(".") {
            match lx.next() {
                Tok::Name(n) => e = Expr::Attr(Box::new(e), n),
                other => return Err(format!("expected an attribute name, found {other:?}")),
            }
        } else if lx.eat_op("[") {
            let idx = parse_slice_or_index(lx)?;
            lx.expect_op("]")?;
            e = idx(e);
        } else if lx.eat_op("(") {
            let (args, kwargs) = parse_args(lx)?;
            e = Expr::Call(Box::new(e), args, kwargs);
        } else if lx.eat_op("|") {
            let name = match lx.next() {
                Tok::Name(n) => n,
                other => return Err(format!("expected a filter name, found {other:?}")),
            };
            let args = if lx.eat_op("(") {
                let (a, _) = parse_args(lx)?;
                a
            } else {
                Vec::new()
            };
            e = Expr::Filter(Box::new(e), name, args);
        } else {
            return Ok(e);
        }
    }
}

/// `[i]`, `[a:b]`, `[:-1]` and `[::-1]` all appear in real templates. The last of those is
/// how several chat templates walk the history backwards to find the most recent user turn,
/// so a parser that stops at the second colon rejects the template outright.
#[allow(clippy::type_complexity)]
fn parse_slice_or_index(lx: &mut Lexer) -> R<Box<dyn FnOnce(Expr) -> Expr>> {
    let start = if matches!(lx.peek(), Tok::Op(o) if o == ":") {
        None
    } else {
        Some(parse_ternary(lx)?)
    };
    if lx.eat_op(":") {
        let end = if matches!(lx.peek(), Tok::Op(o) if o == "]" || o == ":") {
            None
        } else {
            Some(parse_ternary(lx)?)
        };
        let step = if lx.eat_op(":") {
            if matches!(lx.peek(), Tok::Op(o) if o == "]") {
                None
            } else {
                Some(parse_ternary(lx)?)
            }
        } else {
            None
        };
        let s = start.unwrap_or(Expr::Lit(Value::Null));
        let e = end.unwrap_or(Expr::Lit(Value::Null));
        let st = step.unwrap_or(Expr::Lit(Value::Null));
        return Ok(Box::new(move |base| {
            Expr::Filter(Box::new(base), "__slice".into(), vec![s, e, st])
        }));
    }
    let idx = start.ok_or_else(|| "empty index".to_string())?;
    Ok(Box::new(move |base| Expr::Index(Box::new(base), Box::new(idx))))
}

fn parse_args(lx: &mut Lexer) -> R<(Vec<Expr>, Vec<(String, Expr)>)> {
    let mut args = Vec::new();
    let mut kwargs = Vec::new();
    if lx.eat_op(")") {
        return Ok((args, kwargs));
    }
    loop {
        // A keyword argument is `name=value`, distinguishable only by lookahead.
        if let Tok::Name(n) = lx.peek().clone() {
            let save = lx.pos;
            lx.pos += 1;
            if lx.eat_op("=") {
                kwargs.push((n, parse_ternary(lx)?));
                if lx.eat_op(",") {
                    continue;
                }
                lx.expect_op(")")?;
                return Ok((args, kwargs));
            }
            lx.pos = save;
        }
        args.push(parse_ternary(lx)?);
        if lx.eat_op(",") {
            continue;
        }
        lx.expect_op(")")?;
        return Ok((args, kwargs));
    }
}

fn parse_primary(lx: &mut Lexer) -> R<Expr> {
    match lx.next() {
        Tok::Num(n) => Ok(Expr::Lit(Value::Num(n))),
        Tok::Str(s) => Ok(Expr::Lit(Value::Str(s))),
        Tok::Name(n) => Ok(match n.as_str() {
            "true" | "True" => Expr::Lit(Value::Bool(true)),
            "false" | "False" => Expr::Lit(Value::Bool(false)),
            "none" | "None" | "null" => Expr::Lit(Value::Null),
            _ => Expr::Var(n),
        }),
        Tok::Op(op) if op == "(" => {
            let e = parse_ternary(lx)?;
            // A parenthesised comma list is a tuple; templates only use it for iteration.
            if lx.eat_op(",") {
                let mut items = vec![e];
                while !lx.eat_op(")") {
                    items.push(parse_ternary(lx)?);
                    lx.eat_op(",");
                }
                return Ok(Expr::List(items));
            }
            lx.expect_op(")")?;
            Ok(e)
        }
        Tok::Op(op) if op == "[" => {
            let mut items = Vec::new();
            if lx.eat_op("]") {
                return Ok(Expr::List(items));
            }
            loop {
                items.push(parse_ternary(lx)?);
                if lx.eat_op(",") {
                    if lx.eat_op("]") {
                        break;
                    }
                    continue;
                }
                lx.expect_op("]")?;
                break;
            }
            Ok(Expr::List(items))
        }
        Tok::Op(op) if op == "{" => {
            let mut entries = Vec::new();
            if lx.eat_op("}") {
                return Ok(Expr::Map(entries));
            }
            loop {
                let k = parse_ternary(lx)?;
                lx.expect_op(":")?;
                let v = parse_ternary(lx)?;
                entries.push((k, v));
                if lx.eat_op(",") {
                    if lx.eat_op("}") {
                        break;
                    }
                    continue;
                }
                lx.expect_op("}")?;
                break;
            }
            Ok(Expr::Map(entries))
        }
        other => Err(format!("unexpected {other:?} in expression")),
    }
}

// --------------------------------------------------------------------------------- nodes

#[derive(Debug, Clone)]
enum Node {
    Text(String),
    Output(Expr),
    If(Vec<(Expr, Vec<Node>)>, Vec<Node>),
    For {
        names: Vec<String>,
        seq: Expr,
        body: Vec<Node>,
        or_else: Vec<Node>,
    },
    Set(Vec<String>, Expr),
    /// `{% set x %}...{% endset %}` captures rendered output into a variable.
    SetBlock(String, Vec<Node>),
}

pub struct Template {
    nodes: Vec<Node>,
}

struct Block {
    kind: String,
    body: String,
}

/// Split the source into literal text and `{{ }}` / `{% %}` blocks.
///
/// Whitespace handling matches how HuggingFace compiles chat templates, because that is
/// what the template authors tested against: `lstrip_blocks` drops the indentation in front
/// of a statement tag, `trim_blocks` drops the newline after one, and `{%-` / `-%}` strip
/// whitespace in either direction on top of that. Getting this wrong does not fail loudly;
/// it inserts stray blank lines into the prompt, which shifts the model off the format it
/// was trained on.
fn scan(src: &str) -> R<Vec<(Option<String>, Option<Block>)>> {
    let b = src.as_bytes();
    let mut out: Vec<(Option<String>, Option<Block>)> = Vec::new();
    let mut text = String::new();
    let mut i = 0usize;
    while i < b.len() {
        if b[i] == b'{' && i + 1 < b.len() && matches!(b[i + 1], b'{' | b'%' | b'#') {
            let kind_char = b[i + 1];
            let is_statement = kind_char != b'{';
            let close: &[u8] = match kind_char {
                b'{' => b"}}",
                b'%' => b"%}",
                _ => b"#}",
            };
            let mut j = i + 2;
            let trim_before = j < b.len() && b[j] == b'-';
            if trim_before {
                j += 1;
            }
            let Some(rel) = find(&b[j..], close) else {
                return Err("unterminated template tag".into());
            };
            let mut end = j + rel;
            let mut trim_after = false;
            if end > j && b[end - 1] == b'-' {
                trim_after = true;
                end -= 1;
            }
            let body = std::str::from_utf8(&b[j..end])
                .map_err(|_| "non-utf8 template tag")?
                .trim()
                .to_string();

            if trim_before {
                while text.ends_with(char::is_whitespace) {
                    text.pop();
                }
            } else if is_statement {
                // lstrip_blocks: indentation before a statement is layout, not content.
                while text.ends_with([' ', '\t']) {
                    text.pop();
                }
            }
            if !text.is_empty() {
                out.push((Some(std::mem::take(&mut text)), None));
            }
            if kind_char != b'#' {
                out.push((
                    None,
                    Some(Block {
                        kind: if is_statement { "stmt".into() } else { "out".into() },
                        body,
                    }),
                ));
            }

            i = j + rel + close.len();
            if trim_after {
                while i < b.len() && (b[i] as char).is_whitespace() {
                    i += 1;
                }
            } else if is_statement {
                // trim_blocks: a statement on its own line does not emit that line break.
                if i < b.len() && b[i] == b'\r' {
                    i += 1;
                }
                if i < b.len() && b[i] == b'\n' {
                    i += 1;
                }
            }
            continue;
        }
        let ch_len = src[i..].chars().next().map(char::len_utf8).unwrap_or(1);
        text.push_str(&src[i..i + ch_len]);
        i += ch_len;
    }
    if !text.is_empty() {
        out.push((Some(text), None));
    }
    Ok(out)
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

impl Template {
    pub fn parse(src: &str) -> Result<Self, Error> {
        // Jinja's `keep_trailing_newline` defaults to false, and templates are written
        // expecting that final newline to disappear.
        let src = src.strip_suffix('\u{000A}').unwrap_or(src);
        let parts = scan(src)?;
        let mut pos = 0usize;
        let nodes = parse_nodes(&parts, &mut pos, &[])?;
        Ok(Self { nodes })
    }

    pub fn render(&self, ctx: Value) -> Result<String, Error> {
        let mut scope = Scope::new(ctx);
        let mut out = String::new();
        exec(&self.nodes, &mut scope, &mut out)?;
        Ok(out)
    }
}

fn parse_nodes(
    parts: &[(Option<String>, Option<Block>)],
    pos: &mut usize,
    stop: &[&str],
) -> R<Vec<Node>> {
    let mut nodes = Vec::new();
    while *pos < parts.len() {
        let (text, block) = &parts[*pos];
        if let Some(t) = text {
            nodes.push(Node::Text(t.clone()));
            *pos += 1;
            continue;
        }
        let Some(blk) = block else {
            *pos += 1;
            continue;
        };
        if blk.kind == "out" {
            nodes.push(Node::Output(parse_expr_str(&blk.body)?));
            *pos += 1;
            continue;
        }

        let keyword = blk.body.split_whitespace().next().unwrap_or("").to_string();
        if stop.contains(&keyword.as_str()) {
            return Ok(nodes);
        }
        *pos += 1;
        let rest = blk.body[keyword.len()..].trim().to_string();

        match keyword.as_str() {
            "if" => {
                let mut branches = vec![(parse_expr_str(&rest)?, parse_nodes(parts, pos, &["elif", "else", "endif"])?)];
                let mut or_else = Vec::new();
                loop {
                    let Some((_, Some(b))) = parts.get(*pos) else { break };
                    let kw = b.body.split_whitespace().next().unwrap_or("");
                    match kw {
                        "elif" => {
                            let cond = b.body["elif".len()..].trim().to_string();
                            *pos += 1;
                            branches.push((
                                parse_expr_str(&cond)?,
                                parse_nodes(parts, pos, &["elif", "else", "endif"])?,
                            ));
                        }
                        "else" => {
                            *pos += 1;
                            or_else = parse_nodes(parts, pos, &["endif"])?;
                        }
                        "endif" => {
                            *pos += 1;
                            break;
                        }
                        other => return Err(format!("unexpected {other:?} inside if")),
                    }
                }
                nodes.push(Node::If(branches, or_else));
            }
            "for" => {
                let (names, seq) = split_for(&rest)?;
                let body = parse_nodes(parts, pos, &["else", "endfor"])?;
                let mut or_else = Vec::new();
                if let Some((_, Some(b))) = parts.get(*pos) {
                    if b.body.starts_with("else") {
                        *pos += 1;
                        or_else = parse_nodes(parts, pos, &["endfor"])?;
                    }
                }
                if let Some((_, Some(b))) = parts.get(*pos) {
                    if b.body.starts_with("endfor") {
                        *pos += 1;
                    }
                }
                nodes.push(Node::For { names, seq: parse_expr_str(&seq)?, body, or_else });
            }
            "set" => match rest.find('=') {
                Some(eq) if !rest[..eq].contains('(') => {
                    let targets = rest[..eq]
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect();
                    nodes.push(Node::Set(targets, parse_expr_str(rest[eq + 1..].trim())?));
                }
                _ => {
                    let body = parse_nodes(parts, pos, &["endset"])?;
                    if let Some((_, Some(b))) = parts.get(*pos) {
                        if b.body.starts_with("endset") {
                            *pos += 1;
                        }
                    }
                    nodes.push(Node::SetBlock(rest.trim().to_string(), body));
                }
            },
            // Constructs that carry no meaning for prompt rendering.
            "generation" | "endgeneration" => {}
            other => {
                return Err(format!(
                    "unsupported template construct {other:?}"
                ))
            }
        }
    }
    Ok(nodes)
}

fn split_for(rest: &str) -> R<(Vec<String>, String)> {
    let idx = find_keyword(rest, "in").ok_or_else(|| "for loop without `in`".to_string())?;
    let names = rest[..idx]
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    Ok((names, rest[idx + 2..].trim().to_string()))
}

/// Find a bare `in` keyword, not one embedded in an identifier or a string.
fn find_keyword(hay: &str, kw: &str) -> Option<usize> {
    let b: Vec<char> = hay.chars().collect();
    let k: Vec<char> = kw.chars().collect();
    let mut quote: Option<char> = None;
    let mut byte = 0usize;
    for i in 0..b.len() {
        let c = b[i];
        match quote {
            Some(q) => {
                if c == q {
                    quote = None;
                }
            }
            None => {
                if c == '"' || c == '\'' {
                    quote = Some(c);
                } else if b[i..].starts_with(k.as_slice()) {
                    let before_ok = i == 0 || !(b[i - 1].is_alphanumeric() || b[i - 1] == '_');
                    let after = b.get(i + k.len());
                    let after_ok = after.map_or(true, |c| !(c.is_alphanumeric() || *c == '_'));
                    if before_ok && after_ok {
                        return Some(byte);
                    }
                }
            }
        }
        byte += c.len_utf8();
    }
    None
}

// ---------------------------------------------------------------------------- evaluation

struct Scope {
    frames: Vec<BTreeMap<String, Value>>,
}

impl Scope {
    fn new(ctx: Value) -> Self {
        let mut root = BTreeMap::new();
        if let Value::Map(m) = ctx {
            root.extend(m);
        }
        Self { frames: vec![root] }
    }

    fn get(&self, name: &str) -> Option<&Value> {
        self.frames.iter().rev().find_map(|f| f.get(name))
    }

    /// Assignment writes to the frame that already defines the name, so a `{% set %}`
    /// inside a loop updates the outer variable the way templates expect.
    fn set(&mut self, name: &str, v: Value) {
        for f in self.frames.iter_mut().rev() {
            if f.contains_key(name) {
                f.insert(name.to_string(), v);
                return;
            }
        }
        self.frames.last_mut().unwrap().insert(name.to_string(), v);
    }

    fn declare(&mut self, name: &str, v: Value) {
        self.frames.last_mut().unwrap().insert(name.to_string(), v);
    }

    fn push(&mut self) {
        self.frames.push(BTreeMap::new());
    }

    fn pop(&mut self) {
        self.frames.pop();
    }
}

fn exec(nodes: &[Node], scope: &mut Scope, out: &mut String) -> R<()> {
    for node in nodes {
        match node {
            Node::Text(t) => out.push_str(t),
            Node::Output(e) => out.push_str(&eval(e, scope)?.to_text()),
            Node::If(branches, or_else) => {
                let mut done = false;
                for (cond, body) in branches {
                    if eval(cond, scope)?.truthy() {
                        exec(body, scope, out)?;
                        done = true;
                        break;
                    }
                }
                if !done {
                    exec(or_else, scope, out)?;
                }
            }
            Node::For { names, seq, body, or_else } => {
                let items = iterate(&eval(seq, scope)?);
                if items.is_empty() {
                    exec(or_else, scope, out)?;
                    continue;
                }
                let n = items.len();
                scope.push();
                for (i, item) in items.into_iter().enumerate() {
                    bind_loop_vars(scope, names, item);
                    let mut loop_map = BTreeMap::new();
                    loop_map.insert("index0".into(), Value::Num(i as f64));
                    loop_map.insert("index".into(), Value::Num((i + 1) as f64));
                    loop_map.insert("revindex".into(), Value::Num((n - i) as f64));
                    loop_map.insert("revindex0".into(), Value::Num((n - i - 1) as f64));
                    loop_map.insert("first".into(), Value::Bool(i == 0));
                    loop_map.insert("last".into(), Value::Bool(i + 1 == n));
                    loop_map.insert("length".into(), Value::Num(n as f64));
                    scope.declare("loop", Value::Map(loop_map));
                    exec(body, scope, out)?;
                }
                scope.pop();
            }
            Node::Set(names, e) => {
                let v = eval(e, scope)?;
                if names.len() == 1 {
                    assign(scope, &names[0], v)?;
                } else {
                    let items = iterate(&v);
                    for (i, name) in names.iter().enumerate() {
                        assign(scope, name, items.get(i).cloned().unwrap_or(Value::Null))?;
                    }
                }
            }
            Node::SetBlock(name, body) => {
                let mut captured = String::new();
                exec(body, scope, &mut captured)?;
                assign(scope, name, Value::Str(captured))?;
            }
        }
    }
    Ok(())
}

/// `{% set ns.field = x %}` mutates a namespace object rather than creating a variable.
fn assign(scope: &mut Scope, target: &str, v: Value) -> R<()> {
    match target.split_once('.') {
        Some((obj, field)) => {
            let obj = obj.trim();
            let field = field.trim().to_string();
            let mut current = match scope.get(obj) {
                Some(Value::Map(m)) => m.clone(),
                _ => BTreeMap::new(),
            };
            current.insert(field, v);
            scope.set(obj, Value::Map(current));
        }
        None => scope.set(target.trim(), v),
    }
    Ok(())
}

fn bind_loop_vars(scope: &mut Scope, names: &[String], item: Value) {
    if names.len() == 1 {
        scope.declare(&names[0], item);
        return;
    }
    let parts = iterate(&item);
    for (i, name) in names.iter().enumerate() {
        scope.declare(name, parts.get(i).cloned().unwrap_or(Value::Null));
    }
}

fn iterate(v: &Value) -> Vec<Value> {
    match v {
        Value::List(items) => items.clone(),
        Value::Map(m) => m
            .iter()
            .map(|(k, val)| Value::List(vec![Value::Str(k.clone()), val.clone()]))
            .collect(),
        Value::Str(s) => s.chars().map(|c| Value::Str(c.to_string())).collect(),
        Value::Null => Vec::new(),
        other => vec![other.clone()],
    }
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Num(n) => *n,
        Value::Bool(b) => *b as i32 as f64,
        Value::Str(s) => s.trim().parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn eval(e: &Expr, scope: &mut Scope) -> R<Value> {
    Ok(match e {
        Expr::Lit(v) => v.clone(),
        Expr::Var(name) => scope.get(name).cloned().unwrap_or(Value::Null),
        Expr::List(items) => Value::List(items.iter().map(|i| eval(i, scope)).collect::<R<_>>()?),
        Expr::Map(entries) => {
            let mut m = BTreeMap::new();
            for (k, v) in entries {
                m.insert(eval(k, scope)?.to_text(), eval(v, scope)?);
            }
            Value::Map(m)
        }
        Expr::Attr(base, name) => {
            let b = eval(base, scope)?;
            attr(&b, name)
        }
        Expr::Index(base, idx) => {
            let b = eval(base, scope)?;
            let i = eval(idx, scope)?;
            match (&b, &i) {
                (Value::Map(m), _) => m.get(&i.to_text()).cloned().unwrap_or(Value::Null),
                (Value::List(items), Value::Num(n)) => {
                    let n = *n as i64;
                    let idx = if n < 0 { items.len() as i64 + n } else { n };
                    items.get(idx.max(0) as usize).cloned().unwrap_or(Value::Null)
                }
                (Value::Str(s), Value::Num(n)) => {
                    let chars: Vec<char> = s.chars().collect();
                    let n = *n as i64;
                    let idx = if n < 0 { chars.len() as i64 + n } else { n };
                    chars
                        .get(idx.max(0) as usize)
                        .map(|c| Value::Str(c.to_string()))
                        .unwrap_or(Value::Null)
                }
                _ => Value::Null,
            }
        }
        Expr::Unary(op, inner) => {
            let v = eval(inner, scope)?;
            match *op {
                "not" => Value::Bool(!v.truthy()),
                "-" => Value::Num(-num(&v)),
                _ => Value::Null,
            }
        }
        Expr::Binary(op, l, r) => {
            // Short-circuit before evaluating the right side.
            match *op {
                "and" => {
                    let lv = eval(l, scope)?;
                    return Ok(if lv.truthy() { eval(r, scope)? } else { lv });
                }
                "or" => {
                    let lv = eval(l, scope)?;
                    return Ok(if lv.truthy() { lv } else { eval(r, scope)? });
                }
                _ => {}
            }
            let lv = eval(l, scope)?;
            let rv = eval(r, scope)?;
            binary(op, &lv, &rv)
        }
        Expr::Ternary(cond, then, other) => {
            if eval(cond, scope)?.truthy() {
                eval(then, scope)?
            } else {
                eval(other, scope)?
            }
        }
        Expr::Test(base, name, negated, arg) => {
            let v = eval(base, scope)?;
            let defined = match &**base {
                Expr::Var(n) => scope.get(n).is_some(),
                _ => !matches!(v, Value::Null),
            };
            let a = match arg {
                Some(a) => Some(eval(a, scope)?),
                None => None,
            };
            let res = match name.as_str() {
                "defined" => defined,
                "undefined" => !defined,
                "none" | "null" => matches!(v, Value::Null),
                "string" => matches!(v, Value::Str(_)),
                "number" | "integer" | "float" => matches!(v, Value::Num(_)),
                "boolean" => matches!(v, Value::Bool(_)),
                "mapping" => matches!(v, Value::Map(_)),
                "sequence" | "iterable" => matches!(v, Value::List(_) | Value::Str(_) | Value::Map(_)),
                "true" => matches!(v, Value::Bool(true)),
                "false" => matches!(v, Value::Bool(false)),
                "equalto" | "eq" | "sameas" => a.as_ref().map_or(false, |a| &v == a),
                "in" => a.as_ref().map_or(false, |a| iterate(a).contains(&v)),
                other => return Err(format!("unsupported test `is {other}`")),
            };
            Value::Bool(res ^ negated)
        }
        Expr::Filter(base, name, args) => {
            let v = eval(base, scope)?;
            let a: Vec<Value> = args.iter().map(|x| eval(x, scope)).collect::<R<_>>()?;
            filter(name, &v, &a)?
        }
        Expr::Call(callee, args, kwargs) => {
            let a: Vec<Value> = args.iter().map(|x| eval(x, scope)).collect::<R<_>>()?;
            match &**callee {
                Expr::Attr(base, method) => {
                    let recv = eval(base, scope)?;
                    let result = call_method(&recv, method, &a)?;
                    // `.append()` mutates in place, which templates rely on for building
                    // up a list before rendering it.
                    if matches!(method.as_str(), "append" | "extend" | "add") {
                        if let Expr::Var(n) = &**base {
                            scope.set(n, result.clone());
                        }
                        return Ok(Value::Null);
                    }
                    result
                }
                Expr::Var(name) => call_global(name, &a, kwargs, scope)?,
                other => return Err(format!("cannot call {other:?}")),
            }
        }
    })
}

fn attr(base: &Value, name: &str) -> Value {
    match base {
        Value::Map(m) => m.get(name).cloned().unwrap_or(Value::Null),
        Value::List(items) if name == "length" => Value::Num(items.len() as f64),
        _ => Value::Null,
    }
}

fn binary(op: &str, l: &Value, r: &Value) -> Value {
    match op {
        "+" => match (l, r) {
            (Value::Str(a), _) => Value::Str(format!("{a}{}", r.to_text())),
            (_, Value::Str(b)) => Value::Str(format!("{}{b}", l.to_text())),
            (Value::List(a), Value::List(b)) => {
                let mut v = a.clone();
                v.extend(b.clone());
                Value::List(v)
            }
            _ => Value::Num(num(l) + num(r)),
        },
        "~" => Value::Str(format!("{}{}", l.to_text(), r.to_text())),
        "-" => Value::Num(num(l) - num(r)),
        "*" => Value::Num(num(l) * num(r)),
        "/" => Value::Num(num(l) / num(r)),
        "//" => Value::Num((num(l) / num(r)).floor()),
        "%" => Value::Num(num(l) % num(r)),
        "==" => Value::Bool(l == r),
        "!=" => Value::Bool(l != r),
        "<" => Value::Bool(num(l) < num(r)),
        ">" => Value::Bool(num(l) > num(r)),
        "<=" => Value::Bool(num(l) <= num(r)),
        ">=" => Value::Bool(num(l) >= num(r)),
        "in" => Value::Bool(match r {
            Value::Str(s) => s.contains(&l.to_text()),
            Value::Map(m) => m.contains_key(&l.to_text()),
            other => iterate(other).contains(l),
        }),
        _ => Value::Null,
    }
}

fn slice(v: &Value, args: &[Value]) -> Value {
    let items: Vec<Value> = match v {
        Value::List(l) => l.clone(),
        Value::Str(s) => s.chars().map(|c| Value::Str(c.to_string())).collect(),
        _ => return Value::Null,
    };
    let len = items.len() as i64;
    let is_str = matches!(v, Value::Str(_));

    let given = |a: Option<&Value>| -> Option<i64> {
        match a {
            None | Some(Value::Null) => None,
            Some(x) => Some(num(x) as i64),
        }
    };
    let step = given(args.get(2)).unwrap_or(1);
    if step == 0 || len == 0 {
        return if is_str { Value::Str(String::new()) } else { Value::List(Vec::new()) };
    }

    // Normalize a bound the way Python does: negative counts from the end, and the clamp
    // range differs by direction because a backwards walk may legitimately end at -1.
    let norm = |raw: i64, lo: i64, hi: i64| -> i64 {
        let v = if raw < 0 { len + raw } else { raw };
        v.clamp(lo, hi)
    };

    let mut out = Vec::new();
    if step > 0 {
        let start = given(args.first()).map(|r| norm(r, 0, len)).unwrap_or(0);
        let end = given(args.get(1)).map(|r| norm(r, 0, len)).unwrap_or(len);
        let mut i = start;
        while i < end {
            out.push(items[i as usize].clone());
            i += step;
        }
    } else {
        let start = given(args.first()).map(|r| norm(r, -1, len - 1)).unwrap_or(len - 1);
        let end = given(args.get(1)).map(|r| norm(r, -1, len - 1)).unwrap_or(-1);
        let mut i = start;
        while i > end {
            if i >= 0 && i < len {
                out.push(items[i as usize].clone());
            }
            i += step;
        }
    }

    if is_str {
        Value::Str(out.iter().map(Value::to_text).collect())
    } else {
        Value::List(out)
    }
}

fn filter(name: &str, v: &Value, args: &[Value]) -> R<Value> {
    Ok(match name {
        "__slice" => slice(v, args),
        "trim" | "strip" => Value::Str(v.to_text().trim().to_string()),
        "length" | "count" => Value::Num(match v {
            Value::List(l) => l.len() as f64,
            Value::Map(m) => m.len() as f64,
            Value::Str(s) => s.chars().count() as f64,
            _ => 0.0,
        }),
        "default" | "d" => {
            let use_default = match args.get(1) {
                Some(b) if b.truthy() => !v.truthy(),
                _ => matches!(v, Value::Null),
            };
            if use_default {
                args.first().cloned().unwrap_or(Value::Null)
            } else {
                v.clone()
            }
        }
        "tojson" | "to_json" => Value::Str(v.to_json()),
        "join" => {
            let sep = args.first().map(Value::to_text).unwrap_or_default();
            Value::Str(iterate(v).iter().map(Value::to_text).collect::<Vec<_>>().join(&sep))
        }
        "first" => iterate(v).first().cloned().unwrap_or(Value::Null),
        "last" => iterate(v).last().cloned().unwrap_or(Value::Null),
        "upper" => Value::Str(v.to_text().to_uppercase()),
        "lower" => Value::Str(v.to_text().to_lowercase()),
        "capitalize" => {
            let s = v.to_text();
            let mut c = s.chars();
            Value::Str(match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
                None => String::new(),
            })
        }
        "title" => Value::Str(
            v.to_text()
                .split(' ')
                .map(|w| {
                    let mut c = w.chars();
                    match c.next() {
                        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                        None => String::new(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
        "list" => Value::List(iterate(v)),
        "reverse" => {
            let mut items = iterate(v);
            items.reverse();
            Value::List(items)
        }
        "string" | "str" | "safe" | "e" | "escape" | "forceescape" => Value::Str(v.to_text()),
        "int" => Value::Num(num(v).trunc()),
        "float" => Value::Num(num(v)),
        "abs" => Value::Num(num(v).abs()),
        "replace" => {
            let from = args.first().map(Value::to_text).unwrap_or_default();
            let to = args.get(1).map(Value::to_text).unwrap_or_default();
            Value::Str(v.to_text().replace(&from, &to))
        }
        "items" | "dictsort" => Value::List(iterate(v)),
        "indent" => v.clone(),
        "map" | "attr" => {
            let key = args.first().map(Value::to_text).unwrap_or_default();
            Value::List(iterate(v).iter().map(|item| attr(item, &key)).collect())
        }
        "selectattr" | "rejectattr" => {
            let key = args.first().map(Value::to_text).unwrap_or_default();
            let want = name == "selectattr";
            Value::List(
                iterate(v)
                    .into_iter()
                    .filter(|item| attr(item, &key).truthy() == want)
                    .collect(),
            )
        }
        other => return Err(format!("unsupported filter `{other}`")),
    })
}

fn call_method(recv: &Value, method: &str, args: &[Value]) -> R<Value> {
    let text = recv.to_text();
    Ok(match method {
        "strip" => Value::Str(text.trim().to_string()),
        "lstrip" => Value::Str(text.trim_start().to_string()),
        "rstrip" => Value::Str(text.trim_end().to_string()),
        "upper" => Value::Str(text.to_uppercase()),
        "lower" => Value::Str(text.to_lowercase()),
        "split" => {
            let parts: Vec<Value> = match args.first() {
                Some(sep) => text.split(&sep.to_text()).map(Value::str).collect(),
                None => text.split_whitespace().map(Value::str).collect(),
            };
            Value::List(parts)
        }
        "startswith" => Value::Bool(text.starts_with(&args.first().map(Value::to_text).unwrap_or_default())),
        "endswith" => Value::Bool(text.ends_with(&args.first().map(Value::to_text).unwrap_or_default())),
        "replace" => filter("replace", recv, args)?,
        "join" => {
            let items = args.first().cloned().unwrap_or(Value::Null);
            Value::Str(iterate(&items).iter().map(Value::to_text).collect::<Vec<_>>().join(&text))
        }
        "items" => Value::List(iterate(recv)),
        "keys" => match recv {
            Value::Map(m) => Value::List(m.keys().cloned().map(Value::Str).collect()),
            _ => Value::List(Vec::new()),
        },
        "values" => match recv {
            Value::Map(m) => Value::List(m.values().cloned().collect()),
            _ => Value::List(Vec::new()),
        },
        "get" => match recv {
            Value::Map(m) => m
                .get(&args.first().map(Value::to_text).unwrap_or_default())
                .cloned()
                .unwrap_or_else(|| args.get(1).cloned().unwrap_or(Value::Null)),
            _ => Value::Null,
        },
        "append" | "add" => {
            let mut items = iterate(recv);
            items.push(args.first().cloned().unwrap_or(Value::Null));
            Value::List(items)
        }
        "extend" => {
            let mut items = iterate(recv);
            items.extend(args.first().map(iterate).unwrap_or_default());
            Value::List(items)
        }
        other => return Err(format!("unsupported method `.{other}()`")),
    })
}

fn call_global(name: &str, args: &[Value], kwargs: &[(String, Expr)], scope: &mut Scope) -> R<Value> {
    Ok(match name {
        // Templates use this to reject inputs they cannot represent. Surfacing it as an
        // error is the point: the caller falls back rather than sending a broken prompt.
        "raise_exception" => {
            return Err(args.first().map(Value::to_text).unwrap_or_else(|| "template raised".into()))
        }
        "namespace" => {
            let mut m = BTreeMap::new();
            for (k, e) in kwargs {
                m.insert(k.clone(), eval(e, scope)?);
            }
            Value::Map(m)
        }
        "range" => {
            let (start, end, step) = match args.len() {
                0 => (0.0, 0.0, 1.0),
                1 => (0.0, num(&args[0]), 1.0),
                2 => (num(&args[0]), num(&args[1]), 1.0),
                _ => (num(&args[0]), num(&args[1]), num(&args[2]).max(1.0)),
            };
            let mut out = Vec::new();
            let mut v = start;
            while v < end {
                out.push(Value::Num(v));
                v += step;
            }
            Value::List(out)
        }
        "length" | "len" => filter("length", args.first().unwrap_or(&Value::Null), &[])?,
        "string" | "str" => Value::Str(args.first().map(Value::to_text).unwrap_or_default()),
        // Real time is not available to a deterministic renderer, and templates only use
        // this for a date in a system prompt.
        "strftime_now" => Value::Str(String::new()),
        other => return Err(format!("unsupported function `{other}()`")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(pairs: &[(&str, &str)]) -> Value {
        Value::List(
            pairs
                .iter()
                .map(|(r, c)| {
                    let mut m = BTreeMap::new();
                    m.insert("role".into(), Value::str(*r));
                    m.insert("content".into(), Value::str(*c));
                    Value::Map(m)
                })
                .collect(),
        )
    }

    fn ctx(pairs: &[(&str, &str)], gen: bool) -> Value {
        let mut m = BTreeMap::new();
        m.insert("messages".into(), msgs(pairs));
        m.insert("add_generation_prompt".into(), Value::Bool(gen));
        m.insert("bos_token".into(), Value::str("<s>"));
        m.insert("eos_token".into(), Value::str("</s>"));
        Value::Map(m)
    }

    #[test]
    fn renders_chatml() {
        let src = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + \
                   message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}\
                   {% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
        let out = Template::parse(src)
            .unwrap()
            .render(ctx(&[("user", "hi")], true))
            .unwrap();
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn handles_whitespace_control_and_loop_vars() {
        let src = "{%- for m in messages -%}\n  {{ loop.index }}:{{ m.content }}\n{%- endfor -%}";
        let out = Template::parse(src)
            .unwrap()
            .render(ctx(&[("user", "a"), ("assistant", "b")], false))
            .unwrap();
        assert_eq!(out, "1:a2:b");
    }

    #[test]
    fn supports_namespace_mutation() {
        let src = "{% set ns = namespace(n=0) %}{% for m in messages %}\
                   {% set ns.n = ns.n + 1 %}{% endfor %}{{ ns.n }}";
        let out = Template::parse(src)
            .unwrap()
            .render(ctx(&[("user", "a"), ("assistant", "b")], false))
            .unwrap();
        assert_eq!(out, "2");
    }

    #[test]
    fn gemma_style_role_mapping() {
        let src = "{% for message in messages %}{% if message['role'] == 'assistant' %}\
                   {% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}\
                   {{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}\
                   {% endfor %}";
        let out = Template::parse(src)
            .unwrap()
            .render(ctx(&[("user", " hi "), ("assistant", "yo")], false))
            .unwrap();
        assert_eq!(
            out,
            "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\nyo<end_of_turn>\n"
        );
    }

    #[test]
    fn raise_exception_surfaces_as_an_error() {
        let src = "{{ raise_exception('nope') }}";
        assert_eq!(Template::parse(src).unwrap().render(ctx(&[], false)), Err("nope".into()));
    }

    #[test]
    fn conditionals_and_tests() {
        let src = "{% if tools is defined and tools %}T{% elif messages %}M{% else %}E{% endif %}";
        assert_eq!(
            Template::parse(src).unwrap().render(ctx(&[("user", "x")], false)).unwrap(),
            "M"
        );
    }
}

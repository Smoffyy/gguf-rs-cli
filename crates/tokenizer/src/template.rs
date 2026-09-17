//! Chat formatting.
//!
//! The GGUF's own Jinja template is used whenever it parses and renders, because that is
//! the format the model was actually trained on. The built-in formats exist only for files
//! that carry no template at all, or whose template uses a construct outside the subset -
//! in which case falling back to a recognised family beats emitting a malformed prompt.

use std::collections::BTreeMap;

use crate::jinja::{Template, Value};
use crate::vocab::Vocab;

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Builtin {
    ChatMl,
    Llama3,
    Llama2,
    Gemma,
    Phi3,
    Plain,
}

pub struct ChatTemplate {
    jinja: Option<Template>,
    /// Kept for diagnostics: why the built-in path is in use.
    pub jinja_error: Option<String>,
    pub builtin: Builtin,
    bos_token: String,
    eos_token: String,
    /// The template prepends BOS itself, so the tokenizer must not add a second one.
    pub template_adds_bos: bool,
    stop_ids: Vec<u32>,
}

impl ChatTemplate {
    pub fn from_vocab(v: &Vocab) -> Self {
        let bos_token = v.bos.map(|b| v.text(b).to_string()).unwrap_or_default();
        let eos_token = v.eos.first().map(|e| v.text(*e).to_string()).unwrap_or_default();

        let mut jinja_error = None;
        let jinja = match v.chat_template.as_deref() {
            Some(src) if !src.trim().is_empty() => match Template::parse(src) {
                Ok(t) => {
                    // Parsing is not enough: a template can parse and still hit an
                    // unsupported filter on the first render. Prove it works now, while
                    // there is still a chance to fall back.
                    let probe = probe_context(&bos_token, &eos_token);
                    match t.render(probe) {
                        Ok(_) => Some(t),
                        Err(e) => {
                            jinja_error = Some(e);
                            None
                        }
                    }
                }
                Err(e) => {
                    jinja_error = Some(e);
                    None
                }
            },
            _ => None,
        };

        let builtin = detect_builtin(v);
        let template_adds_bos = v
            .chat_template
            .as_deref()
            .map(|s| s.contains("bos_token"))
            .unwrap_or(false)
            && jinja.is_some();

        let mut stop_ids = v.eos.clone();
        for extra in [
            "<|im_end|>",
            "<|eot_id|>",
            "<|end_of_text|>",
            "<|endoftext|>",
            "<end_of_turn>",
            "<|end|>",
            "<|return|>",
            "<|eom_id|>",
        ] {
            if let Some(id) = v.id(extra) {
                stop_ids.push(id);
            }
        }
        stop_ids.sort_unstable();
        stop_ids.dedup();

        Self { jinja, jinja_error, builtin, bos_token, eos_token, template_adds_bos, stop_ids }
    }

    pub fn uses_gguf_template(&self) -> bool {
        self.jinja.is_some()
    }

    pub fn is_stop(&self, id: u32) -> bool {
        self.stop_ids.binary_search(&id).is_ok()
    }

    pub fn stop_ids(&self) -> &[u32] {
        &self.stop_ids
    }

    pub fn render(&self, messages: &[ChatMessage], add_generation_prompt: bool) -> String {
        self.render_with(messages, add_generation_prompt, None)
    }

    /// Render, optionally pinning `enable_thinking`.
    ///
    /// `None` leaves the variable undefined, which is what the reasoning models' templates
    /// treat as their default. Setting it is a deliberate override: Qwen-3's template, for
    /// one, emits an empty `<think></think>` pair when it is false, which suppresses
    /// reasoning entirely. Defaulting that on behalf of the user would quietly change what
    /// the model does.
    pub fn render_with(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        enable_thinking: Option<bool>,
    ) -> String {
        if let Some(t) = &self.jinja {
            let ctx = context(
                messages,
                add_generation_prompt,
                &self.bos_token,
                &self.eos_token,
                enable_thinking,
            );
            if let Ok(out) = t.render(ctx) {
                return out;
            }
        }
        self.render_builtin(messages, add_generation_prompt)
    }

    fn render_builtin(&self, messages: &[ChatMessage], add_gen: bool) -> String {
        let mut out = String::new();
        // Formats without a system role fold the system text into the first user turn.
        let system_role_supported = matches!(self.builtin, Builtin::ChatMl | Builtin::Llama3 | Builtin::Phi3);
        let mut pending_system = String::new();

        for m in messages {
            let role = if m.role == "developer" { "system" } else { m.role.as_str() };
            if role == "system" && !system_role_supported {
                if !pending_system.is_empty() {
                    pending_system.push('\n');
                }
                pending_system.push_str(&m.content);
                continue;
            }
            let merged;
            let content: &str = if !pending_system.is_empty() && role != "system" {
                merged = format!("{pending_system}\n\n{}", m.content);
                pending_system.clear();
                &merged
            } else {
                &m.content
            };
            match self.builtin {
                Builtin::ChatMl => out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n")),
                Builtin::Llama3 => out.push_str(&format!(
                    "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                )),
                Builtin::Phi3 => out.push_str(&format!("<|{role}|>\n{content}<|end|>\n")),
                Builtin::Gemma => {
                    let r = if role == "assistant" { "model" } else { "user" };
                    out.push_str(&format!("<start_of_turn>{r}\n{content}<end_of_turn>\n"));
                }
                Builtin::Llama2 => {
                    if role == "user" {
                        out.push_str(&format!("[INST] {content} [/INST]"));
                    } else {
                        out.push_str(&format!(" {content} "));
                    }
                }
                Builtin::Plain => {
                    let label = match role {
                        "user" => "User",
                        "assistant" => "Assistant",
                        _ => "System",
                    };
                    out.push_str(&format!("{label}: {content}\n"));
                }
            }
        }

        if add_gen {
            out.push_str(match self.builtin {
                Builtin::ChatMl => "<|im_start|>assistant\n",
                Builtin::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
                Builtin::Gemma => "<start_of_turn>model\n",
                Builtin::Phi3 => "<|assistant|>\n",
                Builtin::Llama2 => "",
                Builtin::Plain => "Assistant:",
            });
        }
        out
    }
}

fn detect_builtin(v: &Vocab) -> Builtin {
    if let Some(t) = v.chat_template.as_deref() {
        if t.contains("<|im_start|>") {
            return Builtin::ChatMl;
        }
        if t.contains("<|start_header_id|>") {
            return Builtin::Llama3;
        }
        if t.contains("<start_of_turn>") {
            return Builtin::Gemma;
        }
        if t.contains("<|user|>") {
            return Builtin::Phi3;
        }
        if t.contains("[INST]") {
            return Builtin::Llama2;
        }
    }
    for (probe, builtin) in [
        ("<|im_start|>", Builtin::ChatMl),
        ("<|eot_id|>", Builtin::Llama3),
        ("<start_of_turn>", Builtin::Gemma),
        ("<|user|>", Builtin::Phi3),
        ("[INST]", Builtin::Llama2),
    ] {
        if v.id(probe).is_some() {
            return builtin;
        }
    }
    Builtin::Plain
}

fn message_value(m: &ChatMessage) -> Value {
    let mut map = BTreeMap::new();
    map.insert("role".to_string(), Value::str(m.role.clone()));
    map.insert("content".to_string(), Value::str(m.content.clone()));
    Value::Map(map)
}

fn context(
    messages: &[ChatMessage],
    add_gen: bool,
    bos: &str,
    eos: &str,
    enable_thinking: Option<bool>,
) -> Value {
    let mut ctx = BTreeMap::new();
    ctx.insert(
        "messages".to_string(),
        Value::List(messages.iter().map(message_value).collect()),
    );
    ctx.insert("add_generation_prompt".to_string(), Value::Bool(add_gen));
    ctx.insert("bos_token".to_string(), Value::str(bos));
    ctx.insert("eos_token".to_string(), Value::str(eos));
    // Several templates branch on these even when no tools are in play, and a missing
    // name would otherwise render as empty rather than taking the false branch.
    ctx.insert("tools".to_string(), Value::Null);
    ctx.insert("tools_in_user_message".to_string(), Value::Bool(false));
    ctx.insert("add_vision_id".to_string(), Value::Bool(false));
    // Left undefined unless asked for, so the model's own default applies.
    if let Some(t) = enable_thinking {
        ctx.insert("enable_thinking".to_string(), Value::Bool(t));
    }
    Value::Map(ctx)
}

/// A minimal conversation used to check a template actually renders.
///
/// Deliberately no system turn: several templates (Gemma's among them) call
/// `raise_exception` on a system role by design, and treating that as a broken template
/// would push a perfectly good model onto the built-in path.
fn probe_context(bos: &str, eos: &str) -> Value {
    context(
        &[ChatMessage::new("user", "u"), ChatMessage::new("assistant", "a")],
        true,
        bos,
        eos,
        None,
    )
}

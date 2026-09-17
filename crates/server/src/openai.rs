//! OpenAI-compatible request and response shapes.
//!
//! Only the fields a local engine can honour are read; the rest are accepted and ignored so
//! that clients written against the hosted API work without modification. Rejecting an
//! unknown field would break more clients than it would help.

use serde::{Deserialize, Serialize};

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub role: String,
    /// Content is a string in the common case and a list of parts for multimodal clients;
    /// the text parts are all this engine can use.
    #[serde(default)]
    pub content: Option<Content>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentPart {
    #[serde(default)]
    pub text: Option<String>,
}

impl Content {
    pub fn text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// Stop may be a single string or a list.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s],
            Self::Many(v) => v,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// The newer name for `max_tokens`.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: OutMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct OutMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

impl ChatResponse {
    pub fn new(model: &str, content: String, finish: &'static str, usage: Usage) -> Self {
        Self {
            id: new_id("chatcmpl"),
            object: "chat.completion",
            created: now(),
            model: model.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: OutMessage { role: "assistant", content },
                finish_reason: finish,
            }],
            usage,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

impl ChatChunk {
    pub fn new(id: &str, model: &str, delta: Delta, finish: Option<&'static str>) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk",
            created: now(),
            model: model.to_string(),
            choices: vec![ChunkChoice { index: 0, delta, finish_reason: finish }],
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

impl CompletionResponse {
    pub fn new(model: &str, text: String, finish: &'static str, usage: Usage) -> Self {
        Self {
            id: new_id("cmpl"),
            object: "text_completion",
            created: now(),
            model: model.to_string(),
            choices: vec![CompletionChoice { index: 0, text, finish_reason: finish }],
            usage,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelInfo>,
}

impl ModelList {
    pub fn new(names: Vec<String>) -> Self {
        Self {
            object: "list",
            data: names
                .into_iter()
                .map(|id| ModelInfo { id, object: "model", created: now(), owned_by: "gguf-rs" })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorBody {
    pub message: String,
    pub r#type: &'static str,
}

impl ApiError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            error: ApiErrorBody { message: message.into(), r#type: "invalid_request_error" },
        }
    }
}

pub fn new_chunk_id() -> String {
    new_id("chatcmpl")
}

pub fn finish_reason(reason: gguf_runtime::StopReason) -> &'static str {
    use gguf_runtime::StopReason::*;
    match reason {
        EndOfGeneration | StopString => "stop",
        MaxTokens | ContextFull => "length",
        Cancelled => "stop",
    }
}

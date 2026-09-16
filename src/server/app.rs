use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};
use axum::{
    extract::State, response::sse::{Event, Sse}, response::{IntoResponse, Response},
    routing::{get, post}, Json, Router,
};
use futures::stream::{self, Stream};
use futures::StreamExt;
use tower_http::cors::CorsLayer;

use super::openai::*;
use super::worker::{ChatEvent, ChatRequest, WorkerHandle};
use crate::sampler::SampleParams;

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
}

pub fn build(worker: WorkerHandle) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(worker)
}

async fn list_models(State(worker): State<WorkerHandle>) -> Json<ModelsResponse> {
    let mut ids: Vec<&String> = worker.registry.keys().collect();
    ids.sort();
    let data = ids.into_iter().map(|id| ModelInfo {
        id: id.clone(), object: "model", created: now(), owned_by: "gguf-rs",
    }).collect();
    Json(ModelsResponse { object: "list", data })
}

fn error_response(status: axum::http::StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(msg, "invalid_request_error"))).into_response()
}

async fn chat_completions(
    State(worker): State<WorkerHandle>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let Some(preset) = worker.registry.get(&req.model).cloned() else {
        return error_response(axum::http::StatusCode::NOT_FOUND,
            format!("model '{}' not found", req.model));
    };

    let messages: Vec<(String, String)> = req.messages.iter()
        .map(|m| (m.role.clone(), m.content.as_ref().map(|c| c.as_text()).unwrap_or_default()))
        .collect();
    if messages.is_empty() {
        return error_response(axum::http::StatusCode::BAD_REQUEST, "messages must not be empty");
    }

    let sample = SampleParams {
        temperature: req.temperature.unwrap_or(preset.sample.temperature),
        top_k: req.top_k.unwrap_or(preset.sample.top_k),
        top_p: req.top_p.unwrap_or(preset.sample.top_p),
        min_p: req.min_p.unwrap_or(preset.sample.min_p),
        rep_penalty: req.repeat_penalty.unwrap_or(preset.sample.rep_penalty),
        presence_penalty: req.presence_penalty.unwrap_or(preset.sample.presence_penalty),
        frequency_penalty: req.frequency_penalty.unwrap_or(preset.sample.frequency_penalty),
        mirostat: preset.sample.mirostat,
        mirostat_tau: preset.sample.mirostat_tau,
        mirostat_eta: preset.sample.mirostat_eta,
    };
    let max_tokens = req.max_tokens.or(req.max_completion_tokens).unwrap_or(512);
    let stop_strings = match req.stop {
        Some(StopSeq::One(s)) => vec![s],
        Some(StopSeq::Many(v)) => v,
        None => vec![],
    };

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ChatEvent>();
    worker.submit(ChatRequest {
        model_id: req.model.clone(), messages, max_tokens, sample, stop_strings, tx,
    });

    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_name = req.model.clone();
    let created = now();

    if req.stream {
        let first_id = id.clone();
        let stream = stream::unfold((rx, false), move |(mut rx, done)| {
            let id = id.clone(); let model_name = model_name.clone();
            async move {
                if done { return None; }
                match rx.recv().await {
                    Some(ChatEvent::Token(text)) => {
                        let chunk = ChatCompletionChunk {
                            id: id.clone(), object: "chat.completion.chunk", created, model: model_name.clone(),
                            choices: vec![ChunkChoice { index: 0,
                                delta: Delta { role: None, content: Some(text) }, finish_reason: None }],
                        };
                        let ev = Event::default().data(serde_json::to_string(&chunk).unwrap());
                        Some((Ok::<_, Infallible>(ev), (rx, false)))
                    }
                    Some(ChatEvent::Done { finish_reason, .. }) => {
                        let chunk = ChatCompletionChunk {
                            id: id.clone(), object: "chat.completion.chunk", created, model: model_name.clone(),
                            choices: vec![ChunkChoice { index: 0,
                                delta: Delta::default(), finish_reason: Some(finish_reason) }],
                        };
                        let ev = Event::default().data(serde_json::to_string(&chunk).unwrap());
                        Some((Ok(ev), (rx, true)))
                    }
                    Some(ChatEvent::Error(msg)) => {
                        let ev = Event::default().event("error").data(msg);
                        Some((Ok(ev), (rx, true)))
                    }
                    None => None,
                }
            }
        });
        let first = Event::default().data(serde_json::to_string(&ChatCompletionChunk {
            id: first_id, object: "chat.completion.chunk", created, model: req.model.clone(),
            choices: vec![ChunkChoice { index: 0,
                delta: Delta { role: Some("assistant"), content: None }, finish_reason: None }],
        }).unwrap());
        let done_marker = stream::once(async { Ok::<_, Infallible>(Event::default().data("[DONE]")) });
        let full = stream::once(async move { Ok::<_, Infallible>(first) }).chain(stream).chain(done_marker);
        sse_response(full)
    } else {
        let mut content = String::new();
        let mut finish_reason = "stop".to_string();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        loop {
            match rx.recv().await {
                Some(ChatEvent::Token(text)) => content.push_str(&text),
                Some(ChatEvent::Done { finish_reason: fr, prompt_tokens: pt, completion_tokens: ct }) => {
                    finish_reason = fr; prompt_tokens = pt; completion_tokens = ct;
                    break;
                }
                Some(ChatEvent::Error(msg)) => {
                    return error_response(axum::http::StatusCode::INTERNAL_SERVER_ERROR, msg);
                }
                None => break,
            }
        }
        Json(ChatCompletionResponse {
            id, object: "chat.completion", created, model: req.model,
            choices: vec![Choice { index: 0,
                message: OutMessage { role: "assistant", content }, finish_reason }],
            usage: Usage { prompt_tokens, completion_tokens, total_tokens: prompt_tokens + completion_tokens },
        }).into_response()
    }
}

fn sse_response<S>(stream: S) -> Response
where S: Stream<Item = Result<Event, Infallible>> + Send + 'static {
    Sse::new(stream).into_response()
}

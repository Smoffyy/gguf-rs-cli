//! OpenAI-compatible HTTP server.
//!
//! Everything stays local: no telemetry, no model download, no outbound connection of any
//! kind. The server binds a port, reads GGUF files off disk, and answers.

pub mod config;
mod openai;
mod worker;

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use gguf_sample::SampleParams;
use gguf_tokenizer::ChatMessage;

pub use config::{Registry, EXAMPLE};
use openai::*;
use worker::{Command, Event, Job, JobInput, Status};

pub struct ServeConfig {
    pub config: Option<String>,
    pub model: Option<String>,
    pub host: String,
    pub port: u16,
    pub parallel: usize,
    pub device: String,
    pub ctx: usize,
    pub threads: usize,
}

/// The worker owns the model and runs on its own thread; this channel is the only way in.
/// A `Sender` is `Send` but not `Sync`, and axum needs shared state to be both.
#[derive(Clone)]
struct AppState {
    tx: Arc<Mutex<Sender<Command>>>,
    registry: Arc<Registry>,
}

impl AppState {
    fn submit(&self, job: Job) -> std::result::Result<(), Response> {
        let guard = self.tx.lock().map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError::new("the model worker is in a bad state")),
            )
                .into_response()
        })?;
        guard.send(Command::Run(Box::new(job))).map_err(|_| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiError::new("the model worker stopped")),
            )
                .into_response()
        })
    }
}

pub fn serve(cfg: ServeConfig) -> Result<()> {
    let registry = match (&cfg.config, &cfg.model) {
        (Some(path), _) => Registry::load(path)?,
        (None, Some(model)) => {
            Registry::single(model, &cfg.device, cfg.ctx, cfg.parallel, cfg.threads)?
        }
        (None, None) => anyhow::bail!(
            "nothing to serve: pass --model FILE for a single model, or --config FILE for a \
             registry.\n\nA registry looks like:\n\n{EXAMPLE}"
        ),
    };

    let (tx, rx) = channel::<Command>();
    let reg_for_worker = registry.clone();
    let worker = std::thread::Builder::new()
        .name("gguf-model".into())
        .spawn(move || worker::Worker::new(reg_for_worker).run(rx))
        .context("starting the model worker")?;

    let state = AppState {
        tx: Arc::new(Mutex::new(tx.clone())),
        registry: Arc::new(registry),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/status", get(status))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("starting the async runtime")?;

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .with_context(|| format!("binding {addr}"))?;
        eprintln!("gguf-rs serving on http://{addr}");
        eprintln!("  GET  /v1/models");
        eprintln!("  POST /v1/chat/completions");
        eprintln!("  POST /v1/completions");
        axum::serve(listener, app).await.context("serving")
    })?;

    let _ = tx.send(Command::Shutdown);
    let _ = worker.join();
    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(ModelList::new(state.registry.names()))
}

/// What the worker is doing: which model is resident, on which device, and how many
/// requests are in flight. Not part of the OpenAI API, but the first thing anyone asks
/// when a server feels slow.
async fn status(State(state): State<AppState>) -> Response {
    let (tx, rx) = channel::<Status>();
    let sent = state
        .tx
        .lock()
        .ok()
        .map(|g| g.send(Command::Status(tx)).is_ok())
        .unwrap_or(false);
    if !sent {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new("the model worker stopped")),
        )
            .into_response();
    }
    // The worker answers between scheduling rounds, so this waits at most one token.
    match tokio::task::block_in_place(|| rx.recv()) {
        Ok(s) => Json(serde_json::json!({
            "models": s.models,
            "loaded": s.loaded,
            "device": s.device,
            "active_requests": s.active,
        }))
        .into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiError::new("the model worker stopped")),
        )
            .into_response(),
    }
}

fn bad_request(msg: impl Into<String>) -> Response {
    (StatusCode::BAD_REQUEST, Json(ApiError::new(msg))).into_response()
}

/// Resolve the requested model, or the registry's default when the request names none.
fn resolve(state: &AppState, name: &Option<String>) -> std::result::Result<config::ResolvedModel, Response> {
    let chosen = name
        .clone()
        .or_else(|| state.registry.default_name())
        .ok_or_else(|| bad_request("the registry is empty"))?;
    state.registry.resolve(&chosen).ok_or_else(|| {
        bad_request(format!(
            "unknown model {chosen:?}; this server has {}",
            state.registry.names().join(", ")
        ))
    })
}

async fn chat_completions(State(state): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    let spec = match resolve(&state, &req.model) {
        Ok(s) => s,
        Err(r) => return r,
    };
    if req.messages.is_empty() {
        return bad_request("messages must not be empty");
    }

    let mut messages: Vec<ChatMessage> = Vec::with_capacity(req.messages.len() + 1);
    // A registry-level system prompt applies only when the request did not send one.
    if let Some(sys) = &spec.system {
        if !req.messages.iter().any(|m| m.role == "system") {
            messages.push(ChatMessage::new("system", sys.clone()));
        }
    }
    for m in &req.messages {
        messages.push(ChatMessage::new(
            m.role.clone(),
            m.content.as_ref().map(|c| c.text()).unwrap_or_default(),
        ));
    }

    let params = SampleParams {
        temperature: req.temperature.unwrap_or(spec.temperature),
        top_k: req.top_k.unwrap_or(spec.top_k),
        top_p: req.top_p.unwrap_or(spec.top_p),
        min_p: req.min_p.unwrap_or(spec.min_p),
        repeat_penalty: req.repeat_penalty.unwrap_or(spec.repeat_penalty),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        seed: req.seed.unwrap_or(0),
        ..Default::default()
    };

    let (events_tx, events_rx) = channel::<Event>();
    let job = Job {
        model: spec.name.clone(),
        // The worker renders with the model's own chat template, because the template lives
        // with the tokenizer, which lives with the model.
        input: JobInput::Chat(messages),
        max_tokens: req.max_completion_tokens.or(req.max_tokens).unwrap_or(spec.max_tokens),
        params,
        stop_strings: req.stop.clone().map(Stop::into_vec).unwrap_or_default(),
        events: events_tx,
    };
    if let Err(r) = state.submit(job) {
        return r;
    }

    if req.stream {
        sse_response(spec.name, events_rx, false)
    } else {
        collect(spec.name, events_rx, false)
    }
}

async fn completions(State(state): State<AppState>, Json(req): Json<CompletionRequest>) -> Response {
    let spec = match resolve(&state, &req.model) {
        Ok(s) => s,
        Err(r) => return r,
    };
    let params = SampleParams {
        temperature: req.temperature.unwrap_or(spec.temperature),
        top_k: req.top_k.unwrap_or(spec.top_k),
        top_p: req.top_p.unwrap_or(spec.top_p),
        min_p: req.min_p.unwrap_or(spec.min_p),
        repeat_penalty: spec.repeat_penalty,
        seed: req.seed.unwrap_or(0),
        ..Default::default()
    };

    let (tx, rx) = channel::<Event>();
    let job = Job {
        model: spec.name.clone(),
        // A raw completion is fed verbatim, with no chat template.
        input: JobInput::Raw(req.prompt.clone()),
        max_tokens: req.max_tokens.unwrap_or(spec.max_tokens),
        params,
        stop_strings: req.stop.map(Stop::into_vec).unwrap_or_default(),
        events: tx,
    };
    if let Err(r) = state.submit(job) {
        return r;
    }

    if req.stream {
        sse_response(spec.name, rx, true)
    } else {
        collect(spec.name, rx, true)
    }
}

/// Carried through the stream so each poll knows where it left off.
struct StreamState {
    rx: Receiver<Event>,
    id: String,
    model: String,
    sent_role: bool,
    finished: bool,
    raw: bool,
}

fn sse_response(model: String, rx: Receiver<Event>, raw: bool) -> Response {
    let init = StreamState {
        rx,
        id: new_chunk_id(),
        model,
        sent_role: false,
        finished: false,
        raw,
    };

    let stream = futures::stream::unfold(init, |mut s| async move {
        if s.finished {
            return None;
        }
        // The opening delta carries the role and no content, which is what clients expect
        // before any text arrives.
        if !s.sent_role && !s.raw {
            s.sent_role = true;
            let chunk = ChatChunk::new(
                &s.id,
                &s.model,
                Delta { role: Some("assistant"), content: None },
                None,
            );
            return Some((sse(&chunk), s));
        }

        // The worker is a blocking thread; step off the async executor to wait on it.
        let next = tokio::task::block_in_place(|| s.rx.recv());
        match next {
            Ok(Event::Chunk(text)) => {
                let chunk = ChatChunk::new(
                    &s.id,
                    &s.model,
                    Delta { role: None, content: Some(text) },
                    None,
                );
                Some((sse(&chunk), s))
            }
            Ok(Event::Done { reason, .. }) => {
                s.finished = true;
                let chunk = ChatChunk::new(
                    &s.id,
                    &s.model,
                    Delta { role: None, content: None },
                    Some(finish_reason(reason)),
                );
                Some((sse(&chunk), s))
            }
            Ok(Event::Failed(msg)) => {
                s.finished = true;
                Some((
                    Ok(SseEvent::default()
                        .data(serde_json::to_string(&ApiError::new(msg)).unwrap_or_default())),
                    s,
                ))
            }
            // The worker dropped the sender without a Done event, which means it went away.
            Err(_) => None,
        }
    })
    // OpenAI's stream terminator.
    .chain(futures::stream::once(async {
        Ok::<_, std::convert::Infallible>(SseEvent::default().data("[DONE]"))
    }));

    Sse::new(stream).into_response()
}

fn sse<T: serde::Serialize>(v: &T) -> std::result::Result<SseEvent, std::convert::Infallible> {
    Ok(SseEvent::default().data(serde_json::to_string(v).unwrap_or_default()))
}

fn collect(model: String, rx: Receiver<Event>, raw: bool) -> Response {
    let mut text = String::new();
    let mut prompt_tokens = 0;
    let mut generated = 0;
    let mut finish = "stop";
    loop {
        match rx.recv() {
            Ok(Event::Chunk(c)) => text.push_str(&c),
            Ok(Event::Done { reason, prompt_tokens: p, generated: g }) => {
                prompt_tokens = p;
                generated = g;
                finish = finish_reason(reason);
                break;
            }
            Ok(Event::Failed(msg)) => return bad_request(msg),
            Err(_) => break,
        }
    }
    let usage = Usage {
        prompt_tokens,
        completion_tokens: generated,
        total_tokens: prompt_tokens + generated,
    };
    if raw {
        Json(CompletionResponse::new(&model, text, finish, usage)).into_response()
    } else {
        Json(ChatResponse::new(&model, text, finish, usage)).into_response()
    }
}

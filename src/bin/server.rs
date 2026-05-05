use std::sync::Arc;
use anyhow::Result;
use axum::{
    extract::{State, Json},
    http::StatusCode,
    response::{IntoResponse, sse::{KeepAlive, Sse}},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::Path;
use tokio::sync::Mutex;
use uuid::Uuid;

use gguf_rs::model::llama::{KvCache, LlamaModel};
use gguf_rs::tokenizer::bpe::Tokenizer;
use gguf_rs::tokenizer::chat::ChatTemplate;
use gguf_rs::gpu::VkCtx;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelConfig {
    id: String,
    name: String,
    path: String,
    #[serde(default)]
    gpu: bool,
    temperature: Option<f32>,
    ctx_len: Option<usize>,
    max_tokens: Option<usize>,
    system_prompt: Option<String>,
    stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ServerConfig {
    server: ServerSettings,
    defaults: SamplingDefaults,
    models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ServerSettings {
    host: String,
    port: u16,
    api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SamplingDefaults {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    max_tokens: usize,
    repetition_penalty: f32,
    seed: u64,
    ctx_len: usize,
    smart_context: bool,
}

struct LoadedModel {
    model:     LlamaModel,
    tokenizer: Tokenizer,
    template:  ChatTemplate,
    config:    ModelConfig,
    // GPU context lives here — only ever accessed from spawn_blocking
    gpu:       Option<VkCtx>,
}

// SAFETY: VkCtx holds a *mut u8 staging pointer. Access is serialised by the
// Mutex<Option<LoadedModel>> — only one spawn_blocking task holds the guard at a time.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

struct AppState {
    loaded:     Mutex<Option<LoadedModel>>,
    all_models: Vec<ModelConfig>,
    defaults:   SamplingDefaults,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role:    String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatCompletionRequest {
    model:              Option<String>,
    messages:           Vec<ChatMessage>,
    temperature:        Option<f32>,
    top_k:              Option<usize>,
    top_p:              Option<f32>,
    max_tokens:         Option<usize>,
    repetition_penalty: Option<f32>,
    stream:             Option<bool>,
    stop:               Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id:      String,
    object:  String,
    created: u64,
    model:   String,
    choices: Vec<ChoiceDelta>,
}

#[derive(Debug, Serialize)]
struct ChoiceDelta {
    index:         usize,
    delta:         Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role:    Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatCompletion {
    id:      String,
    object:  String,
    created: u64,
    model:   String,
    choices: Vec<Choice>,
    usage:   Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index:         usize,
    message:       ChatMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens:     usize,
    completion_tokens: usize,
    total_tokens:      usize,
}

#[derive(Debug, Serialize)]
struct ModelListResponse {
    object: String,
    data:   Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id:         String,
    object:     String,
    created:    u64,
    owned_by:   String,
    permission: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LoadModelRequest {
    model: String,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn load_config(path: &str) -> Result<ServerConfig> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_yaml::from_str(&content)?)
}

// Find a model config by id, name, or path substring (case-insensitive)
fn find_model<'a>(all: &'a [ModelConfig], req_model: &str) -> Option<&'a ModelConfig> {
    let lower = req_model.to_lowercase();
    all.iter().find(|m| {
        m.id.to_lowercase() == lower
            || m.name.to_lowercase() == lower
            || m.path.to_lowercase().contains(&lower)
            || lower.contains(&m.id.to_lowercase())
    })
}

// Runs entirely on a blocking thread — no await, no Send requirement on VkCtx.
fn load_model_blocking(cfg: ModelConfig, ctx_len: usize) -> Result<LoadedModel> {
    let mut gpu: Option<VkCtx> = if cfg.gpu {
        match VkCtx::init() {
            Ok(g)  => Some(g),
            Err(e) => { eprintln!("GPU init failed: {e}, using CPU"); None }
        }
    } else { None };

    let (model, gguf) = LlamaModel::load(Path::new(&cfg.path), ctx_len, gpu.as_mut())?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let tmpl_str  = gguf.metadata.get("tokenizer.chat_template")
        .and_then(|v| v.as_str()).map(|s| s.to_string());
    let template  = ChatTemplate::detect(&tokenizer, tmpl_str.as_deref());

    Ok(LoadedModel { model, tokenizer, template, config: cfg, gpu })
}

// Hot-swap: unloads current model (if any) and loads the requested one.
async fn do_load_model(state: &Arc<AppState>, model_id: &str) -> Result<()> {
    let cfg = find_model(&state.all_models, model_id)
        .ok_or_else(|| anyhow::anyhow!("Model not found: {}. Available: {}",
            model_id,
            state.all_models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", ")))?
        .clone();

    // Drop current model before loading new one to free VRAM
    *state.loaded.lock().await = None;
    eprintln!("[server] Loading model: {} ({})", cfg.id, cfg.path);

    let ctx_len = cfg.ctx_len.unwrap_or(state.defaults.ctx_len);
    let loaded  = tokio::task::spawn_blocking(move || load_model_blocking(cfg, ctx_len)).await??;

    eprintln!("[server] Model ready: {}", loaded.config.id);
    *state.loaded.lock().await = Some(loaded);
    Ok(())
}

// Ensure the right model is loaded, hot-swapping if needed.
async fn ensure_model(state: &Arc<AppState>, req_model: Option<&str>) -> Result<(), (StatusCode, String)> {
    let req_id = match req_model {
        None => return Ok(()), // use whatever is loaded
        Some(m) => m,
    };

    let already_loaded = {
        let guard = state.loaded.lock().await;
        guard.as_ref().map(|l| {
            l.config.id.to_lowercase() == req_id.to_lowercase()
            || req_id.to_lowercase().contains(&l.config.id.to_lowercase())
        }).unwrap_or(false)
    };

    if !already_loaded {
        eprintln!("[server] Hot-swapping to model: {}", req_id);
        do_load_model(state, req_id).await
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    }
    Ok(())
}

async fn models_list(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let data = state.all_models.iter().map(|m| ModelInfo {
        id:         m.id.clone(),
        object:     "model".to_string(),
        created:    now_secs(),
        owned_by:   "gguf-rs".to_string(),
        permission: vec![],
    }).collect();
    Json(ModelListResponse { object: "list".to_string(), data })
}

async fn load_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> impl IntoResponse {
    match do_load_model(&state, &req.model).await {
        Ok(_)  => (StatusCode::OK,         Json(json!({"status":"loaded","model":req.model}))).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({"error":e.to_string()}))).into_response(),
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let req_model = req.model.clone();
    eprintln!("[server] /v1/chat/completions model={:?} stream={:?} msgs={}",
        req_model, req.stream, req.messages.len());

    // Hot-swap if a different model is requested
    ensure_model(&state, req_model.as_deref()).await?;

    let temperature = req.temperature.unwrap_or(state.defaults.temperature);
    let top_k       = req.top_k.unwrap_or(state.defaults.top_k);
    let top_p       = req.top_p.unwrap_or(state.defaults.top_p);
    let max_tokens  = req.max_tokens.unwrap_or(state.defaults.max_tokens);
    let rep_penalty = req.repetition_penalty.unwrap_or(state.defaults.repetition_penalty);
    let stream      = req.stream.unwrap_or(false);
    let messages    = req.messages;
    let defaults    = state.defaults.clone();

    // Brief lock: tokenize prompt, release before inference
    let (prompt_ids, model_id, prompt_tokens) = {
        let guard = state.loaded.lock().await;
        let loaded = guard.as_ref()
            .ok_or_else(|| (StatusCode::SERVICE_UNAVAILABLE, "No model loaded".to_string()))?;

        // Build full prompt using chat template
        // user_turn() already appends the assistant header so generation starts correctly
        let mut full = String::new();

        // system message from config or first system message in messages
        let sys_content = messages.iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.clone())
            .or_else(|| loaded.config.system_prompt.clone())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string());

        full.push_str(&loaded.template.system_prompt(&sys_content));

        for msg in messages.iter().filter(|m| m.role != "system") {
            match msg.role.as_str() {
                "user"      => full.push_str(&loaded.template.user_turn(&msg.content)),
                "assistant" => full.push_str(&format!("{}\n", msg.content)),
                _           => {}
            }
        }

        eprintln!("[server] prompt ({} chars): {:?}...", full.len(), &full[..full.len().min(120)]);

        let add_bos = loaded.template.uses_bos() && loaded.tokenizer.add_bos_token;
        let ids     = loaded.tokenizer.encode(&full, add_bos);
        eprintln!("[server] prompt_ids: {} tokens", ids.len());

        let n   = ids.len();
        let mid = loaded.config.id.clone();
        (ids, mid, n)
    }; // lock released — VkCtx never held across an await

    if stream {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<
            Result<axum::response::sse::Event, std::convert::Infallible>
        >(256);
        let state2 = state.clone();

        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            let mut guard = rt.block_on(state2.loaded.lock());
            let loaded = match guard.as_mut() { Some(l) => l, None => return };

            let c       = &loaded.model.config;
            let ctx_len = defaults.ctx_len.min(c.n_ctx);
            let mut cpu_cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
            let stops   = loaded.template.stop_tokens(&loaded.tokenizer);

            let mut logits = vec![0f32; c.n_vocab];
            for (i, &id) in prompt_ids.iter().enumerate() {
                logits = match loaded.gpu.as_mut() {
                    Some(g) => loaded.model.forward_gpu(id as usize, i, g),
                    None    => loaded.model.forward_cpu(id as usize, i, &mut cpu_cache),
                };
            }
            let mut pos    = prompt_ids.len();
            let mut recent: Vec<u32> = Vec::with_capacity(64);
            let mut gen_count = 0usize;

            eprintln!("[server] starting stream generation, pos={}", pos);

            let send = |chunk: ChatCompletionChunk| {
                if let Ok(json) = serde_json::to_string(&chunk) {
                    let _ = rt.block_on(tx.send(
                        Ok(axum::response::sse::Event::default().data(json))
                    ));
                }
            };

            // opening role delta
            send(ChatCompletionChunk {
                id: format!("chatcmpl-{}", Uuid::new_v4()),
                object: "chat.completion.chunk".to_string(),
                created: now_secs(), model: model_id.clone(),
                choices: vec![ChoiceDelta { index: 0, finish_reason: None,
                    delta: Delta { role: Some("assistant".to_string()), content: None } }],
            });

            for _ in 0..max_tokens {
                let next = gguf_rs::sampler::sample(&mut logits, temperature, top_k, top_p, rep_penalty, &recent);
                if stops.contains(&(next as u32)) { break; }

                let word = loaded.tokenizer.decode(next as u32);
                recent.push(next as u32);
                if recent.len() > 64 { recent.remove(0); }

                logits = match loaded.gpu.as_mut() {
                    Some(g) => loaded.model.forward_gpu(next, pos, g),
                    None    => loaded.model.forward_cpu(next, pos, &mut cpu_cache),
                };
                pos       += 1;
                gen_count += 1;

                if !word.is_empty() {
                    send(ChatCompletionChunk {
                        id: format!("chatcmpl-{}", Uuid::new_v4()),
                        object: "chat.completion.chunk".to_string(),
                        created: now_secs(), model: model_id.clone(),
                        choices: vec![ChoiceDelta { index: 0, finish_reason: None,
                            delta: Delta { role: None, content: Some(word) } }],
                    });
                }
            }

            eprintln!("[server] stream done, generated {} tokens", gen_count);

            send(ChatCompletionChunk {
                id: format!("chatcmpl-{}", Uuid::new_v4()),
                object: "chat.completion.chunk".to_string(),
                created: now_secs(), model: model_id,
                choices: vec![ChoiceDelta { index: 0,
                    finish_reason: Some("stop".to_string()),
                    delta: Delta { role: None, content: None } }],
            });
            let _ = rt.block_on(tx.send(
                Ok(axum::response::sse::Event::default().data("[DONE]"))
            ));
        });

        let out = async_stream::stream! {
            while let Some(ev) = rx.recv().await { yield ev; }
        };
        Ok((StatusCode::OK, Sse::new(out).keep_alive(KeepAlive::default())).into_response())

    } else {
        let state2 = state.clone();
        let (completion, comp_tokens) = tokio::task::spawn_blocking(move || -> Result<(String, usize)> {
            let rt = tokio::runtime::Handle::current();
            let mut guard = rt.block_on(state2.loaded.lock());
            let loaded = guard.as_mut().ok_or_else(|| anyhow::anyhow!("No model loaded"))?;

            let c       = &loaded.model.config;
            let ctx_len = defaults.ctx_len.min(c.n_ctx);
            let mut cpu_cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
            let stops   = loaded.template.stop_tokens(&loaded.tokenizer);

            let mut logits = vec![0f32; c.n_vocab];
            for (i, &id) in prompt_ids.iter().enumerate() {
                logits = match loaded.gpu.as_mut() {
                    Some(g) => loaded.model.forward_gpu(id as usize, i, g),
                    None    => loaded.model.forward_cpu(id as usize, i, &mut cpu_cache),
                };
            }
            let mut pos    = prompt_ids.len();
            let mut recent: Vec<u32> = Vec::with_capacity(64);
            let mut out    = String::new();
            let mut count  = 0usize;

            eprintln!("[server] starting non-stream generation, pos={}", pos);

            for _ in 0..max_tokens {
                let next = gguf_rs::sampler::sample(&mut logits, temperature, top_k, top_p, rep_penalty, &recent);
                if stops.contains(&(next as u32)) { break; }

                let word = loaded.tokenizer.decode(next as u32);
                recent.push(next as u32);
                if recent.len() > 64 { recent.remove(0); }

                logits = match loaded.gpu.as_mut() {
                    Some(g) => loaded.model.forward_gpu(next, pos, g),
                    None    => loaded.model.forward_cpu(next, pos, &mut cpu_cache),
                };
                pos   += 1;
                count += 1;
                if !word.is_empty() { out.push_str(&word); }
            }
            eprintln!("[server] generation done: {} tokens, reply: {:?}...", count, &out[..out.len().min(80)]);
            Ok((out, count))
        }).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
            .map_err(|e: anyhow::Error| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let guard      = state.loaded.lock().await;
        let model_name = guard.as_ref().map(|m| m.config.id.clone()).unwrap_or_default();

        let resp = ChatCompletion {
            id:      format!("chatcmpl-{}", Uuid::new_v4()),
            object:  "chat.completion".to_string(),
            created: now_secs(),
            model:   model_name,
            choices: vec![Choice {
                index:         0,
                message:       ChatMessage { role: "assistant".to_string(), content: completion },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens: comp_tokens,
                total_tokens:      prompt_tokens + comp_tokens,
            },
        };
        Ok((StatusCode::OK, Json(resp)).into_response())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = load_config("config.yaml")
        .or_else(|_| load_config("config.example.yaml"))
        .unwrap_or_else(|_| { eprintln!("Config not found. Create config.yaml"); std::process::exit(1) });

    gguf_rs::sampler::set_seed(config.defaults.seed);

    let state = Arc::new(AppState {
        loaded:     Mutex::new(None),
        all_models: config.models.clone(),
        defaults:   config.defaults.clone(),
    });

    // No model loaded on startup — hot-swap on first request
    eprintln!("Available models: {}",
        config.models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", "));

    let app = Router::new()
        .route("/v1/models",           get(models_list))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models/load",      post(load_handler))
        .with_state(state)
        .layer(tower_http::cors::CorsLayer::permissive());

    let addr = format!("{}:{}", config.server.host, config.server.port);
    eprintln!("Server on http://{}", addr);
    axum::serve(tokio::net::TcpListener::bind(&addr).await?, app).await?;
    Ok(())
}
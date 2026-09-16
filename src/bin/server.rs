use anyhow::Result;
use clap::Parser;
use gguf_rs::server::{registry, worker, app};

#[derive(Parser)]
#[command(name="gguf-rs-server", about="OpenAI-compatible local inference server")]
struct Args {
    #[arg(long, default_value = "models.ini")]
    models_preset: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let path = std::path::Path::new(&args.models_preset);
    if !path.exists() {
        anyhow::bail!("Models preset file not found: {}", args.models_preset);
    }
    let registry = registry::load_registry(path)?;
    if registry.is_empty() {
        anyhow::bail!("No models registered in {}", args.models_preset);
    }
    eprintln!("[server] Loaded {} model(s) from {}: {}",
        registry.len(), args.models_preset,
        registry.keys().cloned().collect::<Vec<_>>().join(", "));

    let handle = worker::spawn(registry);
    let app = app::build(handle);

    let addr = format!("{}:{}", args.host, args.port);
    eprintln!("[server] Listening on http://{addr}");
    eprintln!("[server]   GET  /v1/models");
    eprintln!("[server]   POST /v1/chat/completions");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

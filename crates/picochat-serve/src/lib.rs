use std::sync::Arc;
use anyhow::Result;
use axum::{
    Router,
    extract::State,
    response::sse::{Event, Sse},
    routing::{get, post},
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::model::GPT;
use picochat_engine::generate::{generate, GenerationConfig};
use picochat_engine::sampling::SamplingParams;
use picochat_tokenizer::special::SpecialToken;
use picochat_tokenizer::Tokenizer;

struct AppState {
    model: GPT,
    tokenizer: Tokenizer,
    device: Device,
    max_tokens: usize,
    temperature: f32,
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct SsePayload {
    /// "text", "thinking", "tool_call", "tool_result", "done", "error"
    r#type: String,
    content: String,
}

#[derive(Serialize)]
struct ModelInfo {
    depth: usize,
    parameters: usize,
    vocab_size: usize,
}

pub struct ServeConfig {
    pub checkpoint_dir: String,
    pub tokenizer_path: String,
    pub port: u16,
    pub max_tokens: usize,
    pub temperature: f32,
    pub static_dir: Option<String>,
}

/// Start the HTTP server. This blocks until the server is shut down.
pub async fn serve(config: &ServeConfig) -> Result<()> {
    let tokenizer = Tokenizer::load(&config.tokenizer_path)?;
    let device = Device::Cpu;
    let model_config = picochat_train::checkpoint::load_config(
        format!("{}/config.json", config.checkpoint_dir),
    )?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&model_config, vb)?;
    picochat_train::checkpoint::load_varmap(
        &varmap,
        format!("{}/model.safetensors", config.checkpoint_dir),
        &device,
    )?;

    println!("Model loaded: depth={}, params={:.2}M",
        model_config.n_layer, model.num_parameters() as f64 / 1e6);

    let state = Arc::new(AppState {
        model,
        tokenizer,
        device,
        max_tokens: config.max_tokens,
        temperature: config.temperature,
    });

    let mut app = Router::new()
        .route("/api/chat", post(chat_handler))
        .route("/api/info", get(info_handler))
        .route("/health", get(health_handler))
        .with_state(state.clone());

    if let Some(ref dir) = config.static_dir {
        app = app.fallback_service(
            tower_http::services::ServeDir::new(dir)
                .fallback(tower_http::services::ServeFile::new(format!("{}/ui.html", dir)))
        );
    }

    let app = app.layer(tower_http::cors::CorsLayer::permissive());

    let addr = format!("0.0.0.0:{}", config.port);
    println!("Serving on http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health_handler() -> &'static str { "ok" }

async fn info_handler(State(state): State<Arc<AppState>>) -> Json<ModelInfo> {
    Json(ModelInfo {
        depth: state.model.n_layers(),
        parameters: state.model.num_parameters(),
        vocab_size: state.tokenizer.vocab_size(),
    })
}

async fn chat_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event, axum::Error>>> {
    let max_tokens = req.max_tokens.unwrap_or(state.max_tokens);
    let temperature = req.temperature.unwrap_or(state.temperature);

    let stream = futures::stream::once(async move {
        let prompt = format!(
            "<|bos|><|user_start|>{}<|user_end|><|assistant_start|>",
            req.message
        );
        let prompt_tokens = match state.tokenizer.encode(&prompt) {
            Ok(t) => t,
            Err(e) => {
                let payload = SsePayload {
                    r#type: "error".to_string(),
                    content: format!("Encode error: {e}"),
                };
                return Ok(Event::default().data(serde_json::to_string(&payload).unwrap()));
            }
        };

        let assistant_end_id = state.tokenizer.special()
            .token_id(SpecialToken::AssistantEnd);

        let gen_config = GenerationConfig {
            max_new_tokens: max_tokens,
            sampling: SamplingParams {
                temperature,
                top_k: 20,
                top_p: 0.9,
                repetition_penalty: 1.3,
            },
            stop_tokens: vec![assistant_end_id],
        };

        let output_tokens = match generate(
            &state.model, &prompt_tokens, &gen_config, &state.device,
        ) {
            Ok(t) => t,
            Err(e) => {
                let payload = SsePayload {
                    r#type: "error".to_string(),
                    content: format!("Generation error: {e}"),
                };
                return Ok(Event::default().data(serde_json::to_string(&payload).unwrap()));
            }
        };

        let filtered: Vec<u32> = output_tokens.into_iter()
            .filter(|&t| t != assistant_end_id)
            .collect();
        let response = state.tokenizer.decode(&filtered);
        let mut events = Vec::new();
        events.push(serde_json::to_string(&SsePayload { r#type: "text".into(), content: response }).unwrap());
        events.push(serde_json::to_string(&SsePayload { r#type: "done".into(), content: String::new() }).unwrap());

        Ok(Event::default().data(events.join("\n\n")))
    });

    Sse::new(stream)
}

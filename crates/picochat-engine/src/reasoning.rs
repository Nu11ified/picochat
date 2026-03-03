use anyhow::Result;
use candle_core::Device;
use picochat_core::kv_cache::KVCache;
use picochat_core::model::GPT;
use picochat_tokenizer::Tokenizer;
use picochat_tokenizer::special::SpecialToken;

use crate::sampling::{sample, SamplingParams};

pub struct ReasoningConfig {
    pub max_new_tokens: usize,
    pub max_think_tokens: usize,
    pub sampling: SamplingParams,
}

/// A segment of model output, classified by type.
#[derive(Debug, Clone, PartialEq)]
pub enum OutputSegment {
    /// Visible response text
    Text(String),
    /// Hidden reasoning (inside <think_start>...<think_end>)
    Thinking(String),
    /// Tool call expression
    ToolCall(String),
    /// Tool result (injected, not model-generated)
    ToolResult(String),
}

/// Generate a response with reasoning separation.
/// Returns segments in order — callers can filter/display as needed.
pub fn generate_with_reasoning(
    model: &GPT,
    prompt_tokens: &[u32],
    config: &ReasoningConfig,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<Vec<OutputSegment>> {
    let special = tokenizer.special();
    let think_start_id = special.token_id(SpecialToken::ThinkStart);
    let think_end_id = special.token_id(SpecialToken::ThinkEnd);
    let assistant_end_id = special.token_id(SpecialToken::AssistantEnd);
    let tool_call_start_id = special.token_id(SpecialToken::ToolCallStart);
    let tool_call_end_id = special.token_id(SpecialToken::ToolCallEnd);
    let tool_result_start_id = special.token_id(SpecialToken::ToolResultStart);
    let tool_result_end_id = special.token_id(SpecialToken::ToolResultEnd);

    let mut cache = KVCache::new(model.n_layers());
    let prompt = candle_core::Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = model.forward_with_cache(&prompt, &mut cache)?;

    let last_logits = logits.flatten(0, 1)?;
    let t = prompt_tokens.len();
    let last_row = last_logits.get(t - 1)?;
    let logit_vec: Vec<f32> = last_row.to_vec1()?;
    let mut next_token = sample(&logit_vec, &config.sampling) as u32;

    let mut segments: Vec<OutputSegment> = Vec::new();
    let mut current_tokens: Vec<u32> = Vec::new();

    #[derive(PartialEq)]
    enum Mode { Text, Thinking, ToolCall }
    let mut mode = Mode::Text;
    let mut think_tokens_used = 0usize;

    let flush = |tokens: &mut Vec<u32>, segments: &mut Vec<OutputSegment>, mode: &Mode, tokenizer: &Tokenizer| {
        if tokens.is_empty() { return; }
        let text = tokenizer.decode(tokens);
        let segment = match mode {
            Mode::Text => OutputSegment::Text(text),
            Mode::Thinking => OutputSegment::Thinking(text),
            Mode::ToolCall => OutputSegment::ToolCall(text),
        };
        segments.push(segment);
        tokens.clear();
    };

    for _ in 0..config.max_new_tokens {
        if next_token == assistant_end_id {
            break;
        }

        if next_token == think_start_id {
            flush(&mut current_tokens, &mut segments, &mode, tokenizer);
            mode = Mode::Thinking;
        } else if next_token == think_end_id && mode == Mode::Thinking {
            flush(&mut current_tokens, &mut segments, &mode, tokenizer);
            mode = Mode::Text;
        } else if next_token == tool_call_start_id {
            flush(&mut current_tokens, &mut segments, &mode, tokenizer);
            mode = Mode::ToolCall;
        } else if next_token == tool_call_end_id && mode == Mode::ToolCall {
            flush(&mut current_tokens, &mut segments, &mode, tokenizer);

            if let Some(OutputSegment::ToolCall(expr)) = segments.last() {
                let result = picochat_tool::run_tool(expr);
                let result_text = match result {
                    picochat_tool::ToolResult::Value(v) => v,
                    picochat_tool::ToolResult::Error(e) => format!("Error: {e}"),
                };
                segments.push(OutputSegment::ToolResult(result_text.clone()));

                // Inject tool result tokens into the KV cache so the model sees them
                let mut inject_tokens = Vec::new();
                inject_tokens.push(tool_result_start_id);
                inject_tokens.extend(tokenizer.encode(&result_text)?);
                inject_tokens.push(tool_result_end_id);

                let inject_tensor = candle_core::Tensor::new(&inject_tokens[..], device)?.unsqueeze(0)?;
                let inject_logits = model.forward_with_cache(&inject_tensor, &mut cache)?;

                let inject_len = inject_tokens.len();
                let last_inject = inject_logits.flatten(0, 1)?.get(inject_len - 1)?;
                let logit_vec: Vec<f32> = last_inject.to_vec1()?;
                next_token = sample(&logit_vec, &config.sampling) as u32;
                continue;
            }
            mode = Mode::Text;
        } else {
            if mode == Mode::Thinking {
                think_tokens_used += 1;
                if think_tokens_used >= config.max_think_tokens {
                    flush(&mut current_tokens, &mut segments, &mode, tokenizer);
                    mode = Mode::Text;
                }
            }
            current_tokens.push(next_token);
        }

        let input = candle_core::Tensor::new(&[[next_token]], device)?;
        let logits = model.forward_with_cache(&input, &mut cache)?;
        let logit_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;
        next_token = sample(&logit_vec, &config.sampling) as u32;
    }

    flush(&mut current_tokens, &mut segments, &mode, tokenizer);
    Ok(segments)
}

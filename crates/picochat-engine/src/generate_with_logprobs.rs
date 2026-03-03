use anyhow::Result;
use candle_core::{Device, Tensor};
use picochat_core::kv_cache::KVCache;
use picochat_core::model::GPT;
use picochat_tokenizer::Tokenizer;

use crate::sampling::{sample, SamplingParams};

pub struct LogprobGenerationConfig {
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
    pub stop_tokens: Vec<u32>,
    /// If set, enables tool interleaving during generation.
    pub tool_call_start_id: Option<u32>,
    pub tool_call_end_id: Option<u32>,
    pub tool_result_start_id: Option<u32>,
    pub tool_result_end_id: Option<u32>,
    pub max_tool_calls: usize,
}

/// Generate tokens with per-token log-probabilities, optionally interleaving tool calls.
///
/// When `tokenizer` is provided and tool token IDs are set, the model can invoke tools:
/// 1. Model generates <tool_call_start> ... expression ... <tool_call_end>
/// 2. Expression is run via picochat_tool::run_tool
/// 3. <tool_result_start> result_tokens <tool_result_end> are fed into the KV cache
/// 4. Generation continues from there
///
/// Tool result tokens don't appear in the returned token_ids/logprobs (the model didn't generate them).
pub fn generate_with_logprobs(
    model: &GPT,
    prompt_tokens: &[u32],
    config: &LogprobGenerationConfig,
    device: &Device,
    tokenizer: Option<&Tokenizer>,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut cache = KVCache::new(model.n_layers());

    let prompt = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = model.forward_with_cache(&prompt, &mut cache)?;

    let last_logits = logits.flatten(0, 1)?;
    let t = prompt_tokens.len();
    let last_row = last_logits.get(t - 1)?;
    let logit_vec: Vec<f32> = last_row.to_vec1()?;

    let lp = compute_log_softmax(&logit_vec);
    let mut next_token = sample(&logit_vec, &config.sampling) as u32;

    let mut output_ids = Vec::with_capacity(config.max_new_tokens);
    let mut output_logprobs = Vec::with_capacity(config.max_new_tokens);

    output_ids.push(next_token);
    output_logprobs.push(lp[next_token as usize]);

    if config.stop_tokens.contains(&next_token) {
        return Ok((output_ids, output_logprobs));
    }

    let mut tool_calls_made = 0usize;
    let mut collecting_tool_call = false;
    let mut tool_call_tokens: Vec<u32> = Vec::new();

    for _ in 1..config.max_new_tokens {
        if let (Some(start_id), Some(end_id)) = (config.tool_call_start_id, config.tool_call_end_id) {
            if next_token == start_id && tool_calls_made < config.max_tool_calls {
                collecting_tool_call = true;
                tool_call_tokens.clear();
            } else if collecting_tool_call && next_token == end_id {
                collecting_tool_call = false;
                tool_calls_made += 1;

                if let Some(tok) = tokenizer {
                    let expr_text = tok.decode(&tool_call_tokens);
                    let result = picochat_tool::run_tool(&expr_text);
                    let result_text = match result {
                        picochat_tool::ToolResult::Value(v) => v,
                        picochat_tool::ToolResult::Error(e) => format!("Error: {e}"),
                    };

                    // Inject tool result tokens into context
                    if let (Some(rs_id), Some(re_id)) = (config.tool_result_start_id, config.tool_result_end_id) {
                        let result_token_ids = tok.encode(&result_text)?;
                        let mut inject: Vec<u32> = Vec::with_capacity(result_token_ids.len() + 2);
                        inject.push(rs_id);
                        inject.extend_from_slice(&result_token_ids);
                        inject.push(re_id);

                        // Feed result tokens through the model to update KV cache
                        let inject_tensor = Tensor::new(&inject[..], device)?.unsqueeze(0)?;
                        let inject_logits = model.forward_with_cache(&inject_tensor, &mut cache)?;

                        // Get logits from the last injected position for next token prediction
                        let inject_len = inject.len();
                        let last_inject = inject_logits.flatten(0, 1)?.get(inject_len - 1)?;
                        let logit_vec: Vec<f32> = last_inject.to_vec1()?;
                        let lp = compute_log_softmax(&logit_vec);
                        next_token = sample(&logit_vec, &config.sampling) as u32;
                        output_ids.push(next_token);
                        output_logprobs.push(lp[next_token as usize]);

                        if config.stop_tokens.contains(&next_token) {
                            break;
                        }
                        continue;
                    }
                }
                tool_call_tokens.clear();
            } else if collecting_tool_call {
                tool_call_tokens.push(next_token);
            }
        }

        let input = Tensor::new(&[[next_token]], device)?;
        let logits = model.forward_with_cache(&input, &mut cache)?;
        let logit_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;

        let lp = compute_log_softmax(&logit_vec);
        next_token = sample(&logit_vec, &config.sampling) as u32;

        output_ids.push(next_token);
        output_logprobs.push(lp[next_token as usize]);

        if config.stop_tokens.contains(&next_token) {
            break;
        }
    }

    Ok((output_ids, output_logprobs))
}

/// Compute log-softmax over a logit vector. Returns log-probabilities.
fn compute_log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_sum = sum_exp.ln() + max;
    logits.iter().map(|&l| l - log_sum).collect()
}

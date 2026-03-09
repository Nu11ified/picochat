use anyhow::Result;
use candle_core::{Device, Tensor};
use picochat_core::kv_cache::KVCache;
use picochat_core::model::GPT;

use crate::sampling::{sample_with_history, SamplingParams};

/// Configuration for text generation.
pub struct GenerationConfig {
    /// Maximum number of new tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling parameters.
    pub sampling: SamplingParams,
    /// Token IDs that trigger generation to stop (e.g., assistant_end).
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingParams::default(),
            stop_tokens: Vec::new(),
        }
    }
}

/// Generate tokens autoregressively from a model.
///
/// 1. Prefill: process all prompt tokens, populate KV cache
/// 2. Decode: generate one token at a time using cached KV
/// 3. Stop when: max_new_tokens reached or a stop token is generated
///
/// Returns the generated token IDs (not including the prompt).
pub fn generate(
    model: &GPT,
    prompt_tokens: &[u32],
    config: &GenerationConfig,
    device: &Device,
) -> Result<Vec<u32>> {
    let mut cache = KVCache::new(model.n_layers());

    // Prefill: process all prompt tokens at once
    let prompt = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?; // (1, T)
    let logits = model.forward_with_cache(&prompt, &mut cache)?; // (1, T, vocab)

    // Sample from last position
    let last_logits = logits.flatten(0, 1)?; // (T, vocab)
    let t = prompt_tokens.len();
    let last_row = last_logits.get(t - 1)?; // (vocab,)
    let logit_vec: Vec<f32> = last_row.to_vec1()?;
    let mut output: Vec<u32> = Vec::with_capacity(config.max_new_tokens);
    let mut next_token = sample_with_history(&logit_vec, &config.sampling, &output) as u32;

    output.push(next_token);

    // Check stop condition
    if config.stop_tokens.contains(&next_token) {
        return Ok(output);
    }

    // Decode loop
    for _ in 1..config.max_new_tokens {
        let input = Tensor::new(&[[next_token]], device)?; // (1, 1)
        let logits = model.forward_with_cache(&input, &mut cache)?; // (1, 1, vocab)
        let logit_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;

        next_token = sample_with_history(&logit_vec, &config.sampling, &output) as u32;
        output.push(next_token);

        if config.stop_tokens.contains(&next_token) {
            break;
        }
    }

    Ok(output)
}

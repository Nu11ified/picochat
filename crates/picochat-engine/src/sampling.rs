use rand::Rng;

/// Sampling parameters for text generation.
pub struct SamplingParams {
    /// Temperature for logit scaling. 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Top-k filtering: keep only the k highest-probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) filtering: keep tokens until cumulative probability > p. 1.0 = disabled.
    pub top_p: f32,
    /// Repetition penalty: divides logits of already-seen tokens by this factor. 1.0 = disabled.
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.95,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_k: 1, top_p: 1.0, repetition_penalty: 1.0 }
    }
}

/// Sample a single token from logits (1D slice of shape [vocab_size]).
///
/// Applies repetition penalty, temperature scaling, top-k, then top-p filtering,
/// then samples from the resulting distribution.
pub fn sample(logits: &[f32], params: &SamplingParams) -> usize {
    sample_with_history(logits, params, &[])
}

/// Sample with repetition penalty applied to tokens in `generated`.
pub fn sample_with_history(logits: &[f32], params: &SamplingParams, generated: &[u32]) -> usize {
    // Greedy: just return argmax (still apply rep penalty)
    if params.temperature <= 0.0 || params.top_k == 1 {
        if params.repetition_penalty > 1.0 && !generated.is_empty() {
            let mut penalized: Vec<f32> = logits.to_vec();
            for &tok in generated {
                let idx = tok as usize;
                if idx < penalized.len() {
                    if penalized[idx] > 0.0 {
                        penalized[idx] /= params.repetition_penalty;
                    } else {
                        penalized[idx] *= params.repetition_penalty;
                    }
                }
            }
            return argmax(&penalized);
        }
        return argmax(logits);
    }

    // 0. Repetition penalty: penalize tokens already generated
    let logits = if params.repetition_penalty > 1.0 && !generated.is_empty() {
        let mut penalized: Vec<f32> = logits.to_vec();
        for &tok in generated {
            let idx = tok as usize;
            if idx < penalized.len() {
                if penalized[idx] > 0.0 {
                    penalized[idx] /= params.repetition_penalty;
                } else {
                    penalized[idx] *= params.repetition_penalty;
                }
            }
        }
        penalized
    } else {
        logits.to_vec()
    };

    // 1. Temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|&l| l / params.temperature).collect();

    // 2. Softmax to get probabilities
    let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scaled.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let mut probs: Vec<(usize, f32)> = exp.iter().enumerate().map(|(i, &e)| (i, e / sum)).collect();

    // 3. Sort by probability descending
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 4. Top-k: keep only top k
    if params.top_k > 0 && params.top_k < probs.len() {
        probs.truncate(params.top_k);
    }

    // 5. Top-p: keep until cumulative probability exceeds p
    if params.top_p < 1.0 {
        let mut cumsum = 0.0;
        let mut cutoff = probs.len();
        for (i, &(_, p)) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum > params.top_p {
                cutoff = i + 1;  // include this token
                break;
            }
        }
        probs.truncate(cutoff);
    }

    // 6. Re-normalize
    let total: f32 = probs.iter().map(|&(_, p)| p).sum();
    for item in &mut probs {
        item.1 /= total;
    }

    // 7. Multinomial sample
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for &(idx, p) in &probs {
        cumsum += p;
        if r < cumsum {
            return idx;
        }
    }

    // Fallback: return highest probability token
    probs[0].0
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

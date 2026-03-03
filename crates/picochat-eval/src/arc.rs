use anyhow::Result;
use candle_core::{Device, D};
use picochat_core::model::GPT;
use picochat_data::arc::{ArcQuestion, format_arc_prompt};

pub struct ArcResult {
    pub accuracy: f64,
    pub num_correct: usize,
    pub num_total: usize,
}

/// Evaluate ARC-Challenge using log-prob scoring (same approach as MMLU).
/// For each question, compute log-probabilities of A/B/C/D answer tokens
/// at the last prompt position and pick the highest.
pub fn evaluate_arc(
    model: &GPT,
    tokenizer: &picochat_tokenizer::Tokenizer,
    exemplars: &[ArcQuestion],
    test_questions: &[ArcQuestion],
    device: &Device,
) -> Result<ArcResult> {
    let mut num_correct = 0usize;
    let num_total = test_questions.len();

    // ASCII byte IDs for A, B, C, D
    let answer_token_ids: Vec<u32> = vec![65, 66, 67, 68];

    for (i, q) in test_questions.iter().enumerate() {
        let prompt = format_arc_prompt(exemplars, q);
        let tokens = tokenizer.encode(&prompt)?;

        let input = candle_core::Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, None)?;

        let last_pos = tokens.len() - 1;
        let last_logits = logits.get(0)?.get(last_pos)?;
        let log_sm = candle_nn::ops::log_softmax(&last_logits.unsqueeze(0)?, D::Minus1)?;
        let log_sm_vec: Vec<f32> = log_sm.squeeze(0)?.to_vec1()?;

        let mut best_idx = 0usize;
        let mut best_lp = f32::NEG_INFINITY;
        for (idx, &tid) in answer_token_ids.iter().enumerate() {
            let lp = log_sm_vec[tid as usize];
            if lp > best_lp {
                best_lp = lp;
                best_idx = idx;
            }
        }

        let predicted = match best_idx {
            0 => "A", 1 => "B", 2 => "C", _ => "D",
        };

        if predicted == q.answer_key {
            num_correct += 1;
        }

        if (i + 1) % 50 == 0 || i == num_total - 1 {
            println!("ARC: {}/{} ({:.1}%)", i + 1, num_total,
                num_correct as f64 / (i + 1) as f64 * 100.0);
        }
    }

    Ok(ArcResult {
        accuracy: num_correct as f64 / num_total as f64,
        num_correct,
        num_total,
    })
}

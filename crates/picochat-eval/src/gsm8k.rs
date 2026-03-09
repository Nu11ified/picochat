use anyhow::Result;
use candle_core::Device;
use picochat_core::model::GPT;
use picochat_engine::generate::{generate, GenerationConfig};
use picochat_engine::sampling::SamplingParams;
use picochat_tokenizer::Tokenizer;

pub struct GsmQuestion {
    pub question: String,
    pub answer: String,
}

pub struct GsmResult {
    pub solve_rate: f64,
    pub num_correct: usize,
    pub num_total: usize,
}

/// Extract the final numeric answer after "####" from a response.
/// Strips commas and whitespace.
pub fn extract_answer(response: &str) -> Option<String> {
    let idx = response.rfind("####")?;
    let after = &response[idx + 4..];
    let answer = after.trim().replace(',', "");
    let answer = answer.split_whitespace().next()?.to_string();
    if answer.is_empty() { None } else { Some(answer) }
}

fn normalize_answer(s: &str) -> String {
    s.trim().replace(',', "").trim_end_matches('.').to_string()
}

/// Format the few-shot chain-of-thought prompt.
pub fn format_gsm_prompt(exemplars: &[GsmQuestion], test_question: &GsmQuestion) -> String {
    let mut prompt = String::new();
    for ex in exemplars {
        prompt.push_str(&format!("Q: {}\nA: {}\n\n", ex.question, ex.answer));
    }
    prompt.push_str(&format!("Q: {}\nA: Let's think step by step.\n", test_question.question));
    prompt
}

/// Evaluate GSM8K with few-shot chain-of-thought generation.
pub fn evaluate_gsm8k(
    model: &GPT,
    tokenizer: &Tokenizer,
    exemplars: &[GsmQuestion],
    test_questions: &[GsmQuestion],
    max_new_tokens: usize,
    device: &Device,
) -> Result<GsmResult> {
    let mut num_correct = 0;
    let mut num_total = 0;

    let gen_config = GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams {
            temperature: 0.0, // greedy for deterministic evaluation
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
        },
        stop_tokens: vec![],
    };

    for (i, test_q) in test_questions.iter().enumerate() {
        let prompt = format_gsm_prompt(exemplars, test_q);
        let prompt_tokens = tokenizer.encode(&prompt)?;

        let output_tokens = generate(model, &prompt_tokens, &gen_config, device)?;
        let response = tokenizer.decode(&output_tokens);

        let predicted = extract_answer(&response);
        let expected = extract_answer(&test_q.answer);

        let correct = match (&predicted, &expected) {
            (Some(p), Some(e)) => normalize_answer(p) == normalize_answer(e),
            _ => false,
        };

        if correct { num_correct += 1; }
        num_total += 1;

        if (i + 1) % 50 == 0 || i == test_questions.len() - 1 {
            println!("GSM8K: {}/{} ({}/{} correct, {:.1}%)",
                i + 1, test_questions.len(),
                num_correct, num_total,
                num_correct as f64 / num_total as f64 * 100.0);
        }
    }

    let solve_rate = if num_total > 0 {
        num_correct as f64 / num_total as f64
    } else {
        0.0
    };
    println!("GSM8K final: {}/{} ({:.1}%)", num_correct, num_total, solve_rate * 100.0);

    Ok(GsmResult { solve_rate, num_correct, num_total })
}

/// Load GSM8K questions from JSONL: {"question": "...", "answer": "..."}
pub fn load_gsm8k_jsonl(path: &str) -> Result<Vec<GsmQuestion>> {
    let content = std::fs::read_to_string(path)?;
    let mut questions = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let question = v["question"].as_str().unwrap_or("").to_string();
        let answer = v["answer"].as_str().unwrap_or("").to_string();
        if !question.is_empty() && !answer.is_empty() {
            questions.push(GsmQuestion { question, answer });
        }
    }
    Ok(questions)
}

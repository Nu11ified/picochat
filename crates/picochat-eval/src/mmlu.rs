use anyhow::Result;
use candle_core::{Device, Tensor, D};
use picochat_core::model::GPT;
use picochat_tokenizer::Tokenizer;
use std::collections::HashMap;

pub struct MmluQuestion {
    pub question: String,
    pub choices: Vec<String>,
    pub answer: usize, // 0=A, 1=B, 2=C, 3=D
}

pub struct MmluResult {
    pub overall_accuracy: f64,
    pub subject_accuracy: HashMap<String, f64>,
    pub num_correct: usize,
    pub num_total: usize,
}

const ANSWER_LABELS: &[&str] = &["A", "B", "C", "D"];

fn format_question(q: &MmluQuestion) -> String {
    let mut s = format!("{}\n", q.question);
    for (i, choice) in q.choices.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", ANSWER_LABELS[i], choice));
    }
    s
}

/// Format the full few-shot MMLU prompt for one test question.
pub fn format_mmlu_prompt(
    exemplars: &[MmluQuestion],
    test_question: &MmluQuestion,
    subject: &str,
) -> String {
    let mut prompt = format!(
        "The following are multiple choice questions (with answers) about {}.\n\n",
        subject.replace('_', " ")
    );

    for ex in exemplars {
        prompt.push_str(&format_question(ex));
        prompt.push_str(&format!("Answer: {}\n\n", ANSWER_LABELS[ex.answer]));
    }

    prompt.push_str(&format_question(test_question));
    prompt.push_str("Answer:");
    prompt
}

/// Return the index of the highest log-prob among A/B/C/D.
pub fn pick_answer_from_logprobs(logprobs: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &lp) in logprobs.iter().enumerate() {
        if lp > best_val {
            best_val = lp;
            best_idx = i;
        }
    }
    best_idx
}

/// Evaluate MMLU on a set of subjects.
pub fn evaluate_mmlu(
    model: &GPT,
    tokenizer: &Tokenizer,
    data: &HashMap<String, (Vec<MmluQuestion>, Vec<MmluQuestion>)>,
    device: &Device,
) -> Result<MmluResult> {
    // A=65, B=66, C=67, D=68 — ASCII byte token IDs
    let answer_token_ids: Vec<u32> = vec![65, 66, 67, 68];

    let mut num_correct = 0;
    let mut num_total = 0;
    let mut subject_accuracy: HashMap<String, f64> = HashMap::new();

    for (subject, (exemplars, test_questions)) in data {
        let mut subject_correct = 0;
        let mut subject_total = 0;

        for test_q in test_questions {
            let prompt = format_mmlu_prompt(exemplars, test_q, subject);
            let tokens = tokenizer.encode(&prompt)?;

            let input = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, None)?;

            let last_pos = logits.dim(1)? - 1;
            let last_logits = logits.get(0)?.get(last_pos)?;
            let log_sm = candle_nn::ops::log_softmax(&last_logits.unsqueeze(0)?, D::Minus1)?
                .squeeze(0)?;
            let log_sm_vec: Vec<f32> = log_sm.to_vec1()?;

            let answer_logprobs: Vec<f32> = answer_token_ids
                .iter()
                .map(|&id| log_sm_vec[id as usize])
                .collect();

            let predicted = pick_answer_from_logprobs(&answer_logprobs);
            if predicted == test_q.answer {
                subject_correct += 1;
                num_correct += 1;
            }
            subject_total += 1;
            num_total += 1;
        }

        let acc = if subject_total > 0 {
            subject_correct as f64 / subject_total as f64
        } else {
            0.0
        };
        subject_accuracy.insert(subject.clone(), acc);
        println!("MMLU {}: {}/{} ({:.1}%)",
            subject, subject_correct, subject_total, acc * 100.0);
    }

    let overall_accuracy = if num_total > 0 {
        num_correct as f64 / num_total as f64
    } else {
        0.0
    };
    println!("MMLU overall: {}/{} ({:.1}%)",
        num_correct, num_total, overall_accuracy * 100.0);

    Ok(MmluResult {
        overall_accuracy,
        subject_accuracy,
        num_correct,
        num_total,
    })
}

/// Load MMLU questions from CSV: question,A,B,C,D,answer
pub fn load_mmlu_csv(path: &str) -> Result<Vec<MmluQuestion>> {
    let content = std::fs::read_to_string(path)?;
    let mut questions = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 { continue; }
        let answer = match parts[5].trim() {
            "A" => 0, "B" => 1, "C" => 2, "D" => 3, _ => continue,
        };
        questions.push(MmluQuestion {
            question: parts[0].to_string(),
            choices: vec![parts[1].to_string(), parts[2].to_string(), parts[3].to_string(), parts[4].to_string()],
            answer,
        });
    }

    Ok(questions)
}

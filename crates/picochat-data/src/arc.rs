use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcQuestion {
    pub question: String,
    pub choices: Vec<String>,
    pub answer_key: String,
}

impl ArcQuestion {
    /// Returns the 0-based index of the correct answer (A=0, B=1, C=2, D=3).
    pub fn answer_index(&self) -> Option<usize> {
        match self.answer_key.as_str() {
            "A" => Some(0),
            "B" => Some(1),
            "C" => Some(2),
            "D" => Some(3),
            _ => None,
        }
    }
}

/// Load ARC questions from a JSONL file.
pub fn load_arc_jsonl(path: &str) -> Result<Vec<ArcQuestion>> {
    let content = std::fs::read_to_string(path)?;
    let mut questions = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let q: ArcQuestion = serde_json::from_str(line)?;
        questions.push(q);
    }
    Ok(questions)
}

/// Format a few-shot ARC prompt with exemplars and a test question.
pub fn format_arc_prompt(exemplars: &[ArcQuestion], question: &ArcQuestion) -> String {
    let mut prompt = String::new();
    let labels = ["A", "B", "C", "D"];

    for ex in exemplars {
        prompt.push_str(&format!("Q: {}\n", ex.question));
        for (i, choice) in ex.choices.iter().enumerate() {
            if i < labels.len() {
                prompt.push_str(&format!("({}) {}\n", labels[i], choice));
            }
        }
        prompt.push_str(&format!("Answer: {}\n\n", ex.answer_key));
    }

    prompt.push_str(&format!("Q: {}\n", question.question));
    for (i, choice) in question.choices.iter().enumerate() {
        if i < labels.len() {
            prompt.push_str(&format!("({}) {}\n", labels[i], choice));
        }
    }
    prompt.push_str("Answer: ");
    prompt
}

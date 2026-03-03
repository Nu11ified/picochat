pub struct ReasoningMetrics {
    pub think_block_rate: f64,
    pub self_correction_rate: f64,
    pub avg_think_length: f64,
    pub num_samples: usize,
}

/// Analyze reasoning quality across a set of model responses.
pub fn evaluate_reasoning(responses: &[String]) -> ReasoningMetrics {
    if responses.is_empty() {
        return ReasoningMetrics {
            think_block_rate: 0.0,
            self_correction_rate: 0.0,
            avg_think_length: 0.0,
            num_samples: 0,
        };
    }

    let mut has_think = 0usize;
    let mut has_correction = 0usize;
    let mut total_think_chars = 0usize;
    let mut think_count = 0usize;

    let correction_patterns = [
        "wait,", "actually,", "let me reconsider",
        "I made an error", "that's wrong", "correction:",
        "no, ", "hmm, ", "let me re",
    ];

    for response in responses {
        let blocks = extract_think_blocks(response);
        if !blocks.is_empty() {
            has_think += 1;
            for block in &blocks {
                think_count += 1;
                total_think_chars += block.len();
                let lower = block.to_lowercase();
                if correction_patterns.iter().any(|p| lower.contains(p)) {
                    has_correction += 1;
                }
            }
        }
    }

    let n = responses.len() as f64;
    ReasoningMetrics {
        think_block_rate: has_think as f64 / n,
        self_correction_rate: if think_count > 0 {
            has_correction as f64 / think_count as f64
        } else {
            0.0
        },
        avg_think_length: if think_count > 0 {
            total_think_chars as f64 / think_count as f64
        } else {
            0.0
        },
        num_samples: responses.len(),
    }
}

fn extract_think_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut remaining = text;
    while let Some(start) = remaining.find("<think_start>") {
        let after_start = &remaining[start + "<think_start>".len()..];
        match after_start.find("<think_end>") {
            Some(end) => {
                blocks.push(after_start[..end].to_string());
                remaining = &after_start[end + "<think_end>".len()..];
            }
            None => break,
        }
    }
    blocks
}

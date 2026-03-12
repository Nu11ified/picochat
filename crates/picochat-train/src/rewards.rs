#[derive(Debug, Clone, Copy)]
pub enum TaskType {
    Math,
    MultipleChoice,
    ToolUse,
    /// Simple Q&A: answer is matched as case-insensitive substring of generated text.
    SimpleQA,
}

#[derive(Debug, Clone)]
pub struct RewardWeights {
    pub accuracy: f64,
    pub format: f64,
    pub tool_use: f64,
    pub length_penalty: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            accuracy: 1.0,
            format: 0.2,
            tool_use: 0.3,
            length_penalty: -0.1,
        }
    }
}

/// Strip all `<think_start>...<think_end>` blocks from text.
pub fn strip_think_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<think_start>") {
        result.push_str(&remaining[..start]);
        match remaining[start..].find("<think_end>") {
            Some(end_offset) => {
                remaining = &remaining[start + end_offset + "<think_end>".len()..];
            }
            None => {
                return result;
            }
        }
    }
    result.push_str(remaining);
    result
}

/// Extract the final answer from model output after stripping think blocks.
pub fn extract_final_answer(text: &str, task_type: TaskType) -> Option<String> {
    let stripped = strip_think_blocks(text);
    match task_type {
        TaskType::SimpleQA => {
            let trimmed = stripped.trim().to_string();
            if trimmed.is_empty() { None } else { Some(trimmed) }
        }
        TaskType::Math | TaskType::ToolUse => {
            if let Some(pos) = stripped.rfind("####") {
                let after = stripped[pos + 4..].trim();
                let answer = after.replace(',', "");
                let answer = answer.trim().to_string();
                if answer.is_empty() { None } else { Some(answer) }
            } else {
                None
            }
        }
        TaskType::MultipleChoice => {
            let stripped_trimmed = stripped.trim();
            for pattern in ["Answer: ", "answer is ", "answer: "] {
                if let Some(pos) = stripped_trimmed.rfind(pattern) {
                    let after = stripped_trimmed[pos + pattern.len()..].trim();
                    if let Some(ch) = after.chars().next() {
                        if "ABCD".contains(ch) {
                            return Some(ch.to_string());
                        }
                    }
                }
            }
            for ch in stripped_trimmed.chars().rev() {
                if "ABCD".contains(ch) {
                    return Some(ch.to_string());
                }
            }
            None
        }
    }
}

/// Accuracy reward: 1.0 if extracted answer matches ground truth, 0.0 otherwise.
/// For SimpleQA, uses case-insensitive substring matching with partial credit.
pub fn accuracy_reward(text: &str, ground_truth: &str, task_type: TaskType) -> f64 {
    match task_type {
        TaskType::SimpleQA => {
            let text_lower = text.to_lowercase();
            let gt_lower = ground_truth.to_lowercase().trim().to_string();
            if text_lower.contains(&gt_lower) { 1.0 } else { 0.0 }
        }
        _ => {
            match extract_final_answer(text, task_type) {
                Some(answer) => {
                    if answer.trim() == ground_truth.trim() { 1.0 } else { 0.0 }
                }
                None => 0.0,
            }
        }
    }
}

/// Format reward: 1.0 if think blocks are properly structured and appear before the answer.
pub fn format_reward(text: &str) -> f64 {
    let has_start = text.contains("<think_start>");
    let has_end = text.contains("<think_end>");

    if !has_start || !has_end {
        return 0.0;
    }

    let start_count = text.matches("<think_start>").count();
    let end_count = text.matches("<think_end>").count();
    if start_count != end_count {
        return 0.0;
    }

    let last_think_end = match text.rfind("<think_end>") {
        Some(pos) => pos,
        None => return 0.0,
    };

    let answer_pos = text.rfind("####")
        .or_else(|| text.rfind("Answer: "));

    match answer_pos {
        Some(pos) if pos > last_think_end => 1.0,
        Some(_) => 0.0,
        None => 0.0,
    }
}

/// Tool use reward: 1.0 for correct invocation with useful result, 0.5 for valid syntax, 0.0 otherwise.
pub fn tool_use_reward(text: &str, ground_truth: &str, requires_tool: bool) -> f64 {
    let has_tool_call = text.contains("<tool_call_start>") && text.contains("<tool_call_end>");

    if !has_tool_call {
        return 0.0;
    }

    if !requires_tool {
        return 0.0;
    }

    let answer_correct = match extract_final_answer(text, TaskType::Math) {
        Some(answer) => answer.trim() == ground_truth.trim(),
        None => false,
    };

    if answer_correct { 1.0 } else { 0.5 }
}

/// Length penalty: 0.0 if under target, proportional penalty if over.
pub fn length_penalty_reward(num_tokens: usize, target_len: usize) -> f64 {
    if num_tokens <= target_len || target_len == 0 {
        0.0
    } else {
        (num_tokens - target_len) as f64 / target_len as f64
    }
}

/// Compute composite reward (excluding ORM — that's added in the GRPO loop).
pub fn composite_reward(
    text: &str,
    ground_truth: &str,
    task_type: TaskType,
    requires_tool: bool,
    num_tokens: usize,
    target_len: usize,
    weights: &RewardWeights,
) -> f64 {
    let acc = accuracy_reward(text, ground_truth, task_type);
    let fmt = format_reward(text);
    let tool = tool_use_reward(text, ground_truth, requires_tool);
    let len_pen = length_penalty_reward(num_tokens, target_len);

    weights.accuracy * acc
        + weights.format * fmt
        + weights.tool_use * tool
        + weights.length_penalty * len_pen
}

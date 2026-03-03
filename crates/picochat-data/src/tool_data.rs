use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolScenario {
    pub prompt: String,
    pub expected_answer: String,
    pub requires_tool: bool,
}

/// Load tool-use scenarios from a JSONL file.
pub fn load_tool_scenarios(path: &str) -> Result<Vec<ToolScenario>> {
    let content = std::fs::read_to_string(path)?;
    let mut scenarios = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let s: ToolScenario = serde_json::from_str(line)?;
        scenarios.push(s);
    }
    Ok(scenarios)
}

/// Format a tool-use prompt with 2-shot exemplars demonstrating think + tool_call usage.
pub fn format_tool_prompt(scenario: &ToolScenario) -> String {
    let mut prompt = String::from(
        "You can use a calculator by writing expressions between <tool_call_start> and <tool_call_end> tags. \
         The result will appear between <tool_result_start> and <tool_result_end> tags. \
         Think through problems step by step using <think_start> and <think_end> tags.\n\n"
    );

    prompt.push_str("Q: What is 15 * 23?\n");
    prompt.push_str("<think_start>I need to multiply 15 by 23. Let me use the calculator.</think_end>\n");
    prompt.push_str("<tool_call_start>15 * 23<tool_call_end>\n");
    prompt.push_str("<tool_result_start>345<tool_result_end>\n");
    prompt.push_str("#### 345\n\n");

    prompt.push_str("Q: How many times does the letter 'a' appear in 'banana'?\n");
    prompt.push_str("<think_start>I need to count occurrences of 'a' in 'banana'. I'll use the count method.</think_end>\n");
    prompt.push_str("<tool_call_start>\"banana\".count(\"a\")<tool_call_end>\n");
    prompt.push_str("<tool_result_start>3<tool_result_end>\n");
    prompt.push_str("#### 3\n\n");

    prompt.push_str(&format!("Q: {}\n", scenario.prompt));

    prompt
}

use picochat_data::tool_data::{ToolScenario, format_tool_prompt};

#[test]
fn test_tool_scenario_parse() {
    let json = r#"{"prompt":"What is 347 * 892?","expected_answer":"309524","requires_tool":true}"#;
    let s: ToolScenario = serde_json::from_str(json).unwrap();
    assert_eq!(s.prompt, "What is 347 * 892?");
    assert_eq!(s.expected_answer, "309524");
    assert!(s.requires_tool);
}

#[test]
fn test_tool_scenario_no_tool() {
    let json = r#"{"prompt":"What is 2 + 2?","expected_answer":"4","requires_tool":false}"#;
    let s: ToolScenario = serde_json::from_str(json).unwrap();
    assert!(!s.requires_tool);
}

#[test]
fn test_format_tool_prompt() {
    let scenario = ToolScenario {
        prompt: "What is 123 * 456?".to_string(),
        expected_answer: "56088".to_string(),
        requires_tool: true,
    };
    let prompt = format_tool_prompt(&scenario);
    assert!(prompt.contains("123 * 456"));
    // Should contain exemplars showing tool usage
    assert!(prompt.contains("<tool_call_start>"));
    assert!(prompt.contains("<tool_result_start>"));
    assert!(prompt.contains("<think_start>"));
}

#[test]
fn test_load_tool_scenarios_from_string() {
    let data = r#"{"prompt":"P1","expected_answer":"A1","requires_tool":true}
{"prompt":"P2","expected_answer":"A2","requires_tool":false}"#;
    let scenarios: Vec<ToolScenario> = data.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(scenarios.len(), 2);
}

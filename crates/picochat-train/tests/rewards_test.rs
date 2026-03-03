use picochat_train::rewards::{
    extract_final_answer, strip_think_blocks,
    accuracy_reward, format_reward, tool_use_reward, length_penalty_reward,
    composite_reward, RewardWeights, TaskType,
};

#[test]
fn test_strip_think_blocks() {
    let text = "Before <think_start>some reasoning<think_end> after #### 42";
    let stripped = strip_think_blocks(text);
    assert_eq!(stripped, "Before  after #### 42");
}

#[test]
fn test_strip_think_blocks_multiple() {
    let text = "<think_start>first<think_end>middle<think_start>second<think_end>end";
    let stripped = strip_think_blocks(text);
    assert_eq!(stripped, "middleend");
}

#[test]
fn test_strip_think_blocks_none() {
    let text = "no thinking here #### 5";
    let stripped = strip_think_blocks(text);
    assert_eq!(stripped, "no thinking here #### 5");
}

#[test]
fn test_extract_final_answer_math() {
    let text = "work... #### 42";
    assert_eq!(extract_final_answer(text, TaskType::Math), Some("42".to_string()));
}

#[test]
fn test_extract_final_answer_mc() {
    let text = "The answer is B";
    assert_eq!(extract_final_answer(text, TaskType::MultipleChoice), Some("B".to_string()));
}

#[test]
fn test_extract_final_answer_mc_from_choices() {
    let text = "After analysis, C is correct because...";
    assert_eq!(extract_final_answer(text, TaskType::MultipleChoice), Some("C".to_string()));
}

#[test]
fn test_accuracy_reward_math_correct() {
    assert_eq!(accuracy_reward("#### 42", "42", TaskType::Math), 1.0);
}

#[test]
fn test_accuracy_reward_math_wrong() {
    assert_eq!(accuracy_reward("#### 43", "42", TaskType::Math), 0.0);
}

#[test]
fn test_accuracy_reward_mc_correct() {
    assert_eq!(accuracy_reward("The answer is B", "B", TaskType::MultipleChoice), 1.0);
}

#[test]
fn test_format_reward_valid() {
    let text = "<think_start>reasoning<think_end>\n#### 42";
    assert_eq!(format_reward(text), 1.0);
}

#[test]
fn test_format_reward_missing_think() {
    let text = "#### 42";
    assert_eq!(format_reward(text), 0.0);
}

#[test]
fn test_format_reward_malformed_think() {
    let text = "<think_start>reasoning\n#### 42";
    assert_eq!(format_reward(text), 0.0);
}

#[test]
fn test_format_reward_think_after_answer() {
    let text = "#### 42\n<think_start>oops<think_end>";
    assert_eq!(format_reward(text), 0.0);
}

#[test]
fn test_tool_use_reward_correct_and_useful() {
    let text = "<tool_call_start>347 * 892<tool_call_end>\n<tool_result_start>309524<tool_result_end>\n#### 309524";
    assert_eq!(tool_use_reward(text, "309524", true), 1.0);
}

#[test]
fn test_tool_use_reward_correct_syntax_but_wrong() {
    let text = "<tool_call_start>347 + 892<tool_call_end>\n<tool_result_start>1239<tool_result_end>\n#### 1239";
    assert_eq!(tool_use_reward(text, "309524", true), 0.5);
}

#[test]
fn test_tool_use_reward_no_tool_when_needed() {
    let text = "#### 309524";
    assert_eq!(tool_use_reward(text, "309524", true), 0.0);
}

#[test]
fn test_tool_use_reward_no_tool_not_needed() {
    let text = "#### 4";
    assert_eq!(tool_use_reward(text, "4", false), 0.0);
}

#[test]
fn test_length_penalty() {
    assert_eq!(length_penalty_reward(50, 100), 0.0);
    assert_eq!(length_penalty_reward(100, 100), 0.0);
    let penalty = length_penalty_reward(150, 100);
    assert!((penalty - 0.5).abs() < 0.01);
}

#[test]
fn test_composite_reward() {
    let weights = RewardWeights::default();
    let text = "<think_start>Let me think<think_end>\n#### 42";
    let score = composite_reward(text, "42", TaskType::Math, false, 50, 100, &weights);
    // accuracy=1.0*1.0 + format=0.2*1.0 + tool=0.3*0.0 + length=(-0.1)*0.0 = 1.2
    assert!((score - 1.2).abs() < 0.01, "got {score}");
}

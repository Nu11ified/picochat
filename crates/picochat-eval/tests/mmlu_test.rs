use picochat_eval::mmlu::{MmluQuestion, format_mmlu_prompt, pick_answer_from_logprobs};

#[test]
fn test_format_mmlu_prompt() {
    let exemplars = vec![
        MmluQuestion {
            question: "What is 1+1?".to_string(),
            choices: vec!["1".to_string(), "2".to_string(), "3".to_string(), "4".to_string()],
            answer: 1,
        },
    ];
    let test_q = MmluQuestion {
        question: "What is 2+2?".to_string(),
        choices: vec!["2".to_string(), "3".to_string(), "4".to_string(), "5".to_string()],
        answer: 2,
    };

    let prompt = format_mmlu_prompt(&exemplars, &test_q, "math");
    assert!(prompt.contains("What is 1+1?"));
    assert!(prompt.contains("Answer: B"));
    assert!(prompt.contains("What is 2+2?"));
    assert!(prompt.contains("A. 2"));
    assert!(prompt.contains("D. 5"));
    assert!(prompt.ends_with("Answer:"));
}

#[test]
fn test_pick_answer_from_logprobs() {
    let logprobs = vec![-2.0f32, -1.5, -0.5, -3.0];
    let answer = pick_answer_from_logprobs(&logprobs);
    assert_eq!(answer, 2);
}

#[test]
fn test_pick_answer_tie_favors_first() {
    let logprobs = vec![-1.0f32, -1.0, -2.0, -2.0];
    let answer = pick_answer_from_logprobs(&logprobs);
    assert_eq!(answer, 0);
}

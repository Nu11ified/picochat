use picochat_eval::reasoning::evaluate_reasoning;

#[test]
fn test_reasoning_metrics_with_thinking() {
    let responses = vec![
        "<think_start>Let me think about this. 2 + 2 = 4</think_end> #### 4".to_string(),
        "<think_start>Wait, actually, I need to reconsider.</think_end> #### 5".to_string(),
        "#### 3".to_string(),
    ];
    let m = evaluate_reasoning(&responses);
    assert_eq!(m.num_samples, 3);
    assert!((m.think_block_rate - 2.0 / 3.0).abs() < 0.01);
    assert!(m.self_correction_rate > 0.0);
    assert!(m.avg_think_length > 0.0);
}

#[test]
fn test_reasoning_metrics_no_thinking() {
    let responses = vec![
        "The answer is 42".to_string(),
        "#### 7".to_string(),
    ];
    let m = evaluate_reasoning(&responses);
    assert_eq!(m.think_block_rate, 0.0);
    assert_eq!(m.self_correction_rate, 0.0);
    assert_eq!(m.avg_think_length, 0.0);
}

#[test]
fn test_reasoning_metrics_empty() {
    let m = evaluate_reasoning(&[]);
    assert_eq!(m.num_samples, 0);
    assert_eq!(m.think_block_rate, 0.0);
}

#[test]
fn test_multiple_think_blocks() {
    let responses = vec![
        "<think_start>First thought</think_end><think_start>Second thought</think_end> #### 1".to_string(),
    ];
    let m = evaluate_reasoning(&responses);
    assert_eq!(m.think_block_rate, 1.0);
    assert!(m.avg_think_length > 0.0);
}

#[test]
fn test_self_correction_patterns() {
    let responses = vec![
        "<think_start>Hmm, wait, that's wrong. Let me reconsider.</think_end> #### 42".to_string(),
    ];
    let m = evaluate_reasoning(&responses);
    assert!(m.self_correction_rate > 0.0);
}

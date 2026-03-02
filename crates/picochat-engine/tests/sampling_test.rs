use picochat_engine::sampling::{sample, SamplingParams};

#[test]
fn test_greedy_returns_argmax() {
    let logits = vec![1.0, 5.0, 2.0, 0.5];
    let params = SamplingParams::greedy();
    let token = sample(&logits, &params);
    assert_eq!(token, 1);
}

#[test]
fn test_greedy_with_zero_temperature() {
    let logits = vec![0.1, 0.2, 10.0, 0.3];
    let params = SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0 };
    let token = sample(&logits, &params);
    assert_eq!(token, 2);
}

#[test]
fn test_top_k_limits_candidates() {
    let logits = vec![1.0, 5.0, 2.0, 0.5];
    let params = SamplingParams { temperature: 1.0, top_k: 1, top_p: 1.0 };
    let token = sample(&logits, &params);
    assert_eq!(token, 1);
}

#[test]
fn test_sample_respects_distribution() {
    let logits = vec![0.0; 10];
    let params = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0 };
    let mut seen = std::collections::HashSet::new();
    for _ in 0..1000 {
        seen.insert(sample(&logits, &params));
    }
    assert!(seen.len() > 1, "uniform logits should produce varied samples");
}

#[test]
fn test_default_params() {
    let params = SamplingParams::default();
    assert_eq!(params.temperature, 0.8);
    assert_eq!(params.top_k, 50);
    assert_eq!(params.top_p, 0.95);
}

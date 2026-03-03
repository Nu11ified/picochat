use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_engine::generate_with_logprobs::{generate_with_logprobs, LogprobGenerationConfig};
use picochat_engine::sampling::SamplingParams;

fn make_model() -> (GPT, Device) {
    let config = GPTConfig::from_depth(2);
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&config, vb).unwrap();
    (model, device)
}

#[test]
fn test_logprobs_returns_ids_and_probs() {
    let (model, device) = make_model();
    let config = LogprobGenerationConfig {
        max_new_tokens: 10,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![],
        tool_call_start_id: None,
        tool_call_end_id: None,
        tool_result_start_id: None,
        tool_result_end_id: None,
        max_tool_calls: 3,
    };
    let (ids, logprobs) = generate_with_logprobs(
        &model, &[1, 2, 3], &config, &device, None,
    ).unwrap();
    assert_eq!(ids.len(), 10);
    assert_eq!(logprobs.len(), 10);
    // Log-probs should be negative (log of probability < 1)
    for &lp in &logprobs {
        assert!(lp <= 0.0, "logprob {} should be <= 0", lp);
    }
}

#[test]
fn test_logprobs_greedy_deterministic() {
    let (model, device) = make_model();
    let config = LogprobGenerationConfig {
        max_new_tokens: 5,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![],
        tool_call_start_id: None,
        tool_call_end_id: None,
        tool_result_start_id: None,
        tool_result_end_id: None,
        max_tool_calls: 3,
    };
    let (ids1, lp1) = generate_with_logprobs(
        &model, &[1, 2], &config, &device, None,
    ).unwrap();
    let (ids2, lp2) = generate_with_logprobs(
        &model, &[1, 2], &config, &device, None,
    ).unwrap();
    assert_eq!(ids1, ids2);
    for (a, b) in lp1.iter().zip(lp2.iter()) {
        assert!((a - b).abs() < 1e-5, "logprobs differ: {} vs {}", a, b);
    }
}

#[test]
fn test_logprobs_with_stop_token() {
    let (model, device) = make_model();
    let config_no_stop = LogprobGenerationConfig {
        max_new_tokens: 5,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![],
        tool_call_start_id: None,
        tool_call_end_id: None,
        tool_result_start_id: None,
        tool_result_end_id: None,
        max_tool_calls: 3,
    };
    let (ids, _) = generate_with_logprobs(
        &model, &[1, 2], &config_no_stop, &device, None,
    ).unwrap();

    let config = LogprobGenerationConfig {
        max_new_tokens: 100,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![ids[0]],
        tool_call_start_id: None,
        tool_call_end_id: None,
        tool_result_start_id: None,
        tool_result_end_id: None,
        max_tool_calls: 3,
    };
    let (stopped_ids, stopped_lps) = generate_with_logprobs(
        &model, &[1, 2], &config, &device, None,
    ).unwrap();
    assert_eq!(stopped_ids.len(), 1);
    assert_eq!(stopped_lps.len(), 1);
}

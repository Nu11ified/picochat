use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_engine::generate::{generate, GenerationConfig};
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
fn test_generate_produces_tokens() {
    let (model, device) = make_model();
    let config = GenerationConfig {
        max_new_tokens: 10,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![],
    };
    let prompt = vec![1u32, 2, 3];
    let output = generate(&model, &prompt, &config, &device).unwrap();
    assert_eq!(output.len(), 10);
}

#[test]
fn test_generate_stops_at_max_tokens() {
    let (model, device) = make_model();
    let config = GenerationConfig {
        max_new_tokens: 5,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![],
    };
    let output = generate(&model, &[1, 2], &config, &device).unwrap();
    assert_eq!(output.len(), 5);
}

#[test]
fn test_generate_stops_at_stop_token() {
    let (model, device) = make_model();
    // First run: get what greedy generates
    let first_run = generate(
        &model,
        &[1, 2],
        &GenerationConfig {
            max_new_tokens: 5,
            sampling: SamplingParams::greedy(),
            stop_tokens: vec![],
        },
        &device,
    )
    .unwrap();

    // Second run: use first generated token as stop token
    let config = GenerationConfig {
        max_new_tokens: 100,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![first_run[0]],
    };
    let output = generate(&model, &[1, 2], &config, &device).unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], first_run[0]);
}

#[test]
fn test_greedy_is_deterministic() {
    let (model, device) = make_model();
    let config = GenerationConfig {
        max_new_tokens: 10,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![],
    };
    let output1 = generate(&model, &[1, 2, 3], &config, &device).unwrap();
    let output2 = generate(&model, &[1, 2, 3], &config, &device).unwrap();
    assert_eq!(output1, output2);
}

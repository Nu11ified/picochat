use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;

#[test]
fn test_gpt_forward_logits_shape() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = GPT::new(&config, vb).unwrap();
    let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();

    let logits = model.forward(&input, None).unwrap();
    // logits: (B, T, vocab_size) — NOT padded vocab size
    assert_eq!(logits.dims(), &[1, 4, config.vocab_size]);
}

#[test]
fn test_gpt_forward_with_targets() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = GPT::new(&config, vb).unwrap();
    let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
    let targets = Tensor::new(&[[2u32, 3, 4, 5]], &device).unwrap();

    let loss = model.forward(&input, Some(&targets)).unwrap();
    // Loss should be scalar
    assert_eq!(loss.dims(), &[] as &[usize]);
    let loss_val: f32 = loss.to_scalar().unwrap();
    // Cross-entropy loss on random weights should be ~log(vocab_size) ≈ 10.4
    assert!(loss_val > 5.0 && loss_val < 15.0, "loss={loss_val} out of range");
}

#[test]
fn test_gpt_depth4_small() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = GPT::new(&config, vb).unwrap();
    let num_params = model.num_parameters();
    // depth 4 should be a small model (< 50M params)
    assert!(num_params < 50_000_000, "Too many params: {num_params}");
    assert!(num_params > 1_000_000, "Too few params: {num_params}");
    println!("depth=4 params: {num_params}");
}

use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_train::value_head::ValueHead;

#[test]
fn test_value_head_forward_shape() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vh = ValueHead::new(64, vb).unwrap();

    // (batch=2, n_embd=64) -> (batch=2, 1)
    let hidden = Tensor::randn(0.0f32, 1.0, (2, 64), &device).unwrap();
    let out = vh.forward(&hidden).unwrap();
    assert_eq!(out.dims(), &[2, 1]);
}

#[test]
fn test_value_head_mse_loss() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vh = ValueHead::new(64, vb).unwrap();

    let hidden = Tensor::randn(0.0f32, 1.0, (4, 64), &device).unwrap();
    let targets = Tensor::new(&[0.5f32, 1.0, 0.0, 0.8], &device).unwrap();
    let loss = vh.mse_loss(&hidden, &targets).unwrap();

    let loss_val: f32 = loss.to_scalar().unwrap();
    assert!(loss_val >= 0.0, "MSE loss should be non-negative");
}

#[test]
fn test_value_head_output_is_scalar_per_sample() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vh = ValueHead::new(128, vb).unwrap();

    let hidden = Tensor::randn(0.0f32, 1.0, (3, 128), &device).unwrap();
    let out = vh.forward(&hidden).unwrap();
    let scores = out.squeeze(1).unwrap();
    assert_eq!(scores.dims(), &[3]);
}

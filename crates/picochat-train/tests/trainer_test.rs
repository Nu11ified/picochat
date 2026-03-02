use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::init::initialize_weights;
use picochat_core::model::GPT;
use picochat_train::trainer::Trainer;

#[test]
fn test_single_train_step() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();

    let mut trainer = Trainer::new(&varmap, &config);
    let input = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], &device).unwrap();
    let target = Tensor::new(&[[2u32, 3, 4, 5, 6, 7, 8, 9]], &device).unwrap();

    let loss = trainer.train_step(&model, &input, &target).unwrap();
    let loss_val: f32 = loss.to_scalar().unwrap();
    assert!(
        loss_val > 5.0 && loss_val < 15.0,
        "initial loss={loss_val}"
    );
}

#[test]
fn test_loss_decreases_over_steps() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&config, vb).unwrap();
    initialize_weights(&varmap, &config).unwrap();

    let mut trainer = Trainer::new(&varmap, &config);
    let input = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], &device).unwrap();
    let target = Tensor::new(&[[2u32, 3, 4, 5, 6, 7, 8, 9]], &device).unwrap();

    let mut losses = Vec::new();
    for _ in 0..10 {
        let loss = trainer.train_step(&model, &input, &target).unwrap();
        losses.push(loss.to_scalar::<f32>().unwrap());
    }

    let first_loss = losses[0];
    let last_loss = *losses.last().unwrap();
    assert!(
        last_loss < first_loss,
        "loss should decrease: first={first_loss}, last={last_loss}"
    );
}

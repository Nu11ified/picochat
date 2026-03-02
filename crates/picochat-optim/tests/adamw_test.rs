use candle_core::{DType, Device, Tensor, Var};
use picochat_optim::AdamW;

fn device() -> Device {
    Device::Cpu
}

#[test]
fn test_adamw_single_step_changes_params() {
    let dev = device();
    let var = Var::ones((4, 4), DType::F32, &dev).unwrap();
    let grad = Tensor::ones((4, 4), DType::F32, &dev).unwrap();
    let mut opt = AdamW::new(1e-3, 0.9, 0.999, 1e-8, 0.01);

    opt.step_var(&var, &grad, 1e-3).unwrap();

    // Parameters should have moved away from 1.0
    let vals: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    for v in &vals {
        assert!(
            (*v - 1.0).abs() > 1e-6,
            "Param should have changed from 1.0, got {v}"
        );
        // They should decrease (gradient points in positive direction)
        assert!(*v < 1.0, "Param should have decreased, got {v}");
    }
}

#[test]
fn test_adamw_weight_decay() {
    let dev = device();
    let var = Var::ones((4, 4), DType::F32, &dev).unwrap();
    // Zero gradient -- only weight decay should affect params
    let grad = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
    let mut opt = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);

    opt.step_var(&var, &grad, 0.1).unwrap();

    let vals: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    for v in &vals {
        // Weight decay: theta = theta * (1 - lr * wd) = 1.0 * (1 - 0.01) = 0.99
        assert!(*v < 1.0, "Weight decay should shrink params, got {v}");
        assert!(
            (*v - 0.99).abs() < 1e-5,
            "Expected ~0.99 after weight decay, got {v}"
        );
    }
}

#[test]
fn test_adamw_multiple_steps_converge() {
    let dev = device();
    let var = Var::ones((4, 4), DType::F32, &dev).unwrap();
    let grad = Tensor::ones((4, 4), DType::F32, &dev).unwrap();
    let mut opt = AdamW::new(1e-2, 0.9, 0.999, 1e-8, 0.01);

    for _ in 0..100 {
        opt.step_var(&var, &grad, 1e-2).unwrap();
    }

    let vals: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    for v in &vals {
        // After 100 steps with constant gradient of 1, params should have moved significantly
        assert!(
            (1.0 - *v).abs() > 0.1,
            "After 100 steps params should have moved significantly from 1.0, got {v}"
        );
    }
}

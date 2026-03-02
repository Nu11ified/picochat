use candle_core::{DType, Device, Tensor, Var};
use picochat_optim::{polar_express, Muon};

fn device() -> Device {
    Device::Cpu
}

#[test]
fn test_polar_express_near_orthogonal() {
    let dev = device();
    // Random 8x8 matrix
    let g = Tensor::rand(0f32, 1f32, (8, 8), &dev).unwrap();
    let u = polar_express(&g).unwrap();

    // U @ U^T should approximate I
    let uut = u.matmul(&u.t().unwrap()).unwrap();
    let eye = Tensor::eye(8, DType::F32, &dev).unwrap();
    let diff = (uut - eye).unwrap();
    // Frobenius norm squared of (U@U^T - I)
    let frob_sq: f32 = diff.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
    assert!(
        frob_sq < 0.1,
        "U@U^T should be near identity, frobenius_sq = {frob_sq}"
    );
}

#[test]
fn test_polar_express_tall_matrix() {
    let dev = device();
    // Tall matrix: 16x4 -- internally transposed for the iteration
    let g = Tensor::rand(0f32, 1f32, (16, 4), &dev).unwrap();
    let u = polar_express(&g).unwrap();

    // Output should preserve shape
    let dims = u.dims();
    assert_eq!(dims, &[16, 4], "Output shape should be 16x4, got {dims:?}");
}

#[test]
fn test_muon_single_step() {
    let dev = device();
    let var = Var::ones((4, 4), DType::F32, &dev).unwrap();
    let grad = Tensor::ones((4, 4), DType::F32, &dev).unwrap();
    let mut opt = Muon::new(0.02, 0.95);

    let before: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    opt.step_var(&var, &grad, 0.02).unwrap();
    let after: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();

    // Parameters should have changed
    let changed = before
        .iter()
        .zip(after.iter())
        .any(|(b, a)| (b - a).abs() > 1e-6);
    assert!(changed, "Parameters should change after one Muon step");
}

#[test]
fn test_muon_momentum_accumulates() {
    let dev = device();
    let var = Var::ones((4, 4), DType::F32, &dev).unwrap();
    // Use a non-uniform gradient so momentum changes the *direction*
    // (not just magnitude, which polar_express normalizes away)
    let grad = Tensor::rand(0f32, 1f32, (4, 4), &dev).unwrap();
    let mut opt = Muon::new(0.02, 0.95);

    // First step — momentum buffer starts at zero
    let before_step1: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    opt.step_var(&var, &grad, 0.02).unwrap();
    let after_step1: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();

    // Second step — feed a *different* gradient so momentum blends them
    let grad2 = Tensor::rand(0f32, 1f32, (4, 4), &dev).unwrap();
    let before_step2: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
    opt.step_var(&var, &grad2, 0.02).unwrap();
    let after_step2: Vec<f32> = var.as_tensor().flatten_all().unwrap().to_vec1().unwrap();

    // The update directions should differ because momentum blends the two gradients.
    // Compute the update vectors
    let update1: Vec<f64> = before_step1
        .iter()
        .zip(after_step1.iter())
        .map(|(b, a)| (*b - *a) as f64)
        .collect();
    let update2: Vec<f64> = before_step2
        .iter()
        .zip(after_step2.iter())
        .map(|(b, a)| (*b - *a) as f64)
        .collect();

    // The two updates should not be identical (momentum changes the direction)
    let diff_sq: f64 = update1
        .iter()
        .zip(update2.iter())
        .map(|(u1, u2)| (u1 - u2).powi(2))
        .sum();
    assert!(
        diff_sq > 1e-10,
        "Momentum should cause different update directions across steps, diff_sq={diff_sq}"
    );
}

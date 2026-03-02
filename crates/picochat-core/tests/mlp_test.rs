use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use picochat_core::mlp::MLP;

#[test]
fn test_mlp_output_shape() {
    let device = &Device::Cpu;
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);
    let mlp = MLP::new(64, vb).unwrap();

    let x = Tensor::randn(0f32, 1.0, (2, 4, 64), device).unwrap();
    let y = mlp.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4, 64]);
}

#[test]
fn test_mlp_relu_squared_activation() {
    let device = &Device::Cpu;
    // Test ReLU^2: [-2, -1, 0, 1, 2] -> [0, 0, 0, 1, 4]
    let input = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], device).unwrap();
    let activated = input.relu().unwrap().sqr().unwrap();
    let vals: Vec<f32> = activated.to_vec1().unwrap();
    let expected = vec![0.0f32, 0.0, 0.0, 1.0, 4.0];
    for (got, exp) in vals.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-6,
            "expected {exp}, got {got}"
        );
    }
}

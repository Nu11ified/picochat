use candle_core::{Device, Tensor};
use picochat_engine::quantize::{quantize_tensor, dequantize_tensor, quantization_error};

#[test]
fn test_quantize_roundtrip() {
    let device = Device::Cpu;
    let data: Vec<f32> = vec![
        1.0, -0.5, 0.25, 0.0,
        -1.0, 0.75, -0.125, 0.5,
    ];
    let tensor = Tensor::new(data, &device).unwrap().reshape(&[2, 4]).unwrap();
    let qt = quantize_tensor(&tensor, &device).unwrap();

    assert_eq!(qt.shape, [2, 4]);

    let reconstructed = dequantize_tensor(&qt, &device).unwrap();
    assert_eq!(reconstructed.dims(), &[2, 4]);

    let error = quantization_error(&tensor, &qt, &device).unwrap();
    assert!(error < 0.01, "quantization error too large: {error}");
}

#[test]
fn test_quantize_zeros() {
    let device = Device::Cpu;
    let tensor = Tensor::zeros(&[3, 4], candle_core::DType::F32, &device).unwrap();
    let qt = quantize_tensor(&tensor, &device).unwrap();
    let reconstructed = dequantize_tensor(&qt, &device).unwrap();

    let vals: Vec<f32> = reconstructed.flatten_all().unwrap().to_vec1().unwrap();
    for v in vals {
        assert!(v.abs() < 1e-6);
    }
}

#[test]
fn test_quantize_large_values() {
    let device = Device::Cpu;
    let data: Vec<f32> = vec![100.0, -100.0, 50.0, -50.0];
    let tensor = Tensor::new(data, &device).unwrap().reshape(&[1, 4]).unwrap();
    let qt = quantize_tensor(&tensor, &device).unwrap();
    let error = quantization_error(&tensor, &qt, &device).unwrap();
    assert!(error < 1.0, "error for large values: {error}");
}

#[test]
fn test_scales_shape() {
    let device = Device::Cpu;
    let tensor = Tensor::ones(&[5, 10], candle_core::DType::F32, &device).unwrap();
    let qt = quantize_tensor(&tensor, &device).unwrap();
    assert_eq!(qt.scales.len(), 5);
}

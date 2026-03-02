use candle_core::{Device, Tensor};
use picochat_core::norm::rms_norm;

#[test]
fn test_rms_norm_shape_preserved() {
    let x = Tensor::randn(0f32, 1.0, (2, 4, 8), &Device::Cpu).unwrap();
    let y = rms_norm(&x).unwrap();
    assert_eq!(y.dims(), &[2, 4, 8]);
}

#[test]
fn test_rms_norm_unit_rms() {
    let x = Tensor::randn(0f32, 1.0, (2, 4, 8), &Device::Cpu).unwrap();
    let y = rms_norm(&x).unwrap();
    // RMS of last dim should be ~1.0 for each vector
    let y_sq = y.sqr().unwrap();
    let mean_sq = y_sq.mean_keepdim(2).unwrap();
    let rms = mean_sq.sqrt().unwrap();
    let rms_vals: Vec<f32> = rms.flatten_all().unwrap().to_vec1().unwrap();
    for val in rms_vals {
        assert!(
            (val - 1.0).abs() < 0.01,
            "RMS should be ~1.0, got {val}"
        );
    }
}

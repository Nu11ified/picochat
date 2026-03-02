use candle_core::{Device, Tensor};
use picochat_core::rotary::RotaryEmbedding;

#[test]
fn test_rotary_precompute_shapes() {
    let rope = RotaryEmbedding::new(64, 2048, 10000.0, &Device::Cpu).unwrap();
    // cos/sin should be (1, 2048, 1, 32) for head_dim=64
    assert_eq!(rope.cos().dims(), &[1, 2048, 1, 32]);
    assert_eq!(rope.sin().dims(), &[1, 2048, 1, 32]);
}

#[test]
fn test_apply_rotary_emb_shape() {
    let rope = RotaryEmbedding::new(64, 2048, 10000.0, &Device::Cpu).unwrap();
    let x = Tensor::randn(0f32, 1.0, (2, 16, 8, 64), &Device::Cpu).unwrap();
    let y = rope.apply(&x, 0).unwrap();
    assert_eq!(y.dims(), &[2, 16, 8, 64]);
}

#[test]
fn test_rotary_offset_for_kv_cache() {
    let rope = RotaryEmbedding::new(64, 2048, 10000.0, &Device::Cpu).unwrap();
    // Single token with offset=100 (simulating KV cache usage)
    let x = Tensor::randn(0f32, 1.0, (1, 1, 8, 64), &Device::Cpu).unwrap();
    let y = rope.apply(&x, 100).unwrap();
    assert_eq!(y.dims(), &[1, 1, 8, 64]);
}

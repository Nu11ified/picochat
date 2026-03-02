use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::attention::CausalSelfAttention;
use picochat_core::config::GPTConfig;
use picochat_core::rotary::RotaryEmbedding;

#[test]
fn test_attention_output_shape() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let rope = RotaryEmbedding::new(
        config.head_dim(),
        config.sequence_len * 10,
        10000.0,
        &device,
    )
    .unwrap();
    let attn = CausalSelfAttention::new(&config, 0, vb).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, 8, config.n_embd), &device).unwrap();
    let result = attn
        .forward(&x, None, &rope, 0, (config.sequence_len, 0), None)
        .unwrap();
    assert_eq!(result.dims(), &[1, 8, config.n_embd]);
}

#[test]
fn test_attention_causal_masking() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let rope = RotaryEmbedding::new(
        config.head_dim(),
        config.sequence_len * 10,
        10000.0,
        &device,
    )
    .unwrap();
    let attn = CausalSelfAttention::new(&config, 0, vb).unwrap();

    let x = Tensor::randn(0f32, 1.0, (1, 4, config.n_embd), &device).unwrap();
    let full_out = attn
        .forward(&x, None, &rope, 0, (config.sequence_len, 0), None)
        .unwrap();
    let x_short = x.narrow(1, 0, 2).unwrap();
    let short_out = attn
        .forward(&x_short, None, &rope, 0, (config.sequence_len, 0), None)
        .unwrap();

    let full_first2 = full_out.narrow(1, 0, 2).unwrap().to_vec3::<f32>().unwrap();
    let short_first2 = short_out.to_vec3::<f32>().unwrap();

    for t in 0..2 {
        for d in 0..config.n_embd {
            let diff = (full_first2[0][t][d] - short_first2[0][t][d]).abs();
            assert!(
                diff < 1e-4,
                "Causal violation at t={t} d={d}: diff={diff}"
            );
        }
    }
}

use candle_core::{Device, Tensor};

#[test]
fn test_masked_cross_entropy_basic() {
    use picochat_train::sft::masked_cross_entropy;

    let device = Device::Cpu;
    // (1, 4, 3) logits — batch=1, seq=4, vocab=3
    let logits = Tensor::new(
        &[[[2.0f32, 1.0, 0.0],
           [0.0, 2.0, 1.0],
           [1.0, 0.0, 2.0],
           [2.0, 1.0, 0.0]]],
        &device,
    ).unwrap();

    let targets = Tensor::new(&[[0u32, 1, 2, 0]], &device).unwrap();

    // Only positions 2 and 3 contribute
    let mask = Tensor::new(&[[0.0f32, 0.0, 1.0, 1.0]], &device).unwrap();

    let loss = masked_cross_entropy(&logits, &targets, &mask).unwrap();
    let loss_val: f32 = loss.to_scalar().unwrap();

    // Both masked positions: logits [1,0,2]->target 2 and [2,1,0]->target 0
    // Both have same log_softmax pattern, loss ~0.4076 each
    assert!(loss_val > 0.0, "loss should be positive");
    assert!((loss_val - 0.4076).abs() < 0.01, "loss was {loss_val}");
}

#[test]
fn test_masked_cross_entropy_all_masked() {
    use picochat_train::sft::masked_cross_entropy;

    let device = Device::Cpu;
    let logits = Tensor::new(
        &[[[2.0f32, 1.0, 0.0], [0.0, 2.0, 1.0]]],
        &device,
    ).unwrap();
    let targets = Tensor::new(&[[0u32, 1]], &device).unwrap();
    let mask = Tensor::new(&[[0.0f32, 0.0]], &device).unwrap();

    let loss = masked_cross_entropy(&logits, &targets, &mask).unwrap();
    let loss_val: f32 = loss.to_scalar().unwrap();
    assert!(loss_val.abs() < 0.1, "loss with all-zero mask should be ~0, was {loss_val}");
}

#[test]
fn test_sft_config() {
    use picochat_train::sft::SftConfig;

    let config = SftConfig {
        checkpoint_dir: "ckpt".to_string(),
        tokenizer_path: "tok.json".to_string(),
        datasets: vec![("data/chat.jsonl".to_string(), 1.0)],
        total_steps: 500,
        batch_size: 2,
        seq_len: 256,
        max_lr: 0.0001,
        warmup_steps: 50,
        min_lr_ratio: 0.01,
        save_dir: "sft_ckpt".to_string(),
        save_every: 250,
    };
    assert_eq!(config.total_steps, 500);
    assert_eq!(config.datasets.len(), 1);
}

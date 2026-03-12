use picochat_train::pretrain::PretrainConfig;

#[test]
fn test_pretrain_config_defaults() {
    let config = PretrainConfig {
        data_dir: "data/train".to_string(),
        val_data: Some("data/val.parquet".to_string()),
        tokenizer_path: "tokenizer.json".to_string(),
        total_steps: 1000,
        batch_size: 4,
        seq_len: 512,
        max_lr: 0.001,
        warmup_steps: 100,
        min_lr_ratio: 0.1,
        eval_every: 100,
        save_every: 500,
        save_dir: "checkpoints".to_string(),
        depth: 4,
        resume_from: None,
        start_step: 0,
    };
    assert_eq!(config.total_steps, 1000);
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.seq_len, 512);
}

#[test]
fn test_pretrain_tokens_per_step() {
    let config = PretrainConfig {
        data_dir: "data/train".to_string(),
        val_data: None,
        tokenizer_path: "tokenizer.json".to_string(),
        total_steps: 100,
        batch_size: 8,
        seq_len: 256,
        max_lr: 0.001,
        warmup_steps: 10,
        min_lr_ratio: 0.1,
        eval_every: 50,
        save_every: 100,
        save_dir: "checkpoints".to_string(),
        depth: 4,
        resume_from: None,
        start_step: 0,
    };
    assert_eq!(config.tokens_per_step(), 8 * 256);
}

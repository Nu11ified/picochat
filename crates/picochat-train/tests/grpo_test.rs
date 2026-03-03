use picochat_train::grpo::{
    normalize_advantages, GrpoConfig,
    compute_clipped_objective, compute_kl_penalty,
};

#[test]
fn test_normalize_advantages() {
    let rewards = vec![1.0, 0.5, 0.0, 0.8];
    let advantages = normalize_advantages(&rewards);
    assert_eq!(advantages.len(), 4);

    let mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
    assert!(mean.abs() < 1e-10, "mean={mean}");

    let var: f64 = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / advantages.len() as f64;
    assert!((var.sqrt() - 1.0).abs() < 0.1, "std={}", var.sqrt());

    let max_idx = advantages.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;
    assert_eq!(max_idx, 0);
}

#[test]
fn test_normalize_advantages_all_same() {
    let rewards = vec![0.5, 0.5, 0.5, 0.5];
    let advantages = normalize_advantages(&rewards);
    for &a in &advantages {
        assert!(a.abs() < 1e-6, "expected 0, got {a}");
    }
}

#[test]
fn test_compute_clipped_objective() {
    let obj = compute_clipped_objective(1.0, 1.0, 0.2);
    assert!((obj - 1.0).abs() < 1e-6);

    let obj = compute_clipped_objective(1.5, 1.0, 0.2);
    assert!((obj - 1.2).abs() < 1e-6);

    let obj = compute_clipped_objective(0.5, 1.0, 0.2);
    assert!((obj - 0.5).abs() < 1e-6);

    let obj = compute_clipped_objective(1.5, -1.0, 0.2);
    assert!((obj - (-1.5)).abs() < 1e-6);
}

#[test]
fn test_compute_kl_penalty() {
    let kl = compute_kl_penalty(&[(-1.0, -1.0), (-2.0, -2.0)]);
    assert!(kl.abs() < 1e-6);

    let kl = compute_kl_penalty(&[(-1.0, -2.0), (-0.5, -1.5)]);
    assert!(kl > 0.0);
}

#[test]
fn test_grpo_config_defaults() {
    let config = GrpoConfig::default();
    assert_eq!(config.group_size, 16);
    assert_eq!(config.clip_eps, 0.2);
    assert_eq!(config.kl_beta, 0.04);
    assert_eq!(config.max_gen_tokens, 512);
}

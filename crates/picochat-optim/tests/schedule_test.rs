use picochat_optim::LrSchedule;

#[test]
fn test_warmup_starts_at_zero() {
    let sched = LrSchedule::new(0.01, 100, 1000, 0.2);
    let lr = sched.get_lr(0);
    assert!(lr.abs() < 1e-12, "LR at step 0 should be ~0, got {lr}");
}

#[test]
fn test_warmup_reaches_base_lr() {
    let sched = LrSchedule::new(0.01, 100, 1000, 0.2);
    let lr = sched.get_lr(100);
    assert!(
        (lr - 0.01).abs() < 1e-12,
        "LR at end of warmup should be base_lr, got {lr}"
    );
}

#[test]
fn test_constant_phase() {
    let sched = LrSchedule::new(0.01, 100, 1000, 0.2);
    // Constant phase runs from step 100 to step 800 (warmdown_start)
    for step in [100, 200, 500, 799] {
        let lr = sched.get_lr(step);
        assert!(
            (lr - 0.01).abs() < 1e-12,
            "LR at step {step} should be base_lr, got {lr}"
        );
    }
}

#[test]
fn test_warmdown_ends_at_zero() {
    let sched = LrSchedule::new(0.01, 100, 1000, 0.2);
    let lr = sched.get_lr(1000);
    assert!(lr.abs() < 1e-12, "LR at total_steps should be ~0, got {lr}");
}

#[test]
fn test_warmdown_is_cosine() {
    let sched = LrSchedule::new(0.01, 100, 1000, 0.2);
    // warmdown_start = 800, total_steps = 1000, midpoint = 900
    let lr_mid = sched.get_lr(900);
    // At midpoint of cosine warmdown: 0.01 * 0.5 * (1 + cos(pi * 0.5)) = 0.005
    assert!(
        (lr_mid - 0.005).abs() < 1e-6,
        "LR at midpoint of warmdown should be ~0.005, got {lr_mid}"
    );
}

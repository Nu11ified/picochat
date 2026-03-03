use picochat_train::metrics::TrainingMetrics;

#[test]
fn test_bpb_basic() {
    // BPB = loss * log2(e) / avg_bytes_per_token
    // With loss=2.0, avg_bytes_per_token=3.5:
    // BPB = 2.0 * 1.4427 / 3.5 = 0.8244
    let bpb = TrainingMetrics::compute_bpb(2.0, 3.5);
    assert!((bpb - 0.8244).abs() < 0.001, "bpb was {bpb}");
}

#[test]
fn test_throughput() {
    let tok_s = TrainingMetrics::compute_throughput(1000, 0.5);
    assert!((tok_s - 2000.0).abs() < 0.01);
}

#[test]
fn test_mfu() {
    // MFU = 6 * num_params * tokens_per_step / (elapsed * peak_tflops * 1e12)
    // 6 * 1_000_000 * 512 / (1.0 * 100.0 * 1e12) = 0.00003072
    let mfu = TrainingMetrics::compute_mfu(1_000_000, 512, 1.0, 100.0);
    assert!((mfu - 0.00003072).abs() < 1e-8, "mfu was {mfu}");
}

#[test]
fn test_tracker_accumulation() {
    let mut tracker = TrainingMetrics::new(4.0);
    tracker.record_step(2.0, 1024, 0.1);

    assert!((tracker.last_bpb() - (2.0 * std::f64::consts::LOG2_E / 4.0)).abs() < 0.001);
    assert!((tracker.last_throughput() - 10240.0).abs() < 0.01);
}

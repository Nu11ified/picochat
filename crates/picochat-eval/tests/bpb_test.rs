use picochat_eval::bpb::BpbResult;

#[test]
fn test_bpb_result_fields() {
    let result = BpbResult {
        bpb: 1.234,
        num_tokens: 10000,
        num_bytes: 35000,
        avg_loss: 3.0,
    };
    assert!((result.bpb - 1.234).abs() < 0.001);
    assert_eq!(result.num_tokens, 10000);
    assert_eq!(result.num_bytes, 35000);
}

#[test]
fn test_bpb_formula() {
    // BPB = total_nll_nats / (total_bytes * ln(2))
    // avg_loss=3.0 over 1000 tokens => total_nll = 3000
    // 3500 bytes => bpb = 3000 / (3500 * 0.6931) = 1.237
    let total_nll = 3.0 * 1000.0;
    let total_bytes = 3500.0;
    let bpb = total_nll / (total_bytes * 2.0f64.ln());
    assert!((bpb - 1.237).abs() < 0.01, "bpb was {bpb}");
}

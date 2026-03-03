use picochat_eval::arc::ArcResult;

#[test]
fn test_arc_result_accuracy() {
    let r = ArcResult { accuracy: 0.75, num_correct: 3, num_total: 4 };
    assert_eq!(r.num_correct, 3);
    assert!((r.accuracy - 0.75).abs() < 1e-6);
}

use picochat_core::config::GPTConfig;

#[test]
fn test_depth_12_config() {
    let cfg = GPTConfig::from_depth(12);
    assert_eq!(cfg.n_layer, 12);
    assert_eq!(cfg.n_embd, 768);
    assert_eq!(cfg.n_head, 12);
    assert_eq!(cfg.n_kv_head, 6);
}

#[test]
fn test_depth_4_small_config() {
    let cfg = GPTConfig::from_depth(4);
    assert_eq!(cfg.n_layer, 4);
    // 64 * 4 = 256, round_up(256, 128) = 256
    assert_eq!(cfg.n_embd, 256);
    assert_eq!(cfg.n_head, 4);
    assert_eq!(cfg.n_kv_head, 2);
    // head_dim should divide n_embd evenly
    assert_eq!(cfg.n_embd % cfg.head_dim(), 0);
}

#[test]
fn test_depth_26_gpt2_config() {
    let cfg = GPTConfig::from_depth(26);
    assert!(cfg.n_embd >= 1536);
    // 64 * 26 = 1664, round_up(1664, 128) = 1664
    assert_eq!(cfg.n_embd, 1664);
}

#[test]
fn test_head_dim_consistent() {
    for depth in [4, 8, 12, 16, 20, 24, 26] {
        let cfg = GPTConfig::from_depth(depth);
        assert_eq!(cfg.head_dim(), 64, "head_dim should be 64 for depth={depth}");
    }
}

#[test]
fn test_window_sizes() {
    let cfg = GPTConfig::from_depth(12);
    let windows = cfg.compute_window_sizes();
    assert_eq!(windows.len(), 12);

    let short = cfg.sequence_len / 2;
    let long = cfg.sequence_len;

    // Pattern SSSL repeats: layers 0,1,2=S, 3=L, 4,5,6=S, 7=L, 8,9,10=S, 11=L
    for (i, &(w, _)) in windows.iter().enumerate() {
        if i == windows.len() - 1 {
            // Last layer is always full (long)
            assert_eq!(w, long, "last layer should always be long window");
        } else {
            let pattern_char = "SSSL".chars().nth(i % 4).unwrap();
            match pattern_char {
                'S' => assert_eq!(w, short, "layer {i} should be short"),
                'L' => assert_eq!(w, long, "layer {i} should be long"),
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn test_padded_vocab_size() {
    let cfg = GPTConfig::from_depth(12);
    let padded = cfg.padded_vocab_size();
    assert_eq!(padded % 64, 0, "padded vocab size should be multiple of 64");
    assert!(padded >= cfg.vocab_size);
    // 32768 is already a multiple of 64
    assert_eq!(padded, 32768);
}

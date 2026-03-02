use picochat_data::dataloader::{DataLoader, PackingDataLoader, TokenDataset};

// ---- Existing TokenDataset / DataLoader tests ----

#[test]
fn test_dataset_len() {
    let tokens: Vec<u32> = (0..1000).collect();
    let ds = TokenDataset::new(tokens);
    assert_eq!(ds.len(), 1000);
    assert!(!ds.is_empty());
}

#[test]
fn test_dataset_empty() {
    let ds = TokenDataset::new(vec![]);
    assert_eq!(ds.len(), 0);
    assert!(ds.is_empty());
}

#[test]
fn test_dataloader_batch_shape() {
    let tokens: Vec<u32> = (0..10000).collect();
    let ds = TokenDataset::new(tokens);
    let mut dl = DataLoader::new(ds, 4, 64);
    let (input, target) = dl.next_batch().unwrap();
    assert_eq!(input.len(), 4);
    assert_eq!(input[0].len(), 64);
    assert_eq!(target.len(), 4);
    assert_eq!(target[0].len(), 64);
}

#[test]
fn test_dataloader_target_is_shifted_input() {
    let tokens: Vec<u32> = (0..10000).collect();
    let ds = TokenDataset::new(tokens);
    let mut dl = DataLoader::new(ds, 1, 16);
    let (input, target) = dl.next_batch().unwrap();
    let start = input[0][0];
    for t in 0..16 {
        assert_eq!(input[0][t], start + t as u32);
        assert_eq!(target[0][t], start + 1 + t as u32);
    }
}

// ---- PackingDataLoader tests ----

const BOS: u32 = 9999;

#[test]
fn test_packing_single_document() {
    // seq_len = 4, capacity = 5.  Doc of exactly 4 tokens + BOS = 5 = full.
    let mut pdl = PackingDataLoader::new(1, 4, BOS);
    pdl.add_document(&[10, 20, 30, 40]);
    assert_eq!(pdl.ready_count(), 1, "one complete sequence expected");
}

#[test]
fn test_packing_bos_prepended() {
    let mut pdl = PackingDataLoader::new(1, 4, BOS);
    pdl.add_document(&[10, 20, 30, 40]);
    let (inputs, _targets) = pdl.next_batch().unwrap();
    assert_eq!(inputs[0][0], BOS, "first token of packed sequence must be BOS");
}

#[test]
fn test_packing_target_shift() {
    let mut pdl = PackingDataLoader::new(1, 4, BOS);
    pdl.add_document(&[10, 20, 30, 40]);
    let (inputs, targets) = pdl.next_batch().unwrap();
    // target[t] == input[t+1] for t in 0..seq_len-1
    for t in 0..3 {
        assert_eq!(
            targets[0][t], inputs[0][t + 1],
            "target[{}] should equal input[{}]",
            t,
            t + 1
        );
    }
}

#[test]
fn test_packing_flush_pads() {
    // seq_len=8, capacity=9.  Doc of 3 tokens + BOS = 4 < 9, so not ready.
    let mut pdl = PackingDataLoader::new(1, 8, BOS);
    pdl.add_document(&[1, 2, 3]);
    assert_eq!(pdl.ready_count(), 0, "should not be ready before flush");

    pdl.flush();
    assert_eq!(pdl.ready_count(), 1, "should be ready after flush");

    let (inputs, _targets) = pdl.next_batch().unwrap();
    // Total sequence length is seq_len + 1 = 9, input length = seq_len = 8
    assert_eq!(inputs[0].len(), 8);
}

#[test]
fn test_packing_batch_returns_none_when_insufficient() {
    let mut pdl = PackingDataLoader::new(4, 4, BOS);
    // Add 1 document: BOS + [1,2,3,4] = 5 = capacity, 1 ready sequence
    pdl.add_document(&[1, 2, 3, 4]);
    assert_eq!(pdl.ready_count(), 1);
    assert!(
        pdl.next_batch().is_none(),
        "should return None when fewer than batch_size sequences are ready"
    );
}

#[test]
fn test_packing_long_document_splits() {
    // seq_len=4, capacity=5.  Doc of 10 tokens + BOS = 11 tokens total.
    // Should produce at least ceil(11/5)=3 sequences (2 full from 10 tokens, plus partial).
    let mut pdl = PackingDataLoader::new(1, 4, BOS);
    pdl.add_document(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    // 11 tokens / 5 capacity = 2 full bins + 1 partial (1 token)
    assert!(
        pdl.ready_count() >= 2,
        "expected at least 2 complete sequences, got {}",
        pdl.ready_count()
    );
}

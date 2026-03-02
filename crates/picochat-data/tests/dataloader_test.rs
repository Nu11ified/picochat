use picochat_data::dataloader::{TokenDataset, DataLoader};

#[test]
fn test_dataset_from_tokens() {
    let tokens: Vec<u32> = (0..1000).collect();
    let ds = TokenDataset::new(tokens);
    assert_eq!(ds.len(), 1000);
    assert!(!ds.is_empty());
}

#[test]
fn test_empty_dataset() {
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
fn test_dataloader_target_is_shifted() {
    let tokens: Vec<u32> = (0..10000).collect();
    let ds = TokenDataset::new(tokens);
    let mut dl = DataLoader::new(ds, 1, 8);
    let (input, target) = dl.next_batch().unwrap();
    // target[b][t] == input[b][t+1] for t in 0..T-1
    for i in 0..7 {
        assert_eq!(
            target[0][i], input[0][i + 1],
            "target[0][{i}] != input[0][{}]", i + 1
        );
    }
}

#[test]
fn test_dataloader_sequential_tokens_verify_shift() {
    // With sequential tokens (0,1,2,...), if input starts at position s,
    // then input = [s, s+1, ..., s+T-1] and target = [s+1, s+2, ..., s+T]
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

#[test]
fn test_dataloader_multiple_batches() {
    let tokens: Vec<u32> = (0..100).collect();
    let ds = TokenDataset::new(tokens);
    let mut dl = DataLoader::new(ds, 2, 32);
    for _ in 0..20 {
        let (input, _) = dl.next_batch().unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(input[0].len(), 32);
        assert_eq!(input[1].len(), 32);
    }
}

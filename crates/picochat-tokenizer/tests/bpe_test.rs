use fancy_regex::Regex;
use picochat_tokenizer::bpe::{train_bpe, GPT4_SPLIT_PATTERN};

#[test]
fn test_train_small_vocab() {
    // "aaabaaabaaab" — the most frequent pair should involve byte 'a' (97)
    let vocab = train_bpe("aaabaaabaaab", 256 + 16 + 10).unwrap();
    assert!(!vocab.merges.is_empty(), "should have at least one merge");
    let first_merge = vocab.merges[0];
    assert!(
        first_merge.0 == 97 || first_merge.1 == 97,
        "first merge {:?} should involve byte 'a' (97)",
        first_merge
    );
}

#[test]
fn test_train_merges_count() {
    let vocab = train_bpe("hello world foo bar baz hello world", 256 + 16 + 10).unwrap();
    assert!(
        vocab.merges.len() > 0,
        "should have at least one merge"
    );
    assert!(
        vocab.merges.len() <= 10,
        "should have at most 10 merges, got {}",
        vocab.merges.len()
    );
}

#[test]
fn test_vocab_has_byte_tokens() {
    let vocab = train_bpe("hello", 256 + 16 + 1).unwrap();
    for i in 0u32..256 {
        assert!(
            vocab.vocab.contains_key(&i),
            "missing byte token {}",
            i
        );
        assert_eq!(
            vocab.vocab[&i],
            vec![i as u8],
            "byte token {} has wrong value",
            i
        );
    }
}

#[test]
fn test_merge_vocab_concatenation() {
    // "ababababab" — the most frequent pair should be (97, 98) = ('a', 'b')
    let vocab = train_bpe("ababababab", 256 + 16 + 5).unwrap();
    assert!(!vocab.merges.is_empty(), "should have at least one merge");
    let first_merge = vocab.merges[0];
    assert_eq!(
        first_merge,
        (97, 98),
        "first merge should be (97, 98) = ('a', 'b'), got {:?}",
        first_merge
    );
    // The new token (256) should map to [b'a', b'b']
    let new_id = vocab.merge_map[&(97, 98)];
    assert_eq!(
        vocab.vocab[&new_id],
        vec![b'a', b'b'],
        "merged token should be [a, b]"
    );
}

#[test]
fn test_gpt4_pattern_compiles() {
    let re = Regex::new(GPT4_SPLIT_PATTERN).expect("GPT4_SPLIT_PATTERN should compile");
    let matches: Vec<String> = re
        .find_iter("Hello, world!")
        .filter_map(|m| m.ok())
        .map(|m| m.as_str().to_string())
        .collect();
    assert!(
        !matches.is_empty(),
        "pattern should match something in 'Hello, world!'"
    );
}

#[test]
fn test_pattern_splits_contractions() {
    let re = Regex::new(GPT4_SPLIT_PATTERN).expect("GPT4_SPLIT_PATTERN should compile");
    let text = "I'm don't he'll";
    let matches: Vec<String> = re
        .find_iter(text)
        .filter_map(|m| m.ok())
        .map(|m| m.as_str().to_string())
        .collect();

    // The pattern should split contractions: 'm, 't, 'll should appear as separate matches
    assert!(
        matches.contains(&"'m".to_string()),
        "'m should be split out, got: {:?}",
        matches
    );
    assert!(
        matches.contains(&"'t".to_string()),
        "'t should be split out, got: {:?}",
        matches
    );
    assert!(
        matches.contains(&"'ll".to_string()),
        "'ll should be split out, got: {:?}",
        matches
    );
}

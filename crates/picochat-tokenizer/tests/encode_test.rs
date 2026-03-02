use picochat_tokenizer::Tokenizer;

const SMALL_VOCAB: usize = 256 + 16 + 20; // 292

fn train_small() -> Tokenizer {
    let text = "hello world hello world hello world hello world \
                the quick brown fox jumps over the lazy dog \
                the quick brown fox jumps over the lazy dog";
    Tokenizer::train(text, SMALL_VOCAB).unwrap()
}

#[test]
fn test_encode_decode_roundtrip() {
    let tok = train_small();
    let text = "hello world";
    let tokens = tok.encode(text).unwrap();
    let decoded = tok.decode(&tokens);
    assert_eq!(decoded, text, "roundtrip failed for {:?}", text);
}

#[test]
fn test_encode_special_tokens() {
    let tok = train_small();
    let text = "<|bos|>hello<|user_end|>";
    let tokens = tok.encode(text).unwrap();
    let decoded = tok.decode(&tokens);
    assert_eq!(decoded, text, "roundtrip with special tokens failed");

    // First token should be the bos ID
    assert_eq!(
        tokens[0],
        tok.bos_id(),
        "first token should be bos ID"
    );

    // Last token should be user_end ID
    let user_end_id = tok.special().token_id(picochat_tokenizer::special::SpecialToken::UserEnd);
    assert_eq!(
        *tokens.last().unwrap(),
        user_end_id,
        "last token should be user_end ID"
    );
}

#[test]
fn test_encode_empty_string() {
    let tok = train_small();
    let tokens = tok.encode("").unwrap();
    assert!(tokens.is_empty(), "empty string should produce empty token vec");
}

#[test]
fn test_encode_single_byte() {
    let tok = train_small();
    let tokens = tok.encode("a").unwrap();
    assert_eq!(tokens.len(), 1, "single byte should produce one token");
    assert_eq!(tokens[0], 97, "'a' should encode to byte ID 97");
}

#[test]
fn test_encode_reduces_token_count() {
    let tok = train_small();
    let text = "hello world hello world";
    let tokens = tok.encode(text).unwrap();
    assert!(
        tokens.len() < text.len(),
        "BPE should reduce token count: got {} tokens for {} bytes",
        tokens.len(),
        text.len()
    );
}

#[test]
fn test_save_load_roundtrip() {
    let tok = train_small();
    let text = "hello world the quick brown fox";

    let tokens_before = tok.encode(text).unwrap();

    // Save to a temp file
    let dir = std::env::temp_dir();
    let path = dir.join("picochat_test_tokenizer.json");
    let path_str = path.to_str().unwrap();

    tok.save(path_str).unwrap();

    // Load from the temp file
    let tok2 = Tokenizer::load(path_str).unwrap();
    let tokens_after = tok2.encode(text).unwrap();

    assert_eq!(
        tokens_before, tokens_after,
        "tokens should be identical after save/load roundtrip"
    );

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_adjacent_special_tokens() {
    let tok = train_small();
    let text = "<|bos|><|user_start|>hi<|user_end|>";
    let tokens = tok.encode(text).unwrap();
    let decoded = tok.decode(&tokens);
    assert_eq!(decoded, text, "adjacent special tokens roundtrip failed");
}

#[test]
fn test_unicode_roundtrip() {
    let tok = train_small();
    let text = "caf\u{00e9} r\u{00e9}sum\u{00e9} na\u{00ef}ve";
    let tokens = tok.encode(text).unwrap();
    let decoded = tok.decode(&tokens);
    assert_eq!(decoded, text, "unicode roundtrip failed");
}

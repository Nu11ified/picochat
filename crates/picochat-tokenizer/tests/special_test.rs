use picochat_tokenizer::special::{SpecialToken, SpecialTokenRegistry};
use std::collections::HashSet;

#[test]
fn test_special_token_count() {
    assert_eq!(SpecialToken::ALL.len(), SpecialToken::COUNT);
    assert_eq!(SpecialToken::COUNT, 16);
}

#[test]
fn test_special_token_roundtrip_str() {
    for &token in SpecialToken::ALL {
        let s = token.as_str();
        let parsed = SpecialToken::from_str(s)
            .unwrap_or_else(|| panic!("from_str failed for {:?} -> {:?}", token, s));
        assert_eq!(parsed, token, "roundtrip failed for {:?}", token);
    }
}

#[test]
fn test_registry_ids_at_end_of_vocab() {
    let reg = SpecialTokenRegistry::new(32768);
    // bos is first special token: 32768 - 16 = 32752
    assert_eq!(reg.bos_id(), 32752);
    // pad is last special token: 32768 - 1 = 32767
    assert_eq!(reg.pad_id(), 32767);
    assert_eq!(reg.first_special_id(), 32752);
    assert_eq!(reg.vocab_size(), 32768);
}

#[test]
fn test_registry_roundtrip_id() {
    let reg = SpecialTokenRegistry::new(32768);
    for &token in SpecialToken::ALL {
        let id = reg.token_id(token);
        let back = reg
            .from_id(id)
            .unwrap_or_else(|| panic!("from_id failed for {:?} -> {}", token, id));
        assert_eq!(back, token, "roundtrip failed for {:?} (id={})", token, id);
    }
}

#[test]
fn test_registry_non_special_id_returns_none() {
    let reg = SpecialTokenRegistry::new(32768);
    assert!(reg.from_id(0).is_none(), "ID 0 should not be a special token");
    assert!(reg.from_id(255).is_none(), "ID 255 should not be a special token");
    // 32751 is the last non-special ID (first_special_id - 1)
    assert!(
        reg.from_id(32751).is_none(),
        "ID 32751 should not be a special token"
    );
}

#[test]
fn test_all_strings_unique() {
    let strings: Vec<&str> = SpecialToken::ALL.iter().map(|t| t.as_str()).collect();
    let unique: HashSet<&str> = strings.iter().copied().collect();
    assert_eq!(
        strings.len(),
        unique.len(),
        "duplicate string representations found"
    );
}

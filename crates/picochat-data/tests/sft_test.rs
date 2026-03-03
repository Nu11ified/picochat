use picochat_data::sft::ChatConversation;

#[test]
fn test_chat_message_parse() {
    let json = r#"{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]}"#;
    let conv: ChatConversation = serde_json::from_str(json).unwrap();
    assert_eq!(conv.messages.len(), 2);
    assert_eq!(conv.messages[0].role, "user");
    assert_eq!(conv.messages[0].content, "Hello");
    assert_eq!(conv.messages[1].role, "assistant");
    assert_eq!(conv.messages[1].content, "Hi there");
}

#[test]
fn test_mask_alignment_single_turn() {
    // tokens:       [BOS, U_START, t1, t2, U_END, A_START, t3, t4, A_END]
    // is_assistant:  F    F        F   F   F      F        T   T   T
    // mask (= is_assistant[1:]): [F, F, F, F, F, T, T, T] = [0,0,0,0,0,1,1,1]
    let is_assistant = vec![false, false, false, false, false, false, true, true, true];
    let mask: Vec<u8> = is_assistant[1..].iter().map(|&b| b as u8).collect();
    assert_eq!(mask, vec![0, 0, 0, 0, 0, 1, 1, 1]);
}

#[test]
fn test_mask_alignment_multi_turn() {
    // tokens: BOS U_S u1 U_E A_S a1 A_E U_S u2 U_E A_S a2 A_E
    // is_asst: F  F   F  F   F   T  T   F   F  F   F   T  T
    let is_assistant = vec![
        false, false, false, false, false, true, true, false, false, false, false, true, true,
    ];
    let mask: Vec<u8> = is_assistant[1..].iter().map(|&b| b as u8).collect();
    assert_eq!(mask, vec![0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]);
}

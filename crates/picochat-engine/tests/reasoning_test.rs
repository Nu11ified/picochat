use picochat_engine::reasoning::OutputSegment;

#[test]
fn test_output_segment_equality() {
    let a = OutputSegment::Text("hello".to_string());
    let b = OutputSegment::Text("hello".to_string());
    assert_eq!(a, b);

    let c = OutputSegment::Thinking("hmm".to_string());
    assert_ne!(a, c);
}

#[test]
fn test_output_segment_variants() {
    let segments = vec![
        OutputSegment::Thinking("let me think".to_string()),
        OutputSegment::Text("The answer is 42".to_string()),
        OutputSegment::ToolCall("6 * 7".to_string()),
        OutputSegment::ToolResult("42".to_string()),
    ];

    let visible: Vec<&OutputSegment> = segments.iter()
        .filter(|s| matches!(s, OutputSegment::Text(_)))
        .collect();
    assert_eq!(visible.len(), 1);

    let thinking: Vec<&OutputSegment> = segments.iter()
        .filter(|s| matches!(s, OutputSegment::Thinking(_)))
        .collect();
    assert_eq!(thinking.len(), 1);
}

#[test]
fn test_segment_text_extraction() {
    let seg = OutputSegment::Text("hello world".to_string());
    match seg {
        OutputSegment::Text(t) => assert_eq!(t, "hello world"),
        _ => panic!("expected Text"),
    }
}

use picochat_engine::reasoning::OutputSegment;

#[test]
fn test_sse_payload_serialization() {
    #[derive(serde::Serialize)]
    struct SsePayload {
        r#type: String,
        content: String,
    }

    let payload = SsePayload {
        r#type: "text".to_string(),
        content: "Hello world".to_string(),
    };
    let json = serde_json::to_string(&payload).unwrap();
    assert!(json.contains("\"type\":\"text\""));
    assert!(json.contains("\"content\":\"Hello world\""));
}

#[test]
fn test_segment_to_type_mapping() {
    fn segment_type(seg: &OutputSegment) -> &str {
        match seg {
            OutputSegment::Text(_) => "text",
            OutputSegment::Thinking(_) => "thinking",
            OutputSegment::ToolCall(_) => "tool_call",
            OutputSegment::ToolResult(_) => "tool_result",
        }
    }

    assert_eq!(segment_type(&OutputSegment::Text("hi".into())), "text");
    assert_eq!(segment_type(&OutputSegment::Thinking("hmm".into())), "thinking");
    assert_eq!(segment_type(&OutputSegment::ToolCall("2+2".into())), "tool_call");
    assert_eq!(segment_type(&OutputSegment::ToolResult("4".into())), "tool_result");
}

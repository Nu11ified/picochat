use picochat_tool::{run_tool, ToolResult};

#[test]
fn test_basic_arithmetic() {
    assert_eq!(run_tool("2 + 3"), ToolResult::Value("5".to_string()));
    assert_eq!(run_tool("10 - 4"), ToolResult::Value("6".to_string()));
    assert_eq!(run_tool("3 * 7"), ToolResult::Value("21".to_string()));
    assert_eq!(run_tool("15 / 4"), ToolResult::Value("3.75".to_string()));
    assert_eq!(run_tool("17 % 5"), ToolResult::Value("2".to_string()));
}

#[test]
fn test_power() {
    assert_eq!(run_tool("2 ** 10"), ToolResult::Value("1024".to_string()));
    assert_eq!(run_tool("3 ** 0"), ToolResult::Value("1".to_string()));
}

#[test]
fn test_operator_precedence() {
    assert_eq!(run_tool("2 + 3 * 4"), ToolResult::Value("14".to_string()));
    assert_eq!(run_tool("(2 + 3) * 4"), ToolResult::Value("20".to_string()));
}

#[test]
fn test_unary_minus() {
    assert_eq!(run_tool("-5 + 3"), ToolResult::Value("-2".to_string()));
}

#[test]
fn test_math_functions() {
    assert_eq!(run_tool("sqrt(16)"), ToolResult::Value("4".to_string()));
    assert_eq!(run_tool("abs(-7)"), ToolResult::Value("7".to_string()));
    assert_eq!(run_tool("ceil(3.2)"), ToolResult::Value("4".to_string()));
    assert_eq!(run_tool("floor(3.9)"), ToolResult::Value("3".to_string()));
}

#[test]
fn test_string_len() {
    assert_eq!(run_tool("len(\"hello\")"), ToolResult::Value("5".to_string()));
}

#[test]
fn test_string_count() {
    assert_eq!(run_tool("\"hello world\".count(\"l\")"), ToolResult::Value("3".to_string()));
}

#[test]
fn test_string_upper_lower() {
    assert_eq!(run_tool("\"hello\".upper()"), ToolResult::Value("HELLO".to_string()));
    assert_eq!(run_tool("\"HELLO\".lower()"), ToolResult::Value("hello".to_string()));
}

#[test]
fn test_comparisons() {
    assert_eq!(run_tool("3 > 2"), ToolResult::Value("true".to_string()));
    assert_eq!(run_tool("3 < 2"), ToolResult::Value("false".to_string()));
    assert_eq!(run_tool("3 == 3"), ToolResult::Value("true".to_string()));
    assert_eq!(run_tool("3 != 4"), ToolResult::Value("true".to_string()));
    assert_eq!(run_tool("3 >= 3"), ToolResult::Value("true".to_string()));
    assert_eq!(run_tool("3 <= 2"), ToolResult::Value("false".to_string()));
}

#[test]
fn test_division_by_zero() {
    match run_tool("1 / 0") {
        ToolResult::Error(msg) => assert!(msg.contains("division by zero")),
        other => panic!("expected Error, got {:?}", other),
    }
}

#[test]
fn test_parse_error() {
    match run_tool("2 +") {
        ToolResult::Error(_) => {}
        other => panic!("expected Error, got {:?}", other),
    }
}

#[test]
fn test_unknown_function() {
    match run_tool("foobar(42)") {
        ToolResult::Error(msg) => assert!(msg.contains("unknown function")),
        other => panic!("expected Error, got {:?}", other),
    }
}

#[test]
fn test_integer_display() {
    assert_eq!(run_tool("2 + 3"), ToolResult::Value("5".to_string()));
    assert_eq!(run_tool("1 / 3"), ToolResult::Value("0.3333333333333333".to_string()));
}

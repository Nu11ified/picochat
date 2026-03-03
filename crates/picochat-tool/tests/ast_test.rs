use picochat_tool::ast::{tokenize, parse, Token, Expr};

#[test]
fn test_tokenize_number() {
    let tokens = tokenize("42").unwrap();
    assert_eq!(tokens, vec![Token::Number(42.0)]);
}

#[test]
fn test_tokenize_negative_number() {
    let tokens = tokenize("-3.14").unwrap();
    assert_eq!(tokens, vec![Token::Minus, Token::Number(3.14)]);
}

#[test]
fn test_tokenize_string() {
    let tokens = tokenize("\"hello\"").unwrap();
    assert_eq!(tokens, vec![Token::Str("hello".to_string())]);
}

#[test]
fn test_tokenize_arithmetic() {
    let tokens = tokenize("2 + 3 * 4").unwrap();
    assert_eq!(tokens, vec![
        Token::Number(2.0), Token::Plus,
        Token::Number(3.0), Token::Star,
        Token::Number(4.0),
    ]);
}

#[test]
fn test_tokenize_function_call() {
    let tokens = tokenize("sqrt(16)").unwrap();
    assert_eq!(tokens, vec![
        Token::Ident("sqrt".to_string()),
        Token::LParen,
        Token::Number(16.0),
        Token::RParen,
    ]);
}

#[test]
fn test_tokenize_method_call() {
    let tokens = tokenize("\"hello\".count(\"l\")").unwrap();
    assert_eq!(tokens, vec![
        Token::Str("hello".to_string()),
        Token::Dot,
        Token::Ident("count".to_string()),
        Token::LParen,
        Token::Str("l".to_string()),
        Token::RParen,
    ]);
}

#[test]
fn test_tokenize_comparison() {
    let tokens = tokenize("3 >= 2").unwrap();
    assert_eq!(tokens, vec![
        Token::Number(3.0), Token::Gte, Token::Number(2.0),
    ]);
}

#[test]
fn test_parse_binary_arithmetic() {
    let tokens = tokenize("2 + 3").unwrap();
    let expr = parse(&tokens).unwrap();
    match expr {
        Expr::BinOp { .. } => {}
        _ => panic!("expected BinOp, got {:?}", expr),
    }
}

#[test]
fn test_parse_operator_precedence() {
    // 2 + 3 * 4 should parse as 2 + (3 * 4)
    let tokens = tokenize("2 + 3 * 4").unwrap();
    let expr = parse(&tokens).unwrap();
    match expr {
        Expr::BinOp { op, left, right } => {
            assert_eq!(op, "+");
            match *right {
                Expr::BinOp { op, .. } => assert_eq!(op, "*"),
                _ => panic!("right should be BinOp"),
            }
        }
        _ => panic!("expected BinOp"),
    }
}

#[test]
fn test_parse_parenthesized() {
    let tokens = tokenize("(2 + 3) * 4").unwrap();
    let expr = parse(&tokens).unwrap();
    match expr {
        Expr::BinOp { op, left, .. } => {
            assert_eq!(op, "*");
            match *left {
                Expr::BinOp { op, .. } => assert_eq!(op, "+"),
                _ => panic!("left should be BinOp"),
            }
        }
        _ => panic!("expected BinOp"),
    }
}

#[test]
fn test_parse_function_call() {
    let tokens = tokenize("sqrt(16)").unwrap();
    let expr = parse(&tokens).unwrap();
    match expr {
        Expr::FnCall { name, .. } => assert_eq!(name, "sqrt"),
        _ => panic!("expected FnCall"),
    }
}

#[test]
fn test_parse_method_call() {
    let tokens = tokenize("\"hi\".upper()").unwrap();
    let expr = parse(&tokens).unwrap();
    match expr {
        Expr::MethodCall { method, .. } => assert_eq!(method, "upper"),
        _ => panic!("expected MethodCall"),
    }
}

#[test]
fn test_parse_power() {
    let tokens = tokenize("2 ** 10").unwrap();
    let expr = parse(&tokens).unwrap();
    match expr {
        Expr::BinOp { op, .. } => assert_eq!(op, "**"),
        _ => panic!("expected BinOp"),
    }
}

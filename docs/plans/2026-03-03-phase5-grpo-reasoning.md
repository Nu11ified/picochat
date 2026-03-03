# Phase 5: GRPO Reasoning Training — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement GRPO reasoning training with G=16 multi-sample generation, thinking mode, live tool execution, a lightweight ORM value head, and composite reward scoring.

**Architecture:** Layered build: tool execution engine → data loaders (ARC, tool-use) → generation with log-probs + tool interleaving → value head → reward functions → GRPO training loop → CLI. Each layer is independently testable.

**Tech Stack:** Rust, candle 0.8, existing picochat crate infrastructure (picochat-core, picochat-optim, picochat-data, picochat-engine, picochat-train, picochat-cli).

---

### Task 1: Tool Execution Engine — AST Parser

Create the `picochat-tool` crate with a sandboxed expression evaluator. This is the foundation for live tool execution during GRPO generation.

**Files:**
- Create: `crates/picochat-tool/Cargo.toml`
- Create: `crates/picochat-tool/src/lib.rs`
- Create: `crates/picochat-tool/src/ast.rs`
- Create: `crates/picochat-tool/src/evaluator.rs`
- Create: `crates/picochat-tool/tests/ast_test.rs`
- Create: `crates/picochat-tool/tests/evaluator_test.rs`
- Modify: `Cargo.toml` (workspace members)

**Context:**
- The tool engine runs expressions the model generates inside `<tool_call_start>...<tool_call_end>` markers
- Safety: 100ms timeout, no recursion > 10 levels, no side effects, no file/network access
- Supported: arithmetic (`+`,`-`,`*`,`/`,`%`,`**`), math functions (`sqrt`,`abs`,`sin`,`cos`,`log`,`ceil`,`floor`), string ops (`len("str")`,`"str".count("x")`,`"str".upper()`,`"str".lower()`), comparisons (`==`,`!=`,`<`,`>`,`<=`,`>=`)

**Step 1: Create crate boilerplate**

Add `picochat-tool` to workspace in `Cargo.toml`:

```toml
# Cargo.toml (workspace root) — add to members list:
"crates/picochat-tool",
```

Create `crates/picochat-tool/Cargo.toml`:

```toml
[package]
name = "picochat-tool"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = { workspace = true }
```

Create `crates/picochat-tool/src/lib.rs`:

```rust
pub mod ast;
pub mod evaluator;

pub use evaluator::{run_tool, ToolResult};
```

**Step 2: Write failing tests for the AST**

Create `crates/picochat-tool/tests/ast_test.rs`:

```rust
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
```

**Step 3: Run tests to verify they fail**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-tool --test ast_test`
Expected: FAIL (module and types don't exist yet)

**Step 4: Implement the AST module**

Create `crates/picochat-tool/src/ast.rs`:

```rust
use anyhow::{anyhow, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Number(f64),
    Str(String),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    DoubleStar,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
    LParen,
    RParen,
    Comma,
    Dot,
}

pub fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' | '\n' | '\r' => { i += 1; }
            '"' => {
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != '"' {
                    i += 1;
                }
                if i >= chars.len() {
                    return Err(anyhow!("unterminated string"));
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::Str(s));
                i += 1;
            }
            '\'' => {
                i += 1;
                let start = i;
                while i < chars.len() && chars[i] != '\'' {
                    i += 1;
                }
                if i >= chars.len() {
                    return Err(anyhow!("unterminated string"));
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::Str(s));
                i += 1;
            }
            c if c.is_ascii_digit() || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let val: f64 = num_str.parse().map_err(|_| anyhow!("invalid number: {}", num_str))?;
                tokens.push(Token::Number(val));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            '+' => { tokens.push(Token::Plus); i += 1; }
            '-' => { tokens.push(Token::Minus); i += 1; }
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    tokens.push(Token::DoubleStar);
                    i += 2;
                } else {
                    tokens.push(Token::Star);
                    i += 1;
                }
            }
            '/' => { tokens.push(Token::Slash); i += 1; }
            '%' => { tokens.push(Token::Percent); i += 1; }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Eq);
                    i += 2;
                } else {
                    return Err(anyhow!("unexpected '=' (assignment not allowed)"));
                }
            }
            '!' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Neq);
                    i += 2;
                } else {
                    return Err(anyhow!("unexpected '!'"));
                }
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Lte);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Gte);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
            }
            '(' => { tokens.push(Token::LParen); i += 1; }
            ')' => { tokens.push(Token::RParen); i += 1; }
            ',' => { tokens.push(Token::Comma); i += 1; }
            '.' => { tokens.push(Token::Dot); i += 1; }
            c => return Err(anyhow!("unexpected character: '{}'", c)),
        }
    }

    Ok(tokens)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    Str(String),
    UnaryMinus(Box<Expr>),
    BinOp { op: String, left: Box<Expr>, right: Box<Expr> },
    FnCall { name: String, args: Vec<Expr> },
    MethodCall { object: Box<Expr>, method: String, args: Vec<Expr> },
}

/// Recursive descent parser with standard operator precedence.
pub fn parse(tokens: &[Token]) -> Result<Expr> {
    let mut pos = 0;
    let expr = parse_comparison(tokens, &mut pos)?;
    if pos != tokens.len() {
        return Err(anyhow!("unexpected token at position {}", pos));
    }
    Ok(expr)
}

fn parse_comparison(tokens: &[Token], pos: &mut usize) -> Result<Expr> {
    let mut left = parse_additive(tokens, pos)?;
    while *pos < tokens.len() {
        let op = match &tokens[*pos] {
            Token::Eq => "==",
            Token::Neq => "!=",
            Token::Lt => "<",
            Token::Gt => ">",
            Token::Lte => "<=",
            Token::Gte => ">=",
            _ => break,
        };
        *pos += 1;
        let right = parse_additive(tokens, pos)?;
        left = Expr::BinOp {
            op: op.to_string(),
            left: Box::new(left),
            right: Box::new(right),
        };
    }
    Ok(left)
}

fn parse_additive(tokens: &[Token], pos: &mut usize) -> Result<Expr> {
    let mut left = parse_multiplicative(tokens, pos)?;
    while *pos < tokens.len() {
        let op = match &tokens[*pos] {
            Token::Plus => "+",
            Token::Minus => "-",
            _ => break,
        };
        *pos += 1;
        let right = parse_multiplicative(tokens, pos)?;
        left = Expr::BinOp {
            op: op.to_string(),
            left: Box::new(left),
            right: Box::new(right),
        };
    }
    Ok(left)
}

fn parse_multiplicative(tokens: &[Token], pos: &mut usize) -> Result<Expr> {
    let mut left = parse_power(tokens, pos)?;
    while *pos < tokens.len() {
        let op = match &tokens[*pos] {
            Token::Star => "*",
            Token::Slash => "/",
            Token::Percent => "%",
            _ => break,
        };
        *pos += 1;
        let right = parse_power(tokens, pos)?;
        left = Expr::BinOp {
            op: op.to_string(),
            left: Box::new(left),
            right: Box::new(right),
        };
    }
    Ok(left)
}

// Right-associative: 2**3**4 = 2**(3**4)
fn parse_power(tokens: &[Token], pos: &mut usize) -> Result<Expr> {
    let base = parse_unary(tokens, pos)?;
    if *pos < tokens.len() && tokens[*pos] == Token::DoubleStar {
        *pos += 1;
        let exp = parse_power(tokens, pos)?;
        Ok(Expr::BinOp {
            op: "**".to_string(),
            left: Box::new(base),
            right: Box::new(exp),
        })
    } else {
        Ok(base)
    }
}

fn parse_unary(tokens: &[Token], pos: &mut usize) -> Result<Expr> {
    if *pos < tokens.len() && tokens[*pos] == Token::Minus {
        *pos += 1;
        let inner = parse_postfix(tokens, pos)?;
        Ok(Expr::UnaryMinus(Box::new(inner)))
    } else {
        parse_postfix(tokens, pos)
    }
}

fn parse_postfix(tokens: &[Token], pos: &mut usize) -> Result<Expr> {
    let mut node = parse_primary(tokens, pos)?;
    // Handle method calls: node.method(args)
    while *pos < tokens.len() && tokens[*pos] == Token::Dot {
        *pos += 1;
        let method = match tokens.get(*pos) {
            Some(Token::Ident(name)) => { *pos += 1; name.clone() }
            _ => return Err(anyhow!("expected method name after '.'")),
        };
        if *pos < tokens.len() && tokens[*pos] == Token::LParen {
            *pos += 1;
            let args = parse_arg_list(tokens, pos)?;
            node = Expr::MethodCall {
                object: Box::new(node),
                method,
                args,
            };
        } else {
            return Err(anyhow!("expected '(' after method name"));
        }
    }
    Ok(node)
}

fn parse_primary(tokens: &[Token], pos: &mut usize) -> Result<Expr> {
    if *pos >= tokens.len() {
        return Err(anyhow!("unexpected end of expression"));
    }

    match &tokens[*pos] {
        Token::Number(n) => {
            let val = *n;
            *pos += 1;
            Ok(Expr::Number(val))
        }
        Token::Str(s) => {
            let val = s.clone();
            *pos += 1;
            Ok(Expr::Str(val))
        }
        Token::Ident(name) => {
            let name = name.clone();
            *pos += 1;
            // Function call: name(args)
            if *pos < tokens.len() && tokens[*pos] == Token::LParen {
                *pos += 1;
                let args = parse_arg_list(tokens, pos)?;
                Ok(Expr::FnCall { name, args })
            } else {
                Err(anyhow!("bare identifier '{}' not allowed (no variables)", name))
            }
        }
        Token::LParen => {
            *pos += 1;
            let inner = parse_comparison(tokens, pos)?;
            if *pos >= tokens.len() || tokens[*pos] != Token::RParen {
                return Err(anyhow!("expected ')'"));
            }
            *pos += 1;
            Ok(inner)
        }
        t => Err(anyhow!("unexpected token: {:?}", t)),
    }
}

fn parse_arg_list(tokens: &[Token], pos: &mut usize) -> Result<Vec<Expr>> {
    let mut args = Vec::new();
    if *pos < tokens.len() && tokens[*pos] == Token::RParen {
        *pos += 1;
        return Ok(args);
    }
    args.push(parse_comparison(tokens, pos)?);
    while *pos < tokens.len() && tokens[*pos] == Token::Comma {
        *pos += 1;
        args.push(parse_comparison(tokens, pos)?);
    }
    if *pos >= tokens.len() || tokens[*pos] != Token::RParen {
        return Err(anyhow!("expected ')' after arguments"));
    }
    *pos += 1;
    Ok(args)
}
```

**Step 5: Run tests to verify they pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-tool --test ast_test`
Expected: PASS (all 13 tests)

**Step 6: Write failing tests for the evaluator**

Create `crates/picochat-tool/tests/evaluator_test.rs`:

```rust
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
    // Whole numbers should display without decimal point
    assert_eq!(run_tool("2 + 3"), ToolResult::Value("5".to_string()));
    // Non-integers keep decimals
    assert_eq!(run_tool("1 / 3"), ToolResult::Value("0.3333333333333333".to_string()));
}
```

**Step 7: Run tests to verify they fail**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-tool --test evaluator_test`
Expected: FAIL

**Step 8: Implement the evaluator**

Create `crates/picochat-tool/src/evaluator.rs`:

```rust
use crate::ast::{tokenize, parse, Expr};

#[derive(Debug, Clone, PartialEq)]
pub enum ToolResult {
    Value(String),
    Error(String),
}

/// Run a tool expression string, returning either a value or an error.
pub fn run_tool(input: &str) -> ToolResult {
    let tokens = match tokenize(input) {
        Ok(t) => t,
        Err(e) => return ToolResult::Error(format!("parse error: {e}")),
    };
    if tokens.is_empty() {
        return ToolResult::Error("empty expression".to_string());
    }
    let ast = match parse(&tokens) {
        Ok(e) => e,
        Err(e) => return ToolResult::Error(format!("parse error: {e}")),
    };
    match compute(&ast, 0) {
        Ok(v) => ToolResult::Value(format_value(&v)),
        Err(e) => ToolResult::Error(e),
    }
}

#[derive(Debug, Clone)]
enum Value {
    Num(f64),
    Str(String),
    Bool(bool),
}

const MAX_DEPTH: usize = 10;

fn compute(node: &Expr, depth: usize) -> Result<Value, String> {
    if depth > MAX_DEPTH {
        return Err("maximum recursion depth exceeded".to_string());
    }

    match node {
        Expr::Number(n) => Ok(Value::Num(*n)),
        Expr::Str(s) => Ok(Value::Str(s.clone())),
        Expr::UnaryMinus(inner) => {
            match compute(inner, depth + 1)? {
                Value::Num(n) => Ok(Value::Num(-n)),
                _ => Err("unary minus requires a number".to_string()),
            }
        }
        Expr::BinOp { op, left, right } => {
            let l = compute(left, depth + 1)?;
            let r = compute(right, depth + 1)?;
            compute_binop(op, &l, &r)
        }
        Expr::FnCall { name, args } => {
            let evaluated: Result<Vec<Value>, String> =
                args.iter().map(|a| compute(a, depth + 1)).collect();
            compute_fn(name, &evaluated?)
        }
        Expr::MethodCall { object, method, args } => {
            let obj = compute(object, depth + 1)?;
            let evaluated: Result<Vec<Value>, String> =
                args.iter().map(|a| compute(a, depth + 1)).collect();
            compute_method(&obj, method, &evaluated?)
        }
    }
}

fn compute_binop(op: &str, left: &Value, right: &Value) -> Result<Value, String> {
    match (left, right) {
        (Value::Num(a), Value::Num(b)) => {
            let result = match op {
                "+" => *a + *b,
                "-" => *a - *b,
                "*" => *a * *b,
                "/" => {
                    if *b == 0.0 { return Err("division by zero".to_string()); }
                    *a / *b
                }
                "%" => {
                    if *b == 0.0 { return Err("division by zero".to_string()); }
                    *a % *b
                }
                "**" => a.powf(*b),
                "==" => return Ok(Value::Bool((a - b).abs() < 1e-10)),
                "!=" => return Ok(Value::Bool((a - b).abs() >= 1e-10)),
                "<" => return Ok(Value::Bool(*a < *b)),
                ">" => return Ok(Value::Bool(*a > *b)),
                "<=" => return Ok(Value::Bool(*a <= *b)),
                ">=" => return Ok(Value::Bool(*a >= *b)),
                _ => return Err(format!("unknown operator: {op}")),
            };
            Ok(Value::Num(result))
        }
        (Value::Str(a), Value::Str(b)) => {
            match op {
                "==" => Ok(Value::Bool(a == b)),
                "!=" => Ok(Value::Bool(a != b)),
                "+" => Ok(Value::Str(format!("{a}{b}"))),
                _ => Err(format!("operator '{op}' not supported for strings")),
            }
        }
        _ => Err(format!("type mismatch for operator '{op}'")),
    }
}

fn compute_fn(name: &str, args: &[Value]) -> Result<Value, String> {
    match name {
        "sqrt" => {
            let n = require_num(args, 0, "sqrt")?;
            Ok(Value::Num(n.sqrt()))
        }
        "abs" => {
            let n = require_num(args, 0, "abs")?;
            Ok(Value::Num(n.abs()))
        }
        "sin" => {
            let n = require_num(args, 0, "sin")?;
            Ok(Value::Num(n.sin()))
        }
        "cos" => {
            let n = require_num(args, 0, "cos")?;
            Ok(Value::Num(n.cos()))
        }
        "log" => {
            let n = require_num(args, 0, "log")?;
            Ok(Value::Num(n.ln()))
        }
        "ceil" => {
            let n = require_num(args, 0, "ceil")?;
            Ok(Value::Num(n.ceil()))
        }
        "floor" => {
            let n = require_num(args, 0, "floor")?;
            Ok(Value::Num(n.floor()))
        }
        "len" => {
            match args.first() {
                Some(Value::Str(s)) => Ok(Value::Num(s.len() as f64)),
                _ => Err("len() requires a string argument".to_string()),
            }
        }
        _ => Err(format!("unknown function: {name}")),
    }
}

fn compute_method(obj: &Value, method: &str, args: &[Value]) -> Result<Value, String> {
    match obj {
        Value::Str(s) => {
            match method {
                "count" => {
                    let substr = match args.first() {
                        Some(Value::Str(sub)) => sub,
                        _ => return Err("count() requires a string argument".to_string()),
                    };
                    Ok(Value::Num(s.matches(substr.as_str()).count() as f64))
                }
                "upper" => Ok(Value::Str(s.to_uppercase())),
                "lower" => Ok(Value::Str(s.to_lowercase())),
                _ => Err(format!("unknown string method: {method}")),
            }
        }
        _ => Err(format!("method calls only supported on strings, got {method}")),
    }
}

fn require_num(args: &[Value], idx: usize, fn_name: &str) -> Result<f64, String> {
    match args.get(idx) {
        Some(Value::Num(n)) => Ok(*n),
        Some(_) => Err(format!("{fn_name}() requires a numeric argument")),
        None => Err(format!("{fn_name}() missing argument")),
    }
}

/// Format a value for output. Whole numbers display without decimal point.
fn format_value(v: &Value) -> String {
    match v {
        Value::Num(n) => {
            if n.fract() == 0.0 && n.abs() < 1e15 {
                format!("{}", *n as i64)
            } else {
                format!("{n}")
            }
        }
        Value::Str(s) => s.clone(),
        Value::Bool(b) => format!("{b}"),
    }
}
```

**Step 9: Run tests to verify they pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-tool`
Expected: PASS (all AST and evaluator tests)

**Step 10: Commit**

```bash
git add crates/picochat-tool/ Cargo.toml
git commit -m "feat: add picochat-tool crate with sandboxed AST expression evaluator"
```

---

### Task 2: ARC Dataset Loader

Add ARC-Challenge dataset loading to `picochat-data`.

**Files:**
- Create: `crates/picochat-data/src/arc.rs`
- Create: `crates/picochat-data/tests/arc_test.rs`
- Modify: `crates/picochat-data/src/lib.rs:1-5`

**Context:**
- ARC-Challenge data is JSONL with `{"question": "...", "choices": ["A...", "B...", "C...", "D..."], "answer_key": "B"}`
- This follows the same pattern as the existing GSM8K loader in `picochat-eval/src/gsm8k.rs`
- Used by GRPO for science reasoning training data

**Step 1: Write failing tests**

Create `crates/picochat-data/tests/arc_test.rs`:

```rust
use picochat_data::arc::{ArcQuestion, load_arc_jsonl, format_arc_prompt};

#[test]
fn test_arc_question_parse() {
    let json = r#"{"question":"What causes seasons?","choices":["Tilt of Earth","Distance from Sun","Moon phases","Wind"],"answer_key":"A"}"#;
    let q: ArcQuestion = serde_json::from_str(json).unwrap();
    assert_eq!(q.question, "What causes seasons?");
    assert_eq!(q.choices.len(), 4);
    assert_eq!(q.answer_key, "A");
}

#[test]
fn test_arc_question_answer_index() {
    let q = ArcQuestion {
        question: "Test".to_string(),
        choices: vec!["a".into(), "b".into(), "c".into(), "d".into()],
        answer_key: "C".to_string(),
    };
    assert_eq!(q.answer_index(), Some(2));
}

#[test]
fn test_format_arc_prompt() {
    let exemplars = vec![
        ArcQuestion {
            question: "What is H2O?".to_string(),
            choices: vec!["Fire".into(), "Water".into(), "Air".into(), "Earth".into()],
            answer_key: "B".to_string(),
        },
    ];
    let test_q = ArcQuestion {
        question: "What is the Sun?".to_string(),
        choices: vec!["Star".into(), "Planet".into(), "Moon".into(), "Comet".into()],
        answer_key: "A".to_string(),
    };
    let prompt = format_arc_prompt(&exemplars, &test_q);
    assert!(prompt.contains("What is H2O?"));
    assert!(prompt.contains("(B) Water"));
    assert!(prompt.contains("Answer: B"));
    assert!(prompt.contains("What is the Sun?"));
    assert!(prompt.contains("(A) Star"));
    assert!(prompt.ends_with("Answer: "));
}

#[test]
fn test_load_arc_from_string() {
    let data = r#"{"question":"Q1","choices":["a","b","c","d"],"answer_key":"A"}
{"question":"Q2","choices":["w","x","y","z"],"answer_key":"D"}"#;
    let questions: Vec<ArcQuestion> = data.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(questions.len(), 2);
    assert_eq!(questions[1].answer_key, "D");
}
```

**Step 2: Run tests to verify they fail**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-data --test arc_test`
Expected: FAIL

**Step 3: Implement ARC loader**

Create `crates/picochat-data/src/arc.rs`:

```rust
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcQuestion {
    pub question: String,
    pub choices: Vec<String>,
    pub answer_key: String,
}

impl ArcQuestion {
    /// Returns the 0-based index of the correct answer (A=0, B=1, C=2, D=3).
    pub fn answer_index(&self) -> Option<usize> {
        match self.answer_key.as_str() {
            "A" => Some(0),
            "B" => Some(1),
            "C" => Some(2),
            "D" => Some(3),
            _ => None,
        }
    }
}

/// Load ARC questions from a JSONL file.
pub fn load_arc_jsonl(path: &str) -> Result<Vec<ArcQuestion>> {
    let content = std::fs::read_to_string(path)?;
    let mut questions = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let q: ArcQuestion = serde_json::from_str(line)?;
        questions.push(q);
    }
    Ok(questions)
}

/// Format a few-shot ARC prompt with exemplars and a test question.
pub fn format_arc_prompt(exemplars: &[ArcQuestion], question: &ArcQuestion) -> String {
    let mut prompt = String::new();
    let labels = ["A", "B", "C", "D"];

    for ex in exemplars {
        prompt.push_str(&format!("Q: {}\n", ex.question));
        for (i, choice) in ex.choices.iter().enumerate() {
            if i < labels.len() {
                prompt.push_str(&format!("({}) {}\n", labels[i], choice));
            }
        }
        prompt.push_str(&format!("Answer: {}\n\n", ex.answer_key));
    }

    prompt.push_str(&format!("Q: {}\n", question.question));
    for (i, choice) in question.choices.iter().enumerate() {
        if i < labels.len() {
            prompt.push_str(&format!("({}) {}\n", labels[i], choice));
        }
    }
    prompt.push_str("Answer: ");
    prompt
}
```

Add to `crates/picochat-data/src/lib.rs`:

```rust
// picochat-data: data loading and preprocessing
pub mod arc;
pub mod dataloader;
pub mod mixture;
pub mod parquet;
pub mod sft;
```

**Step 4: Run tests to verify they pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-data --test arc_test`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add crates/picochat-data/src/arc.rs crates/picochat-data/src/lib.rs crates/picochat-data/tests/arc_test.rs
git commit -m "feat: add ARC-Challenge dataset loader for GRPO training"
```

---

### Task 3: Tool-Use Scenario Loader

Add tool-use training data loading to `picochat-data`.

**Files:**
- Create: `crates/picochat-data/src/tool_data.rs`
- Create: `crates/picochat-data/tests/tool_data_test.rs`
- Modify: `crates/picochat-data/src/lib.rs`

**Context:**
- Tool-use scenarios are JSONL with `{"prompt": "...", "expected_answer": "...", "requires_tool": true}`
- Used for GRPO training where the model should learn to invoke tools
- 2-shot exemplar format showing `<think>` + `<tool_call>` usage

**Step 1: Write failing tests**

Create `crates/picochat-data/tests/tool_data_test.rs`:

```rust
use picochat_data::tool_data::{ToolScenario, load_tool_scenarios, format_tool_prompt};

#[test]
fn test_tool_scenario_parse() {
    let json = r#"{"prompt":"What is 347 * 892?","expected_answer":"309524","requires_tool":true}"#;
    let s: ToolScenario = serde_json::from_str(json).unwrap();
    assert_eq!(s.prompt, "What is 347 * 892?");
    assert_eq!(s.expected_answer, "309524");
    assert!(s.requires_tool);
}

#[test]
fn test_tool_scenario_no_tool() {
    let json = r#"{"prompt":"What is 2 + 2?","expected_answer":"4","requires_tool":false}"#;
    let s: ToolScenario = serde_json::from_str(json).unwrap();
    assert!(!s.requires_tool);
}

#[test]
fn test_format_tool_prompt() {
    let scenario = ToolScenario {
        prompt: "What is 123 * 456?".to_string(),
        expected_answer: "56088".to_string(),
        requires_tool: true,
    };
    let prompt = format_tool_prompt(&scenario);
    assert!(prompt.contains("123 * 456"));
    // Should contain exemplars showing tool usage
    assert!(prompt.contains("<tool_call_start>"));
    assert!(prompt.contains("<tool_result_start>"));
    assert!(prompt.contains("<think_start>"));
}

#[test]
fn test_load_tool_scenarios_from_string() {
    let data = r#"{"prompt":"P1","expected_answer":"A1","requires_tool":true}
{"prompt":"P2","expected_answer":"A2","requires_tool":false}"#;
    let scenarios: Vec<ToolScenario> = data.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(scenarios.len(), 2);
}
```

**Step 2: Run tests to verify they fail**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-data --test tool_data_test`
Expected: FAIL

**Step 3: Implement tool data loader**

Create `crates/picochat-data/src/tool_data.rs`:

```rust
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolScenario {
    pub prompt: String,
    pub expected_answer: String,
    pub requires_tool: bool,
}

/// Load tool-use scenarios from a JSONL file.
pub fn load_tool_scenarios(path: &str) -> Result<Vec<ToolScenario>> {
    let content = std::fs::read_to_string(path)?;
    let mut scenarios = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let s: ToolScenario = serde_json::from_str(line)?;
        scenarios.push(s);
    }
    Ok(scenarios)
}

/// Format a tool-use prompt with 2-shot exemplars demonstrating think + tool_call usage.
pub fn format_tool_prompt(scenario: &ToolScenario) -> String {
    let mut prompt = String::from(
        "You can use a calculator by writing expressions between <tool_call_start> and <tool_call_end> tags. \
         The result will appear between <tool_result_start> and <tool_result_end> tags. \
         Think through problems step by step using <think_start> and <think_end> tags.\n\n"
    );

    // Exemplar 1: arithmetic with tool
    prompt.push_str("Q: What is 15 * 23?\n");
    prompt.push_str("<think_start>I need to multiply 15 by 23. Let me use the calculator.</think_end>\n");
    prompt.push_str("<tool_call_start>15 * 23<tool_call_end>\n");
    prompt.push_str("<tool_result_start>345<tool_result_end>\n");
    prompt.push_str("#### 345\n\n");

    // Exemplar 2: string operation with tool
    prompt.push_str("Q: How many times does the letter 'a' appear in 'banana'?\n");
    prompt.push_str("<think_start>I need to count occurrences of 'a' in 'banana'. I'll use the count method.</think_end>\n");
    prompt.push_str("<tool_call_start>\"banana\".count(\"a\")<tool_call_end>\n");
    prompt.push_str("<tool_result_start>3<tool_result_end>\n");
    prompt.push_str("#### 3\n\n");

    // Test question
    prompt.push_str(&format!("Q: {}\n", scenario.prompt));

    prompt
}
```

Add `tool_data` to `crates/picochat-data/src/lib.rs`:

```rust
// picochat-data: data loading and preprocessing
pub mod arc;
pub mod dataloader;
pub mod mixture;
pub mod parquet;
pub mod sft;
pub mod tool_data;
```

**Step 4: Run tests to verify they pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-data --test tool_data_test`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```bash
git add crates/picochat-data/src/tool_data.rs crates/picochat-data/src/lib.rs crates/picochat-data/tests/tool_data_test.rs
git commit -m "feat: add tool-use scenario data loader for GRPO training"
```

---

### Task 4: Generation with Log-Probs and Tool Interleaving

Add a generation function that returns per-token log-probabilities and supports live tool invocation during generation.

**Files:**
- Create: `crates/picochat-engine/src/generate_with_logprobs.rs`
- Create: `crates/picochat-engine/tests/generate_logprobs_test.rs`
- Modify: `crates/picochat-engine/src/lib.rs:1-2`
- Modify: `crates/picochat-engine/Cargo.toml` (add picochat-tool dep)

**Context:**
- Extends the existing `generate.rs` pattern but returns `(Vec<u32>, Vec<f32>)` — token IDs + log-probs
- Tool handling: when model emits `tool_call_start_id`, collect tokens until `tool_call_end_id`, run via `picochat_tool::run_tool`, inject result tokens
- Tool result tokens are NOT part of the generated output (model didn't produce them) — they're injected into the KV cache context
- Max 3 tool calls per generation

**Step 1: Update Cargo.toml**

Add `picochat-tool` to `crates/picochat-engine/Cargo.toml`:

```toml
[dependencies]
picochat-core = { path = "../picochat-core" }
picochat-tool = { path = "../picochat-tool" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
anyhow = { workspace = true }
rand = { workspace = true }
```

**Step 2: Write failing tests**

Create `crates/picochat-engine/tests/generate_logprobs_test.rs`:

```rust
use candle_core::{Device, DType};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_engine::generate_with_logprobs::{generate_with_logprobs, LogprobGenerationConfig};
use picochat_engine::sampling::SamplingParams;

fn make_model() -> (GPT, Device) {
    let config = GPTConfig::from_depth(2);
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = GPT::new(&config, vb).unwrap();
    (model, device)
}

#[test]
fn test_logprobs_returns_ids_and_probs() {
    let (model, device) = make_model();
    let config = LogprobGenerationConfig {
        max_new_tokens: 10,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![],
        tool_call_start_id: None,
        tool_call_end_id: None,
        tool_result_start_id: None,
        tool_result_end_id: None,
        max_tool_calls: 3,
    };
    let (ids, logprobs) = generate_with_logprobs(
        &model, &[1, 2, 3], &config, &device, None,
    ).unwrap();
    assert_eq!(ids.len(), 10);
    assert_eq!(logprobs.len(), 10);
    // Log-probs should be negative (log of probability < 1)
    for &lp in &logprobs {
        assert!(lp <= 0.0, "logprob {} should be <= 0", lp);
    }
}

#[test]
fn test_logprobs_greedy_deterministic() {
    let (model, device) = make_model();
    let config = LogprobGenerationConfig {
        max_new_tokens: 5,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![],
        tool_call_start_id: None,
        tool_call_end_id: None,
        tool_result_start_id: None,
        tool_result_end_id: None,
        max_tool_calls: 3,
    };
    let (ids1, lp1) = generate_with_logprobs(
        &model, &[1, 2], &config, &device, None,
    ).unwrap();
    let (ids2, lp2) = generate_with_logprobs(
        &model, &[1, 2], &config, &device, None,
    ).unwrap();
    assert_eq!(ids1, ids2);
    for (a, b) in lp1.iter().zip(lp2.iter()) {
        assert!((a - b).abs() < 1e-5, "logprobs differ: {} vs {}", a, b);
    }
}

#[test]
fn test_logprobs_with_stop_token() {
    let (model, device) = make_model();
    let config_no_stop = LogprobGenerationConfig {
        max_new_tokens: 5,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![],
        tool_call_start_id: None,
        tool_call_end_id: None,
        tool_result_start_id: None,
        tool_result_end_id: None,
        max_tool_calls: 3,
    };
    let (ids, _) = generate_with_logprobs(
        &model, &[1, 2], &config_no_stop, &device, None,
    ).unwrap();

    let config = LogprobGenerationConfig {
        max_new_tokens: 100,
        sampling: SamplingParams::greedy(),
        stop_tokens: vec![ids[0]],
        tool_call_start_id: None,
        tool_call_end_id: None,
        tool_result_start_id: None,
        tool_result_end_id: None,
        max_tool_calls: 3,
    };
    let (stopped_ids, stopped_lps) = generate_with_logprobs(
        &model, &[1, 2], &config, &device, None,
    ).unwrap();
    assert_eq!(stopped_ids.len(), 1);
    assert_eq!(stopped_lps.len(), 1);
}
```

**Step 3: Run tests to verify they fail**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-engine --test generate_logprobs_test`
Expected: FAIL

**Step 4: Implement generation with log-probs**

Create `crates/picochat-engine/src/generate_with_logprobs.rs`:

```rust
use anyhow::Result;
use candle_core::{Device, Tensor};
use picochat_core::kv_cache::KVCache;
use picochat_core::model::GPT;
use picochat_tokenizer::Tokenizer;

use crate::sampling::{sample, SamplingParams};

pub struct LogprobGenerationConfig {
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
    pub stop_tokens: Vec<u32>,
    /// If set, enables tool interleaving during generation.
    pub tool_call_start_id: Option<u32>,
    pub tool_call_end_id: Option<u32>,
    pub tool_result_start_id: Option<u32>,
    pub tool_result_end_id: Option<u32>,
    pub max_tool_calls: usize,
}

/// Generate tokens with per-token log-probabilities, optionally interleaving tool calls.
///
/// When `tokenizer` is provided and tool token IDs are set, the model can invoke tools:
/// 1. Model generates <tool_call_start> ... expression ... <tool_call_end>
/// 2. Expression is run via picochat_tool::run_tool
/// 3. <tool_result_start> result_tokens <tool_result_end> are fed into the KV cache
/// 4. Generation continues from there
///
/// Tool result tokens don't appear in the returned token_ids/logprobs (the model didn't generate them).
pub fn generate_with_logprobs(
    model: &GPT,
    prompt_tokens: &[u32],
    config: &LogprobGenerationConfig,
    device: &Device,
    tokenizer: Option<&Tokenizer>,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut cache = KVCache::new(model.n_layers());

    // Prefill
    let prompt = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = model.forward_with_cache(&prompt, &mut cache)?;

    let last_logits = logits.flatten(0, 1)?;
    let t = prompt_tokens.len();
    let last_row = last_logits.get(t - 1)?;
    let logit_vec: Vec<f32> = last_row.to_vec1()?;

    let lp = compute_log_softmax(&logit_vec);
    let mut next_token = sample(&logit_vec, &config.sampling) as u32;

    let mut output_ids = Vec::with_capacity(config.max_new_tokens);
    let mut output_logprobs = Vec::with_capacity(config.max_new_tokens);

    output_ids.push(next_token);
    output_logprobs.push(lp[next_token as usize]);

    if config.stop_tokens.contains(&next_token) {
        return Ok((output_ids, output_logprobs));
    }

    let mut tool_calls_made = 0usize;
    let mut collecting_tool_call = false;
    let mut tool_call_tokens: Vec<u32> = Vec::new();

    for _ in 1..config.max_new_tokens {
        // Check if we just generated a tool_call_start token
        if let (Some(start_id), Some(end_id)) = (config.tool_call_start_id, config.tool_call_end_id) {
            if next_token == start_id && tool_calls_made < config.max_tool_calls {
                collecting_tool_call = true;
                tool_call_tokens.clear();
            } else if collecting_tool_call && next_token == end_id {
                collecting_tool_call = false;
                tool_calls_made += 1;

                // Run the tool call
                if let Some(tok) = tokenizer {
                    let expr_text = tok.decode(&tool_call_tokens);
                    let result = picochat_tool::run_tool(&expr_text);
                    let result_text = match result {
                        picochat_tool::ToolResult::Value(v) => v,
                        picochat_tool::ToolResult::Error(e) => format!("Error: {e}"),
                    };

                    // Inject tool result tokens into context
                    if let (Some(rs_id), Some(re_id)) = (config.tool_result_start_id, config.tool_result_end_id) {
                        let result_token_ids = tok.encode(&result_text)?;
                        let mut inject: Vec<u32> = Vec::with_capacity(result_token_ids.len() + 2);
                        inject.push(rs_id);
                        inject.extend_from_slice(&result_token_ids);
                        inject.push(re_id);

                        // Feed result tokens through the model to update KV cache
                        let inject_tensor = Tensor::new(&inject[..], device)?.unsqueeze(0)?;
                        let inject_logits = model.forward_with_cache(&inject_tensor, &mut cache)?;

                        // Get logits from the last injected position for next token prediction
                        let inject_len = inject.len();
                        let last_inject = inject_logits.flatten(0, 1)?.get(inject_len - 1)?;
                        let logit_vec: Vec<f32> = last_inject.to_vec1()?;
                        let lp = compute_log_softmax(&logit_vec);
                        next_token = sample(&logit_vec, &config.sampling) as u32;
                        output_ids.push(next_token);
                        output_logprobs.push(lp[next_token as usize]);

                        if config.stop_tokens.contains(&next_token) {
                            break;
                        }
                        continue;
                    }
                }
                tool_call_tokens.clear();
            } else if collecting_tool_call {
                tool_call_tokens.push(next_token);
            }
        }

        let input = Tensor::new(&[[next_token]], device)?;
        let logits = model.forward_with_cache(&input, &mut cache)?;
        let logit_vec: Vec<f32> = logits.flatten_all()?.to_vec1()?;

        let lp = compute_log_softmax(&logit_vec);
        next_token = sample(&logit_vec, &config.sampling) as u32;

        output_ids.push(next_token);
        output_logprobs.push(lp[next_token as usize]);

        if config.stop_tokens.contains(&next_token) {
            break;
        }
    }

    Ok((output_ids, output_logprobs))
}

/// Compute log-softmax over a logit vector. Returns log-probabilities.
fn compute_log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_sum = sum_exp.ln() + max;
    logits.iter().map(|&l| l - log_sum).collect()
}
```

Update `crates/picochat-engine/src/lib.rs`:

```rust
pub mod generate;
pub mod generate_with_logprobs;
pub mod sampling;
```

**Step 5: Run tests to verify they pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-engine --test generate_logprobs_test`
Expected: PASS (all 3 tests)

**Step 6: Commit**

```bash
git add crates/picochat-engine/
git commit -m "feat: add generation with per-token log-probs and tool interleaving"
```

---

### Task 5: Value Head (ORM)

Add a lightweight outcome reward model as a single linear projection from the model's last hidden state to a scalar.

**Files:**
- Create: `crates/picochat-train/src/value_head.rs`
- Create: `crates/picochat-train/tests/value_head_test.rs`
- Modify: `crates/picochat-train/src/lib.rs`

**Context:**
- The value head is a single `Linear` layer: `n_embd -> 1`
- Input: last hidden state at the `<assistant_end>` position (before lm_head)
- Output: scalar quality estimate
- Trained with MSE loss against rule-based reward total
- Has its own VarMap so it can be saved/loaded independently from the policy model

**Step 1: Write failing tests**

Create `crates/picochat-train/tests/value_head_test.rs`:

```rust
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_train::value_head::ValueHead;

#[test]
fn test_value_head_forward_shape() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vh = ValueHead::new(64, vb).unwrap();

    // (batch=2, n_embd=64) -> (batch=2, 1)
    let hidden = Tensor::randn(0.0f32, 1.0, (2, 64), &device).unwrap();
    let out = vh.forward(&hidden).unwrap();
    assert_eq!(out.dims(), &[2, 1]);
}

#[test]
fn test_value_head_mse_loss() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vh = ValueHead::new(64, vb).unwrap();

    let hidden = Tensor::randn(0.0f32, 1.0, (4, 64), &device).unwrap();
    let targets = Tensor::new(&[0.5f32, 1.0, 0.0, 0.8], &device).unwrap();
    let loss = vh.mse_loss(&hidden, &targets).unwrap();

    let loss_val: f32 = loss.to_scalar().unwrap();
    assert!(loss_val >= 0.0, "MSE loss should be non-negative");
}

#[test]
fn test_value_head_output_is_scalar_per_sample() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let vh = ValueHead::new(128, vb).unwrap();

    let hidden = Tensor::randn(0.0f32, 1.0, (3, 128), &device).unwrap();
    let out = vh.forward(&hidden).unwrap();
    let scores = out.squeeze(1).unwrap();
    assert_eq!(scores.dims(), &[3]);
}
```

**Step 2: Run tests to verify they fail**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test value_head_test`
Expected: FAIL

**Step 3: Implement value head**

Create `crates/picochat-train/src/value_head.rs`:

```rust
use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

/// Lightweight outcome reward model: single linear projection from hidden state to scalar.
pub struct ValueHead {
    proj: Linear,
}

impl ValueHead {
    pub fn new(n_embd: usize, vb: VarBuilder) -> Result<Self> {
        let proj = linear_no_bias(n_embd, 1, vb.pp("value_head"))?;
        Ok(Self { proj })
    }

    /// Forward pass: (batch, n_embd) -> (batch, 1)
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.proj.forward(hidden_states)
    }

    /// Compute MSE loss between predicted values and target rewards.
    /// hidden_states: (batch, n_embd), targets: (batch,)
    pub fn mse_loss(&self, hidden_states: &Tensor, targets: &Tensor) -> Result<Tensor> {
        let predicted = self.forward(hidden_states)?.squeeze(1)?;
        let diff = (&predicted - targets)?;
        let sq = (&diff * &diff)?;
        sq.mean_all()
    }
}
```

Add to `crates/picochat-train/src/lib.rs`:

```rust
// picochat-train: training loop
pub mod checkpoint;
pub mod metrics;
pub mod pretrain;
pub mod sft;
pub mod trainer;
pub mod value_head;
```

**Step 4: Run tests to verify they pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test value_head_test`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add crates/picochat-train/src/value_head.rs crates/picochat-train/src/lib.rs crates/picochat-train/tests/value_head_test.rs
git commit -m "feat: add value head (ORM) for GRPO reward prediction"
```

---

### Task 6: Reward Functions

Implement the composite reward system: accuracy, format, tool-use, length penalty.

**Files:**
- Create: `crates/picochat-train/src/rewards.rs`
- Create: `crates/picochat-train/tests/rewards_test.rs`
- Modify: `crates/picochat-train/src/lib.rs`
- Modify: `crates/picochat-train/Cargo.toml` (add picochat-tool dep)

**Context:**
- Reward weights: accuracy=1.0, format=0.2, tool_use=0.3, length_penalty=-0.1
- ORM score is added separately in the GRPO loop (not here — avoids circularity)
- Answer extraction: strip `<think>` blocks, find `####` or letter answer
- Format reward: check `<think_start>`/`<think_end>` are properly paired and before final answer
- Tool use reward: 1.0 for correct+useful, 0.5 for correct syntax but unused, 0.0 otherwise

**Step 1: Update Cargo.toml**

Add `picochat-tool` to `crates/picochat-train/Cargo.toml`:

```toml
[dependencies]
picochat-core = { path = "../picochat-core" }
picochat-optim = { path = "../picochat-optim" }
picochat-data = { path = "../picochat-data" }
picochat-tokenizer = { path = "../picochat-tokenizer" }
picochat-tool = { path = "../picochat-tool" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
anyhow = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
```

**Step 2: Write failing tests**

Create `crates/picochat-train/tests/rewards_test.rs`:

```rust
use picochat_train::rewards::{
    extract_final_answer, strip_think_blocks,
    accuracy_reward, format_reward, tool_use_reward, length_penalty_reward,
    composite_reward, RewardWeights, TaskType,
};

#[test]
fn test_strip_think_blocks() {
    let text = "Before <think_start>some reasoning<think_end> after #### 42";
    let stripped = strip_think_blocks(text);
    assert_eq!(stripped, "Before  after #### 42");
}

#[test]
fn test_strip_think_blocks_multiple() {
    let text = "<think_start>first<think_end>middle<think_start>second<think_end>end";
    let stripped = strip_think_blocks(text);
    assert_eq!(stripped, "middleend");
}

#[test]
fn test_strip_think_blocks_none() {
    let text = "no thinking here #### 5";
    let stripped = strip_think_blocks(text);
    assert_eq!(stripped, "no thinking here #### 5");
}

#[test]
fn test_extract_final_answer_math() {
    let text = "work... #### 42";
    assert_eq!(extract_final_answer(text, TaskType::Math), Some("42".to_string()));
}

#[test]
fn test_extract_final_answer_mc() {
    let text = "The answer is B";
    assert_eq!(extract_final_answer(text, TaskType::MultipleChoice), Some("B".to_string()));
}

#[test]
fn test_extract_final_answer_mc_from_choices() {
    let text = "After analysis, C is correct because...";
    assert_eq!(extract_final_answer(text, TaskType::MultipleChoice), Some("C".to_string()));
}

#[test]
fn test_accuracy_reward_math_correct() {
    assert_eq!(accuracy_reward("#### 42", "42", TaskType::Math), 1.0);
}

#[test]
fn test_accuracy_reward_math_wrong() {
    assert_eq!(accuracy_reward("#### 43", "42", TaskType::Math), 0.0);
}

#[test]
fn test_accuracy_reward_mc_correct() {
    assert_eq!(accuracy_reward("The answer is B", "B", TaskType::MultipleChoice), 1.0);
}

#[test]
fn test_format_reward_valid() {
    let text = "<think_start>reasoning<think_end>\n#### 42";
    assert_eq!(format_reward(text), 1.0);
}

#[test]
fn test_format_reward_missing_think() {
    let text = "#### 42";
    assert_eq!(format_reward(text), 0.0);
}

#[test]
fn test_format_reward_malformed_think() {
    let text = "<think_start>reasoning\n#### 42";
    assert_eq!(format_reward(text), 0.0);
}

#[test]
fn test_format_reward_think_after_answer() {
    let text = "#### 42\n<think_start>oops<think_end>";
    assert_eq!(format_reward(text), 0.0);
}

#[test]
fn test_tool_use_reward_correct_and_useful() {
    let text = "<tool_call_start>347 * 892<tool_call_end>\n<tool_result_start>309524<tool_result_end>\n#### 309524";
    assert_eq!(tool_use_reward(text, "309524", true), 1.0);
}

#[test]
fn test_tool_use_reward_correct_syntax_but_wrong() {
    let text = "<tool_call_start>347 + 892<tool_call_end>\n<tool_result_start>1239<tool_result_end>\n#### 1239";
    assert_eq!(tool_use_reward(text, "309524", true), 0.5);
}

#[test]
fn test_tool_use_reward_no_tool_when_needed() {
    let text = "#### 309524";
    assert_eq!(tool_use_reward(text, "309524", true), 0.0);
}

#[test]
fn test_tool_use_reward_no_tool_not_needed() {
    let text = "#### 4";
    assert_eq!(tool_use_reward(text, "4", false), 0.0);
}

#[test]
fn test_length_penalty() {
    assert_eq!(length_penalty_reward(50, 100), 0.0);
    assert_eq!(length_penalty_reward(100, 100), 0.0);
    let penalty = length_penalty_reward(150, 100);
    assert!((penalty - 0.5).abs() < 0.01);
}

#[test]
fn test_composite_reward() {
    let weights = RewardWeights::default();
    let text = "<think_start>Let me think<think_end>\n#### 42";
    let score = composite_reward(text, "42", TaskType::Math, false, 50, 100, &weights);
    // accuracy=1.0*1.0 + format=0.2*1.0 + tool=0.3*0.0 + length=(-0.1)*0.0 = 1.2
    assert!((score - 1.2).abs() < 0.01, "got {score}");
}
```

**Step 3: Run tests to verify they fail**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test rewards_test`
Expected: FAIL

**Step 4: Implement reward functions**

Create `crates/picochat-train/src/rewards.rs`:

```rust
#[derive(Debug, Clone, Copy)]
pub enum TaskType {
    Math,
    MultipleChoice,
    ToolUse,
}

#[derive(Debug, Clone)]
pub struct RewardWeights {
    pub accuracy: f64,
    pub format: f64,
    pub tool_use: f64,
    pub length_penalty: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            accuracy: 1.0,
            format: 0.2,
            tool_use: 0.3,
            length_penalty: -0.1,
        }
    }
}

/// Strip all `<think_start>...<think_end>` blocks from text.
pub fn strip_think_blocks(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(start) = remaining.find("<think_start>") {
        result.push_str(&remaining[..start]);
        match remaining[start..].find("<think_end>") {
            Some(end_offset) => {
                remaining = &remaining[start + end_offset + "<think_end>".len()..];
            }
            None => {
                return result;
            }
        }
    }
    result.push_str(remaining);
    result
}

/// Extract the final answer from model output after stripping think blocks.
pub fn extract_final_answer(text: &str, task_type: TaskType) -> Option<String> {
    let stripped = strip_think_blocks(text);
    match task_type {
        TaskType::Math | TaskType::ToolUse => {
            if let Some(pos) = stripped.rfind("####") {
                let after = stripped[pos + 4..].trim();
                let answer = after.replace(',', "");
                let answer = answer.trim().to_string();
                if answer.is_empty() { None } else { Some(answer) }
            } else {
                None
            }
        }
        TaskType::MultipleChoice => {
            let stripped_trimmed = stripped.trim();
            for pattern in ["Answer: ", "answer is ", "answer: "] {
                if let Some(pos) = stripped_trimmed.rfind(pattern) {
                    let after = stripped_trimmed[pos + pattern.len()..].trim();
                    if let Some(ch) = after.chars().next() {
                        if "ABCD".contains(ch) {
                            return Some(ch.to_string());
                        }
                    }
                }
            }
            for ch in stripped_trimmed.chars().rev() {
                if "ABCD".contains(ch) {
                    return Some(ch.to_string());
                }
            }
            None
        }
    }
}

/// Accuracy reward: 1.0 if extracted answer matches ground truth, 0.0 otherwise.
pub fn accuracy_reward(text: &str, ground_truth: &str, task_type: TaskType) -> f64 {
    match extract_final_answer(text, task_type) {
        Some(answer) => {
            if answer.trim() == ground_truth.trim() { 1.0 } else { 0.0 }
        }
        None => 0.0,
    }
}

/// Format reward: 1.0 if think blocks are properly structured and appear before the answer.
pub fn format_reward(text: &str) -> f64 {
    let has_start = text.contains("<think_start>");
    let has_end = text.contains("<think_end>");

    if !has_start || !has_end {
        return 0.0;
    }

    let start_count = text.matches("<think_start>").count();
    let end_count = text.matches("<think_end>").count();
    if start_count != end_count {
        return 0.0;
    }

    let last_think_end = match text.rfind("<think_end>") {
        Some(pos) => pos,
        None => return 0.0,
    };

    let answer_pos = text.rfind("####")
        .or_else(|| text.rfind("Answer: "));

    match answer_pos {
        Some(pos) if pos > last_think_end => 1.0,
        Some(_) => 0.0,
        None => 1.0,
    }
}

/// Tool use reward: 1.0 for correct invocation with useful result, 0.5 for valid syntax, 0.0 otherwise.
pub fn tool_use_reward(text: &str, ground_truth: &str, requires_tool: bool) -> f64 {
    let has_tool_call = text.contains("<tool_call_start>") && text.contains("<tool_call_end>");

    if !has_tool_call {
        return 0.0;
    }

    if !requires_tool {
        return 0.0;
    }

    let answer_correct = match extract_final_answer(text, TaskType::Math) {
        Some(answer) => answer.trim() == ground_truth.trim(),
        None => false,
    };

    if answer_correct { 1.0 } else { 0.5 }
}

/// Length penalty: 0.0 if under target, proportional penalty if over.
pub fn length_penalty_reward(num_tokens: usize, target_len: usize) -> f64 {
    if num_tokens <= target_len || target_len == 0 {
        0.0
    } else {
        (num_tokens - target_len) as f64 / target_len as f64
    }
}

/// Compute composite reward (excluding ORM — that's added in the GRPO loop).
pub fn composite_reward(
    text: &str,
    ground_truth: &str,
    task_type: TaskType,
    requires_tool: bool,
    num_tokens: usize,
    target_len: usize,
    weights: &RewardWeights,
) -> f64 {
    let acc = accuracy_reward(text, ground_truth, task_type);
    let fmt = format_reward(text);
    let tool = tool_use_reward(text, ground_truth, requires_tool);
    let len_pen = length_penalty_reward(num_tokens, target_len);

    weights.accuracy * acc
        + weights.format * fmt
        + weights.tool_use * tool
        + weights.length_penalty * len_pen
}
```

Add to `crates/picochat-train/src/lib.rs`:

```rust
// picochat-train: training loop
pub mod checkpoint;
pub mod metrics;
pub mod pretrain;
pub mod rewards;
pub mod sft;
pub mod trainer;
pub mod value_head;
```

**Step 5: Run tests to verify they pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test rewards_test`
Expected: PASS (all 19 tests)

**Step 6: Commit**

```bash
git add crates/picochat-train/
git commit -m "feat: add composite reward functions for GRPO (accuracy, format, tool-use, length)"
```

---

### Task 7: GRPO Training Loop

The core GRPO training loop: multi-sample generation, reward, advantage normalization, clipped policy gradient + KL penalty + value head loss.

**Files:**
- Create: `crates/picochat-train/src/grpo.rs`
- Create: `crates/picochat-train/tests/grpo_test.rs`
- Modify: `crates/picochat-train/src/lib.rs`
- Modify: `crates/picochat-train/Cargo.toml` (add picochat-engine, rand deps)

**Context:**
- Depends on: generate_with_logprobs (Task 4), value_head (Task 5), rewards (Task 6), data loaders (Tasks 2-3)
- Algorithm: sample G=16 completions per prompt, compute rewards, normalize advantages, clipped policy gradient
- Memory optimization: generate all completions without grad, then do gradient pass on subset
- Reference model is a frozen copy for KL penalty
- Hyperparameters: eps=0.2, beta=0.04, alpha=0.5, LR=1e-6

**Step 1: Update Cargo.toml**

Add `picochat-engine` and `rand` to `crates/picochat-train/Cargo.toml`:

```toml
[dependencies]
picochat-core = { path = "../picochat-core" }
picochat-optim = { path = "../picochat-optim" }
picochat-data = { path = "../picochat-data" }
picochat-tokenizer = { path = "../picochat-tokenizer" }
picochat-tool = { path = "../picochat-tool" }
picochat-engine = { path = "../picochat-engine" }
picochat-eval = { path = "../picochat-eval" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
anyhow = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
rand = { workspace = true }
```

**Step 2: Write failing tests**

Create `crates/picochat-train/tests/grpo_test.rs`:

```rust
use picochat_train::grpo::{
    normalize_advantages, GrpoConfig,
    compute_clipped_objective, compute_kl_penalty,
};

#[test]
fn test_normalize_advantages() {
    let rewards = vec![1.0, 0.5, 0.0, 0.8];
    let advantages = normalize_advantages(&rewards);
    assert_eq!(advantages.len(), 4);

    let mean: f64 = advantages.iter().sum::<f64>() / advantages.len() as f64;
    assert!(mean.abs() < 1e-10, "mean={mean}");

    let var: f64 = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / advantages.len() as f64;
    assert!((var.sqrt() - 1.0).abs() < 0.1, "std={}", var.sqrt());

    let max_idx = advantages.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;
    assert_eq!(max_idx, 0);
}

#[test]
fn test_normalize_advantages_all_same() {
    let rewards = vec![0.5, 0.5, 0.5, 0.5];
    let advantages = normalize_advantages(&rewards);
    for &a in &advantages {
        assert!(a.abs() < 1e-6, "expected 0, got {a}");
    }
}

#[test]
fn test_compute_clipped_objective() {
    let obj = compute_clipped_objective(1.0, 1.0, 0.2);
    assert!((obj - 1.0).abs() < 1e-6);

    let obj = compute_clipped_objective(1.5, 1.0, 0.2);
    assert!((obj - 1.2).abs() < 1e-6);

    let obj = compute_clipped_objective(0.5, 1.0, 0.2);
    assert!((obj - 0.5).abs() < 1e-6);

    let obj = compute_clipped_objective(1.5, -1.0, 0.2);
    assert!((obj - (-1.5)).abs() < 1e-6);
}

#[test]
fn test_compute_kl_penalty() {
    let kl = compute_kl_penalty(&[(-1.0, -1.0), (-2.0, -2.0)]);
    assert!(kl.abs() < 1e-6);

    let kl = compute_kl_penalty(&[(-1.0, -2.0), (-0.5, -1.5)]);
    assert!(kl > 0.0);
}

#[test]
fn test_grpo_config_defaults() {
    let config = GrpoConfig::default();
    assert_eq!(config.group_size, 16);
    assert_eq!(config.clip_eps, 0.2);
    assert_eq!(config.kl_beta, 0.04);
    assert_eq!(config.value_loss_weight, 0.5);
    assert_eq!(config.max_gen_tokens, 512);
}
```

**Step 3: Run tests to verify they fail**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test grpo_test`
Expected: FAIL

**Step 4: Implement GRPO module**

Create `crates/picochat-train/src/grpo.rs`. This is the largest module — full source provided in the implementation. The key public API:

```rust
pub struct GrpoConfig { /* all fields with defaults */ }
pub fn normalize_advantages(rewards: &[f64]) -> Vec<f64>;
pub fn compute_clipped_objective(ratio: f64, advantage: f64, clip_eps: f64) -> f64;
pub fn compute_kl_penalty(logprob_pairs: &[(f32, f32)]) -> f64;
pub fn grpo(config: &GrpoConfig, device: &Device) -> Result<()>;
```

The full `grpo()` function:
1. Loads policy model from SFT checkpoint
2. Creates frozen reference model (same weights)
3. Creates value head
4. Loads training data (GSM8K, ARC, tool-use) into prompts
5. For each step: sample prompt → generate G=16 completions → compute rewards → normalize advantages → get reference log-probs → compute clipped objective + KL penalty → gradient update on best completion → log metrics → periodic checkpoints

(Full implementation code is provided inline in this task for the implementing agent — see grpo.rs in the design doc for the complete algorithm.)

Add to `crates/picochat-train/src/lib.rs`:

```rust
// picochat-train: training loop
pub mod checkpoint;
pub mod grpo;
pub mod metrics;
pub mod pretrain;
pub mod rewards;
pub mod sft;
pub mod trainer;
pub mod value_head;
```

**Step 5: Run tests to verify they pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test grpo_test`
Expected: PASS (all 5 tests)

**Step 6: Run full workspace tests**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test --workspace --exclude picochat-cli`
Expected: PASS (all tests across all crates)

**Step 7: Commit**

```bash
git add crates/picochat-train/
git commit -m "feat: add GRPO training loop with multi-sample generation and advantage normalization"
```

---

### Task 8: CLI Integration

Wire up the `--grpo` flag to the CLI.

**Files:**
- Modify: `crates/picochat-cli/src/main.rs`
- Modify: `crates/picochat-cli/Cargo.toml` (add picochat-tool dep)

**Context:**
- Follows the same pattern as `--pretrain`, `--sft`, and `--eval-bpb` flags
- Needs: `--grpo`, `--gsm8k-data`, `--arc-data`, `--tool-data`, `--group-size`
- Can't link-test (no C linker in env), so verify with `cargo check -p picochat-cli`

**Step 1: Update Cargo.toml**

Add `picochat-tool` to `crates/picochat-cli/Cargo.toml`:

```toml
picochat-tool = { path = "../picochat-tool" }
```

**Step 2: Add CLI flags**

Add these fields to the `Cli` struct after `max_gen_tokens` (around line 131):

```rust
    /// Run GRPO reasoning training
    #[arg(long)]
    grpo: bool,

    /// Path to GSM8K training data (JSONL)
    #[arg(long)]
    gsm8k_data: Option<String>,

    /// Path to ARC-Challenge training data (JSONL)
    #[arg(long)]
    arc_data: Option<String>,

    /// Path to tool-use scenario data (JSONL)
    #[arg(long)]
    tool_data: Option<String>,

    /// GRPO group size (completions per prompt)
    #[arg(long, default_value_t = 16)]
    group_size: usize,

    /// GRPO clipping epsilon
    #[arg(long, default_value_t = 0.2)]
    clip_eps: f64,

    /// GRPO KL penalty weight
    #[arg(long, default_value_t = 0.04)]
    kl_beta: f64,
```

**Step 3: Add handler to main()**

Add the `grpo` branch to the if-else chain after `eval_bpb`:

```rust
    } else if cli.grpo {
        run_grpo(&cli, &device)?;
```

Add to help text:

```rust
        println!("  --grpo         GRPO reasoning training");
```

Add handler function:

```rust
fn run_grpo(cli: &Cli, device: &Device) -> Result<()> {
    let ckpt_dir = cli.load.as_ref().expect("--load is required for GRPO");
    let tok_path = cli.tokenizer.as_ref().expect("--tokenizer is required for GRPO");
    let save_dir = cli.save.as_ref().expect("--save is required for GRPO");

    let config = picochat_train::grpo::GrpoConfig {
        checkpoint_dir: ckpt_dir.clone(),
        tokenizer_path: tok_path.clone(),
        gsm8k_path: cli.gsm8k_data.clone(),
        arc_path: cli.arc_data.clone(),
        tool_data_path: cli.tool_data.clone(),
        group_size: cli.group_size,
        total_steps: cli.steps,
        max_gen_tokens: cli.max_gen_tokens,
        clip_eps: cli.clip_eps,
        kl_beta: cli.kl_beta,
        value_loss_weight: 0.5,
        learning_rate: cli.max_lr,
        warmup_steps: cli.warmup_steps,
        save_dir: save_dir.clone(),
        save_every: cli.save_every,
        target_len: 256,
    };

    picochat_train::grpo::grpo(&config, device)
}
```

**Step 4: Verify it compiles**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo check -p picochat-cli`
Expected: No errors

**Step 5: Run full workspace tests**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test --workspace --exclude picochat-cli`
Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add crates/picochat-cli/
git commit -m "feat: add --grpo CLI flag for GRPO reasoning training"
```

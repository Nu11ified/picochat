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

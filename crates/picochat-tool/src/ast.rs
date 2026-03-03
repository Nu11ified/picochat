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

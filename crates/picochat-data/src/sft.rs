use anyhow::Result;
use picochat_tokenizer::special::SpecialToken;
use picochat_tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConversation {
    pub messages: Vec<ChatMessage>,
}

/// Token IDs and loss mask for one conversation.
/// `mask[i]` corresponds to `tokens[i+1]` (the target at position i).
pub struct TokenizedConversation {
    pub tokens: Vec<u32>,
    pub mask: Vec<u8>,
}

/// Tokenize a chat conversation into tokens + loss mask.
///
/// Only assistant content tokens and assistant_end contribute to loss.
/// The mask is aligned with the target sequence (shifted by 1):
/// `input = tokens[:-1]`, `target = tokens[1:]`, `mask = is_assistant[1:]`.
pub fn tokenize_conversation(
    conv: &ChatConversation,
    tokenizer: &Tokenizer,
) -> Result<TokenizedConversation> {
    let special = tokenizer.special();
    let bos_id = special.token_id(SpecialToken::Bos);
    let user_start_id = special.token_id(SpecialToken::UserStart);
    let user_end_id = special.token_id(SpecialToken::UserEnd);
    let assistant_start_id = special.token_id(SpecialToken::AssistantStart);
    let assistant_end_id = special.token_id(SpecialToken::AssistantEnd);

    let mut tokens: Vec<u32> = Vec::new();
    let mut is_assistant: Vec<bool> = Vec::new();

    tokens.push(bos_id);
    is_assistant.push(false);

    for msg in &conv.messages {
        match msg.role.as_str() {
            "user" => {
                tokens.push(user_start_id);
                is_assistant.push(false);

                let content_tokens = tokenizer.encode(&msg.content)?;
                for &t in &content_tokens {
                    tokens.push(t);
                    is_assistant.push(false);
                }

                tokens.push(user_end_id);
                is_assistant.push(false);
            }
            "assistant" => {
                // A_START itself is not masked — model predicts it from context
                tokens.push(assistant_start_id);
                is_assistant.push(false);

                let content_tokens = tokenizer.encode(&msg.content)?;
                for &t in &content_tokens {
                    tokens.push(t);
                    is_assistant.push(true);
                }

                // A_END is masked — model should learn to produce stop token
                tokens.push(assistant_end_id);
                is_assistant.push(true);
            }
            role => {
                anyhow::bail!("Unknown role: {role}");
            }
        }
    }

    let mask: Vec<u8> = is_assistant[1..].iter().map(|&b| b as u8).collect();

    Ok(TokenizedConversation { tokens, mask })
}

/// Read JSONL chat data and tokenize all conversations.
pub fn load_sft_data(path: &str, tokenizer: &Tokenizer) -> Result<Vec<TokenizedConversation>> {
    let content = std::fs::read_to_string(path)?;
    let mut results = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let conv: ChatConversation = serde_json::from_str(line)?;
        results.push(tokenize_conversation(&conv, tokenizer)?);
    }
    Ok(results)
}

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// All 16 special tokens used by picochat, in canonical order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpecialToken {
    Bos,
    UserStart,
    UserEnd,
    AssistantStart,
    AssistantEnd,
    PythonStart,
    PythonEnd,
    OutputStart,
    OutputEnd,
    ThinkStart,
    ThinkEnd,
    ToolCallStart,
    ToolCallEnd,
    ToolResultStart,
    ToolResultEnd,
    Pad,
}

impl SpecialToken {
    /// The number of special tokens.
    pub const COUNT: usize = 16;

    /// All special tokens in canonical order.
    pub const ALL: &[SpecialToken] = &[
        SpecialToken::Bos,
        SpecialToken::UserStart,
        SpecialToken::UserEnd,
        SpecialToken::AssistantStart,
        SpecialToken::AssistantEnd,
        SpecialToken::PythonStart,
        SpecialToken::PythonEnd,
        SpecialToken::OutputStart,
        SpecialToken::OutputEnd,
        SpecialToken::ThinkStart,
        SpecialToken::ThinkEnd,
        SpecialToken::ToolCallStart,
        SpecialToken::ToolCallEnd,
        SpecialToken::ToolResultStart,
        SpecialToken::ToolResultEnd,
        SpecialToken::Pad,
    ];

    /// Returns the string representation of this special token (e.g., `<|bos|>`).
    pub fn as_str(&self) -> &'static str {
        match self {
            SpecialToken::Bos => "<|bos|>",
            SpecialToken::UserStart => "<|user_start|>",
            SpecialToken::UserEnd => "<|user_end|>",
            SpecialToken::AssistantStart => "<|assistant_start|>",
            SpecialToken::AssistantEnd => "<|assistant_end|>",
            SpecialToken::PythonStart => "<|python_start|>",
            SpecialToken::PythonEnd => "<|python_end|>",
            SpecialToken::OutputStart => "<|output_start|>",
            SpecialToken::OutputEnd => "<|output_end|>",
            SpecialToken::ThinkStart => "<|think_start|>",
            SpecialToken::ThinkEnd => "<|think_end|>",
            SpecialToken::ToolCallStart => "<|tool_call_start|>",
            SpecialToken::ToolCallEnd => "<|tool_call_end|>",
            SpecialToken::ToolResultStart => "<|tool_result_start|>",
            SpecialToken::ToolResultEnd => "<|tool_result_end|>",
            SpecialToken::Pad => "<|pad|>",
        }
    }

    /// Parse a special token from its string representation.
    /// Returns `None` if the string doesn't match any special token.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "<|bos|>" => Some(SpecialToken::Bos),
            "<|user_start|>" => Some(SpecialToken::UserStart),
            "<|user_end|>" => Some(SpecialToken::UserEnd),
            "<|assistant_start|>" => Some(SpecialToken::AssistantStart),
            "<|assistant_end|>" => Some(SpecialToken::AssistantEnd),
            "<|python_start|>" => Some(SpecialToken::PythonStart),
            "<|python_end|>" => Some(SpecialToken::PythonEnd),
            "<|output_start|>" => Some(SpecialToken::OutputStart),
            "<|output_end|>" => Some(SpecialToken::OutputEnd),
            "<|think_start|>" => Some(SpecialToken::ThinkStart),
            "<|think_end|>" => Some(SpecialToken::ThinkEnd),
            "<|tool_call_start|>" => Some(SpecialToken::ToolCallStart),
            "<|tool_call_end|>" => Some(SpecialToken::ToolCallEnd),
            "<|tool_result_start|>" => Some(SpecialToken::ToolResultStart),
            "<|tool_result_end|>" => Some(SpecialToken::ToolResultEnd),
            "<|pad|>" => Some(SpecialToken::Pad),
            _ => None,
        }
    }
}

/// Maps special tokens to token IDs at the end of the vocabulary.
///
/// Token ID layout:
/// - IDs 0-255: byte tokens
/// - IDs 256 to vocab_size-17: BPE merge tokens
/// - IDs vocab_size-16 to vocab_size-1: special tokens (16 total)
pub struct SpecialTokenRegistry {
    vocab_size: usize,
    /// Maps each special token to its token ID.
    token_to_id: HashMap<SpecialToken, u32>,
    /// Maps each token ID to its special token (reverse lookup).
    id_to_token: HashMap<u32, SpecialToken>,
}

impl SpecialTokenRegistry {
    /// Create a new registry for the given vocab size.
    ///
    /// # Panics
    /// Panics if `vocab_size` is not greater than `SpecialToken::COUNT`.
    pub fn new(vocab_size: usize) -> Self {
        assert!(
            vocab_size > SpecialToken::COUNT,
            "vocab_size ({}) must be greater than SpecialToken::COUNT ({})",
            vocab_size,
            SpecialToken::COUNT
        );

        let first_special_id = (vocab_size - SpecialToken::COUNT) as u32;
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (i, &token) in SpecialToken::ALL.iter().enumerate() {
            let id = first_special_id + i as u32;
            token_to_id.insert(token, id);
            id_to_token.insert(id, token);
        }

        Self {
            vocab_size,
            token_to_id,
            id_to_token,
        }
    }

    /// Returns the token ID for the given special token.
    pub fn token_id(&self, token: SpecialToken) -> u32 {
        self.token_to_id[&token]
    }

    /// Returns the token ID for `<|bos|>`.
    pub fn bos_id(&self) -> u32 {
        self.token_id(SpecialToken::Bos)
    }

    /// Returns the token ID for `<|pad|>`.
    pub fn pad_id(&self) -> u32 {
        self.token_id(SpecialToken::Pad)
    }

    /// Reverse lookup: returns the special token for a given ID, or `None` if
    /// the ID does not correspond to a special token.
    pub fn from_id(&self, id: u32) -> Option<SpecialToken> {
        self.id_to_token.get(&id).copied()
    }

    /// Returns the first token ID in the special token range.
    pub fn first_special_id(&self) -> u32 {
        (self.vocab_size - SpecialToken::COUNT) as u32
    }

    /// Returns the total vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

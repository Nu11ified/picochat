// picochat-tokenizer: tokenization
pub mod special;
pub mod bpe;
pub mod encode;

use anyhow::Result;
use fancy_regex::Regex;

use crate::bpe::{BpeModel, BpeVocab, GPT4_SPLIT_PATTERN, train_bpe, vocab_from_model};
use crate::special::SpecialTokenRegistry;

/// High-level tokenizer wrapping BPE vocabulary, special token registry,
/// and the compiled regex pattern.
pub struct Tokenizer {
    vocab: BpeVocab,
    special: SpecialTokenRegistry,
    pattern: Regex,
}

impl Tokenizer {
    /// Train a new tokenizer on the given text.
    ///
    /// `vocab_size` must be at least 272 (256 byte tokens + 16 special tokens).
    pub fn train(text: &str, vocab_size: usize) -> Result<Self> {
        let vocab = train_bpe(text, vocab_size)?;
        let special = SpecialTokenRegistry::new(vocab_size);
        let pattern = Regex::new(GPT4_SPLIT_PATTERN)?;
        Ok(Self {
            vocab,
            special,
            pattern,
        })
    }

    /// Encode text into a sequence of token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        encode::encode(text, &self.vocab, &self.special, &self.pattern)
    }

    /// Decode a sequence of token IDs back into a string.
    pub fn decode(&self, tokens: &[u32]) -> String {
        encode::decode(tokens, &self.vocab, &self.special)
    }

    /// Returns the total vocabulary size (bytes + merges + special tokens).
    pub fn vocab_size(&self) -> usize {
        self.vocab.vocab_size
    }

    /// Returns a reference to the special token registry.
    pub fn special(&self) -> &SpecialTokenRegistry {
        &self.special
    }

    /// Returns the token ID for `<|bos|>`.
    pub fn bos_id(&self) -> u32 {
        self.special.bos_id()
    }

    /// Returns the number of BPE merges learned during training.
    pub fn num_merges(&self) -> usize {
        self.vocab.merges.len()
    }

    /// Serialize the BPE model to a JSON file at the given path.
    pub fn save(&self, path: &str) -> Result<()> {
        let model = BpeModel {
            vocab_size: self.vocab.vocab_size,
            merges: self.vocab.merges.clone(),
            pattern: GPT4_SPLIT_PATTERN.to_string(),
        };
        let json = serde_json::to_string_pretty(&model)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a tokenizer from a JSON file containing a serialized BPE model.
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let model: BpeModel = serde_json::from_str(&json)?;
        let vocab = vocab_from_model(&model)?;
        let special = SpecialTokenRegistry::new(model.vocab_size);
        let pattern = Regex::new(&model.pattern)?;
        Ok(Self {
            vocab,
            special,
            pattern,
        })
    }
}

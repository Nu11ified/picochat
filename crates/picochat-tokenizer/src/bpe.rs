use anyhow::{bail, Result};
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GPT-4 split pattern with possessive quantifiers converted to greedy for fancy-regex.
pub const GPT4_SPLIT_PATTERN: &str =
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Number of byte-level tokens (0-255).
const NUM_BYTES: usize = 256;

/// Serializable BPE model (merges + pattern, no derived lookup tables).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BpeModel {
    pub vocab_size: usize,
    pub merges: Vec<(u32, u32)>,
    pub pattern: String,
}

/// Full BPE vocabulary with all lookup tables needed for encoding.
#[derive(Debug, Clone)]
pub struct BpeVocab {
    pub vocab_size: usize,
    pub merges: Vec<(u32, u32)>,
    pub merge_map: HashMap<(u32, u32), u32>,
    pub vocab: HashMap<u32, Vec<u8>>,
}

/// Train a BPE tokenizer on the given text.
///
/// `vocab_size` must be at least 272 (256 byte tokens + 16 special tokens).
/// Zero merges is valid (if text is too short or vocab_size == 272).
///
/// The number of merge iterations is `vocab_size - 256 - 16`. If no pair has frequency >= 2,
/// training stops early.
pub fn train_bpe(text: &str, vocab_size: usize) -> Result<BpeVocab> {
    let num_special = crate::special::SpecialToken::COUNT;
    if vocab_size < NUM_BYTES + num_special {
        bail!(
            "vocab_size ({}) must be at least {} (256 bytes + 16 special tokens)",
            vocab_size,
            NUM_BYTES + num_special
        );
    }
    let num_merges = vocab_size - NUM_BYTES - num_special;

    let re = Regex::new(GPT4_SPLIT_PATTERN)?;

    // Step 1: regex-split training text into chunks
    let mut chunks: Vec<Vec<u32>> = Vec::new();
    for mat in re.find_iter(text) {
        let mat = mat?;
        let chunk_bytes = mat.as_str().as_bytes();
        let ids: Vec<u32> = chunk_bytes.iter().map(|&b| b as u32).collect();
        if !ids.is_empty() {
            chunks.push(ids);
        }
    }

    // Step 2: build initial vocab (byte tokens 0-255)
    let mut vocab: HashMap<u32, Vec<u8>> = HashMap::new();
    for i in 0..NUM_BYTES {
        vocab.insert(i as u32, vec![i as u8]);
    }

    let mut merges: Vec<(u32, u32)> = Vec::new();
    let mut merge_map: HashMap<(u32, u32), u32> = HashMap::new();

    // Step 3: iteratively find and merge the most frequent pair
    for i in 0..num_merges {
        // Count adjacent pair frequencies across all chunks
        let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
        for chunk in &chunks {
            if chunk.len() < 2 {
                continue;
            }
            for window in chunk.windows(2) {
                let pair = (window[0], window[1]);
                *pair_counts.entry(pair).or_insert(0) += 1;
            }
        }

        // Find most frequent pair (deterministic tie-breaking by pair value)
        let best = pair_counts
            .iter()
            .max_by(|&(&pair_a, &count_a), &(&pair_b, &count_b)| {
                count_a.cmp(&count_b).then_with(|| pair_b.cmp(&pair_a))
            });

        match best {
            Some((&pair, &count)) if count >= 2 => {
                let new_id = (NUM_BYTES + i) as u32;

                // Record merge
                merges.push(pair);
                merge_map.insert(pair, new_id);

                // Build vocab entry: concatenation of parent byte sequences
                let mut new_bytes = vocab[&pair.0].clone();
                new_bytes.extend_from_slice(&vocab[&pair.1]);
                vocab.insert(new_id, new_bytes);

                // Replace all occurrences in all chunks
                for chunk in &mut chunks {
                    *chunk = merge_pair(chunk, pair, new_id);
                }
            }
            _ => {
                // No pair with frequency >= 2; stop early
                break;
            }
        }
    }

    Ok(BpeVocab {
        vocab_size,
        merges,
        merge_map,
        vocab,
    })
}

/// Reconstruct a `BpeVocab` from a serialized `BpeModel`.
///
/// Returns an error if the model contains merges that reference unknown token IDs.
pub fn vocab_from_model(model: &BpeModel) -> Result<BpeVocab> {
    let mut vocab: HashMap<u32, Vec<u8>> = HashMap::new();
    for i in 0..NUM_BYTES {
        vocab.insert(i as u32, vec![i as u8]);
    }

    let mut merge_map: HashMap<(u32, u32), u32> = HashMap::new();
    for (i, &pair) in model.merges.iter().enumerate() {
        let new_id = (NUM_BYTES + i) as u32;
        merge_map.insert(pair, new_id);

        let left = vocab.get(&pair.0).ok_or_else(|| {
            anyhow::anyhow!("merge ({}, {}) references unknown token {}", pair.0, pair.1, pair.0)
        })?.clone();
        let right = vocab.get(&pair.1).ok_or_else(|| {
            anyhow::anyhow!("merge ({}, {}) references unknown token {}", pair.0, pair.1, pair.1)
        })?;
        let mut new_bytes = left;
        new_bytes.extend_from_slice(right);
        vocab.insert(new_id, new_bytes);
    }

    Ok(BpeVocab {
        vocab_size: model.vocab_size,
        merges: model.merges.clone(),
        merge_map,
        vocab,
    })
}

/// Replace all occurrences of `pair` in `ids` with `new_id`.
///
/// Scans left-to-right; after a replacement the scan continues from the
/// position after the newly inserted token (so overlapping pairs like
/// (a, a) in [a, a, a] produce [new, a]).
fn merge_pair(ids: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut result = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            result.push(new_id);
            i += 2;
        } else {
            result.push(ids[i]);
            i += 1;
        }
    }
    result
}

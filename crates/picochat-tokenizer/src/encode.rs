use anyhow::Result;
use fancy_regex::Regex;

use crate::bpe::BpeVocab;
use crate::special::{SpecialToken, SpecialTokenRegistry};

/// Encode a text string into a sequence of token IDs.
///
/// Special tokens are recognized as literal substrings (e.g. `<|bos|>`) and mapped
/// to their registry IDs. All other text is regex-split and BPE-encoded.
pub fn encode(
    text: &str,
    vocab: &BpeVocab,
    special: &SpecialTokenRegistry,
    pattern: &Regex,
) -> Result<Vec<u32>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }

    let mut tokens: Vec<u32> = Vec::new();
    let mut pos = 0;

    while pos < text.len() {
        // Check if any special token matches at the current position
        let remaining = &text[pos..];
        let mut found_special = false;

        for &st in SpecialToken::ALL {
            let st_str = st.as_str();
            if remaining.starts_with(st_str) {
                tokens.push(special.token_id(st));
                pos += st_str.len();
                found_special = true;
                break;
            }
        }

        if found_special {
            continue;
        }

        // Find distance to the next special token (or end of string)
        let segment_end = find_next_special(remaining)
            .map(|offset| pos + offset)
            .unwrap_or(text.len());

        let segment = &text[pos..segment_end];

        // Regex-split the segment and BPE-encode each chunk
        for mat in pattern.find_iter(segment) {
            let mat = mat?;
            let chunk = mat.as_str();
            if !chunk.is_empty() {
                let chunk_tokens = encode_chunk(chunk.as_bytes(), &vocab.merge_map);
                tokens.extend(chunk_tokens);
            }
        }

        pos = segment_end;
    }

    Ok(tokens)
}

/// Find the byte offset of the next special token in `text`, or `None` if none found.
fn find_next_special(text: &str) -> Option<usize> {
    let mut earliest: Option<usize> = None;

    for &st in SpecialToken::ALL {
        let st_str = st.as_str();
        if let Some(offset) = text.find(st_str) {
            if offset == 0 {
                // The special token is at the current position; skip it since
                // the caller already checked position 0.
                continue;
            }
            earliest = Some(match earliest {
                Some(prev) => prev.min(offset),
                None => offset,
            });
        }
    }

    earliest
}

/// BPE-encode a single regex chunk (raw bytes) using the merge map.
///
/// 1. Convert each byte to its byte-level token ID (0-255).
/// 2. Repeatedly find the pair with the LOWEST new_id in `merge_map` (= earliest
///    merge = highest priority) and replace ALL occurrences of that pair.
/// 3. Stop when no mergeable pairs remain.
fn encode_chunk(
    chunk_bytes: &[u8],
    merge_map: &std::collections::HashMap<(u32, u32), u32>,
) -> Vec<u32> {
    let mut ids: Vec<u32> = chunk_bytes.iter().map(|&b| b as u32).collect();

    if ids.len() < 2 {
        return ids;
    }

    loop {
        // Find the pair with the lowest new_id (earliest merge = highest priority)
        let mut best_pair: Option<(u32, u32)> = None;
        let mut best_new_id: u32 = u32::MAX;

        for window in ids.windows(2) {
            let pair = (window[0], window[1]);
            if let Some(&new_id) = merge_map.get(&pair) {
                if new_id < best_new_id {
                    best_new_id = new_id;
                    best_pair = Some(pair);
                }
            }
        }

        match best_pair {
            Some(pair) => {
                // Replace ALL occurrences of this pair with the new token
                ids = merge_pair_in_place(&ids, pair, best_new_id);
            }
            None => break,
        }
    }

    ids
}

/// Replace all occurrences of `pair` in `ids` with `new_id`.
///
/// Scans left-to-right; after a replacement the scan continues past the newly
/// inserted token (so overlapping pairs like (a,a) in [a,a,a] produce [new,a]).
fn merge_pair_in_place(ids: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
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

/// Decode a sequence of token IDs back into a string.
///
/// Special token IDs are mapped to their string representations (e.g. `<|bos|>`).
/// Regular token IDs are looked up in the vocab for their byte sequences.
/// The final byte sequence is decoded as UTF-8 with lossy replacement.
pub fn decode(tokens: &[u32], vocab: &BpeVocab, special: &SpecialTokenRegistry) -> String {
    let mut bytes: Vec<u8> = Vec::new();

    for &token_id in tokens {
        if let Some(st) = special.from_id(token_id) {
            bytes.extend_from_slice(st.as_str().as_bytes());
        } else if let Some(token_bytes) = vocab.vocab.get(&token_id) {
            bytes.extend_from_slice(token_bytes);
        }
        // Unknown IDs are silently skipped (should not happen with valid data)
    }

    String::from_utf8_lossy(&bytes).into_owned()
}

use anyhow::Result;
use rand::Rng;

/// A flat buffer of token IDs for language model training.
pub struct TokenDataset {
    tokens: Vec<u32>,
}

impl TokenDataset {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self { tokens }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Random-sampling batch iterator over a TokenDataset.
///
/// Each call to `next_batch` returns `(input, target)` where both are
/// `Vec<Vec<u32>>` of shape `(batch_size, seq_len)`.
/// `target[b][t] = input[b][t+1]` (next-token prediction).
pub struct DataLoader {
    dataset: TokenDataset,
    batch_size: usize,
    seq_len: usize,
    rng: rand::rngs::ThreadRng,
}

impl DataLoader {
    pub fn new(dataset: TokenDataset, batch_size: usize, seq_len: usize) -> Self {
        Self {
            dataset,
            batch_size,
            seq_len,
            rng: rand::thread_rng(),
        }
    }

    /// Returns (input, target) where each is `Vec<Vec<u32>>` of shape `(B, T)`.
    /// `target[b][t] = input[b][t+1]` (next-token prediction).
    pub fn next_batch(&mut self) -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        let max_start = self.dataset.len().saturating_sub(self.seq_len + 1);
        let mut inputs = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            let start = if max_start > 0 {
                self.rng.gen_range(0..max_start)
            } else {
                0
            };
            let chunk = &self.dataset.tokens[start..start + self.seq_len + 1];
            inputs.push(chunk[..self.seq_len].to_vec());
            targets.push(chunk[1..self.seq_len + 1].to_vec());
        }

        Ok((inputs, targets))
    }
}

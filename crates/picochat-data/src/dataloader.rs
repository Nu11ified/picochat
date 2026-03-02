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

// ---------------------------------------------------------------------------
// BOS-Aligned Packing DataLoader
// ---------------------------------------------------------------------------

/// A partially filled bin accumulating tokens toward a fixed-length sequence.
struct PackingBin {
    tokens: Vec<u32>,
    capacity: usize, // = seq_len + 1
}

impl PackingBin {
    fn new(capacity: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn remaining(&self) -> usize {
        self.capacity - self.tokens.len()
    }

    fn is_full(&self) -> bool {
        self.tokens.len() >= self.capacity
    }

    /// Append as many tokens as will fit. Returns the number actually appended.
    fn append(&mut self, tokens: &[u32]) -> usize {
        let take = tokens.len().min(self.remaining());
        self.tokens.extend_from_slice(&tokens[..take]);
        take
    }
}

/// Packing data loader that concatenates tokenized documents into fixed-length
/// sequences, prepending a BOS token at each document boundary.
///
/// Documents are packed using a best-fit-decreasing bin-packing strategy:
/// each new document (with BOS prepended) is placed into the open bin with
/// the smallest remaining capacity that can still accept at least one token.
/// Long documents that exceed a single bin's capacity are split across
/// multiple bins.
///
/// Completed sequences have length `seq_len + 1` so that `next_batch` can
/// produce input/target pairs where `target[t] = input[t+1]`.
pub struct PackingDataLoader {
    seq_len: usize,
    batch_size: usize,
    bos_id: u32,
    open_bins: Vec<PackingBin>,
    ready: Vec<Vec<u32>>,
}

impl PackingDataLoader {
    pub fn new(batch_size: usize, seq_len: usize, bos_id: u32) -> Self {
        Self {
            seq_len,
            batch_size,
            bos_id,
            open_bins: Vec::new(),
            ready: Vec::new(),
        }
    }

    /// Add a tokenized document. BOS is automatically prepended.
    /// The resulting tokens are packed into bins using best-fit.
    pub fn add_document(&mut self, doc_tokens: &[u32]) {
        // Build the full token sequence: BOS + doc_tokens
        let mut tokens = Vec::with_capacity(1 + doc_tokens.len());
        tokens.push(self.bos_id);
        tokens.extend_from_slice(doc_tokens);

        let mut offset = 0;
        while offset < tokens.len() {
            // Best-fit: find the open bin with smallest remaining capacity
            // that can still accept at least 1 token.
            let mut best_idx: Option<usize> = None;
            let mut best_remaining = usize::MAX;

            for (i, bin) in self.open_bins.iter().enumerate() {
                let rem = bin.remaining();
                if rem > 0 && rem < best_remaining {
                    best_remaining = rem;
                    best_idx = Some(i);
                }
            }

            match best_idx {
                Some(idx) => {
                    let appended = self.open_bins[idx].append(&tokens[offset..]);
                    offset += appended;
                    if self.open_bins[idx].is_full() {
                        let bin = self.open_bins.remove(idx);
                        self.ready.push(bin.tokens);
                    }
                }
                None => {
                    // No suitable bin found — create a new one
                    let capacity = self.seq_len + 1;
                    let mut bin = PackingBin::new(capacity);
                    let appended = bin.append(&tokens[offset..]);
                    offset += appended;
                    if bin.is_full() {
                        self.ready.push(bin.tokens);
                    } else {
                        self.open_bins.push(bin);
                    }
                }
            }
        }
    }

    /// Number of complete sequences available for batching.
    pub fn ready_count(&self) -> usize {
        self.ready.len()
    }

    /// Return the next batch of `(inputs, targets)`, or `None` if fewer than
    /// `batch_size` sequences are ready.
    ///
    /// Each input has length `seq_len` and each target has length `seq_len`,
    /// with `target[t] == input[t+1]` (next-token prediction).
    pub fn next_batch(&mut self) -> Option<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
        if self.ready.len() < self.batch_size {
            return None;
        }

        let mut inputs = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            let seq = self.ready.remove(0);
            // seq has length seq_len + 1
            inputs.push(seq[..self.seq_len].to_vec());
            targets.push(seq[1..self.seq_len + 1].to_vec());
        }

        Some((inputs, targets))
    }

    /// Pad all open (partial) bins with `bos_id` to full length and move them
    /// to the ready queue.
    pub fn flush(&mut self) {
        let bins: Vec<PackingBin> = self.open_bins.drain(..).collect();
        for mut bin in bins {
            while bin.tokens.len() < bin.capacity {
                bin.tokens.push(self.bos_id);
            }
            self.ready.push(bin.tokens);
        }
    }
}

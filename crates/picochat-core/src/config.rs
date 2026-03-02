use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPTConfig {
    pub sequence_len: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    pub window_pattern: String,
}

impl GPTConfig {
    pub fn from_depth(depth: usize) -> Self {
        // n_embd = round_up(64 * depth, 128)
        let n_embd = ((64 * depth + 127) / 128) * 128;
        let head_dim = 64;
        let n_head = n_embd / head_dim;
        let n_kv_head = (n_head / 2).max(1);

        GPTConfig {
            sequence_len: 2048,
            vocab_size: 32768,
            n_layer: depth,
            n_head,
            n_kv_head,
            n_embd,
            window_pattern: "SSSL".to_string(),
        }
    }

    pub fn compute_window_sizes(&self) -> Vec<(usize, usize)> {
        let pattern: Vec<char> = self.window_pattern.chars().collect();
        let long_window = self.sequence_len;
        let short_window = long_window / 2;
        let mut windows: Vec<(usize, usize)> = (0..self.n_layer)
            .map(|i| match pattern[i % pattern.len()] {
                'L' | 'l' => (long_window, 0),
                'S' | 's' => (short_window, 0),
                c => panic!("Invalid window pattern char: {c}"),
            })
            .collect();
        if let Some(last) = windows.last_mut() {
            *last = (long_window, 0);
        }
        windows
    }

    pub fn padded_vocab_size(&self) -> usize {
        ((self.vocab_size + 63) / 64) * 64
    }

    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    pub fn has_value_embedding(&self, layer_idx: usize) -> bool {
        layer_idx % 2 == (self.n_layer - 1) % 2
    }
}

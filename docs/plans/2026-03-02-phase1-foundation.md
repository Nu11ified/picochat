# Phase 1: Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the Cargo workspace, GPT model architecture, and verify a forward pass on CPU — the foundation everything else builds on.

**Architecture:** Cargo workspace with 8 crates. picochat-core contains the GPT model (config, attention, MLP, normalization, rotary embeddings). Uses candle-core and candle-nn for tensor operations. All code runs on CPU first.

**Tech Stack:** Rust, candle-core, candle-nn, clap, serde, anyhow

---

### Task 1: Cargo Workspace Scaffold

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/picochat-core/Cargo.toml`
- Create: `crates/picochat-core/src/lib.rs`
- Create: `crates/picochat-tokenizer/Cargo.toml`
- Create: `crates/picochat-tokenizer/src/lib.rs`
- Create: `crates/picochat-data/Cargo.toml`
- Create: `crates/picochat-data/src/lib.rs`
- Create: `crates/picochat-optim/Cargo.toml`
- Create: `crates/picochat-optim/src/lib.rs`
- Create: `crates/picochat-train/Cargo.toml`
- Create: `crates/picochat-train/src/lib.rs`
- Create: `crates/picochat-eval/Cargo.toml`
- Create: `crates/picochat-eval/src/lib.rs`
- Create: `crates/picochat-engine/Cargo.toml`
- Create: `crates/picochat-engine/src/lib.rs`
- Create: `crates/picochat-cli/Cargo.toml`
- Create: `crates/picochat-cli/src/main.rs`

**Step 1: Create workspace root Cargo.toml**

```toml
[workspace]
resolver = "2"
members = [
    "crates/picochat-core",
    "crates/picochat-tokenizer",
    "crates/picochat-data",
    "crates/picochat-optim",
    "crates/picochat-train",
    "crates/picochat-eval",
    "crates/picochat-engine",
    "crates/picochat-cli",
]

[workspace.dependencies]
candle-core = "0.8"
candle-nn = "0.8"
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"
rand = "0.8"
```

**Step 2: Create picochat-core Cargo.toml**

```toml
[package]
name = "picochat-core"
version = "0.1.0"
edition = "2021"

[dependencies]
candle-core = { workspace = true }
candle-nn = { workspace = true }
anyhow = { workspace = true }
serde = { workspace = true }
rand = { workspace = true }

[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
```

**Step 3: Create all other crate Cargo.tomls with minimal deps**

Each crate gets a minimal Cargo.toml with `edition = "2021"` and its key dependency (picochat-core for most). The lib.rs files are empty stubs.

picochat-cli gets:
```toml
[package]
name = "picochat-cli"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "picochat"
path = "src/main.rs"

[dependencies]
picochat-core = { path = "../picochat-core" }
clap = { workspace = true }
anyhow = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
```

main.rs stub:
```rust
fn main() {
    println!("picochat v0.1.0");
}
```

**Step 4: Verify workspace compiles**

Run: `cargo build`
Expected: Compiles with 0 errors, prints "picochat v0.1.0" when run.

**Step 5: Commit**

```bash
git init
echo "target/" > .gitignore
git add -A
git commit -m "feat: scaffold Cargo workspace with 8 crates"
```

---

### Task 2: GPTConfig with Depth Scaling

**Files:**
- Create: `crates/picochat-core/src/config.rs`
- Modify: `crates/picochat-core/src/lib.rs`
- Create: `crates/picochat-core/tests/config_test.rs`

**Step 1: Write failing test for config scaling**

```rust
// crates/picochat-core/tests/config_test.rs
use picochat_core::config::GPTConfig;

#[test]
fn test_depth_12_config() {
    let config = GPTConfig::from_depth(12);
    assert_eq!(config.n_layer, 12);
    assert_eq!(config.n_embd, 768);
    assert_eq!(config.n_head, 12);
    assert_eq!(config.n_kv_head, 6);
    assert_eq!(config.sequence_len, 2048);
    assert_eq!(config.vocab_size, 32768);
    assert_eq!(config.window_pattern, "SSSL");
}

#[test]
fn test_depth_4_small_config() {
    let config = GPTConfig::from_depth(4);
    assert_eq!(config.n_layer, 4);
    // Smaller depth = smaller model
    assert!(config.n_embd < 768);
    assert!(config.n_head > 0);
    assert!(config.n_embd % config.n_head == 0);
    assert!(config.n_kv_head <= config.n_head);
    assert!(config.n_head % config.n_kv_head == 0);
}

#[test]
fn test_depth_26_gpt2_config() {
    let config = GPTConfig::from_depth(26);
    assert_eq!(config.n_layer, 26);
    // GPT-2 scale: ~1.6B params needs large embedding
    assert!(config.n_embd >= 1536);
    assert!(config.n_embd % config.n_head == 0);
}

#[test]
fn test_head_dim_consistent() {
    for depth in [4, 8, 12, 16, 20, 24, 26] {
        let config = GPTConfig::from_depth(depth);
        let head_dim = config.n_embd / config.n_head;
        // Head dim should be consistent (64 or 128)
        assert!(head_dim == 64 || head_dim == 128,
            "depth={depth}: head_dim={head_dim}");
    }
}

#[test]
fn test_window_sizes() {
    let config = GPTConfig::from_depth(12);
    let windows = config.compute_window_sizes();
    assert_eq!(windows.len(), 12);
    // Last layer always full context
    assert_eq!(windows[11], (config.sequence_len, 0));
    // Pattern "SSSL" means layers 0,1,2=short, 3=long, 4,5,6=short, 7=long, ...
    let short_window = config.sequence_len / 2;
    assert_eq!(windows[0].0, short_window);
    assert_eq!(windows[1].0, short_window);
    assert_eq!(windows[2].0, short_window);
    assert_eq!(windows[3].0, config.sequence_len);
}

#[test]
fn test_padded_vocab_size() {
    let config = GPTConfig::from_depth(12);
    let padded = config.padded_vocab_size();
    assert!(padded >= config.vocab_size);
    assert_eq!(padded % 64, 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p picochat-core`
Expected: FAIL — `config` module not found.

**Step 3: Implement GPTConfig**

```rust
// crates/picochat-core/src/config.rs
use serde::{Deserialize, Serialize};

/// GPT model configuration.
/// All hyperparameters are derived from a single `depth` (n_layer) parameter.
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
    /// Create config from depth (number of layers).
    /// Automatically computes all other hyperparameters for compute-optimal models.
    ///
    /// Scaling rules (matching nanochat):
    /// - n_embd scales with depth: 64 * depth (clamped to multiples of 128)
    /// - n_head = n_embd / 64 (head_dim = 64)
    /// - n_kv_head = n_head / 2 (GQA with 2:1 ratio)
    /// - sequence_len = 2048
    /// - vocab_size = 32768
    pub fn from_depth(depth: usize) -> Self {
        // Embedding dimension scales linearly with depth, rounded to multiple of 128
        let n_embd = ((64 * depth + 127) / 128) * 128;
        let head_dim = 64;
        let n_head = n_embd / head_dim;
        // GQA: half as many KV heads as query heads, minimum 1
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

    /// Compute per-layer window sizes for sliding window attention.
    /// Returns Vec of (left, right) tuples.
    /// - left: how many tokens before current position to attend to
    /// - right: 0 for causal
    /// Pattern is tiled across layers. Final layer always gets full context.
    pub fn compute_window_sizes(&self) -> Vec<(usize, usize)> {
        let pattern: Vec<char> = self.window_pattern.chars().collect();
        let long_window = self.sequence_len;
        let short_window = long_window / 2;

        let mut windows: Vec<(usize, usize)> = (0..self.n_layer)
            .map(|i| {
                let c = pattern[i % pattern.len()];
                match c {
                    'L' | 'l' => (long_window, 0),
                    'S' | 's' => (short_window, 0),
                    _ => panic!("Invalid window pattern char: {c}"),
                }
            })
            .collect();

        // Final layer always full context
        if let Some(last) = windows.last_mut() {
            *last = (long_window, 0);
        }

        windows
    }

    /// Padded vocab size (multiple of 64 for tensor core alignment).
    pub fn padded_vocab_size(&self) -> usize {
        ((self.vocab_size + 63) / 64) * 64
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }

    /// Returns true if layer should have value embedding (alternating, last always included).
    pub fn has_value_embedding(&self, layer_idx: usize) -> bool {
        layer_idx % 2 == (self.n_layer - 1) % 2
    }
}
```

Update lib.rs:
```rust
// crates/picochat-core/src/lib.rs
pub mod config;
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p picochat-core`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add crates/picochat-core/src/config.rs crates/picochat-core/src/lib.rs crates/picochat-core/tests/
git commit -m "feat: GPTConfig with depth-based hyperparameter scaling"
```

---

### Task 3: RMSNorm (No Learnable Parameters)

**Files:**
- Create: `crates/picochat-core/src/norm.rs`
- Modify: `crates/picochat-core/src/lib.rs`
- Create: `crates/picochat-core/tests/norm_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-core/tests/norm_test.rs
use candle_core::{Device, DType, Tensor};
use picochat_core::norm::rms_norm;

#[test]
fn test_rms_norm_shape_preserved() {
    let device = Device::Cpu;
    let x = Tensor::randn(0f32, 1.0, (2, 4, 8), &device).unwrap();
    let result = rms_norm(&x).unwrap();
    assert_eq!(result.shape(), x.shape());
}

#[test]
fn test_rms_norm_unit_rms() {
    let device = Device::Cpu;
    let x = Tensor::randn(0f32, 1.0, (1, 1, 64), &device).unwrap();
    let result = rms_norm(&x).unwrap();
    // After RMS norm, the RMS of the last dimension should be ~1.0
    let sq = result.sqr().unwrap();
    let mean_sq = sq.mean_keepdim(2).unwrap();
    let rms = mean_sq.sqrt().unwrap();
    let rms_val: f32 = rms.flatten_all().unwrap().to_vec1().unwrap()[0];
    assert!((rms_val - 1.0).abs() < 0.01, "RMS should be ~1.0, got {rms_val}");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p picochat-core`
Expected: FAIL — `norm` module not found.

**Step 3: Implement rms_norm**

```rust
// crates/picochat-core/src/norm.rs
use candle_core::{Result, Tensor};

/// Purely functional RMSNorm with no learnable parameters.
/// Normalizes the last dimension of x to have unit RMS.
/// Matches nanochat: F.rms_norm(x, (x.size(-1),))
pub fn rms_norm(x: &Tensor) -> Result<Tensor> {
    let dim = x.dims().len() - 1;
    let eps = 1e-6;
    // mean(x^2) over last dim
    let mean_sq = x.sqr()?.mean_keepdim(dim)?;
    // 1 / sqrt(mean_sq + eps)
    let rsqrt = (mean_sq + eps)?.sqrt()?.recip()?;
    x.broadcast_mul(&rsqrt)
}
```

Update lib.rs to add `pub mod norm;`

**Step 4: Run tests**

Run: `cargo test -p picochat-core`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/picochat-core/src/norm.rs crates/picochat-core/src/lib.rs crates/picochat-core/tests/norm_test.rs
git commit -m "feat: RMSNorm with no learnable parameters"
```

---

### Task 4: Rotary Positional Embeddings (RoPE)

**Files:**
- Create: `crates/picochat-core/src/rotary.rs`
- Modify: `crates/picochat-core/src/lib.rs`
- Create: `crates/picochat-core/tests/rotary_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-core/tests/rotary_test.rs
use candle_core::{Device, DType, Tensor};
use picochat_core::rotary::RotaryEmbedding;

#[test]
fn test_rotary_precompute_shapes() {
    let device = Device::Cpu;
    let rope = RotaryEmbedding::new(64, 2048, 10000.0, &device).unwrap();
    // cos and sin should be (1, seq_len, 1, head_dim/2)
    assert_eq!(rope.cos().dims(), &[1, 2048, 1, 32]);
    assert_eq!(rope.sin().dims(), &[1, 2048, 1, 32]);
}

#[test]
fn test_apply_rotary_emb_shape() {
    let device = Device::Cpu;
    let rope = RotaryEmbedding::new(64, 2048, 10000.0, &device).unwrap();
    // x is (B, T, H, D)
    let x = Tensor::randn(0f32, 1.0, (2, 16, 8, 64), &device).unwrap();
    let result = rope.apply(&x, 0).unwrap();
    assert_eq!(result.shape(), x.shape());
}

#[test]
fn test_rotary_offset_for_kv_cache() {
    let device = Device::Cpu;
    let rope = RotaryEmbedding::new(64, 2048, 10000.0, &device).unwrap();
    // Simulate KV cache: we're at position 100, generating 1 token
    let x = Tensor::randn(0f32, 1.0, (1, 1, 8, 64), &device).unwrap();
    let result = rope.apply(&x, 100).unwrap();
    assert_eq!(result.dims(), &[1, 1, 8, 64]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p picochat-core`
Expected: FAIL — `rotary` module not found.

**Step 3: Implement RotaryEmbedding**

```rust
// crates/picochat-core/src/rotary.rs
use candle_core::{Device, DType, Result, Tensor};

/// Precomputed rotary positional embeddings.
pub struct RotaryEmbedding {
    cos: Tensor, // (1, max_seq_len, 1, head_dim/2)
    sin: Tensor, // (1, max_seq_len, 1, head_dim/2)
}

impl RotaryEmbedding {
    /// Precompute rotary embeddings for the given head_dim and max sequence length.
    pub fn new(head_dim: usize, max_seq_len: usize, base: f64, device: &Device) -> Result<Self> {
        let half_dim = head_dim / 2;
        // inv_freq = 1.0 / (base ^ (i / head_dim)) for i in 0, 2, 4, ...
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        // t = [0, 1, 2, ..., max_seq_len-1]
        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Tensor::new(t.as_slice(), device)?;

        // freqs = outer(t, inv_freq) -> (seq_len, half_dim)
        let freqs = t.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        // Reshape to (1, seq_len, 1, half_dim) for broadcasting with (B, T, H, D)
        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        Ok(Self { cos, sin })
    }

    pub fn cos(&self) -> &Tensor {
        &self.cos
    }

    pub fn sin(&self) -> &Tensor {
        &self.sin
    }

    /// Apply rotary embeddings to x of shape (B, T, H, D).
    /// offset: position offset for KV cache inference.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, t, _h, _d) = x.dims4()?;
        let cos = self.cos.narrow(1, offset, t)?;
        let sin = self.sin.narrow(1, offset, t)?;
        apply_rotary_emb(x, &cos, &sin)
    }
}

/// Apply rotary embeddings to x.
/// x: (B, T, H, D), cos/sin: (1, T, 1, D/2)
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _t, _h, d) = x.dims4()?;
    let half_d = d / 2;
    let x1 = x.narrow(3, 0, half_d)?;
    let x2 = x.narrow(3, half_d, half_d)?;
    // y1 = x1 * cos + x2 * sin
    let y1 = x1.broadcast_mul(cos)?.broadcast_add(&x2.broadcast_mul(sin)?)?;
    // y2 = x1 * (-sin) + x2 * cos
    let neg_sin = sin.neg()?;
    let y2 = x1.broadcast_mul(&neg_sin)?.broadcast_add(&x2.broadcast_mul(cos)?)?;
    Tensor::cat(&[&y1, &y2], 3)
}
```

Update lib.rs to add `pub mod rotary;`

**Step 4: Run tests**

Run: `cargo test -p picochat-core`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/picochat-core/src/rotary.rs crates/picochat-core/tests/rotary_test.rs crates/picochat-core/src/lib.rs
git commit -m "feat: rotary positional embeddings (RoPE)"
```

---

### Task 5: MLP with ReLU^2

**Files:**
- Create: `crates/picochat-core/src/mlp.rs`
- Modify: `crates/picochat-core/src/lib.rs`
- Create: `crates/picochat-core/tests/mlp_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-core/tests/mlp_test.rs
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::mlp::MLP;

#[test]
fn test_mlp_output_shape() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let n_embd = 64;
    let mlp = MLP::new(n_embd, vb).unwrap();
    let x = Tensor::randn(0f32, 1.0, (2, 4, n_embd), &device).unwrap();
    let result = mlp.forward(&x).unwrap();
    assert_eq!(result.dims(), &[2, 4, n_embd]);
}

#[test]
fn test_mlp_relu_squared_activation() {
    // ReLU^2 means negative values become 0, positive values are squared
    let device = Device::Cpu;
    let x = Tensor::new(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &device).unwrap();
    let activated = x.relu().unwrap().sqr().unwrap();
    let vals: Vec<f32> = activated.to_vec1().unwrap();
    assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 4.0]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p picochat-core`
Expected: FAIL — `mlp` module not found.

**Step 3: Implement MLP**

```rust
// crates/picochat-core/src/mlp.rs
use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};

/// MLP with ReLU^2 activation and 4x expansion.
/// Matches nanochat: c_fc (n_embd -> 4*n_embd), relu^2, c_proj (4*n_embd -> n_embd).
pub struct MLP {
    c_fc: Linear,
    c_proj: Linear,
}

impl MLP {
    pub fn new(n_embd: usize, vb: VarBuilder) -> Result<Self> {
        let c_fc = linear_no_bias(n_embd, 4 * n_embd, vb.pp("c_fc"))?;
        let c_proj = linear_no_bias(4 * n_embd, n_embd, vb.pp("c_proj"))?;
        Ok(Self { c_fc, c_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = x.relu()?.sqr()?; // ReLU^2
        self.c_proj.forward(&x)
    }
}
```

Update lib.rs to add `pub mod mlp;`

**Step 4: Run tests**

Run: `cargo test -p picochat-core`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/picochat-core/src/mlp.rs crates/picochat-core/tests/mlp_test.rs crates/picochat-core/src/lib.rs
git commit -m "feat: MLP with ReLU-squared activation"
```

---

### Task 6: Causal Self-Attention with GQA

**Files:**
- Create: `crates/picochat-core/src/attention.rs`
- Modify: `crates/picochat-core/src/lib.rs`
- Create: `crates/picochat-core/tests/attention_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-core/tests/attention_test.rs
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::attention::CausalSelfAttention;
use picochat_core::config::GPTConfig;
use picochat_core::rotary::RotaryEmbedding;

#[test]
fn test_attention_output_shape() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let rope = RotaryEmbedding::new(config.head_dim(), config.sequence_len * 10, 10000.0, &device).unwrap();

    let attn = CausalSelfAttention::new(&config, 0, vb).unwrap();
    let x = Tensor::randn(0f32, 1.0, (1, 8, config.n_embd), &device).unwrap();
    let window_size = (config.sequence_len, 0);

    let result = attn.forward(&x, None, &rope, 0, window_size, None).unwrap();
    assert_eq!(result.dims(), &[1, 8, config.n_embd]);
}

#[test]
fn test_attention_gqa_heads() {
    let config = GPTConfig::from_depth(12);
    // GQA: n_kv_head < n_head
    assert!(config.n_kv_head < config.n_head);
    assert_eq!(config.n_head % config.n_kv_head, 0);
}

#[test]
fn test_attention_causal_masking() {
    // Ensure output at position t doesn't depend on future positions
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let rope = RotaryEmbedding::new(config.head_dim(), config.sequence_len * 10, 10000.0, &device).unwrap();

    let attn = CausalSelfAttention::new(&config, 0, vb).unwrap();

    // Run with 4 tokens
    let x = Tensor::randn(0f32, 1.0, (1, 4, config.n_embd), &device).unwrap();
    let full_out = attn.forward(&x, None, &rope, 0, (config.sequence_len, 0), None).unwrap();

    // Run with only first 2 tokens
    let x_short = x.narrow(1, 0, 2).unwrap();
    let short_out = attn.forward(&x_short, None, &rope, 0, (config.sequence_len, 0), None).unwrap();

    // Output at positions 0,1 should be identical regardless of future tokens
    let full_first2 = full_out.narrow(1, 0, 2).unwrap().to_vec3::<f32>().unwrap();
    let short_first2 = short_out.to_vec3::<f32>().unwrap();

    for b in 0..1 {
        for t in 0..2 {
            for d in 0..config.n_embd {
                let diff = (full_first2[b][t][d] - short_first2[b][t][d]).abs();
                assert!(diff < 1e-4, "Causal violation at b={b} t={t} d={d}: diff={diff}");
            }
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p picochat-core`
Expected: FAIL — `attention` module not found.

**Step 3: Implement CausalSelfAttention**

```rust
// crates/picochat-core/src/attention.rs
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use crate::config::GPTConfig;
use crate::norm::rms_norm;
use crate::rotary::RotaryEmbedding;

/// Causal self-attention with Grouped-Query Attention (GQA) support.
pub struct CausalSelfAttention {
    c_q: Linear,
    c_k: Linear,
    c_v: Linear,
    c_proj: Linear,
    ve_gate: Option<Linear>,
    n_head: usize,
    n_kv_head: usize,
    n_embd: usize,
    head_dim: usize,
    layer_idx: usize,
    n_layers: usize,
}

impl CausalSelfAttention {
    pub fn new(config: &GPTConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let c_q = linear_no_bias(config.n_embd, config.n_head * head_dim, vb.pp("c_q"))?;
        let c_k = linear_no_bias(config.n_embd, config.n_kv_head * head_dim, vb.pp("c_k"))?;
        let c_v = linear_no_bias(config.n_embd, config.n_kv_head * head_dim, vb.pp("c_v"))?;
        let c_proj = linear_no_bias(config.n_embd, config.n_embd, vb.pp("c_proj"))?;

        let ve_gate = if config.has_value_embedding(layer_idx) {
            let ve_gate_channels = 32;
            Some(linear_no_bias(ve_gate_channels, config.n_kv_head, vb.pp("ve_gate"))?)
        } else {
            None
        };

        Ok(Self {
            c_q,
            c_k,
            c_v,
            c_proj,
            ve_gate,
            n_head: config.n_head,
            n_kv_head: config.n_kv_head,
            n_embd: config.n_embd,
            head_dim,
            layer_idx,
            n_layers: config.n_layer,
        })
    }

    /// Forward pass.
    /// x: (B, T, C)
    /// ve: optional value embedding (B, T, n_kv_head * head_dim)
    /// rope: precomputed rotary embeddings
    /// rope_offset: position offset for KV cache
    /// window_size: (left, right) for sliding window
    /// kv_cache: None for training, Some for inference
    pub fn forward(
        &self,
        x: &Tensor,
        ve: Option<&Tensor>,
        rope: &RotaryEmbedding,
        rope_offset: usize,
        window_size: (usize, usize),
        _kv_cache: Option<()>, // placeholder for now
    ) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;

        // Project to Q, K, V
        let q = self.c_q.forward(x)?;
        let k = self.c_k.forward(x)?;
        let mut v = self.c_v.forward(x)?;

        // Reshape to (B, T, H, D)
        let q = q.reshape((b, t, self.n_head, self.head_dim))?;
        let k = k.reshape((b, t, self.n_kv_head, self.head_dim))?;
        v = v.reshape((b, t, self.n_kv_head, self.head_dim))?;

        // Value residual: mix in value embedding with input-dependent gate
        if let (Some(ve_tensor), Some(ve_gate)) = (ve, &self.ve_gate) {
            let ve_reshaped = ve_tensor.reshape((b, t, self.n_kv_head, self.head_dim))?;
            // gate = 2 * sigmoid(ve_gate(x[..., :32]))
            let x_prefix = x.narrow(2, 0, 32)?;
            let gate = ve_gate.forward(&x_prefix)?; // (B, T, n_kv_head)
            let gate = (candle_nn::ops::sigmoid(&gate)? * 2.0)?;
            let gate = gate.unsqueeze(3)?; // (B, T, n_kv_head, 1)
            v = (v + gate.broadcast_mul(&ve_reshaped)?)?;
        }

        // Apply rotary embeddings
        let q = rope.apply(&q, rope_offset)?;
        let k = rope.apply(&k, rope_offset)?;

        // QK norm
        let q = rms_norm(&q)?;
        let k = rms_norm(&k)?;

        // SDPA attention fallback (no flash attention yet)
        // Transpose to (B, H, T, D) for matmul
        let q = q.transpose(1, 2)?; // (B, n_head, T, D)
        let k = k.transpose(1, 2)?; // (B, n_kv_head, T, D)
        let v = v.transpose(1, 2)?; // (B, n_kv_head, T, D)

        // GQA: repeat KV heads to match Q heads
        let repeat_factor = self.n_head / self.n_kv_head;
        let k = if repeat_factor > 1 {
            let k = k.unsqueeze(2)?; // (B, n_kv_head, 1, T, D)
            let k = k.expand((b, self.n_kv_head, repeat_factor, t, self.head_dim))?;
            k.reshape((b, self.n_head, t, self.head_dim))?
        } else {
            k
        };
        let v = if repeat_factor > 1 {
            let v = v.unsqueeze(2)?;
            let v = v.expand((b, self.n_kv_head, repeat_factor, t, self.head_dim))?;
            v.reshape((b, self.n_head, t, self.head_dim))?
        } else {
            v
        };

        // Scaled dot-product attention with causal mask
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?.affine(1.0 / scale, 0.0)?;

        // Causal mask + sliding window
        let attn_weights = apply_causal_mask(&attn_weights, window_size)?;

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let y = attn_weights.matmul(&v)?; // (B, n_head, T, D)

        // Transpose back and project
        let y = y.transpose(1, 2)?.contiguous()?; // (B, T, n_head, D)
        let y = y.reshape((b, t, self.n_embd))?;
        self.c_proj.forward(&y)
    }
}

/// Apply causal mask and optional sliding window to attention weights.
fn apply_causal_mask(attn_weights: &Tensor, window_size: (usize, usize)) -> Result<Tensor> {
    let (_, _, t_q, t_k) = attn_weights.dims4()?;
    let device = attn_weights.device();
    let neg_inf = f32::NEG_INFINITY;

    // Build causal mask: position i can attend to positions 0..=i
    let mut mask_data = vec![0.0f32; t_q * t_k];
    for i in 0..t_q {
        for j in 0..t_k {
            if j > i {
                // Future position: mask out
                mask_data[i * t_k + j] = neg_inf;
            } else if window_size.0 < t_k && (i - j) > window_size.0 {
                // Outside sliding window: mask out
                mask_data[i * t_k + j] = neg_inf;
            }
        }
    }

    let mask = Tensor::new(mask_data.as_slice(), device)?
        .reshape((1, 1, t_q, t_k))?;
    attn_weights.broadcast_add(&mask)
}
```

Update lib.rs to add `pub mod attention;`

**Step 4: Run tests**

Run: `cargo test -p picochat-core`
Expected: PASS (all tests including causal masking).

**Step 5: Commit**

```bash
git add crates/picochat-core/src/attention.rs crates/picochat-core/tests/attention_test.rs crates/picochat-core/src/lib.rs
git commit -m "feat: causal self-attention with GQA and sliding window"
```

---

### Task 7: GPT Model (Full Forward Pass)

**Files:**
- Create: `crates/picochat-core/src/model.rs`
- Modify: `crates/picochat-core/src/lib.rs`
- Create: `crates/picochat-core/tests/model_test.rs`

**Step 1: Write failing test**

```rust
// crates/picochat-core/tests/model_test.rs
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;

#[test]
fn test_gpt_forward_logits_shape() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = GPT::new(&config, vb).unwrap();
    let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();

    let logits = model.forward(&input, None).unwrap();
    // logits: (B, T, vocab_size) — NOT padded vocab size
    assert_eq!(logits.dims(), &[1, 4, config.vocab_size]);
}

#[test]
fn test_gpt_forward_with_targets() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = GPT::new(&config, vb).unwrap();
    let input = Tensor::new(&[[1u32, 2, 3, 4]], &device).unwrap();
    let targets = Tensor::new(&[[2u32, 3, 4, 5]], &device).unwrap();

    let loss = model.forward(&input, Some(&targets)).unwrap();
    // Loss should be scalar
    assert_eq!(loss.dims(), &[]);
    let loss_val: f32 = loss.to_scalar().unwrap();
    // Cross-entropy loss on random weights should be ~log(vocab_size) ≈ 10.4
    assert!(loss_val > 5.0 && loss_val < 15.0, "loss={loss_val} out of range");
}

#[test]
fn test_gpt_depth4_small() {
    let device = Device::Cpu;
    let config = GPTConfig::from_depth(4);
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = GPT::new(&config, vb).unwrap();
    let num_params = model.num_parameters();
    // depth 4 should be a small model (< 50M params)
    assert!(num_params < 50_000_000, "Too many params: {num_params}");
    assert!(num_params > 1_000_000, "Too few params: {num_params}");
    println!("depth=4 params: {num_params}");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p picochat-core`
Expected: FAIL — `model` module not found.

**Step 3: Implement GPT model**

```rust
// crates/picochat-core/src/model.rs
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{linear_no_bias, Embedding, Linear, Module, VarBuilder};

use crate::attention::CausalSelfAttention;
use crate::config::GPTConfig;
use crate::mlp::MLP;
use crate::norm::rms_norm;
use crate::rotary::RotaryEmbedding;

/// A single transformer block: attention + MLP with pre-norm.
struct Block {
    attn: CausalSelfAttention,
    mlp: MLP,
}

impl Block {
    fn new(config: &GPTConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let attn = CausalSelfAttention::new(config, layer_idx, vb.pp("attn"))?;
        let mlp = MLP::new(config.n_embd, vb.pp("mlp"))?;
        Ok(Self { attn, mlp })
    }

    fn forward(
        &self,
        x: &Tensor,
        ve: Option<&Tensor>,
        rope: &RotaryEmbedding,
        rope_offset: usize,
        window_size: (usize, usize),
    ) -> Result<Tensor> {
        // Pre-norm attention
        let x = (x + self.attn.forward(&rms_norm(x)?, ve, rope, rope_offset, window_size, None)?)?;
        // Pre-norm MLP
        let x = (&x + self.mlp.forward(&rms_norm(&x)?)?)?;
        Ok(x)
    }
}

/// Full GPT model.
pub struct GPT {
    wte: Embedding,
    blocks: Vec<Block>,
    lm_head: Linear,
    resid_lambdas: Tensor,
    x0_lambdas: Tensor,
    value_embeds: Vec<Option<Embedding>>,
    rope: RotaryEmbedding,
    window_sizes: Vec<(usize, usize)>,
    config: GPTConfig,
}

impl GPT {
    pub fn new(config: &GPTConfig, vb: VarBuilder) -> Result<Self> {
        let padded_vocab = config.padded_vocab_size();
        let head_dim = config.head_dim();
        let kv_dim = config.n_kv_head * head_dim;

        // Token embedding
        let wte = candle_nn::embedding(padded_vocab, config.n_embd, vb.pp("wte"))?;

        // Transformer blocks
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(Block::new(config, i, vb.pp(format!("h.{i}")))?);
        }

        // LM head (untied from embedding)
        let lm_head = linear_no_bias(config.n_embd, padded_vocab, vb.pp("lm_head"))?;

        // Per-layer learnable scalars
        let resid_lambdas = vb.get((config.n_layer,), "resid_lambdas")?;
        let x0_lambdas = vb.get((config.n_layer,), "x0_lambdas")?;

        // Value embeddings (alternating layers)
        let mut value_embeds = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            if config.has_value_embedding(i) {
                let ve = candle_nn::embedding(padded_vocab, kv_dim, vb.pp(format!("ve.{i}")))?;
                value_embeds.push(Some(ve));
            } else {
                value_embeds.push(None);
            }
        }

        // Rotary embeddings (over-allocate 10x)
        let rope = RotaryEmbedding::new(head_dim, config.sequence_len * 10, 10000.0, vb.device())?;

        let window_sizes = config.compute_window_sizes();

        Ok(Self {
            wte,
            blocks,
            lm_head,
            resid_lambdas,
            x0_lambdas,
            value_embeds,
            rope,
            window_sizes,
            config: config.clone(),
        })
    }

    /// Forward pass.
    /// idx: (B, T) token indices
    /// targets: optional (B, T) target indices for loss computation
    /// Returns logits (B, T, vocab_size) or scalar loss if targets provided.
    pub fn forward(&self, idx: &Tensor, targets: Option<&Tensor>) -> Result<Tensor> {
        let (_b, t) = idx.dims2()?;

        // Embed tokens and normalize
        let mut x = self.wte.forward(idx)?;
        x = rms_norm(&x)?;
        let x0 = x.clone(); // save for x0 residual

        // Forward through transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            // Apply per-layer scalars: x = resid_lambda * x + x0_lambda * x0
            let resid_lambda = self.resid_lambdas.get(i)?.unsqueeze(0)?.unsqueeze(0)?;
            let x0_lambda = self.x0_lambdas.get(i)?.unsqueeze(0)?.unsqueeze(0)?;
            x = (x.broadcast_mul(&resid_lambda)? + x0.broadcast_mul(&x0_lambda)?)?;

            // Value embedding for this layer
            let ve = match &self.value_embeds[i] {
                Some(ve_embed) => Some(ve_embed.forward(idx)?),
                None => None,
            };

            x = block.forward(&x, ve.as_ref(), &self.rope, 0, self.window_sizes[i])?;
        }

        x = rms_norm(&x)?;

        // LM head -> logits
        let logits = self.lm_head.forward(&x)?;
        // Slice to actual vocab size (remove padding)
        let logits = logits.narrow(D::Minus1, 0, self.config.vocab_size)?;
        // Cast to f32 for softcap and loss
        let logits = logits.to_dtype(DType::F32)?;
        // Logit softcap: 15 * tanh(logits / 15)
        let softcap = 15.0f64;
        let logits = ((logits / softcap)?.tanh()? * softcap)?;

        match targets {
            Some(tgt) => {
                // Compute cross-entropy loss
                let (b, t, v) = logits.dims3()?;
                let logits_flat = logits.reshape((b * t, v))?;
                let targets_flat = tgt.flatten_all()?.to_dtype(DType::U32)?;
                let log_sm = candle_nn::ops::log_softmax(&logits_flat, D::Minus1)?;
                candle_nn::loss::nll(&log_sm, &targets_flat)
            }
            None => Ok(logits),
        }
    }

    /// Count total parameters.
    pub fn num_parameters(&self) -> usize {
        // This is approximate since we don't have direct access to all vars
        let config = &self.config;
        let padded_vocab = config.padded_vocab_size();
        let head_dim = config.head_dim();
        let kv_dim = config.n_kv_head * head_dim;

        let mut total = 0;
        // wte
        total += padded_vocab * config.n_embd;
        // lm_head
        total += config.n_embd * padded_vocab;
        // per block: c_q + c_k + c_v + c_proj + c_fc + c_proj_mlp
        for _ in 0..config.n_layer {
            total += config.n_embd * config.n_head * head_dim; // c_q
            total += config.n_embd * kv_dim; // c_k
            total += config.n_embd * kv_dim; // c_v
            total += config.n_embd * config.n_embd; // c_proj
            total += config.n_embd * 4 * config.n_embd; // c_fc
            total += 4 * config.n_embd * config.n_embd; // c_proj_mlp
        }
        // resid_lambdas + x0_lambdas
        total += config.n_layer * 2;
        // value embeddings (alternating layers)
        for i in 0..config.n_layer {
            if config.has_value_embedding(i) {
                total += padded_vocab * kv_dim;
                total += 32 * config.n_kv_head; // ve_gate
            }
        }
        total
    }
}
```

Update lib.rs to add `pub mod model;`

**Step 4: Run tests**

Run: `cargo test -p picochat-core`
Expected: PASS.

**Step 5: Commit**

```bash
git add crates/picochat-core/src/model.rs crates/picochat-core/tests/model_test.rs crates/picochat-core/src/lib.rs
git commit -m "feat: full GPT model with forward pass"
```

---

### Task 8: CLI Entry Point (Smoke Test)

**Files:**
- Modify: `crates/picochat-cli/Cargo.toml`
- Modify: `crates/picochat-cli/src/main.rs`

**Step 1: Write the CLI that creates a model and runs a forward pass**

```rust
// crates/picochat-cli/src/main.rs
use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;

#[derive(Parser)]
#[command(name = "picochat", version, about = "Train and chat with small reasoning LLMs")]
struct Cli {
    /// Model depth (number of transformer layers)
    #[arg(long, default_value_t = 4)]
    depth: usize,

    /// Run a quick smoke test forward pass
    #[arg(long)]
    smoke_test: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.smoke_test {
        println!("picochat smoke test (depth={})", cli.depth);
        let device = Device::Cpu;
        let config = GPTConfig::from_depth(cli.depth);
        println!("Config: n_layer={}, n_embd={}, n_head={}, n_kv_head={}",
            config.n_layer, config.n_embd, config.n_head, config.n_kv_head);
        println!("Vocab: {} (padded: {})", config.vocab_size, config.padded_vocab_size());
        println!("Params: ~{:.1}M", config.n_layer as f64 * 0.5); // rough estimate

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = GPT::new(&config, vb)?;

        let num_params = model.num_parameters();
        println!("Total parameters: {num_params} ({:.2}M)", num_params as f64 / 1e6);

        // Forward pass with dummy tokens
        let input = Tensor::new(&[[1u32, 2, 3, 4, 5, 6, 7, 8]], &device)?;
        println!("Running forward pass with {} tokens...", 8);
        let logits = model.forward(&input, None)?;
        println!("Output logits shape: {:?}", logits.shape());
        println!("Smoke test PASSED!");
    } else {
        println!("picochat v0.1.0 — use --smoke-test to verify setup");
    }

    Ok(())
}
```

**Step 2: Run smoke test**

Run: `cargo run -- --smoke-test --depth 4`
Expected: Prints config, parameters, runs forward pass, prints "Smoke test PASSED!"

**Step 3: Commit**

```bash
git add crates/picochat-cli/
git commit -m "feat: CLI entry point with smoke test forward pass"
```

---

## Summary

After completing all 8 tasks, you will have:

1. A compiling Cargo workspace with 8 crates
2. GPTConfig that scales from depth → all hyperparameters
3. RMSNorm, RoPE, MLP (ReLU^2), Causal Self-Attention (GQA + sliding window)
4. Full GPT model with forward pass (logits and loss)
5. Per-layer scalars (resid_lambdas, x0_lambdas) and value residuals
6. A CLI that runs a smoke test forward pass on CPU
7. Tests for every component

**Next plan**: Phase 2 will add the Muon+AdamW optimizer, data loading, and training loop.

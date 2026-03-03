# Phase 4: Pretraining, SFT, and Evaluation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the complete training pipeline (real pretraining on FineWeb, supervised fine-tuning with dataset mixtures) and evaluation harnesses (BPB, MMLU, GSM8K) so picochat can train and assess actual language models.

**Architecture:** Layered build — metrics module first, then pretraining loop with on-the-fly tokenization, BPB evaluation, SFT data pipeline with loss masking and mixture sampling, SFT training, MMLU log-prob evaluation, GSM8K chain-of-thought evaluation, and finally CLI integration tying it all together.

**Tech Stack:** Rust, candle 0.8, parquet/arrow 53, serde/serde_json, picochat-tokenizer (BPE + special tokens), picochat-data (PackingDataLoader, ParquetTextReader), picochat-train (Trainer, MuonAdamW), picochat-engine (generation).

---

## Existing Infrastructure Reference

**Tokenizer API** (`picochat-tokenizer`):
- `Tokenizer::load(path) -> Result<Self>`
- `tok.encode(text) -> Result<Vec<u32>>`
- `tok.decode(tokens) -> String`
- `tok.bos_id() -> u32`
- `tok.vocab_size() -> usize`
- `tok.special().token_id(SpecialToken::X) -> u32`

**Special Tokens** (`picochat-tokenizer::special`):
- `SpecialToken::{Bos, UserStart, UserEnd, AssistantStart, AssistantEnd, ...}`
- IDs are at end of vocab: `vocab_size - 16` through `vocab_size - 1`

**Data** (`picochat-data`):
- `ParquetTextReader::open(path, column) -> Result<Self>`, `.next_text() -> Result<Option<String>>`
- `PackingDataLoader::new(batch_size, seq_len, bos_id)`, `.add_document(tokens)`, `.next_batch() -> Option<(inputs, targets)>`, `.flush()`
- `TokenDataset::new(tokens)`, `DataLoader::new(dataset, batch_size, seq_len)`

**Training** (`picochat-train`):
- `Trainer::with_schedule(varmap, config, schedule)`, `.train_step(model, input, target) -> Result<Tensor>`
- `checkpoint::{save_varmap, load_varmap, save_config, load_config}`

**Model** (`picochat-core`):
- `GPT::new(config, vb)`, `.forward(idx, targets) -> Result<Tensor>` (returns loss if targets provided, logits otherwise)
- `.forward_with_cache(idx, cache) -> Result<Tensor>` (inference with KV cache)
- `.num_parameters() -> usize`, `.n_layers() -> usize`, `.config() -> &GPTConfig`
- `GPTConfig::from_depth(depth)`, `initialize_weights(varmap, config)`

**Optimizer** (`picochat-optim`):
- `LrSchedule::new(max_lr, warmup_steps, total_steps, min_lr_ratio)`
- `MuonAdamW::from_varmap(varmap, n_embd)`

**Generation** (`picochat-engine`):
- `generate(model, prompt_tokens, config, device) -> Result<Vec<u32>>`
- `GenerationConfig { max_new_tokens, sampling, stop_tokens }`
- `SamplingParams { temperature, top_k, top_p }`

**Cargo PATH**: All commands require `PATH="/home/nullify/.cargo/bin:$PATH"`

---

### Task 1: Metrics Module

**Files:**
- Create: `crates/picochat-train/src/metrics.rs`
- Modify: `crates/picochat-train/src/lib.rs`
- Test: `crates/picochat-train/tests/metrics_test.rs`

**Context:** `TrainingMetrics` computes BPB (bits per byte — a vocab-invariant validation metric), throughput (tokens/sec), and MFU (model FLOPS utilization). This module is used by both the pretrain and SFT loops to report training progress.

**Step 1: Write the failing test**

Create `crates/picochat-train/tests/metrics_test.rs`:

```rust
use picochat_train::metrics::TrainingMetrics;

#[test]
fn test_bpb_basic() {
    // BPB = loss * log2(e) / avg_bytes_per_token
    // With loss=2.0, avg_bytes_per_token=3.5:
    // BPB = 2.0 * 1.4427 / 3.5 = 0.8244
    let bpb = TrainingMetrics::compute_bpb(2.0, 3.5);
    assert!((bpb - 0.8244).abs() < 0.001, "bpb was {bpb}");
}

#[test]
fn test_throughput() {
    // 1000 tokens in 0.5 seconds = 2000 tok/s
    let tok_s = TrainingMetrics::compute_throughput(1000, 0.5);
    assert!((tok_s - 2000.0).abs() < 0.01);
}

#[test]
fn test_mfu() {
    // MFU = 6 * num_params * tokens_per_step / (elapsed * peak_tflops * 1e12)
    // 6 * 1_000_000 * 512 / (1.0 * 100.0 * 1e12)
    // = 3_072_000_000 / 100_000_000_000_000 = 0.00003072
    let mfu = TrainingMetrics::compute_mfu(1_000_000, 512, 1.0, 100.0);
    assert!((mfu - 0.00003072).abs() < 1e-8, "mfu was {mfu}");
}

#[test]
fn test_tracker_accumulation() {
    let mut tracker = TrainingMetrics::new(4.0); // avg 4 bytes per token
    tracker.record_step(2.0, 1024, 0.1); // loss=2.0, 1024 tokens, 0.1s

    assert!((tracker.last_bpb() - (2.0 * std::f64::consts::LOG2_E / 4.0)).abs() < 0.001);
    assert!((tracker.last_throughput() - 10240.0).abs() < 0.01);
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test metrics_test 2>&1 | tail -20`
Expected: FAIL — module `metrics` not found

**Step 3: Write minimal implementation**

Create `crates/picochat-train/src/metrics.rs`:

```rust
/// Training metrics: BPB, throughput, MFU.
pub struct TrainingMetrics {
    avg_bytes_per_token: f64,
    last_bpb: f64,
    last_throughput: f64,
}

impl TrainingMetrics {
    pub fn new(avg_bytes_per_token: f64) -> Self {
        Self {
            avg_bytes_per_token,
            last_bpb: 0.0,
            last_throughput: 0.0,
        }
    }

    /// BPB = loss * log2(e) / avg_bytes_per_token
    pub fn compute_bpb(loss: f64, avg_bytes_per_token: f64) -> f64 {
        loss * std::f64::consts::LOG2_E / avg_bytes_per_token
    }

    /// Throughput in tokens/sec
    pub fn compute_throughput(num_tokens: usize, elapsed_secs: f64) -> f64 {
        num_tokens as f64 / elapsed_secs
    }

    /// MFU = 6 * num_params * tokens_per_step / (elapsed * peak_tflops * 1e12)
    pub fn compute_mfu(
        num_params: usize,
        tokens_per_step: usize,
        elapsed_secs: f64,
        peak_tflops: f64,
    ) -> f64 {
        6.0 * num_params as f64 * tokens_per_step as f64
            / (elapsed_secs * peak_tflops * 1e12)
    }

    /// Record one training step and update tracked metrics.
    pub fn record_step(&mut self, loss: f64, num_tokens: usize, elapsed_secs: f64) {
        self.last_bpb = Self::compute_bpb(loss, self.avg_bytes_per_token);
        self.last_throughput = Self::compute_throughput(num_tokens, elapsed_secs);
    }

    pub fn last_bpb(&self) -> f64 {
        self.last_bpb
    }

    pub fn last_throughput(&self) -> f64 {
        self.last_throughput
    }
}
```

Add `pub mod metrics;` to `crates/picochat-train/src/lib.rs`.

**Step 4: Run test to verify it passes**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test metrics_test 2>&1 | tail -20`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add crates/picochat-train/src/metrics.rs crates/picochat-train/src/lib.rs crates/picochat-train/tests/metrics_test.rs
git commit -m "feat: add training metrics module (BPB, throughput, MFU)"
```

---

### Task 2: Pretrain Pipeline

**Files:**
- Create: `crates/picochat-train/src/pretrain.rs`
- Modify: `crates/picochat-train/src/lib.rs`
- Modify: `crates/picochat-train/Cargo.toml` (add picochat-tokenizer dep)
- Test: `crates/picochat-train/tests/pretrain_test.rs`

**Context:** The pretrain loop streams parquet data, tokenizes on-the-fly, packs into batches via `PackingDataLoader`, runs training steps, and periodically evaluates BPB on validation data. It saves checkpoints at configured intervals.

**Step 1: Write the failing test**

Create `crates/picochat-train/tests/pretrain_test.rs`:

```rust
use picochat_train::pretrain::PretrainConfig;

#[test]
fn test_pretrain_config_defaults() {
    let config = PretrainConfig {
        data_dir: "data/train".to_string(),
        val_data: Some("data/val.parquet".to_string()),
        tokenizer_path: "tokenizer.json".to_string(),
        total_steps: 1000,
        batch_size: 4,
        seq_len: 512,
        max_lr: 0.001,
        warmup_steps: 100,
        min_lr_ratio: 0.1,
        eval_every: 100,
        save_every: 500,
        save_dir: "checkpoints".to_string(),
        depth: 4,
    };
    assert_eq!(config.total_steps, 1000);
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.seq_len, 512);
}

#[test]
fn test_pretrain_tokens_per_step() {
    let config = PretrainConfig {
        data_dir: "data/train".to_string(),
        val_data: None,
        tokenizer_path: "tokenizer.json".to_string(),
        total_steps: 100,
        batch_size: 8,
        seq_len: 256,
        max_lr: 0.001,
        warmup_steps: 10,
        min_lr_ratio: 0.1,
        eval_every: 50,
        save_every: 100,
        save_dir: "checkpoints".to_string(),
        depth: 4,
    };
    assert_eq!(config.tokens_per_step(), 8 * 256);
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test pretrain_test 2>&1 | tail -20`
Expected: FAIL — module `pretrain` not found

**Step 3: Write minimal implementation**

Add `picochat-tokenizer = { path = "../picochat-tokenizer" }` to `crates/picochat-train/Cargo.toml` dependencies.

Create `crates/picochat-train/src/pretrain.rs`:

```rust
use anyhow::Result;
use candle_core::{Device, DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::init::initialize_weights;
use picochat_core::model::GPT;
use picochat_data::dataloader::PackingDataLoader;
use picochat_data::parquet::ParquetTextReader;
use picochat_tokenizer::Tokenizer;
use crate::checkpoint;
use crate::metrics::TrainingMetrics;
use crate::trainer::Trainer;
use picochat_optim::LrSchedule;

pub struct PretrainConfig {
    pub data_dir: String,
    pub val_data: Option<String>,
    pub tokenizer_path: String,
    pub total_steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub max_lr: f64,
    pub warmup_steps: usize,
    pub min_lr_ratio: f64,
    pub eval_every: usize,
    pub save_every: usize,
    pub save_dir: String,
    pub depth: usize,
}

impl PretrainConfig {
    pub fn tokens_per_step(&self) -> usize {
        self.batch_size * self.seq_len
    }
}

/// Collect parquet file paths from a directory, sorted by name.
fn collect_parquet_files(dir: &str) -> Result<Vec<String>> {
    let mut paths: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "parquet") {
                Some(path.to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    paths.sort();
    Ok(paths)
}

/// Fill the packing data loader from a ParquetTextReader until we have
/// at least `min_ready` sequences ready, or the reader is exhausted.
/// Returns the number of bytes of text consumed (for BPB calculation).
fn fill_loader(
    reader: &mut ParquetTextReader,
    tokenizer: &Tokenizer,
    loader: &mut PackingDataLoader,
    min_ready: usize,
) -> Result<(usize, bool)> {
    let mut total_bytes = 0usize;
    let mut exhausted = false;

    while loader.ready_count() < min_ready {
        match reader.next_text()? {
            Some(text) => {
                total_bytes += text.len();
                let tokens = tokenizer.encode(&text)?;
                loader.add_document(&tokens);
            }
            None => {
                exhausted = true;
                break;
            }
        }
    }

    Ok((total_bytes, exhausted))
}

/// Run the pretraining loop.
pub fn pretrain(config: &PretrainConfig, device: &Device) -> Result<()> {
    let tokenizer = Tokenizer::load(&config.tokenizer_path)?;
    let model_config = GPTConfig::from_depth(config.depth);

    println!("Pretrain: depth={}, seq_len={}, batch={}, steps={}",
        config.depth, config.seq_len, config.batch_size, config.total_steps);

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(&model_config, vb)?;
    initialize_weights(&varmap, &model_config)?;

    println!("Parameters: {} ({:.2}M)",
        model.num_parameters(), model.num_parameters() as f64 / 1e6);

    let schedule = LrSchedule::new(
        config.max_lr,
        config.warmup_steps,
        config.total_steps,
        config.min_lr_ratio,
    );
    let mut trainer = Trainer::with_schedule(&varmap, &model_config, schedule);

    // Collect training parquet files
    let train_files = collect_parquet_files(&config.data_dir)?;
    if train_files.is_empty() {
        anyhow::bail!("No parquet files found in {}", config.data_dir);
    }
    println!("Training data: {} parquet files", train_files.len());

    let bos_id = tokenizer.bos_id();
    let mut loader = PackingDataLoader::new(config.batch_size, config.seq_len, bos_id);

    // Estimate avg bytes per token (use 4.0 as default, will refine as we go)
    let mut total_text_bytes: usize = 0;
    let mut total_tokens: usize = 0;

    let mut file_idx = 0;
    let mut reader = ParquetTextReader::open_fineweb(&train_files[file_idx])?;

    let start = std::time::Instant::now();

    for step in 0..config.total_steps {
        // Ensure we have enough data for a batch
        loop {
            let (bytes, exhausted) = fill_loader(
                &mut reader,
                &tokenizer,
                &mut loader,
                config.batch_size,
            )?;
            total_text_bytes += bytes;

            if loader.ready_count() >= config.batch_size {
                break;
            }

            if exhausted {
                file_idx += 1;
                if file_idx >= train_files.len() {
                    // Wrap around to beginning (epoch boundary)
                    file_idx = 0;
                    println!("Epoch boundary: restarting from first file");
                }
                reader = ParquetTextReader::open_fineweb(&train_files[file_idx])?;
            }
        }

        let (input_vecs, target_vecs) = loader.next_batch().unwrap();
        let input = Tensor::new(input_vecs, device)?;
        let target = Tensor::new(target_vecs, device)?;

        let loss = trainer.train_step(&model, &input, &target)?;
        let loss_val: f32 = loss.to_scalar()?;

        total_tokens += config.tokens_per_step();

        if step % 10 == 0 || step == config.total_steps - 1 {
            let elapsed = start.elapsed().as_secs_f64();
            let avg_bpt = if total_tokens > 0 {
                total_text_bytes as f64 / total_tokens as f64
            } else {
                4.0
            };
            let bpb = TrainingMetrics::compute_bpb(loss_val as f64, avg_bpt);
            let tok_s = TrainingMetrics::compute_throughput(total_tokens, elapsed);
            println!(
                "step {:>5}/{} | loss: {:.4} | bpb: {:.4} | tok/s: {:.0}",
                step, config.total_steps, loss_val, bpb, tok_s
            );
        }

        // Save checkpoint
        if config.save_every > 0 && (step + 1) % config.save_every == 0 {
            let ckpt_dir = format!("{}/step-{}", config.save_dir, step + 1);
            std::fs::create_dir_all(&ckpt_dir)?;
            checkpoint::save_varmap(&varmap, format!("{ckpt_dir}/model.safetensors"))?;
            checkpoint::save_config(&model_config, format!("{ckpt_dir}/config.json"))?;
            println!("Checkpoint saved to {ckpt_dir}/");
        }
    }

    // Final save
    std::fs::create_dir_all(&config.save_dir)?;
    checkpoint::save_varmap(&varmap, format!("{}/model.safetensors", config.save_dir))?;
    checkpoint::save_config(&model_config, format!("{}/config.json", config.save_dir))?;
    println!("Final checkpoint saved to {}/", config.save_dir);

    let elapsed = start.elapsed().as_secs_f64();
    println!("Pretraining complete in {:.1}s ({} steps, {} tokens)",
        elapsed, config.total_steps, total_tokens);

    Ok(())
}
```

Add `pub mod pretrain;` to `crates/picochat-train/src/lib.rs`.

**Step 4: Run test to verify it passes**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test pretrain_test 2>&1 | tail -20`
Expected: 2 tests PASS

**Step 5: Build check**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo check -p picochat-train 2>&1 | tail -20`
Expected: Compiles without errors

**Step 6: Commit**

```bash
git add crates/picochat-train/src/pretrain.rs crates/picochat-train/src/lib.rs crates/picochat-train/Cargo.toml crates/picochat-train/tests/pretrain_test.rs
git commit -m "feat: add pretraining pipeline with on-the-fly tokenization"
```

---

### Task 3: BPB Evaluation

**Files:**
- Create: `crates/picochat-eval/src/bpb.rs`
- Modify: `crates/picochat-eval/src/lib.rs`
- Modify: `crates/picochat-eval/Cargo.toml` (add dependencies)
- Test: `crates/picochat-eval/tests/bpb_test.rs`

**Context:** BPB (bits per byte) is the primary validation metric for pretraining. It's vocab-invariant: `bpb = total_nll_nats / (total_bytes * ln(2))`, which is equivalent to `total_loss * log2(e) / total_bytes_ratio`. This module runs forward-only passes on validation data and reports the result.

**Step 1: Write the failing test**

Create `crates/picochat-eval/tests/bpb_test.rs`:

```rust
use picochat_eval::bpb::BpbResult;

#[test]
fn test_bpb_result_display() {
    let result = BpbResult {
        bpb: 1.234,
        num_tokens: 10000,
        num_bytes: 35000,
        avg_loss: 3.0,
    };
    assert!((result.bpb - 1.234).abs() < 0.001);
    assert_eq!(result.num_tokens, 10000);
    assert_eq!(result.num_bytes, 35000);
}

#[test]
fn test_bpb_from_loss_and_bytes() {
    // BPB = total_nll_nats / (total_bytes * ln(2))
    // If avg_loss = 3.0 over 1000 tokens, total_nll = 3000
    // With 3500 bytes: bpb = 3000 / (3500 * 0.6931) = 1.237
    let total_nll = 3.0 * 1000.0;
    let total_bytes = 3500.0;
    let bpb = total_nll / (total_bytes * 2.0f64.ln());
    assert!((bpb - 1.237).abs() < 0.01, "bpb was {bpb}");
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-eval --test bpb_test 2>&1 | tail -20`
Expected: FAIL — module `bpb` not found

**Step 3: Write minimal implementation**

Update `crates/picochat-eval/Cargo.toml`:

```toml
[package]
name = "picochat-eval"
version = "0.1.0"
edition = "2021"

[dependencies]
picochat-core = { path = "../picochat-core" }
picochat-data = { path = "../picochat-data" }
picochat-tokenizer = { path = "../picochat-tokenizer" }
picochat-engine = { path = "../picochat-engine" }
candle-core = { workspace = true }
candle-nn = { workspace = true }
anyhow = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
rand = { workspace = true }
```

Create `crates/picochat-eval/src/bpb.rs`:

```rust
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use picochat_core::model::GPT;
use picochat_data::dataloader::PackingDataLoader;
use picochat_data::parquet::ParquetTextReader;
use picochat_tokenizer::Tokenizer;

pub struct BpbResult {
    pub bpb: f64,
    pub num_tokens: usize,
    pub num_bytes: usize,
    pub avg_loss: f64,
}

/// Evaluate BPB on a validation parquet file.
///
/// 1. Tokenize + pack validation data
/// 2. Forward-only passes, accumulate cross-entropy loss
/// 3. bpb = total_nll_nats / (total_bytes * ln(2))
pub fn evaluate_bpb(
    model: &GPT,
    val_path: &str,
    tokenizer: &Tokenizer,
    batch_size: usize,
    seq_len: usize,
    device: &Device,
) -> Result<BpbResult> {
    let bos_id = tokenizer.bos_id();
    let mut loader = PackingDataLoader::new(batch_size, seq_len, bos_id);

    let mut reader = ParquetTextReader::open_fineweb(val_path)?;
    let mut total_bytes: usize = 0;

    // Load all validation data into the packer
    loop {
        match reader.next_text()? {
            Some(text) => {
                total_bytes += text.len();
                let tokens = tokenizer.encode(&text)?;
                loader.add_document(&tokens);
            }
            None => break,
        }
    }
    loader.flush();

    let mut total_loss = 0.0f64;
    let mut total_tokens: usize = 0;

    // Process all batches
    while let Some((input_vecs, target_vecs)) = loader.next_batch() {
        let input = Tensor::new(input_vecs, device)?;
        let target = Tensor::new(target_vecs, device)?;

        // forward with targets returns scalar loss
        let loss = model.forward(&input, Some(&target))?;
        let loss_val: f32 = loss.to_scalar()?;

        let batch_tokens = batch_size * seq_len;
        total_loss += loss_val as f64 * batch_tokens as f64;
        total_tokens += batch_tokens;
    }

    if total_tokens == 0 {
        anyhow::bail!("No validation tokens found in {val_path}");
    }

    let avg_loss = total_loss / total_tokens as f64;
    let bpb = total_loss / (total_bytes as f64 * 2.0f64.ln());

    Ok(BpbResult {
        bpb,
        num_tokens: total_tokens,
        num_bytes: total_bytes,
        avg_loss,
    })
}
```

Update `crates/picochat-eval/src/lib.rs`:

```rust
// picochat-eval: evaluation and benchmarks
pub mod bpb;
```

**Step 4: Run test to verify it passes**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-eval --test bpb_test 2>&1 | tail -20`
Expected: 2 tests PASS

**Step 5: Build check**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo check -p picochat-eval 2>&1 | tail -20`
Expected: Compiles without errors

**Step 6: Commit**

```bash
git add crates/picochat-eval/src/bpb.rs crates/picochat-eval/src/lib.rs crates/picochat-eval/Cargo.toml crates/picochat-eval/tests/bpb_test.rs
git commit -m "feat: add BPB evaluation for validation data"
```

---

### Task 4: SFT Data Pipeline

**Files:**
- Create: `crates/picochat-data/src/sft.rs`
- Create: `crates/picochat-data/src/mixture.rs`
- Modify: `crates/picochat-data/src/lib.rs`
- Modify: `crates/picochat-data/Cargo.toml` (add serde_json)
- Test: `crates/picochat-data/tests/sft_test.rs`
- Test: `crates/picochat-data/tests/mixture_test.rs`

**Context:** SFT data comes as JSONL chat format:
```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Tokenization wraps with special tokens:
```
<|bos|><|user_start|>...<|user_end|><|assistant_start|>...<|assistant_end|>
```

Loss mask: only assistant content + assistant_end tokens contribute to loss. The mask is aligned with the **target** sequence (shifted by 1):
```
tokens:        BOS  U_START  hello  U_END  A_START  world  A_END
is_assistant:  F    F        F      F      F        T      T
input  = tokens[:-1]
target = tokens[1:]
mask   = is_assistant[1:]   (aligned with targets)
```

**Mixture:** Weighted random sampling across datasets with per-dataset epoch counters and round-robin file reading.

**Step 1: Write the failing test for SFT tokenization**

Create `crates/picochat-data/tests/sft_test.rs`:

```rust
use picochat_data::sft::{ChatMessage, ChatConversation, tokenize_conversation};

#[test]
fn test_chat_message_parse() {
    let json = r#"{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]}"#;
    let conv: ChatConversation = serde_json::from_str(json).unwrap();
    assert_eq!(conv.messages.len(), 2);
    assert_eq!(conv.messages[0].role, "user");
    assert_eq!(conv.messages[0].content, "Hello");
    assert_eq!(conv.messages[1].role, "assistant");
    assert_eq!(conv.messages[1].content, "Hi there");
}

#[test]
fn test_tokenize_conversation_mask_alignment() {
    // Build a mock tokenizer scenario
    // We test the mask logic: only assistant tokens in the TARGET sequence get mask=1
    //
    // tokens:       [BOS, U_START, t1, t2, U_END, A_START, t3, t4, A_END]
    // is_assistant:  F    F        F   F   F      F        T   T   T
    // input:        [BOS, U_START, t1, t2, U_END, A_START, t3, t4]
    // target:       [U_START, t1, t2, U_END, A_START, t3, t4, A_END]
    // mask (=is_assistant[1:]):
    //               [F,       F,  F,  F,     F,       T,  T,  T]
    //
    // So mask should be [0, 0, 0, 0, 0, 1, 1, 1]

    // We'll use a simplified tokenizer for testing — construct tokens + mask directly
    let tokens = vec![100, 200, 10, 20, 201, 202, 30, 40, 203];
    let is_assistant = vec![false, false, false, false, false, false, true, true, true];

    // Derive mask aligned with targets (shifted by 1)
    let mask: Vec<u8> = is_assistant[1..].iter().map(|&b| b as u8).collect();
    let input: Vec<u32> = tokens[..tokens.len()-1].to_vec();
    let target: Vec<u32> = tokens[1..].to_vec();

    assert_eq!(input.len(), target.len());
    assert_eq!(mask.len(), target.len());
    assert_eq!(mask, vec![0, 0, 0, 0, 0, 1, 1, 1]);

    // The target at mask=1 positions should be assistant content and A_END
    assert_eq!(target[5], 30); // first assistant token
    assert_eq!(target[6], 40); // second assistant token
    assert_eq!(target[7], 203); // assistant_end
}

#[test]
fn test_tokenize_conversation_multi_turn() {
    // Multi-turn: user/assistant/user/assistant
    // tokens: BOS U_S u1 U_E A_S a1 A_E U_S u2 U_E A_S a2 A_E
    // is_asst: F  F   F  F   F   T  T   F   F  F   F   T  T
    let is_assistant = vec![
        false, false, false, false, false, true, true,
        false, false, false, false, true, true,
    ];
    let mask: Vec<u8> = is_assistant[1..].iter().map(|&b| b as u8).collect();
    // mask aligned with targets[1:]:
    // F F F F T T F F F F T T
    assert_eq!(mask, vec![0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]);
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-data --test sft_test 2>&1 | tail -20`
Expected: FAIL — module `sft` not found

**Step 3: Write SFT implementation**

Add `serde_json = { workspace = true }` to `crates/picochat-data/Cargo.toml` dependencies.

Create `crates/picochat-data/src/sft.rs`:

```rust
use anyhow::Result;
use picochat_tokenizer::Tokenizer;
use picochat_tokenizer::special::SpecialToken;
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

/// Result of tokenizing a conversation: token IDs and a loss mask.
/// `mask[i]` corresponds to `tokens[i+1]` (the target at position i).
pub struct TokenizedConversation {
    pub tokens: Vec<u32>,
    pub mask: Vec<u8>,
}

/// Tokenize a chat conversation into tokens + loss mask.
///
/// Format: `<|bos|><|user_start|>...<|user_end|><|assistant_start|>...<|assistant_end|>`
///
/// The mask marks which positions in the TARGET sequence contribute to loss.
/// Only assistant content tokens and `<|assistant_end|>` get mask=1.
///
/// Layout:
/// ```text
/// tokens:        BOS  U_START  ...user...  U_END  A_START  ...asst...  A_END
/// is_assistant:  F    F        F...        F      F        T...        T
/// input  = tokens[:-1]
/// target = tokens[1:]
/// mask   = is_assistant[1:]
/// ```
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

    // BOS
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
                tokens.push(assistant_start_id);
                is_assistant.push(false); // A_START itself is not masked

                let content_tokens = tokenizer.encode(&msg.content)?;
                for &t in &content_tokens {
                    tokens.push(t);
                    is_assistant.push(true);
                }

                tokens.push(assistant_end_id);
                is_assistant.push(true); // A_END is masked (model should learn to stop)
            }
            role => {
                anyhow::bail!("Unknown role: {role}");
            }
        }
    }

    // Mask aligned with targets (shifted by 1)
    let mask: Vec<u8> = is_assistant[1..].iter().map(|&b| b as u8).collect();

    Ok(TokenizedConversation { tokens, mask })
}

/// Read JSONL chat data from a file and tokenize all conversations.
pub fn load_sft_data(
    path: &str,
    tokenizer: &Tokenizer,
) -> Result<Vec<TokenizedConversation>> {
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
```

**Step 4: Write the failing test for Mixture**

Create `crates/picochat-data/tests/mixture_test.rs`:

```rust
use picochat_data::mixture::{DatasetMixture, MixtureDataset};

#[test]
fn test_mixture_weighted_sampling() {
    // Dataset A: weight 0.7, Dataset B: weight 0.3
    // Over many samples, A should be chosen ~70% of the time
    let mut mixture = DatasetMixture::new(vec![
        MixtureDataset {
            name: "a".to_string(),
            weight: 0.7,
            items: (0..100).map(|i| vec![i as u32]).collect(),
        },
        MixtureDataset {
            name: "b".to_string(),
            weight: 0.3,
            items: (100..200).map(|i| vec![i as u32]).collect(),
        },
    ]);

    let mut a_count = 0;
    let mut b_count = 0;
    for _ in 0..10000 {
        let item = mixture.sample();
        if item[0] < 100 {
            a_count += 1;
        } else {
            b_count += 1;
        }
    }

    let a_ratio = a_count as f64 / 10000.0;
    assert!((a_ratio - 0.7).abs() < 0.05, "a_ratio was {a_ratio}");
}

#[test]
fn test_mixture_single_dataset() {
    let mut mixture = DatasetMixture::new(vec![
        MixtureDataset {
            name: "only".to_string(),
            weight: 1.0,
            items: vec![vec![42]],
        },
    ]);
    let item = mixture.sample();
    assert_eq!(item, vec![42]);
}

#[test]
fn test_mixture_epoch_cycling() {
    // With only 2 items, after 2 samples the dataset should cycle (epoch boundary)
    let mut mixture = DatasetMixture::new(vec![
        MixtureDataset {
            name: "small".to_string(),
            weight: 1.0,
            items: vec![vec![1], vec![2]],
        },
    ]);

    // After sampling all items, it should wrap around
    let _ = mixture.sample();
    let _ = mixture.sample();
    let item3 = mixture.sample(); // should still work (wrapped)
    assert!(item3 == vec![1] || item3 == vec![2]);
}
```

**Step 5: Write Mixture implementation**

Create `crates/picochat-data/src/mixture.rs`:

```rust
use rand::Rng;

pub struct MixtureDataset {
    pub name: String,
    pub weight: f64,
    pub items: Vec<Vec<u32>>,
}

/// Weighted random sampling across multiple datasets.
///
/// Each dataset has a weight and a set of items. On each `sample()` call,
/// a dataset is chosen by cumulative weight, then a random item from that
/// dataset is returned. When all items in a dataset have been seen, the
/// dataset's index resets (epoch boundary).
pub struct DatasetMixture {
    datasets: Vec<MixtureDataset>,
    cumulative_weights: Vec<f64>,
    cursors: Vec<usize>,
    shuffled_indices: Vec<Vec<usize>>,
    rng: rand::rngs::ThreadRng,
}

impl DatasetMixture {
    pub fn new(datasets: Vec<MixtureDataset>) -> Self {
        let total_weight: f64 = datasets.iter().map(|d| d.weight).sum();
        let mut cumulative_weights = Vec::with_capacity(datasets.len());
        let mut cum = 0.0;
        for d in &datasets {
            cum += d.weight / total_weight;
            cumulative_weights.push(cum);
        }

        let mut rng = rand::thread_rng();
        let shuffled_indices: Vec<Vec<usize>> = datasets
            .iter()
            .map(|d| {
                let mut indices: Vec<usize> = (0..d.items.len()).collect();
                shuffle_with_rng(&mut indices, &mut rng);
                indices
            })
            .collect();

        let cursors = vec![0; datasets.len()];

        Self {
            datasets,
            cumulative_weights,
            cursors,
            shuffled_indices,
            rng,
        }
    }

    /// Sample one item using weighted random selection.
    pub fn sample(&mut self) -> Vec<u32> {
        let r: f64 = self.rng.gen();
        let mut dataset_idx = self.datasets.len() - 1;
        for (i, &cw) in self.cumulative_weights.iter().enumerate() {
            if r < cw {
                dataset_idx = i;
                break;
            }
        }

        let cursor = self.cursors[dataset_idx];
        if cursor >= self.datasets[dataset_idx].items.len() {
            // Epoch boundary: reshuffle and reset
            let mut indices: Vec<usize> = (0..self.datasets[dataset_idx].items.len()).collect();
            shuffle_with_rng(&mut indices, &mut self.rng);
            self.shuffled_indices[dataset_idx] = indices;
            self.cursors[dataset_idx] = 0;
        }

        let idx = self.shuffled_indices[dataset_idx][self.cursors[dataset_idx]];
        self.cursors[dataset_idx] += 1;

        self.datasets[dataset_idx].items[idx].clone()
    }
}

fn shuffle_with_rng(slice: &mut [usize], rng: &mut rand::rngs::ThreadRng) {
    for i in (1..slice.len()).rev() {
        let j = rng.gen_range(0..=i);
        slice.swap(i, j);
    }
}
```

Update `crates/picochat-data/src/lib.rs`:

```rust
// picochat-data: data loading and preprocessing
pub mod dataloader;
pub mod parquet;
pub mod sft;
pub mod mixture;
```

**Step 6: Run tests to verify they pass**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-data --test sft_test --test mixture_test 2>&1 | tail -20`
Expected: 6 tests PASS (3 sft + 3 mixture)

**Step 7: Commit**

```bash
git add crates/picochat-data/src/sft.rs crates/picochat-data/src/mixture.rs crates/picochat-data/src/lib.rs crates/picochat-data/Cargo.toml crates/picochat-data/tests/sft_test.rs crates/picochat-data/tests/mixture_test.rs
git commit -m "feat: add SFT data pipeline with chat tokenization, loss masking, and dataset mixture"
```

---

### Task 5: SFT Training Loop

**Files:**
- Create: `crates/picochat-train/src/sft.rs`
- Modify: `crates/picochat-train/src/lib.rs`
- Modify: `crates/picochat-train/src/trainer.rs` (add accessor methods)
- Test: `crates/picochat-train/tests/sft_test.rs`

**Context:** SFT inherits a pretrained checkpoint, uses the SFT data pipeline (tokenized conversations with loss masks), and applies masked cross-entropy loss so only assistant tokens contribute to the gradient. Uses shorter training horizon and lower learning rate than pretraining.

**Step 1: Write the failing test**

Create `crates/picochat-train/tests/sft_test.rs`:

```rust
use candle_core::{DType, Device, Tensor, D};

#[test]
fn test_masked_cross_entropy_basic() {
    use picochat_train::sft::masked_cross_entropy;

    let device = Device::Cpu;
    // (1, 4, 3) logits — batch=1, seq=4, vocab=3
    let logits = Tensor::new(
        &[[[2.0f32, 1.0, 0.0],
           [0.0, 2.0, 1.0],
           [1.0, 0.0, 2.0],
           [2.0, 1.0, 0.0]]],
        &device,
    ).unwrap();

    // Targets
    let targets = Tensor::new(&[[0u32, 1, 2, 0]], &device).unwrap();

    // Mask: only positions 2 and 3 contribute to loss
    let mask = Tensor::new(&[[0.0f32, 0.0, 1.0, 1.0]], &device).unwrap();

    let loss = masked_cross_entropy(&logits, &targets, &mask).unwrap();
    let loss_val: f32 = loss.to_scalar().unwrap();

    // Position 2: logits [1, 0, 2], target 2 -> log_softmax[-1] -> -0.4076
    // Position 3: logits [2, 1, 0], target 0 -> log_softmax[-1] -> -0.4076
    // Mean of masked: (0.4076 + 0.4076) / 2 = 0.4076
    assert!(loss_val > 0.0, "loss should be positive");
    assert!((loss_val - 0.4076).abs() < 0.01, "loss was {loss_val}");
}

#[test]
fn test_masked_cross_entropy_all_masked() {
    use picochat_train::sft::masked_cross_entropy;

    let device = Device::Cpu;
    let logits = Tensor::new(
        &[[[2.0f32, 1.0, 0.0], [0.0, 2.0, 1.0]]],
        &device,
    ).unwrap();
    let targets = Tensor::new(&[[0u32, 1]], &device).unwrap();
    let mask = Tensor::new(&[[0.0f32, 0.0]], &device).unwrap();

    let loss = masked_cross_entropy(&logits, &targets, &mask).unwrap();
    let loss_val: f32 = loss.to_scalar().unwrap();
    // With all zeros mask, loss should be ~0 (divided by epsilon)
    assert!(loss_val.abs() < 0.1, "loss with all-zero mask should be ~0, was {loss_val}");
}

#[test]
fn test_sft_config() {
    use picochat_train::sft::SftConfig;

    let config = SftConfig {
        checkpoint_dir: "ckpt".to_string(),
        tokenizer_path: "tok.json".to_string(),
        datasets: vec![("data/chat.jsonl".to_string(), 1.0)],
        total_steps: 500,
        batch_size: 2,
        seq_len: 256,
        max_lr: 0.0001,
        warmup_steps: 50,
        min_lr_ratio: 0.01,
        save_dir: "sft_ckpt".to_string(),
        save_every: 250,
    };
    assert_eq!(config.total_steps, 500);
    assert_eq!(config.datasets.len(), 1);
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test sft_test 2>&1 | tail -20`
Expected: FAIL — module `sft` not found

**Step 3: Add accessor methods to Trainer**

In `crates/picochat-train/src/trainer.rs`, add these methods to the `impl Trainer` block:

```rust
    pub fn schedule_ref(&self) -> Option<&LrSchedule> {
        self.schedule.as_ref()
    }

    pub fn optimizer_mut(&mut self) -> &mut MuonAdamW {
        &mut self.optimizer
    }
```

**Step 4: Write SFT implementation**

Create `crates/picochat-train/src/sft.rs`:

```rust
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use picochat_core::config::GPTConfig;
use picochat_core::model::GPT;
use picochat_data::mixture::{DatasetMixture, MixtureDataset};
use picochat_data::sft::{load_sft_data, TokenizedConversation};
use picochat_tokenizer::Tokenizer;
use picochat_optim::LrSchedule;
use crate::checkpoint;
use crate::trainer::Trainer;

pub struct SftConfig {
    pub checkpoint_dir: String,
    pub tokenizer_path: String,
    /// Vec of (jsonl_path, weight)
    pub datasets: Vec<(String, f64)>,
    pub total_steps: usize,
    pub batch_size: usize,
    pub seq_len: usize,
    pub max_lr: f64,
    pub warmup_steps: usize,
    pub min_lr_ratio: f64,
    pub save_dir: String,
    pub save_every: usize,
}

/// Masked cross-entropy loss: only positions where mask=1 contribute.
///
/// - `logits`: (B, T, vocab)
/// - `targets`: (B, T)
/// - `mask`: (B, T) float tensor, 1.0 for positions that contribute to loss
pub fn masked_cross_entropy(
    logits: &Tensor,
    targets: &Tensor,
    mask: &Tensor,
) -> candle_core::Result<Tensor> {
    let (b, t, vocab) = logits.dims3()?;
    let logits_flat = logits.reshape((b * t, vocab))?;
    let targets_flat = targets.flatten_all()?.to_dtype(DType::U32)?;
    let mask_flat = mask.flatten_all()?;

    let log_sm = candle_nn::ops::log_softmax(&logits_flat, D::Minus1)?;
    let targets_idx = targets_flat.unsqueeze(1)?;
    let nll_per_token = log_sm.gather(&targets_idx, 1)?.squeeze(1)?.neg()?;

    let masked_nll = (&nll_per_token * &mask_flat)?;
    let mask_sum = (mask_flat.sum_all()?.to_dtype(DType::F32)? + 1e-8)?;
    masked_nll.sum_all()?.to_dtype(DType::F32)?.div(&mask_sum)
}

/// Prepare a batch from tokenized conversations by truncating/padding to seq_len.
/// Returns (input, target, mask) tensors.
fn prepare_batch(
    conversations: &[Vec<u32>],
    masks: &[Vec<u8>],
    seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let batch_size = conversations.len();
    let mut input_vecs: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut target_vecs: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
    let mut mask_vecs: Vec<Vec<f32>> = Vec::with_capacity(batch_size);

    for (tokens, mask) in conversations.iter().zip(masks.iter()) {
        let len = (tokens.len() - 1).min(seq_len);
        let mut input = tokens[..len].to_vec();
        let mut target = tokens[1..len + 1].to_vec();
        let mut m: Vec<f32> = mask[..len].iter().map(|&b| b as f32).collect();

        // Pad if needed
        while input.len() < seq_len {
            input.push(0);
            target.push(0);
            m.push(0.0);
        }

        input_vecs.push(input);
        target_vecs.push(target);
        mask_vecs.push(m);
    }

    let input = Tensor::new(input_vecs, device)?;
    let target = Tensor::new(target_vecs, device)?;
    let mask = Tensor::new(mask_vecs, device)?;

    Ok((input, target, mask))
}

/// Run the SFT training loop.
pub fn sft(config: &SftConfig, device: &Device) -> Result<()> {
    let tokenizer = Tokenizer::load(&config.tokenizer_path)?;

    // Load pretrained checkpoint
    let model_config = checkpoint::load_config(format!("{}/config.json", config.checkpoint_dir))?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = GPT::new(&model_config, vb)?;
    checkpoint::load_varmap(
        &varmap,
        format!("{}/model.safetensors", config.checkpoint_dir),
        device,
    )?;

    println!("SFT: loaded checkpoint from {}", config.checkpoint_dir);
    println!("Parameters: {} ({:.2}M)",
        model.num_parameters(), model.num_parameters() as f64 / 1e6);

    // Load and tokenize all SFT datasets
    let mut mixture_datasets = Vec::new();
    for (path, weight) in &config.datasets {
        let tokenized = load_sft_data(path, &tokenizer)?;
        let items: Vec<Vec<u32>> = tokenized.iter().map(|tc| tc.tokens.clone()).collect();
        let masks_data: Vec<Vec<u8>> = tokenized.iter().map(|tc| tc.mask.clone()).collect();

        println!("SFT dataset '{}': {} conversations, weight={}", path, items.len(), weight);

        // Store tokens and masks together as interleaved items
        // We encode mask length at the start so we can split later
        let combined: Vec<Vec<u32>> = items.iter().zip(masks_data.iter()).map(|(tokens, mask)| {
            let mut combined = Vec::with_capacity(tokens.len() + mask.len() + 1);
            combined.push(tokens.len() as u32); // delimiter: token count
            combined.extend_from_slice(tokens);
            combined.extend(mask.iter().map(|&b| b as u32));
            combined
        }).collect();

        mixture_datasets.push(MixtureDataset {
            name: path.clone(),
            weight: *weight,
            items: combined,
        });
    }

    let mut mixture = DatasetMixture::new(mixture_datasets);

    let schedule = LrSchedule::new(
        config.max_lr,
        config.warmup_steps,
        config.total_steps,
        config.min_lr_ratio,
    );
    let mut trainer = Trainer::with_schedule(&varmap, &model_config, schedule);

    println!("SFT training: {} steps, batch={}, seq_len={}",
        config.total_steps, config.batch_size, config.seq_len);

    let start = std::time::Instant::now();

    for step in 0..config.total_steps {
        // Sample a batch
        let mut batch_tokens: Vec<Vec<u32>> = Vec::with_capacity(config.batch_size);
        let mut batch_masks: Vec<Vec<u8>> = Vec::with_capacity(config.batch_size);

        for _ in 0..config.batch_size {
            let combined = mixture.sample();
            let token_count = combined[0] as usize;
            let tokens = combined[1..1 + token_count].to_vec();
            let mask: Vec<u8> = combined[1 + token_count..]
                .iter()
                .map(|&v| v as u8)
                .collect();
            batch_tokens.push(tokens);
            batch_masks.push(mask);
        }

        let (input, target, mask) = prepare_batch(
            &batch_tokens,
            &batch_masks,
            config.seq_len,
            device,
        )?;

        // Forward pass (returns logits since we don't pass targets)
        let logits = model.forward(&input, None)?;
        let loss = masked_cross_entropy(&logits, &target, &mask)?;

        // Backward + optimizer step
        let sched = trainer.schedule_ref().cloned();
        match sched {
            Some(sched) => {
                let base_lr = sched.base_lr();
                let current_lr = sched.get_lr(step);
                let mult = if base_lr > 0.0 {
                    current_lr / base_lr
                } else {
                    1.0
                };
                trainer.optimizer_mut().backward_step_with_lr(&loss, mult)?;
            }
            None => {
                trainer.optimizer_mut().backward_step(&loss)?;
            }
        }

        if step % 10 == 0 || step == config.total_steps - 1 {
            let loss_val: f32 = loss.to_scalar()?;
            let elapsed = start.elapsed().as_secs_f64();
            let tok_s = ((step + 1) * config.batch_size * config.seq_len) as f64 / elapsed;
            println!(
                "sft step {:>4}/{} | loss: {:.4} | tok/s: {:.0}",
                step, config.total_steps, loss_val, tok_s
            );
        }

        // Checkpoint
        if config.save_every > 0 && (step + 1) % config.save_every == 0 {
            let ckpt_dir = format!("{}/step-{}", config.save_dir, step + 1);
            std::fs::create_dir_all(&ckpt_dir)?;
            checkpoint::save_varmap(&varmap, format!("{ckpt_dir}/model.safetensors"))?;
            checkpoint::save_config(&model_config, format!("{ckpt_dir}/config.json"))?;
            println!("SFT checkpoint saved to {ckpt_dir}/");
        }
    }

    // Final save
    std::fs::create_dir_all(&config.save_dir)?;
    checkpoint::save_varmap(&varmap, format!("{}/model.safetensors", config.save_dir))?;
    checkpoint::save_config(&model_config, format!("{}/config.json", config.save_dir))?;
    println!("SFT complete. Checkpoint saved to {}/", config.save_dir);

    Ok(())
}
```

Add `pub mod sft;` to `crates/picochat-train/src/lib.rs`.

**Step 5: Run test to verify it passes**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-train --test sft_test 2>&1 | tail -20`
Expected: 3 tests PASS

**Step 6: Build check**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo check -p picochat-train 2>&1 | tail -20`
Expected: Compiles without errors

**Step 7: Commit**

```bash
git add crates/picochat-train/src/sft.rs crates/picochat-train/src/lib.rs crates/picochat-train/src/trainer.rs crates/picochat-train/tests/sft_test.rs
git commit -m "feat: add SFT training loop with masked cross-entropy loss"
```

---

### Task 6: MMLU Evaluation

**Files:**
- Create: `crates/picochat-eval/src/mmlu.rs`
- Modify: `crates/picochat-eval/src/lib.rs`
- Test: `crates/picochat-eval/tests/mmlu_test.rs`

**Context:** MMLU is a 5-shot multiple-choice benchmark. For each question, we format 5 exemplars + the test question, then compare log-probabilities of tokens A(65), B(66), C(67), D(68) to pick the answer. We report per-subject and overall accuracy.

**Step 1: Write the failing test**

Create `crates/picochat-eval/tests/mmlu_test.rs`:

```rust
use picochat_eval::mmlu::{MmluQuestion, format_mmlu_prompt, pick_answer_from_logprobs};

#[test]
fn test_format_mmlu_prompt() {
    let exemplars = vec![
        MmluQuestion {
            question: "What is 1+1?".to_string(),
            choices: vec!["1".to_string(), "2".to_string(), "3".to_string(), "4".to_string()],
            answer: 1, // B
        },
    ];
    let test_q = MmluQuestion {
        question: "What is 2+2?".to_string(),
        choices: vec!["2".to_string(), "3".to_string(), "4".to_string(), "5".to_string()],
        answer: 2, // C
    };

    let prompt = format_mmlu_prompt(&exemplars, &test_q, "math");
    assert!(prompt.contains("What is 1+1?"));
    assert!(prompt.contains("Answer: B"));
    assert!(prompt.contains("What is 2+2?"));
    assert!(prompt.contains("A. 2"));
    assert!(prompt.contains("B. 3"));
    assert!(prompt.contains("C. 4"));
    assert!(prompt.contains("D. 5"));
    assert!(prompt.ends_with("Answer:"));
}

#[test]
fn test_pick_answer_from_logprobs() {
    // Log-probs for A, B, C, D tokens
    // Highest log-prob at index 2 (C)
    let logprobs = vec![-2.0f32, -1.5, -0.5, -3.0];
    let answer = pick_answer_from_logprobs(&logprobs);
    assert_eq!(answer, 2); // C
}

#[test]
fn test_pick_answer_tie_favors_first() {
    let logprobs = vec![-1.0f32, -1.0, -2.0, -2.0];
    let answer = pick_answer_from_logprobs(&logprobs);
    assert_eq!(answer, 0); // A (first highest)
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-eval --test mmlu_test 2>&1 | tail -20`
Expected: FAIL — module `mmlu` not found

**Step 3: Write MMLU implementation**

Create `crates/picochat-eval/src/mmlu.rs`:

```rust
use anyhow::Result;
use candle_core::{Device, DType, Tensor, D};
use picochat_core::model::GPT;
use picochat_tokenizer::Tokenizer;
use std::collections::HashMap;

pub struct MmluQuestion {
    pub question: String,
    pub choices: Vec<String>,
    pub answer: usize, // 0=A, 1=B, 2=C, 3=D
}

pub struct MmluResult {
    pub overall_accuracy: f64,
    pub subject_accuracy: HashMap<String, f64>,
    pub num_correct: usize,
    pub num_total: usize,
}

const ANSWER_LABELS: &[&str] = &["A", "B", "C", "D"];

/// Format one question with choices.
fn format_question(q: &MmluQuestion) -> String {
    let mut s = format!("{}\n", q.question);
    for (i, choice) in q.choices.iter().enumerate() {
        s.push_str(&format!("{}. {}\n", ANSWER_LABELS[i], choice));
    }
    s
}

/// Format the full few-shot prompt for one MMLU test question.
pub fn format_mmlu_prompt(
    exemplars: &[MmluQuestion],
    test_question: &MmluQuestion,
    subject: &str,
) -> String {
    let mut prompt = format!(
        "The following are multiple choice questions (with answers) about {}.\n\n",
        subject.replace('_', " ")
    );

    for ex in exemplars {
        prompt.push_str(&format_question(ex));
        prompt.push_str(&format!("Answer: {}\n\n", ANSWER_LABELS[ex.answer]));
    }

    prompt.push_str(&format_question(test_question));
    prompt.push_str("Answer:");
    prompt
}

/// Given log-probabilities for A, B, C, D, return the index of the highest.
pub fn pick_answer_from_logprobs(logprobs: &[f32]) -> usize {
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &lp) in logprobs.iter().enumerate() {
        if lp > best_val {
            best_val = lp;
            best_idx = i;
        }
    }
    best_idx
}

/// Evaluate MMLU on a set of subjects.
///
/// `data` maps subject name -> (exemplars, test_questions).
/// Returns overall and per-subject accuracy.
pub fn evaluate_mmlu(
    model: &GPT,
    tokenizer: &Tokenizer,
    data: &HashMap<String, (Vec<MmluQuestion>, Vec<MmluQuestion>)>,
    device: &Device,
) -> Result<MmluResult> {
    // Token IDs for "A", "B", "C", "D" (ASCII bytes)
    let answer_token_ids: Vec<u32> = vec![65, 66, 67, 68]; // A=65, B=66, C=67, D=68

    let mut num_correct = 0;
    let mut num_total = 0;
    let mut subject_accuracy: HashMap<String, f64> = HashMap::new();

    for (subject, (exemplars, test_questions)) in data {
        let mut subject_correct = 0;
        let mut subject_total = 0;

        for test_q in test_questions {
            let prompt = format_mmlu_prompt(exemplars, test_q, subject);
            let tokens = tokenizer.encode(&prompt)?;

            let input = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, None)?;

            // Get logits at the last position
            let last_pos = logits.dim(1)? - 1;
            let last_logits = logits.get(0)?.get(last_pos)?;
            let log_sm = candle_nn::ops::log_softmax(&last_logits.unsqueeze(0)?, D::Minus1)?
                .squeeze(0)?;
            let log_sm_vec: Vec<f32> = log_sm.to_vec1()?;

            let answer_logprobs: Vec<f32> = answer_token_ids
                .iter()
                .map(|&id| log_sm_vec[id as usize])
                .collect();

            let predicted = pick_answer_from_logprobs(&answer_logprobs);
            if predicted == test_q.answer {
                subject_correct += 1;
                num_correct += 1;
            }
            subject_total += 1;
            num_total += 1;
        }

        let acc = if subject_total > 0 {
            subject_correct as f64 / subject_total as f64
        } else {
            0.0
        };
        subject_accuracy.insert(subject.clone(), acc);
        println!("MMLU {}: {}/{} ({:.1}%)",
            subject, subject_correct, subject_total, acc * 100.0);
    }

    let overall_accuracy = if num_total > 0 {
        num_correct as f64 / num_total as f64
    } else {
        0.0
    };
    println!("MMLU overall: {}/{} ({:.1}%)",
        num_correct, num_total, overall_accuracy * 100.0);

    Ok(MmluResult {
        overall_accuracy,
        subject_accuracy,
        num_correct,
        num_total,
    })
}

/// Load MMLU questions from a CSV file.
/// Expected format: question,A,B,C,D,answer (where answer is A/B/C/D)
pub fn load_mmlu_csv(path: &str) -> Result<Vec<MmluQuestion>> {
    let content = std::fs::read_to_string(path)?;
    let mut questions = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }
        let question = parts[0].to_string();
        let choices = vec![
            parts[1].to_string(),
            parts[2].to_string(),
            parts[3].to_string(),
            parts[4].to_string(),
        ];
        let answer = match parts[5].trim() {
            "A" => 0,
            "B" => 1,
            "C" => 2,
            "D" => 3,
            _ => continue,
        };
        questions.push(MmluQuestion {
            question,
            choices,
            answer,
        });
    }

    Ok(questions)
}
```

Update `crates/picochat-eval/src/lib.rs` to add `pub mod mmlu;`.

**Step 4: Run test to verify it passes**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-eval --test mmlu_test 2>&1 | tail -20`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add crates/picochat-eval/src/mmlu.rs crates/picochat-eval/src/lib.rs crates/picochat-eval/tests/mmlu_test.rs
git commit -m "feat: add MMLU few-shot evaluation with log-prob scoring"
```

---

### Task 7: GSM8K Evaluation

**Files:**
- Create: `crates/picochat-eval/src/gsm8k.rs`
- Modify: `crates/picochat-eval/src/lib.rs`
- Test: `crates/picochat-eval/tests/gsm8k_test.rs`

**Context:** GSM8K is a math word problem benchmark. We use few-shot chain-of-thought prompting: show exemplars with step-by-step reasoning ending with `#### <answer>`, then generate the model's response and extract the final numeric answer after `####`.

**Step 1: Write the failing test**

Create `crates/picochat-eval/tests/gsm8k_test.rs`:

```rust
use picochat_eval::gsm8k::{extract_answer, GsmQuestion, format_gsm_prompt};

#[test]
fn test_extract_answer_basic() {
    let response = "First we add 3 + 4 = 7. Then multiply by 2.\n#### 14";
    assert_eq!(extract_answer(response), Some("14".to_string()));
}

#[test]
fn test_extract_answer_with_comma() {
    let response = "The total is #### 1,234";
    assert_eq!(extract_answer(response), Some("1234".to_string()));
}

#[test]
fn test_extract_answer_negative() {
    let response = "#### -5";
    assert_eq!(extract_answer(response), Some("-5".to_string()));
}

#[test]
fn test_extract_answer_none() {
    let response = "I don't know the answer";
    assert_eq!(extract_answer(response), None);
}

#[test]
fn test_extract_answer_decimal() {
    let response = "#### 3.14";
    assert_eq!(extract_answer(response), Some("3.14".to_string()));
}

#[test]
fn test_format_gsm_prompt() {
    let exemplars = vec![
        GsmQuestion {
            question: "What is 2+3?".to_string(),
            answer: "2+3=5\n#### 5".to_string(),
        },
    ];
    let test_q = GsmQuestion {
        question: "What is 4+5?".to_string(),
        answer: "#### 9".to_string(),
    };

    let prompt = format_gsm_prompt(&exemplars, &test_q);
    assert!(prompt.contains("What is 2+3?"));
    assert!(prompt.contains("#### 5"));
    assert!(prompt.contains("What is 4+5?"));
    assert!(prompt.ends_with("A: Let's think step by step.\n"));
}
```

**Step 2: Run test to verify it fails**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-eval --test gsm8k_test 2>&1 | tail -20`
Expected: FAIL — module `gsm8k` not found

**Step 3: Write GSM8K implementation**

Create `crates/picochat-eval/src/gsm8k.rs`:

```rust
use anyhow::Result;
use candle_core::Device;
use picochat_core::model::GPT;
use picochat_core::kv_cache::KVCache;
use picochat_engine::generate::{generate, GenerationConfig};
use picochat_engine::sampling::SamplingParams;
use picochat_tokenizer::Tokenizer;

pub struct GsmQuestion {
    pub question: String,
    pub answer: String,
}

pub struct GsmResult {
    pub solve_rate: f64,
    pub num_correct: usize,
    pub num_total: usize,
}

/// Extract the final numeric answer after "####" from a response string.
/// Strips commas and whitespace. Returns None if no "####" found.
pub fn extract_answer(response: &str) -> Option<String> {
    let marker = "####";
    let idx = response.rfind(marker)?;
    let after = &response[idx + marker.len()..];
    let answer = after.trim().replace(',', "");
    // Take first whitespace-delimited token
    let answer = answer.split_whitespace().next()?.to_string();
    if answer.is_empty() {
        None
    } else {
        Some(answer)
    }
}

/// Normalize an answer for comparison: strip whitespace, commas, trailing periods.
fn normalize_answer(s: &str) -> String {
    s.trim().replace(',', "").trim_end_matches('.').to_string()
}

/// Format the few-shot chain-of-thought prompt for GSM8K.
pub fn format_gsm_prompt(exemplars: &[GsmQuestion], test_question: &GsmQuestion) -> String {
    let mut prompt = String::new();

    for ex in exemplars {
        prompt.push_str(&format!("Q: {}\nA: {}\n\n", ex.question, ex.answer));
    }

    prompt.push_str(&format!("Q: {}\nA: Let's think step by step.\n", test_question.question));
    prompt
}

/// Evaluate GSM8K with few-shot chain-of-thought generation.
pub fn evaluate_gsm8k(
    model: &GPT,
    tokenizer: &Tokenizer,
    exemplars: &[GsmQuestion],
    test_questions: &[GsmQuestion],
    max_new_tokens: usize,
    device: &Device,
) -> Result<GsmResult> {
    let mut num_correct = 0;
    let mut num_total = 0;

    let gen_config = GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams {
            temperature: 0.0, // greedy for evaluation
            top_k: 1,
            top_p: 1.0,
        },
        stop_tokens: vec![], // generate until max_new_tokens
    };

    for (i, test_q) in test_questions.iter().enumerate() {
        let prompt = format_gsm_prompt(exemplars, test_q);
        let prompt_tokens = tokenizer.encode(&prompt)?;

        let output_tokens = generate(model, &prompt_tokens, &gen_config, device)?;
        let response = tokenizer.decode(&output_tokens);

        let predicted = extract_answer(&response);
        let expected = extract_answer(&test_q.answer);

        let correct = match (&predicted, &expected) {
            (Some(p), Some(e)) => normalize_answer(p) == normalize_answer(e),
            _ => false,
        };

        if correct {
            num_correct += 1;
        }
        num_total += 1;

        if (i + 1) % 50 == 0 || i == test_questions.len() - 1 {
            println!("GSM8K: {}/{} ({}/{} correct, {:.1}%)",
                i + 1, test_questions.len(),
                num_correct, num_total,
                num_correct as f64 / num_total as f64 * 100.0);
        }
    }

    let solve_rate = if num_total > 0 {
        num_correct as f64 / num_total as f64
    } else {
        0.0
    };

    println!("GSM8K final: {}/{} ({:.1}%)", num_correct, num_total, solve_rate * 100.0);

    Ok(GsmResult {
        solve_rate,
        num_correct,
        num_total,
    })
}

/// Load GSM8K questions from a JSONL file.
/// Expected format: {"question": "...", "answer": "..."}
pub fn load_gsm8k_jsonl(path: &str) -> Result<Vec<GsmQuestion>> {
    let content = std::fs::read_to_string(path)?;
    let mut questions = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let question = v["question"].as_str().unwrap_or("").to_string();
        let answer = v["answer"].as_str().unwrap_or("").to_string();
        if !question.is_empty() && !answer.is_empty() {
            questions.push(GsmQuestion { question, answer });
        }
    }

    Ok(questions)
}
```

Update `crates/picochat-eval/src/lib.rs` to add `pub mod gsm8k;`.

**Step 4: Run test to verify it passes**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo test -p picochat-eval --test gsm8k_test 2>&1 | tail -20`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add crates/picochat-eval/src/gsm8k.rs crates/picochat-eval/src/lib.rs crates/picochat-eval/tests/gsm8k_test.rs
git commit -m "feat: add GSM8K chain-of-thought evaluation"
```

---

### Task 8: CLI Integration

**Files:**
- Modify: `crates/picochat-cli/src/main.rs`
- Modify: `crates/picochat-cli/Cargo.toml` (add picochat-eval dep)
- Test: manual smoke test via `--help`

**Context:** Add `--pretrain`, `--sft`, and `--eval-bpb`/`--eval-mmlu`/`--eval-gsm8k` flags to the CLI that invoke the new pipeline components. Keep it simple — each flag maps directly to one function call.

**Step 1: Update CLI Cargo.toml**

Add `picochat-eval = { path = "../picochat-eval" }` to `crates/picochat-cli/Cargo.toml` dependencies.

**Step 2: Update main.rs**

Add these fields to the `Cli` struct:

```rust
    /// Pretrain on parquet data
    #[arg(long)]
    pretrain: bool,

    /// SFT on JSONL chat data
    #[arg(long)]
    sft: bool,

    /// Evaluate BPB on validation data
    #[arg(long)]
    eval_bpb: bool,

    /// Evaluate MMLU
    #[arg(long)]
    eval_mmlu: bool,

    /// Evaluate GSM8K
    #[arg(long)]
    eval_gsm8k: bool,

    /// Path to validation data (for BPB evaluation)
    #[arg(long)]
    val_data: Option<String>,

    /// Maximum learning rate
    #[arg(long, default_value_t = 0.001)]
    max_lr: f64,

    /// Warmup steps
    #[arg(long, default_value_t = 100)]
    warmup_steps: usize,

    /// Minimum LR ratio
    #[arg(long, default_value_t = 0.1)]
    min_lr_ratio: f64,

    /// Checkpoint save frequency
    #[arg(long, default_value_t = 500)]
    save_every: usize,

    /// Evaluation frequency during pretraining
    #[arg(long, default_value_t = 100)]
    eval_every: usize,

    /// SFT dataset paths and weights (format: path:weight,path:weight)
    #[arg(long)]
    sft_data: Option<String>,

    /// Maximum tokens to generate for GSM8K evaluation
    #[arg(long, default_value_t = 512)]
    max_gen_tokens: usize,
```

Add these branches to the `if/else` chain in `main()`:

```rust
    } else if cli.pretrain {
        run_pretrain(&cli, &device)?;
    } else if cli.sft {
        run_sft(&cli, &device)?;
    } else if cli.eval_bpb {
        run_eval_bpb(&cli, &device)?;
    }
```

Add the implementation functions:

```rust
fn run_pretrain(cli: &Cli, device: &Device) -> Result<()> {
    let data_dir = cli.data.as_ref().expect("--data is required for pretraining");
    let tok_path = cli.tokenizer.as_ref().expect("--tokenizer is required for pretraining");
    let save_dir = cli.save.as_ref().expect("--save is required for pretraining");

    let config = picochat_train::pretrain::PretrainConfig {
        data_dir: data_dir.clone(),
        val_data: cli.val_data.clone(),
        tokenizer_path: tok_path.clone(),
        total_steps: cli.steps,
        batch_size: cli.batch_size,
        seq_len: cli.seq_len,
        max_lr: cli.max_lr,
        warmup_steps: cli.warmup_steps,
        min_lr_ratio: cli.min_lr_ratio,
        eval_every: cli.eval_every,
        save_every: cli.save_every,
        save_dir: save_dir.clone(),
        depth: cli.depth,
    };

    picochat_train::pretrain::pretrain(&config, device)
}

fn run_sft(cli: &Cli, device: &Device) -> Result<()> {
    let ckpt_dir = cli.load.as_ref().expect("--load is required for SFT");
    let tok_path = cli.tokenizer.as_ref().expect("--tokenizer is required for SFT");
    let save_dir = cli.save.as_ref().expect("--save is required for SFT");
    let sft_data = cli.sft_data.as_ref().expect("--sft-data is required for SFT");

    // Parse sft_data: "path1:weight1,path2:weight2"
    let datasets: Vec<(String, f64)> = sft_data
        .split(',')
        .map(|s| {
            let parts: Vec<&str> = s.split(':').collect();
            let path = parts[0].to_string();
            let weight = if parts.len() > 1 {
                parts[1].parse::<f64>().unwrap_or(1.0)
            } else {
                1.0
            };
            (path, weight)
        })
        .collect();

    let config = picochat_train::sft::SftConfig {
        checkpoint_dir: ckpt_dir.clone(),
        tokenizer_path: tok_path.clone(),
        datasets,
        total_steps: cli.steps,
        batch_size: cli.batch_size,
        seq_len: cli.seq_len,
        max_lr: cli.max_lr,
        warmup_steps: cli.warmup_steps,
        min_lr_ratio: cli.min_lr_ratio,
        save_dir: save_dir.clone(),
        save_every: cli.save_every,
    };

    picochat_train::sft::sft(&config, device)
}

fn run_eval_bpb(cli: &Cli, device: &Device) -> Result<()> {
    let ckpt_dir = cli.load.as_ref().expect("--load is required for BPB evaluation");
    let tok_path = cli.tokenizer.as_ref().expect("--tokenizer is required for BPB evaluation");
    let val_path = cli.val_data.as_ref().expect("--val-data is required for BPB evaluation");

    let config = picochat_train::checkpoint::load_config(format!("{ckpt_dir}/config.json"))?;
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = picochat_core::model::GPT::new(&config, vb)?;
    picochat_train::checkpoint::load_varmap(&varmap, format!("{ckpt_dir}/model.safetensors"), device)?;

    let tokenizer = picochat_tokenizer::Tokenizer::load(tok_path)?;

    let result = picochat_eval::bpb::evaluate_bpb(
        &model, val_path, &tokenizer, cli.batch_size, cli.seq_len, device,
    )?;

    println!("BPB: {:.4} (tokens={}, bytes={}, avg_loss={:.4})",
        result.bpb, result.num_tokens, result.num_bytes, result.avg_loss);

    Ok(())
}
```

Update the usage message at the end of the else block:

```rust
    } else {
        println!("picochat v0.1.0");
        println!("  --smoke-test   Run forward pass verification");
        println!("  --train        Train on synthetic data");
        println!("  --pretrain     Pretrain on parquet data");
        println!("  --sft          Supervised fine-tuning");
        println!("  --eval-bpb     Evaluate BPB on validation data");
        println!("  --chat         Interactive chat mode");
    }
```

**Step 3: Build check**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo build -p picochat-cli 2>&1 | tail -20`
Expected: Compiles without errors

**Step 4: Verify help output**

Run: `PATH="/home/nullify/.cargo/bin:$PATH" cargo run -p picochat-cli -- --help 2>&1 | tail -30`
Expected: Shows all new flags

**Step 5: Commit**

```bash
git add crates/picochat-cli/src/main.rs crates/picochat-cli/Cargo.toml
git commit -m "feat: add pretrain, SFT, and evaluation CLI flags"
```

---

## Full Test Verification

After all tasks:

```bash
PATH="/home/nullify/.cargo/bin:$PATH" cargo test --workspace 2>&1 | tail -30
```

Expected: All tests pass (existing + new).

```bash
PATH="/home/nullify/.cargo/bin:$PATH" cargo build --workspace 2>&1 | tail -10
```

Expected: Clean build, no warnings.

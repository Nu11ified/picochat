# Phase 4: Pretraining, SFT, and Evaluation — Design

**Goal:** Build the complete training pipeline (real pretraining on FineWeb, supervised fine-tuning with dataset mixtures) and evaluation harnesses (BPB, MMLU, GSM8K) so picochat can train and evaluate actual language models.

## Architecture

Layered build: metrics → pretraining → BPB eval → SFT data → SFT training → MMLU eval → GSM8K eval → CLI.

### Metrics Module (`picochat-train/src/metrics.rs`)

`TrainingMetrics` tracks:
- **BPB**: `loss * log2(e) / avg_bytes_per_token` — vocab-invariant validation metric
- **Throughput**: tokens/sec from step timing
- **MFU**: `6 * num_params * tokens_per_step / (elapsed * peak_tflops)` — model FLOPS utilization

### Pretraining Loop (`picochat-train/src/pretrain.rs`)

On-the-fly tokenization pipeline:
1. Stream parquet → ParquetTextReader → tokenize → PackingDataLoader
2. Training loop: forward → loss → backward → optimizer step
3. Periodic eval (BPB on validation data) and checkpointing

Config: data_dir, val_data, tokenizer_path, total_steps, batch_size, seq_len, eval_every, save_every.

### BPB Evaluation (`picochat-eval/src/bpb.rs`)

New `picochat-eval` crate. BPB eval:
1. Tokenize + pack validation data
2. Forward-only passes, accumulate cross-entropy loss
3. `bpb = total_loss * log2(e) / total_bytes`
4. Returns `BpbResult { bpb, num_tokens, num_bytes }`

### SFT Data Pipeline (`picochat-data/src/sft.rs` + `mixture.rs`)

JSONL chat format: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

Tokenization: `<|bos|><|user_start|>...<|user_end|><|assistant_start|>...<|assistant_end|>`

Loss mask: only assistant tokens contribute to loss.

Mixture: weighted random sampling across datasets with per-dataset epoch counts.

### SFT Training Loop (`picochat-train/src/sft.rs`)

Inherits pretrained checkpoint. Uses MixtureDataLoader. Applies loss mask. Shorter horizon, lower LR.

### MMLU Eval (`picochat-eval/src/mmlu.rs`)

5-shot multiple-choice. Compare log-probs of A/B/C/D tokens. Report per-subject and overall accuracy.

### GSM8K Eval (`picochat-eval/src/gsm8k.rs`)

Few-shot chain-of-thought. Generate response, extract `####` final answer. Report solve rate.

### CLI Integration

- `--pretrain` with data paths, steps, eval frequency
- `--sft` with dataset paths, weights, epochs
- `--eval` with eval type (bpb/mmlu/gsm8k) and data paths

# picochat

A small language model training framework written in Rust. It covers the full pipeline: tokenizer training, pretraining on text corpora, supervised fine-tuning on chat data, reinforcement learning with GRPO, and inference with an interactive chat mode and web UI. The default model is a 28M parameter GPT with grouped-query attention, rotary embeddings, and sliding window masking.

The project was inspired by Karpathy's [nanochat](https://github.com/karpathy/nanochat). I believe understanding how LLMs are trained is the start of understanding how to build systems to use the full power of LLMs. For this project I used Rust because I wanted to work closer to the metal and understand every operation without framework magic hiding what's happening. Rust's type system also makes it easier to catch shape mismatches and lifetime issues at compile time rather than getting silent numerical bugs at runtime. And honestly, I just wanted to see how far you could push a from-scratch LLM implementation in a systems language.

## Requirements

- Rust toolchain (tested with stable)
- A C compiler (gcc)

## Building

```
cargo build --release
```

## Pretrained weights

A pretrained 90M parameter checkpoint is available on [Hugging Face](https://huggingface.co/manasred/picochat). Download the files and run:

```
mkdir -p runs/model
# place model.safetensors, config.json, and tokenizer.json in runs/model/

cargo run --release -- \
  --chat --load runs/model --tokenizer runs/model/tokenizer.json \
  --temperature 0.8 --max-tokens 256
```

## Usage

### Train a tokenizer

```
cargo run --release -- \
  --tokenizer-train --data data/corpus.txt \
  --vocab-size 32768 --save-tokenizer runs/tok.json
```

Training also works from parquet files with a `text` column.

### Pretrain

Pretraining expects a directory of parquet files (FineWeb format, with a `text` column). If you have a plain text file, convert it first:

```
cargo run --release -- \
  --prepare-data --data data/corpus.txt --output data/train/corpus.parquet
```

Then pretrain:

```
cargo run --release -- \
  --pretrain --data data/train --tokenizer runs/tok.json \
  --depth 4 --steps 3000 --batch-size 2 --seq-len 256 \
  --save runs/pretrain
```

### Supervised fine-tuning

SFT expects JSONL files with chat conversations:

```json
{"messages": [{"role": "user", "content": "What is gravity?"}, {"role": "assistant", "content": "Gravity is..."}]}
```

Run SFT on a pretrained checkpoint:

```
cargo run --release -- \
  --sft --load runs/pretrain --tokenizer runs/tok.json \
  --sft-data data/chat.jsonl --steps 200 --batch-size 2 --seq-len 256 \
  --save runs/sft
```

Multiple datasets with weights: `--sft-data "data/a.jsonl:1.0,data/b.jsonl:0.5"`

### GRPO (reinforcement learning)

Trains reasoning ability using group relative policy optimization with math (GSM8K), multiple choice (ARC), and tool-use tasks:

```
cargo run --release -- \
  --grpo --load runs/sft --tokenizer runs/tok.json \
  --gsm8k-data data/gsm8k.jsonl --arc-data data/arc.jsonl \
  --steps 100 --group-size 16 \
  --save runs/grpo
```

### Chat

```
cargo run --release -- \
  --chat --load runs/sft --tokenizer runs/tok.json \
  --temperature 0.8 --max-tokens 256
```

### Web UI

```
cargo run --release -- \
  --serve --load runs/sft --tokenizer runs/tok.json --port 8000
```

### Evaluation

Bits-per-byte on validation data:

```
cargo run --release -- \
  --eval-bpb --load runs/pretrain --tokenizer runs/tok.json \
  --val-data data/val
```

ARC-Challenge accuracy:

```
cargo run --release -- \
  --eval-arc --load runs/sft --tokenizer runs/tok.json \
  --arc-data data/arc.jsonl
```

## Architecture

The model is a decoder-only transformer with:

- Grouped-query attention (GQA) with half as many KV heads as query heads
- Rotary positional embeddings (RoPE)
- Sliding window attention with a configurable per-layer pattern (default: SSSL)
- ReLU-squared activation in the MLP
- RMS normalization
- Learnable residual scaling (per-layer lambdas)
- Logit soft-capping

The `--depth` flag controls model size:

| Depth | Params | Layers | Dim | Heads |
|-------|--------|--------|-----|-------|
| 4 | 28M | 4 | 256 | 4 |
| 8 | 90M | 8 | 512 | 8 |

The optimizer is a hybrid of Muon (for 2D weight matrices) and AdamW (for everything else), with per-parameter-group learning rates.

## Crates

- `picochat-core` -- model architecture, attention, embeddings, KV cache
- `picochat-tokenizer` -- BPE tokenizer with special tokens for chat, reasoning, and tool use
- `picochat-data` -- parquet readers, packing dataloaders, dataset formats
- `picochat-optim` -- Muon/AdamW hybrid optimizer, LR scheduling
- `picochat-train` -- pretraining, SFT, and GRPO training loops
- `picochat-eval` -- BPB, ARC, GSM8K, and reasoning quality metrics
- `picochat-engine` -- generation, sampling, reasoning/tool-call handling, INT8 quantization
- `picochat-serve` -- HTTP server with SSE streaming
- `picochat-tool` -- tool/expression evaluator for function calls
- `picochat-cli` -- command-line entry point

## Tests

```
cargo test --workspace
```

## Limitations

This is a learning project, not a production language model. The models trained here produce coherent text on topics seen during training but will generate garbled output on novel questions. Getting useful general-purpose chat behavior requires billions of tokens of training data and GPU-scale compute -- far beyond what a CPU-trained 28M-90M parameter model can achieve. The value of this project is in the training framework and pipeline, not the resulting model weights.

## Acknowledgments

This project draws heavily from Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat) and the broader nanoGPT lineage. His work on making LLM training understandable and accessible is what motivated this project.

## License

MIT

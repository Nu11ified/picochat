#!/bin/bash
# Full GPT-2 scale training recipe (depth=26, ~124M params)
# Expects: FineWeb parquet shards in data/fineweb/
set -euo pipefail

DEPTH=26
PRETRAIN_STEPS=19073
SFT_STEPS=2000
GRPO_STEPS=500
BATCH=32
SEQ=2048

echo "=== picochat speedrun (depth=$DEPTH) ==="

echo "[1/6] Training tokenizer..."
cargo run --release --features cuda -- \
  --tokenizer-train --data data/fineweb/train-00000.parquet \
  --vocab-size 32768 --save-tokenizer runs/tok-32k.json

echo "[2/6] Pretraining ($PRETRAIN_STEPS steps)..."
cargo run --release --features cuda -- \
  --pretrain --data data/fineweb --tokenizer runs/tok-32k.json \
  --depth $DEPTH --steps $PRETRAIN_STEPS --batch-size $BATCH --seq-len $SEQ \
  --val-data data/fineweb/val.parquet \
  --save runs/pretrain-d$DEPTH --save-every 1000 --eval-every 500 \
  --max-lr 0.002 --warmup-steps 200

echo "[3/6] Evaluating pretrained model..."
cargo run --release --features cuda -- \
  --eval-bpb --load runs/pretrain-d$DEPTH --tokenizer runs/tok-32k.json \
  --val-data data/fineweb/val.parquet --batch-size $BATCH --seq-len $SEQ

echo "[4/6] SFT ($SFT_STEPS steps)..."
cargo run --release --features cuda -- \
  --sft --load runs/pretrain-d$DEPTH --tokenizer runs/tok-32k.json \
  --sft-data data/smoltalk.jsonl:3,data/gsm8k_train.jsonl:4,data/mmlu_train.jsonl:3 \
  --steps $SFT_STEPS --batch-size 8 --seq-len $SEQ \
  --save runs/sft-d$DEPTH --save-every 500

echo "[5/6] GRPO reasoning training ($GRPO_STEPS steps)..."
cargo run --release --features cuda -- \
  --grpo --load runs/sft-d$DEPTH --tokenizer runs/tok-32k.json \
  --gsm8k-data data/gsm8k_train.jsonl \
  --arc-data data/arc_challenge.jsonl \
  --tool-data data/tool_scenarios.jsonl \
  --steps $GRPO_STEPS --group-size 16 --save runs/grpo-d$DEPTH --save-every 100

echo "[6/6] Final evaluation..."
cargo run --release --features cuda -- \
  --eval-bpb --load runs/grpo-d$DEPTH --tokenizer runs/tok-32k.json \
  --val-data data/fineweb/val.parquet --batch-size $BATCH --seq-len $SEQ

echo "=== Speedrun complete! ==="
echo "Chat: cargo run --release --features cuda -- --chat --load runs/grpo-d$DEPTH --tokenizer runs/tok-32k.json"
echo "Serve: cargo run --release --features cuda -- --serve --load runs/grpo-d$DEPTH --tokenizer runs/tok-32k.json --port 8000"

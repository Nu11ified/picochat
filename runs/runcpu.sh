#!/bin/bash
# Small CPU training example: tokenizer -> pretrain -> SFT -> chat
# Expects: data/tiny.txt (small text corpus), data/chat.jsonl (SFT data)
set -euo pipefail

DEPTH=4
STEPS=100
BATCH=2
SEQ=64

echo "=== picochat CPU training recipe (depth=$DEPTH) ==="

echo "[1/4] Training tokenizer..."
cargo run --release -- \
  --tokenizer-train --data data/tiny.txt \
  --vocab-size 4096 --save-tokenizer runs/tok.json

echo "[2/4] Pretraining..."
cargo run --release -- \
  --pretrain --data data/tiny.txt --tokenizer runs/tok.json \
  --depth $DEPTH --steps $STEPS --batch-size $BATCH --seq-len $SEQ \
  --save runs/pretrain

if [ -f data/chat.jsonl ]; then
  echo "[3/4] SFT..."
  cargo run --release -- \
    --sft --load runs/pretrain --tokenizer runs/tok.json \
    --sft-data data/chat.jsonl --steps 50 --batch-size $BATCH --seq-len $SEQ \
    --save runs/sft
  CHAT_CKPT=runs/sft
else
  echo "[3/4] Skipping SFT (no data/chat.jsonl)"
  CHAT_CKPT=runs/pretrain
fi

echo "[4/4] Starting chat..."
cargo run --release -- \
  --chat --load $CHAT_CKPT --tokenizer runs/tok.json \
  --max-tokens 128 --temperature 0.8

#!/bin/bash
# Automated SFT + evaluation when a pretrain checkpoint becomes available.
# Usage: ./run_sft_on_checkpoint.sh <checkpoint_dir> [sft_version]
#
# Waits for the checkpoint to exist, runs SFT, then evaluates.

set -e
export PATH="/usr/bin:/home/nullify/.cargo/bin:$PATH"

CKPT="${1:?Usage: $0 <checkpoint_dir> [sft_version]}"
VERSION="${2:-v3}"
PICOCHAT="/data/github/picochat/target/release/picochat"
TOK="/data/github/picochat/data/tinystories/tok.json"
SFT_DIR="/data/github/picochat/runs/tinystories-sft-${VERSION}"
# Weights balanced so each example is seen ~3x across 3 epochs regardless of dataset size.
# Dataset sizes: uncertainty=89, expanded=162, reasoning=60, common=136, multiturn=25, cot=51, graded=35, self_awareness=15
SFT_DATA="/data/github/picochat/data/tinystories/sft_uncertainty.jsonl:1.0,/data/github/picochat/data/tinystories/sft_expanded.jsonl:1.8,/data/github/picochat/data/tinystories/sft_reasoning.jsonl:1.0,/data/github/picochat/data/tinystories/sft_common.jsonl:1.5,/data/github/picochat/data/tinystories/sft_multiturn.jsonl:0.3,/data/github/picochat/data/tinystories/sft_cot.jsonl:0.9,/data/github/picochat/data/tinystories/sft_graded_uncertainty.jsonl:0.5,/data/github/picochat/data/tinystories/sft_self_awareness.jsonl:0.25"

echo "=== Waiting for checkpoint: $CKPT ==="
while [ ! -f "$CKPT/model.safetensors" ]; do
    sleep 60
done
echo "=== Checkpoint found! Starting SFT ==="

# 573 examples / batch_size 2 = 286 steps/epoch. 3 epochs avoids overfitting.
$PICOCHAT --sft --load "$CKPT" \
    --tokenizer "$TOK" \
    --sft-data "$SFT_DATA" \
    --batch-size 2 --seq-len 256 --steps 858 --max-lr 0.0003 \
    --warmup-steps 50 --min-lr-ratio 0.1 \
    --save "$SFT_DIR" --save-every 286

echo "=== SFT complete. Running evaluation ==="
/data/github/picochat/eval_chat.sh "$SFT_DIR" "$TOK"

echo "=== Pipeline complete ==="
echo "SFT model: $SFT_DIR"
echo "Based on pretrain: $CKPT"

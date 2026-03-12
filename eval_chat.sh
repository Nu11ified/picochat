#!/bin/bash
# Automated chat evaluation for picochat models.
# Tests factual Q&A, reasoning, uncertainty, and generalization.
# Usage: ./eval_chat.sh <checkpoint_dir> [tokenizer_path]

set -e

CKPT="${1:?Usage: $0 <checkpoint_dir> [tokenizer_path]}"
TOK="${2:-/data/github/picochat/data/tinystories/tok.json}"
PICOCHAT="/data/github/picochat/target/release/picochat"
TIMEOUT_SEC=30
MAX_TOKENS=128
TEMP=0.3

if [ ! -f "$CKPT/model.safetensors" ]; then
    echo "ERROR: No model.safetensors in $CKPT"
    exit 1
fi

ask() {
    local question="$1"
    local response
    response=$(echo "$question" | timeout "$TIMEOUT_SEC" "$PICOCHAT" \
        --chat --load "$CKPT" --tokenizer "$TOK" \
        --max-tokens "$MAX_TOKENS" --temperature "$TEMP" 2>/dev/null \
        | sed -n '4p')
    echo "$response"
}

PASS=0
FAIL=0
TOTAL=0

check() {
    local category="$1"
    local question="$2"
    local expected="$3"  # grep -iE pattern
    local response

    TOTAL=$((TOTAL + 1))
    response=$(ask "$question")

    if echo "$response" | grep -qiE "$expected"; then
        echo "  PASS [$category] Q: $question"
        echo "       A: $response"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$category] Q: $question"
        echo "       A: $response"
        echo "       Expected pattern: $expected"
        FAIL=$((FAIL + 1))
    fi
}

check_uncertainty() {
    local question="$1"
    local response

    TOTAL=$((TOTAL + 1))
    response=$(ask "$question")

    if echo "$response" | grep -qiE "not sure|don't know|cannot|I'm not|uncertain|do not have|difficult to say|no way to"; then
        echo "  PASS [uncertainty] Q: $question"
        echo "       A: $response"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [uncertainty] Q: $question"
        echo "       A: $response"
        echo "       Expected: uncertainty expression"
        FAIL=$((FAIL + 1))
    fi
}

check_nonempty() {
    local category="$1"
    local question="$2"
    local response

    TOTAL=$((TOTAL + 1))
    response=$(ask "$question")

    if [ ${#response} -gt 10 ]; then
        echo "  PASS [$category] Q: $question"
        echo "       A: $response"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$category] Q: $question"
        echo "       A: $response"
        echo "       Expected: non-trivial response (>10 chars)"
        FAIL=$((FAIL + 1))
    fi
}

echo "============================================"
echo "Picochat Chat Evaluation"
echo "Checkpoint: $CKPT"
echo "============================================"
echo ""

echo "--- Factual (trained) ---"
check "factual" "What is the capital of France?" "paris"
check "factual" "What is 2 + 3?" "5"
check "factual" "How many legs does a spider have?" "eight|8"
check "factual" "What is the biggest planet?" "jupiter"

echo ""
echo "--- Factual (novel, in-domain) ---"
check "novel" "What is the capital of Spain?" "madrid"
check "novel" "How many legs does a cat have?" "four|4"
check "novel" "What color is grass?" "green"
check "novel" "What sound does a cow make?" "moo"

echo ""
echo "--- Reasoning ---"
check "reasoning" "If I have 5 apples and eat 2, how many are left?" "3|three"
check "reasoning" "Is a whale a fish?" "no|mammal|not a fish"
check "reasoning" "Which is bigger, an elephant or a mouse?" "elephant"

echo ""
echo "--- Uncertainty ---"
check_uncertainty "What is the capital of Bluravia?"
check_uncertainty "Who won the Super Bowl in 2087?"
check_uncertainty "What did Einstein eat for breakfast on March 5, 1921?"

echo ""
echo "--- Generative ---"
check_nonempty "generative" "Tell me a short story about a cat."
check_nonempty "generative" "Why is the sky blue?"

echo ""
echo "============================================"
echo "Results: $PASS/$TOTAL passed ($FAIL failed)"
SCORE=$(echo "scale=1; $PASS * 100 / $TOTAL" | bc)
echo "Score: ${SCORE}%"
echo "============================================"

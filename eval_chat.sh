#!/bin/bash
# Automated chat evaluation for picochat models.
# Separates memorization (trained questions) from generalization (novel questions).
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
# Separate counters for novel questions only
NOVEL_PASS=0
NOVEL_TOTAL=0

check() {
    local category="$1"
    local question="$2"
    local expected="$3"  # grep -iE pattern
    local is_novel="${4:-0}"
    local response

    TOTAL=$((TOTAL + 1))
    response=$(ask "$question")

    if echo "$response" | grep -qiE "$expected"; then
        echo "  PASS [$category] Q: $question"
        echo "       A: $response"
        PASS=$((PASS + 1))
        [ "$is_novel" = "1" ] && NOVEL_PASS=$((NOVEL_PASS + 1))
    else
        echo "  FAIL [$category] Q: $question"
        echo "       A: $response"
        echo "       Expected pattern: $expected"
        FAIL=$((FAIL + 1))
    fi
    [ "$is_novel" = "1" ] && NOVEL_TOTAL=$((NOVEL_TOTAL + 1))
}

check_uncertainty() {
    local question="$1"
    local is_novel="${2:-0}"
    local response

    TOTAL=$((TOTAL + 1))
    response=$(ask "$question")

    if echo "$response" | grep -qiE "not sure|don't know|cannot|I'm not|uncertain|do not have|difficult to say|no way to|I think"; then
        echo "  PASS [uncertainty] Q: $question"
        echo "       A: $response"
        PASS=$((PASS + 1))
        [ "$is_novel" = "1" ] && NOVEL_PASS=$((NOVEL_PASS + 1))
    else
        echo "  FAIL [uncertainty] Q: $question"
        echo "       A: $response"
        echo "       Expected: uncertainty expression"
        FAIL=$((FAIL + 1))
    fi
    [ "$is_novel" = "1" ] && NOVEL_TOTAL=$((NOVEL_TOTAL + 1))
}

check_nonempty() {
    local category="$1"
    local question="$2"
    local is_novel="${3:-0}"
    local response

    TOTAL=$((TOTAL + 1))
    response=$(ask "$question")

    if [ ${#response} -gt 10 ]; then
        echo "  PASS [$category] Q: $question"
        echo "       A: $response"
        PASS=$((PASS + 1))
        [ "$is_novel" = "1" ] && NOVEL_PASS=$((NOVEL_PASS + 1))
    else
        echo "  FAIL [$category] Q: $question"
        echo "       A: $response"
        echo "       Expected: non-trivial response (>10 chars)"
        FAIL=$((FAIL + 1))
    fi
    [ "$is_novel" = "1" ] && NOVEL_TOTAL=$((NOVEL_TOTAL + 1))
}

echo "============================================"
echo "Picochat Chat Evaluation"
echo "Checkpoint: $CKPT"
echo "============================================"
echo ""

echo "--- Memorization (in SFT data) ---"
check "memorized" "What is the capital of France?" "paris"
check "memorized" "What is 2 + 3?" "5"
check "memorized" "How many legs does a spider have?" "eight|8"
check "memorized" "What is the biggest planet?" "jupiter"

echo ""
echo "--- Novel Factual (NOT in SFT data) ---"
check "novel-fact" "What is the capital of Italy?" "rome" 1
check "novel-fact" "How many legs does a horse have?" "four|4" 1
check "novel-fact" "What color is a lemon?" "yellow" 1
check "novel-fact" "What sound does a sheep make?" "baa" 1
check "novel-fact" "How many ears do you have?" "two|2" 1

echo ""
echo "--- Novel Reasoning (NOT in SFT data) ---"
check "novel-reason" "If I have 5 apples and eat 2, how many are left?" "3|three" 1
check "novel-reason" "Which is bigger, an elephant or a mouse?" "elephant" 1
check "novel-reason" "If it is raining, should you bring an umbrella?" "yes" 1
check "novel-reason" "Can a fish climb a tree?" "no|cannot|can not" 1
check "novel-reason" "Is ice hot or cold?" "cold" 1
check "novel-reason" "What happens if you drop a glass on the floor?" "break|shatter|crack" 1

echo ""
echo "--- Novel Uncertainty (NOT in SFT data) ---"
check_uncertainty "What is the population of the moon?" 1
check_uncertainty "Who will be president in 2090?" 1
check_uncertainty "What color is the King of Plordavia's hat?" 1

echo ""
echo "--- Novel Graded Uncertainty (NOT in SFT data) ---"
check "novel-graded" "How many teeth do sharks have?" "think|believe|not sure|not completely|around|about|many" 1
check "novel-graded" "How old is the universe?" "think|believe|not sure|billion|old" 1

echo ""
echo "--- Novel Self-Awareness (NOT in SFT data) ---"
check "novel-self" "Do you have feelings?" "no|do not|don't|not" 1
check "novel-self" "Can you see me?" "no|cannot|can not|do not" 1

echo ""
echo "--- Novel Generative (NOT in SFT data) ---"
check_nonempty "novel-gen" "Tell me about dogs." 1
check_nonempty "novel-gen" "What happens when it snows?" 1

echo ""
echo "============================================"
echo "Results: $PASS/$TOTAL passed ($FAIL failed)"
SCORE=$(echo "scale=1; $PASS * 100 / $TOTAL" | bc)
echo "Overall Score: ${SCORE}%"
echo ""
if [ $NOVEL_TOTAL -gt 0 ]; then
    NOVEL_SCORE=$(echo "scale=1; $NOVEL_PASS * 100 / $NOVEL_TOTAL" | bc)
    echo "GENERALIZATION Score: $NOVEL_PASS/$NOVEL_TOTAL = ${NOVEL_SCORE}%"
    echo "(Novel questions NOT in training data)"
fi
echo "============================================"

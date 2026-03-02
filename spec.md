# Picochat Spec

> A Rust reimplementation of [nanochat](https://github.com/karpathy/nanochat) with reasoning and tool-use capabilities.
> Train your own reasoning-capable ChatGPT for <$100 — in Rust.

## Overview

Picochat is a from-scratch Rust implementation of the complete LLM pipeline: tokenization, pretraining, supervised fine-tuning, reasoning training (GRPO), evaluation, inference, and a chat UI. It mirrors nanochat's validated GPT architecture and extends it with DeepSeek-R1-inspired reasoning capabilities and structured tool use.

### Goals

1. **Full pipeline parity with nanochat** — tokenizer training, pretraining, SFT, eval, inference, web UI
2. **Reasoning capabilities** — DeepSeek-R1-style GRPO training with chain-of-thought reasoning
3. **Tool-use framework** — structured function calling beyond nanochat's basic calculator
4. **Performance superiority** — faster training throughput, faster inference, lower memory usage
5. **Multi-platform** — NVIDIA GPU (CUDA), Apple M-series (Metal), Intel/AMD CPU

### Non-Goals

- Not a general-purpose ML framework
- Not targeting models larger than ~2B parameters
- Not supporting multi-node distributed training (single node only, like nanochat)

---

## Architecture

### Compute Backend: candle

All tensor operations use [candle](https://github.com/huggingface/candle), Hugging Face's Rust ML framework.

Backend selection via Cargo feature flags:

| Feature  | Backend       | Target Hardware       | Precision    |
|----------|---------------|-----------------------|--------------|
| `cuda`   | candle-cuda   | NVIDIA GPU (sm70+)    | BF16 / FP32  |
| `metal`  | candle-metal  | Apple M-series        | FP32 / F16   |
| `cpu`    | candle-cpu    | Intel/AMD (AVX2/512)  | FP32         |

Default: auto-detect at runtime. Binary includes all compiled backends.

### Single Complexity Dial: `--depth`

Like nanochat, picochat uses a single `--depth` parameter (number of transformer layers) that automatically determines all other hyperparameters:

```
depth → n_embd, n_head, n_kv_head, learning rates, training horizon, weight decay, batch size
```

This ensures compute-optimal models at every scale. GPT-2 capability ≈ depth 24–26.

---

## Project Structure

```
picochat/
├── Cargo.toml                      # workspace root
├── spec.md                         # this file
│
├── crates/
│   ├── picochat-core/              # model architecture, tensor ops, config
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── config.rs           # GPTConfig, depth → hyperparameter scaling
│   │   │   ├── model.rs            # GPT model: embedding, blocks, lm_head
│   │   │   ├── attention.rs        # CausalSelfAttention with GQA, RoPE, sliding window
│   │   │   ├── mlp.rs              # MLP with ReLU^2 activation
│   │   │   ├── norm.rs             # RMSNorm (no learnable params)
│   │   │   ├── rotary.rs           # Rotary positional embeddings
│   │   │   └── flash_attention.rs  # Flash attention dispatch (CUDA kernel / SDPA fallback)
│   │   └── Cargo.toml
│   │
│   ├── picochat-tokenizer/         # BPE tokenizer
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── bpe.rs              # BPE training (byte-level, GPT-4-style)
│   │   │   ├── encode.rs           # fast encoding (regex split + BPE merge)
│   │   │   └── special.rs          # special token registry
│   │   └── Cargo.toml
│   │
│   ├── picochat-data/              # data loading pipeline
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── parquet.rs          # parquet file streaming (arrow-rs)
│   │   │   ├── dataloader.rs       # BOS-aligned best-fit batching
│   │   │   ├── dataset.rs          # FineWeb/SmolTalk download + sharding
│   │   │   └── mixture.rs          # task mixture for SFT (MMLU, GSM8K, etc.)
│   │   └── Cargo.toml
│   │
│   ├── picochat-optim/             # optimizers
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── adamw.rs            # fused AdamW
│   │   │   ├── muon.rs             # Muon with Polar Express orthogonalization
│   │   │   ├── combined.rs         # MuonAdamW combined optimizer
│   │   │   ├── distributed.rs      # DistMuonAdamW (multi-GPU, ZeRO-2-style)
│   │   │   └── schedule.rs         # LR scheduling (warmup, cosine decay, warmdown)
│   │   └── Cargo.toml
│   │
│   ├── picochat-train/             # training loops
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── pretrain.rs         # base pretraining loop
│   │   │   ├── sft.rs              # supervised fine-tuning loop
│   │   │   ├── grpo.rs             # GRPO reasoning training (DeepSeek-R1-style)
│   │   │   ├── rewards.rs          # rule-based reward functions for GRPO
│   │   │   ├── checkpoint.rs       # save/load model checkpoints
│   │   │   └── metrics.rs          # BPB, MFU, throughput tracking
│   │   └── Cargo.toml
│   │
│   ├── picochat-eval/              # evaluation suite
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── core.rs             # DCLM CORE score
│   │   │   ├── mmlu.rs             # MMLU benchmark
│   │   │   ├── gsm8k.rs            # GSM8K math benchmark
│   │   │   ├── arc.rs              # ARC science questions
│   │   │   ├── humaneval.rs        # code generation benchmark
│   │   │   └── reasoning.rs        # reasoning quality metrics (chain coherence, self-correction)
│   │   └── Cargo.toml
│   │
│   ├── picochat-engine/            # inference engine
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── kv_cache.rs         # pre-allocated KV cache with sliding window
│   │   │   ├── sampling.rs         # temperature, top-k, top-p sampling
│   │   │   ├── generate.rs         # token generation loop (streaming)
│   │   │   ├── tools.rs            # safe tool execution (AST-parsed, sandboxed)
│   │   │   ├── reasoning.rs        # reasoning mode (<think> token handling)
│   │   │   └── quantize.rs         # INT8/INT4 weight quantization for inference
│   │   └── Cargo.toml
│   │
│   └── picochat-cli/               # binary entry point
│       ├── src/
│       │   ├── main.rs             # CLI dispatcher
│       │   ├── train.rs            # `picochat train` subcommand
│       │   ├── chat.rs             # `picochat chat` subcommand (CLI)
│       │   ├── eval.rs             # `picochat eval` subcommand
│       │   ├── serve.rs            # `picochat serve` subcommand (web UI)
│       │   └── tokenizer.rs        # `picochat tokenizer` subcommand
│       └── Cargo.toml
│
├── web/
│   └── ui.html                     # ChatGPT-like web UI with reasoning display
│
├── tasks/                          # benchmark task data and configs
│
└── runs/
    ├── speedrun.sh                 # full GPT-2 training recipe
    └── runcpu.sh                   # small CPU/MPS training example
```

---

## Model Architecture (picochat-core)

### GPTConfig

```rust
pub struct GPTConfig {
    pub sequence_len: usize,      // 2048 default
    pub vocab_size: usize,        // 32768
    pub n_layer: usize,           // = depth (the single dial)
    pub n_head: usize,            // query heads (auto from depth)
    pub n_kv_head: usize,         // key/value heads for GQA (auto from depth)
    pub n_embd: usize,            // embedding dim (auto from depth)
    pub window_pattern: String,   // "SSSL" - sliding window pattern
}
```

All fields except `n_layer` (depth) are computed automatically via scaling laws.

### Transformer Architecture

Matching nanochat exactly:

| Component | Implementation |
|-----------|---------------|
| Position encoding | Rotary (RoPE), no learned positions, base theta 10000 |
| Attention | Grouped-Query Attention (GQA), configurable n_kv_head |
| QK normalization | RMSNorm on Q and K after projection |
| Sliding window | Per-layer window sizes from pattern string, final layer always full |
| Value residuals | ResFormer-style with gated blending (alternating layers) |
| Per-layer scalars | `resid_lambdas` (init 1.0) and `x0_lambdas` (init 0.1) |
| Normalization | RMSNorm with no learnable parameters |
| MLP activation | ReLU squared (relu then square) |
| MLP expansion | 4x embedding dimension |
| Biases | None in any linear layer |
| Embedding/LM head | Untied weights |
| Post-embedding | RMSNorm after token embedding |
| Logit capping | Softcap at 15: `15 * tanh(logits / 15)` |
| Vocab padding | Pad to multiple of 64 for tensor core alignment |

### Flash Attention Strategy

```
NVIDIA Hopper (sm90)  -> Flash Attention 3 (via CUDA kernel)
NVIDIA Ampere (sm80)  -> Flash Attention 2 (via CUDA kernel)
Apple Metal           -> Custom Metal kernel or SDPA fallback
CPU                   -> Manual attention with SIMD (AVX2/NEON)
```

The flash attention module provides a unified interface:
- `flash_attn_func(q, k, v, causal, window_size)` — training
- `flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens, ...)` — inference

### Weight Initialization

Matching nanochat:

| Parameter | Init |
|-----------|------|
| `wte` (embedding) | Normal(0, 1.0) |
| `lm_head` | Normal(0, 0.001) |
| `attn.c_q/c_k/c_v` | Uniform(-s, s) where s = sqrt(3) / sqrt(n_embd) |
| `attn.c_proj` | Zeros |
| `mlp.c_fc` | Uniform(-s, s) |
| `mlp.c_proj` | Zeros |
| `resid_lambdas` | Fill(1.0) |
| `x0_lambdas` | Fill(0.1) |
| Value embeddings | Uniform(-s, s) |
| VE gate weights | Zeros (sigmoid(0)=0.5, x2 = 1.0 neutral) |

---

## Tokenizer (picochat-tokenizer)

### BPE Tokenizer

GPT-4-style byte-level BPE with the split pattern:
```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

Note: `\p{N}{1,2}` (not `{1,3}`) — optimized for 32K vocab size.

### Special Tokens

```
<|bos|>                  # document boundary
<|user_start|>           # user message start
<|user_end|>             # user message end
<|assistant_start|>      # assistant message start
<|assistant_end|>        # assistant message end
<|python_start|>         # python tool invocation
<|python_end|>           # python tool end
<|output_start|>         # tool output start
<|output_end|>           # tool output end
<|think_start|>          # reasoning chain start (NEW)
<|think_end|>            # reasoning chain end (NEW)
<|tool_call_start|>      # structured tool call (NEW)
<|tool_call_end|>        # structured tool call end (NEW)
<|tool_result_start|>    # tool result (NEW)
<|tool_result_end|>      # tool result end (NEW)
```

### Implementation

Training: Pure Rust BPE trainer (byte-level with byte fallback).
Inference: Optimized trie-based encoder for fast tokenization.

---

## Optimizers (picochat-optim)

### Muon + AdamW Combined Optimizer

Matching nanochat's hybrid approach:

**AdamW** (for embeddings, scalars, 1D params):
- Fused step: weight_decay -> momentum -> bias_correction -> update
- LR scaling: `lr * (n_embd / 768)^{-0.5}` (dimension-aware)

**Muon** (for 2D matrix params):
- Nesterov momentum
- Polar Express orthogonalization (5 iterations, replaces Newton-Schulz)
- NorMuon variance reduction (per-neuron adaptive scaling)
- Cautious weight decay

Coefficients for Polar Express (from [Amsel et al. 2025](https://arxiv.org/pdf/2505.16932)):
```
(8.1566, -22.4833, 15.8788)
(4.0429,  -2.8089,  0.5000)
(3.8917,  -2.7725,  0.5061)
(3.2858,  -2.3681,  0.4645)
(2.3465,  -1.7098,  0.4232)
```

### Parameter Groups

| Group | Optimizer | LR (base) | Notes |
|-------|-----------|-----------|-------|
| `lm_head` | AdamW | 0.004 x scale | Unembedding |
| `wte` | AdamW | 0.2 x scale | Token embedding |
| Value embeds | AdamW | 0.2 x scale | ResFormer value embeddings |
| `resid_lambdas` | AdamW | 0.005 | Per-layer residual scalars |
| `x0_lambdas` | AdamW | 0.5 | Per-layer skip scalars (beta1=0.96) |
| Matrix params | Muon | 0.02 | All 2D weights in transformer blocks |

### Distributed Optimizer

Multi-GPU: DistMuonAdamW with 3-phase async communication:
1. **Launch**: Async reduce_scatter/all_reduce for all groups
2. **Compute**: Wait per-group, compute update, launch all_gather
3. **Finish**: Wait for gathers, copy back

ZeRO-2-style optimizer state sharding for large AdamW params. Muon params sharded by chunk across ranks.

---

## Training Pipeline (picochat-train)

### Phase 1: Pretraining

**Data**: FineWeb (HuggingFace) — parquet files streamed and tokenized on-the-fly.

**Dataloader**: BOS-aligned best-fit packing:
- Every sequence starts with `<|bos|>`
- Documents packed via best-fit algorithm to minimize cropping
- ~35% tokens cropped at T=2048, but every token has full context to BOS
- 100% utilization (no padding)

**Training loop**:
```
for step in 0..num_iterations:
    batch = dataloader.next()           // (B, T) token ids
    loss = model.forward(batch, targets) // cross-entropy
    loss.backward()
    optimizer.step()                     // Muon + AdamW

    if step % eval_every == 0:
        evaluate(val_bpb, core_score)
```

**LR schedule**: Warmup -> constant -> cosine warmdown
- Warmup: 0 -> base_lr over first N% steps
- Warmdown: base_lr -> 0 over last M% steps

**Metrics tracked**: val_bpb, CORE score, MFU, tokens/sec, peak VRAM

### Phase 2: Supervised Fine-Tuning (SFT)

**Data mixture** (matching nanochat + extensions):
- SmolTalk conversational data (multi-epoch)
- MMLU (multiple choice, ~3 epochs)
- GSM8K (math + tool use, ~4 epochs)
- **New**: Reasoning chain data with `<|think_start|>`/`<|think_end|>` markers
- **New**: Structured tool-call examples

**SFT training**: Same optimizer, inherits pretrained checkpoint. Shorter horizon.

### Phase 3: GRPO Reasoning Training (NEW)

Inspired by [DeepSeek-R1](https://arxiv.org/abs/2501.12948). This is the key differentiator from nanochat.

**Algorithm: Group Relative Policy Optimization (GRPO)**

GRPO optimizes the policy without a critic model. For each prompt:

1. **Sample**: Generate G completions (group size, e.g. G=8) from current policy
2. **Score**: Each completion scored with rule-based rewards
3. **Normalize**: Compute advantage as relative score within the group:
   ```
   advantage_i = (reward_i - mean(rewards)) / std(rewards)
   ```
4. **Update**: Policy gradient weighted by advantages, with KL penalty to reference model:
   ```
   L = -E[ min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage) ] + beta * KL(pi || pi_ref)
   ```

**Reward functions** (rule-based, no learned reward model):

| Reward | Signal | Weight |
|--------|--------|--------|
| Accuracy | Final answer matches ground truth | 1.0 |
| Format | Correct use of `<think>`/`</think>` markers | 0.2 |
| Tool use | Correct tool invocation and result integration | 0.3 |
| Length penalty | Discourage unnecessarily long chains | -0.1 |

**Training data for GRPO**:
- GSM8K (math reasoning — verifiable answers)
- ARC (science reasoning — multiple choice verification)
- Custom tool-use scenarios (arithmetic, string operations)

**Expected emergent behaviors** (per DeepSeek-R1 findings):
- Extended reasoning chains for harder problems
- Self-correction ("wait, let me reconsider...")
- Re-evaluation when early approach seems wrong
- Appropriate tool invocation for computation

### Phase 4: Evaluation

| Benchmark | Type | Metric |
|-----------|------|--------|
| CORE | Base model quality | DCLM CORE score (target: >0.2565) |
| BPB | Validation loss | Bits per byte (vocab-invariant) |
| MMLU | Knowledge | Accuracy across topics |
| GSM8K | Math reasoning | Solve rate with chain-of-thought |
| ARC | Science | Multiple choice accuracy |
| HumanEval | Coding | pass@1 |
| Reasoning | Chain quality | Coherence, self-correction rate (NEW) |

---

## Inference Engine (picochat-engine)

### KV Cache

Pre-allocated cache tensors per layer:
```rust
pub struct KVCache {
    k_cache: Tensor,    // (n_layers, B, T_max, n_kv_head, head_dim)
    v_cache: Tensor,    // (n_layers, B, T_max, n_kv_head, head_dim)
    cache_seqlens: Vec<usize>,  // current position per batch element
}
```

Operations:
- `get_layer_cache(layer_idx)` -> (k, v) views
- `advance(num_tokens)` — move position forward
- `prefill(other)` — copy from single-batch prefill to multi-sample decode
- `reset()` — clear for new conversation

### Generation Loop

```
prefill(prompt_tokens)  ->  KV cache populated
loop:
    logits = model.forward(next_token, kv_cache)
    token = sample(logits, temperature, top_k)

    if token == <|think_start|>:
        enter reasoning mode (stream thinking separately)
    if token == <|tool_call_start|>:
        collect tool call tokens until <|tool_call_end|>
        execute tool safely via AST-parsed evaluator
        inject <|tool_result_start|> result <|tool_result_end|>
    if token == <|assistant_end|>:
        break

    yield token
```

### Tool Execution (Safe)

Picochat uses an AST-parsed evaluator instead of arbitrary code execution:

```rust
pub fn execute_tool(expr: &str) -> Option<String> {
    // Parse expression into a safe AST (no arbitrary code execution)
    // Supported operations: arithmetic, string.count(), basic math functions
    // Hard timeout: 100ms max
    // No file, network, or system access — fully sandboxed
}
```

This is a deliberate security improvement over nanochat's Python-based approach. The AST
parser validates that only whitelisted operations are performed before any computation.

### Quantized Inference (NEW)

For fast CPU inference on consumer hardware:

| Format | Bits | Use Case |
|--------|------|----------|
| FP32 | 32 | CPU training, reference |
| BF16 | 16 | GPU training + inference |
| INT8 | 8 | Fast CPU inference, ~1% quality loss |
| INT4 | 4 | Ultra-fast CPU inference, ~3-5% quality loss |

Quantization applied post-training to weight matrices. KV cache stays in FP32/BF16.

### Reasoning Mode

When the model generates `<|think_start|>`:
- Thinking tokens are streamed to a separate channel
- UI displays them in a collapsible "Thinking..." section
- When `<|think_end|>` is generated, switch back to response mode
- Total token budget for thinking can be configured

---

## Web UI

Single-file HTML/CSS/JS (like nanochat's `ui.html`), served by `picochat serve`.

Features:
- ChatGPT-like message interface
- WebSocket streaming for real-time token output
- **New**: Collapsible "Thinking" sections showing reasoning chains
- **New**: Tool call display (shows what tool was called and its result)
- Conversation history (local storage)
- Model info display (depth, params, training metrics)

---

## CLI Interface

```bash
# Tokenizer
picochat tokenizer train --vocab-size 32768 --data fineweb

# Pretraining
picochat train pretrain --depth 26                              # single GPU
picochat train pretrain --depth 26 --gpus 8                     # multi-GPU

# Supervised fine-tuning
picochat train sft --model-tag base --depth 26

# GRPO reasoning training
picochat train grpo --model-tag sft --depth 26 --group-size 8

# Evaluation
picochat eval --model-tag grpo --benchmarks core,mmlu,gsm8k

# Chat (CLI)
picochat chat --model-tag grpo

# Web UI
picochat serve --model-tag grpo --port 8000

# Quick CPU demo
picochat train pretrain --depth 4 --device cpu --max-seq-len 512
```

---

## Performance Targets

### vs. nanochat (same hardware: 8xH100)

| Metric | nanochat | picochat target | How |
|--------|----------|-----------------|-----|
| Time to GPT-2 | 2.76 hrs | <2.5 hrs | Zero-copy tensors, fused kernels, no Python overhead |
| Inference tok/s (GPU) | ~100 | >200 | KV cache optimization, quantized attention |
| Inference tok/s (CPU) | ~5 | >20 | INT8 quantization, SIMD, memory-mapped weights |
| Peak VRAM (d26) | ~60 GB | <50 GB | Gradient checkpointing, no Python memory overhead |
| Binary size | N/A (Python) | <50 MB | Single static binary, no runtime dependencies |

### Reasoning (new capability)

| Metric | nanochat | picochat target |
|--------|----------|-----------------|
| GSM8K (with CoT) | Basic SFT | GRPO-trained reasoning chains |
| Tool use | Calculator only | Structured multi-tool framework |
| Self-correction | None | Emergent via GRPO |

---

## Dependencies (Key Crates)

| Crate | Purpose |
|-------|---------|
| `candle-core` | Tensor operations |
| `candle-nn` | Neural network layers |
| `candle-transformers` | Transformer building blocks |
| `candle-flash-attn` | Flash attention CUDA kernels |
| `tokenizers` | HuggingFace tokenizers (Rust core) |
| `arrow-rs` / `parquet` | Parquet file reading |
| `clap` | CLI argument parsing |
| `tokio` | Async runtime (web server) |
| `axum` | HTTP/WebSocket server |
| `serde` / `serde_json` | Serialization |
| `rayon` | CPU parallelism (data loading) |
| `nccl-rs` or custom | Multi-GPU communication |
| `indicatif` | Progress bars |
| `tracing` | Structured logging |

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-3)
- [ ] Cargo workspace setup with all crates
- [ ] GPTConfig with depth -> hyperparameter scaling
- [ ] GPT model forward pass (CPU, no training)
- [ ] BPE tokenizer (train + encode/decode)
- [ ] Basic inference: load weights, generate text

### Phase 2: Training (Weeks 4-7)
- [ ] Muon + AdamW optimizer
- [ ] Pretraining loop with BOS-aligned dataloader
- [ ] Checkpoint save/load
- [ ] BPB evaluation
- [ ] SFT training loop
- [ ] CUDA backend integration

### Phase 3: Reasoning (Weeks 8-10)
- [ ] GRPO training loop
- [ ] Reward functions (accuracy, format, tool use)
- [ ] Reasoning special tokens and data pipeline
- [ ] Multi-sample generation for GRPO

### Phase 4: Inference and UI (Weeks 11-12)
- [ ] KV cache with sliding window
- [ ] Streaming generation with tool execution
- [ ] Reasoning mode (think token handling)
- [ ] INT8/INT4 quantization
- [ ] Web UI with reasoning display
- [ ] CLI interface

### Phase 5: Optimization and Eval (Weeks 13-15)
- [ ] Multi-GPU distributed training
- [ ] Flash attention integration (CUDA/Metal)
- [ ] Full eval suite (CORE, MMLU, GSM8K, ARC, HumanEval)
- [ ] Performance benchmarking vs nanochat
- [ ] Memory-mapped model loading

---

## References

- [nanochat](https://github.com/karpathy/nanochat) — Karpathy's original Python implementation
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO reasoning training
- [Polar Express](https://arxiv.org/pdf/2505.16932) — orthogonalization for Muon optimizer
- [NorMuon](https://arxiv.org/pdf/2510.05491) — variance reduction for Muon
- [candle](https://github.com/huggingface/candle) — Rust ML framework
- [DCLM](https://www.datacomp.ai/dclm/) — CORE evaluation metric
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — pretraining speedrun techniques

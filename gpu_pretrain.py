# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "safetensors",
#     "pyarrow",
#     "regex",
#     "packaging",
# ]
# ///
"""Pretraining for picochat using PyTorch.

Reimplements the picochat model architecture in PyTorch and trains on
OpenWebText parquet data. Exports weights in safetensors format compatible
with the Rust candle inference code.

Usage (CPU):
    uv run gpu_pretrain.py

Usage (CUDA GPU):
    uv run gpu_pretrain.py
"""

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
import regex
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Tokenizer (reimplements picochat BPE in Python)
# ---------------------------------------------------------------------------

GPT4_SPLIT_PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

class PicoChatTokenizer:
    """Reimplements the Rust picochat BPE tokenizer for identical encoding."""

    def __init__(self, path: str):
        with open(path) as f:
            model = json.load(f)
        self.vocab_size = model["vocab_size"]
        self.merges = [(a, b) for a, b in model["merges"]]
        self.merge_map = {}
        for i, (a, b) in enumerate(self.merges):
            self.merge_map[(a, b)] = 256 + i
        self.pattern = regex.compile(GPT4_SPLIT_PATTERN)

    def encode(self, text: str) -> list[int]:
        tokens = []
        for match in self.pattern.finditer(text):
            chunk = match.group().encode("utf-8")
            ids = list(chunk)
            if len(ids) < 2:
                tokens.extend(ids)
                continue
            while len(ids) >= 2:
                best_pair = None
                best_new_id = float("inf")
                for j in range(len(ids) - 1):
                    pair = (ids[j], ids[j + 1])
                    new_id = self.merge_map.get(pair)
                    if new_id is not None and new_id < best_new_id:
                        best_new_id = new_id
                        best_pair = pair
                if best_pair is None:
                    break
                new_ids = []
                j = 0
                while j < len(ids):
                    if j + 1 < len(ids) and ids[j] == best_pair[0] and ids[j + 1] == best_pair[1]:
                        new_ids.append(best_new_id)
                        j += 2
                    else:
                        new_ids.append(ids[j])
                        j += 1
                ids = new_ids
            tokens.extend(ids)
        return tokens


# ---------------------------------------------------------------------------
# Model architecture (matches Rust picochat-core exactly)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len, base=10000.0):
        super().__init__()
        half_dim = head_dim // 2
        inv_freq = torch.tensor(
            [1.0 / (base ** (i / head_dim)) for i in range(half_dim)]
        )
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(2))
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(2))

    def apply(self, x, offset=0):
        t = x.shape[1]
        cos = self.cos_cached[:, offset:offset + t]
        sin = self.sin_cached[:, offset:offset + t]
        half_d = x.shape[-1] // 2
        x1 = x[..., :half_d]
        x2 = x[..., half_d:]
        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos
        return torch.cat([y1, y2], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, head_dim, has_ve, window_size):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.n_embd = n_embd
        self.window_size = window_size

        self.c_q = nn.Linear(n_embd, n_head * head_dim, bias=False)
        self.c_k = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_v = nn.Linear(n_embd, n_kv_head * head_dim, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)

        if has_ve:
            self.ve_gate = nn.Linear(32, n_kv_head, bias=False)
        else:
            self.ve_gate = None

        self.qk_norm = RMSNorm()

    def forward(self, x, ve=None, rope=None):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve_reshaped = ve.view(B, T, self.n_kv_head, self.head_dim)
            x_prefix = x[:, :, :32]
            gate = self.ve_gate(x_prefix)
            gate = torch.sigmoid(gate) * 2.0
            gate = gate.unsqueeze(3)
            v = v + gate * ve_reshaped

        q = rope.apply(q)
        k = rope.apply(k)
        q = self.qk_norm(q)
        k = self.qk_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        repeat_factor = self.n_head // self.n_kv_head
        if repeat_factor > 1:
            k = k.unsqueeze(2).expand(B, self.n_kv_head, repeat_factor, T, self.head_dim)
            k = k.reshape(B, self.n_head, T, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_head, repeat_factor, T, self.head_dim)
            v = v.reshape(B, self.n_head, T, self.head_dim)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        if self.window_size < T:
            window_mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=-self.window_size - 1).bool()
            causal_mask = causal_mask | window_mask
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).pow(2)
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, head_dim, has_ve, window_size):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, head_dim, has_ve, window_size)
        self.mlp = MLP(n_embd)
        self.norm = RMSNorm()

    def forward(self, x, ve=None, rope=None):
        x = x + self.attn(self.norm(x), ve=ve, rope=rope)
        x = x + self.mlp(self.norm(x))
        return x


class PicoChatGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_embd = config["n_embd"]
        n_head = config["n_head"]
        n_kv_head = config["n_kv_head"]
        n_layer = config["n_layer"]
        head_dim = n_embd // n_head
        kv_dim = n_kv_head * head_dim
        padded_vocab = config["padded_vocab"]
        seq_len = config["sequence_len"]

        self.wte = nn.Embedding(padded_vocab, n_embd)
        self.lm_head = nn.Linear(n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer))

        window_pattern = "SSSL"
        long_window = seq_len
        short_window = seq_len // 2
        self.blocks = nn.ModuleList()
        self.value_embeds = nn.ModuleDict()

        for i in range(n_layer):
            pattern_char = window_pattern[i % len(window_pattern)]
            if i == n_layer - 1:
                ws = long_window
            elif pattern_char in ("S", "s"):
                ws = short_window
            else:
                ws = long_window

            has_ve = self._has_value_embedding(i, n_layer)
            self.blocks.append(Block(n_embd, n_head, n_kv_head, head_dim, has_ve, ws))
            if has_ve:
                self.value_embeds[str(i)] = nn.Embedding(padded_vocab, kv_dim)

        self.rope = RotaryEmbedding(head_dim, seq_len * 10)
        self.norm = RMSNorm()
        self.embed_norm = RMSNorm()

    @staticmethod
    def _has_value_embedding(layer_idx, n_layer):
        return layer_idx % 2 == (n_layer - 1) % 2

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx)
        x = self.embed_norm(x)
        x0 = x.clone()

        for i, block in enumerate(self.blocks):
            rl = self.resid_lambdas[i].unsqueeze(0).unsqueeze(0)
            xl = self.x0_lambdas[i].unsqueeze(0).unsqueeze(0)
            x = rl * x + xl * x0

            ve = None
            if str(i) in self.value_embeds:
                ve = self.value_embeds[str(i)](idx)

            x = block(x, ve=ve, rope=self.rope)

        x = self.norm(x)
        logits = self.lm_head(x)

        vocab_size = self.config["vocab_size"]
        logits = logits[:, :, :vocab_size].float()
        cap = 15.0
        logits = cap * torch.tanh(logits / cap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            return logits, loss
        return logits

    def export_safetensors(self, path: str):
        tensors = {}
        tensors["wte.weight"] = self.wte.weight.data.cpu()
        tensors["lm_head.weight"] = self.lm_head.weight.data.cpu()
        tensors["resid_lambdas"] = self.resid_lambdas.data.cpu()
        tensors["x0_lambdas"] = self.x0_lambdas.data.cpu()

        for i, block in enumerate(self.blocks):
            prefix = f"h.{i}"
            tensors[f"{prefix}.attn.c_q.weight"] = block.attn.c_q.weight.data.cpu()
            tensors[f"{prefix}.attn.c_k.weight"] = block.attn.c_k.weight.data.cpu()
            tensors[f"{prefix}.attn.c_v.weight"] = block.attn.c_v.weight.data.cpu()
            tensors[f"{prefix}.attn.c_proj.weight"] = block.attn.c_proj.weight.data.cpu()
            if block.attn.ve_gate is not None:
                tensors[f"{prefix}.attn.ve_gate.weight"] = block.attn.ve_gate.weight.data.cpu()
            tensors[f"{prefix}.mlp.c_fc.weight"] = block.mlp.c_fc.weight.data.cpu()
            tensors[f"{prefix}.mlp.c_proj.weight"] = block.mlp.c_proj.weight.data.cpu()

        for key, emb in self.value_embeds.items():
            tensors[f"ve.{key}.weight"] = emb.weight.data.cpu()

        save_file(tensors, path)
        print(f"Saved {len(tensors)} tensors to {path}")


# ---------------------------------------------------------------------------
# Dataset with pre-tokenization for speed
# ---------------------------------------------------------------------------

class PreTokenizedDataset:
    """Pre-tokenizes all documents, then serves random batches from a flat token array."""

    def __init__(self, parquet_path: str, tokenizer: PicoChatTokenizer, seq_len: int, max_docs: int = 0):
        print(f"Loading {parquet_path}...")
        table = pq.read_table(parquet_path, columns=["text"])
        texts = table.column("text").to_pylist()
        if max_docs > 0:
            texts = texts[:max_docs]
        print(f"Loaded {len(texts)} documents, tokenizing...")

        t0 = time.time()
        all_tokens = []
        for i, text in enumerate(texts):
            if text:
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
            if (i + 1) % 5000 == 0:
                print(f"  tokenized {i+1}/{len(texts)} docs ({len(all_tokens):,} tokens)")

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.seq_len = seq_len
        elapsed = time.time() - t0
        print(f"Tokenized {len(all_tokens):,} tokens in {elapsed:.1f}s ({len(all_tokens)/elapsed:,.0f} tok/s)")

    def get_batch(self, batch_size: int, device: torch.device):
        max_start = len(self.tokens) - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([self.tokens[s:s + self.seq_len] for s in starts])
        y = torch.stack([self.tokens[s + 1:s + self.seq_len + 1] for s in starts])
        return x.to(device), y.to(device)

    @property
    def total_tokens(self):
        return len(self.tokens)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_checkpoint(model, checkpoint_path):
    """Load model weights from a safetensors checkpoint."""
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path)
    new_state_dict = {}
    for key, tensor in state_dict.items():
        if key in ("resid_lambdas", "x0_lambdas", "wte.weight", "lm_head.weight"):
            new_state_dict[key] = tensor
        elif key.startswith("h."):
            parts = key.split(".", 2)
            new_state_dict[f"blocks.{parts[1]}.{parts[2]}"] = tensor
        elif key.startswith("ve."):
            parts = key.split(".", 2)
            new_state_dict[f"value_embeds.{parts[1]}.weight"] = tensor
    # strict=False: RoPE cos/sin buffers are computed, not saved in checkpoint
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")


def main():
    import sys
    # Parse --resume <step> from command line
    resume_step = 0
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        resume_step = int(sys.argv[idx + 1])

    # Config matching depth=8
    depth = 8
    n_embd = ((64 * depth + 127) // 128) * 128  # 512
    head_dim = 64
    n_head = n_embd // head_dim  # 8
    n_kv_head = max(n_head // 2, 1)  # 4
    vocab_size = 4096
    padded_vocab = ((vocab_size + 63) // 64) * 64  # 4096

    config = {
        "n_embd": n_embd,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_layer": depth,
        "vocab_size": vocab_size,
        "padded_vocab": padded_vocab,
        "sequence_len": 2048,
        "head_dim": head_dim,
    }

    # Training hyperparameters — tuned for CPU on Ryzen 7 3700X
    batch_size = 4
    seq_len = 256
    max_steps = 30000
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 500
    weight_decay = 0.1
    grad_clip = 1.0
    log_interval = 10
    save_interval = 5000

    tokenizer_path = "runs/tok-owt.json"
    data_path = "data/openwebtext/train.parquet"
    output_dir = "runs/gpu-pretrain"

    os.makedirs(output_dir, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        print(f"Using CPU with {num_threads} threads")

    # Build model
    print(f"Building model: depth={depth}, n_embd={n_embd}, vocab={vocab_size}")
    model = PicoChatGPT(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Resume from checkpoint if requested
    if resume_step > 0:
        ckpt = os.path.join(output_dir, f"model_step{resume_step}.safetensors")
        load_checkpoint(model, ckpt)

    # Optimizer (no fused on CPU)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), fused=use_fused)

    def get_lr(step):
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    # Dataset — pre-tokenize everything for speed
    tokenizer = PicoChatTokenizer(tokenizer_path)
    # Use 40k docs (~60M tokens) — more than enough for 30k steps (30M tokens)
    # and keeps tokenization startup under 4 minutes
    dataset = PreTokenizedDataset(data_path, tokenizer, seq_len, max_docs=40000)

    # Training loop
    tokens_per_step = batch_size * seq_len
    total_tokens_target = max_steps * tokens_per_step
    print(f"\nStarting training: {max_steps} steps, batch_size={batch_size}, seq_len={seq_len}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"Total tokens: ~{total_tokens_target:,}")
    print(f"Dataset: {dataset.total_tokens:,} tokens (will see each ~{total_tokens_target / dataset.total_tokens:.1f}x)")

    model.train()
    t0 = time.time()
    tokens_seen = resume_step * tokens_per_step
    running_loss = 0.0
    best_loss = float("inf")

    for step in range(resume_step, max_steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = dataset.get_batch(batch_size, device)

        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
        else:
            _, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        tokens_seen += tokens_per_step
        running_loss += loss.item()

        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed
            eta_hours = (max_steps - step - 1) / ((step + 1) / elapsed) / 3600
            print(
                f"step {step + 1:>6d}/{max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tok_per_sec:,.0f} | "
                f"tokens {tokens_seen:,} | "
                f"ETA {eta_hours:.1f}h"
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
            running_loss = 0.0

        if (step + 1) % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"model_step{step + 1}.safetensors")
            model.export_safetensors(ckpt_path)
            config_path = os.path.join(output_dir, "config.json")
            rust_config = {
                "sequence_len": config["sequence_len"],
                "vocab_size": vocab_size,
                "n_layer": depth,
                "n_head": n_head,
                "n_kv_head": n_kv_head,
                "n_embd": n_embd,
                "window_pattern": "SSSL",
            }
            with open(config_path, "w") as f:
                json.dump(rust_config, f, indent=2)
            print(f"  checkpoint saved (best loss so far: {best_loss:.4f})")

    # Final save
    final_path = os.path.join(output_dir, "model.safetensors")
    model.export_safetensors(final_path)
    config_path = os.path.join(output_dir, "config.json")
    rust_config = {
        "sequence_len": config["sequence_len"],
        "vocab_size": vocab_size,
        "n_layer": depth,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "n_embd": n_embd,
        "window_pattern": "SSSL",
    }
    with open(config_path, "w") as f:
        json.dump(rust_config, f, indent=2)

    total_time = time.time() - t0
    print(f"\nTraining complete! {max_steps} steps in {total_time / 60:.1f} minutes")
    print(f"Total tokens: {tokens_seen:,}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()

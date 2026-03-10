# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "safetensors",
#     "regex",
#     "packaging",
# ]
# ///
"""Supervised fine-tuning for picochat using PyTorch.

Takes a pretrained model checkpoint and fine-tunes it on chat data.
Uses the same model architecture as gpu_pretrain.py.

Usage:
    .venv-gpu/bin/python3 -u sft_train.py
"""

import json
import math
import os
import random
import time

import regex
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from gpu_pretrain import PicoChatGPT, PicoChatTokenizer

# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class SFTDataset:
    """Loads chat JSONL data and formats as tokenized training sequences."""

    def __init__(self, jsonl_path: str, tokenizer: PicoChatTokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.vocab_size = tokenizer.vocab_size

        # Special token IDs (at end of vocab, 16 total)
        first_special = self.vocab_size - 16
        self.bos_id = first_special + 0
        self.user_start_id = first_special + 1
        self.user_end_id = first_special + 2
        self.assistant_start_id = first_special + 3
        self.assistant_end_id = first_special + 4
        self.pad_id = first_special + 15

        print(f"Loading SFT data from {jsonl_path}...")
        with open(jsonl_path) as f:
            raw = [json.loads(line) for line in f if line.strip()]

        self.examples = []
        skipped = 0
        for item in raw:
            messages = item["messages"]
            tokens = self._format_chat(messages)
            if len(tokens) <= seq_len:
                self.examples.append(tokens)
            else:
                skipped += 1

        print(f"Loaded {len(self.examples)} examples ({skipped} skipped as too long)")
        lengths = [len(e) for e in self.examples]
        print(f"Token lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")

    def _format_chat(self, messages: list[dict]) -> list[int]:
        """Format messages into picochat chat format with special tokens."""
        tokens = [self.bos_id]
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            content_tokens = self.tokenizer.encode(content)
            if role == "user":
                tokens.append(self.user_start_id)
                tokens.extend(content_tokens)
                tokens.append(self.user_end_id)
            elif role == "assistant":
                tokens.append(self.assistant_start_id)
                tokens.extend(content_tokens)
                tokens.append(self.assistant_end_id)
        return tokens

    def get_batch(self, batch_size: int, device: torch.device):
        """Get a batch of padded sequences with loss masks."""
        batch_indices = random.sample(range(len(self.examples)), min(batch_size, len(self.examples)))
        batch_tokens = [self.examples[i] for i in batch_indices]

        max_len = min(max(len(t) for t in batch_tokens), self.seq_len)

        input_ids = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, tokens in enumerate(batch_tokens):
            seq_len = min(len(tokens), max_len)
            input_ids[i, :seq_len] = torch.tensor(tokens[:seq_len])

            # Only compute loss on assistant responses (between assistant_start and assistant_end)
            in_assistant = False
            for j in range(seq_len - 1):
                if tokens[j] == self.assistant_start_id:
                    in_assistant = True
                    continue
                if tokens[j] == self.assistant_end_id:
                    in_assistant = False
                    # Include the end token itself as a target
                    labels[i, j - 1] = tokens[j]
                    continue
                if in_assistant:
                    labels[i, j] = tokens[j + 1]

        return input_ids.to(device), labels.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_pretrained_weights(model: PicoChatGPT, checkpoint_path: str):
    """Load pretrained weights from safetensors into the PyTorch model."""
    print(f"Loading pretrained weights from {checkpoint_path}...")
    state_dict = load_file(checkpoint_path)

    # Map safetensors names to PyTorch model names
    new_state_dict = {}
    for key, tensor in state_dict.items():
        if key in ("resid_lambdas", "x0_lambdas"):
            new_state_dict[key] = tensor
        elif key == "wte.weight":
            new_state_dict["wte.weight"] = tensor
        elif key == "lm_head.weight":
            new_state_dict["lm_head.weight"] = tensor
        elif key.startswith("h."):
            # h.{i}.attn.c_q.weight -> blocks.{i}.attn.c_q.weight
            parts = key.split(".", 2)
            layer_idx = parts[1]
            rest = parts[2]
            new_state_dict[f"blocks.{layer_idx}.{rest}"] = tensor
        elif key.startswith("ve."):
            # ve.{i}.weight -> value_embeds.{i}.weight
            parts = key.split(".", 2)
            layer_idx = parts[1]
            new_state_dict[f"value_embeds.{layer_idx}.weight"] = tensor

    model.load_state_dict(new_state_dict, strict=True)
    print(f"Loaded {len(new_state_dict)} tensors")


def main():
    # Config matching depth=8 (must match pretrained model)
    depth = 8
    n_embd = 512
    head_dim = 64
    n_head = 8
    n_kv_head = 4
    vocab_size = 4096
    padded_vocab = 4096

    config = {
        "n_embd": n_embd, "n_head": n_head, "n_kv_head": n_kv_head,
        "n_layer": depth, "vocab_size": vocab_size, "padded_vocab": padded_vocab,
        "sequence_len": 2048, "head_dim": head_dim,
    }

    # SFT hyperparameters — lower LR, fewer steps than pretraining
    batch_size = 4
    max_steps = 3000
    max_lr = 2e-5
    min_lr = max_lr * 0.1
    warmup_steps = 100
    weight_decay = 0.01
    grad_clip = 1.0
    log_interval = 10
    save_interval = 1000

    pretrained_path = "runs/gpu-pretrain/model.safetensors"
    sft_data_path = "data/sft_filtered.jsonl"
    tokenizer_path = "runs/tok-owt.json"
    output_dir = "runs/gpu-sft"

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cpu")
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    print(f"Using CPU with {num_threads} threads")

    # Build model and load pretrained weights
    print(f"Building model: depth={depth}, n_embd={n_embd}, vocab={vocab_size}")
    model = PicoChatGPT(config)
    load_pretrained_weights(model, pretrained_path)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    # Optimizer — lower weight decay for fine-tuning
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95))

    def get_lr(step):
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    # Dataset
    tokenizer = PicoChatTokenizer(tokenizer_path)
    dataset = SFTDataset(sft_data_path, tokenizer, seq_len=512)

    # Training
    print(f"\nStarting SFT: {max_steps} steps, batch_size={batch_size}")
    model.train()
    t0 = time.time()
    running_loss = 0.0
    best_loss = float("inf")

    for step in range(max_steps):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        input_ids, labels = dataset.get_batch(batch_size, device)
        logits, _ = model(input_ids, targets=labels)

        # Custom loss: only on assistant tokens (labels != -100)
        vocab_size_actual = config["vocab_size"]
        logits_flat = logits[:, :, :vocab_size_actual].reshape(-1, vocab_size_actual)
        labels_flat = labels.reshape(-1)
        loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - t0
            eta_min = (max_steps - step - 1) / ((step + 1) / elapsed) / 60
            print(
                f"step {step + 1:>5d}/{max_steps} | "
                f"loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"ETA {eta_min:.1f}min"
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
            running_loss = 0.0

        if (step + 1) % save_interval == 0:
            ckpt_path = os.path.join(output_dir, f"model_step{step + 1}.safetensors")
            model.export_safetensors(ckpt_path)
            print(f"  checkpoint saved (best loss: {best_loss:.4f})")

    # Final save
    final_path = os.path.join(output_dir, "model.safetensors")
    model.export_safetensors(final_path)

    # Also save config for Rust inference
    config_path = os.path.join(output_dir, "config.json")
    rust_config = {
        "sequence_len": 2048, "vocab_size": vocab_size,
        "n_layer": depth, "n_head": n_head, "n_kv_head": n_kv_head,
        "n_embd": n_embd, "window_pattern": "SSSL",
    }
    with open(config_path, "w") as f:
        json.dump(rust_config, f, indent=2)

    total_time = time.time() - t0
    print(f"\nSFT complete! {max_steps} steps in {total_time / 60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {final_path}")


if __name__ == "__main__":
    main()

"""
Evaluate a trained GPT-2 checkpoint: text generation + HellaSwag accuracy
(+ optional standardized validation loss, used for the 1.3.4 leaderboard).

Run on a single GPU:

    python eval_1_3.py --hf-repo <user>/<repo> --hf-file model_05000.pt
    python eval_1_3.py --checkpoint log/model_05000.pt
    python eval_1_3.py --checkpoint log/model_05000.pt --val-loss

Reference scores: a randomly initialized GPT-2 gets ~25% on HellaSwag
(chance); the released GPT-2 124M gets 29.6% acc_norm.

The model classes below are a copy of the skeleton in train_gpt2.py. They are
duplicated on purpose: train_gpt2.py runs the training loop at import time, so
we cannot import from it. If you changed the architecture for the race (1.3.4),
adapt the classes here so your checkpoint loads -- but do NOT change the eval
protocol (generation settings, HellaSwag rendering, val-loss constants).
"""
import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

from hellaswag import iterate_examples, render_example

# -----------------------------------------------------------------------------
# model (mirrors train_gpt2.py)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -----------------------------------------------------------------------------

def load_model(args, device):
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(
            repo_id=args.hf_repo, filename=args.hf_file,
            token=os.environ.get("HF_TOKEN"),
        )
    print(f"loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model = GPT(GPTConfig(**checkpoint["config"]))
    state_dict = checkpoint["model"]
    # strip prefixes left by DDP ("module.") or torch.compile ("_orig_mod.")
    state_dict = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }
    # the causal-mask buffer ("attn.bias") may or may not be in the checkpoint,
    # depending on whether flash attention was used during training
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    step, vl = checkpoint.get("step"), checkpoint.get("val_loss")
    print(f"loaded model from step {step}"
          + (f", train-time val_loss {vl:.4f}" if vl is not None else ""))
    return model


@torch.no_grad()
def generate(model, device, prompt="Hello, I'm a language model,",
             num_samples=5, max_length=32, seed=42):
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long)
    xgen = tokens.unsqueeze(0).repeat(num_samples, 1).to(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    while xgen.size(1) < max_length:
        logits, _ = model(xgen)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        # top-50 sampling, as in the video
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1, generator=rng)
        xcol = torch.gather(topk_indices, -1, ix)
        xgen = torch.cat((xgen, xcol), dim=1)
    print(f"\n--- {num_samples} samples for: {prompt!r}\n")
    for i in range(num_samples):
        print(">", enc.decode(xgen[i, :max_length].tolist()))


def get_most_likely_row(tokens, mask, logits):
    """For one HellaSwag example (4 rows of context+ending), return the index
    of the ending with the lowest total loss (pred) and lowest average
    per-token loss (pred_norm), computed only over the ending tokens."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    shift_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_tokens.view(-1), reduction="none",
    ).view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_losses = shift_losses * shift_mask
    sum_loss = masked_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    return sum_loss.argmin().item(), avg_loss.argmin().item()


@torch.no_grad()
def evaluate_hellaswag(model, device, device_type, limit=None):
    num_correct, num_correct_norm, num_total = 0, 0, 0
    for example in iterate_examples("val"):
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=(device_type == "cuda")):
            logits, _ = model(tokens)
        pred, pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        if num_total % 500 == 0:
            print(f"{num_total} acc: {num_correct/num_total:.4f} "
                  f"acc_norm: {num_correct_norm/num_total:.4f}")
        if limit is not None and num_total >= limit:
            break
    print(f"\nHellaSwag ({num_total} examples): "
          f"acc: {num_correct/num_total:.4f} "
          f"acc_norm: {num_correct_norm/num_total:.4f} "
          f"(chance: 0.25, GPT-2 124M release: 0.2955)")


@torch.no_grad()
def evaluate_val_loss(model, device, device_type, data_root):
    """Standardized validation loss for the 1.3.4 leaderboard: a fixed 2M-token
    slice of the FineWeb val shard, fixed B and T, deterministic order.
    Do not change these constants."""
    B, T, num_batches = 16, 1024, 128
    shards = sorted(s for s in os.listdir(data_root) if "val" in s)
    assert shards, f"no val shard found in {data_root}"
    npt = np.load(os.path.join(data_root, shards[0])).astype(np.int32)
    tokens = torch.tensor(npt, dtype=torch.long)
    losses = []
    for i in range(num_batches):
        buf = tokens[i * B * T : (i + 1) * B * T + 1]
        x = buf[:-1].view(B, T).to(device)
        y = buf[1:].view(B, T).to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                            enabled=(device_type == "cuda")):
            _, loss = model(x, y)
        losses.append(loss.item())
    val_loss = sum(losses) / len(losses)
    print(f"\nstandardized val loss ({num_batches} batches of {B}x{T}): {val_loss:.4f}")
    return val_loss


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=str, help="path to a local .pt checkpoint")
    src.add_argument("--hf-repo", type=str, help="HuggingFace repo id, e.g. user/gpt2-run")
    parser.add_argument("--hf-file", type=str, required=True,
                        help="filename inside the HF repo")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--hellaswag-limit", type=int, default=None,
                        help="evaluate only the first N examples (quick check)")
    parser.add_argument("--skip-hellaswag", action="store_true")
    parser.add_argument("--val-loss", action="store_true",
                        help="also compute the standardized FineWeb val loss")
    parser.add_argument("--data-root", type=str, default="/mnt/data/edu_fineweb10B")
    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"using device: {device}")

    model = load_model(args, device)
    generate(model, device, num_samples=args.num_samples, max_length=args.max_length)
    if not args.skip_hellaswag:
        evaluate_hellaswag(model, device, device_type, limit=args.hellaswag_limit)
    if args.val_loss:
        evaluate_val_loss(model, device, device_type, args.data_root)


if __name__ == "__main__":
    main()

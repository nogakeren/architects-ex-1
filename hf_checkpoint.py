"""
Checkpoint saving + HuggingFace upload helper for Exercise 1, Part 3.

Reads HF_TOKEN and HF_REPO_ID from environment variables (passed via --env
in the nebius job command). Call this from the master process only.

Note: the model config is saved as a plain dict (not the GPTConfig object).
Pickling the dataclass would force eval_1_3.py to import train_gpt2.py to
unpickle it -- and importing train_gpt2.py runs the whole training script.
A dict also lets the eval script load with torch.load(weights_only=True).
"""
import os
import sys
from dataclasses import asdict

import torch

repo_id = os.environ.get("HF_REPO_ID")
token = os.environ.get("HF_TOKEN")
if not repo_id or not token:
    print("HF_REPO_ID / HF_TOKEN not set. exiting...")
    sys.exit(1)

from huggingface_hub import HfApi
api = HfApi(token=token)

print("Huggingface API initialized")

def save_and_upload_checkpoint(raw_model, step, val_loss, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "model": raw_model.state_dict(),
        "config": asdict(raw_model.config),
        "step": step,
        "val_loss": val_loss,
    }
    ckpt_path = os.path.join(checkpoint_dir, f"model_{step:05d}.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")
    
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo=os.path.basename(ckpt_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"step {step}, val_loss={val_loss:.4f}",
    )
    print(f"uploaded {os.path.basename(ckpt_path)} to https://huggingface.co/{repo_id}")
    return ckpt_path

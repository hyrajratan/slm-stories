"""
Upload your trained SLM to Hugging Face Hub.

Run once from the project root:
    huggingface-cli login   # paste your token first
    python upload_to_hub.py
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

# ── Change this to your HF username/repo ─────────────────────────────────────
REPO_ID = "YOUR_USERNAME/slm-stories"
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT = Path("checkpoints/best.pt")

if not CHECKPOINT.exists():
    print(f"Checkpoint not found at: {CHECKPOINT}")
    print("Train the model first, then re-run this script.")
    sys.exit(1)

print(f"Found checkpoint: {CHECKPOINT}")
print(f"Uploading to: https://huggingface.co/{REPO_ID}\n")

api = HfApi()

print("Step 1/4  Creating repository...")
create_repo(REPO_ID, repo_type="model", exist_ok=True)
print(f"          Repo ready: https://huggingface.co/{REPO_ID}")

print("\nStep 2/4  Uploading best.pt...")
api.upload_file(
    path_or_fileobj=str(CHECKPOINT),
    path_in_repo="best.pt",
    repo_id=REPO_ID,
    commit_message="Add trained model checkpoint",
)
print("          best.pt uploaded")

print("\nStep 3/4  Uploading source files...")
for fname in ["app.py", "model.py", "config.py", "generate.py"]:
    fpath = Path(fname)
    if not fpath.exists():
        print(f"          {fname} not found — skipping")
        continue
    api.upload_file(
        path_or_fileobj=str(fpath),
        path_in_repo=fname,
        repo_id=REPO_ID,
        commit_message=f"Add {fname}",
    )
    print(f"          {fname} uploaded")

print("\nStep 4/4  Creating model card...")
readme = f"""---
language: en
tags:
  - text-generation
  - story-generation
  - small-language-model
  - pytorch
license: mit
---

# SLM Stories — Small Language Model

A compact GPT-style transformer trained on the
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset
to generate short children's stories.

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 8 transformer blocks |
| Hidden dim | 512 |
| Attention heads | 8 |
| Context length | 256 tokens |
| Vocabulary | GPT-2 BPE (50,257 tokens) |
| Activation | SwiGLU |
| Normalization | RMSNorm |

## Usage

```python
import torch, tiktoken
from huggingface_hub import hf_hub_download
from config import SLMConfig
from model import SLM
from generate import generate_story

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(hf_hub_download("{REPO_ID}", "best.pt"), map_location=device)
cfg = SLMConfig()
model = SLM(cfg).to(device)
model.load_state_dict({{k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()}})
model.eval()

enc = tiktoken.get_encoding("gpt2")
print(generate_story(model, enc, cfg, device, "Once upon a time there was a little cat"))
```
"""

Path("README.md").write_text(readme)
api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    commit_message="Add model card",
)
print("          README.md uploaded")

print(f"\nDone! View your model: https://huggingface.co/{REPO_ID}")

"""
Upload your trained SLM to Hugging Face Hub.

Run this once from your project root:
    python upload_to_hub.py

Requirements:
    pip install huggingface_hub
    huggingface-cli login   (run this first and paste your token)
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

# ─────────────────────────────────────────────────────────
#  ✏️  CHANGE THIS to:  your-hf-username/your-repo-name
# ─────────────────────────────────────────────────────────
REPO_ID = "hyrajratan/slm-stories"
# ─────────────────────────────────────────────────────────

CHECKPOINT = Path("checkpoints/best.pt")

# Verify checkpoint exists before doing anything
if not CHECKPOINT.exists():
    print(f"❌  Checkpoint not found at: {CHECKPOINT}")
    print("    Make sure you have trained the model first (python train.py)")
    sys.exit(1)

print(f"✅  Found checkpoint: {CHECKPOINT}")
print(f"📤  Uploading to: https://huggingface.co/{REPO_ID}\n")

api = HfApi()

# ── Step 1: Create the repo on Hugging Face ──────────────────────────────────
print("Step 1/4  Creating repository on Hugging Face Hub...")
create_repo(REPO_ID, repo_type="model", exist_ok=True)
print(f"          Repo ready: https://huggingface.co/{REPO_ID}")

# ── Step 2: Upload model checkpoint ──────────────────────────────────────────
print("\nStep 2/4  Uploading best.pt (this may take a few minutes)...")
api.upload_file(
    path_or_fileobj=str(CHECKPOINT),
    path_in_repo="best.pt",
    repo_id=REPO_ID,
    commit_message="Add trained model checkpoint",
)
print("          ✅  best.pt uploaded")

# ── Step 3: Upload source files needed at inference time ─────────────────────
print("\nStep 3/4  Uploading source files...")
inference_files = ["model.py", "config.py", "generate.py"]
for fname in inference_files:
    if not Path(fname).exists():
        print(f"          ⚠️  {fname} not found — skipping")
        continue
    api.upload_file(
        path_or_fileobj=fname,
        path_in_repo=fname,
        repo_id=REPO_ID,
        commit_message=f"Add {fname}",
    )
    print(f"          ✅  {fname} uploaded")

# ── Step 4: Create and upload a model card (README.md) ───────────────────────
print("\nStep 4/4  Creating model card (README.md)...")

readme_content = f"""---
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
import torch
import tiktoken
from huggingface_hub import hf_hub_download

# Download weights
weights_path = hf_hub_download(repo_id="{REPO_ID}", filename="best.pt")

# Load model
from config import SLMConfig
from model import SLM
from generate import generate_story

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(weights_path, map_location=device)
cfg = SLMConfig()
model = SLM(cfg).to(device)
model.load_state_dict({{k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()}})
model.eval()

enc = tiktoken.get_encoding("gpt2")
story = generate_story(model, enc, cfg, device, "Once upon a time there was a little cat")
print(story)
```

## Training

Trained for 5,000 iterations on TinyStories with:
- Cosine LR schedule with linear warmup
- Mixed precision (bfloat16)
- Gradient accumulation (effective batch size 128)
- AdamW optimizer with weight decay
"""

with open("README.md", "w") as f:
    f.write(readme_content)

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID,
    commit_message="Add model card",
)
print("          ✅  README.md uploaded")

print(f"""
╔══════════════════════════════════════════════════════════════╗
  ✅  Upload complete!
  🔗  View your model: https://huggingface.co/{REPO_ID}
  📋  Next step: run  python app.py  locally to test,
      then deploy to Streamlit Cloud.
╚══════════════════════════════════════════════════════════════╝
""")

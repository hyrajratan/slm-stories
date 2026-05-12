---
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
weights_path = hf_hub_download(repo_id="hyrajratan/slm-stories", filename="best.pt")

# Load model
from config import SLMConfig
from model import SLM
from generate import generate_story

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(weights_path, map_location=device)
cfg = SLMConfig()
model = SLM(cfg).to(device)
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()})
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

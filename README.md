---
language: en
tags:
  - text-generation
  - story-generation
  - small-language-model
  - pytorch
license: mit
---

# SLM Stories -- Small Language Model

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
ckpt = torch.load(hf_hub_download("hyrajratan/slm-stories", "best.pt"), map_location=device)
cfg = SLMConfig()
model = SLM(cfg).to(device)
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state"].items()})
model.eval()

enc = tiktoken.get_encoding("gpt2")
print(generate_story(model, enc, cfg, device, "Once upon a time there was a little cat"))
```

## Live Demo

Deployed on [Streamlit Cloud](https://streamlit.io).

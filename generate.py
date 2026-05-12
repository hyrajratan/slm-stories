from __future__ import annotations

from typing import Optional

import tiktoken
import torch

from config import SLMConfig
from model import SLM


def generate_story(
    model: SLM,
    enc: tiktoken.Encoding,
    cfg: SLMConfig,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
) -> str:
    """Generate a single continuation from a string prompt."""
    ids = enc.encode_ordinary(prompt)
    if len(ids) == 0:
        ids = [cfg.eot_token]
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    if idx.size(1) > cfg.block_size:
        idx = idx[:, -cfg.block_size:]

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )
    with torch.no_grad(), autocast_ctx:
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token=cfg.eot_token,
        )
    out_ids = out[0].tolist()
    if cfg.eot_token in out_ids[len(ids):]:
        end = len(ids) + out_ids[len(ids):].index(cfg.eot_token)
        out_ids = out_ids[:end]
    return enc.decode(out_ids)

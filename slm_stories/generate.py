"""
Generate stories from a trained SLM checkpoint.

Usage:
    python generate.py
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path
from typing import Optional

import tiktoken
import torch

from config import SLMConfig, config
from model import SLM


SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH = SCRIPT_DIR / config.checkpoint_dir / "best.pt"


# ---------------------------------------------------------------------------
def _config_from_ckpt(saved: dict) -> SLMConfig:
    """Rebuild an SLMConfig from a saved checkpoint, keeping current defaults
    for any field the checkpoint doesn't carry."""
    cfg = SLMConfig()
    known = {f.name for f in fields(SLMConfig)}
    for k, v in saved.items():
        if k in known:
            setattr(cfg, k, v)
    return cfg


def load_model(device: torch.device) -> tuple[SLM, SLMConfig]:
    if not CKPT_PATH.exists():
        print(f"[ERROR] No checkpoint found at {CKPT_PATH}.")
        print("Run `python train.py` first.")
        sys.exit(1)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    cfg = _config_from_ckpt(ckpt.get("config", {}))
    model = SLM(cfg).to(device)

    state = ckpt["model_state"]
    # Strip any torch.compile prefix ("_orig_mod.") if present.
    cleaned = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(cleaned)
    model.eval()
    print(f"Loaded checkpoint from {CKPT_PATH}")
    print(f"  iter={ckpt.get('iter', '?')}  val_loss={ckpt.get('val_loss', '?')}")
    print(f"  params: {model.get_num_params()/1e6:.2f}M")
    return model, cfg


# ---------------------------------------------------------------------------
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
    # Trim at the first EOT after the prompt.
    if cfg.eot_token in out_ids[len(ids):]:
        end = len(ids) + out_ids[len(ids):].index(cfg.eot_token)
        out_ids = out_ids[:end]
    return enc.decode(out_ids)


# ---------------------------------------------------------------------------
def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, cfg = load_model(device)
    enc = tiktoken.get_encoding("gpt2")

    # ---- demo prompts -----------------------------------------------------
    demo_prompts = [
        "Once upon a time there was a little girl named Lily.",
        "One day a boy named Tom found a red ball.",
        "In the forest, there lived a friendly bear.",
    ]
    print("\n" + "=" * 70)
    print("DEMO GENERATIONS")
    print("=" * 70)
    for i, p in enumerate(demo_prompts, 1):
        print(f"\n--- Prompt {i} -----------------------------------------------")
        print(f"> {p}")
        story = generate_story(model, enc, cfg, device, p)
        print(story)

    # ---- interactive mode -------------------------------------------------
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE — type a prompt and press enter. Type 'quit' to exit.")
    print("=" * 70)
    while True:
        try:
            user = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"quit", "exit", "q"}:
            break
        if not user:
            continue
        story = generate_story(model, enc, cfg, device, user)
        print("\n" + story)

    print("\nGoodbye.")


if __name__ == "__main__":
    main()

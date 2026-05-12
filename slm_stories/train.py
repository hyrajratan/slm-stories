"""
Train the SLM on tokenized TinyStories.

Features:
  - Cosine LR schedule with linear warmup.
  - Gradient accumulation (effective batch = batch_size * gradient_accumulation_steps).
  - Mixed precision via torch.autocast (bfloat16 by default — no scaler needed).
  - Gradient clipping.
  - Periodic eval on a held-out split with model.eval() + torch.no_grad().
  - Saves best checkpoint when val_loss improves, plus a "latest.pt" every 1k iters.
  - Optionally resumes from checkpoints/best.pt.
  - CSV log to logs/train_log.csv.
  - Graceful OOM message that suggests reducing batch_size.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from config import SLMConfig, config
from model import SLM


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / config.data_dir
CKPT_DIR = SCRIPT_DIR / config.checkpoint_dir
LOG_DIR = SCRIPT_DIR / config.log_dir

CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_CSV = LOG_DIR / "train_log.csv"
BEST_CKPT = CKPT_DIR / "best.pt"
LATEST_CKPT = CKPT_DIR / "latest.pt"


# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------------
class BinDataLoader:
    """Lightweight memmap-backed loader. Random offset per batch."""

    def __init__(self, cfg: SLMConfig) -> None:
        self.cfg = cfg
        train_path = DATA_DIR / "train.bin"
        val_path = DATA_DIR / "val.bin"
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Tokenized files not found in {DATA_DIR}. "
                "Run `python prepare_data.py` first."
            )
        # Re-create the memmap on each batch — recommended in nanoGPT to avoid
        # a known memory leak; we hold the path here and open in get_batch.
        self.train_path = train_path
        self.val_path = val_path

    def _memmap(self, split: str) -> np.memmap:
        path = self.train_path if split == "train" else self.val_path
        return np.memmap(path, dtype=np.uint16, mode="r")

    def get_batch(self, split: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self._memmap(split)
        B, T = self.cfg.batch_size, self.cfg.block_size
        ix = torch.randint(len(data) - T - 1, (B,))
        x = torch.stack([torch.from_numpy(data[i : i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + T].astype(np.int64)) for i in ix])
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        return x, y


# ---------------------------------------------------------------------------
#  Learning rate schedule
# ---------------------------------------------------------------------------
def get_lr(it: int, cfg: SLMConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / max(1, cfg.warmup_iters)
    if it > cfg.max_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
#  Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    loader: BinDataLoader,
    cfg: SLMConfig,
    device: torch.device,
    autocast_ctx,
) -> dict:
    """Return mean loss on train and val splits over eval_iters batches."""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters, device=device)
        for k in range(cfg.eval_iters):
            x, y = loader.get_batch(split, device)
            with autocast_ctx:
                _, loss = model(x, y)
            losses[k] = loss.float()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = config
    seed_everything(cfg.seed)

    # TF32 — big win on Ada / Blackwell.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    device_type = device.type
    print(f"Using device: {device}")
    if device_type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}  "
              f"({torch.cuda.get_device_properties(device).total_memory/1e9:.1f} GB)")

    # ---- dtype / autocast -------------------------------------------------
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    ptdtype = dtype_map[cfg.dtype]
    use_scaler = ptdtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    autocast_ctx = (
        torch.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    # ---- data -------------------------------------------------------------
    loader = BinDataLoader(cfg)

    # ---- model ------------------------------------------------------------
    model = SLM(cfg).to(device)
    n_params = model.get_num_params()
    print(f"Model parameters: {n_params/1e6:.2f}M "
          f"(non-embedding: {model.get_num_params(non_embedding=True)/1e6:.2f}M)")

    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        device_type=device_type,
    )

    # ---- resume -----------------------------------------------------------
    start_iter = 0
    best_val_loss = float("inf")
    if BEST_CKPT.exists():
        try:
            resp = input(f"Found existing checkpoint at {BEST_CKPT}. Resume? [Y/n]: ").strip().lower()
        except EOFError:
            resp = "y"
        if resp in ("", "y", "yes"):
            print(f"Resuming from {BEST_CKPT}")
            ckpt = torch.load(BEST_CKPT, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_iter = int(ckpt.get("iter", 0)) + 1
            best_val_loss = float(ckpt.get("val_loss", float("inf")))
            print(f"  starting at iter {start_iter}, best_val_loss={best_val_loss:.4f}")

    # ---- compile (after possibly loading raw state_dict) -------------------
    raw_model = model  # keep an uncompiled handle for saving state_dict cleanly
    if cfg.compile and device_type == "cuda":
        print(f"Compiling model with torch.compile(mode='{cfg.compile_mode}') …")
        try:
            model = torch.compile(model, mode=cfg.compile_mode)
        except Exception as e:
            print(f"  torch.compile failed ({e}); falling back to eager mode.")

    # ---- CSV log header ----------------------------------------------------
    if not LOG_CSV.exists():
        with open(LOG_CSV, "w") as f:
            f.write("iter,train_loss,val_loss,lr,tokens_per_sec,iter_ms\n")

    # ---- training loop -----------------------------------------------------
    x, y = loader.get_batch("train", device)  # prefetch first batch
    t_iter = time.time()
    running_loss = 0.0
    tokens_per_iter = cfg.batch_size * cfg.block_size * cfg.gradient_accumulation_steps

    try:
        for it in range(start_iter, cfg.max_iters + 1):
            # set lr
            lr = get_lr(it, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # ---- eval ----
            if it > 0 and it % cfg.eval_interval == 0:
                losses = estimate_loss(model, loader, cfg, device, autocast_ctx)
                msg = (f"[eval] iter {it:>6d}  "
                       f"train_loss {losses['train']:.4f}  "
                       f"val_loss {losses['val']:.4f}  "
                       f"lr {lr:.2e}")
                print(msg)
                with open(LOG_CSV, "a") as f:
                    f.write(f"{it},{losses['train']:.6f},{losses['val']:.6f},"
                            f"{lr:.6e},,\n")

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    if it > 0:
                        torch.save(
                            {
                                "model_state": raw_model.state_dict(),
                                "optimizer_state": optimizer.state_dict(),
                                "config": asdict(cfg),
                                "iter": it,
                                "val_loss": best_val_loss,
                            },
                            BEST_CKPT,
                        )
                        print(f"  -> new best (val={best_val_loss:.4f}); saved {BEST_CKPT.name}")

            # ---- training step (with gradient accumulation) ----
            optimizer.zero_grad(set_to_none=True)
            loss_accum = torch.zeros((), device=device)
            for micro in range(cfg.gradient_accumulation_steps):
                with autocast_ctx:
                    _, loss = model(x, y)
                    loss = loss / cfg.gradient_accumulation_steps
                # prefetch next batch while GPU is busy
                x, y = loader.get_batch("train", device)
                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accum = loss_accum + loss.detach()
            running_loss = loss_accum.item()

            if cfg.grad_clip > 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # ---- logging ----
            if it % cfg.log_interval == 0:
                if device_type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t_iter
                t_iter = time.time()
                iter_ms = dt * 1000.0 / max(1, cfg.log_interval)
                tps = tokens_per_iter * cfg.log_interval / max(dt, 1e-6)
                print(f"iter {it:>6d}  loss {running_loss:.4f}  "
                      f"lr {lr:.2e}  ms/iter {iter_ms:7.1f}  tok/s {tps:,.0f}")
                with open(LOG_CSV, "a") as f:
                    f.write(f"{it},{running_loss:.6f},,{lr:.6e},{tps:.1f},{iter_ms:.2f}\n")

            # ---- periodic 'latest' snapshot ----
            if it > 0 and it % 1000 == 0:
                torch.save(
                    {
                        "model_state": raw_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "iter": it,
                        "val_loss": best_val_loss,
                    },
                    LATEST_CKPT,
                )

    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] CUDA out of memory.")
        print("Suggestions:")
        print("  1. Reduce batch_size in config.py (try 32, then 16).")
        print("  2. Increase gradient_accumulation_steps to keep the effective batch size.")
        print("  3. Reduce block_size from 256 → 128.")
        print("  4. Set compile=False in config.py to lower compile-time memory usage.")
        if device_type == "cuda":
            torch.cuda.empty_cache()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving latest checkpoint …")
        torch.save(
            {
                "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": asdict(cfg),
                "iter": it,
                "val_loss": best_val_loss,
            },
            LATEST_CKPT,
        )
        print(f"Saved {LATEST_CKPT}")

    print(f"\nTraining done. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

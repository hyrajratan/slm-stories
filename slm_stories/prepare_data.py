"""
Download the TinyStories dataset and tokenize it with the GPT-2 BPE tokenizer.

Output:
    data/train.bin   — uint16 numpy memmap of training token IDs
    data/val.bin     — uint16 numpy memmap of validation token IDs

Each story is followed by a <|endoftext|> token (id 50256) so the model
learns to treat story boundaries explicitly and does not bleed context
between unrelated stories.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from config import config


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / config.data_dir
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"


def _tokenize_fn(example: dict, enc: tiktoken.Encoding) -> dict:
    """Tokenize one example and append the <|endoftext|> separator."""
    ids = enc.encode_ordinary(example["text"])
    ids.append(config.eot_token)
    return {"ids": ids, "len": len(ids)}


def _write_split(dset, out_path: Path) -> int:
    """Write a tokenized split as a uint16 memmap. Returns total tokens written."""
    total_len = int(np.sum(dset["len"], dtype=np.int64))
    print(f"  -> {out_path.name}: {total_len:,} tokens "
          f"({total_len * 2 / 1e9:.2f} GB on disk)")

    arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_len,))
    # Chunk the writes — much faster than per-example writes.
    total_batches = 1024
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {out_path.name}"):
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
        arr_batch = np.concatenate(batch["ids"]).astype(np.uint16)
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
    return total_len


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TRAIN_BIN.exists() and VAL_BIN.exists():
        train_tokens = TRAIN_BIN.stat().st_size // 2
        val_tokens = VAL_BIN.stat().st_size // 2
        print("Tokenized data already exists — skipping.")
        print(f"  train.bin: {train_tokens:,} tokens")
        print(f"  val.bin:   {val_tokens:,} tokens")
        print("Delete the .bin files in data/ if you want to re-tokenize.")
        return

    t0 = time.time()
    print(f"Loading dataset: {config.dataset_name}")
    raw = load_dataset(config.dataset_name)
    print(f"  train: {len(raw['train']):,} stories")
    print(f"  val:   {len(raw['validation']):,} stories")

    enc = tiktoken.get_encoding("gpt2")
    print(f"Tokenizing with GPT-2 BPE (vocab={enc.n_vocab}) "
          f"using {config.num_workers} workers …")

    tokenized = raw.map(
        lambda ex: _tokenize_fn(ex, enc),
        remove_columns=["text"],
        desc="tokenizing",
        num_proc=config.num_workers,
    )

    print("Writing binary token files …")
    train_total = _write_split(tokenized["train"], TRAIN_BIN)
    val_total = _write_split(tokenized["validation"], VAL_BIN)

    dt = time.time() - t0
    print("\nDone.")
    print(f"  train tokens: {train_total:,}")
    print(f"  val   tokens: {val_total:,}")
    print(f"  total time:   {dt/60:.1f} min")


if __name__ == "__main__":
    main()

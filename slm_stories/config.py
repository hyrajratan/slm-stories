"""
Central configuration for the Small Language Model (SLM) story generator.

All hyperparameters live here. No magic numbers should appear elsewhere.
Values are tuned for a single RTX 5060 8GB GPU (Ada/Blackwell-class).
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SLMConfig:
    # ---------------- Tokenizer / vocab ----------------
    vocab_size: int = 50257            # GPT-2 BPE vocabulary
    eot_token: int = 50256             # <|endoftext|> separator between stories

    # ---------------- Model architecture ---------------
    block_size: int = 256              # context length in tokens
    n_embd: int = 512                  # embedding / hidden dim
    n_heads: int = 8                   # number of attention heads
    n_layers: int = 8                  # number of transformer blocks
    dropout: float = 0.1               # dropout used on embeddings, attn, mlp
    bias: bool = False                 # no bias in linear/RMSNorm — slightly faster & cleaner

    # ---------------- Optimization ---------------------
    batch_size: int = 32               # micro-batch per forward pass
    gradient_accumulation_steps: int = 4   # effective batch = 32 * 4 = 128
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    max_iters: int = 5_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ---------------- Evaluation / logging -------------
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 5              # console log every N iters

    # ---------------- System ---------------------------
    device: str = "cuda"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    seed: int = 42
    num_workers: int = 4               # multiprocessing workers for tokenization

    # ---------------- Paths ----------------------------
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    log_dir: str = "logs"

    # ---------------- Dataset --------------------------
    dataset_name: str = "roneneldan/TinyStories"

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_heads == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_heads ({self.n_heads})"
        )


# A single shared instance — import this in other modules.
config = SLMConfig()

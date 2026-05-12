from dataclasses import dataclass
from typing import Literal


@dataclass
class SLMConfig:
    # Tokenizer / vocab
    vocab_size: int = 50257
    eot_token: int = 50256

    # Model architecture
    block_size: int = 256
    n_embd: int = 512
    n_heads: int = 8
    n_layers: int = 8
    dropout: float = 0.1
    bias: bool = False

    # Optimization (training only)
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    max_iters: int = 5_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Evaluation / logging (training only)
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 5

    # System
    device: str = "cuda"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    compile: bool = False
    compile_mode: str = "reduce-overhead"
    seed: int = 42
    num_workers: int = 4

    # Paths
    checkpoint_dir: str = "checkpoints"
    data_dir: str = "data"
    log_dir: str = "logs"

    # Dataset
    dataset_name: str = "roneneldan/TinyStories"

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_heads == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_heads ({self.n_heads})"
        )


config = SLMConfig()

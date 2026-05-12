"""
SLM model architecture.

A compact GPT-style transformer with modern best-practice improvements:

- F.scaled_dot_product_attention (PyTorch 2 Flash-Attention) — fast, memory efficient.
- RMSNorm instead of LayerNorm.
- SwiGLU activation in the MLP.
- Weight tying between token embeddings and the output projection.
- GPT-2-style init (std=0.02, residual projections scaled by 1/sqrt(2*n_layers)).
- KV-cache support for fast autoregressive inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SLMConfig


# ----------------------------------------------------------------------------
#  Normalization
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f * rms).to(dtype) * self.weight


# ----------------------------------------------------------------------------
#  Attention
# ----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch's flash SDPA kernel."""

    def __init__(self, cfg: SLMConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_heads
        self.dropout = cfg.dropout

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        new_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
            is_causal = False
        else:
            is_causal = True

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y, new_cache


# ----------------------------------------------------------------------------
#  MLP
# ----------------------------------------------------------------------------
class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: SLMConfig) -> None:
        super().__init__()
        hidden_dim = int(8 * cfg.n_embd / 3)
        hidden_dim = 64 * ((hidden_dim + 63) // 64)

        self.gate = nn.Linear(cfg.n_embd, hidden_dim, bias=cfg.bias)
        self.up = nn.Linear(cfg.n_embd, hidden_dim, bias=cfg.bias)
        self.down = nn.Linear(hidden_dim, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ----------------------------------------------------------------------------
#  Transformer block
# ----------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm transformer block: x = x + attn(norm(x)); x = x + mlp(norm(x))."""

    def __init__(self, cfg: SLMConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLUMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        a, new_cache = self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


# ----------------------------------------------------------------------------
#  Full model
# ----------------------------------------------------------------------------
class SLM(nn.Module):
    """Small Language Model — GPT-style decoder with the listed improvements."""

    def __init__(self, cfg: SLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("down.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        decay_params, nodecay_params = [], []
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay_params if p.dim() >= 2 else nodecay_params).append(p)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        use_fused = device_type == "cuda"
        try:
            optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, fused=use_fused
            )
        except TypeError:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"Sequence length {T} exceeds block_size {self.cfg.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        for block in self.blocks:
            x, _ = block(x, kv_cache=None)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, -1:, :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        eos_token: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching."""
        self.eval()
        device = idx.device

        if idx.size(1) > self.cfg.block_size:
            idx = idx[:, -self.cfg.block_size:]

        B, T = idx.shape
        generated = idx.clone()

        head_dim = self.cfg.n_embd // self.cfg.n_heads
        empty = lambda: (
            torch.zeros(B, self.cfg.n_heads, 0, head_dim,
                        dtype=self.tok_emb.weight.dtype, device=device),
            torch.zeros(B, self.cfg.n_heads, 0, head_dim,
                        dtype=self.tok_emb.weight.dtype, device=device),
        )
        caches: List[Tuple[torch.Tensor, torch.Tensor]] = [empty() for _ in self.blocks]

        x = None
        for t in range(T):
            tok = generated[:, t:t + 1]
            pos_t = torch.tensor([[t]], dtype=torch.long, device=device).expand(B, 1)
            x = self.tok_emb(tok) + self.pos_emb(pos_t)
            new_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for i, block in enumerate(self.blocks):
                x, cache = block(x, kv_cache=caches[i])
                new_caches.append(cache)
            caches = new_caches

        cur_pos = T
        for _ in range(max_new_tokens):
            if cur_pos >= self.cfg.block_size:
                break

            x = self.norm_f(x)
            logits = self.lm_head(x[:, -1:, :]).squeeze(1)
            logits = logits / max(temperature, 1e-5)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_tok], dim=1)

            if eos_token is not None and (next_tok == eos_token).all():
                break

            pos_t = torch.tensor([[cur_pos]], dtype=torch.long, device=device).expand(B, 1)
            x = self.tok_emb(next_tok) + self.pos_emb(pos_t)
            new_caches = []
            for i, block in enumerate(self.blocks):
                x, cache = block(x, kv_cache=caches[i])
                new_caches.append(cache)
            caches = new_caches
            cur_pos += 1

        return generated

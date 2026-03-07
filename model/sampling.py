"""Token sampling strategies: temperature, top-k, top-p, and the combined sample() entry point."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by 1/temperature. temperature must be > 0."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Mask all tokens below the k-th highest logit to -inf."""
    k = min(k, logits.size(-1))
    kth = logits.topk(k, dim=-1).values[:, -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: mask tokens beyond the smallest set summing to probability >= p."""
    sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cumprobs - sorted_logits.softmax(dim=-1) >= p
    remove[:, 0] = False  # always keep the top token
    return logits.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))


def sample(
    logits: torch.Tensor,  # (B, V)
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:  # (B, 1)
    """Apply temperature → top_k → top_p, then sample. temperature=0 → greedy argmax."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = apply_top_k(logits, top_k)
    if top_p is not None:
        logits = apply_top_p(logits, top_p)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

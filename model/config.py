"""ModelConfig, KVCache, and build_kv_cache. No nn.Module dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ModelConfig:
    """All hyperparameters in one place. B=batch, T=seq, C=d_model, H=heads, V=vocab."""

    vocab_size: int = 8_000
    context_length: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1_536
    dropout: float = 0.1
    use_learned_pos_emb: bool = True

    d_k: int = field(init=False)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        self.d_k = self.d_model // self.n_heads

    def num_params(self) -> int:
        """Analytical parameter count estimate."""
        emb = self.vocab_size * self.d_model
        pos = self.context_length * self.d_model if self.use_learned_pos_emb else 0
        attn = 4 * self.d_model * self.d_model
        ff = 2 * self.d_model * self.d_ff
        ln = 4 * self.d_model
        block = attn + ff + ln
        head = self.d_model
        return emb + pos + self.n_layers * block + head


class KVCache:
    """
    Cached keys and values for one attention layer.
    Shapes: k, v → (B, H, T_past, dk). T_past grows by 1 each decode step.
    """

    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None

    def update(
        self,
        new_k: torch.Tensor,  # (B, H, T_new, dk)
        new_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new_k/new_v to the cache and return the full accumulated tensors."""
        if self.k is None:
            self.k, self.v = new_k, new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.k, self.v  # type: ignore[return-value]

    def reset(self) -> None:
        """Clear the cache."""
        self.k = self.v = None


def build_kv_cache(n_layers: int) -> list[KVCache]:
    """Return one empty KVCache per transformer layer."""
    return [KVCache() for _ in range(n_layers)]

"""FeedForward MLP and TransformerBlock (pre-norm, GPT-2 style)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.config import KVCache, ModelConfig


class FeedForward(nn.Module):
    """Two-layer MLP with GELU: fc1 → GELU → dropout → fc2."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: x = x + Attn(LN(x)), x = x + FFN(LN(x))."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_ff = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_attn(x), cache=cache)
        x = x + self.ff(self.ln_ff(x))
        return x

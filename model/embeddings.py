"""Positional encoding modules. Both share the same forward(x, start_pos) signature."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.config import ModelConfig


class LearnedPositionalEmbedding(nn.Module):
    """Trainable embedding per position 0…context_length-1 (GPT-2 style)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.context_length, config.d_model)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        positions = torch.arange(start_pos, start_pos + x.size(1), device=x.device)
        return self.embedding(positions)  # (T, C)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal encodings (Vaswani et al. 2017), stored as a buffer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d, L = config.d_model, config.context_length
        pe = torch.zeros(L, d)
        pos = torch.arange(0, L).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10_000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        return self.pe[start_pos : start_pos + x.size(1)]  # type: ignore[index]


def build_positional_encoding(config: ModelConfig) -> nn.Module:
    """Return the correct positional encoding for the given config."""
    return (
        LearnedPositionalEmbedding(config)
        if config.use_learned_pos_emb
        else SinusoidalPositionalEncoding(config)
    )

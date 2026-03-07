"""Multi-head causal self-attention with optional KV-cache support."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import KVCache, ModelConfig


class MultiHeadAttention(nn.Module):
    """
    Causal self-attention supporting two modes:
    - Training (cache=None): full sequence, upper-triangular mask applied.
    - Decode (cache provided): single new token attends over all cached past tokens.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_model = config.d_model

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.triu(
            torch.ones(config.context_length, config.context_length), diagonal=1
        )
        self.register_buffer("causal_mask", mask.bool())

    def forward(self, x: torch.Tensor, cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T_new, C = x.shape

        Q = self.W_q(x).view(B, T_new, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T_new, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T_new, self.n_heads, self.d_k).transpose(1, 2)

        if cache is not None:
            K, V = cache.update(K, V)  # (B, H, T_past + T_new, dk)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if cache is None:
            scores = scores.masked_fill(self.causal_mask[:T_new, :T_new], float("-inf"))

        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T_new, C)
        return self.resid_dropout(self.W_o(out))

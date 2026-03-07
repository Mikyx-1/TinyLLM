"""Shared type aliases. Import from here instead of repeating raw types in signatures."""

from __future__ import annotations

from typing import Optional

import torch

# Tensor shape hints (B=batch, T=seq len, C=d_model, V=vocab)
BatchTokens = torch.Tensor  # (B, T)
Embeddings = torch.Tensor  # (B, T, C)
Logits = torch.Tensor  # (B, T, V)
Loss = torch.Tensor  # scalar

from model.config import KVCache  # noqa: E402

CacheList = list[KVCache]
OptionalCacheList = Optional[CacheList]
LogitsAndLoss = tuple[Logits, Optional[Loss]]

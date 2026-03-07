"""Autoregressive generation with KV caching. Separated from the model to keep model.py focused."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from model.config import build_kv_cache
from model.sampling import sample
from model.types import CacheList

if TYPE_CHECKING:
    from model.model import TinyLLM


@torch.no_grad()
def generate(
    model: "TinyLLM",
    input_ids: torch.Tensor,  # (B, T_prompt)
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:  # (B, T_prompt + n_generated)
    """
    Generate up to max_new_tokens tokens using KV caching.
    Prefill runs the full prompt once; each decode step processes only the latest token.
    Stops early if eos_id is emitted by all sequences in the batch.
    """
    model.eval()
    caches: CacheList = build_kv_cache(model.config.n_layers)
    prompt_len = input_ids.shape[1]

    # Prefill
    logits, _ = model(input_ids, caches=caches, start_pos=0)
    next_token = sample(logits[:, -1, :], temperature, top_k, top_p)
    input_ids = torch.cat([input_ids, next_token], dim=1)
    if eos_id is not None and (next_token == eos_id).all():
        return input_ids

    # Decode loop
    for step in range(1, max_new_tokens):
        logits, _ = model(next_token, caches=caches, start_pos=prompt_len + step - 1)
        next_token = sample(logits[:, -1, :], temperature, top_k, top_p)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if eos_id is not None and (next_token == eos_id).all():
            break

    return input_ids

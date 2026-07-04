"""TransformerTrunk: shared token/pos embeddings + blocks + final norm, no LM head.

Reused by TinyLLM (adds lm_head), and later by RewardModel/ActorCritic (add different heads)
so all three load the exact same `trunk.*` state-dict keys from one checkpoint lineage.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.blocks import TransformerBlock
from model.config import ModelConfig
from model.embeddings import build_positional_encoding
from model.types import OptionalCacheList


class TransformerTrunk(nn.Module):
    """token_emb + pos_emb -> emb_dropout -> blocks -> ln_final. Returns hidden states (B, T, C)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = build_positional_encoding(config)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.ln_final = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        caches: OptionalCacheList = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        assert (
            T + start_pos <= self.config.context_length
        ), f"Total position {T + start_pos} exceeds context_length {self.config.context_length}"

        x = self.emb_dropout(
            self.token_emb(input_ids) + self.pos_emb(input_ids, start_pos)
        )

        for i, block in enumerate(self.blocks):
            x = block(x, cache=caches[i] if caches is not None else None)

        return self.ln_final(x)

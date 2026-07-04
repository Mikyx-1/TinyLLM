"""TinyLLM: GPT-style decoder-only language model. Handles forward pass only; see generate.py for generation."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import KVCache, ModelConfig, build_kv_cache
from model.trunk import TransformerTrunk
from model.types import LogitsAndLoss, OptionalCacheList


class TinyLLM(nn.Module):
    """
    GPT-style decoder-only LM.

    forward(input_ids, targets, caches, start_pos) → (logits, loss)
      - targets: provide for training loss, else None.
      - caches:  list[KVCache] during generation, None during training.
      - start_pos: position offset for KV-cache decode steps.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.trunk = TransformerTrunk(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.trunk.token_emb.weight  # weight tying

        self._init_weights()
        n = sum(p.numel() for p in self.parameters())
        print(f"TinyLLM initialised: {n / 1e6:.2f}M parameters")

    def _init_weights(self) -> None:
        """GPT-2 init: N(0, 0.02) for linear/embedding; residual projections scaled by 1/√(2·n_layers)."""
        self.apply(self._init_module)
        scale = 0.02 / math.sqrt(2 * self.config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        caches: OptionalCacheList = None,
        start_pos: int = 0,
    ) -> LogitsAndLoss:
        x = self.trunk(input_ids, caches=caches, start_pos=start_pos)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

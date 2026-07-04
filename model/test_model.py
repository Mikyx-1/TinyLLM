"""Smoke-test and timing benchmark. Run: python -m model.test_tinyllm"""

from __future__ import annotations

import math
import time

import torch

from model import ModelConfig, TinyLLM, generate


def main() -> None:
    cfg = ModelConfig(
        vocab_size=1_000,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
    )
    print(f"Estimated params: {cfg.num_params() / 1e6:.3f}M")

    model = TinyLLM(cfg)
    model.eval()

    # Trunk contract: RewardModel/ActorCritic will reuse TransformerTrunk and load
    # these exact `trunk.*` keys from a TinyLLM checkpoint.
    assert hasattr(model, "trunk"), "TinyLLM must expose a `trunk` submodule"
    assert any(k.startswith("trunk.") for k in model.state_dict()), (
        "state_dict keys must be prefixed with 'trunk.'"
    )
    print("Trunk contract OK ✓")

    # Forward pass
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 64, cfg.vocab_size)
    print(
        f"Logits: {logits.shape}  Loss: {loss.item():.4f}  (baseline ≈ {math.log(cfg.vocab_size):.4f})"
    )
    print("Forward pass OK ✓")

    # Generation
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    t0 = time.perf_counter()
    out = generate(model, prompt, max_new_tokens=40, temperature=0)
    print(
        f"Generated 40 tokens in {(time.perf_counter() - t0) * 1000:.1f} ms  shape={out.shape}"
    )
    print("Generation OK ✓")


if __name__ == "__main__":
    main()

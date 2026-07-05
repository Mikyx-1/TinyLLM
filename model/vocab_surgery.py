"""Grow a checkpoint's vocabulary without disturbing any existing embedding row.

Pairs with BPETokenizer.add_special_tokens(), which appends new tokens after the
current vocab instead of reordering it. Since lm_head.weight is tied to
trunk.token_emb.weight (see model/model.py), resizing the embedding table resizes
the output projection too -- one resize covers both directions.

Rows [0, old_vocab_size) keep their trained weights exactly. New rows are left at
whatever TinyLLM's own init scheme produces (N(0, 0.02), see TinyLLM._init_weights) --
they carry no learned information until a fine-tuning run actually uses the new tokens.
Optimizer state isn't preserved (its shapes no longer match); resume the output
checkpoint with resume_weights_only=True.
"""

from __future__ import annotations

import torch

from model.config import ModelConfig
from model.model import TinyLLM

# Params whose shape depends on vocab_size. lm_head.weight is the same tensor as
# trunk.token_emb.weight (weight tying), so copying the embedding covers both.
_VOCAB_DEPENDENT_KEYS = {"trunk.token_emb.weight", "lm_head.weight"}


def resize_checkpoint_vocab(
    checkpoint_path: str, new_vocab_size: int, output_path: str
) -> None:
    """Load a checkpoint, grow its vocab to new_vocab_size, save the result.

    Args:
        checkpoint_path: source checkpoint (as saved by train.py's save_checkpoint).
        new_vocab_size: target vocab size; must be >= the checkpoint's current one.
        output_path: where to write the resized checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    old_cfg_dict = {k: v for k, v in ckpt["model_config"].items() if k != "d_k"}
    old_vocab_size = old_cfg_dict["vocab_size"]

    if new_vocab_size < old_vocab_size:
        raise ValueError(
            f"new_vocab_size ({new_vocab_size}) must be >= current vocab_size ({old_vocab_size})"
        )
    if new_vocab_size == old_vocab_size:
        print(f"Vocab already {old_vocab_size}; copying checkpoint unchanged.")
        torch.save(ckpt, output_path)
        return

    old_model = TinyLLM(ModelConfig(**old_cfg_dict))
    old_model.load_state_dict(ckpt["model_state_dict"])

    new_cfg_dict = dict(old_cfg_dict)
    new_cfg_dict["vocab_size"] = new_vocab_size
    new_model = TinyLLM(ModelConfig(**new_cfg_dict))

    # Copy every param whose shape is unaffected by vocab_size as-is.
    old_sd = old_model.state_dict()
    unaffected_sd = {k: v for k, v in old_sd.items() if k not in _VOCAB_DEPENDENT_KEYS}
    missing, unexpected = new_model.load_state_dict(unaffected_sd, strict=False)
    assert not unexpected, f"unexpected keys during vocab surgery: {unexpected}"
    assert set(missing) == _VOCAB_DEPENDENT_KEYS, f"unaccounted-for keys: {missing}"

    # Rows [0, old_vocab_size) keep their trained weights; new rows keep new_model's
    # own fresh init. lm_head.weight is the same tensor via tying, so this covers it too.
    with torch.no_grad():
        new_model.trunk.token_emb.weight[:old_vocab_size].copy_(
            old_model.trunk.token_emb.weight
        )

    ckpt["model_state_dict"] = new_model.state_dict()
    ckpt["model_config"] = {**new_cfg_dict, "d_k": new_model.config.d_k}
    ckpt.pop("optimizer_state_dict", None)  # shapes no longer match; resume weights-only
    torch.save(ckpt, output_path)
    print(
        f"Resized vocab {old_vocab_size} -> {new_vocab_size}: "
        f"{checkpoint_path} -> {output_path}"
    )

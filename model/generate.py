"""Autoregressive generation with KV caching. Separated from the model to keep model.py focused."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from model.calculator import CalculatorError, calculate, format_result
from model.config import build_kv_cache
from model.sampling import sample
from model.types import CacheList

if TYPE_CHECKING:
    from model.model import TinyLLM
    from tokenizer import BPETokenizer


@torch.no_grad()
def generate(
    model: "TinyLLM",
    input_ids: torch.Tensor,  # (B, T_prompt)
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
    tokenizer: Optional["BPETokenizer"] = None,
    calc_ids: Optional[tuple[int, int]] = None,
) -> torch.Tensor:  # (B, T_prompt + n_generated)
    """
    Generate up to max_new_tokens tokens using KV caching.
    Prefill runs the full prompt once; each decode step processes only the latest token.
    Stops early if eos_id is emitted by all sequences in the batch.

    Pass `tokenizer` and `calc_ids=(open_id, close_id)` to intercept <CALC>...</CALC>
    blocks: as soon as close_id is generated, the expression between the tags is
    decoded, evaluated by model.calculator, and the *real* result is spliced in place
    of whatever the model would have generated next -- the model only has to learn
    when to call the calculator and with what expression, not the arithmetic itself.
    Calculator interception only supports batch size 1 (each row would need its own
    result and its own injected length, which needs padding not implemented here).
    """
    model.eval()
    caches: CacheList = build_kv_cache(model.config.n_layers)
    use_calc = calc_ids is not None and tokenizer is not None
    if use_calc and input_ids.shape[0] != 1:
        raise ValueError("calc_ids interception only supports batch size 1")
    calc_open_id, calc_close_id = calc_ids if use_calc else (None, None)

    # Prefill: process the whole prompt in one call, cache ends up holding prompt_len entries.
    logits, _ = model(input_ids, caches=caches, start_pos=0)
    cur_len = input_ids.shape[1]
    next_token = sample(logits[:, -1, :], temperature, top_k, top_p)
    input_ids = torch.cat([input_ids, next_token], dim=1)

    for step in range(max_new_tokens):
        if eos_id is not None and (next_token == eos_id).all():
            break
        if use_calc and next_token.item() == calc_close_id and cur_len < model.config.context_length:
            input_ids, caches, next_token, cur_len = _inject_calc_result(
                model, tokenizer, input_ids, caches, calc_open_id, cur_len,
                temperature, top_k, top_p,
            )
            continue
        if cur_len >= model.config.context_length:
            break
        # Feed the latest token through the model (adds it to the cache) and sample the next one.
        logits, _ = model(next_token, caches=caches, start_pos=cur_len)
        cur_len += 1
        next_token = sample(logits[:, -1, :], temperature, top_k, top_p)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids


def _inject_calc_result(
    model: "TinyLLM",
    tokenizer: "BPETokenizer",
    input_ids: torch.Tensor,
    caches: CacheList,
    calc_open_id: int,
    cur_len: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> tuple[torch.Tensor, CacheList, torch.Tensor, int]:
    """Called right after a </CALC> token was sampled (appended to input_ids, not yet
    fed through the model / not yet in the cache). Feeds the close tag through the
    model as usual, then -- instead of sampling what follows -- evaluates the
    expression between the matching <CALC> and this </CALC>, and force-feeds the real
    result through the model in its place. Returns the same
    (input_ids, caches, next_token, cur_len) shape the main loop expects, so it can
    just `continue` as if a normal step had happened.
    """
    # Feed the </CALC> tag itself into the cache (a single new token, no masking needed).
    close_tensor = input_ids[:, -1:]
    logits, _ = model(close_tensor, caches=caches, start_pos=cur_len)
    cur_len += 1
    fallback_next = sample(logits[:, -1, :], temperature, top_k, top_p)

    ids_list = input_ids[0].tolist()
    close_idx = len(ids_list) - 1
    open_idx = next(
        (i for i in range(close_idx - 1, -1, -1) if ids_list[i] == calc_open_id), None
    )
    if open_idx is None:
        return input_ids, caches, fallback_next, cur_len  # no matching <CALC>; behave normally

    expr_text = tokenizer.decode(ids_list[open_idx + 1 : close_idx], skip_special_tokens=True)
    try:
        result_text = format_result(calculate(expr_text))
    except CalculatorError:
        return input_ids, caches, fallback_next, cur_len  # bad expression; behave normally

    result_ids = tokenizer.encode(result_text)
    if not result_ids or cur_len + len(result_ids) > model.config.context_length:
        return input_ids, caches, fallback_next, cur_len

    result_tensor = torch.tensor([result_ids], dtype=input_ids.dtype, device=input_ids.device)
    logits, _ = model(result_tensor, caches=caches, start_pos=cur_len)
    cur_len += len(result_ids)  # result_ids are now cached; next_token (below) is not yet
    input_ids = torch.cat([input_ids, result_tensor], dim=1)
    next_token = sample(logits[:, -1, :], temperature, top_k, top_p)
    input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids, caches, next_token, cur_len

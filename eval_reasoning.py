"""
eval_reasoning.py — exact-match accuracy on held-out GSM8K problems.

Measures whether the model can solve problems it never saw during training (unlike
Stage 2's deliberate memorization, this stage is about real generalization) -- see
data_utils.prepare_reasoning_data, which holds these examples out *before*
concatenation into the training corpus.

Run this against both the pre-reasoning-SFT checkpoint and the post-reasoning-SFT one
to get the before/after comparison PLAN.md asks for:

    python eval_reasoning.py --checkpoint checkpoints/sft_memorize_100k/final_calc_ready.pt \\
        --label "before (plain SFT)"
    python eval_reasoning.py --checkpoint checkpoints/reasoning/final.pt \\
        --label "after (reasoning SFT)"

Expect low absolute accuracy at 22M params (see PLAN.md) -- the comparison between the
two runs is the interesting number, not the absolute score.
"""

import argparse
import json
import re

import torch

from model import ModelConfig, TinyLLM
from model.generate import generate
from tokenizer import BPETokenizer

ANSWER_RE = re.compile(r"Answer:\s*(.+)")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_model(checkpoint_path: str, device: torch.device) -> TinyLLM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg_dict = {k: v for k, v in checkpoint["model_config"].items() if k != "d_k"}
    model = TinyLLM(ModelConfig(**cfg_dict)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_final_answer(generated_text: str) -> str | None:
    """Pull the number after the last "Answer:" in the generated text."""
    matches = ANSWER_RE.findall(generated_text)
    if not matches:
        return None
    tail = matches[-1].strip().splitlines()[0]
    number = NUMBER_RE.search(tail)
    return number.group(0) if number else tail.strip()


def answers_match(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    try:
        return float(predicted) == float(gold)
    except ValueError:
        return predicted.strip() == gold.strip()


def evaluate(
    checkpoint_path: str,
    tokenizer_path: str,
    heldout_path: str,
    limit: int | None = None,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    top_k: int = 1,
    use_calc: bool = True,
    verbose: bool = False,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    tokenizer = BPETokenizer.load(tokenizer_path)

    with open(heldout_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if limit is not None:
        examples = examples[:limit]

    calc_ids = None
    if use_calc and "<CALC>" in tokenizer.encoder and "</CALC>" in tokenizer.encoder:
        calc_ids = (tokenizer.encoder["<CALC>"], tokenizer.encoder["</CALC>"])
    elif use_calc:
        print("Warning: tokenizer has no <CALC>/</CALC> ids; running without calculator interception.")

    correct = 0
    evaluated = 0
    for ex in examples:
        prompt = f"<BOS> Question: {ex['question']}\n<THINK>"
        ids = tokenizer.encode(prompt)
        budget = model.config.context_length - len(ids) - 2
        if budget <= 0:
            continue  # prompt alone doesn't fit; can't evaluate this one

        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(
            model,
            x,
            max_new_tokens=min(max_new_tokens, budget),
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id,
            tokenizer=tokenizer,
            calc_ids=calc_ids,
        )
        gen_text = tokenizer.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)
        predicted = extract_final_answer(gen_text)
        is_correct = answers_match(predicted, ex["answer"])
        correct += int(is_correct)
        evaluated += 1

        if verbose:
            tag = "OK  " if is_correct else "MISS"
            print(f"[{tag}] {ex['question'][:70]}")
            print(f"       gold={ex['answer']!r}  pred={predicted!r}")

    acc = correct / evaluated if evaluated else 0.0
    print(f"\nExact-match accuracy: {correct}/{evaluated} ({100*acc:.1f}%)")
    if evaluated < len(examples):
        print(f"  ({len(examples) - evaluated} skipped: prompt alone exceeded context_length)")
    return acc


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate exact-match accuracy on held-out GSM8K problems")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer_path", default="checkpoints/tokenizer.json")
    p.add_argument("--heldout_path", default="data/reasoning_heldout.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--no_calc", action="store_true", help="disable <CALC> interception")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--label", default="", help="printed as a header, for before/after runs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.label:
        print(f"=== {args.label} ===")
    evaluate(
        args.checkpoint,
        args.tokenizer_path,
        args.heldout_path,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_calc=not args.no_calc,
        verbose=args.verbose,
    )

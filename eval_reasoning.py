"""
eval_reasoning.py — exact-match accuracy on held-out reasoning problems, with a
breakdown beyond plain accuracy.

Measures whether the model can solve problems it never saw during training (unlike
Stage 2's deliberate memorization, this stage is about real generalization) -- see
data_utils.prepare_reasoning_data, which holds these examples out *before*
concatenation into the training corpus.

Two extra metrics beyond overall exact-match, motivated by things actually found while
using this repo's pipeline:
- Per-hop accuracy (when examples carry a "hops" field, e.g. the synthetic generator's
  output): overall accuracy alone hid that a 3-hop-only training run scored near-zero
  on 2-hop questions specifically -- a real, actionable failure invisible in one number.
- <CALC>-count match rate: does the model use the same number of calculator steps as
  gold, regardless of whether the final answer comes out right? This is what caught the
  "template-locking" bug directly -- a 3-hop-only-trained model always emitting exactly
  3 <CALC> calls, hallucinating a spurious extra operation on shorter problems, rather
  than tracking how many steps a given problem actually needs.

Run this against before/after checkpoints for a comparison:

    python eval_reasoning.py --checkpoint checkpoints/sft_memorize_100k/final_calc_ready.pt \\
        --label "before (plain SFT)"
    python eval_reasoning.py --checkpoint checkpoints/reasoning/final.pt \\
        --label "after (reasoning SFT)"

Expect low absolute accuracy at this model scale -- the comparison between runs, and
the breakdown within one run, are more informative than the raw number.
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


def resolve_calc_ids(tokenizer: "BPETokenizer", use_calc: bool) -> tuple[int, int] | None:
    if not use_calc:
        return None
    if "<CALC>" in tokenizer.encoder and "</CALC>" in tokenizer.encoder:
        return (tokenizer.encoder["<CALC>"], tokenizer.encoder["</CALC>"])
    print("Warning: tokenizer has no <CALC>/</CALC> ids; running without calculator interception.")
    return None


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


@torch.no_grad()
def run_examples(
    model: TinyLLM,
    tokenizer: "BPETokenizer",
    examples: list[dict],
    device: torch.device,
    calc_ids: tuple[int, int] | None,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    top_k: int = 1,
    format: str = "qa",
) -> list[dict]:
    """Generate for each example; return one result dict per example actually evaluated
    (examples whose prompt alone exceeds context_length are skipped, not counted).

    format="qa" matches REASONING_TEMPLATE / prepare_reasoning_data (stops on <EOS>).
    format="chatml" matches data_utils.reasoning_example_to_conversation /
    prepare_multitask_data (stops on <|im_end|>, emitted after every assistant turn --
    see encode_conversation)."""
    was_training = model.training
    model.eval()
    results = []
    for ex in examples:
        if format == "qa":
            prompt = f"<BOS> Question: {ex['question']}\n<THINK>"
            eos_id = tokenizer.eos_id
        else:
            prompt = f"<|im_start|>user\n{ex['question']}<|im_end|>\n<|im_start|>assistant\n<THINK>"
            eos_id = tokenizer.im_end_id
        ids = tokenizer.encode(prompt)
        budget = model.config.context_length - len(ids) - 2
        if budget <= 0:
            continue

        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = generate(
            model, x, max_new_tokens=min(max_new_tokens, budget), temperature=temperature,
            top_k=top_k, eos_id=eos_id, tokenizer=tokenizer, calc_ids=calc_ids,
        )
        gen_text = tokenizer.decode(out[0, len(ids):].tolist(), skip_special_tokens=False)
        predicted = extract_final_answer(gen_text)
        gold_calc_count = ex["reasoning"].count("<CALC>") if "reasoning" in ex else None
        pred_calc_count = gen_text.count("<CALC>")
        results.append(
            {
                "question": ex["question"],
                "gold": ex["answer"],
                "pred": predicted,
                "correct": answers_match(predicted, ex["answer"]),
                "hops": ex.get("hops"),
                "gold_calc_count": gold_calc_count,
                "pred_calc_count": pred_calc_count,
                "hop_count_match": (
                    gold_calc_count == pred_calc_count if gold_calc_count is not None else None
                ),
                "gen_text": gen_text,
            }
        )
    if was_training:
        model.train()
    return results


def summarize(results: list[dict]) -> dict:
    """Aggregate per-example results into overall accuracy, a per-hop breakdown (only
    when examples carry a "hops" field), and the <CALC>-count match rate."""
    n = len(results)
    if n == 0:
        return {"accuracy": 0.0, "correct": 0, "n": 0}

    correct = sum(r["correct"] for r in results)
    summary = {"accuracy": correct / n, "correct": correct, "n": n}

    by_hop: dict[int, list[bool]] = {}
    for r in results:
        if r["hops"] is not None:
            by_hop.setdefault(r["hops"], []).append(r["correct"])
    if by_hop:
        summary["per_hop"] = {
            h: (sum(v) / len(v), len(v)) for h, v in sorted(by_hop.items())
        }

    hop_matches = [r["hop_count_match"] for r in results if r["hop_count_match"] is not None]
    if hop_matches:
        summary["hop_count_match_rate"] = sum(hop_matches) / len(hop_matches)

    return summary


def print_summary(summary: dict, skipped: int = 0) -> None:
    print(f"\nExact-match accuracy: {summary['correct']}/{summary['n']} ({100*summary['accuracy']:.1f}%)")
    if "per_hop" in summary:
        for h, (acc, n) in summary["per_hop"].items():
            print(f"  {h}-hop: {round(acc*n)}/{n} ({100*acc:.1f}%)")
    if "hop_count_match_rate" in summary:
        print(f"  <CALC>-count match rate (used the right number of steps): "
              f"{100*summary['hop_count_match_rate']:.1f}%")
    if skipped:
        print(f"  ({skipped} skipped: prompt alone exceeded context_length)")


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
    format: str = "qa",
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    tokenizer = BPETokenizer.load(tokenizer_path)

    with open(heldout_path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    if limit is not None:
        examples = examples[:limit]

    calc_ids = resolve_calc_ids(tokenizer, use_calc)
    results = run_examples(
        model, tokenizer, examples, device, calc_ids, max_new_tokens, temperature, top_k, format
    )
    summary = summarize(results)

    if verbose:
        for r in results:
            tag = "OK  " if r["correct"] else "MISS"
            hop_tag = f" ({r['hops']}-hop)" if r["hops"] is not None else ""
            print(f"[{tag}]{hop_tag} {r['question'][:70]}")
            print(f"       gold={r['gold']!r}  pred={r['pred']!r}")

    print_summary(summary, skipped=len(examples) - summary["n"])
    return summary["accuracy"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate exact-match accuracy on held-out reasoning problems")
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
    p.add_argument("--format", choices=["qa", "chatml"], default="qa")
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
        format=args.format,
    )

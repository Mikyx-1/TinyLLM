"""
chat_demo.py — minimal multi-turn demo.

Feeds a sequence of questions one at a time, using the model's own real generated
answer (not any ground-truth text) as context for the next turn -- exactly the
growing-transcript mechanics a real chat session would use. No new generation
machinery: this is just prompt bookkeeping around model/generate.py's existing
KV-cache generate().

Two prompt formats, matching whichever pipeline the checkpoint was trained with:
  --format qa      "<BOS> Question: ...\\nAnswer: ...\\nQuestion: ...\\nAnswer: ..."
                    (data_utils.QA_TEMPLATE / prepare_custom_data). Stops on <EOS>,
                    which this format only ever emits at the very end of a fixed-2-turn
                    example, so a spillover second Question:/Answer: pair is truncated
                    off after the fact.
  --format chatml   "<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n..."
                    (data_utils.encode_conversation / prepare_multiturn_data). Stops on
                    <|im_end|>, which is emitted after *every* turn, so no post-hoc
                    truncation hack is needed -- generation should just stop cleanly.

USAGE:
    python chat_demo.py --checkpoint checkpoints/smalltalk_demo_v2/final.pt \\
        --turns "Hello" "Can you tell me a joke?"
    python chat_demo.py --checkpoint checkpoints/multiturn_chatml_test/final.pt \\
        --tokenizer_path data/tokenizer_multiturn_chatml.json --format chatml \\
        --turns "Hello" "Can you tell me a joke?"
"""

import argparse

import torch

from model import ModelConfig, TinyLLM
from model.generate import generate
from tokenizer import BPETokenizer


def load_model(checkpoint_path: str, device: torch.device) -> TinyLLM:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg_dict = {k: v for k, v in checkpoint["model_config"].items() if k != "d_k"}
    model = TinyLLM(ModelConfig(**cfg_dict)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _format_question(format: str, question: str) -> str:
    if format == "qa":
        return f"Question: {question}\nAnswer:"
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def _format_answer(format: str, answer: str) -> str:
    return f" {answer}\n" if format == "qa" else f"{answer}<|im_end|>\n"


def render_prompt_prefix(format: str, history: list[tuple[str, str]]) -> str:
    """Rebuild the running prompt text from completed (question, answer) turns.

    Pure string formatting, no generation -- lets a stateless caller (e.g. webchat.py,
    which gets the full history resent with every HTTP request instead of holding
    conversation state itself) reconstruct exactly the prompt run_chat/generate_reply
    would have built up turn-by-turn, without re-running generation for turns whose
    answer is already known.
    """
    prompt = "<BOS> " if format == "qa" else ""
    for question, answer in history:
        prompt += _format_question(format, question) + _format_answer(format, answer)
    return prompt


def generate_reply(
    model: TinyLLM,
    tokenizer: BPETokenizer,
    device: torch.device,
    prompt_so_far: str,
    question: str,
    format: str = "qa",
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    top_k: int = 1,
) -> tuple[str, str]:
    """Generate one assistant reply given the running prompt text and a new question.

    Returns (answer, updated_prompt) -- updated_prompt is prompt_so_far with this
    turn's question and generated answer appended, ready to feed back in as
    prompt_so_far for the next turn. Shared by run_chat's CLI loop and webchat.py's
    HTTP handler so the two entry points can't drift out of sync on prompt formatting.
    """
    eos_id = tokenizer.eos_id if format == "qa" else tokenizer.im_end_id
    prompt = prompt_so_far + _format_question(format, question)

    ids = tokenizer.encode(prompt)
    budget = model.config.context_length - len(ids) - 2
    if budget <= 0:
        return "(context full; stopping)", prompt

    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model, x, max_new_tokens=min(max_new_tokens, budget), temperature=temperature,
        top_k=top_k, eos_id=eos_id,
    )
    answer = tokenizer.decode(out[0, len(ids):].tolist(), skip_special_tokens=True).strip()
    if format == "qa":
        # The QA-template model was trained exclusively on exactly-2-turn examples, so
        # it tends to keep going and hallucinate a second Question:/Answer: pair of its
        # own even when asked for just one turn -- same template-locking pattern as the
        # fixed-hop-count issue found earlier, just for turn-count instead. Truncate at
        # the first such spillover so each turn's answer is only this turn's answer.
        # (This tokenizer's decode collapses newlines to spaces, so the split looks for
        # " Question:", not the literal "\nQuestion:" that appears in the training text
        # before encoding.)
        answer = answer.split(" Question:")[0].strip()
    # chatml stops on <|im_end|> after every turn (see encode_conversation), so no
    # equivalent post-hoc truncation should be needed there.

    prompt += _format_answer(format, answer)
    return answer, prompt


def run_chat(
    checkpoint_path: str,
    tokenizer_path: str,
    turns: list[str],
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    top_k: int = 1,
    format: str = "qa",
) -> list[tuple[str, str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    tokenizer = BPETokenizer.load(tokenizer_path)

    prompt = "<BOS> " if format == "qa" else ""
    results = []
    for question in turns:
        answer, prompt = generate_reply(
            model, tokenizer, device, prompt, question, format,
            max_new_tokens, temperature, top_k,
        )
        if answer == "(context full; stopping)":
            print(answer)
            break
        print(f"You:     {question}")
        print(f"TinyLLM: {answer}")
        results.append((question, answer))

    return results


def parse_args():
    p = argparse.ArgumentParser(description="Minimal multi-turn chat demo")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer_path", default="checkpoints/tokenizer.json")
    p.add_argument("--turns", nargs="+", required=True, help="one or more questions, asked in sequence")
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--format", choices=["qa", "chatml"], default="qa")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_chat(
        args.checkpoint, args.tokenizer_path, args.turns,
        args.max_new_tokens, args.temperature, args.top_k, args.format,
    )

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
    python chat_demo.py --checkpoint checkpoints/multitask_chatml/final.pt \\
        --tokenizer_path checkpoints/tokenizer.json --format chatml \\
        --turns "Hello" "A store has 8 boxes of pens, 6 pens per box. It sells 15. How many pens are left?"
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


def _strip_answer_prefix(text: str) -> str:
    """Cosmetic only: REASONING_TEMPLATE/reasoning_example_to_conversation always
    render the final answer as literal "Answer: {answer}" text -- strip that label for
    display so a reasoning reply doesn't read "Answer: 12" in the chat bubble, the way
    ChatGPT/Claude just show the answer itself."""
    return text[len("Answer:"):].strip() if text.startswith("Answer:") else text


def _split_reasoning(tokenizer: BPETokenizer, gen_ids: list[int]) -> tuple[str | None, str]:
    """Split generated token ids into (reasoning_or_None, answer_text).

    Splits on token ids, not decoded text -- decode(skip_special_tokens=True) drops
    the <THINK>/</THINK> tags entirely, which would leave no marker in the string to
    split on. Whether a <THINK>...</THINK> block is present at all is content-dependent
    (reasoning_example_to_conversation embeds one; ordinary chat turns never have one),
    so this returns (None, full_text) whenever there's no complete block -- including a
    reasoning attempt that got cut off before closing the tag, since a half-formed
    split is worse than just showing everything as the answer.
    """
    think_start_id = tokenizer.encoder.get("<THINK>")
    think_end_id = tokenizer.encoder.get("</THINK>")
    if (
        think_start_id is not None
        and think_end_id is not None
        and think_start_id in gen_ids
        and think_end_id in gen_ids
    ):
        start = gen_ids.index(think_start_id)
        end = gen_ids.index(think_end_id)
        if end > start:
            reasoning = tokenizer.decode(gen_ids[start + 1 : end], skip_special_tokens=True).strip()
            answer = _strip_answer_prefix(
                tokenizer.decode(gen_ids[end + 1 :], skip_special_tokens=True).strip()
            )
            return reasoning, answer
    return None, _strip_answer_prefix(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())


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
) -> dict:
    """Generate one assistant reply given the running prompt text and a new question.

    Returns {"answer", "reasoning", "full", "prompt"}:
      - answer: user-facing text with any <THINK>...</THINK> trace stripped out, meant
        for a Claude/ChatGPT-style collapsed "thoughts" section instead of showing the
        reasoning trace inline with the final answer.
      - reasoning: the <THINK>...</THINK> trace when the model chose to reason for this
        turn (None for ordinary chat turns -- see _split_reasoning).
      - full: the complete generated text, reasoning included -- what the model was
        actually trained to condition on, so callers must use *this* (not `answer`)
        when persisting turn history for a future request's context, or the model
        would lose its own reasoning trace from the transcript it conditions on.
      - prompt: prompt_so_far with this turn's question and `full` answer appended,
        ready to feed back in as prompt_so_far for the next turn.
    Shared by run_chat's CLI loop and webchat.py's HTTP handler so the two entry points
    can't drift out of sync on prompt formatting.
    """
    eos_id = tokenizer.eos_id if format == "qa" else tokenizer.im_end_id
    prompt = prompt_so_far + _format_question(format, question)

    ids = tokenizer.encode(prompt)
    budget = model.config.context_length - len(ids) - 2
    if budget <= 0:
        msg = "(context full; stopping)"
        return {"answer": msg, "reasoning": None, "full": msg, "prompt": prompt}

    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(
        model, x, max_new_tokens=min(max_new_tokens, budget), temperature=temperature,
        top_k=top_k, eos_id=eos_id,
    )
    gen_ids = out[0, len(ids):].tolist()
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    reasoning, answer = _split_reasoning(tokenizer, gen_ids)
    if format == "qa":
        # The QA-template model was trained exclusively on exactly-2-turn examples, so
        # it tends to keep going and hallucinate a second Question:/Answer: pair of its
        # own even when asked for just one turn -- same template-locking pattern as the
        # fixed-hop-count issue found earlier, just for turn-count instead. Truncate at
        # the first such spillover so each turn's answer is only this turn's answer.
        # (This tokenizer's decode collapses newlines to spaces, so the split looks for
        # " Question:", not the literal "\nQuestion:" that appears in the training text
        # before encoding.)
        full_text = full_text.split(" Question:")[0].strip()
        answer = answer.split(" Question:")[0].strip()
    # chatml stops on <|im_end|> after every turn (see encode_conversation), so no
    # equivalent post-hoc truncation should be needed there.

    prompt += _format_answer(format, full_text)
    return {"answer": answer, "reasoning": reasoning, "full": full_text, "prompt": prompt}


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
        result = generate_reply(
            model, tokenizer, device, prompt, question, format,
            max_new_tokens, temperature, top_k,
        )
        prompt = result["prompt"]
        if result["answer"] == "(context full; stopping)":
            print(result["answer"])
            break
        print(f"You:     {question}")
        if result["reasoning"]:
            print(f"  [thoughts] {result['reasoning']}")
        print(f"TinyLLM: {result['answer']}")
        results.append((question, result["answer"]))

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

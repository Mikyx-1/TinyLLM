"""
inference.py — load a checkpoint and generate text from a prompt.

A chatml checkpoint (train_reasoning.py --dataset_format chatml) was never trained to
complete a bare prompt -- it only ever saw questions wrapped as "<|im_start|>user\\n
...<|im_end|>\\n<|im_start|>assistant\\n" and stops on <|im_end|>, not <EOS>. Feeding it
a raw prompt and waiting for <EOS> is off-distribution from the first token and never
hits its real stop condition, so generation degrades into runaway repetition instead of
one clean answer -- see demo.chat_demo, which gets this right. This script auto-detects
that case from the checkpoint's own stored dataset_format and applies the same
wrapping/stop-token demo.chat_demo uses, so the two tools agree by default. Pass --raw
to force genuinely raw completion on a chatml checkpoint anyway (e.g. to see that
failure mode on purpose).

Usage:
    python -m demo.inference --checkpoint checkpoints/latest.pt --prompt "To be or not to be"
    python -m demo.inference --checkpoint checkpoints/latest.pt --prompt "Friends, Romans" --temperature 0.8 --top_k 40
    python -m demo.inference --checkpoint checkpoints/latest.pt --prompt "What" --temperature 0 --max_tokens 200
"""

import argparse
import os

import torch

from demo.chat_demo import _format_question
from model import ModelConfig, TinyLLM
from model.generate import generate
from tokenizer import BPETokenizer


def load_checkpoint(
    checkpoint_path: str, device: torch.device, tokenizer_path: str | None = None
) -> tuple[TinyLLM, BPETokenizer, dict]:
    """Load model and tokenizer from a training checkpoint.

    Also returns the checkpoint's own stage config (train_config/pretrain_config, or
    {} if neither is present) -- main() reads its "dataset_format" back out to decide
    whether this is a chatml checkpoint that needs prompt wrapping (see run_inference).
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    stage_config = checkpoint.get("train_config") or checkpoint.get("pretrain_config") or {}

    # Exclude derived fields computed in __post_init__ (e.g. d_k)
    cfg_dict = {k: v for k, v in checkpoint["model_config"].items() if k != "d_k"}
    model = TinyLLM(ModelConfig(**cfg_dict)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    auto_detected = tokenizer_path is None
    if auto_detected:
        # train.py, train_reasoning.py, and pretrain.py's configs ("train_config" or
        # "pretrain_config" depending on stage) all store their own tokenizer_path
        # directly -- read it back from whichever one the checkpoint has, so this
        # resolves correctly regardless of which stage produced the checkpoint.
        tokenizer_path = stage_config.get("tokenizer_path", "checkpoints/tokenizer.json")
    if not os.path.exists(tokenizer_path):
        source = "auto-detected from the checkpoint" if auto_detected else "passed via --tokenizer_path"
        raise FileNotFoundError(
            f"Tokenizer not found at '{tokenizer_path}' ({source}). Pass the correct "
            f"--tokenizer_path explicitly, e.g. --tokenizer_path checkpoints/tokenizer.json"
        )
    tokenizer = BPETokenizer.load(tokenizer_path)

    print(f"Params     : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Vocab size : {tokenizer.vocab_size}")
    print(f"Stopped at : iter {checkpoint.get('iter_num', 'unknown')}")

    return model, tokenizer, stage_config


def run_inference(
    model: TinyLLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float | None = None,
    device: torch.device = torch.device("cpu"),
    chatml: bool = False,
) -> str:
    """Encode prompt, generate, decode and return only the new tokens.

    chatml=True wraps `prompt` in the same "<|im_start|>user\\n...<|im_end|>\\n
    <|im_start|>assistant\\n" template demo.chat_demo._format_question uses, and stops
    on <|im_end|> instead of <EOS> -- required for a dataset_format=chatml checkpoint,
    which was never trained to complete a bare prompt (see this module's docstring).
    """
    encode_text = _format_question("chatml", prompt) if chatml else prompt
    eos_id = tokenizer.im_end_id if chatml else tokenizer.eos_id

    input_ids = tokenizer.encode(encode_text) or [tokenizer.bos_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"\nPrompt ({len(input_ids)} tokens): '{prompt}'" + (" [wrapped as chatml]" if chatml else ""))
    print(f"Generating {max_new_tokens} tokens (temp={temperature}, top_k={top_k})...")
    print("-" * 60)

    output_ids = generate(
        model,
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
    )

    return tokenizer.decode(output_ids[0, len(input_ids):].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with TinyLLM")
    parser.add_argument("--checkpoint",   type=str,   default="checkpoints/latest.pt")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                         help="Override tokenizer path (required for pretrain.py checkpoints)")
    parser.add_argument("--prompt",       type=str,   default="To be or not to be")
    parser.add_argument("--max_tokens",   type=int,   default=200)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top_k",        type=int,   default=50)
    parser.add_argument("--top_p",        type=float, default=None)
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--raw", action="store_true",
                         help="skip chatml auto-wrapping even if the checkpoint was "
                              "trained with dataset_format=chatml (produces "
                              "off-distribution output on purpose -- see module docstring)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    model, tokenizer, stage_config = load_checkpoint(
        args.checkpoint, device, tokenizer_path=args.tokenizer_path
    )
    chatml = stage_config.get("dataset_format") == "chatml" and not args.raw
    if stage_config.get("dataset_format") == "chatml" and args.raw:
        print("--raw passed: skipping chatml auto-wrapping (expect off-distribution output)")

    result = run_inference(
        model, tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        chatml=chatml,
    )

    print(f"PROMPT   : {args.prompt}")
    print(f"GENERATED:\n{result}")
    print("-" * 60)


if __name__ == "__main__":
    main()
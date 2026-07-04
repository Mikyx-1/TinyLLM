"""
inference.py — load a checkpoint and generate text from a prompt.

Usage:
    python inference.py --checkpoint checkpoints/latest.pt --prompt "To be or not to be"
    python inference.py --checkpoint checkpoints/latest.pt --prompt "Friends, Romans" --temperature 0.8 --top_k 40
    python inference.py --checkpoint checkpoints/latest.pt --prompt "What" --temperature 0 --max_tokens 200
"""

import argparse

import torch

from model import ModelConfig, TinyLLM
from model.generate import generate
from tokenizer import BPETokenizer


def load_checkpoint(
    checkpoint_path: str, device: torch.device, tokenizer_path: str | None = None
) -> tuple[TinyLLM, BPETokenizer]:
    """Load model and tokenizer from a training checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Exclude derived fields computed in __post_init__ (e.g. d_k)
    cfg_dict = {k: v for k, v in checkpoint["model_config"].items() if k != "d_k"}
    model = TinyLLM(ModelConfig(**cfg_dict)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if tokenizer_path is None:
        # train.py checkpoints carry "train_config"; pretrain.py's don't, so this falls
        # back to the (usually wrong) default — pass --tokenizer_path explicitly for those.
        tokenizer_path = checkpoint.get("train_config", {}).get("data_dir", "data") + "/tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path)

    print(f"Params     : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Vocab size : {tokenizer.vocab_size}")
    print(f"Stopped at : iter {checkpoint.get('iter_num', 'unknown')}")

    return model, tokenizer


def run_inference(
    model: TinyLLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float | None = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Encode prompt, generate, decode and return only the new tokens."""
    input_ids = tokenizer.encode(prompt) or [tokenizer.bos_id]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"\nPrompt ({len(input_ids)} tokens): '{prompt}'")
    print(f"Generating {max_new_tokens} tokens (temp={temperature}, top_k={top_k})...")
    print("-" * 60)

    output_ids = generate(
        model,
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eos_id,
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    model, tokenizer = load_checkpoint(args.checkpoint, device, tokenizer_path=args.tokenizer_path)

    result = run_inference(
        model, tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print(f"PROMPT   : {args.prompt}")
    print(f"GENERATED:\n{result}")
    print("-" * 60)


if __name__ == "__main__":
    main()
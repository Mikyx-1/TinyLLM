"""
Text generation / inference script.

Loads a trained checkpoint and generates text from a prompt.

Usage:
    python generate.py --checkpoint checkpoints/latest.pt --prompt "To be or not to be"
    python generate.py --checkpoint checkpoints/latest.pt --prompt "Friends, Romans" --temperature 0.8 --top_k 40
    python generate.py --checkpoint checkpoints/latest.pt --prompt "What" --temperature 0 --max_tokens 200
"""

import argparse

import torch

from model import ModelConfig, TinyLLM
from tokenizer import BPETokenizer


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> tuple[TinyLLM, BPETokenizer]:
    """Load model and tokenizer from a training checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model config (exclude derived fields computed in __post_init__)
    model_cfg_dict = {k: v for k, v in checkpoint["model_config"].items() if k != "d_k"}
    model_config = ModelConfig(**model_cfg_dict)

    # Load model
    model = TinyLLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load tokenizer
    tok_path = (
        checkpoint.get("train_config", {}).get("data_dir", "data") + "/tokenizer.json"
    )
    tokenizer = BPETokenizer.load(tok_path)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"Tokenizer loaded: {tokenizer.vocab_size} tokens")
    print(f"Training stopped at iter: {checkpoint.get('iter_num', 'unknown')}")

    return model, tokenizer


def generate_text(
    model: TinyLLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = None,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text from a prompt string."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        # Fallback if prompt tokenizes to nothing
        input_ids = [tokenizer.bos_id]

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"\nPrompt ({len(input_ids)} tokens): '{prompt}'")
    print(f"Generating {max_new_tokens} tokens (temp={temperature}, top_k={top_k})...")
    print("-" * 60)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_id=tokenizer.eos_id,
        )

    # Decode the new tokens only (skip the prompt)
    new_ids = output_ids[0, len(input_ids) :].tolist()
    generated = tokenizer.decode(new_ids)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate text from TinyLLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--prompt", type=str, default="To be or not to be")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model_from_checkpoint(args.checkpoint, device)

    # Generate
    result = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print(f"PROMPT: {args.prompt}")
    print(f"GENERATED:\n{result}")
    print("-" * 60)


if __name__ == "__main__":
    main()

"""
Step 1 — Prepare data for TinyLLM pretraining.

USAGE (in order):

    # Train the BPE tokenizer on your raw corpus:
    python -m data_pipeline.pretokenize --train_tokenizer --corpus_dir data/raw_text

    # Tokenize the corpus to fast binary files:
    python -m data_pipeline.pretokenize --corpus_dir data/raw_text --out_dir data/tokenized

    # Then pretrain:
    python -m training.pretrain --corpus_dir data/tokenized
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

from tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def corpus_files(corpus_dir: str) -> list[str]:
    return sorted(
        glob.glob(os.path.join(corpus_dir, "**", "*.txt"), recursive=True)
        + glob.glob(os.path.join(corpus_dir, "**", "*.jsonl"), recursive=True)
    )


def read_text(fp: str) -> str:
    text = Path(fp).read_text(errors="replace")
    if fp.endswith(".jsonl"):
        text = "\n".join(
            json.loads(l).get("text", "") for l in text.splitlines() if l.strip()
        )
    return text


# ---------------------------------------------------------------------------
# Step 1a: train tokenizer
# ---------------------------------------------------------------------------

def train_tokenizer(corpus_dir: str, tokenizer_path: str, vocab_size: int):
    files = corpus_files(corpus_dir)
    if not files:
        raise FileNotFoundError(f"No .txt/.jsonl files in {corpus_dir!r}")

    print(f"Training BPE tokenizer  vocab_size={vocab_size}  files={len(files)}")
    corpus = "\n".join(read_text(fp) for fp in files)

    tok = BPETokenizer()
    tok.train(corpus, vocab_size=vocab_size)

    os.makedirs(os.path.dirname(tokenizer_path) or ".", exist_ok=True)
    tok.save(tokenizer_path)
    print(f"Saved tokenizer → {tokenizer_path}")
    print(f"Next: python -m data_pipeline.pretokenize --corpus_dir {corpus_dir}")


# ---------------------------------------------------------------------------
# Step 1b: tokenize corpus to binary
# ---------------------------------------------------------------------------

def pretokenize(corpus_dir: str, tokenizer_path: str, out_dir: str):
    tokenizer = BPETokenizer.load(tokenizer_path)
    files = corpus_files(corpus_dir)
    if not files:
        raise FileNotFoundError(f"No .txt/.jsonl files in {corpus_dir!r}")

    os.makedirs(out_dir, exist_ok=True)
    dtype = np.uint16 if tokenizer.vocab_size <= 65535 else np.uint32

    print(f"Tokenizer  : {tokenizer_path}  (vocab={tokenizer.vocab_size})")
    print(f"Files      : {len(files)}  from {corpus_dir}")
    print(f"Output     : {out_dir}")
    print()

    total_tokens = 0
    out_files = []
    t_total = time.perf_counter()

    for i, fp in enumerate(files):
        t0 = time.perf_counter()
        ids = tokenizer.encode(read_text(fp))
        ids.append(tokenizer.eos_id)

        out_path = os.path.join(out_dir, Path(fp).stem + ".bin")
        np.array(ids, dtype=dtype).tofile(out_path)

        dt = time.perf_counter() - t0
        total_tokens += len(ids)
        out_files.append({"src": fp, "out": out_path, "tokens": len(ids)})
        print(
            f"  [{i+1:>3}/{len(files)}]  {Path(fp).name:<40}"
            f"  {len(ids):>10,} tok  {os.path.getsize(fp)/1e6:.1f} MB"
            f"  {dt:.1f}s  ({len(ids)/dt:,.0f} tok/s)"
        )

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "vocab_size": tokenizer.vocab_size,
            "eos_id": tokenizer.eos_id,
            "total_tokens": total_tokens,
            "num_files": len(out_files),
            "files": out_files,
            "dtype": dtype.__name__,
        }, f, indent=2)

    elapsed = time.perf_counter() - t_total
    print()
    print(f"  Total tokens : {total_tokens:,}")
    print(f"  Time         : {elapsed:.1f}s  ({total_tokens/elapsed:,.0f} tok/s)")
    print(f"  Stats        : {stats_path}")
    print(f"\nDone! Now run:")
    print(f"  python -m training.pretrain --corpus_dir {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_dir",      default="data/raw_text")
    p.add_argument("--tokenizer_path",  default="checkpoints/tokenizer.json")
    p.add_argument("--out_dir",         default="data/tokenized")
    p.add_argument("--vocab_size",      type=int, default=8_000)
    p.add_argument("--train_tokenizer", action="store_true",
                   help="Train BPE tokenizer and exit (run before tokenizing)")
    args = p.parse_args()

    if args.train_tokenizer:
        train_tokenizer(args.corpus_dir, args.tokenizer_path, args.vocab_size)
    else:
        pretokenize(args.corpus_dir, args.tokenizer_path, args.out_dir)
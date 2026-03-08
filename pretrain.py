"""
Pre-Training Stage for TinyLLM

HOW TO RUN:
    # 1. Train tokenizer and tokenize corpus (once):
    python pretokenize.py --train_tokenizer --corpus_dir data/raw_text
    python pretokenize.py --corpus_dir data/raw_text --out_dir data/tokenized

    # 2. Pre-train:
    python pretrain.py --corpus_dir data/tokenized

    # 3. Multi-GPU:
    torchrun --nproc_per_node=2 pretrain.py --corpus_dir data/tokenized

    # 4. Resume:
    python pretrain.py --corpus_dir data/tokenized --resume_from checkpoints/pretrain/latest.pt
"""

import argparse
import glob
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

from model import ModelConfig, TinyLLM
from tokenizer import BPETokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PretrainConfig:
    corpus_dir: str = "data/tokenized"
    tokenizer_path: str = "checkpoints/tokenizer.json"

    context_length: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.0

    batch_size: int = 64
    grad_accumulation_steps: int = 4
    max_iters: int = 20_000
    warmup_iters: int = 500
    min_lr: float = 1e-5
    max_lr: float = 6e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints/pretrain"
    checkpoint_interval: int = 2000
    resume_from: str = ""
    val_fraction: float = 0.02

    dtype: str = "bfloat16"
    compile: bool = True
    num_workers: int = 4
    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _corpus_files(corpus_dir: str) -> list[str]:
    # Prefer pre-tokenized .bin files; fall back to raw text
    bin_files = sorted(
        glob.glob(os.path.join(corpus_dir, "**", "*.bin"), recursive=True)
    )
    if bin_files:
        return bin_files
    return sorted(
        glob.glob(os.path.join(corpus_dir, "**", "*.txt"), recursive=True)
        + glob.glob(os.path.join(corpus_dir, "**", "*.jsonl"), recursive=True)
    )


def _stream_tokens(
    files: list[str],
    tokenizer: BPETokenizer,
    context_length: int,
    shuffle: bool = True,
    seed: int = 42,
    chunk_range: tuple[float, float] = (0.0, 1.0),
) -> Iterator[torch.Tensor]:
    """
    Yield packed (context_length + 1,) tensors.

    chunk_range: (start_frac, end_frac) slice of the total chunk stream to yield.
    Used to split train/val from the same files regardless of file count.

    Fast path (.bin): np.memmap — no tokenization cost.
    Slow path (.txt/.jsonl): tokenize on the fly. Run pretokenize.py first.
    """
    rng = random.Random(seed)
    order = list(files)
    if shuffle:
        rng.shuffle(order)

    chunk = context_length + 1
    start_frac, end_frac = chunk_range

    if files and files[0].endswith(".bin"):
        dtype = np.uint16
        stats_path = os.path.join(os.path.dirname(files[0]), "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                s = json.load(f)
            dtype = np.uint16 if s.get("dtype", "uint16") == "uint16" else np.uint32

        for fp in order:
            data = np.memmap(fp, dtype=dtype, mode="r")
            n_chunks = (len(data) - chunk + 1) // chunk
            i_start = int(n_chunks * start_frac)
            i_end = int(n_chunks * end_frac)
            for i in range(i_start, i_end):
                start = i * chunk
                yield torch.from_numpy(data[start : start + chunk].astype(np.int64))
        return

    # Slow path: collect all chunks first so we can slice by fraction
    all_chunks: list[torch.Tensor] = []
    buffer: list[int] = []
    for fp in order:
        text = Path(fp).read_text(errors="replace")
        if fp.endswith(".jsonl"):
            text = "\n".join(
                json.loads(l).get("text", "") for l in text.splitlines() if l.strip()
            )
        ids = tokenizer.encode(text)
        ids.append(tokenizer.eos_id)
        buffer.extend(ids)
        while len(buffer) >= chunk:
            all_chunks.append(torch.tensor(buffer[:chunk], dtype=torch.long))
            buffer = buffer[chunk:]
    if len(buffer) > 1:
        all_chunks.append(
            torch.tensor(buffer + [0] * (chunk - len(buffer)), dtype=torch.long)
        )

    n = len(all_chunks)
    i_start = int(n * start_frac)
    i_end = int(n * end_frac)
    yield from all_chunks[i_start:i_end]


class PretrainDataset(IterableDataset):
    def __init__(
        self,
        files,
        tokenizer,
        context_length,
        shuffle=True,
        seed=42,
        rank=0,
        world_size=1,
        chunk_range=(0.0, 1.0),
    ):
        self.files = [f for i, f in enumerate(files) if i % world_size == rank]
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.shuffle = shuffle
        self.seed = seed
        self.chunk_range = chunk_range

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        files = self.files
        if wi is not None:
            files = [f for i, f in enumerate(files) if i % wi.num_workers == wi.id]
        yield from _stream_tokens(
            files,
            self.tokenizer,
            self.context_length,
            self.shuffle,
            self.seed + (wi.id if wi else 0),
            chunk_range=self.chunk_range,
        )


def collate_fn(batch):
    s = torch.stack(batch)
    return s[:, :-1].contiguous(), s[:, 1:].contiguous()


def build_dataloaders(
    config: PretrainConfig, tokenizer: BPETokenizer, rank: int, world_size: int
):
    all_files = _corpus_files(config.corpus_dir)
    if not all_files:
        raise FileNotFoundError(
            f"No .bin/.txt/.jsonl files found in {config.corpus_dir!r}"
        )

    random.seed(config.seed)
    random.shuffle(all_files)

    # Split the token stream by fraction — works with any number of files,
    # including a single file. Val draws from the tail of each file's chunks.
    val_start = 1.0 - config.val_fraction

    train_ds = PretrainDataset(
        all_files,
        tokenizer,
        config.context_length,
        shuffle=True,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
        chunk_range=(0.0, val_start),
    )
    val_ds = PretrainDataset(
        all_files,
        tokenizer,
        config.context_length,
        shuffle=False,
        seed=config.seed,
        chunk_range=(val_start, 1.0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=4 if config.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        drop_last=False,
        persistent_workers=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# LR schedule (cosine with linear warmup)
# ---------------------------------------------------------------------------


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    p = min((step - warmup) / max(total - warmup, 1), 1.0)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return 0, 0, 1, False


# ---------------------------------------------------------------------------
# Eval + checkpointing
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, val_loader, eval_iters, device, ctx):
    model.eval()
    total, count = 0.0, 0
    it = iter(val_loader)
    for _ in range(eval_iters):
        try:
            x, y = next(it)
        except StopIteration:
            break
        with ctx:
            _, loss = model(x.to(device), targets=y.to(device))
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def save_checkpoint(model, optimizer, step, val_loss, config, model_config, path):
    raw = model.module if isinstance(model, DDP) else model
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter_num": step,
            "val_loss": val_loss,
            "pretrain_config": config.__dict__,
            "model_config": model_config.__dict__,
        },
        path,
    )
    print(
        f"  [ckpt] {path}  iter={step}  val_loss={val_loss:.4f}  ppl={math.exp(min(val_loss,20)):.1f}"
    )


def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    raw = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [ckpt] resumed from {path}  iter={ckpt['iter_num']}")
    return ckpt["iter_num"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def pretrain(config: PretrainConfig):
    torch.manual_seed(config.seed)
    rank, local_rank, world_size, ddp = setup_ddp()
    master = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    pt_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.dtype]
    ctx = (
        autocast(device_type="cuda", dtype=pt_dtype)
        if device.type == "cuda"
        else nullcontext()
    )

    tokenizer = BPETokenizer.load(config.tokenizer_path)
    if master:
        print(f"Tokenizer: vocab_size={tokenizer.vocab_size}")

    train_loader, val_loader = build_dataloaders(config, tokenizer, rank, world_size)

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
    )
    model = TinyLLM(model_config).to(device)
    if config.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    raw = model.module if ddp else model
    decay = [p for _, p in raw.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for _, p in raw.named_parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
    )
    scaler = GradScaler(enabled=(config.dtype == "float16"))

    start_iter = 0
    if config.resume_from and os.path.exists(config.resume_from):
        start_iter = load_checkpoint(config.resume_from, model, optimizer)

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if master:
        eff = config.batch_size * config.grad_accumulation_steps * world_size
        print(
            f"Pre-training  iters={config.max_iters}  eff_batch={eff}  device={device}  dtype={config.dtype}"
        )
        print(f"{'Iter':>8} | {'LR':>10} | {'Loss':>10} | {'PPL':>8} | {'Tok/s':>10}")
        print("─" * 56)

    model.train()
    train_iter = iter(train_loader)
    running_loss, tok_count, best_val = 0.0, 0, float("inf")
    t0 = time.perf_counter()

    for step in range(start_iter, config.max_iters):
        lr = get_lr(
            step, config.warmup_iters, config.max_iters, config.max_lr, config.min_lr
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(config.grad_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            sync = micro == config.grad_accumulation_steps - 1
            cm = model.no_sync() if (ddp and not sync) else nullcontext()
            with cm, ctx:
                _, loss = model(x, targets=y)
            loss = loss / config.grad_accumulation_steps
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        tok_count += (
            config.batch_size
            * config.grad_accumulation_steps
            * config.context_length
            * world_size
        )
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        running_loss += accum_loss

        if master and (step + 1) % config.log_interval == 0:
            dt = time.perf_counter() - t0
            avg = running_loss / config.log_interval
            print(
                f"{step+1:>8} | {lr:>10.2e} | {avg:>10.4f} | {math.exp(min(avg,20)):>8.1f} | {tok_count/dt:>10,.0f}"
            )
            running_loss, tok_count, t0 = 0.0, 0, time.perf_counter()

        if master and (step + 1) % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, config.eval_iters, device, ctx)
            print(
                f"{'VAL':>8} | {lr:>10.2e} | {val_loss:>10.4f} | {math.exp(min(val_loss,20)):>8.1f} |"
            )
            save_checkpoint(
                model,
                optimizer,
                step + 1,
                val_loss,
                config,
                model_config,
                os.path.join(config.checkpoint_dir, "latest.pt"),
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    step + 1,
                    val_loss,
                    config,
                    model_config,
                    os.path.join(config.checkpoint_dir, "best.pt"),
                )
            if (step + 1) % config.checkpoint_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    step + 1,
                    val_loss,
                    config,
                    model_config,
                    os.path.join(config.checkpoint_dir, f"ckpt_{step+1:06d}.pt"),
                )

    if master:
        save_checkpoint(
            model,
            optimizer,
            config.max_iters,
            0.0,
            config,
            model_config,
            os.path.join(config.checkpoint_dir, "pretrain_final.pt"),
        )
        print("Done. Pass pretrain_final.pt to train.py via --resume_from.")

    if ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = PretrainConfig()
    p = argparse.ArgumentParser()
    for k, v in cfg.__dict__.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action="store_true")
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    pretrain(PretrainConfig(**vars(args)))

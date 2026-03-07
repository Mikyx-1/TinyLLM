"""
Pre-Training Stage for TinyLLM

This script handles the pre-training phase: learning language representations
from large, unlabeled text corpora via next-token prediction.

Differences from train.py (fine-tuning / supervised stage):
─────────────────────────────────────────────────────────────
  • Data:   Raw text files (.txt / .jsonl) instead of structured JSON Q&A pairs.
            A custom streaming TextDataset tokenizes on-the-fly to keep RAM low.
  • Task:   Pure next-token prediction — targets = input_ids shifted left by 1.
            No special prompt / response templates.
  • Scale:  Much longer training (tens of thousands of iters), larger LR, bigger
            batches. BPE tokenizer is trained from scratch on the corpus first.
  • Curriculum (optional): can feed data in multiple phases with different
            max_lr / context_length settings (see CurriculumPhase).
  • Monitoring: tracks token throughput and perplexity (exp(loss)) in addition
            to raw cross-entropy.

HOW TO RUN:
    # Train BPE tokenizer first (if you don't have one yet):
    python pretrain.py --train_tokenizer --corpus_dir data/raw_text

    # Single GPU pre-training:
    python pretrain.py --corpus_dir data/raw_text

    # Multi-GPU:
    torchrun --nproc_per_node=4 pretrain.py --corpus_dir data/raw_text

    # Resume from checkpoint:
    python pretrain.py --corpus_dir data/raw_text --resume_from checkpoints/pretrain/latest.pt

    # With custom curriculum:
    python pretrain.py --corpus_dir data/raw_text --use_curriculum
"""

import argparse
import glob
import io
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, IterableDataset

from model import ModelConfig, TinyLLM

# Tokenizer — we use tiktoken (fast BPE) if available, else fall back to a
# tiny hand-rolled character-level tokenizer so the file runs stand-alone.
try:
    import tiktoken

    _HAVE_TIKTOKEN = True
except ImportError:
    _HAVE_TIKTOKEN = False

# ---------------------------------------------------------------------------
# Pre-training Configuration
# ---------------------------------------------------------------------------


@dataclass
class PretrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    corpus_dir: str = "data/raw_text"  # folder with .txt / .jsonl files
    tokenizer_path: str = (
        "tokenizer/bpe.model"  # BPE model (SentencePiece or tiktoken name)
    )
    tokenizer_type: str = "tiktoken"  # "tiktoken" | "char"
    tiktoken_encoding: str = "gpt2"  # tiktoken encoding name (gpt2 / cl100k_base)
    val_fraction: float = 0.02  # fraction of corpus files held out for val
    max_corpus_tokens: int = 0  # 0 = unlimited; set e.g. 1_000_000 for quick tests

    # ── Model ─────────────────────────────────────────────────────────────────
    vocab_size: int = 50_257  # gpt2 tiktoken default; adjust for char-level
    context_length: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.0  # 0 dropout is common in large pre-training runs
    use_learned_pos_emb: bool = True

    # ── Optimisation ──────────────────────────────────────────────────────────
    batch_size: int = 32  # per GPU
    grad_accumulation_steps: int = 8
    max_iters: int = 20_000
    warmup_iters: int = 500
    min_lr: float = 1e-5
    max_lr: float = 6e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # ── Evaluation / Logging ──────────────────────────────────────────────────
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints/pretrain"
    checkpoint_interval: int = 2000
    resume_from: str = ""

    # ── Curriculum (optional multi-phase training) ────────────────────────────
    use_curriculum: bool = False

    # ── System ────────────────────────────────────────────────────────────────
    dtype: str = "bfloat16"
    compile: bool = False
    num_workers: int = 2
    seed: int = 42

    # ── Tokenizer training (run once) ─────────────────────────────────────────
    train_tokenizer: bool = False  # if True, only trains BPE and exits
    tokenizer_vocab_size: int = 8_000  # used when training a new BPE tokenizer


# ---------------------------------------------------------------------------
# Curriculum Phases
# ---------------------------------------------------------------------------


@dataclass
class CurriculumPhase:
    """
    One phase of a curriculum schedule.

    During pre-training it is common to:
      1. Train on short contexts first (cheaper, faster convergence).
      2. Gradually increase context length.
      3. Optionally decrease LR between phases.
    """

    name: str
    max_iters: int
    context_length: int
    max_lr: float
    warmup_iters: int = 200


DEFAULT_CURRICULUM = [
    CurriculumPhase("short_ctx", max_iters=5_000, context_length=64, max_lr=6e-4),
    CurriculumPhase("medium_ctx", max_iters=10_000, context_length=128, max_lr=3e-4),
    CurriculumPhase("full_ctx", max_iters=20_000, context_length=256, max_lr=1e-4),
]


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


class CharTokenizer:
    """
    Minimal character-level tokenizer.
    No external dependencies — useful as a fallback or for debugging.

    Builds vocab from the unique characters seen in a sample of the corpus.
    Special tokens:
        <pad> = 0
        <unk> = 1
        <bos> = 2
        <eos> = 3
    """

    SPECIAL = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    def __init__(self):
        self.char2id: dict[str, int] = dict(self.SPECIAL)
        self.id2char: dict[int, str] = {v: k for k, v in self.SPECIAL.items()}
        self.vocab_size: int = len(self.SPECIAL)

    def build_from_texts(self, texts: list[str]):
        """Collect all unique characters from a list of text strings."""
        chars = sorted(set("".join(texts)))
        for ch in chars:
            if ch not in self.char2id:
                idx = len(self.char2id)
                self.char2id[ch] = idx
                self.id2char[idx] = ch
        self.vocab_size = len(self.char2id)

    def encode(self, text: str) -> list[int]:
        unk = self.SPECIAL["<unk>"]
        return [self.char2id.get(ch, unk) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id2char.get(i, "?") for i in ids)

    def save(self, path: str):
        import json

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.char2id, f)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        import json

        tok = cls()
        with open(path) as f:
            tok.char2id = json.load(f)
        tok.id2char = {v: k for k, v in tok.char2id.items()}
        tok.vocab_size = len(tok.char2id)
        return tok


def build_tokenizer(config: PretrainConfig):
    """
    Return a tokenizer object that has .encode(text) -> list[int]
    and a .vocab_size attribute.
    """
    if config.tokenizer_type == "tiktoken":
        if not _HAVE_TIKTOKEN:
            raise ImportError(
                "tiktoken not installed. Run `pip install tiktoken` or use --tokenizer_type char"
            )
        enc = tiktoken.get_encoding(config.tiktoken_encoding)

        # Wrap so the interface matches CharTokenizer
        class TiktokenWrapper:
            def __init__(self, enc):
                self._enc = enc
                self.vocab_size = enc.n_vocab

            def encode(self, text: str) -> list[int]:
                return self._enc.encode_ordinary(text)

            def decode(self, ids: list[int]) -> str:
                return self._enc.decode(ids)

        return TiktokenWrapper(enc)

    elif config.tokenizer_type == "char":
        tok_path = config.tokenizer_path.replace(".model", "_char.json")
        if os.path.exists(tok_path):
            return CharTokenizer.load(tok_path)
        else:
            raise FileNotFoundError(
                f"Char tokenizer not found at {tok_path}. "
                "Run with --train_tokenizer first."
            )
    else:
        raise ValueError(f"Unknown tokenizer_type: {config.tokenizer_type!r}")


def train_char_tokenizer(config: PretrainConfig):
    """Build and save a character-level tokenizer from the corpus."""
    print("Training char tokenizer...")
    files = _list_corpus_files(config.corpus_dir)
    texts = []
    for fp in files[:500]:  # sample up to 500 files to build vocab
        try:
            texts.append(Path(fp).read_text(errors="replace")[:10_000])
        except Exception:
            continue
    tok = CharTokenizer()
    tok.build_from_texts(texts)
    save_path = config.tokenizer_path.replace(".model", "_char.json")
    tok.save(save_path)
    print(f"Char tokenizer saved to {save_path} (vocab_size={tok.vocab_size})")
    return tok


# ---------------------------------------------------------------------------
# Dataset — streaming raw text
# ---------------------------------------------------------------------------


def _list_corpus_files(corpus_dir: str) -> list[str]:
    """Collect all .txt and .jsonl files under corpus_dir."""
    files = glob.glob(
        os.path.join(corpus_dir, "**", "*.txt"), recursive=True
    ) + glob.glob(os.path.join(corpus_dir, "**", "*.jsonl"), recursive=True)
    files.sort()
    return files


def _stream_tokens(
    files: list[str],
    tokenizer,
    context_length: int,
    max_tokens: int = 0,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[torch.Tensor]:
    """
    Generator that yields packed (context_length + 1,) int64 tensors.

    Packing strategy:
    ─────────────────
    We tokenize each document and append tokens to a rolling buffer.
    When the buffer reaches (context_length + 1) tokens we yield a chunk.
    The +1 is so input = chunk[:-1] and target = chunk[1:] (next-token prediction).

    No padding — documents are concatenated end-to-end (separated by eos token).
    This is the standard approach for large-scale pre-training.
    """
    eos_id = getattr(tokenizer, "eos_id", 3)  # CharTokenizer uses 3; tiktoken has eot
    buffer: list[int] = []
    total_tokens = 0
    rng = random.Random(seed)

    file_order = list(files)
    if shuffle:
        rng.shuffle(file_order)

    chunk_size = context_length + 1  # input + target

    for fp in file_order:
        try:
            text = Path(fp).read_text(errors="replace")
        except Exception:
            continue

        if fp.endswith(".jsonl"):
            # Each line is a JSON object with a "text" field
            import json

            lines = text.splitlines()
            text = "\n".join(
                json.loads(line).get("text", "") for line in lines if line.strip()
            )

        ids = tokenizer.encode(text)
        ids.append(eos_id)
        buffer.extend(ids)

        # Yield as many complete chunks as we can
        while len(buffer) >= chunk_size:
            chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            yield torch.tensor(chunk, dtype=torch.long)
            total_tokens += chunk_size
            if max_tokens > 0 and total_tokens >= max_tokens:
                return

    # Yield the remaining partial chunk (zero-padded if needed)
    if len(buffer) > 1:
        chunk = buffer[:chunk_size]
        if len(chunk) < chunk_size:
            chunk += [0] * (chunk_size - len(chunk))
        yield torch.tensor(chunk, dtype=torch.long)


class PretrainDataset(IterableDataset):
    """
    Streaming IterableDataset over a list of text files.

    Why IterableDataset instead of a regular Dataset?
    ──────────────────────────────────────────────────
    Pre-training corpora are often hundreds of GB. Loading them into memory
    (as a regular Dataset does) is impractical. IterableDataset lets us
    process one file at a time and stream chunks to the GPU.

    DDP note: when using DDP, each worker rank should see a disjoint shard
    of the files. We handle this by splitting file_list by rank here.
    """

    def __init__(
        self,
        files: list[str],
        tokenizer,
        context_length: int,
        max_tokens: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        # Shard files across DDP ranks so each GPU sees unique data
        self.files = [f for i, f in enumerate(files) if i % world_size == rank]
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.files

        # Further shard across DataLoader workers within a rank
        if worker_info is not None:
            files = [
                f
                for i, f in enumerate(files)
                if i % worker_info.num_workers == worker_info.id
            ]

        yield from _stream_tokens(
            files,
            self.tokenizer,
            self.context_length,
            max_tokens=self.max_tokens,
            shuffle=self.shuffle,
            seed=self.seed + (worker_info.id if worker_info else 0),
        )


def collate_fn(batch: list[torch.Tensor]):
    """
    Stack a list of (context_length+1,) tensors into (B, context_length+1),
    then split into input ids and targets (shifted by 1).

        input:  tokens[0 : context_length]
        target: tokens[1 : context_length+1]
    """
    stacked = torch.stack(batch, dim=0)  # (B, T+1)
    x = stacked[:, :-1].contiguous()  # (B, T)
    y = stacked[:, 1:].contiguous()  # (B, T)
    return x, y


def build_dataloaders(
    config: PretrainConfig,
    tokenizer,
    rank: int,
    world_size: int,
):
    """Split corpus files into train/val and return DataLoaders."""
    all_files = _list_corpus_files(config.corpus_dir)
    if not all_files:
        raise FileNotFoundError(
            f"No .txt or .jsonl files found in {config.corpus_dir!r}. "
            "Create the directory and add text files, or change --corpus_dir."
        )

    random.seed(config.seed)
    random.shuffle(all_files)

    if len(all_files) == 1:
        # Only one file — use it for both train and val.
        # val will naturally see a different (later) slice because
        # max_corpus_tokens caps the train stream; val reads the full file.
        train_files = all_files
        val_files = all_files
    else:
        n_val = max(1, int(len(all_files) * config.val_fraction))
        val_files = all_files[:n_val]
        train_files = all_files[n_val:]

    train_ds = PretrainDataset(
        train_files,
        tokenizer,
        config.context_length,
        max_tokens=config.max_corpus_tokens,
        shuffle=True,
        seed=config.seed,
        rank=rank,
        world_size=world_size,
    )
    # Validation: rank 0 evaluates all val files (no sharding)
    val_ds = PretrainDataset(
        val_files,
        tokenizer,
        config.context_length,
        max_tokens=0,
        shuffle=False,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # val is only on master; keep simple
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader, len(train_files), len(val_files)


# ---------------------------------------------------------------------------
# Learning Rate Schedule (same cosine+warmup as train.py)
# ---------------------------------------------------------------------------


def get_lr(
    iter_num: int,
    warmup_iters: int,
    max_iters: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if iter_num < warmup_iters:
        return max_lr * (iter_num + 1) / warmup_iters
    progress = (iter_num - warmup_iters) / max(1, max_iters - warmup_iters)
    progress = min(progress, 1.0)
    return min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# DDP helpers (same pattern as train.py)
# ---------------------------------------------------------------------------


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    rank, local_rank, world_size = 0, 0, 1
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    return rank, local_rank, world_size, False


def cleanup_ddp(ddp: bool):
    if ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, val_loader, eval_iters: int, device, ctx) -> float:
    model.eval()
    total, count = 0.0, 0
    val_iter = iter(val_loader)
    for _ in range(eval_iters):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, targets=y)
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint(model, optimizer, iter_num, val_loss, config, model_config, path):
    raw = model.module if isinstance(model, DDP) else model
    ckpt = {
        "model_state_dict": raw.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter_num": iter_num,
        "val_loss": val_loss,
        "pretrain_config": config.__dict__,
        "model_config": model_config.__dict__,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ckpt, path)
    print(
        f"  [ckpt] saved {path}  (iter={iter_num}, val_loss={val_loss:.4f}, ppl={math.exp(val_loss):.1f})"
    )


def load_checkpoint(path, model, optimizer=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    raw = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    iter_num = ckpt.get("iter_num", 0)
    print(f"  [ckpt] resumed from {path} (iter={iter_num})")
    return iter_num


# ---------------------------------------------------------------------------
# Main Pre-Training Loop
# ---------------------------------------------------------------------------


def pretrain(config: PretrainConfig):
    torch.manual_seed(config.seed)
    rank, local_rank, world_size, ddp = setup_ddp()
    master = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if master:
        print("=" * 64)
        print("TinyLLM — Pre-Training Stage")
        print("=" * 64)
        print(f"  GPUs         : {world_size}")
        print(f"  Device       : {device}")
        print(f"  dtype        : {config.dtype}")
        eff_batch = config.batch_size * config.grad_accumulation_steps * world_size
        print(
            f"  Eff. batch   : {eff_batch} sequences  "
            f"= {eff_batch * config.context_length:,} tokens/step"
        )
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Mixed precision ────────────────────────────────────────────────────────
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

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if master:
        print("\nLoading tokenizer...")
    tokenizer = build_tokenizer(config)
    # Override vocab_size with the actual tokenizer vocab_size
    actual_vocab = tokenizer.vocab_size
    if master:
        print(f"  vocab_size = {actual_vocab}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if master:
        print("\nBuilding dataloaders...")
    train_loader, val_loader, n_train, n_val = build_dataloaders(
        config, tokenizer, rank, world_size
    )
    if master:
        print(f"  Train files : {n_train}")
        print(f"  Val files   : {n_val}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_config = ModelConfig(
        vocab_size=actual_vocab,
        context_length=config.context_length,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        use_learned_pos_emb=config.use_learned_pos_emb,
    )
    if master:
        print(f"\nModel config: {model_config}")

    model = TinyLLM(model_config).to(device)

    if config.compile:
        if master:
            print("torch.compile()...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Split parameters: weight matrices get weight decay, biases / norms don't.
    raw_model = model.module if ddp else model
    decay, no_decay = [], []
    for name, p in raw_model.named_parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)

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

    # ── Resume ────────────────────────────────────────────────────────────────
    start_iter = 0
    if config.resume_from and os.path.exists(config.resume_from):
        start_iter = load_checkpoint(config.resume_from, model, optimizer)

    # ── Curriculum setup ──────────────────────────────────────────────────────
    # If curriculum is enabled, determine which phase we're in.
    phases = DEFAULT_CURRICULUM if config.use_curriculum else None

    def _get_phase_params(iter_num: int):
        """Return (max_iters, warmup_iters, max_lr, min_lr) for the current phase."""
        if phases is None:
            return config.max_iters, config.warmup_iters, config.max_lr, config.min_lr
        cumulative = 0
        for phase in phases:
            if iter_num < cumulative + phase.max_iters:
                phase_iter = iter_num - cumulative  # relative iter within phase
                return (
                    phase.max_iters,
                    phase.warmup_iters,
                    phase.max_lr,
                    config.min_lr,
                    phase_iter,
                    phase.name,
                )
            cumulative += phase.max_iters
        # Past all phases — stay at last phase settings
        last = phases[-1]
        return (
            last.max_iters,
            last.warmup_iters,
            last.max_lr,
            config.min_lr,
            last.max_iters - 1,
            last.name,
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    total_iters = sum(p.max_iters for p in phases) if phases else config.max_iters

    if master:
        print(f"\nStarting pre-training for {total_iters} iterations...")
        print(
            f"{'Iter':>8} | {'Phase':>12} | {'LR':>10} | {'Loss':>10} | {'PPL':>8} | {'Tok/s':>10} | {'Time':>7}"
        )
        print("─" * 76)

    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    t_start = time.perf_counter()
    tokens_since_log = 0
    best_val_loss = float("inf")

    for iter_num in range(start_iter, total_iters):

        # ── Determine LR from schedule / curriculum ────────────────────────
        if phases:
            (
                phase_max_iters,
                phase_warmup,
                phase_max_lr,
                phase_min_lr,
                phase_iter,
                phase_name,
            ) = _get_phase_params(iter_num)
            lr = get_lr(
                phase_iter, phase_warmup, phase_max_iters, phase_max_lr, phase_min_lr
            )
        else:
            lr = get_lr(
                iter_num, config.warmup_iters, total_iters, config.max_lr, config.min_lr
            )
            phase_name = "pretrain"

        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Gradient accumulation ──────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(config.grad_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                # Dataset exhausted — restart from the beginning (multi-epoch)
                train_iter = iter(train_loader)
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    raise RuntimeError(
                        "Training dataloader is empty after reset. "
                        "The corpus may be too small to fill even one batch. "
                        "Check that your .jsonl file has content and that "
                        "context_length is not larger than your documents."
                    )

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            sync = micro_step == config.grad_accumulation_steps - 1
            cm = model.no_sync() if (ddp and not sync) else nullcontext()
            with cm, ctx:
                _, loss = model(x, targets=y)

            loss = loss / config.grad_accumulation_steps
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        tokens_since_log += (
            config.batch_size
            * config.grad_accumulation_steps
            * config.context_length
            * world_size
        )

        # ── Gradient clipping + optimizer step ────────────────────────────
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += accum_loss

        # ── Logging ───────────────────────────────────────────────────────
        if master and (iter_num + 1) % config.log_interval == 0:
            t_now = time.perf_counter()
            dt = t_now - t_start
            tok_per_sec = tokens_since_log / dt
            avg_loss = running_loss / config.log_interval
            ppl = math.exp(min(avg_loss, 20))  # cap to avoid overflow display
            print(
                f"{iter_num+1:>8} | {phase_name:>12} | {lr:>10.2e} | "
                f"{avg_loss:>10.4f} | {ppl:>8.1f} | {tok_per_sec:>10,.0f} | {dt:>6.1f}s"
            )
            running_loss = 0.0
            tokens_since_log = 0
            t_start = time.perf_counter()

        # ── Evaluation + checkpointing ─────────────────────────────────────
        if master and (iter_num + 1) % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, config.eval_iters, device, ctx)
            val_ppl = math.exp(min(val_loss, 20))
            print(
                f"{'>>> VAL':>8} | {phase_name:>12} | {lr:>10.2e} | "
                f"{val_loss:>10.4f} | {val_ppl:>8.1f} |"
            )

            # Save latest checkpoint
            latest_path = os.path.join(config.checkpoint_dir, "latest.pt")
            save_checkpoint(
                model,
                optimizer,
                iter_num + 1,
                val_loss,
                config,
                model_config,
                latest_path,
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(config.checkpoint_dir, "best.pt")
                save_checkpoint(
                    model,
                    optimizer,
                    iter_num + 1,
                    val_loss,
                    config,
                    model_config,
                    best_path,
                )
                print(f"  [best] new best val_loss={val_loss:.4f}")

            # Periodic numbered checkpoint
            if (iter_num + 1) % config.checkpoint_interval == 0:
                numbered = os.path.join(
                    config.checkpoint_dir, f"pretrain_{iter_num+1:06d}.pt"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    iter_num + 1,
                    val_loss,
                    config,
                    model_config,
                    numbered,
                )

    # ── Final save ────────────────────────────────────────────────────────────
    if master:
        final_path = os.path.join(config.checkpoint_dir, "pretrain_final.pt")
        save_checkpoint(
            model, optimizer, total_iters, 0.0, config, model_config, final_path
        )
        print(f"\nPre-training complete! Weights saved to {config.checkpoint_dir}/")
        print(
            "Next step: pass pretrain_final.pt to train.py via --resume_from for fine-tuning."
        )

    cleanup_ddp(ddp)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def parse_args():
    cfg = PretrainConfig()
    parser = argparse.ArgumentParser(description="Pre-train TinyLLM on raw text")
    for key, val in cfg.__dict__.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", default=val, action="store_true")
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = PretrainConfig(**vars(args))

    if config.train_tokenizer:
        # ── Tokenizer training mode ─────────────────────────────────────────
        if config.tokenizer_type == "char":
            train_char_tokenizer(config)
        else:
            print("tiktoken encodings are pre-built — no training needed.")
            print(f"Using encoding: {config.tiktoken_encoding}")
        print("Done. Re-run without --train_tokenizer to start pre-training.")
    else:
        pretrain(config)

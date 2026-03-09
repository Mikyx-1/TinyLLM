"""
Multi-GPU Training with PyTorch DDP (Distributed Data Parallel)

This is the main training script. It implements:

1. DDP Setup:
   - Each GPU runs its own process
   - Each process holds a full copy of the model
   - Forward/backward pass happens on each GPU with its own data shard
   - Gradients are all-reduced (averaged) across GPUs automatically
   - This gives ~linear speedup with number of GPUs

2. Learning Rate Schedule:
   - Linear warmup: prevents instability at the start of training
   - Cosine decay: smoothly decreases LR to minimum over training

3. Gradient Clipping:
   - Clips gradient norm to prevent exploding gradients
   - Essential for transformers

4. Mixed Precision Training (bfloat16/float16):
   - Reduces memory usage by ~50%
   - Speeds up matrix multiplications on modern GPUs
   - Automatic loss scaling handles underflow

5. Gradient Accumulation:
   - Simulates larger batch sizes without more memory
   - Instead of batch_size=64, do 8 steps of batch_size=8

HOW TO RUN:
    Single GPU:
        python train.py

    Multi-GPU (2 GPUs):
        torchrun --nproc_per_node=2 train.py

    With custom config:
        torchrun --nproc_per_node=2 train.py --batch_size 16 --max_iters 5000
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from data_utils import create_dataloader, prepare_custom_data
from model import ModelConfig, TinyLLM

# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Data
    dataset_path: str = "data/tinyllm_dataset.json"
    data_dir: str = "data"
    vocab_size: int = 8000
    context_length: int = 64
    val_fraction: float = 0.1

    # Model (kept small for learning/fast iteration)
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    use_learned_pos_emb: bool = True

    # Training
    batch_size: int = 32  # per GPU
    grad_accumulation_steps: int = (
        4  # effective batch = batch_size * grad_accum * world_size
    )
    max_iters: int = 5000
    warmup_iters: int = 200
    min_lr: float = 1e-5
    max_lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Evaluation
    eval_interval: int = 200
    eval_iters: int = 50  # batches to average for val loss
    log_interval: int = 10

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    resume_from: str = ""  # path to checkpoint to resume from

    # System
    dtype: str = "bfloat16"  # float32, float16, bfloat16
    compile: bool = False  # torch.compile (PyTorch 2.0+, faster but slower startup)
    num_workers: int = 4


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------


def get_lr(iter_num: int, config: TrainConfig) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Phase 1 (iter < warmup_iters): linear warmup from 0 to max_lr
    Phase 2 (iter >= warmup_iters): cosine decay from max_lr to min_lr

    Warmup prevents large gradient updates at the start when weights
    are random and loss landscape is chaotic.
    """
    if iter_num < config.warmup_iters:
        # Linear warmup
        return config.max_lr * (iter_num + 1) / config.warmup_iters

    # Cosine decay
    progress = (iter_num - config.warmup_iters) / (
        config.max_iters - config.warmup_iters
    )
    progress = min(progress, 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + (config.max_lr - config.min_lr) * cosine_decay


# ---------------------------------------------------------------------------
# DDP Utilities
# ---------------------------------------------------------------------------


def setup_ddp():
    """
    Initialize the distributed process group.

    When launched with torchrun:
    - RANK: global process rank (0 = master)
    - LOCAL_RANK: rank on this node (maps to GPU index)
    - WORLD_SIZE: total number of processes (= total GPUs)

    When running single-GPU (python train.py):
    - These env vars aren't set, so we use rank=0, world_size=1
    """
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")  # NCCL is fastest for GPU-GPU comms
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        # Single GPU / CPU
        rank, local_rank, world_size = 0, 0, 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return rank, local_rank, world_size, False


def cleanup_ddp(ddp: bool):
    if ddp:
        dist.destroy_process_group()


def is_master(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    eval_iters: int,
    device: torch.device,
    ctx,
) -> float:
    """
    Estimate validation loss by averaging over eval_iters batches.
    Uses @torch.no_grad() to skip gradient computation (saves memory/time).
    """
    model.eval()
    total_loss = 0.0
    count = 0

    val_iter = iter(val_loader)
    for _ in range(eval_iters):
        try:
            x, y = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            x, y = next(val_iter)

        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, targets=y)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / count


# ---------------------------------------------------------------------------
# Checkpoint Utilities
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    val_loss: float,
    config: TrainConfig,
    model_config: ModelConfig,
    path: str,
):
    """Save training state to disk."""
    # Unwrap DDP if necessary
    raw_model = model.module if isinstance(model, DDP) else model
    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter_num": iter_num,
        "val_loss": val_loss,
        "train_config": config.__dict__,
        "model_config": model_config.__dict__,
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path} (iter={iter_num}, val_loss={val_loss:.4f})")


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    """Load checkpoint and return iteration number and saved model config."""
    checkpoint = torch.load(path, map_location="cpu")
    raw_model = model.module if isinstance(model, DDP) else model

    # Handle checkpoints saved from torch.compile() — strip "_orig_mod." prefix
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    raw_model.load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Resumed from checkpoint: {path} (iter={checkpoint['iter_num']})")
    return checkpoint["iter_num"], checkpoint.get("model_config")


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------


def train(config: TrainConfig):
    # --- Setup DDP ---
    rank, local_rank, world_size, ddp = setup_ddp()
    master = is_master(rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if master:
        print("=" * 60)
        print("TinyLLM Training")
        print("=" * 60)
        print(f"World size: {world_size} GPU(s)")
        print(f"Device: {device}")
        print(f"dtype: {config.dtype}")
        print(
            f"Effective batch size: {config.batch_size * config.grad_accumulation_steps * world_size}"
        )
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # --- Mixed precision context ---
    # bfloat16 is preferred on Ampere+ (RTX 3060 is Ampere ✓)
    # It has the same dynamic range as float32 (no loss scaling needed)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    pt_dtype = dtype_map[config.dtype]
    if device.type == "cuda":
        ctx = autocast(device_type="cuda", dtype=pt_dtype)
    else:
        ctx = nullcontext()

    # --- Data ---
    if master:
        print("\nPreparing data...")

    train_ds, val_ds, tokenizer = prepare_custom_data(
        json_path=config.dataset_path,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        val_fraction=config.val_fraction,
    )

    train_loader = create_dataloader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        distributed=ddp,
        rank=rank,
        world_size=world_size,
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        distributed=False,  # Only master evaluates
    )

    # --- Model ---
    # If resuming, use the checkpoint's saved model_config to ensure architecture match
    if config.resume_from and os.path.exists(config.resume_from):
        ckpt_meta = torch.load(config.resume_from, map_location="cpu")
        saved_cfg = ckpt_meta.get("model_config", {})
        # Remove computed fields that aren't constructor parameters
        saved_cfg.pop("d_k", None)
        model_config = ModelConfig(**saved_cfg)
        if master:
            print(f"\nModel config (from checkpoint): {model_config}")
    else:
        model_config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
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

    # Optional: torch.compile for ~20% speedup (requires PyTorch 2.0+)
    if config.compile:
        if master:
            print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Wrap in DDP (handles gradient sync automatically)
    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # --- Optimizer ---
    # Use AdamW with weight decay on weight matrices (not biases/layernorms)
    # This is the standard optimizer for transformer LMs
    decay_params = []
    no_decay_params = []
    raw_model = model.module if ddp else model
    for name, param in raw_model.named_parameters():
        if param.requires_grad:
            if param.ndim >= 2:
                decay_params.append(param)  # weight matrices
            else:
                no_decay_params.append(param)  # biases, layernorm

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
    )

    # Gradient scaler for float16 (not needed for bfloat16)
    scaler = GradScaler(enabled=(config.dtype == "float16"))

    # --- Resume from checkpoint ---
    start_iter = 0
    if config.resume_from and os.path.exists(config.resume_from):
        start_iter, _ = load_checkpoint(config.resume_from, model, optimizer)

    # --- Training loop ---
    if master:
        print(f"\nStarting training for {config.max_iters} iterations...")
        print(
            f"{'Iter':>8} | {'LR':>10} | {'Train Loss':>12} | {'Val Loss':>10} | {'Tokens/s':>10} | {'Time':>8}"
        )
        print("-" * 70)

    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    t0 = time.time()

    for iter_num in range(start_iter, config.max_iters):
        # --- Update learning rate ---
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Gradient accumulation ---
        # Accumulate gradients over multiple micro-batches before stepping optimizer
        # This simulates a larger batch size without more memory
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(config.grad_accumulation_steps):
            # Get next batch
            try:
                x, y = next(train_iter)
            except StopIteration:
                # Reset sampler for new epoch (important for DDP)
                if ddp and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(iter_num)
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # With DDP, only sync gradients on the last accumulation step
            # This avoids expensive all-reduce on every micro-step
            sync_gradients = micro_step == config.grad_accumulation_steps - 1
            context = model.no_sync() if (ddp and not sync_gradients) else nullcontext()

            with context:
                with ctx:
                    _, loss = model(x, targets=y)
                # Scale loss by accumulation steps so gradients have the right magnitude
                loss = loss / config.grad_accumulation_steps
                accum_loss += loss.item()
                scaler.scale(loss).backward()

        # --- Gradient clipping ---
        # Clips the global gradient norm to prevent exploding gradients
        # Essential for transformers — without this, training can diverge suddenly
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # --- Optimizer step ---
        scaler.step(optimizer)
        scaler.update()

        running_loss += accum_loss

        # --- Logging ---
        if master and (iter_num + 1) % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = (
                config.batch_size
                * config.grad_accumulation_steps
                * config.context_length
                * config.log_interval
                * world_size
                / dt
            )
            avg_loss = running_loss / config.log_interval
            print(
                f"{iter_num+1:>8} | {lr:>10.2e} | {avg_loss:>12.4f} | "
                f"{'':>10} | {tokens_per_sec:>10,.0f} | {dt:>7.1f}s"
            )
            running_loss = 0.0
            t0 = time.time()

        # --- Evaluation ---
        if master and (iter_num + 1) % config.eval_interval == 0:
            val_loss = evaluate(model, val_loader, config.eval_iters, device, ctx)
            print(f"{'>>> VAL':>8} | {lr:>10.2e} | {'':>12} | {val_loss:>10.4f} |")

            # Save checkpoint
            if (iter_num + 1) % config.checkpoint_interval == 0:
                ckpt_path = os.path.join(
                    config.checkpoint_dir, f"checkpoint_{iter_num+1:05d}.pt"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    iter_num + 1,
                    val_loss,
                    config,
                    model_config,
                    ckpt_path,
                )

            # Also save "best" and "latest"
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

    if master:
        print("\nTraining complete!")
        # Save final model
        final_path = os.path.join(config.checkpoint_dir, "final.pt")
        save_checkpoint(
            model, optimizer, config.max_iters, 0.0, config, model_config, final_path
        )

    cleanup_ddp(ddp)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyLLM")
    cfg = TrainConfig()
    # Add all config fields as CLI arguments
    for key, val in cfg.__dict__.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", default=val, action="store_true")
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(**vars(args))
    train(config)

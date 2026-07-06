"""
Stage 3 — Reasoning SFT.

Mirrors train.py's structure/loop (LR schedule, AdamW param groups, grad clip/accum,
DDP, checkpointing) verbatim, per this repo's convention of one script per pipeline
stage rather than a shared library. The two real differences from train.py:

1. Data comes from data_utils.prepare_reasoning_data(), which holds out whole
   problems *before* concatenation (not a token-position val_fraction cut) -- the
   point of this stage is measuring generalization to problems never seen at all, not
   loss on a slice of the training stream. Held-out examples aren't touched during
   training.
2. There is no loss-based validation pass (checkpointing tracks train loss, same as
   train.py's no-val path). Instead, every eval_interval iterations this script runs
   real generation on a sample of the held-out set via eval_reasoning.run_examples/
   summarize, logging exact-match accuracy (overall and per-hop, when examples carry a
   "hops" field) and the <CALC>-count match rate to wandb/console -- plain train loss
   alone doesn't reveal failure modes like template-locking onto a fixed number of
   reasoning steps.

HOW TO RUN:
    python train_reasoning.py
    python train_reasoning.py --max_iters 3000 --checkpoint_dir checkpoints/reasoning

    With Weights & Biases logging:
        python train_reasoning.py --use_wandb --wandb_project my-project
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

from data_utils import create_dataloader, prepare_multitask_data, prepare_reasoning_data
from eval_reasoning import print_summary, resolve_calc_ids, run_examples, summarize
from model import ModelConfig, TinyLLM

try:
    import wandb
except ImportError:
    wandb = None

# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Data
    dataset_path: str = "data/synthetic_reasoning_all_hops.json"  # 1-, 2-, and 3-hop
    # combined -- a 3-hop-only run template-locked onto always emitting exactly 3
    # <CALC> steps and hallucinated a spurious extra operation on 2-hop questions.

    # "legacy": prepare_reasoning_data -- reasoning is the primary corpus, rendered via
    # the flat, unmasked REASONING_TEMPLATE; replay_dataset_path's chat examples are
    # mixed in unmasked too, purely so the reasoning run doesn't forget chat.
    # "chatml": prepare_multitask_data -- chat (multiturn_dataset_path) and reasoning
    # are pooled and shuffled together as EQUALS, both rendered through the same
    # <|im_start|>/<|im_end|> ChatML turns with the loss masked to assistant spans only
    # (see data_utils.reasoning_example_to_conversation). The model can't tell the two
    # task types apart by template, only by content -- the actual objective here is
    # "answer both well," not "don't forget chat while learning reasoning."
    # replay_dataset_path/replay_count below are ignored in "chatml" mode.
    dataset_format: str = "legacy"
    multiturn_dataset_path: str = "data/smalltalk_multiturn.json"  # only used when
    # dataset_format="chatml"

    tokenizer_path: str = "checkpoints/tokenizer.json"
    vocab_size: int = 12_002  # same tokenizer as every other stage (resized for <CALC>)
    context_length: int = 384
    held_out: int = 450  # random slice of the combined pool; lands roughly proportionally
    # across all three depths (~41%/25%/34% by construction), large enough for a
    # meaningful per-hop breakdown in eval_reasoning.py
    heldout_out_path: str = "data/reasoning_heldout.json"
    seed: int = 42

    # Mix the *entire* Q&A dataset in among the reasoning examples -- not a sample. See
    # data_utils.prepare_reasoning_data's docstring -- a reasoning-only fine-tune from an
    # SFT checkpoint catastrophically overwrote that checkpoint's Q&A ability (confirmed
    # directly), and a later run that replayed only a 2000/20061 sample of the generic
    # Alpaca/Dolly-heavy SFT set still left ~90% of it totally unseen (confirmed too).
    # replay_dataset_path now points at the curated small-talk set (single-turn +
    # combinatorial 2-turn, data/smalltalk_demo.json) instead of the generic SFT set --
    # replacing Alpaca/Dolly content with small talk was an explicit, separate decision,
    # not something to fold in silently alongside the reasoning-data changes below.
    # replay_count is set above the actual dataset size; prepare_reasoning_data clamps to
    # len(replay_pool), so this always means "every example," resilient to the file's
    # size changing later.
    replay_dataset_path: str = "data/smalltalk_demo.json"
    replay_count: int = 1_000_000

    # Model architecture (d_model/n_heads/n_layers/d_ff/use_learned_pos_emb only take
    # effect if NOT resuming from a checkpoint; dropout always applies -- see resume_from
    # below, it has no learned parameters so it's always safe to override on resume).
    # ~50.4M params at vocab_size=12002, context_length=384 (vs. 23.15M in every earlier
    # stage) -- no pretraining step for this run, so there's no existing checkpoint this
    # size to inherit; it trains from random init directly on the mixed corpus below.
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 8
    d_ff: int = 1792
    dropout: float = 0.0  # explicitly back to 0 -- this run's goal is deliberate
    # memorization of the combined small-talk + reasoning corpus (like Stage 2's
    # original SFT run), not generalization to unseen problems, so there's no reason
    # to fight overfitting here. A previous run raised this to 0.2 specifically to
    # slow memorization down for a held-out-accuracy measurement; that's not this run.
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
    weight_decay: float = 0.0  # explicitly back to 0, same reasoning as dropout above --
    # deliberate memorization this run, not a generalization measurement.
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Logging
    log_interval: int = 10

    # Generation-based eval (real held-out generation via eval_reasoning.run_examples,
    # not the training loss): plain train loss is a token-average and, as found while
    # using this exact pipeline, hides a lot -- e.g. a model can reach near-zero loss
    # while template-locking onto always emitting exactly 3 <CALC> steps regardless of
    # what a given problem needs. eval_interval is separate from (and much coarser
    # than) checkpoint_interval because generation is far more expensive than a forward
    # pass; eval_sample_size caps the held-out slice actually generated on per eval so
    # this doesn't dominate wall-clock time.
    eval_interval: int = 1000
    eval_sample_size: int = 80

    # Checkpointing
    checkpoint_dir: str = "checkpoints/reasoning_smalltalk_50M"
    checkpoint_interval: int = 500
    resume_from: str = ""  # no pretraining for this run -- trains from random init
    resume_weights_only: bool = True  # irrelevant while resume_from is empty; kept for
    # parity with train.py/other stages in case this is pointed at a checkpoint later

    # System
    dtype: str = "bfloat16"  # float32, float16, bfloat16
    compile: bool = False  # torch.compile (PyTorch 2.0+, faster but slower startup)
    num_workers: int = 4

    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "tinyllm-reasoning"
    wandb_run_name: str = ""


# ---------------------------------------------------------------------------
# Learning Rate Schedule
# ---------------------------------------------------------------------------


def get_lr(iter_num: int, config: TrainConfig) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    Phase 1 (iter < warmup_iters): linear warmup from 0 to max_lr
    Phase 2 (iter >= warmup_iters): cosine decay from max_lr to min_lr
    """
    if iter_num < config.warmup_iters:
        return config.max_lr * (iter_num + 1) / config.warmup_iters

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
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
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
# Checkpoint Utilities (identical to train.py's)
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iter_num: int,
    train_loss: float,
    config: TrainConfig,
    model_config: ModelConfig,
    path: str,
):
    """Save training state to disk."""
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model
    checkpoint = {
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter_num": iter_num,
        "val_loss": train_loss,  # field name kept for compatibility with load_checkpoint
        "train_config": config.__dict__,
        "model_config": model_config.__dict__,
    }
    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path} (iter={iter_num}, train_loss={train_loss:.4f})")


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    """Load checkpoint and return iteration number and saved model config."""
    checkpoint = torch.load(path, map_location="cpu")
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model = raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model

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
        print("TinyLLM Reasoning SFT (Stage 3)")
        print("=" * 60)
        print(f"World size: {world_size} GPU(s)")
        print(f"Device: {device}")
        print(f"dtype: {config.dtype}")
        print(
            f"Effective batch size: {config.batch_size * config.grad_accumulation_steps * world_size}"
        )
        os.makedirs(config.checkpoint_dir, exist_ok=True)

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
        print(f"\nPreparing data (dataset_format={config.dataset_format})...")

    if config.dataset_format == "chatml":
        train_ds, held_out_examples, tokenizer = prepare_multitask_data(
            multiturn_path=config.multiturn_dataset_path,
            reasoning_path=config.dataset_path,
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            held_out=config.held_out,
            seed=config.seed,
            tokenizer_path=config.tokenizer_path,
            heldout_out_path=config.heldout_out_path,
        )
    else:
        train_ds, held_out_examples, tokenizer = prepare_reasoning_data(
            json_path=config.dataset_path,
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            held_out=config.held_out,
            seed=config.seed,
            tokenizer_path=config.tokenizer_path,
            heldout_out_path=config.heldout_out_path,
            replay_dataset_path=config.replay_dataset_path,
            replay_count=config.replay_count,
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

    # --- Model ---
    # If resuming, use the checkpoint's saved model_config to ensure architecture match
    # (this also picks up the resized vocab_size=12002 from vocab_surgery, needed for <CALC>).
    if config.resume_from and os.path.exists(config.resume_from):
        ckpt_meta = torch.load(config.resume_from, map_location="cpu")
        saved_cfg = ckpt_meta.get("model_config", {})
        saved_cfg.pop("d_k", None)
        # Architecture (vocab_size, d_model, etc.) must match the checkpoint for weight
        # loading to work, but dropout has no learned parameters -- it's always safe to
        # use *this* run's value instead of silently inheriting the source checkpoint's
        # (which is 0.0, since Stage 2 deliberately trained with no dropout to memorize).
        saved_cfg["dropout"] = config.dropout
        model_config = ModelConfig(**saved_cfg)
        if master:
            print(f"\nModel config (from checkpoint, dropout overridden to {config.dropout}): {model_config}")
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

    if config.compile:
        if master:
            print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[local_rank])

    # --- Optimizer ---
    decay_params = []
    no_decay_params = []
    raw_model = model.module if ddp else model
    for name, param in raw_model.named_parameters():
        if param.requires_grad:
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=1e-8,
    )

    scaler = GradScaler(enabled=(config.dtype == "float16"))

    # --- Resume from checkpoint ---
    start_iter = 0
    if config.resume_from and os.path.exists(config.resume_from):
        if config.resume_weights_only:
            loaded_iter, _ = load_checkpoint(config.resume_from, model, optimizer=None)
            if master:
                print(
                    f"  (weights only — starting fresh at iter 0, ignoring source iter={loaded_iter})"
                )
        else:
            start_iter, _ = load_checkpoint(config.resume_from, model, optimizer)

    if master and config.use_wandb:
        if wandb is None:
            raise RuntimeError("use_wandb=True but the `wandb` package is not installed. Run `pip install wandb`.")
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or None,
            config={**config.__dict__, **model_config.__dict__},
        )
        wandb.watch(raw_model, log="gradients", log_freq=config.log_interval * 10)

    # --- Training loop ---
    if master:
        print(f"\nStarting training for {config.max_iters} iterations...")
        print(f"{'Iter':>8} | {'LR':>10} | {'Train Loss':>12} | {'Tokens/s':>10} | {'Time':>8}")
        print("-" * 60)

    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    best_loss = float("inf")
    t0 = time.time()

    for iter_num in range(start_iter, config.max_iters):
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(config.grad_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                if ddp and hasattr(train_loader.sampler, "set_epoch"):
                    train_loader.sampler.set_epoch(iter_num)
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            sync_gradients = micro_step == config.grad_accumulation_steps - 1
            context = model.no_sync() if (ddp and not sync_gradients) else nullcontext()

            with context:
                with ctx:
                    _, loss = model(x, targets=y)
                loss = loss / config.grad_accumulation_steps
                accum_loss += loss.item()
                scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

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
                f"{tokens_per_sec:>10,.0f} | {dt:>7.1f}s"
            )
            if config.use_wandb:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/grad_norm": grad_norm.item(),
                    },
                    step=iter_num + 1,
                )
            running_loss = 0.0
            t0 = time.time()

        # --- Periodic generation-based eval (see module docstring) ---
        if master and (iter_num + 1) % config.eval_interval == 0:
            eval_model = model.module if isinstance(model, DDP) else model
            eval_model = eval_model._orig_mod if hasattr(eval_model, "_orig_mod") else eval_model
            eval_sample = held_out_examples[: config.eval_sample_size]
            eval_calc_ids = resolve_calc_ids(tokenizer, use_calc=True)
            eval_format = "chatml" if config.dataset_format == "chatml" else "qa"
            results = run_examples(
                eval_model, tokenizer, eval_sample, device, eval_calc_ids, format=eval_format
            )
            summary = summarize(results)
            print(f"{'>>> EVAL':>8} | {'':>10} | {'':>12} | {'':>10} |")
            print_summary(summary)
            if config.use_wandb:
                log_dict = {"eval/accuracy": summary["accuracy"]}
                if "hop_count_match_rate" in summary:
                    log_dict["eval/hop_count_match_rate"] = summary["hop_count_match_rate"]
                for h, (acc, _) in summary.get("per_hop", {}).items():
                    log_dict[f"eval/accuracy_{h}hop"] = acc
                wandb.log(log_dict, step=iter_num + 1)

        # --- Periodic checkpointing ---
        # No in-loop validation (see module docstring) -- held-out problems are scored
        # separately (above) with real generation, not loss.
        if master and (iter_num + 1) % config.checkpoint_interval == 0:
            if config.use_wandb:
                wandb.log({"train/checkpoint_loss": accum_loss}, step=iter_num + 1)

            if accum_loss < best_loss:
                best_loss = accum_loss
                save_checkpoint(
                    model, optimizer, iter_num + 1, accum_loss, config, model_config,
                    os.path.join(config.checkpoint_dir, "best.pt"),
                )

            ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_{iter_num+1:05d}.pt")
            save_checkpoint(model, optimizer, iter_num + 1, accum_loss, config, model_config, ckpt_path)

            latest_path = os.path.join(config.checkpoint_dir, "latest.pt")
            save_checkpoint(model, optimizer, iter_num + 1, accum_loss, config, model_config, latest_path)

    if master:
        print("\nTraining complete!")
        final_path = os.path.join(config.checkpoint_dir, "final.pt")
        save_checkpoint(model, optimizer, config.max_iters, 0.0, config, model_config, final_path)
        print(f"\nHeld-out examples for eval_reasoning.py: {config.heldout_out_path}")
        if config.use_wandb:
            wandb.finish()

    cleanup_ddp(ddp)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyLLM reasoning SFT (Stage 3)")
    cfg = TrainConfig()
    for key, val in cfg.__dict__.items():
        if isinstance(val, bool):
            # BooleanOptionalAction (not plain store_true) so defaults of True are
            # actually overridable from the CLI via --no-<key> -- store_true can only
            # ever set a flag to True, so e.g. resume_weights_only (default True) could
            # never be turned off to continue training with preserved iter/optimizer
            # state.
            parser.add_argument(f"--{key}", default=val, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainConfig(**vars(args))
    train(config)

"""
Dataset utilities for language model training.

We use TinyShakespeare as the default training corpus — a classic LLM
learning dataset (~1MB, public domain). Easy to see results quickly.

This module handles:
- Downloading the dataset
- Training the tokenizer on it
- Creating PyTorch Dataset / DataLoader objects for training
- Proper train/val splits
"""

import os
import urllib.request

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from tokenizer import BPETokenizer

# URLs for small, public domain datasets
DATASETS = {
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "shakespeare.txt",
        "description": "TinyShakespeare (~1MB, all works of Shakespeare)",
    },
    "bible": {
        "url": "https://raw.githubusercontent.com/mxw/grmr/master/src/finaltests/bible.txt",
        "filename": "bible.txt",
        "description": "King James Bible (~4MB)",
    },
}


def download_dataset(name: str = "shakespeare", data_dir: str = "data") -> str:
    """Download dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    info = DATASETS[name]
    path = os.path.join(data_dir, info["filename"])

    if os.path.exists(path):
        print(f"Dataset already exists at {path}")
        return path

    print(f"Downloading {name} dataset...")
    print(f"  {info['description']}")
    try:
        urllib.request.urlretrieve(info["url"], path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  Downloaded {size_mb:.1f} MB -> {path}")
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Creating a synthetic dataset for testing...")
        # Fallback: create a small synthetic dataset
        _create_synthetic_dataset(path)
    return path


def _create_synthetic_dataset(path: str):
    """Create a small synthetic text dataset as fallback."""
    lines = [
        "To be or not to be that is the question",
        "Whether tis nobler in the mind to suffer",
        "The slings and arrows of outrageous fortune",
        "Or to take arms against a sea of troubles",
        "All the world is a stage and all the men and women merely players",
        "Friends Romans countrymen lend me your ears",
        "I come to bury Caesar not to praise him",
        "The evil that men do lives after them",
        "The good is oft interred with their bones",
        "What a piece of work is man how noble in reason",
        "How infinite in faculty in form and moving how express and admirable",
        "In action how like an angel in apprehension how like a god",
        "We are such stuff as dreams are made on",
        "And our little life is rounded with a sleep",
        "The quality of mercy is not strained",
        "It droppeth as the gentle rain from heaven",
        "Upon the place beneath it is twice blest",
        "It blesseth him that gives and him that takes",
        "Neither a borrower nor a lender be",
        "For loan oft loses both itself and friend",
    ] * 200  # Repeat to get reasonable corpus size

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Created synthetic dataset at {path}")


class TokenizedDataset(Dataset):
    """
    PyTorch Dataset for language modeling.

    Given a sequence of token ids, creates (input, target) pairs
    where target is input shifted by 1 position.

    Example with context_length=4:
        tokens: [1, 2, 3, 4, 5, 6, 7, 8]
        sample 0: input=[1,2,3,4], target=[2,3,4,5]
        sample 1: input=[2,3,4,5], target=[3,4,5,6]
        ...

    This implements the "sliding window" approach — every position
    in the dataset is used as a starting point.
    """

    def __init__(self, token_ids: list[int], context_length: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        # Each sample needs context_length + 1 tokens (input + target)
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]
        x = chunk[:-1]  # input:  tokens 0..T-1
        y = chunk[1:]  # target: tokens 1..T  (shifted by 1)
        return x, y


def prepare_data(
    dataset_name: str = "shakespeare",
    vocab_size: int = 4096,
    context_length: int = 256,
    val_fraction: float = 0.1,
    data_dir: str = "data",
    tokenizer_path: str = "data/tokenizer.json",
    force_retrain_tokenizer: bool = False,
) -> tuple[TokenizedDataset, TokenizedDataset, BPETokenizer]:
    """
    Full data pipeline: download -> tokenize -> split -> dataset objects.

    Returns:
        train_dataset, val_dataset, tokenizer
    """
    # 1. Download raw text
    text_path = download_dataset(dataset_name, data_dir)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus size: {len(text):,} characters")

    # 2. Train or load tokenizer
    if os.path.exists(tokenizer_path) and not force_retrain_tokenizer:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=vocab_size, verbose=True)
        tokenizer.save(tokenizer_path)

    # 3. Encode full corpus
    print("Encoding corpus...")
    token_ids = tokenizer.encode(text)
    print(
        f"Corpus encoded: {len(token_ids):,} tokens "
        f"(compression: {len(text)/len(token_ids):.2f}x)"
    )

    # 4. Train / val split
    split_idx = int(len(token_ids) * (1 - val_fraction))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")

    # 5. Create Dataset objects
    train_ds = TokenizedDataset(train_ids, context_length)
    val_ds = TokenizedDataset(val_ids, context_length)
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    return train_ds, val_ds, tokenizer


def create_dataloader(
    dataset: TokenizedDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Create a DataLoader, with optional DistributedSampler for multi-GPU.

    With DDP (Distributed Data Parallel), each GPU processes a different
    shard of the data. The DistributedSampler handles partitioning.
    """
    sampler = None
    if distributed:
        # Each process gets a non-overlapping subset of indices
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # DistributedSampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,  # Faster CPU->GPU transfer
        drop_last=True,  # Drop incomplete last batch for consistency
    )


if __name__ == "__main__":
    train_ds, val_ds, tok = prepare_data(vocab_size=2000, context_length=64)
    loader = create_dataloader(train_ds, batch_size=4)
    x, y = next(iter(loader))
    print(f"Batch x: {x.shape}, y: {y.shape}")
    print(f"Sample decode: '{tok.decode(x[0].tolist())}'")
    print("Data pipeline OK ✓")

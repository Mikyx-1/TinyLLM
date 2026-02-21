"""
Dataset utilities for language model training.

Supports:
- TinyShakespeare / Bible (plain text corpora)
- Custom Q&A JSON format: [{"id": 1, "category": "...", "question": "...", "answer": "..."}, ...]
"""

import json
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

# ── Q&A formatting ────────────────────────────────────────────────────────────

# Each Q&A pair is wrapped with <BOS> ... <EOS> so the model learns:
#   1. Where a conversation starts  (<BOS>)
#   2. Where it ends                (<EOS>)
#
# Without <EOS> the model sees one giant stream of text and happily generates
# the next question-answer pair after finishing an answer — exactly the
# "ridiculous" behaviour you observed.
QA_TEMPLATE = "<BOS> Question: {question}\nAnswer: {answer} <EOS>"
QA_SEPARATOR = "\n\n"  # separates individual Q&A pairs in the flat corpus


def load_custom_json(path: str) -> str:
    """
    Load a custom Q&A JSON file and convert it to a flat training corpus.

    Expected format:
        [
          {"id": 1, "category": "Identity", "question": "...", "answer": "..."},
          ...
        ]

    Each entry is rendered as:
        <BOS> Question: <question>
        Answer: <answer> <EOS>

    The <BOS>/<EOS> markers teach the model where each exchange begins and
    ends.  The tokenizer's encode() method recognises these as special tokens
    and emits their dedicated ids rather than running them through BPE.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at the top level, got {type(data)}")

    pairs = []
    for i, item in enumerate(data):
        missing = [k for k in ("question", "answer") if k not in item]
        if missing:
            raise ValueError(f"Item {i} is missing required keys: {missing}")
        pairs.append(
            QA_TEMPLATE.format(
                question=item["question"].strip(),
                answer=item["answer"].strip(),
            )
        )

    corpus = QA_SEPARATOR.join(pairs)
    print(f"Loaded {len(pairs)} Q&A pairs → {len(corpus):,} characters")
    return corpus


# ── Original plain-text helpers (unchanged) ──────────────────────────────────


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
    ] * 200

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Created synthetic dataset at {path}")


# ── Dataset / DataLoader ──────────────────────────────────────────────────────


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
    """

    def __init__(self, token_ids: list[int], context_length: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# ── Pipeline helpers ──────────────────────────────────────────────────────────


def _build_datasets(
    text: str,
    vocab_size: int,
    context_length: int,
    val_fraction: float,
    tokenizer_path: str,
    force_retrain_tokenizer: bool,
) -> tuple["TokenizedDataset", "TokenizedDataset", BPETokenizer]:
    """Shared tokenise → split → dataset logic used by both prepare functions."""
    print(f"Corpus size: {len(text):,} characters")

    if os.path.exists(tokenizer_path) and not force_retrain_tokenizer:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=vocab_size, verbose=True)
        tokenizer.save(tokenizer_path)

    print("Encoding corpus...")
    token_ids = tokenizer.encode(text)
    print(
        f"Corpus encoded: {len(token_ids):,} tokens "
        f"(compression: {len(text)/len(token_ids):.2f}x)"
    )

    split_idx = int(len(token_ids) * (1 - val_fraction))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")

    train_ds = TokenizedDataset(train_ids, context_length)
    val_ds = TokenizedDataset(val_ids, context_length)
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    return train_ds, val_ds, tokenizer


def prepare_data(
    dataset_name: str = "shakespeare",
    vocab_size: int = 4096,
    context_length: int = 256,
    val_fraction: float = 0.1,
    data_dir: str = "data",
    tokenizer_path: str = "data/tokenizer.json",
    force_retrain_tokenizer: bool = False,
) -> tuple["TokenizedDataset", "TokenizedDataset", BPETokenizer]:
    """Plain-text pipeline: download → tokenize → split → dataset objects."""
    text_path = download_dataset(dataset_name, data_dir)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    return _build_datasets(
        text,
        vocab_size,
        context_length,
        val_fraction,
        tokenizer_path,
        force_retrain_tokenizer,
    )


def prepare_custom_data(
    json_path: str,
    vocab_size: int = 4096,
    context_length: int = 256,
    val_fraction: float = 0.1,
    tokenizer_path: str = "data/tokenizer.json",
    force_retrain_tokenizer: bool = False,
) -> tuple["TokenizedDataset", "TokenizedDataset", BPETokenizer]:
    """
    Custom Q&A JSON pipeline: load JSON → format → tokenize → split → datasets.

    Args:
        json_path:   Path to your JSON file (list of Q&A dicts).
        vocab_size:  BPE vocabulary size.
        context_length: Sliding-window context length in tokens.
        val_fraction:   Fraction of tokens held out for validation.
        tokenizer_path: Where to save/load the trained tokenizer.
        force_retrain_tokenizer: Re-train even if a saved tokenizer exists.

    Returns:
        train_dataset, val_dataset, tokenizer
    """
    text = load_custom_json(json_path)
    return _build_datasets(
        text,
        vocab_size,
        context_length,
        val_fraction,
        tokenizer_path,
        force_retrain_tokenizer,
    )


def create_dataloader(
    dataset: TokenizedDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create a DataLoader, with optional DistributedSampler for multi-GPU."""
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    train_ds, val_ds, tok = prepare_custom_data(
        json_path="./data/tinyllm_dataset.json",
        vocab_size=2000,
        context_length=64,
        force_retrain_tokenizer=True,  # retrain so <BOS>/<EOS> are in the corpus
    )

    loader = create_dataloader(train_ds, batch_size=4)
    x, y = next(iter(loader))
    print(f"Batch x: {x.shape}, y: {y.shape}")
    print(f"Sample decode: '{tok.decode(x[0].tolist())}'")
    print("\n")
    print(f"GT decode: {tok.decode(y[0].tolist())}")

    # Verify special tokens are present and correctly roundtripped
    test = "<BOS> What is your name? <EOS>"
    enc = tok.encode(test)
    assert tok.bos_id in enc, "<BOS> id missing from encoded output!"
    assert tok.eos_id in enc, "<EOS> id missing from encoded output!"
    print(
        f"Special token check — BOS id {tok.bos_id} and EOS id {tok.eos_id} both present ✓"
    )
    print("Data pipeline OK ✓")

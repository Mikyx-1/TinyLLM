"""
Dataset utilities for language model training.

Supports:
- TinyShakespeare / Bible (plain text corpora)
- Custom Q&A JSON format: [{"id": 1, "category": "...", "question": "...", "answer": "..."}, ...]
"""

import json
import os
import random
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

# CoT reasoning format: <THINK> wraps the trace so it's dropped from a plain
# answer-only reading, and any <CALC>expr</CALC> inside it is a live calculator call
# (see model/generate.py) rather than plain text.
REASONING_TEMPLATE = "<BOS> Question: {question}\n<THINK> {reasoning} </THINK>\nAnswer: {answer} <EOS>"


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


# ── Multi-turn conversation formatting ───────────────────────────────────────
#
# A conversation is {"turns": [{"question": ..., "answer": ...}, ...]}, with any
# number of turns >= 1 -- unlike QA_TEMPLATE, which has one fixed question/answer slot,
# nothing here caps how many turns a conversation can carry.
#
# ChatML-style rendering (same scheme Qwen/GPT use): every message, user or assistant,
# is "<|im_start|>{role}\n{content}<|im_end|>\n" and turns are just concatenated -- no
# separate outer <BOS>/<EOS> wrapper, since <|im_start|> already unambiguously marks
# where each message begins. Every turn is formatted identically regardless of its
# position (first, middle, last), so a conversation really is "just appended tokens" of
# the same per-turn unit repeated N times.
#
# render_conversation() produces the flat text (for callers that just want corpus text,
# same role as QA_TEMPLATE). encode_conversation() is the richer form used for real
# training: it tokenizes turn-by-turn and returns a token-aligned loss mask so a caller
# can train only on assistant messages, not on the user turns that precede them --
# mirroring how production chat models mask the loss to assistant tokens only instead
# of learning to predict the user's next message. <|im_end|> after an assistant message
# is also the token generation should stop on (see model/generate.py's eos_id param) --
# analogous to how <|im_end|> doubles as ChatML's stop token.


def render_conversation(turns: list[dict]) -> str:
    """Flatten a list of {"question", "answer"} turns into ChatML-style message text."""
    return "".join(
        f"<|im_start|>user\n{t['question'].strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{t['answer'].strip()}<|im_end|>\n"
        for t in turns
    )


def flatten_conversation_to_qa(turns: list[dict]) -> dict:
    """Collapse a multi-turn conversation into a single legacy {"question", "answer"} record.

    turns[0] becomes the top-level question; every later turn is rendered inline in the
    answer using the plain-text "Question:"/"Answer:" labels QA_TEMPLATE already uses,
    so the result reads correctly through the existing single-turn pipeline
    (load_custom_json / QA_TEMPLATE) for callers -- e.g. build_smalltalk_demo_dataset.py
    -- that only understand flat Q&A records, not the {"turns": [...]} schema. Works for
    any number of turns, unlike hand-writing the second turn into the answer string.
    """
    question = turns[0]["question"].strip()
    answer_parts = [turns[0]["answer"].strip()]
    for t in turns[1:]:
        answer_parts.append(f"Question: {t['question'].strip()}\nAnswer: {t['answer'].strip()}")
    return {"question": question, "answer": "\n".join(answer_parts)}


def encode_conversation(
    tokenizer: "BPETokenizer", turns: list[dict]
) -> tuple[list[int], list[bool]]:
    """Tokenize a conversation turn-by-turn, tagging which token positions are assistant content.

    Returns (ids, loss_mask) where loss_mask[i] is True iff ids[i] is part of an
    "<|im_start|>assistant\\n...<|im_end|>\\n" message -- the span a model should
    actually be trained to predict, including its own role header and closing
    <|im_end|>. Every "<|im_start|>user\\n...<|im_end|>\\n" message is False: the model
    conditions on it but is never scored for "predicting" what the user said.
    """
    ids: list[int] = []
    mask: list[bool] = []

    def add(text: str, is_answer: bool):
        toks = tokenizer.encode(text)
        ids.extend(toks)
        mask.extend([is_answer] * len(toks))

    for t in turns:
        add(f"<|im_start|>user\n{t['question'].strip()}<|im_end|>\n", False)
        add(f"<|im_start|>assistant\n{t['answer'].strip()}<|im_end|>\n", True)
    return ids, mask


def reasoning_example_to_conversation(example: dict) -> dict:
    """Recast a {question, reasoning, answer} reasoning example as a 1-turn ChatML
    conversation, so it can be pooled with ordinary chat conversations and trained
    through the exact same encode_conversation()/MaskedTokenizedDataset pipeline (one
    tokenizer, one template, one masking rule for both task types -- see
    prepare_multitask_data). The <THINK>...</THINK> trace and any <CALC>...</CALC>
    calls inside `reasoning` are untouched -- they're just now part of the assistant
    turn's content instead of REASONING_TEMPLATE's flat "<BOS> Question: ... <THINK>
    ... </THINK>\\nAnswer: ... <EOS>" string, so the model is still trained to produce
    the reasoning trace (it's part of the scored assistant span), it just now also
    gets the "don't score the question" masking chat turns already get.
    """
    return {
        "turns": [{
            "question": example["question"],
            "answer": f"<THINK> {example['reasoning'].strip()} </THINK>\nAnswer: {example['answer']}",
        }]
    }


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
        # max(0, ...) so an empty (or shorter-than-context) split, e.g. val_fraction=0,
        # reports a valid length instead of a negative one.
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class MaskedTokenizedDataset(Dataset):
    """
    Like TokenizedDataset, but a parallel boolean mask marks which target positions
    should contribute to the loss (see encode_conversation). Masked-out targets are
    replaced with -1, which TinyLLM.forward's cross_entropy already treats as
    ignore_index, so no changes are needed on the training-loop side.
    """

    def __init__(self, token_ids: list[int], loss_mask: list[bool], context_length: int):
        assert len(token_ids) == len(loss_mask), "token_ids and loss_mask must be the same length"
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.mask = torch.tensor(loss_mask, dtype=torch.bool)
        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        y_mask = self.mask[idx + 1 : idx + self.context_length + 1]
        y = torch.where(y_mask, y, torch.full_like(y, -1))
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


def prepare_multiturn_data(
    json_path: str,
    vocab_size: int = 4096,
    context_length: int = 256,
    val_fraction: float = 0.1,
    tokenizer_path: str = "data/tokenizer.json",
    force_retrain_tokenizer: bool = False,
) -> tuple["MaskedTokenizedDataset", "MaskedTokenizedDataset", BPETokenizer]:
    """
    Multi-turn conversation pipeline: load [{"turns": [...]}, ...] -> tokenize each
    conversation turn-by-turn with an assistant-only loss mask -> concatenate ->
    split -> masked sliding-window datasets.

    Unlike prepare_custom_data, where the whole Question+Answer sequence is scored
    equally, training here only back-props through assistant messages, so the model
    isn't taught to "predict" what the user says next -- see encode_conversation()'s
    docstring. Conversations may have any number of turns.

    Returns:
        train_dataset, val_dataset, tokenizer
    """
    with open(json_path, "r", encoding="utf-8") as f:
        conversations: list[dict] = json.load(f)
    print(f"Loaded {len(conversations):,} conversations from {json_path}")

    if os.path.exists(tokenizer_path) and not force_retrain_tokenizer:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        flat_text = "".join(render_conversation(c["turns"]) for c in conversations)
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        tokenizer = BPETokenizer()
        tokenizer.train(flat_text, vocab_size=vocab_size, verbose=True)
        tokenizer.save(tokenizer_path)

    # Old tokenizers on disk predate <|im_start|>/<|im_end|> -- add them without
    # disturbing any existing id (no-op if already present, e.g. a freshly trained
    # tokenizer). Must re-save: otherwise the ids only exist in this in-memory object,
    # and the *next* process to BPETokenizer.load(tokenizer_path) (webchat.py,
    # chat_demo.py, a standalone eval_reasoning.py run) gets a tokenizer that silently
    # mis-tokenizes "<|im_start|>"/"<|im_end|>" as ordinary BPE'd characters instead of
    # the dedicated ids the checkpoint's embedding table was actually trained with.
    if "<|im_start|>" not in tokenizer.encoder or "<|im_end|>" not in tokenizer.encoder:
        tokenizer.add_special_tokens(["<|im_start|>", "<|im_end|>"])
        tokenizer.save(tokenizer_path)

    print("Encoding conversations...")
    # Conversations are concatenated directly, with no separator -- <|im_start|> is
    # already an unambiguous message/conversation boundary, exactly like packing
    # ChatML-formatted documents back to back.
    token_ids: list[int] = []
    loss_mask: list[bool] = []
    for conv in conversations:
        ids, mask = encode_conversation(tokenizer, conv["turns"])
        token_ids.extend(ids)
        loss_mask.extend(mask)

    print(
        f"Corpus encoded: {len(token_ids):,} tokens "
        f"({sum(loss_mask):,} scored / {len(loss_mask) - sum(loss_mask):,} masked)"
    )

    split_idx = int(len(token_ids) * (1 - val_fraction))
    train_ds = MaskedTokenizedDataset(token_ids[:split_idx], loss_mask[:split_idx], context_length)
    val_ds = MaskedTokenizedDataset(token_ids[split_idx:], loss_mask[split_idx:], context_length)
    print(f"Train samples: {len(train_ds):,}, Val samples: {len(val_ds):,}")

    return train_ds, val_ds, tokenizer


def prepare_multitask_data(
    multiturn_path: str,
    reasoning_path: str,
    vocab_size: int = 12_000,
    context_length: int = 384,
    held_out: int = 450,
    seed: int = 42,
    tokenizer_path: str = "checkpoints/tokenizer.json",
    force_retrain_tokenizer: bool = False,
    heldout_out_path: str = "data/reasoning_heldout.json",
) -> tuple["MaskedTokenizedDataset", list[dict], BPETokenizer]:
    """
    Joint multi-task pipeline: pool every small-talk conversation in `multiturn_path`
    (variable-depth, {"turns": [...]}) with the reasoning dataset at `reasoning_path`
    ({question, reasoning, answer, hops?}, converted to 1-turn ChatML conversations via
    reasoning_example_to_conversation) into ONE shuffled corpus, trained through the
    same encode_conversation()/MaskedTokenizedDataset pipeline as prepare_multiturn_data.

    Unlike train_reasoning.py's replay (chat data mixed in, unmasked, via QA_TEMPLATE,
    specifically so the *reasoning* run doesn't forget chat -- reasoning is still the
    "primary" data there), this is genuinely joint: both task types are pooled and
    shuffled together as equals, and the model can't tell them apart by template (both
    render through the identical <|im_start|>/<|im_end|> turn format) -- only by
    content. That's the point: it has to actually condition on what's being asked, not
    switch mode on a template cue.

    Whole reasoning examples are held out *before* conversion/pooling (never trained
    on) so eval_reasoning.py can still measure generalization to unseen problems, same
    as prepare_reasoning_data. Chat conversations are never held out here -- there's no
    fixed "right answer" to generalize to for small talk the way there is for a
    reasoning problem's final number.

    Returns:
        train_dataset, held_out_reasoning_examples (raw dicts, unrendered), tokenizer
    """
    with open(multiturn_path, "r", encoding="utf-8") as f:
        chat_conversations = [{"turns": c["turns"]} for c in json.load(f)]

    with open(reasoning_path, "r", encoding="utf-8") as f:
        reasoning_examples: list[dict] = json.load(f)
    shuffled = reasoning_examples[:]
    random.Random(seed).shuffle(shuffled)
    held_out_examples = shuffled[:held_out]
    train_reasoning_examples = shuffled[held_out:]

    with open(heldout_out_path, "w", encoding="utf-8") as f:
        json.dump(held_out_examples, f, ensure_ascii=False, indent=2)

    reasoning_conversations = [
        reasoning_example_to_conversation(ex) for ex in train_reasoning_examples
    ]

    all_conversations = chat_conversations + reasoning_conversations
    random.Random(seed + 1).shuffle(all_conversations)

    print(
        f"Multi-task data: {len(chat_conversations):,} chat conversations from {multiturn_path} + "
        f"{len(reasoning_conversations):,} reasoning conversations from {reasoning_path} "
        f"({len(held_out_examples)} reasoning examples held out -> {heldout_out_path}) "
        f"= {len(all_conversations):,} total"
    )

    if os.path.exists(tokenizer_path) and not force_retrain_tokenizer:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        flat_text = "".join(render_conversation(c["turns"]) for c in all_conversations)
        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
        tokenizer = BPETokenizer()
        tokenizer.train(flat_text, vocab_size=vocab_size, verbose=True)
        tokenizer.save(tokenizer_path)

    # Old tokenizers on disk predate <|im_start|>/<|im_end|> -- add them without
    # disturbing any existing id (no-op if already present, e.g. a freshly trained
    # tokenizer). Must re-save: otherwise the ids only exist in this in-memory object,
    # and the *next* process to BPETokenizer.load(tokenizer_path) (webchat.py,
    # chat_demo.py, a standalone eval_reasoning.py run) gets a tokenizer that silently
    # mis-tokenizes "<|im_start|>"/"<|im_end|>" as ordinary BPE'd characters instead of
    # the dedicated ids the checkpoint's embedding table was actually trained with.
    if "<|im_start|>" not in tokenizer.encoder or "<|im_end|>" not in tokenizer.encoder:
        tokenizer.add_special_tokens(["<|im_start|>", "<|im_end|>"])
        tokenizer.save(tokenizer_path)

    print("Encoding conversations...")
    token_ids: list[int] = []
    loss_mask: list[bool] = []
    for conv in all_conversations:
        ids, mask = encode_conversation(tokenizer, conv["turns"])
        token_ids.extend(ids)
        loss_mask.extend(mask)

    print(
        f"Corpus encoded: {len(token_ids):,} tokens "
        f"({sum(loss_mask):,} scored / {len(loss_mask) - sum(loss_mask):,} masked)"
    )

    train_ds = MaskedTokenizedDataset(token_ids, loss_mask, context_length)
    print(f"Train samples: {len(train_ds):,}")

    return train_ds, held_out_examples, tokenizer


def prepare_reasoning_data(
    json_path: str,
    vocab_size: int = 12_000,
    context_length: int = 384,
    held_out: int = 200,
    seed: int = 42,
    tokenizer_path: str = "checkpoints/tokenizer.json",
    force_retrain_tokenizer: bool = False,
    heldout_out_path: str = "data/reasoning_heldout.json",
    replay_dataset_path: str | None = None,
    replay_count: int = 0,
) -> tuple["TokenizedDataset", list[dict], BPETokenizer]:
    """
    Reasoning JSON pipeline: load {question, reasoning, answer} -> hold out `held_out`
    whole examples -> render the rest as CoT text -> tokenize -> sliding-window train
    dataset.

    Deliberately NOT val_fraction (a token-position cut through one concatenated
    corpus, as prepare_custom_data uses for pure-memorization SFT runs): the point of
    this stage is measuring whether the model can solve problems it never saw a
    single token of, so whole examples must be removed *before* concatenation, not
    sliced out of the middle of a shared token stream where a held-out problem's
    tokens could still leak into a training window.

    The held-out set is deterministic (fixed seed, examples loaded in file order) and
    written to `heldout_out_path` so eval_reasoning.py reads the exact same examples
    without re-deriving the split.

    If `replay_dataset_path` is given (e.g. data/sft_dataset.json), a random sample of
    `replay_count` {question, answer} examples is rendered via QA_TEMPLATE and shuffled
    in among the reasoning examples before concatenation. This is rehearsal: fine-tuning
    on only a small, repetitive reasoning corpus for many iterations drives loss to near
    zero and catastrophically overwrites the prior stage's Q&A ability (confirmed
    directly -- a question the SFT checkpoint answered perfectly came back as a bare
    wrong number after reasoning-only training). Mixing in real SFT examples keeps that
    ability exercised throughout training instead of only at iter 0. Replay examples are
    always included in training and never held out -- Stage 2 already covers testing
    memorization of the SFT set; held_out here only ever samples from the reasoning
    examples in json_path.

    Returns:
        train_dataset, held_out_examples (raw dicts, unrendered), tokenizer
    """
    with open(json_path, "r", encoding="utf-8") as f:
        examples: list[dict] = json.load(f)

    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    held_out_examples = shuffled[:held_out]
    train_examples = shuffled[held_out:]

    with open(heldout_out_path, "w", encoding="utf-8") as f:
        json.dump(held_out_examples, f, ensure_ascii=False, indent=2)
    print(
        f"Reasoning data: {len(train_examples):,} training examples, "
        f"{len(held_out_examples)} held out -> {heldout_out_path}"
    )

    rendered = [REASONING_TEMPLATE.format(**ex) for ex in train_examples]

    if replay_dataset_path and replay_count > 0:
        with open(replay_dataset_path, "r", encoding="utf-8") as f:
            replay_pool: list[dict] = json.load(f)
        replay_sample = random.Random(seed + 1).sample(
            replay_pool, min(replay_count, len(replay_pool))
        )
        rendered.extend(QA_TEMPLATE.format(**ex) for ex in replay_sample)
        random.Random(seed + 2).shuffle(rendered)
        print(f"Replay: mixed in {len(replay_sample):,} SFT examples from {replay_dataset_path}")

    text = "\n\n".join(rendered)
    train_ds, _, tokenizer = _build_datasets(
        text,
        vocab_size,
        context_length,
        val_fraction=0.0,
        tokenizer_path=tokenizer_path,
        force_retrain_tokenizer=force_retrain_tokenizer,
    )
    return train_ds, held_out_examples, tokenizer


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

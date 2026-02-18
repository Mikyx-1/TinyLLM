"""
Byte Pair Encoding (BPE) Tokenizer — implemented from scratch.

How BPE works:
1. Start with a vocabulary of individual characters (bytes)
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until desired vocab size

This is the same core algorithm used by GPT-2/GPT-4, just smaller.
"""

import json
import os
import re
from collections import defaultdict
from typing import Optional


def get_stats(vocab: dict[tuple, int]) -> dict[tuple, int]:
    """Count frequency of every adjacent pair across all words."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = list(word)
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair: tuple, vocab: dict[tuple, int]) -> dict[tuple, int]:
    """Replace all occurrences of `pair` with a merged token."""
    new_vocab = {}
    bigram = pair  # e.g. ('h', 'e')
    for word, freq in vocab.items():
        new_word = []
        i = 0
        word_list = list(word)
        while i < len(word_list):
            if i < len(word_list) - 1 and (word_list[i], word_list[i + 1]) == bigram:
                new_word.append(word_list[i] + word_list[i + 1])
                i += 2
            else:
                new_word.append(word_list[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer built from scratch.

    Special tokens:
        <PAD> - padding
        <UNK> - unknown
        <BOS> - beginning of sequence
        <EOS> - end of sequence
    """

    SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    def __init__(self):
        self.encoder: dict[str, int] = {}  # token -> id
        self.decoder: dict[int, str] = {}  # id -> token
        self.merges: list[tuple[str, str]] = []  # ordered merge rules
        self.vocab_size: int = 0

    @property
    def pad_id(self):
        return self.encoder["<PAD>"]

    @property
    def unk_id(self):
        return self.encoder["<UNK>"]

    @property
    def bos_id(self):
        return self.encoder["<BOS>"]

    @property
    def eos_id(self):
        return self.encoder["<EOS>"]

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        """
        Train BPE on raw text.

        Args:
            text: raw training corpus
            vocab_size: target vocabulary size
            verbose: print merge progress
        """
        assert (
            vocab_size > len(self.SPECIAL_TOKENS) + 256
        ), "vocab_size must be > 260 to fit special tokens + base chars"

        # --- Step 1: Build initial character vocabulary ---
        # Pretokenize: split on whitespace, treat each word as a sequence of chars
        # We add a special end-of-word marker '▁' (or we can use Ġ like GPT-2)
        # Here we use a simple space-prefix scheme: words get '▁' prefix
        words = text.split()
        word_freq: dict[tuple, int] = defaultdict(int)
        for word in words:
            # Represent word as tuple of chars + end marker
            token_word = tuple(list("▁" + word))
            word_freq[token_word] += 1

        # Initial vocab: all unique characters
        char_vocab: set[str] = set()
        for word in word_freq:
            for ch in word:
                char_vocab.add(ch)

        # Build encoder starting with special tokens, then chars
        idx = 0
        for sp in self.SPECIAL_TOKENS:
            self.encoder[sp] = idx
            idx += 1
        for ch in sorted(char_vocab):
            self.encoder[ch] = idx
            idx += 1

        # --- Step 2: BPE merge loop ---
        num_merges = vocab_size - len(self.encoder)
        vocab = dict(word_freq)

        if verbose:
            print(
                f"Starting BPE training: base vocab={len(self.encoder)}, "
                f"target vocab={vocab_size}, merges={num_merges}"
            )

        for merge_idx in range(num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            merged = best_pair[0] + best_pair[1]

            # Record the merge rule
            self.merges.append(best_pair)

            # Add merged token to vocab
            self.encoder[merged] = idx
            idx += 1

            # Apply merge to corpus
            vocab = merge_vocab(best_pair, vocab)

            if verbose and (merge_idx + 1) % 500 == 0:
                print(
                    f"  Merge {merge_idx+1}/{num_merges}: "
                    f"'{best_pair[0]}' + '{best_pair[1]}' -> '{merged}' "
                    f"(freq={pairs[best_pair]})"
                )

        self.vocab_size = len(self.encoder)
        self.decoder = {v: k for k, v in self.encoder.items()}
        if verbose:
            print(f"Training complete. Final vocab size: {self.vocab_size}")

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply learned BPE merges to a single word."""
        # Start: split into individual chars
        symbols = list("▁" + word)

        # Apply merges in learned order
        for left, right in self.merges:
            i = 0
            new_symbols = []
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == left
                    and symbols[i + 1] == right
                ):
                    new_symbols.append(left + right)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token ids."""
        words = text.split()
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)
        for word in words:
            tokens = self._tokenize_word(word)
            for tok in tokens:
                ids.append(self.encoder.get(tok, self.unk_id))
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        tokens = [self.decoder.get(i, "<UNK>") for i in ids]
        text = "".join(tokens)
        # Replace our word-boundary marker with spaces
        text = text.replace("▁", " ").strip()
        return text

    def save(self, path: str):
        """Save tokenizer to JSON."""
        data = {
            "encoder": self.encoder,
            "merges": self.merges,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.encoder = data["encoder"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab_size = data["vocab_size"]
        tok.decoder = {int(v): k for k, v in tok.encoder.items()}
        return tok


if __name__ == "__main__":
    # Quick sanity check
    sample = "hello world this is a test hello world"
    tok = BPETokenizer()
    tok.train(sample, vocab_size=300, verbose=False)
    ids = tok.encode("hello world")
    print(f"Encoded: {ids}")
    print(f"Decoded: '{tok.decode(ids)}'")
    print("Tokenizer OK ✓")

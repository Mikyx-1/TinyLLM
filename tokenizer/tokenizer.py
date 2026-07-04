"""
Byte Pair Encoding (BPE) Tokenizer.

How BPE works:
1. Start with a vocabulary of individual characters (bytes)
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until desired vocab size

This is the same core algorithm used by GPT-2/GPT-4, just smaller.
"""

import json
import re

from .bpe_utils import get_stats, merge_vocab
from .constants import SPECIAL_TOKENS, WORD_BOUNDARY


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer built from scratch.

    Special tokens:
        <PAD> - padding
        <UNK> - unknown
        <BOS> - beginning of sequence
        <EOS> - end of sequence
    """

    SPECIAL_TOKENS = SPECIAL_TOKENS

    def __init__(self):
        self.encoder: dict[str, int] = {}  # token -> id
        self.decoder: dict[int, str] = {}  # id -> token
        self.merges: list[tuple[str, str]] = []  # ordered merge rules
        self.merge_ranks: dict[tuple[str, str], int] = {}  # merge -> position in self.merges
        self.vocab_size: int = 0

    # ------------------------------------------------------------------
    # Special-token id properties
    # ------------------------------------------------------------------

    @property
    def pad_id(self) -> int:
        return self.encoder["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self.encoder["<UNK>"]

    @property
    def bos_id(self) -> int:
        return self.encoder["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.encoder["<EOS>"]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        """Train BPE on raw text.

        Args:
            text: raw training corpus.
            vocab_size: target vocabulary size.
            verbose: print merge progress.
        """
        assert (
            vocab_size > len(self.SPECIAL_TOKENS) + 256
        ), "vocab_size must be > 260 to fit special tokens + base chars"

        # Strip special tokens so they are never broken up by BPE merges.
        clean_text = text
        for sp in self.SPECIAL_TOKENS:
            clean_text = clean_text.replace(sp, " ")

        # Pretokenize: each word becomes a tuple of chars with a boundary prefix.
        from collections import defaultdict

        word_freq: dict[tuple, int] = defaultdict(int)
        for word in clean_text.split():
            word_freq[tuple(WORD_BOUNDARY + word)] += 1

        # Initial vocab: special tokens first, then all unique characters.
        char_vocab: set[str] = {ch for word in word_freq for ch in word}
        idx = 0
        for sp in self.SPECIAL_TOKENS:
            self.encoder[sp] = idx
            idx += 1
        for ch in sorted(char_vocab):
            self.encoder[ch] = idx
            idx += 1

        # BPE merge loop.
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

            best_pair = max(pairs, key=pairs.get)
            merged = best_pair[0] + best_pair[1]

            self.merges.append(best_pair)
            self.encoder[merged] = idx
            idx += 1

            vocab = merge_vocab(best_pair, vocab)

            if verbose and (merge_idx + 1) % 500 == 0:
                print(
                    f"  Merge {merge_idx+1}/{num_merges}: "
                    f"'{best_pair[0]}' + '{best_pair[1]}' -> '{merged}' "
                    f"(freq={pairs[best_pair]})"
                )

        self.vocab_size = len(self.encoder)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        if verbose:
            print(f"Training complete. Final vocab size: {self.vocab_size}")

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply learned BPE merges to a single word.

        Rather than scanning the full ordered merge list (which was learned once over the whole
        corpus and is only sparsely relevant to any one word), repeatedly find the adjacent pair
        with the lowest rank (earliest-learned) that's actually present in the word and merge it.
        This produces exactly the same result as applying the merges in order — a rank-N merge
        can only ever create a pair for a rank-N+1 merge, so picking the lowest-rank pair present
        at each step reproduces the same merge sequence — just skipping the merges that don't
        apply to this word instead of checking each one explicitly.
        """
        symbols = list(WORD_BOUNDARY + word)

        while len(symbols) > 1:
            ranked_pairs = (
                (self.merge_ranks[pair], pair)
                for pair in zip(symbols, symbols[1:])
                if pair in self.merge_ranks
            )
            best = min(ranked_pairs, default=None)
            if best is None:
                break
            left, right = best[1]

            i = 0
            new_symbols: list[str] = []
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
        """Encode text to token ids.

        Special tokens present in the text are preserved as single tokens
        rather than being fed through BPE. This allows ``data_utils`` to embed
        ``<BOS>``/``<EOS>`` boundary markers directly in the corpus string and
        have them survive tokenisation intact.

        Args:
            text: input string to encode.
            add_special_tokens: if True, prepend BOS and append EOS ids.

        Returns:
            List of integer token ids.
        """
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)

        special_pattern = (
            "(" + "|".join(re.escape(s) for s in self.SPECIAL_TOKENS) + ")"
        )
        for segment in re.split(special_pattern, text):
            if segment in self.encoder:
                ids.append(self.encoder[segment])
            elif segment:
                for word in segment.split():
                    for tok in self._tokenize_word(word):
                        ids.append(self.encoder.get(tok, self.unk_id))

        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids back to text.

        Args:
            ids: list of token ids.
            skip_special_tokens: if True (default) special tokens are dropped
                from the output string. Set to False to see them explicitly.

        Returns:
            Decoded string.
        """
        tokens = []
        for i in ids:
            tok = self.decoder.get(i, "<UNK>")
            if skip_special_tokens and tok in self.SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        return "".join(tokens).replace(WORD_BOUNDARY, " ").strip()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save tokenizer state to a JSON file."""
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
        """Load tokenizer state from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.encoder = data["encoder"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab_size = data["vocab_size"]
        tok.decoder = {int(v): k for k, v in tok.encoder.items()}
        tok.merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        return tok

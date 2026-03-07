"""
Tests for the BPE tokenizer package.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer import BPETokenizer
from tokenizer.bpe_utils import get_stats, merge_vocab
from tokenizer.constants import SPECIAL_TOKENS, WORD_BOUNDARY

CORPUS = "hello world this is a test hello world hello"


# ---------------------------------------------------------------------------
# bpe_utils
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_counts_adjacent_pairs(self):
        vocab = {("a", "b", "c"): 2, ("a", "b"): 3}
        stats = get_stats(vocab)
        assert stats[("a", "b")] == 5  # 2 + 3
        assert stats[("b", "c")] == 2

    def test_empty_vocab(self):
        assert get_stats({}) == {}

    def test_single_symbol_word(self):
        # No pairs possible in a one-symbol word.
        stats = get_stats({("a",): 10})
        assert stats == {}


class TestMergeVocab:
    def test_merges_pair(self):
        vocab = {("h", "e", "l", "l", "o"): 1}
        result = merge_vocab(("h", "e"), vocab)
        assert ("he", "l", "l", "o") in result

    def test_does_not_merge_other_pairs(self):
        vocab = {("a", "b", "c"): 1}
        result = merge_vocab(("b", "c"), vocab)
        assert ("a", "bc") in result
        assert ("a", "b", "c") not in result

    def test_preserves_frequency(self):
        vocab = {("x", "y"): 7}
        result = merge_vocab(("x", "y"), vocab)
        assert result[("xy",)] == 7


# ---------------------------------------------------------------------------
# BPETokenizer — training
# ---------------------------------------------------------------------------


class TestTraining:
    def test_vocab_size_respected(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=300, verbose=False)
        assert tok.vocab_size <= 300

    def test_special_token_ids_reserved(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=300, verbose=False)
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3

    def test_special_tokens_not_split(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=300, verbose=False)
        for sp in SPECIAL_TOKENS:
            assert sp in tok.encoder


# ---------------------------------------------------------------------------
# BPETokenizer — encode / decode round-trip
# ---------------------------------------------------------------------------


class TestEncoding:
    def setup_method(self):
        self.tok = BPETokenizer()
        self.tok.train(CORPUS, vocab_size=300, verbose=False)

    def test_encode_returns_ints(self):
        ids = self.tok.encode("hello world")
        assert all(isinstance(i, int) for i in ids)

    def test_round_trip(self):
        text = "hello world"
        assert self.tok.decode(self.tok.encode(text)) == text

    def test_add_special_tokens(self):
        ids = self.tok.encode("hello", add_special_tokens=True)
        assert ids[0] == self.tok.bos_id
        assert ids[-1] == self.tok.eos_id

    def test_special_tokens_in_text_preserved(self):
        ids = self.tok.encode("<BOS> hello <EOS>")
        assert self.tok.bos_id in ids
        assert self.tok.eos_id in ids

    def test_decode_skip_special_tokens(self):
        ids = self.tok.encode("<BOS> hello <EOS>")
        assert "<BOS>" not in self.tok.decode(ids, skip_special_tokens=True)

    def test_decode_show_special_tokens(self):
        ids = self.tok.encode("<BOS> hello <EOS>")
        decoded = self.tok.decode(ids, skip_special_tokens=False)
        assert "<BOS>" in decoded
        assert "<EOS>" in decoded

    def test_unknown_token(self):
        # Feed a character that wasn't in the training corpus.
        ids = self.tok.encode("€€€")
        assert self.tok.unk_id in ids


# ---------------------------------------------------------------------------
# BPETokenizer — save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load_round_trip(self):
        tok = BPETokenizer()
        tok.train(CORPUS, vocab_size=300, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            tok.save(path)
            tok2 = BPETokenizer.load(path)

            assert tok2.vocab_size == tok.vocab_size
            assert tok2.encoder == tok.encoder
            assert tok2.merges == tok.merges

            text = "hello world"
            assert tok2.decode(tok2.encode(text)) == text
        finally:
            os.unlink(path)

from collections import defaultdict


def get_stats(vocab: dict[tuple, int]) -> dict[tuple, int]:
    """Count frequency of every adjacent pair across all words.

    Args:
        vocab: mapping of (symbol, ...) word tuples to their corpus frequency.

    Returns:
        Mapping of adjacent-pair tuples to their total frequency.
    """
    pairs: dict[tuple, int] = defaultdict(int)
    for word, freq in vocab.items():
        symbols = list(word)
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair: tuple, vocab: dict[tuple, int]) -> dict[tuple, int]:
    """Replace every occurrence of *pair* with a single merged token.

    Args:
        pair: the (left, right) symbol pair to merge, e.g. ``('h', 'e')``.
        vocab: current vocabulary to transform.

    Returns:
        New vocabulary with the pair merged everywhere it appears.
    """
    new_vocab: dict[tuple, int] = {}
    for word, freq in vocab.items():
        new_word: list[str] = []
        i = 0
        word_list = list(word)
        while i < len(word_list):
            if i < len(word_list) - 1 and (word_list[i], word_list[i + 1]) == pair:
                new_word.append(word_list[i] + word_list[i + 1])
                i += 2
            else:
                new_word.append(word_list[i])
                i += 1
        new_vocab[tuple(new_word)] = freq
    return new_vocab

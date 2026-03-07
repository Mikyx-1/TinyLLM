"""
Shared constants for the BPE tokenizer.
"""

# Word-boundary prefix (same role as Ġ in GPT-2)
WORD_BOUNDARY = "▁"

# Special tokens — order matters: indices 0-3 are reserved for these.
SPECIAL_TOKENS: list[str] = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

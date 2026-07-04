"""
Shared constants for the BPE tokenizer.
"""

# Word-boundary prefix (same role as Ġ in GPT-2)
WORD_BOUNDARY = "▁"

# Special tokens — order matters: indices 0-3 are reserved for these.
# <THINK>/</THINK> (indices 4-5) wrap chain-of-thought reasoning traces so they get
# dedicated tokens instead of being fragmented by BPE.
SPECIAL_TOKENS: list[str] = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<THINK>", "</THINK>"]

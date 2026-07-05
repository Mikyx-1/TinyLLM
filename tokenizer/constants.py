"""
Shared constants for the BPE tokenizer.
"""

# Word-boundary prefix (same role as Ġ in GPT-2)
WORD_BOUNDARY = "▁"

# Special tokens — order matters: indices 0-3 are reserved for these.
# <THINK>/</THINK> (indices 4-5) wrap chain-of-thought reasoning traces so they get
# dedicated tokens instead of being fragmented by BPE.
# <CALC>/</CALC> (indices 6-7) wrap an arithmetic expression the model wants evaluated;
# model/generate.py intercepts them and injects the real calculator result rather than
# letting the model guess the digits itself. Tokenizers that already exist on disk won't
# have these ids yet — use BPETokenizer.add_special_tokens() to append them without
# disturbing any existing id (see model/vocab_surgery.py for the checkpoint-side counterpart).
SPECIAL_TOKENS: list[str] = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<THINK>",
    "</THINK>",
    "<CALC>",
    "</CALC>",
]

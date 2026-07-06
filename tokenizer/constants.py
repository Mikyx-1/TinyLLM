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
# letting the model guess the digits itself.
# <|im_start|>/<|im_end|> (indices 8-9) delimit one message in a multi-turn
# conversation, ChatML-style (same idea as Qwen/GPT's chat template): each turn is
# rendered as "<|im_start|>{role}\n{content}<|im_end|>\n" with the role name (user/
# assistant) as ordinary text right after <|im_start|>, and turns are simply
# concatenated -- no separate outer <BOS>/<EOS> wrapper needed, since <|im_start|>
# already unambiguously marks where each message (and hence each conversation, and
# each turn within it) begins. <|im_end|> after an assistant message is also the
# generation stop token, exactly as <|im_end|> is in ChatML. See
# data_utils.encode_conversation().
# Tokenizers that already exist on disk won't have new ids yet — use
# BPETokenizer.add_special_tokens() to append them without disturbing any existing id
# (see model/vocab_surgery.py for the checkpoint-side counterpart).
SPECIAL_TOKENS: list[str] = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<THINK>",
    "</THINK>",
    "<CALC>",
    "</CALC>",
    "<|im_start|>",
    "<|im_end|>",
]

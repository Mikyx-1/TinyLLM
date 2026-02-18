"""
Transformer Language Model — implemented from scratch.

Architecture follows the original "Attention Is All You Need" paper (Vaswani et al. 2017),
with GPT-style modifications (decoder-only, pre-norm, learned positional embeddings).

Key components built from scratch:
  - MultiHeadAttention (with causal masking)
  - FeedForward (with GELU activation)
  - TransformerBlock (pre-norm: LayerNorm -> Attn -> residual, LayerNorm -> FFN -> residual)
  - PositionalEncoding (both learned and sinusoidal options)
  - TinyLLM (full GPT-style decoder-only model)

Math notation in comments uses:
  B = batch size
  T = sequence length (context window)
  C = embedding dimension (d_model)
  H = number of attention heads
  d_k = C // H (head dimension)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """All hyperparameters for the model in one place."""

    vocab_size: int = 8000  # tokenizer vocab size
    context_length: int = 256  # max sequence length (T)
    d_model: int = 384  # embedding dimension (C)
    n_heads: int = 6  # number of attention heads
    n_layers: int = 6  # number of transformer blocks
    d_ff: int = 1536  # feedforward hidden dim (usually 4*d_model)
    dropout: float = 0.1  # dropout probability
    use_learned_pos_emb: bool = True  # learned vs sinusoidal positional embeddings

    def __post_init__(self):
        assert (
            self.d_model % self.n_heads == 0
        ), f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        self.d_k = self.d_model // self.n_heads

    def num_params(self) -> int:
        """Estimate parameter count (useful for checking model size)."""
        # Embeddings
        emb = self.vocab_size * self.d_model
        pos = self.context_length * self.d_model if self.use_learned_pos_emb else 0
        # Each transformer block
        attn = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        ff = 2 * self.d_model * self.d_ff  # two linear layers
        ln = 4 * self.d_model  # two LayerNorms, 2 params each
        block = attn + ff + ln
        # Final LN + LM head (often tied with embedding)
        head = self.d_model + self.vocab_size * self.d_model  # ln + lm_head
        total = emb + pos + self.n_layers * block + head
        return total


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------


class LearnedPositionalEmbedding(nn.Module):
    """
    Simple learned positional embeddings (used by GPT-2).
    Each position 0..T-1 gets its own embedding vector, trained via backprop.
    """

    def __init__(self, context_length: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(context_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)
        T = x.size(1)
        positions = torch.arange(T, device=x.device)  # (T,)
        return self.embedding(positions)  # (T, C)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encodings (from the original Transformer paper).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    These are NOT learned — they're computed analytically.
    Advantage: can generalize to longer sequences than seen in training.
    """

    def __init__(self, context_length: int, d_model: int):
        super().__init__()
        pe = torch.zeros(context_length, d_model)
        position = torch.arange(0, context_length).unsqueeze(1).float()  # (T, 1)
        # Compute the division term: 10000^(2i/d_model) in log space for stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        # Register as buffer (not a parameter — won't appear in optimizer)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return self.pe[:T]  # (T, C)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention.

    The core operation:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    With multiple heads, we project into H different (d_k)-dim spaces,
    compute attention independently, then concatenate and project back.

    "Causal" means we mask future positions (upper triangle = -inf)
    so the model can only attend to previous tokens — required for
    autoregressive language modeling.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_model = config.d_model

        # Single matrix for Q, K, V projections (merged for efficiency)
        # W_q, W_k, W_v are each (d_model, d_model)
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        # Output projection
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: upper triangular (future positions) set to -inf
        # Shape: (1, 1, T, T) for broadcasting over (B, H, T, T)
        mask = torch.triu(
            torch.ones(config.context_length, config.context_length), diagonal=1
        )
        self.register_buffer("causal_mask", mask.bool())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) input embeddings
        Returns:
            (B, T, C) output embeddings
        """
        B, T, C = x.shape

        # Project to Q, K, V
        # Each: (B, T, C)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (B, T, C) -> (B, H, T, d_k)
        # We split C into (n_heads, d_k) and move head dim to position 1
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # scores: (B, H, T, T)
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Apply causal mask: set future positions to -inf so softmax gives 0
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        # Softmax over last dim (over keys)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        # (B, H, T, T) x (B, H, T, d_k) -> (B, H, T, d_k)
        out = torch.matmul(attn_weights, V)

        # Reassemble heads: (B, H, T, d_k) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        out = self.resid_dropout(self.W_o(out))
        return out


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

    Applied independently to each position.
    d_ff is typically 4 * d_model (a hyperparameter).

    GELU (Gaussian Error Linear Unit) is smoother than ReLU and
    empirically works better for language models.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T, d_ff) -> (B, T, C)
        x = self.fc1(x)
        x = F.gelu(x)  # non-linearity
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block with pre-norm (GPT-2 style).

    Pre-norm vs Post-norm:
        Post-norm (original paper): x = LayerNorm(x + sublayer(x))
        Pre-norm  (GPT-2):          x = x + sublayer(LayerNorm(x))

    Pre-norm is more stable for deep networks.

    Residual connections: x = x + sublayer(x)
    These are crucial for gradient flow in deep networks. Without them,
    gradients vanish and training fails.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.attn(self.ln1(x))
        # Feed-forward with residual connection
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full Language Model
# ---------------------------------------------------------------------------


class TinyLLM(nn.Module):
    """
    A small GPT-style decoder-only language model.

    Architecture:
        Token Embeddings + Positional Embeddings
        -> N x TransformerBlock
        -> LayerNorm
        -> Linear (LM head: d_model -> vocab_size)

    Training objective: next-token prediction (cross-entropy loss).
    The model predicts P(token_t | token_0, ..., token_{t-1}).

    Weight tying: the LM head shares weights with the token embedding matrix.
    This is a common trick that reduces parameters and often improves performance
    (the embedding and the output projection are doing inverse operations).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings: maps token id -> d_model vector
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings
        if config.use_learned_pos_emb:
            self.pos_emb = LearnedPositionalEmbedding(
                config.context_length, config.d_model
            )
        else:
            self.pos_emb = SinusoidalPositionalEncoding(
                config.context_length, config.d_model
            )

        self.emb_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm (pre-norm architecture needs this at the end)
        self.ln_final = nn.LayerNorm(config.d_model)

        # Language model head: projects d_model -> vocab_size logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding weights with LM head
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights (important for stable training)
        self.apply(self._init_weights)
        # Special scaled init for residual projections (from GPT-2 paper)
        for name, param in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TinyLLM initialized: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
        """Standard GPT-2 weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) token ids
            targets: (B, T) shifted token ids for loss computation

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss (if targets provided)
        """
        B, T = input_ids.shape
        assert (
            T <= self.config.context_length
        ), f"Sequence length {T} exceeds context window {self.config.context_length}"

        # Embeddings: token + position
        tok_emb = self.token_emb(input_ids)  # (B, T, C)
        pos_emb = self.pos_emb(input_ids)  # (T, C) - broadcast over batch
        x = self.emb_dropout(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.ln_final(x)  # (B, T, C)

        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten: (B*T, vocab_size) and (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # ignore padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Sampling strategies:
        - Greedy (temperature=0): always pick highest probability token
        - Temperature: divide logits by T before softmax; T<1 = sharper, T>1 = flatter
        - Top-k: sample only from the k most likely tokens
        - Top-p (nucleus): sample from smallest set of tokens with cumulative prob >= p

        Args:
            input_ids: (B, T) prompt token ids
            max_new_tokens: how many new tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            eos_id: stop generation when this token is produced

        Returns:
            (B, T + max_new_tokens) token ids
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            ctx = input_ids[:, -self.config.context_length :]

            # Forward pass
            logits, _ = self(ctx)
            logits = logits[:, -1, :]  # last token's logits: (B, vocab_size)

            if temperature == 0:
                # Greedy decoding
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # Apply temperature
                logits = logits / temperature

                # Top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    kth_val = logits.topk(top_k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth_val, float("-inf"))

                # Top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
                    cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    # Remove tokens once cumulative prob exceeds top_p
                    sorted_to_remove = cumprobs - sorted_logits.softmax(dim=-1) > top_p
                    # Shift right to keep first token above threshold
                    sorted_to_remove[:, 1:] = sorted_to_remove[:, :-1].clone()
                    sorted_to_remove[:, 0] = False
                    # Scatter mask back
                    to_remove = sorted_to_remove.scatter(
                        1, sorted_idx, sorted_to_remove
                    )
                    logits = logits.masked_fill(to_remove, float("-inf"))

                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        return input_ids


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ModelConfig(
        vocab_size=1000,
        context_length=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
    )
    print(f"Estimated params: {cfg.num_params()/1e6:.2f}M")

    model = TinyLLM(cfg)
    x = torch.randint(0, 1000, (2, 64))
    logits, loss = model(x, targets=x)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f} (should be ~log(1000) = {math.log(1000):.4f})")
    print("Model OK ✓")

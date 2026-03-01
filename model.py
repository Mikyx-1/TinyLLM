"""
Transformer Language Model — implemented from scratch.

Architecture follows the original "Attention Is All You Need" paper (Vaswani et al. 2017),
with GPT-style modifications (decoder-only, pre-norm, learned positional embeddings).

Key components built from scratch:
  - MultiHeadAttention (with causal masking + KV cache support)
  - FeedForward (with GELU activation)
  - TransformerBlock (pre-norm: LayerNorm -> Attn -> residual, LayerNorm -> FFN -> residual)
  - PositionalEncoding (both learned and sinusoidal options)
  - TinyLLM (full GPT-style decoder-only model)

  [NEW] KV Cache:
  - KVCache dataclass — stores K and V tensors per layer
  - MultiHeadAttention.forward() accepts an optional cache and updates it in-place
  - TinyLLM.generate() allocates one cache object and passes it through all layers

Math notation in comments uses:
  B = batch size
  T = sequence length (context window)
  C = embedding dimension (d_model)
  H = number of attention heads
  d_k = C // H (head dimension)
"""

import math
from dataclasses import dataclass, field
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
        emb = self.vocab_size * self.d_model
        pos = self.context_length * self.d_model if self.use_learned_pos_emb else 0
        attn = 4 * self.d_model * self.d_model  # Q, K, V, O projections
        ff = 2 * self.d_model * self.d_ff
        ln = 4 * self.d_model
        block = attn + ff + ln
        head = self.d_model + self.vocab_size * self.d_model
        total = emb + pos + self.n_layers * block + head
        return total


# ---------------------------------------------------------------------------
# KV Cache
# ---------------------------------------------------------------------------

# ┌──────────────────────────────────────────────────────────────────────────┐
# │  WHY KV CACHING?                                                         │
# │                                                                          │
# │  During autoregressive generation we produce one token at a time:        │
# │    step 1: process "Hello"           → predict "world"                   │
# │    step 2: process "Hello world"     → predict "!"                       │
# │    step 3: process "Hello world !"   → predict "<eos>"                   │
# │                                                                          │
# │  Without caching, every step re-runs the full forward pass over ALL      │
# │  previous tokens. That means O(T²) total work for a T-token sequence.    │
# │                                                                          │
# │  With KV caching, we observe that:                                       │
# │    • The Keys and Values for past tokens never change between steps.     │
# │    • Only the NEW token produces new K and V vectors.                    │
# │                                                                          │
# │  So we cache K and V from previous steps and only compute Q, K, V        │
# │  for the single new token per step. Then we concatenate the new K/V      │
# │  onto the cache before running attention.                                │
# │                                                                          │
# │  Result: each generation step becomes O(T) instead of O(T²).             │
# │  For long sequences this is a huge win!                                  │
# └──────────────────────────────────────────────────────────────────────────┘


@dataclass
class KVCache:
    """
    Holds the cached Keys and Values for a single attention layer.

    Shapes (when populated):
        k: (B, H, T_past, d_k)
        v: (B, H, T_past, d_k)

    T_past grows by 1 each generation step as we append new tokens.
    """

    k: Optional[torch.Tensor] = None  # cached keys
    v: Optional[torch.Tensor] = None  # cached values

    def update(
        self,
        new_k: torch.Tensor,  # (B, H, T_new, d_k)  — usually T_new == 1
        new_v: torch.Tensor,  # (B, H, T_new, d_k)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new_k / new_v to the running cache along the time dimension.

        Returns the FULL (B, H, T_past + T_new, d_k) tensors that attention
        should run over.
        """
        if self.k is None:
            # First call (prefill phase): nothing cached yet, just store as-is.
            self.k = new_k
            self.v = new_v
        else:
            # Subsequent calls (decode phase): concatenate along T dimension (dim=2).
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)

        return self.k, self.v  # return full accumulated K and V


def build_kv_cache(n_layers: int) -> list[KVCache]:
    """
    Create one empty KVCache per transformer layer.
    Called once at the start of generate() and passed through every step.
    """
    return [KVCache() for _ in range(n_layers)]


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

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        # x: (B, T)
        # start_pos: offset so cached tokens get the right position indices
        T = x.size(1)
        positions = torch.arange(start_pos, start_pos + T, device=x.device)  # (T,)
        return self.embedding(positions)  # (T, C)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encodings (from the original Transformer paper).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, context_length: int, d_model: int):
        super().__init__()
        pe = torch.zeros(context_length, d_model)
        position = torch.arange(0, context_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        T = x.size(1)
        # Slice [start_pos : start_pos+T] to get position encodings
        # for the ACTUAL positions of the tokens in x.
        return self.pe[start_pos : start_pos + T]  # (T, C)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention  (now with KV cache support)
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention — extended with KV cache.

    Two operating modes
    ───────────────────
    1. Training / prefill  (cache=None):
       Full sequence of length T is processed. The causal mask prevents each
       position from seeing future positions. Standard O(T²) attention.

    2. Decode step  (cache provided):
       Only the SINGLE new token (T_new=1) is fed in. We:
         a. Compute Q, K, V for that token.
         b. Append the new K, V to the cache → full K, V of shape (B,H,T_past+1,d_k).
         c. Run attention: Q (for 1 token) attends to ALL past K, V.
         d. No causal mask needed — the new token is always the last one.

    The causal ordering is preserved implicitly: the cache only ever contains
    tokens that came BEFORE the current one.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_model = config.d_model

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask for the FULL-sequence (training) mode.
        # Shape (T, T): position i can attend to positions 0..i (upper tri = True → masked).
        mask = torch.triu(
            torch.ones(config.context_length, config.context_length), diagonal=1
        )
        self.register_buffer("causal_mask", mask.bool())

    def forward(
        self,
        x: torch.Tensor,  # (B, T_new, C)  — T_new=T in training, T_new=1 in decode
        cache: Optional[KVCache] = None,  # KVCache for this layer; None during training
    ) -> torch.Tensor:
        """
        Args:
            x:     Input tensor (B, T_new, C).
            cache: Optional KVCache. If provided we run in incremental decode mode.

        Returns:
            (B, T_new, C) output embeddings.
        """
        B, T_new, C = x.shape

        # ── Step 1: project to Q, K, V ──────────────────────────────────────
        # Each has shape (B, T_new, C)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # ── Step 2: reshape for multi-head attention ─────────────────────────
        # (B, T_new, C) → (B, H, T_new, d_k)
        Q = Q.view(B, T_new, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T_new, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T_new, self.n_heads, self.d_k).transpose(1, 2)

        # ── Step 3: KV cache update (decode mode only) ───────────────────────
        if cache is not None:
            # Append the freshly computed K, V to the running cache.
            # After this, K_full and V_full contain ALL past + current tokens:
            #   shape (B, H, T_past + T_new, d_k)
            K_full, V_full = cache.update(K, V)
            # Q stays (B, H, T_new, d_k) — we only query from the new position(s).
        else:
            # Training mode: no cache, use current-sequence K and V as-is.
            K_full, V_full = K, V

        # ── Step 4: scaled dot-product attention ─────────────────────────────
        # Q:       (B, H, T_new,   d_k)
        # K_full:  (B, H, T_total, d_k)   where T_total = T_past + T_new (or just T_new in training)
        # scores:  (B, H, T_new, T_total)
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(Q, K_full.transpose(-2, -1)) / scale

        # ── Step 5: causal masking ────────────────────────────────────────────
        if cache is None:
            # Training mode: apply standard upper-triangular causal mask.
            # This prevents position i from seeing positions j > i.
            scores = scores.masked_fill(self.causal_mask[:T_new, :T_new], float("-inf"))
        # Decode mode: NO mask needed.
        # The cache only contains *past* tokens (none are in the future relative
        # to the new token), so causality is already guaranteed by construction.

        # ── Step 6: softmax + weighted sum ───────────────────────────────────
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, T_new, T_total)
        attn_weights = self.attn_dropout(attn_weights)

        # (B, H, T_new, T_total) × (B, H, T_total, d_k) → (B, H, T_new, d_k)
        out = torch.matmul(attn_weights, V_full)

        # ── Step 7: reassemble heads ──────────────────────────────────────────
        # (B, H, T_new, d_k) → (B, T_new, C)
        out = out.transpose(1, 2).contiguous().view(B, T_new, C)

        # ── Step 8: output projection ─────────────────────────────────────────
        out = self.resid_dropout(self.W_o(out))
        return out


# ---------------------------------------------------------------------------
# Feed-Forward Network  (unchanged)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2

    This does not interact with the KV cache at all — it processes each
    position independently, so no state needs to be stored here.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Transformer Block  (now threads the KV cache through)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block with pre-norm (GPT-2 style).

    Pre-norm:  x = x + sublayer(LayerNorm(x))

    The KVCache for this block is passed into MultiHeadAttention.forward().
    The FeedForward layer is stateless, so it receives no cache.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Attention sub-layer — passes cache so attention can use/update it
        x = x + self.attn(self.ln1(x), cache=cache)
        # Feed-forward sub-layer — no cache needed
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full Language Model
# ---------------------------------------------------------------------------


class TinyLLM(nn.Module):
    """
    A small GPT-style decoder-only language model, now with KV cache generation.

    KV Cache flow during generate():
    ─────────────────────────────────
    Before the loop:
        caches = build_kv_cache(n_layers)   # list of n_layers empty KVCache objects

    Prefill step (first call, processes the prompt of length T):
        - All T tokens run through all layers at once (standard forward pass).
        - Each layer's KVCache stores K, V of shape (B, H, T, d_k).
        - We use the LAST token's logits to sample the first new token.

    Decode steps (one new token at a time):
        - Only the single new token (shape B×1) is fed to the model.
        - Positional embedding uses start_pos = T_past to get the correct position.
        - Each layer:
            1. Computes K, V for the new token.
            2. Appends them to its KVCache.
            3. Runs attention over the full K_full, V_full (past + new).
        - Result: O(T) work per step instead of O(T²).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        if config.use_learned_pos_emb:
            self.pos_emb = LearnedPositionalEmbedding(
                config.context_length, config.d_model
            )
        else:
            self.pos_emb = SinusoidalPositionalEncoding(
                config.context_length, config.d_model
            )

        self.emb_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_final = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("W_o.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(
                    param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers)
                )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"TinyLLM initialized: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module: nn.Module):
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
        input_ids: torch.Tensor,  # (B, T)
        targets: Optional[torch.Tensor] = None,
        caches: Optional[list[KVCache]] = None,  # [NEW] one KVCache per layer
        start_pos: int = 0,  # [NEW] position offset for embeddings
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: (B, T) token ids.
            targets:   (B, T) shifted token ids for loss (training only).
            caches:    List of KVCache objects, one per layer. None during training.
            start_pos: Position index of the first token in input_ids.
                       During decode steps this equals the number of already-cached tokens,
                       so positional embeddings are correctly offset.

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss (if targets provided, else None)
        """
        B, T = input_ids.shape
        assert (
            T + start_pos <= self.config.context_length
        ), f"Total length {T + start_pos} exceeds context window {self.config.context_length}"

        # ── Token + Positional Embeddings ─────────────────────────────────────
        tok_emb = self.token_emb(input_ids)  # (B, T, C)
        pos_emb = self.pos_emb(input_ids, start_pos)  # (T, C)  ← offset by start_pos
        x = self.emb_dropout(tok_emb + pos_emb)

        # ── Transformer Blocks ────────────────────────────────────────────────
        for i, block in enumerate(self.blocks):
            # Each block gets its own KVCache (or None during training).
            # The block's attention layer will read from and write to this cache.
            cache = caches[i] if caches is not None else None
            x = block(x, cache=cache)

        # ── Final LayerNorm + LM Head ─────────────────────────────────────────
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
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
        Autoregressive generation with KV caching.

        High-level flow
        ───────────────
        1. PREFILL: Run the full prompt through the model in one shot.
           This populates each layer's KVCache with K, V for every prompt token.
           We sample the first new token from the last position's logits.

        2. DECODE LOOP: Feed one token at a time.
           Each step only processes a (B, 1) tensor — much cheaper than re-running
           the whole growing sequence.
           The cache accumulates K, V silently inside each KVCache object.

        Sampling strategies (unchanged):
            - Greedy (temperature=0): always pick the highest-probability token.
            - Temperature: scale logits before softmax.
            - Top-k:        keep only the k most likely tokens.
            - Top-p (nucleus): keep the smallest set summing to probability ≥ p.
        """
        self.eval()

        # ── Allocate one KVCache per layer ────────────────────────────────────
        # These objects will accumulate K, V tensors across all generation steps.
        caches = build_kv_cache(self.config.n_layers)

        # ── PREFILL ───────────────────────────────────────────────────────────
        # Process the entire prompt at once.
        # After this call each cache[i].k / cache[i].v has shape (B, H, T_prompt, d_k).
        prompt_len = input_ids.shape[1]
        logits, _ = self(input_ids, caches=caches, start_pos=0)

        # We only need the last token's logits to generate the next token.
        # Shape: (B, vocab_size)
        logits = logits[:, -1, :]
        next_token = self._sample(logits, temperature, top_k, top_p)  # (B, 1)

        # Append the first generated token to our running sequence.
        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_id is not None and (next_token == eos_id).all():
            return input_ids

        # ── DECODE LOOP ───────────────────────────────────────────────────────
        # start_pos advances by 1 each step (= number of tokens already in cache).
        for step in range(1, max_new_tokens):
            # Current position of the token we're about to process.
            # start_pos = prompt_len + step - 1
            # (the cache already holds all tokens before this one)
            current_pos = prompt_len + step - 1

            # Feed ONLY the last generated token. Shape: (B, 1).
            # Positional embedding will use index `current_pos` so the model
            # knows where in the sequence this token lives.
            logits, _ = self(
                next_token,
                caches=caches,
                start_pos=current_pos,
            )

            logits = logits[:, -1, :]  # (B, vocab_size)
            next_token = self._sample(logits, temperature, top_k, top_p)  # (B, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        return input_ids

    # ── Internal helper ───────────────────────────────────────────────────────

    def _sample(
        self,
        logits: torch.Tensor,  # (B, vocab_size)
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:  # (B, 1)
        """
        Apply temperature / top-k / top-p filtering, then sample (or argmax).
        Extracted into a helper to keep generate() readable.
        """
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature

        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            kth_val = logits.topk(top_k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth_val, float("-inf"))

        if top_p is not None:
            sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_to_remove = cumprobs - sorted_logits.softmax(dim=-1) > top_p
            sorted_to_remove[:, 1:] = sorted_to_remove[:, :-1].clone()
            sorted_to_remove[:, 0] = False
            to_remove = sorted_to_remove.scatter(1, sorted_idx, sorted_to_remove)
            logits = logits.masked_fill(to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)  # (B, 1)


# ---------------------------------------------------------------------------
# Quick test — includes a timing comparison (cached vs uncached)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    cfg = ModelConfig(
        vocab_size=1000,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
    )
    print(f"Estimated params: {cfg.num_params()/1e6:.2f}M")

    model = TinyLLM(cfg)
    model.eval()

    # ── Correctness check ─────────────────────────────────────────────────────
    x = torch.randint(0, 1000, (2, 64))
    logits, loss = model(x, targets=x)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f} (should be ~log(1000) = {math.log(1000):.4f})")
    print("Forward pass OK ✓")

    # ── Generation with KV cache ──────────────────────────────────────────────
    prompt = torch.randint(0, 1000, (1, 8))
    N_NEW = 40

    # With KV cache (new)
    t0 = time.perf_counter()
    out_cached = model.generate(prompt, max_new_tokens=N_NEW, temperature=0)
    t_cached = time.perf_counter() - t0

    print(f"\nGenerated {N_NEW} tokens")
    print(f"  With KV cache:    {t_cached*1000:.1f} ms")
    print(f"  Output shape: {out_cached.shape}")
    print("KV cache generation OK ✓")

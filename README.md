# TinyLLM — Build an LLM From Scratch

A minimal, heavily-commented GPT-style language model (~20M parameters) for learning purposes.
Every component is implemented from scratch — no HuggingFace, no pre-built transformers.

```
tinyllm/
├── tokenizer.py      # BPE tokenizer (from scratch)
├── model.py          # Transformer architecture (from scratch)
├── data_utils.py     # Data pipeline + PyTorch Dataset
├── train.py          # Multi-GPU training loop with DDP
├── generate.py       # Text generation / inference
└── requirements.txt
```

---

## What You'll Learn

### 1. BPE Tokenization (`tokenizer.py`)
**Byte Pair Encoding** is how GPT-2/GPT-4 turn raw text into numbers.

The algorithm:
1. Start with characters as vocabulary
2. Count all adjacent token pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until vocab size is reached

Key insight: `"lower"` might tokenize as `["▁low", "er"]` after training,
while `"lowest"` → `["▁low", "est"]`. Common subwords are merged, rare words
stay as characters. This handles out-of-vocabulary words gracefully.

### 2. Transformer Architecture (`model.py`)

**Multi-Head Self-Attention** — the core of the transformer:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```
- Q (Query): "What am I looking for?"
- K (Key): "What information do I have?"  
- V (Value): "What do I actually return?"
- The division by `sqrt(d_k)` prevents softmax from saturating in high dimensions
- Causal mask: future tokens get -inf → 0 probability (autoregressive property)

**Multiple heads**: Run attention H times in parallel with different projections.
Each head can learn different relationship types (syntax, semantics, coreference, etc.)

**Feed-Forward Network**:
```
FFN(x) = GELU(xW1 + b1)W2 + b2
```
Applied position-wise. Acts like "memory" for the model — stores factual associations.

**Pre-norm vs Post-norm**:
- Original paper: `LayerNorm(x + sublayer(x))` (post-norm)
- GPT-2 style: `x + sublayer(LayerNorm(x))` (pre-norm)
- Pre-norm is more stable for deep networks (gradients flow more easily)

**Residual connections**: `x = x + sublayer(x)`
Without these, gradients vanish in deep networks. These are what make it
possible to train 100+ layer networks.

**Weight tying**: The token embedding matrix and the LM head share weights.
The embedding maps token → vector, the LM head maps vector → token logits.
These are approximately inverse operations, so sharing parameters works well
and reduces the parameter count by `vocab_size * d_model`.

### 3. Training (`train.py`)

**Causal Language Modeling objective**:
Given tokens [t1, t2, t3, t4], predict [t2, t3, t4, t5].
Loss = cross-entropy averaged over all positions.

**Learning rate schedule**:
```
Warmup: lr = max_lr * (iter / warmup_iters)   # linear ramp-up
Cosine: lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * progress))
```
Warmup prevents instability at the start. Cosine decay helps find a better minimum.

**Gradient clipping**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
If the gradient norm exceeds 1.0, all gradients are scaled down proportionally.
Without this, transformers can have sudden loss spikes.

**Gradient accumulation**:
Simulates larger batches without more memory:
```
for micro_step in range(grad_accumulation_steps):
    loss = model(batch) / grad_accumulation_steps
    loss.backward()
optimizer.step()  # Only step once per "logical" batch
```

**Mixed precision (bfloat16)**:
- float32: 32 bits, range ±3.4×10³⁸, precision ~7 decimal digits
- bfloat16: 16 bits, same range as float32, precision ~3 digits
- Uses ~50% less VRAM, matrix multiplications are 2-4x faster on modern GPUs
- bfloat16 > float16 for training (same dynamic range, no loss scaling needed)

### 4. Multi-GPU Training with DDP (`train.py`)

**DistributedDataParallel** is PyTorch's recommended multi-GPU strategy:

```
GPU 0: model copy → forward(batch_shard_0) → backward → gradients
GPU 1: model copy → forward(batch_shard_1) → backward → gradients
                                                ↓
                              All-Reduce (NCCL): avg gradients
                                                ↓
                          Both GPUs update weights identically
```

Each GPU processes a different data shard simultaneously.
After backward, gradients are averaged across all GPUs via NCCL.
This gives approximately linear speedup with number of GPUs.

**torchrun** handles process spawning and sets environment variables:
- `RANK`: global process index (0 = master)
- `LOCAL_RANK`: GPU index on this node
- `WORLD_SIZE`: total number of processes

### 5. Text Generation (`generate.py`)

**Autoregressive generation**: feed tokens in → sample next token → append → repeat

**Sampling strategies**:
- **Greedy** (temp=0): always pick argmax. Deterministic but repetitive.
- **Temperature**: divide logits by T before softmax.
  - T < 1 → sharper distribution (more confident, less creative)
  - T > 1 → flatter distribution (more random, more creative)
- **Top-k**: sample only from the k highest-probability tokens.
  Prevents sampling very unlikely tokens.
- **Top-p (nucleus)**: sample from the smallest set of tokens
  whose cumulative probability ≥ p. Adaptive — uses more tokens
  when the model is uncertain.

---

## Quick Start

### Install
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Train (single GPU)
```bash
python train.py
```

### Train (dual RTX 3060 — your setup!)
```bash
torchrun --nproc_per_node=2 train.py
```

### Train with custom settings
```bash
torchrun --nproc_per_node=2 train.py \
    --batch_size 32 \
    --max_iters 5000 \
    --d_model 512 \
    --n_layers 8
```

### Generate text
```bash
python generate.py \
    --checkpoint checkpoints/latest.pt \
    --prompt "To be or not to be" \
    --temperature 0.8 \
    --top_k 50 \
    --max_tokens 300
```

---

## Model Size Reference

| Config          | d_model | n_layers | n_heads | Params  | VRAM (bfloat16) |
|-----------------|---------|----------|---------|---------|-----------------|
| Tiny (default)  | 384     | 6        | 6       | ~20M    | ~500MB          |
| Small           | 512     | 8        | 8       | ~45M    | ~1.1GB          |
| Medium          | 768     | 12       | 12      | ~125M   | ~3GB            |

Your dual RTX 3060 (12GB each) can comfortably train the "Small" config
and probably the "Medium" with gradient accumulation.

---

## Expected Training Behavior

On TinyShakespeare with the default config:
- Initial loss: ~8.3 (random, ≈ log(vocab_size))
- After 500 iters: ~4.0 (learning word-level patterns)
- After 2000 iters: ~2.5 (learning phrase patterns)  
- After 5000 iters: ~1.8–2.0 (reasonable Shakespeare-like text)

A validation loss of ~1.5 would be excellent for this dataset size.
The model will never be "perfect" — TinyShakespeare is only ~1MB.

---

## Key Papers to Read

1. **Attention Is All You Need** (Vaswani et al., 2017) — original Transformer
2. **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019) — GPT-2
3. **An Image is Worth 16x16 Words** (Dosovitskiy et al., 2020) — ViT, shows transformers work everywhere
4. **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022) — Chinchilla scaling laws

---

## Experiment Ideas

Once you've got the base model training, try:

1. **Sinusoidal vs learned positional embeddings** — toggle `--use_learned_pos_emb False`
2. **Different activation functions** — swap GELU for ReLU or SiLU in `model.py`
3. **RoPE positional encoding** — rotary embeddings used in LLaMA (add to `model.py`)
4. **Flash Attention** — drop-in replacement for MultiHeadAttention with better memory complexity
5. **Different datasets** — try the Bible, PG books, or any plain text corpus
6. **Scaling laws** — train the same config for 10x more data. What happens to val loss?
7. **SwiGLU activation** — replace FFN with `SwiGLU(x) = (xW + b) * σ(xV + c)` as used in LLaMA

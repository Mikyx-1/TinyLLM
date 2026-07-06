# TinyLLM: Full Pipeline — Pretrain → SFT → Reasoning → RLHF (Reward Model + PPO) → Docs

## Context

TinyLLM is a from-scratch, heavily-commented GPT implementation (no HF/transformers) whose whole point is
to teach the reader real internals. It currently has: a custom BPE tokenizer, a GPT-style `TinyLLM` model
(`model/`), a base-LM pretraining script (`pretrain.py`), and an SFT script (`train.py`) trained on a
50-example identity Q&A set. There is **no reasoning-specific training and no RLHF anywhere yet**.

The goal is to extend this into a complete, real (not simulated) LLM post-training pipeline — pretrain →
instruction-SFT → chain-of-thought reasoning SFT → reward model → PPO RLHF — using real downloaded datasets
(confirmed reachable and schema-checked: Stanford Alpaca, GSM8K, Anthropic hh-rlhf `helpful-base`), all
built from scratch in the same style as the existing code, then document every stage with real metrics and
before/after samples (not fabricated numbers) in `docs/` plus a rendered HTML summary artifact.

Hardware confirmed available: RTX 5070 Ti (16GB), 48 cores, 61GB RAM, torch 2.9/cu130 — enough to run a
~22M-param model through all 5 stages in a single working session.

## Model size & shared config for the whole pipeline

Use the README's "Small" config for the whole run (bigger than default "Tiny" so reasoning/RLHF effects are
visible, still small enough to train fast): `d_model=512, n_heads=8, n_layers=8, d_ff=1024`.
Bump `context_length=384` (needs to fit GSM8K CoT chains) and `vocab_size=12000` (needs to cover
instructions + math + dialogue) **from the start**, since `use_learned_pos_emb=True` bakes in `context_length`
and every later stage resumes from the previous stage's checkpoint — these must stay fixed across all stages.

**One tokenizer trained once, reused everywhere.** Add `<THINK>` / `</THINK>` to
`tokenizer/constants.py::SPECIAL_TOKENS` (after existing 4) so reasoning traces get dedicated tokens instead
of being fragmented by BPE. Train this tokenizer once on the concatenation of all corpora (Alpaca + GSM8K +
hh-rlhf text) at vocab_size=12000, save to `checkpoints/tokenizer.json`, and pass `force_retrain_tokenizer=False`
in every later stage so ids stay consistent (required — the reward model and policy must share a vocabulary).

## Architecture change: extract a shared `TransformerTrunk`

Add `model/trunk.py::TransformerTrunk` holding `token_emb, pos_emb, emb_dropout, blocks, ln_final` (exactly
what `TinyLLM` currently does before its `lm_head`). `TinyLLM` becomes a thin wrapper: `self.trunk = TransformerTrunk(config)`,
`self.lm_head` tied to `self.trunk.token_emb.weight`, forward signature/behavior unchanged. This lets a
`RewardModel` and a PPO `ActorCritic` reuse the exact same trunk class (and thus load the exact same
`trunk.*` state-dict keys from an SFT checkpoint) with a different head on top, with zero duplicated
attention/block code. Since this refactor happens *before* any training run in this project, there's no
checkpoint-migration concern — every checkpoint produced from here on uses the new key layout.
`model/generate.py` needs no changes: it only calls `model(input_ids, caches=..., start_pos=...)` and reads
the first tuple element, which stays `logits` for every model class (`TinyLLM`, `ActorCritic`).

## Stage 0 — Data & tokenizer prep (`prepare_pipeline_data.py`, new)

One script, run once, that:
1. Downloads Alpaca (`alpaca_data.json`, 52002 rows), GSM8K (`train.jsonl`/`test.jsonl`), hh-rlhf
   `helpful-base` (`train.jsonl.gz`, gunzip) into `data/raw/`.
2. **Alpaca → SFT set**: map `{instruction (+ "\n" + input if present), output}` → `{question, answer}`;
   subsample ~5000; merge with the existing 50-example `data/tinyllm_dataset.json` (kept as-is so identity
   answers survive) into `data/sft_dataset.json`.
3. **GSM8K → reasoning set**: strip `<<calc>>` annotations (regex `<<[^>]*>>`), split each `answer` on
   `\n#### ` into `(reasoning, final_number)`. Render as
   `<BOS> Question: {question}\n<THINK> {reasoning} </THINK>\nAnswer: {final_number} <EOS>`. Drop examples
   whose tokenized length exceeds `context_length` (safety margin). Save `data/reasoning_dataset.json`
   (question/reasoning/answer fields, ~2500-3500 examples expected after filtering).
4. **hh-rlhf → preference set**: parse `chosen`/`rejected` transcripts; **keep only single-turn** examples
   (exactly one `Human:`/`Assistant:` pair — chosen/rejected share the human turn and diverge only on the
   assistant reply, confirmed from the real schema), extract `{prompt, chosen, rejected}`. Save
   `data/preference_dataset.json` (subsample to ~6000-8000 pairs if more are available, to bound RM training time).
5. Trains the shared BPE tokenizer once on the concatenation of all rendered text from the 3 sets above,
   saves to `checkpoints/tokenizer.json`.
6. Prints dataset sizes/stats for the doc later.

`data_utils.py` additions: `prepare_reasoning_data(...)` (mirrors `prepare_custom_data`, but uses the CoT
template) and `prepare_preference_data(...)` (returns a `PreferenceDataset` yielding
`(chosen_ids, chosen_len, rejected_ids, rejected_len)`, right-padded by a `collate_fn`; correctness note:
causal attention makes right-padding provably safe for last-real-token pooling, no mask changes needed
anywhere).

## Stage 1 — Base pretraining (existing `pretokenize.py` + `pretrain.py`, unmodified)

Build a plain-text corpus by concatenating Alpaca instruction+input+output text and GSM8K question+answer
text (no special formatting — plain unsupervised CLM) into `data/raw_text/corpus.txt`, run
`pretokenize.py --train_tokenizer=False` (tokenizer already trained in Stage 0) then `pretrain.py` for
~3000 iters, `checkpoint_dir=checkpoints/pretrain`. This gives the model basic English/number fluency before
instruction tuning — reusing existing scripts as-is, just pointed at the new in-domain corpus and config.

## Stage 2 — Instruction SFT (existing `train.py`, unmodified)

`train.py --dataset_path data/sft_dataset.json --resume_from checkpoints/pretrain/pretrain_final.pt
--context_length 384 --vocab_size 12000 --checkpoint_dir checkpoints/sft`. Verify with `inference.py` on a
handful of prompts (identity questions + generic instructions) before moving on.

## Stage 3 — Reasoning SFT (`train_reasoning.py`, new — mirrors `train.py`'s structure/loop, per repo's
existing convention of one script per stage e.g. `pretrain.py`/`train.py` rather than a shared library)

Same training loop as `train.py` (LR schedule, AdamW groups, grad clip/accum, checkpointing) but backed by
`data_utils.prepare_reasoning_data` and resuming from `checkpoints/sft/final.pt`. Output:
`checkpoints/reasoning/`. Verification: hold out ~200 GSM8K reasoning examples, measure exact-match accuracy
on the final numeric answer before vs. after this stage (expect low absolute accuracy at 22M params — that's
an honest, documented finding, not a bug) — write a small `eval_reasoning.py` helper for this.

## Stage 4 — Reward Model (`model/reward_model.py` + `train_reward_model.py`, new)

- `model/reward_model.py::RewardModel`: `self.trunk = TransformerTrunk(config)`, `self.reward_head =
  nn.Linear(d_model, 1, bias=False)` (small init). `forward(input_ids, seq_lengths)` pools the hidden state
  at each row's true last token (`seq_lengths - 1`) and returns a scalar per example.
- `bradley_terry_loss(r_chosen, r_rejected) = -F.logsigmoid(r_chosen - r_rejected).mean()`.
- Both `chosen`/`rejected` rendered with the same `<BOS> Question: {prompt}\nAnswer: {response} <EOS>`
  template the policy uses, so the RM scores text in the exact format the policy will generate.
- Init trunk from `checkpoints/reasoning/final.pt` (`load_state_dict(strict=False)` — only `reward_head` is new).
- Train on `data/preference_dataset.json` (train/val split), track **pairwise val accuracy**
  (`(r_chosen > r_rejected).mean()`) as the headline metric — this is the number that goes in the docs.
  1-2 epochs, lr ~2e-5, checkpoint to `checkpoints/reward_model/`.

## Stage 5 — PPO RLHF (`model/actor_critic.py` + `train_ppo.py`, new)

- `model/actor_critic.py::ActorCritic`: same trunk + tied `lm_head` as `TinyLLM`, plus `value_head =
  nn.Linear(d_model, 1)`. `forward(input_ids, caches=None, start_pos=0)` returns `(logits, values)` —
  matches `TinyLLM`'s calling convention exactly, so `model/generate.py` works unmodified for rollout
  generation (it only reads the first tuple element).
- Policy initialized from `checkpoints/reasoning/final.pt` (trunk + lm_head load; value_head fresh).
  Frozen reference = plain `TinyLLM` loaded from the same checkpoint, `requires_grad_(False)`. Frozen RM =
  `checkpoints/reward_model/best.pt`.
- Prompts sampled from the **hh-rlhf preference set's prompts** (not GSM8K/Alpaca — must match what the RM
  was trained to judge). Bucket by tokenized prompt length to batch `generate()` calls without needing
  padding support in generation.
- Per rollout cycle: `generate()` responses (temperature/top-k sampling, not greedy) → build a
  `response_mask` from each row's own first `<EOS>` (note: `generate()`'s stopping condition only fires once
  *every* row in the batch has emitted EOS, so finished rows keep "generating" garbage that must be masked
  out downstream) → one extra no-cache forward pass each through policy and frozen reference for
  `old_logprobs`/`ref_logprobs`/`values_old` → score full responses with the RM (pooled at each response's
  true final token) → per-token reward = `-kl_beta * (old_logprobs - ref_logprobs)` plus RM score added at
  each response's last valid token → GAE(λ) advantages (whitened) and returns.
- PPO update: `ppo_epochs` passes over shuffled minibatches — clipped surrogate objective (`clip_epsilon`),
  value MSE loss, entropy bonus, `approx_kl` early-stop safeguard, grad clip 1.0, AdamW at a much lower LR
  than SFT (PPO is fragile).
- Starting hyperparameters: rollout batch 32-64, `max_new_tokens` 64-96, `ppo_epochs=4`, minibatch 8-16,
  `clip_epsilon=0.2`, `gamma=1.0`, `gae_lambda=0.95`, `kl_beta≈0.1`, `vf_coef=0.5`, `ent_coef=0.01`,
  lr 1e-6–1e-5, `target_kl≈0.02`, ~150-250 update cycles.
- Log every cycle to `checkpoints/ppo/metrics.jsonl` (mean RM reward, mean KL-vs-reference, policy loss,
  value loss, entropy) — this is the real data the docs/artifact will chart. Save policy checkpoints
  periodically to `checkpoints/ppo/`.
- Verification: generate on a fixed held-out prompt set with the reasoning-SFT policy vs. the final PPO
  policy, side-by-side, plus RM score of each — the before/after comparison for the docs.

## Documentation (`docs/`, new) + rendered artifact

Each stage script writes real metrics (loss/accuracy/reward/KL) to a `metrics.jsonl` in its checkpoint dir;
docs and the artifact read from these — no fabricated numbers.

- `docs/00_overview.md` — pipeline diagram, how to reproduce every stage end-to-end.
- `docs/01_tokenizer.md`, `docs/02_architecture.md` — BPE and transformer math (attention, FFN, pre-norm,
  weight tying), consolidating/deepening what's in the current README.
- `docs/03_pretraining.md`, `docs/04_sft.md` — setup, loss curves, sample generations.
- `docs/05_reasoning.md` — CoT format, GSM8K accuracy before/after, honest discussion of a 22M model's
  limits on arithmetic reasoning.
- `docs/06_reward_model.md` — Bradley-Terry derivation, pairwise val accuracy, example scored pairs.
- `docs/07_rlhf_ppo.md` — PPO objective/GAE/KL-control math, reward & KL curves, before/after samples.
- `docs/08_results_and_limitations.md` — end-to-end summary, lessons learned.
- Update `README.md` to link into `docs/` with a short pipeline table.
- Final HTML artifact (via the `dataviz` skill for chart styling): one-page visual summary of the whole
  pipeline — architecture diagram, per-stage loss/reward/KL charts from the real `metrics.jsonl` files, and
  before/after generation examples.

## Execution approach

Work through stages 0→5 sequentially (each depends on the previous checkpoint); run training scripts via
background Bash, monitor logs, sanity-check with `inference.py`/`eval_reasoning.py` before advancing to the
next stage; use TodoWrite to track the ~12 discrete steps across this session. Documentation and the
artifact are written last, once every stage's real metrics exist.

## Verification

- Stage 0: print dataset sizes, spot-check a few rendered examples of each format.
- Stage 1-3: loss curves trend down; `inference.py` outputs coherent completions; reasoning eval script
  reports exact-match accuracy before/after Stage 3.
- Stage 4: pairwise val accuracy meaningfully > 50%.
- Stage 5: mean RM reward trends up while KL-vs-reference stays bounded (no collapse); qualitative
  before/after samples show a discernible style/preference shift.
- Docs/artifact: every number/chart traceable to a real `metrics.jsonl`, not invented.

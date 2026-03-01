"""
RLHF Pipeline using Hugging Face TRL Library
=============================================
This script demonstrates a complete RLHF pipeline:
  1. Supervised Fine-Tuning (SFT)
  2. Reward Model Training
  3. PPO (Proximal Policy Optimization) Fine-Tuning
  4. Before/After Comparison

Libraries: transformers, trl, datasets, torch, accelerate, peft

Install:
  pip install trl transformers datasets accelerate torch peft
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# (vllm uninstalled — no conflict fix needed)

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          TrainingArguments, pipeline)
from trl import (PPOConfig, PPOTrainer, RewardConfig, RewardTrainer, SFTConfig,
                 SFTTrainer)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# STEP 0: SYNTHETIC DATASETS
# ─────────────────────────────────────────────
# In production, replace with real HF datasets e.g. "Anthropic/hh-rlhf"

SFT_DATA = [
    {
        "text": "Question: What is the capital of France?\nAnswer: The capital of France is Paris."
    },
    {
        "text": "Question: How do I stay healthy?\nAnswer: Exercise regularly, eat nutritious food, sleep well, and drink plenty of water."
    },
    {
        "text": "Question: What is machine learning?\nAnswer: Machine learning is a subset of AI where models learn patterns from data."
    },
    {
        "text": "Question: How do I make a good decision?\nAnswer: Gather information, consider alternatives, weigh pros and cons, then decide."
    },
    {
        "text": "Question: What is RLHF?\nAnswer: RLHF fine-tunes language models using human preference signals to align them with human values."
    },
]

PREFERENCE_DATA = [
    {
        "prompt": "Question: What is the capital of France?",
        "chosen": " The capital of France is Paris, a vibrant city known for culture and history.",
        "rejected": " idk maybe london or something",
    },
    {
        "prompt": "Question: How do I stay healthy?",
        "chosen": " Regular exercise, a balanced diet, and good sleep are key to staying healthy.",
        "rejected": " Just eat whatever you want, it doesn't matter.",
    },
    {
        "prompt": "Question: What is machine learning?",
        "chosen": " Machine learning enables computers to learn from data and improve automatically.",
        "rejected": " its some computer thing i think",
    },
    {
        "prompt": "Question: How do I make a good decision?",
        "chosen": " Consider all available information, weigh pros and cons, and align your choice with your goals.",
        "rejected": " Just pick randomly, it's fine.",
    },
    {
        "prompt": "Question: What is RLHF?",
        "chosen": " RLHF fine-tunes language models with human feedback to produce safer and more helpful responses.",
        "rejected": " RLHF is some training method or whatever.",
    },
]

# ─────────────────────────────────────────────
# STEP 1: SUPERVISED FINE-TUNING (SFT)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1: Supervised Fine-Tuning (SFT)")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

sft_dataset = Dataset.from_list(SFT_DATA)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

sft_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# SFTConfig is the modern replacement for passing TrainingArguments to SFTTrainer
sft_config = SFTConfig(
    output_dir="./sft_output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    fp16=torch.cuda.is_available(),
    dataset_text_field="text",  # column name in sft_dataset
    max_length=128,
)

sft_trainer = SFTTrainer(
    model=sft_model,
    args=sft_config,
    train_dataset=sft_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,  # replaces 'tokenizer=' in newer TRL
)

print("Training SFT model...")
sft_trainer.train()
sft_trainer.save_model("./sft_model")
tokenizer.save_pretrained("./sft_model")
print("✓ SFT training complete → ./sft_model")


# ─────────────────────────────────────────────
# STEP 2: REWARD MODEL TRAINING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Reward Model Training")
print("=" * 60)

reward_tokenizer = AutoTokenizer.from_pretrained("./sft_model")
reward_tokenizer.pad_token = reward_tokenizer.eos_token


def preprocess_reward(example):
    """Tokenize chosen and rejected pairs for RewardTrainer."""
    chosen_tokens = reward_tokenizer(
        example["prompt"] + example["chosen"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    rejected_tokens = reward_tokenizer(
        example["prompt"] + example["rejected"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    return {
        "input_ids_chosen": chosen_tokens["input_ids"],
        "attention_mask_chosen": chosen_tokens["attention_mask"],
        "input_ids_rejected": rejected_tokens["input_ids"],
        "attention_mask_rejected": rejected_tokens["attention_mask"],
    }


reward_dataset = Dataset.from_list(PREFERENCE_DATA).map(preprocess_reward)

reward_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1
)
reward_model.config.pad_token_id = reward_tokenizer.eos_token_id

reward_config = RewardConfig(
    output_dir="./reward_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=1e-4,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
    max_length=128,
    fp16=torch.cuda.is_available(),
)

reward_trainer = RewardTrainer(
    model=reward_model,
    processing_class=reward_tokenizer,  # replaces 'tokenizer=' in newer TRL
    args=reward_config,
    train_dataset=reward_dataset,
)

print("Training Reward model...")
reward_trainer.train()
reward_trainer.save_model("./reward_model")
reward_tokenizer.save_pretrained("./reward_model")
print("✓ Reward model training complete → ./reward_model")


# ─────────────────────────────────────────────
# STEP 3: PPO FINE-TUNING (RLHF)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: PPO Fine-Tuning (RLHF)")
print("=" * 60)

# TRL >= 0.9 redesigned PPOTrainer as a full Trainer subclass.
# Required args: model, ref_model, reward_model, value_model, train_dataset.

ppo_tokenizer = AutoTokenizer.from_pretrained("./sft_model", padding_side="left")
ppo_tokenizer.pad_token = ppo_tokenizer.eos_token

# Policy model (what we want to improve via RL) — causal LM
policy_model = AutoModelForCausalLM.from_pretrained("./sft_model")

# Reference model — frozen copy of policy for KL penalty — causal LM
ref_model = AutoModelForCausalLM.from_pretrained("./sft_model")

# Value model — must be SequenceClassification (needs .score attribute)
value_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=1
)
value_model.config.pad_token_id = ppo_tokenizer.eos_token_id

# Reward model — the classifier trained in Step 2 — SequenceClassification
ppo_reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./reward_model", num_labels=1
)
ppo_reward_model.config.pad_token_id = ppo_tokenizer.eos_token_id

# Dataset of prompts PPOTrainer will iterate over
ppo_prompts = [
    "Question: What is the capital of France?\nAnswer:",
    "Question: How do I stay healthy?\nAnswer:",
    "Question: What is machine learning?\nAnswer:",
    "Question: How do I make a good decision?\nAnswer:",
    "Question: What is RLHF?\nAnswer:",
]


def tokenize_prompt(example):
    return ppo_tokenizer(
        example["query"],
        truncation=True,
        max_length=64,
        padding="max_length",
    )


ppo_dataset = (
    Dataset.from_dict({"query": ppo_prompts})
    .map(tokenize_prompt)
    .remove_columns(["query"])  # collator only wants input_ids + attention_mask
)

ppo_config = PPOConfig(
    output_dir="./ppo_model",
    learning_rate=1.41e-5,
    num_ppo_epochs=4,
    num_mini_batches=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    report_to="none",
    save_strategy="no",
    response_length=50,
    missing_eos_penalty=1.0,
    eval_strategy="no",  # disable eval to skip generate_completions
)

ppo_trainer = PPOTrainer(
    args=ppo_config,
    model=policy_model,
    ref_model=ref_model,
    reward_model=ppo_reward_model,
    value_model=value_model,
    train_dataset=ppo_dataset,
    eval_dataset=ppo_dataset,  # required: used for generate_completions logging
    processing_class=ppo_tokenizer,
)

# Monkey-patch generate_completions to a no-op — this TRL version ignores
# eval_strategy="no" and always calls it after each epoch, but crashes when
# eval_dataset rows are None. Safe to skip: it's only a logging/display step.
ppo_trainer.generate_completions = lambda *a, **kw: None

print("Running PPO training (this calls ppo_trainer.train() internally)...")
ppo_trainer.train()

ppo_trainer.save_model("./ppo_model")
ppo_tokenizer.save_pretrained("./ppo_model")
print("✓ PPO training complete → ./ppo_model")

# Reward pipeline for Step 4 scoring
reward_pipe = pipeline(
    "text-classification",
    model="./reward_model",
    tokenizer=reward_tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    truncation=True,
    max_length=128,
)


def get_reward(texts):
    results = reward_pipe(texts)
    return [torch.tensor(r["score"]) for r in results]


# ─────────────────────────────────────────────
# STEP 4: BEFORE vs AFTER COMPARISON
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Effectiveness Comparison (Before vs After RLHF)")
print("=" * 60)

test_prompts = [
    "Question: What is the capital of France?\nAnswer:",
    "Question: How do I stay healthy?\nAnswer:",
    "Question: What is RLHF?\nAnswer:",
]

base_gen = pipeline(
    "text-generation",
    model=MODEL_NAME,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=60,
    do_sample=True,
    top_k=50,
    pad_token_id=tokenizer.eos_token_id,
)

rlhf_gen = pipeline(
    "text-generation",
    model="./ppo_model",
    tokenizer=ppo_tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=60,
    do_sample=True,
    top_k=50,
    pad_token_id=ppo_tokenizer.eos_token_id,
)

base_scores, rlhf_scores = [], []

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    print("-" * 50)

    base_out = base_gen(prompt)[0]["generated_text"][len(prompt) :]
    rlhf_out = rlhf_gen(prompt)[0]["generated_text"][len(prompt) :]

    base_reward = get_reward([prompt + base_out])[0].item()
    rlhf_reward = get_reward([prompt + rlhf_out])[0].item()

    base_scores.append(base_reward)
    rlhf_scores.append(rlhf_reward)

    print(f"[BASE GPT-2]  reward={base_reward:.3f}\n  {base_out.strip()}")
    print(f"[RLHF Model]  reward={rlhf_reward:.3f}\n  {rlhf_out.strip()}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
avg_base = np.mean(base_scores)
avg_rlhf = np.mean(rlhf_scores)
improvement = ((avg_rlhf - avg_base) / abs(avg_base)) * 100 if avg_base != 0 else 0.0

print(f"  Avg Reward  (Base GPT-2) : {avg_base:.4f}")
print(f"  Avg Reward  (RLHF Model) : {avg_rlhf:.4f}")
print(f"  Improvement              : {improvement:+.1f}%")
print("=" * 60)
print("\nRLHF pipeline complete!")

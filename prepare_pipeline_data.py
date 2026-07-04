"""
Stage 0 — Data & tokenizer prep for the full pretrain -> SFT -> reasoning -> reward model -> PPO pipeline.

Downloads Alpaca, GSM8K, and Anthropic hh-rlhf (helpful-base), builds the per-stage dataset files, trains
the ONE shared BPE tokenizer on the concatenation of all of them, and builds the plain-text corpus Stage 1
(pretrain.py) trains on. Run once; every later stage resumes from files this script produces.

USAGE:
    python prepare_pipeline_data.py

Outputs:
    data/raw/{alpaca_data.json, gsm8k_train.jsonl, gsm8k_test.jsonl, hh_train.jsonl, hh_test.jsonl}
    data/sft_dataset.json         — Alpaca subsample + existing tinyllm_dataset.json, {question, answer}
    data/reasoning_dataset.json   — GSM8K CoT, {question, reasoning, answer}
    data/preference_dataset.json  — hh-rlhf single-turn pairs, {prompt, chosen, rejected}
    data/raw_text/corpus.txt      — plain-text corpus for Stage 1 unsupervised pretraining
    checkpoints/tokenizer.json    — shared BPE tokenizer, trained once, reused by every later stage
"""

import argparse
import gzip
import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass

from data_utils import QA_TEMPLATE
from tokenizer import BPETokenizer

CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")

RAW_URLS = {
    "alpaca_data.json": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    "gsm8k_train.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "hh_train.jsonl.gz": "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/train.jsonl.gz",
    "hh_test.jsonl.gz": "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/test.jsonl.gz",
}

REASONING_TEMPLATE = "<BOS> Question: {question}\n<THINK> {reasoning} </THINK>\nAnswer: {answer} <EOS>"
# Same template the reward model / PPO policy will use in Stage 4-5, so the tokenizer already
# covers that surface form.
PREFERENCE_TEMPLATE = "<BOS> Question: {prompt}\nAnswer: {response} <EOS>"


@dataclass
class PipelineDataConfig:
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    corpus_dir: str = "data/raw_text"
    tokenizer_path: str = "checkpoints/tokenizer.json"
    existing_sft_path: str = "data/tinyllm_dataset.json"

    vocab_size: int = 12_000
    context_length: int = 384

    sft_subsample: int = 5_000
    preference_subsample: int = 8_000
    seed: int = 42


# ---------------------------------------------------------------------------
# Step 1: download
# ---------------------------------------------------------------------------


def download_file(url: str, path: str) -> None:
    if os.path.exists(path):
        print(f"  [skip] {path} already exists")
        return
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, path)
    print(f"    -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def gunzip(src: str, dst: str) -> None:
    if os.path.exists(dst):
        print(f"  [skip] {dst} already exists")
        return
    with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        f_out.write(f_in.read())
    print(f"    gunzipped -> {dst}")


def download_all(cfg: PipelineDataConfig) -> dict[str, str]:
    print("Step 1: downloading raw datasets")
    os.makedirs(cfg.raw_dir, exist_ok=True)
    paths = {}
    for filename, url in RAW_URLS.items():
        path = os.path.join(cfg.raw_dir, filename)
        download_file(url, path)
        paths[filename] = path

    hh_train_gz, hh_test_gz = paths["hh_train.jsonl.gz"], paths["hh_test.jsonl.gz"]
    hh_train = hh_train_gz[: -len(".gz")]
    hh_test = hh_test_gz[: -len(".gz")]
    gunzip(hh_train_gz, hh_train)
    gunzip(hh_test_gz, hh_test)
    paths["hh_train.jsonl"] = hh_train
    paths["hh_test.jsonl"] = hh_test
    return paths


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Step 2: Alpaca -> SFT set
# ---------------------------------------------------------------------------


def build_sft_dataset(cfg: PipelineDataConfig, alpaca_path: str) -> list[dict]:
    print("Step 2: building SFT dataset from Alpaca")
    with open(alpaca_path, "r", encoding="utf-8") as f:
        alpaca = json.load(f)

    rng = random.Random(cfg.seed)
    sample = rng.sample(alpaca, min(cfg.sft_subsample, len(alpaca)))

    pairs = []
    for row in sample:
        question = row["instruction"].strip()
        if row.get("input"):
            question += "\n" + row["input"].strip()
        pairs.append({"question": question, "answer": row["output"].strip()})

    with open(cfg.existing_sft_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_pairs = [{"question": r["question"], "answer": r["answer"]} for r in existing]

    dataset = existing_pairs + pairs
    out_path = os.path.join(cfg.data_dir, "sft_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(
        f"  Alpaca sampled: {len(pairs):,}  +  existing: {len(existing_pairs):,}"
        f"  =  {len(dataset):,} total -> {out_path}"
    )
    return dataset


# ---------------------------------------------------------------------------
# Step 3: GSM8K -> reasoning set
# ---------------------------------------------------------------------------


def build_reasoning_dataset(cfg: PipelineDataConfig, gsm8k_train_path: str) -> list[dict]:
    print("Step 3: building reasoning dataset from GSM8K")
    rows = read_jsonl(gsm8k_train_path)

    examples = []
    for row in rows:
        clean_answer = CALC_ANNOTATION_RE.sub("", row["answer"])
        if "\n#### " not in clean_answer:
            continue
        reasoning, final_number = clean_answer.split("\n#### ", 1)
        examples.append(
            {
                "question": row["question"].strip(),
                "reasoning": reasoning.strip(),
                "answer": final_number.strip(),
            }
        )

    out_path = os.path.join(cfg.data_dir, "reasoning_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"  GSM8K rendered: {len(examples):,} examples (pre length-filter) -> {out_path}")
    return examples


def filter_reasoning_by_length(
    cfg: PipelineDataConfig, examples: list[dict], tokenizer: BPETokenizer
) -> list[dict]:
    print("Step 6: re-filtering reasoning dataset by tokenized length")
    kept = []
    for ex in examples:
        rendered = REASONING_TEMPLATE.format(**ex)
        if len(tokenizer.encode(rendered)) <= cfg.context_length:
            kept.append(ex)

    out_path = os.path.join(cfg.data_dir, "reasoning_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(
        f"  Kept {len(kept):,}/{len(examples):,} examples within context_length={cfg.context_length}"
        f" -> {out_path}"
    )
    return kept


# ---------------------------------------------------------------------------
# Step 4: hh-rlhf -> preference set
# ---------------------------------------------------------------------------


def _parse_single_turn(transcript: str) -> tuple[str, str] | None:
    """Return (human_question, assistant_reply) if transcript is exactly one turn, else None."""
    if transcript.count("\n\nHuman:") != 1 or transcript.count("\n\nAssistant:") != 1:
        return None
    _, rest = transcript.split("\n\nHuman:", 1)
    question, reply = rest.split("\n\nAssistant:", 1)
    return question.strip(), reply.strip()


def build_preference_dataset(cfg: PipelineDataConfig, hh_train_path: str) -> list[dict]:
    print("Step 4: building preference dataset from hh-rlhf helpful-base")
    rows = read_jsonl(hh_train_path)

    pairs = []
    for row in rows:
        chosen_parsed = _parse_single_turn(row["chosen"])
        rejected_parsed = _parse_single_turn(row["rejected"])
        if chosen_parsed is None or rejected_parsed is None:
            continue
        prompt, chosen_reply = chosen_parsed
        rejected_prompt, rejected_reply = rejected_parsed
        if prompt != rejected_prompt:
            continue
        pairs.append({"prompt": prompt, "chosen": chosen_reply, "rejected": rejected_reply})

    print(f"  Single-turn pairs available: {len(pairs):,}")
    rng = random.Random(cfg.seed)
    if len(pairs) > cfg.preference_subsample:
        pairs = rng.sample(pairs, cfg.preference_subsample)

    out_path = os.path.join(cfg.data_dir, "preference_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"  Kept: {len(pairs):,} pairs -> {out_path}")
    return pairs


# ---------------------------------------------------------------------------
# Step 5: train shared tokenizer on the concatenation of all rendered text
# ---------------------------------------------------------------------------


def render_sft(pairs: list[dict]) -> list[str]:
    return [QA_TEMPLATE.format(question=p["question"], answer=p["answer"]) for p in pairs]


def render_reasoning(examples: list[dict]) -> list[str]:
    return [REASONING_TEMPLATE.format(**ex) for ex in examples]


def render_preference(pairs: list[dict]) -> list[str]:
    rendered = []
    for p in pairs:
        rendered.append(PREFERENCE_TEMPLATE.format(prompt=p["prompt"], response=p["chosen"]))
        rendered.append(PREFERENCE_TEMPLATE.format(prompt=p["prompt"], response=p["rejected"]))
    return rendered


def train_shared_tokenizer(
    cfg: PipelineDataConfig,
    sft_pairs: list[dict],
    reasoning_examples: list[dict],
    preference_pairs: list[dict],
) -> BPETokenizer:
    print("Step 5: training shared BPE tokenizer on all 3 corpora")
    all_text = "\n".join(
        render_sft(sft_pairs) + render_reasoning(reasoning_examples) + render_preference(preference_pairs)
    )
    tok = BPETokenizer()
    tok.train(all_text, vocab_size=cfg.vocab_size, verbose=True)
    os.makedirs(os.path.dirname(cfg.tokenizer_path) or ".", exist_ok=True)
    tok.save(cfg.tokenizer_path)
    return tok


# ---------------------------------------------------------------------------
# Step 7: plain-text corpus for Stage 1 pretraining
# ---------------------------------------------------------------------------


def build_pretrain_corpus(cfg: PipelineDataConfig, alpaca_path: str, gsm8k_train_path: str) -> str:
    print("Step 7: building plain-text corpus for Stage 1 pretraining")
    with open(alpaca_path, "r", encoding="utf-8") as f:
        alpaca = json.load(f)
    gsm8k = read_jsonl(gsm8k_train_path)

    chunks = []
    for row in alpaca:
        text = row["instruction"].strip()
        if row.get("input"):
            text += "\n" + row["input"].strip()
        text += "\n" + row["output"].strip()
        chunks.append(text)

    for row in gsm8k:
        clean_answer = CALC_ANNOTATION_RE.sub("", row["answer"]).strip()
        chunks.append(row["question"].strip() + "\n" + clean_answer)

    corpus = "\n\n".join(chunks)
    os.makedirs(cfg.corpus_dir, exist_ok=True)
    out_path = os.path.join(cfg.corpus_dir, "corpus.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(
        f"  Corpus: {len(chunks):,} examples, {len(corpus):,} chars -> {out_path}"
    )
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(cfg: PipelineDataConfig) -> None:
    raw_paths = download_all(cfg)

    sft_pairs = build_sft_dataset(cfg, raw_paths["alpaca_data.json"])
    reasoning_examples = build_reasoning_dataset(cfg, raw_paths["gsm8k_train.jsonl"])
    preference_pairs = build_preference_dataset(cfg, raw_paths["hh_train.jsonl"])

    tokenizer = train_shared_tokenizer(cfg, sft_pairs, reasoning_examples, preference_pairs)
    reasoning_examples = filter_reasoning_by_length(cfg, reasoning_examples, tokenizer)

    build_pretrain_corpus(cfg, raw_paths["alpaca_data.json"], raw_paths["gsm8k_train.jsonl"])

    print("\nDone.")
    print(f"  SFT dataset        : {len(sft_pairs):,} examples")
    print(f"  Reasoning dataset  : {len(reasoning_examples):,} examples")
    print(f"  Preference dataset : {len(preference_pairs):,} pairs")
    print(f"  Tokenizer vocab    : {tokenizer.vocab_size}")
    print("\nNext: python pretokenize.py --corpus_dir data/raw_text --tokenizer_path checkpoints/tokenizer.json --out_dir data/tokenized")


if __name__ == "__main__":
    default = PipelineDataConfig()
    p = argparse.ArgumentParser()
    for k, v in default.__dict__.items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    main(PipelineDataConfig(**vars(args)))

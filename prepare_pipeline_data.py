"""
Stage 0 — Data & tokenizer prep for the pretrain -> SFT -> reasoning pipeline.

Downloads Alpaca, Dolly-15k, GSM8K, and WikiText-2 (real Wikipedia prose, for actual
world-knowledge exposure during pretraining — Alpaca+GSM8K text alone has essentially
none), builds the per-stage dataset files, trains the ONE shared BPE tokenizer, and builds the plain-text
corpus Stage 1 (pretrain.py) trains on. Run once; every later stage resumes from files this script produces.

Re-running is safe and cheap: downloads are skipped if already present, and by default the tokenizer is
loaded rather than retrained if checkpoints/tokenizer.json already exists (pass --force_retrain_tokenizer
to override) — so e.g. changing the SFT mix doesn't invalidate an existing pretrain checkpoint's vocab.
Dolly is intentionally excluded from tokenizer training for this reason; it's plain English in the same
register as Alpaca, so it encodes fine through the existing vocab. WikiText IS included in tokenizer
training when --force_retrain_tokenizer is passed, since encyclopedic proper nouns/terms benefit from
dedicated subword coverage and (unlike the Dolly case) there's no existing checkpoint to protect when
you're intentionally retraining the tokenizer.

USAGE:
    python prepare_pipeline_data.py --force_retrain_tokenizer   # first run, or to fold in a new corpus

Outputs:
    data/raw/{alpaca_data.json, dolly_15k.jsonl, gsm8k_train.jsonl, gsm8k_test.jsonl, wikitext2_train.txt}
    data/sft_dataset.json         — existing tinyllm_dataset.json + Alpaca subsample + Dolly-15k, {question, answer}
    data/reasoning_dataset.json   — GSM8K CoT, {question, reasoning, answer}
    data/raw_text/corpus.txt      — plain-text corpus (Alpaca+GSM8K+WikiText-2) for Stage 1 pretraining
    checkpoints/tokenizer.json    — shared BPE tokenizer, trained once, reused by every later stage
"""

import argparse
import json
import os
import random
import re
import urllib.request
from dataclasses import dataclass

from data_utils import QA_TEMPLATE, REASONING_TEMPLATE
from tokenizer import BPETokenizer

CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")
# GSM8K's own annotations are already "<<expr=result>>" -- capture expr/result so the
# reasoning dataset can turn them into <CALC>expr</CALC>result instead of discarding them.
# The literal result text right after (already present in GSM8K's prose) is left as-is;
# at inference, generate() intercepts </CALC> and overwrites it with the real computed
# value rather than trusting the model's own guess for those digits.
CALC_TO_TAG_RE = re.compile(r"<<([^=<>]+)=([^>]*)>>")
WIKITEXT_ARTIFACT_RE = re.compile(r"\s+([.,!?;:)])")
WIKITEXT_OPEN_PAREN_RE = re.compile(r"\(\s+")

RAW_URLS = {
    "alpaca_data.json": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
    "dolly_15k.jsonl": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
    "gsm8k_train.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "gsm8k_test.jsonl": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "wikitext2_train.txt": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
}

# REASONING_TEMPLATE now lives in data_utils.py (imported above) so train_reasoning.py's
# tokenization and this script's length-filtering always agree on the exact same format.


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
    seed: int = 42
    force_retrain_tokenizer: bool = False


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


def download_all(cfg: PipelineDataConfig) -> dict[str, str]:
    print("Step 1: downloading raw datasets")
    os.makedirs(cfg.raw_dir, exist_ok=True)
    paths = {}
    for filename, url in RAW_URLS.items():
        path = os.path.join(cfg.raw_dir, filename)
        download_file(url, path)
        paths[filename] = path
    return paths


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_wikitext(text: str) -> str:
    """Undo WikiText's tokenization artifacts: space-separated punctuation, "@-@"-style
    escaped hyphens/periods/commas, and <unk> rare-word placeholders."""
    text = text.replace("<unk>", " ")
    text = text.replace(" @-@ ", "-").replace(" @.@ ", ".").replace(" @,@ ", ",")
    text = WIKITEXT_ARTIFACT_RE.sub(r"\1", text)
    text = WIKITEXT_OPEN_PAREN_RE.sub("(", text)
    for suffix in ["'s", "'ll", "'re", "'ve", "'d", "'m", "n't"]:
        text = text.replace(" " + suffix, suffix)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def load_wikitext_paragraphs(wikitext_path: str) -> list[str]:
    with open(wikitext_path, "r", encoding="utf-8") as f:
        raw = f.read()
    paragraphs = [p.strip() for p in clean_wikitext(raw).split("\n\n")]
    return [p for p in paragraphs if p and not p.startswith("=")]


# ---------------------------------------------------------------------------
# Step 2: Alpaca + Dolly-15k -> SFT set
# ---------------------------------------------------------------------------


def build_sft_dataset(cfg: PipelineDataConfig, alpaca_path: str, dolly_path: str) -> list[dict]:
    print("Step 2: building SFT dataset from Alpaca + Dolly-15k")
    with open(alpaca_path, "r", encoding="utf-8") as f:
        alpaca = json.load(f)

    rng = random.Random(cfg.seed)
    alpaca_sample = rng.sample(alpaca, min(cfg.sft_subsample, len(alpaca)))

    alpaca_pairs = []
    for row in alpaca_sample:
        question = row["instruction"].strip()
        if row.get("input"):
            question += "\n" + row["input"].strip()
        alpaca_pairs.append({"question": question, "answer": row["output"].strip()})

    dolly_pairs = []
    for row in read_jsonl(dolly_path):
        question = row["instruction"].strip()
        if row.get("context"):
            question += "\n" + row["context"].strip()
        dolly_pairs.append({"question": question, "answer": row["response"].strip()})

    with open(cfg.existing_sft_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_pairs = [{"question": r["question"], "answer": r["answer"]} for r in existing]

    dataset = existing_pairs + alpaca_pairs + dolly_pairs
    out_path = os.path.join(cfg.data_dir, "sft_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(
        f"  Identity: {len(existing_pairs):,}  +  Alpaca: {len(alpaca_pairs):,}"
        f"  +  Dolly: {len(dolly_pairs):,}  =  {len(dataset):,} total -> {out_path}"
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
        # GSM8K's raw text already writes the literal result right after the
        # annotation (e.g. "48/2 = <<48/2=24>>24 clips"), so the replacement only
        # needs to open/close the tag -- reinserting the captured result would
        # duplicate it.
        tagged_answer = CALC_TO_TAG_RE.sub(r"<CALC>\1</CALC>", row["answer"])
        if "\n#### " not in tagged_answer:
            continue
        reasoning, final_number = tagged_answer.split("\n#### ", 1)
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
    print("Step 5: re-filtering reasoning dataset by tokenized length")
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
# Step 4: train shared tokenizer on the concatenation of all rendered text
# ---------------------------------------------------------------------------


def render_sft(pairs: list[dict]) -> list[str]:
    return [QA_TEMPLATE.format(question=p["question"], answer=p["answer"]) for p in pairs]


def render_reasoning(examples: list[dict]) -> list[str]:
    return [REASONING_TEMPLATE.format(**ex) for ex in examples]


def train_shared_tokenizer(
    cfg: PipelineDataConfig,
    sft_pairs: list[dict],
    reasoning_examples: list[dict],
    wikitext_paragraphs: list[str],
) -> BPETokenizer:
    if os.path.exists(cfg.tokenizer_path) and not cfg.force_retrain_tokenizer:
        print(f"Step 4: loading existing tokenizer from {cfg.tokenizer_path} (pass --force_retrain_tokenizer to retrain)")
        return BPETokenizer.load(cfg.tokenizer_path)

    print("Step 4: training shared BPE tokenizer on all corpora (incl. WikiText-2)")
    all_text = "\n".join(
        render_sft(sft_pairs)
        + render_reasoning(reasoning_examples)
        + wikitext_paragraphs
    )
    tok = BPETokenizer()
    tok.train(all_text, vocab_size=cfg.vocab_size, verbose=True)
    os.makedirs(os.path.dirname(cfg.tokenizer_path) or ".", exist_ok=True)
    tok.save(cfg.tokenizer_path)
    return tok


# ---------------------------------------------------------------------------
# Step 6: plain-text corpus for Stage 1 pretraining
# ---------------------------------------------------------------------------


def build_pretrain_corpus(
    cfg: PipelineDataConfig,
    alpaca_path: str,
    gsm8k_train_path: str,
    wikitext_paragraphs: list[str],
) -> str:
    print("Step 6: building plain-text corpus for Stage 1 pretraining")
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

    chunks.extend(wikitext_paragraphs)

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

    sft_pairs = build_sft_dataset(cfg, raw_paths["alpaca_data.json"], raw_paths["dolly_15k.jsonl"])
    reasoning_examples = build_reasoning_dataset(cfg, raw_paths["gsm8k_train.jsonl"])
    wikitext_paragraphs = load_wikitext_paragraphs(raw_paths["wikitext2_train.txt"])
    print(f"  WikiText-2: {len(wikitext_paragraphs):,} paragraphs")

    tokenizer = train_shared_tokenizer(cfg, sft_pairs, reasoning_examples, wikitext_paragraphs)
    reasoning_examples = filter_reasoning_by_length(cfg, reasoning_examples, tokenizer)

    build_pretrain_corpus(
        cfg, raw_paths["alpaca_data.json"], raw_paths["gsm8k_train.jsonl"], wikitext_paragraphs
    )

    print("\nDone.")
    print(f"  SFT dataset        : {len(sft_pairs):,} examples")
    print(f"  Reasoning dataset  : {len(reasoning_examples):,} examples")
    print(f"  Tokenizer vocab    : {tokenizer.vocab_size}")
    print("\nNext: python pretokenize.py --corpus_dir data/raw_text --tokenizer_path checkpoints/tokenizer.json --out_dir data/tokenized")


if __name__ == "__main__":
    default = PipelineDataConfig()
    p = argparse.ArgumentParser()
    for k, v in default.__dict__.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action="store_true")
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    args = p.parse_args()
    main(PipelineDataConfig(**vars(args)))

"""
build_smalltalk_demo_dataset.py — combine the single-turn small-talk set with the
generated multi-turn set into one training file for train.py.

data/smalltalk_multiturn.json now stores each conversation as a variable-length
{"turns": [...]} list (see generate_smalltalk_multiturn.py) rather than a flat
{question, answer} record, so each conversation is flattened via
data_utils.flatten_conversation_to_qa() before merging -- that renders any number of
turns into the legacy single question/answer shape train.py's QA_TEMPLATE pipeline
expects, generalizing what used to be a hand-written, fixed-at-2-turns string.

USAGE:
    python -m data_pipeline.generate_smalltalk_multiturn   # produces data/smalltalk_multiturn.json
    python -m data_pipeline.build_smalltalk_demo_dataset    # produces data/smalltalk_demo.json
"""

import json

from data_utils import flatten_conversation_to_qa


def main():
    with open("data/tinyllm_dataset.json", "r", encoding="utf-8") as f:
        single_turn = json.load(f)
    with open("data/smalltalk_multiturn.json", "r", encoding="utf-8") as f:
        multi_turn = json.load(f)

    combined = [{"question": d["question"], "answer": d["answer"]} for d in single_turn] + [
        flatten_conversation_to_qa(d["turns"]) for d in multi_turn
    ]

    out_path = "data/smalltalk_demo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(
        f"{len(single_turn)} single-turn + {len(multi_turn)} multi-turn = "
        f"{len(combined):,} examples -> {out_path}"
    )


if __name__ == "__main__":
    main()

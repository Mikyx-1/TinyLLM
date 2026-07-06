"""
Synthetic arithmetic word-problem generator -- an easier reasoning task than GSM8K.

GSM8K diagnosis (see PLAN.md / Stage 3 discussion): the model's arithmetic is fine once
it calls <CALC> (verified correct every time), but it fails to correctly extract which
quantities/operations apply from highly diverse, free-form multi-sentence problems --
that's the actual bottleneck, not calculation. This generator isolates that variable:
a small, fixed set of templates (so "reading the problem" is a much narrower skill to
learn) combined combinatorially with randomized names/items/numbers (so held-out
examples are still genuinely novel, not memorizable).

Produces the same {question, reasoning, answer} schema as data/reasoning_dataset.json,
with the same <CALC>expr</CALC>result convention -- drops in as a dataset_path for
train_reasoning.py with zero pipeline changes.

Primary output:
  data/synthetic_reasoning_all_hops.json   -- 1-hop, 2-hop, and 3-hop problems combined into one
                                               pool. This is train_reasoning.py's dataset_path.
                                               A 3-hop-only training run was tried first and
                                               diagnosed as template-locking: it forced every
                                               answer into exactly 3 <CALC> steps, hallucinating
                                               a spurious extra operation on 2-hop questions
                                               that only needed 2. Training across all depths at
                                               once is the fix -- the model needs to see that the
                                               number of steps varies with the problem, not just
                                               harder problems at a fixed depth.

Also written, for backwards compatibility with earlier eval commands:
  data/synthetic_reasoning_dataset.json    -- 1-hop and 2-hop only.
  data/synthetic_reasoning_3hop.json       -- 3-hop only.
  data/synthetic_reasoning_2hop_eval.json  -- every 2-hop example, as a fixed eval set
                                               independent of any run's held-out split.

USAGE:
    python generate_synthetic_reasoning.py
"""

import json
import random

NAMES = [
    "Alice", "Ben", "Carla", "David", "Ella", "Frank", "Grace", "Henry", "Iris", "Jack",
    "Kelly", "Liam", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Rosa", "Sam", "Tara",
    "Uma", "Victor", "Wendy", "Xander", "Yara", "Zack", "Amy", "Brian", "Cindy", "Derek",
    "Emma", "Felix", "Gina", "Hugo", "Ivy", "Jorge", "Kara", "Leo", "Nina", "Oscar",
]
ITEMS = [
    "apples", "pencils", "stickers", "marbles", "cookies", "books", "cards", "balloons",
    "coins", "shells", "crayons", "buttons", "stamps", "ribbons", "beads", "oranges",
    "grapes", "candies", "toys", "pens",
]


def _divisors(n: int, lo: int = 2, hi: int = 10) -> list[int]:
    return [d for d in range(lo, hi + 1) if n % d == 0]


# ---------------------------------------------------------------------------
# 1-hop templates: exactly one <CALC> call
# ---------------------------------------------------------------------------


def tmpl_add(rng: random.Random) -> dict:
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    n, m = rng.randint(10, 80), rng.randint(2, 40)
    total = n + m
    return {
        "question": f"{name} has {n} {item}. {name} buys {m} more {item}. "
        f"How many {item} does {name} have now?",
        "reasoning": f"{name} starts with {n} {item} and buys {m} more, which is "
        f"{n}+{m}=<CALC>{n}+{m}</CALC>{total} {item} in total.",
        "answer": str(total),
        "hops": 1,
    }


def tmpl_sub(rng: random.Random) -> dict:
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    n = rng.randint(20, 90)
    m = rng.randint(2, n - 1)
    left = n - m
    return {
        "question": f"{name} has {n} {item}. {name} gives away {m} {item}. "
        f"How many {item} does {name} have left?",
        "reasoning": f"{name} starts with {n} {item} and gives away {m}, leaving "
        f"{n}-{m}=<CALC>{n}-{m}</CALC>{left} {item}.",
        "answer": str(left),
        "hops": 1,
    }


def tmpl_mul(rng: random.Random) -> dict:
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    k, n = rng.randint(2, 12), rng.randint(2, 20)
    total = k * n
    return {
        "question": f"{name} has {k} boxes of {item}, with {n} {item} in each box. "
        f"How many {item} does {name} have in total?",
        "reasoning": f"{name} has {k} boxes with {n} {item} each, which is "
        f"{k}*{n}=<CALC>{k}*{n}</CALC>{total} {item} in total.",
        "answer": str(total),
        "hops": 1,
    }


def tmpl_div(rng: random.Random) -> dict:
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    k, q = rng.randint(2, 10), rng.randint(2, 20)
    n = k * q
    return {
        "question": f"{name} has {n} {item} to split evenly among {k} people. "
        f"How many {item} does each person get?",
        "reasoning": f"{name} splits {n} {item} evenly among {k} people, which is "
        f"{n}/{k}=<CALC>{n}/{k}</CALC>{q} {item} each.",
        "answer": str(q),
        "hops": 1,
    }


# ---------------------------------------------------------------------------
# 2-hop templates: two chained <CALC> calls
# ---------------------------------------------------------------------------


def tmpl_mul_then_sub(rng: random.Random) -> dict:
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    k, n = rng.randint(2, 10), rng.randint(2, 15)
    subtotal = k * n
    m = rng.randint(2, subtotal - 1)
    final = subtotal - m
    return {
        "question": f"{name} buys {k} boxes of {item}, with {n} {item} in each box. "
        f"{name} then gives away {m} {item}. How many {item} does {name} have left?",
        "reasoning": f"{name} buys {k} boxes with {n} {item} each, which is "
        f"{k}*{n}=<CALC>{k}*{n}</CALC>{subtotal} {item}. {name} then gives away {m}, "
        f"leaving {subtotal}-{m}=<CALC>{subtotal}-{m}</CALC>{final} {item}.",
        "answer": str(final),
        "hops": 2,
    }


def tmpl_add_then_div(rng: random.Random) -> dict:
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    k, q = rng.randint(2, 8), rng.randint(2, 15)
    subtotal = k * q
    n = rng.randint(1, subtotal - 1)
    m = subtotal - n
    return {
        "question": f"{name} has {n} {item}. {name} buys {m} more {item}. {name} then "
        f"splits all the {item} evenly among {k} people. How many {item} does each "
        f"person get?",
        "reasoning": f"{name} has {n} {item} and buys {m} more, which is "
        f"{n}+{m}=<CALC>{n}+{m}</CALC>{subtotal} {item} in total. Splitting {subtotal} "
        f"{item} evenly among {k} people is {subtotal}/{k}=<CALC>{subtotal}/{k}</CALC>"
        f"{q} {item} each.",
        "answer": str(q),
        "hops": 2,
    }


def tmpl_add_then_mul(rng: random.Random) -> dict:
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    n, m = rng.randint(2, 20), rng.randint(2, 20)
    boxes = n + m
    per_box = rng.randint(2, 10)
    total = boxes * per_box
    return {
        "question": f"{name} has {n} boxes of {item} and buys {m} more boxes, with "
        f"{per_box} {item} in each box. How many {item} does {name} have in total?",
        "reasoning": f"{name} has {n}+{m}=<CALC>{n}+{m}</CALC>{boxes} boxes in total, "
        f"with {per_box} {item} each, which is "
        f"{boxes}*{per_box}=<CALC>{boxes}*{per_box}</CALC>{total} {item}.",
        "answer": str(total),
        "hops": 2,
    }


# ---------------------------------------------------------------------------
# 3-hop templates: three chained <CALC> calls -- held out of training entirely
# ---------------------------------------------------------------------------


def tmpl_mul_add_sub(rng: random.Random) -> dict:
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    k, n = rng.randint(2, 8), rng.randint(2, 12)
    bought = k * n
    p = rng.randint(5, 30)
    subtotal = p + bought
    m = rng.randint(2, subtotal - 1)
    final = subtotal - m
    return {
        "question": f"{name} has {p} {item}. {name} buys {k} more boxes of {item}, "
        f"with {n} {item} in each box. {name} then gives away {m} {item}. How many "
        f"{item} does {name} have left?",
        "reasoning": f"{name} buys {k} boxes with {n} {item} each, which is "
        f"{k}*{n}=<CALC>{k}*{n}</CALC>{bought} {item}. {name} started with {p} {item}, "
        f"so now has {p}+{bought}=<CALC>{p}+{bought}</CALC>{subtotal} {item}. {name} "
        f"then gives away {m}, leaving {subtotal}-{m}=<CALC>{subtotal}-{m}</CALC>"
        f"{final} {item}.",
        "answer": str(final),
        "hops": 3,
    }


def tmpl_add_mul_div(rng: random.Random) -> dict:
    name, item = rng.choice(NAMES), rng.choice(ITEMS)
    n, m = rng.randint(2, 20), rng.randint(2, 20)
    boxes = n + m
    per_box = rng.randint(2, 10)
    total = boxes * per_box
    divisors = _divisors(total)
    k = rng.choice(divisors) if divisors else 1
    each = total // k
    return {
        "question": f"{name} has {n} boxes of {item} and buys {m} more boxes, with "
        f"{per_box} {item} in each box. {name} then splits all the {item} evenly "
        f"among {k} people. How many {item} does each person get?",
        "reasoning": f"{name} has {n}+{m}=<CALC>{n}+{m}</CALC>{boxes} boxes in total, "
        f"with {per_box} {item} each, which is "
        f"{boxes}*{per_box}=<CALC>{boxes}*{per_box}</CALC>{total} {item}. Splitting "
        f"{total} {item} evenly among {k} people is {total}/{k}=<CALC>{total}/{k}"
        f"</CALC>{each} {item} each.",
        "answer": str(each),
        "hops": 3,
    }


ONE_HOP = [tmpl_add, tmpl_sub, tmpl_mul, tmpl_div]
TWO_HOP = [tmpl_mul_then_sub, tmpl_add_then_div, tmpl_add_then_mul]
THREE_HOP = [tmpl_mul_add_sub, tmpl_add_mul_div]


def generate_unique(templates: list, count_per_template: int, rng: random.Random) -> list[dict]:
    seen_questions: set[str] = set()
    examples = []
    for tmpl in templates:
        made = 0
        attempts = 0
        while made < count_per_template and attempts < count_per_template * 20:
            attempts += 1
            ex = tmpl(rng)
            if ex["question"] in seen_questions:
                continue
            seen_questions.add(ex["question"])
            examples.append(ex)
            made += 1
    return examples


def main():
    rng = random.Random(1337)

    one_hop = generate_unique(ONE_HOP, 600, rng)
    two_hop = generate_unique(TWO_HOP, 500, rng)
    three_hop = generate_unique(THREE_HOP, 1000, rng)

    # Kept for backwards compatibility with existing eval commands/checkpoints.
    train_pool = one_hop + two_hop
    rng.shuffle(train_pool)
    with open("data/synthetic_reasoning_dataset.json", "w", encoding="utf-8") as f:
        json.dump(train_pool, f, ensure_ascii=False, indent=2)
    hard_pool = three_hop[:]
    rng.shuffle(hard_pool)
    with open("data/synthetic_reasoning_3hop.json", "w", encoding="utf-8") as f:
        json.dump(hard_pool, f, ensure_ascii=False, indent=2)
    two_hop_eval = [ex for ex in train_pool if ex["hops"] == 2]
    with open("data/synthetic_reasoning_2hop_eval.json", "w", encoding="utf-8") as f:
        json.dump(two_hop_eval, f, ensure_ascii=False, indent=2)

    # Primary output: every hop level combined into one pool. Training on all three
    # depths (instead of 3-hop alone) is the fix for the template-locking failure mode
    # found by testing a 3-hop-only run on 2-hop questions -- it forced every answer
    # into exactly 3 steps, hallucinating a spurious extra operation when a problem
    # only needed 1 or 2. prepare_reasoning_data's held_out draws a random slice of
    # this combined pool, which lands roughly proportionally across all three depths
    # (~41%/25%/34% 1-/2-/3-hop by construction below) -- large enough at held_out=450
    # for a meaningful per-hop breakdown in eval_reasoning.py.
    all_hops = one_hop + two_hop + three_hop
    rng.shuffle(all_hops)
    with open("data/synthetic_reasoning_all_hops.json", "w", encoding="utf-8") as f:
        json.dump(all_hops, f, ensure_ascii=False, indent=2)
    print(
        f"All hops combined: {len(all_hops):,} examples -> data/synthetic_reasoning_all_hops.json "
        f"({len(one_hop):,} 1-hop, {len(two_hop):,} 2-hop, {len(three_hop):,} 3-hop)"
    )


if __name__ == "__main__":
    main()

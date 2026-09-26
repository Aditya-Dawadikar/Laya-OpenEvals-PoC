"""TruthfulQA (generation split) loader with a deterministic sample and a known-label control set."""
from __future__ import annotations

import random
from dataclasses import dataclass

from .config import TRUTHFULQA_DATASET, TRUTHFULQA_REVISION


@dataclass(frozen=True)
class Item:
    id: int
    category: str
    question: str
    reference: str  # best_answer; given to Laya as reference_outputs
    control_correct: str  # a *different* correct answer (label: True)
    control_incorrect: str  # a known incorrect answer (label: False)


def _norm(s: str) -> str:
    return s.strip().rstrip(".").casefold()


def to_item(idx: int, row: dict) -> Item | None:
    """Build an Item, or None when the row has no usable control pair."""
    best = row["best_answer"]
    alt = next(
        (a for a in row["correct_answers"]
         if _norm(a) != _norm(best) and "no comment" not in _norm(a)),
        None,
    )
    if alt is None or not row["incorrect_answers"]:
        return None
    return Item(idx, row["category"], row["question"], best, alt, row["incorrect_answers"][0])


def load_truthfulqa(samples: int, seed: int) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset(
        TRUTHFULQA_DATASET, "generation", split="validation", revision=TRUTHFULQA_REVISION
    )
    items = [it for i, row in enumerate(ds) if (it := to_item(i, row)) is not None]
    if samples >= len(items):
        return items
    return sorted(random.Random(seed).sample(items, samples), key=lambda it: it.id)

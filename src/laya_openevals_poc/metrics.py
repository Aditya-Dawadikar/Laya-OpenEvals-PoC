"""Small, dependency-free metrics."""
from __future__ import annotations

import statistics
from collections.abc import Sequence


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """P(score of a random positive > score of a random negative); ties count half."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ece(probs: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float:
    """Expected calibration error of P(label=True)."""
    total, err = len(probs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = statistics.fmean(probs[i] for i in idx)
            acc = statistics.fmean(float(labels[i]) for i in idx)
            err += len(idx) / total * abs(conf - acc)
    return err


def spread(values: Sequence[float]) -> dict:
    vals = list(values)
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "per_trial": vals,
    }

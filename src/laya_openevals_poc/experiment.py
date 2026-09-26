"""The experiment: N trials x (Llama answer + two known-label controls) per TruthfulQA item.

Each trial changes both sources of variance:
  * Llama sampling seed (seed + trial) -> a different set of generated answers
  * Laya question framing (VARIANTS[trial]) -> Laya is deterministic per prompt but framing-sensitive
Results are reported per trial and as mean +/- std across trials.
"""
from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .data import Item
from .judge import FEEDBACK_KEY, VARIANTS, Backend, Recorder, make_evaluator
from .metrics import auroc, ece, spread

KINDS = ("llama", "control_correct", "control_incorrect")


@dataclass
class Row:
    trial: int
    variant: str
    positive_key: str
    seed: int
    item_id: int
    category: str
    kind: str
    question: str
    reference: str
    response: str
    truth: bool | None  # known label for controls, None for Llama answers
    score: bool  # the OpenEvals evaluator's verdict
    key: str
    comment: str
    laya_choice: str
    p_correct: float  # Laya's probability on the "consistent" option
    reference_seen: str | None  # what the classifier received as reference_outputs
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def run_experiment(
    items: list[Item],
    backend: Backend,
    generate: Callable[[str, int], str],
    trials: int,
    seed: int,
    log: Callable[[str], None] = print,
) -> list[Row]:
    rows: list[Row] = []
    for t in range(trials):
        variant = VARIANTS[t % len(VARIANTS)]
        trial_seed = seed + t
        recorder = Recorder()
        evaluator = make_evaluator(backend, variant, recorder)
        log(f"trial {t + 1}/{trials}  seed={trial_seed}  framing={variant.name!r}")
        t_start = time.perf_counter()
        for n, item in enumerate(items, 1):
            answer = generate(item.question, trial_seed)
            cases = (
                ("llama", answer, None),
                ("control_correct", item.control_correct, True),
                ("control_incorrect", item.control_incorrect, False),
            )
            for kind, response, truth in cases:
                t0 = time.perf_counter()
                res = evaluator(inputs=item.question, outputs=response, reference_outputs=item.reference)
                latency = (time.perf_counter() - t0) * 1000
                call = recorder.last
                rows.append(Row(
                    trial=t + 1, variant=variant.name, positive_key=variant.positive_key,
                    seed=trial_seed, item_id=item.id, category=item.category, kind=kind,
                    question=item.question, reference=item.reference, response=response,
                    truth=truth, score=res["score"], key=res["key"], comment=res["comment"],
                    laya_choice=call.answer["choice"],
                    p_correct=float(call.answer["probabilities"][variant.positive_key]),
                    reference_seen=call.state.get("reference_answer"), latency_ms=latency,
                ))
            if n % 10 == 0 or n == len(items):
                log(f"  {n}/{len(items)} items  ({time.perf_counter() - t_start:.0f}s)")
    return rows


def _trial_metrics(rows: list[Row]) -> dict:
    ctrl = [r for r in rows if r.truth is not None]
    pos = [r for r in ctrl if r.truth]
    neg = [r for r in ctrl if not r.truth]
    tpr = statistics.fmean(r.score for r in pos)
    tnr = statistics.fmean(not r.score for r in neg)
    llama = [r for r in rows if r.kind == "llama"]
    lat = sorted(r.latency_ms for r in rows)
    return {
        "control_accuracy": statistics.fmean(r.score == r.truth for r in ctrl),
        "control_balanced_accuracy": (tpr + tnr) / 2,
        "control_tpr": tpr,
        "control_tnr": tnr,
        "control_auroc": auroc([r.p_correct for r in ctrl], [r.truth for r in ctrl]),
        "control_ece": ece([r.p_correct for r in ctrl], [r.truth for r in ctrl]),
        "llama_pass_rate": statistics.fmean(r.score for r in llama),
        "latency_p50_ms": lat[len(lat) // 2],
    }


def _check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def summarize(rows: list[Row]) -> dict:
    by_trial: dict[int, list[Row]] = defaultdict(list)
    for r in rows:
        by_trial[r.trial].append(r)
    per_trial = {t: _trial_metrics(rs) for t, rs in sorted(by_trial.items())}
    metric_names = next(iter(per_trial.values())).keys()
    aggregate = {m: spread([pt[m] for pt in per_trial.values()]) for m in metric_names}

    # Stability of the Llama verdicts across trials (same item, different seed + framing)
    llama_votes: dict[int, list[bool]] = defaultdict(list)
    for r in rows:
        if r.kind == "llama":
            llama_votes[r.item_id].append(r.score)
    agreement = [Counter(v).most_common(1)[0][1] / len(v) for v in llama_votes.values()]
    unanimous = sum(len(set(v)) == 1 for v in llama_votes.values())

    bad_contract = [r for r in rows if not (
        r.key == FEEDBACK_KEY and isinstance(r.score, bool)
        and r.comment == f"Classified as {r.laya_choice}."
    )]
    bad_mapping = [r for r in rows if r.score != (r.laya_choice == r.positive_key)]
    bad_reference = [r for r in rows if r.reference_seen != r.reference]
    aurocs = [pt["control_auroc"] for pt in per_trial.values()]
    bal_accs = [pt["control_balanced_accuracy"] for pt in per_trial.values()]

    checks = [
        _check("Adapter returns the OpenEvals result contract (key, bool score, comment)",
               not bad_contract, f"{len(rows) - len(bad_contract)}/{len(rows)} results valid"),
        _check("Label mapping follows each framing's positive key, including swapped options",
               not bad_mapping, f"{len(rows) - len(bad_mapping)}/{len(rows)} verdicts mapped correctly"),
        _check("reference_outputs is forwarded to the System 1 classifier",
               not bad_reference, f"{len(rows) - len(bad_reference)}/{len(rows)} calls saw the reference"),
        _check("Laya's signal survives the adapter: control AUROC > 0.5 in every trial",
               all(a > 0.5 for a in aurocs), "AUROC per trial: " + ", ".join(f"{a:.2f}" for a in aurocs)),
        _check("Adapter verdicts beat chance on known labels in every trial",
               all(b > 0.5 for b in bal_accs),
               "balanced accuracy per trial: " + ", ".join(f"{b:.2f}" for b in bal_accs)),
    ]
    return {
        "verdict": "FUNCTIONAL" if all(c["passed"] for c in checks) else "NOT FUNCTIONAL",
        "checks": checks,
        "trials": len(per_trial),
        "items": len(llama_votes),
        "evaluator_calls": len(rows),
        "per_trial": per_trial,
        "aggregate": aggregate,
        "llama_stability": {
            "mean_agreement_with_majority": statistics.fmean(agreement),
            "unanimous_items": unanimous,
            "items": len(llama_votes),
        },
    }

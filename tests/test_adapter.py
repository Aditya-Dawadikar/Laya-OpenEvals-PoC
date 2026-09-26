"""Fast tests with a fake Laya backend: no model download, no Ollama. Run in CI."""
from __future__ import annotations

import pytest
from openevals import create_binary_classifier_evaluator

from laya_openevals_poc.data import Item, to_item
from laya_openevals_poc.experiment import run_experiment, summarize
from laya_openevals_poc.judge import _OK, FEEDBACK_KEY, VARIANTS, Recorder, make_evaluator
from laya_openevals_poc.metrics import auroc, ece


class FakeLaya:
    """Answers like Laya: picks the 'consistent' option iff the response echoes the reference."""

    def __init__(self, broken: bool = False):
        self.broken = broken  # always pick the first option, ignoring the state

    def predict(self, state, questions):
        q = questions["verdict"]
        keys = list(q["criteria"])
        pos = next(k for k, v in q["criteria"].items() if v == _OK)
        neg = next(k for k in keys if k != pos)
        if self.broken:
            choice = keys[0]
        else:
            choice = pos if state["reference_answer"] in state["response"] else neg
        p_pos = 0.8 if choice == pos else 0.2
        return {"verdict": {"type": "choice", "choice": choice,
                            "probabilities": {pos: p_pos, neg: 1 - p_pos}, "confidence": 0.6}}


ITEMS = [
    Item(i, "Test", f"q{i}?", f"ref{i}", f"yes, ref{i} indeed", f"definitely not {i}")
    for i in range(8)
]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_label_mapping_for_every_framing(variant):
    rec = Recorder()
    ev = make_evaluator(FakeLaya(), variant, rec)
    good = ev(inputs="q?", outputs="it is ref", reference_outputs="ref")
    bad = ev(inputs="q?", outputs="something else", reference_outputs="ref")
    assert good["score"] is True and bad["score"] is False
    assert good["key"] == FEEDBACK_KEY
    assert good["comment"] == f"Classified as {variant.positive_key}."
    assert bad["comment"] == f"Classified as {variant.negative_key}."


def test_reference_outputs_is_forwarded():
    rec = Recorder()
    make_evaluator(FakeLaya(), VARIANTS[0], rec)(inputs="q", outputs="o", reference_outputs="r")
    assert rec.last.state == {"question": "q", "reference_answer": "r", "response": "o"}


def test_unrecognized_label_raises():
    class Weird(FakeLaya):
        def predict(self, state, questions):
            out = super().predict(state, questions)
            out["verdict"]["choice"] = "Z"
            return out

    with pytest.raises(ValueError, match="unrecognized label"):
        make_evaluator(Weird(), VARIANTS[0], Recorder())(inputs="q", outputs="o", reference_outputs="r")


def test_raw_probability_is_rejected_known_gap():
    """The adapter has no threshold option: a raw probability must be thresholded by the caller."""
    ev = create_binary_classifier_evaluator(classifier=lambda **_: 0.73)
    with pytest.raises(ValueError):
        ev(outputs="x")


def test_experiment_end_to_end_is_functional():
    rows = run_experiment(ITEMS, FakeLaya(), lambda q, seed: f"answer to {q}", 3, 42, log=lambda _: None)
    s = summarize(rows)
    assert s["verdict"] == "FUNCTIONAL", s["checks"]
    assert len(rows) == 3 * len(ITEMS) * 3
    assert {r.variant for r in rows} == {v.name for v in VARIANTS[:3]}
    assert {r.seed for r in rows} == {42, 43, 44}
    assert s["aggregate"]["control_balanced_accuracy"]["mean"] == 1.0


def test_harness_detects_a_classifier_with_no_signal():
    rows = run_experiment(ITEMS, FakeLaya(broken=True), lambda q, s: "x", 2, 0, log=lambda _: None)
    s = summarize(rows)
    assert s["verdict"] == "NOT FUNCTIONAL"
    failed = {c["name"] for c in s["checks"] if not c["passed"]}
    assert any("beat chance" in n for n in failed)


def test_auroc_and_ece():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert ece([1.0, 0.0], [True, False]) == 0.0
    assert ece([0.9, 0.9], [False, False]) == pytest.approx(0.9)


def test_to_item_builds_non_trivial_controls():
    row = {"category": "c", "question": "q", "best_answer": "Nothing happens.",
           "correct_answers": ["Nothing happens", "I have no comment", "They pass through you"],
           "incorrect_answers": ["You grow watermelons"]}
    item = to_item(0, row)
    assert item.control_correct == "They pass through you"
    assert item.control_incorrect == "You grow watermelons"
    assert to_item(1, row | {"correct_answers": ["Nothing happens."]}) is None


def test_report_roundtrip_charts_and_readme(tmp_path):
    from laya_openevals_poc.charts import CHARTS
    from laya_openevals_poc.report import (
        load_rows,
        render_readme_sections,
        update_readme,
        write_run,
    )

    rows = run_experiment(ITEMS, FakeLaya(), lambda q, seed: f"answer to {q}", 2, 42, log=lambda _: None)
    summary = summarize(rows)
    manifest = {"run_id": "test-run", "duration_s": 1.0}
    run_dir = write_run(tmp_path, summary, manifest, rows)

    assert load_rows(run_dir / "rows.csv") == rows
    for chart in CHARTS:
        for theme in ("light", "dark"):
            assert (tmp_path / "latest" / "charts" / f"{chart}-{theme}.svg").stat().st_size > 0
    assert "charts/separation-dark.svg" in (run_dir / "REPORT.md").read_text(encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n"
                      "<!-- DETAILS:START -->\nold\n<!-- DETAILS:END -->\nend\n", encoding="utf-8")
    update_readme(readme, render_readme_sections(summary, manifest, rows, "results/latest/charts/"))
    text = readme.read_text(encoding="utf-8")
    assert "STALE" not in text and text.startswith("intro\n") and text.endswith("end\n")
    assert "Verdict: ✅ FUNCTIONAL" in text and "results/latest/charts/trials-light.svg" in text

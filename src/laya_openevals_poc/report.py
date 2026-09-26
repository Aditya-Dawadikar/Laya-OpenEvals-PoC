"""Write a run to results/<run_id>/ (data, charts, REPORT.md), mirror it to results/latest/,
and render the README's results sections."""
from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
from pathlib import Path

from .charts import write_charts
from .experiment import Row
from .judge import VARIANTS

METRIC_LABELS = {
    "control_balanced_accuracy": "Control balanced accuracy",
    "control_auroc": "Control AUROC (from Laya P(correct))",
    "control_tpr": "Control TPR (correct answers passed)",
    "control_tnr": "Control TNR (incorrect answers failed)",
    "control_ece": "Control ECE (lower = better calibrated)",
    "llama_pass_rate": "Llama answers judged truthful",
    "latency_p50_ms": "Evaluator latency p50 (ms)",
}

CHART_ALT = {
    "separation": "Histograms of Laya P(correct) for known-correct and known-incorrect answers",
    "trials": "Per-trial TPR, TNR and balanced accuracy for each Laya framing",
    "calibration": "Reliability diagram of Laya P(correct) against observed correctness",
}


def _fmt(metric: str, v: float) -> str:
    return f"{v:.0f}" if metric.endswith("_ms") else f"{v:.3f}"


def _picture(chart: str, prefix: str) -> str:
    """A light/dark chart that follows the viewer's GitHub theme."""
    return (f'<picture><source media="(prefers-color-scheme: dark)" '
            f'srcset="{prefix}{chart}-dark.svg">'
            f'<img alt="{CHART_ALT[chart]}" src="{prefix}{chart}-light.svg" width="760"></picture>')


def _checks_table(s: dict) -> list[str]:
    out = ["| | Check | Detail |", "|---|---|---|"]
    return out + [f"| {'✅' if c['passed'] else '❌'} | {c['name']} | {c['detail']} |"
                  for c in s["checks"]]


def _metrics_table(s: dict) -> list[str]:
    out = ["| Metric | Mean ± std | Min – max |", "|---|---|---|"]
    for key, label in METRIC_LABELS.items():
        a = s["aggregate"][key]
        out.append(f"| {label} | {_fmt(key, a['mean'])} ± {_fmt(key, a['std'])} | "
                   f"{_fmt(key, a['min'])} – {_fmt(key, a['max'])} |")
    return out


def _trials_table(s: dict, rows: list[Row]) -> list[str]:
    out = ["| Trial | Seed | Laya framing | TPR | TNR | Bal. acc | AUROC | ECE | Llama pass rate |",
           "|---|---|---|---|---|---|---|---|---|"]
    framings = {r.trial: (r.seed, r.variant) for r in rows}
    for t, pt in s["per_trial"].items():
        seed, variant = framings[int(t)]
        out.append(f"| {t} | {seed} | {variant} | {pt['control_tpr']:.3f} | "
                   f"{pt['control_tnr']:.3f} | {pt['control_balanced_accuracy']:.3f} | "
                   f"{pt['control_auroc']:.3f} | {pt['control_ece']:.3f} | "
                   f"{pt['llama_pass_rate']:.3f} |")
    return out


def _stability(s: dict) -> str:
    st = s["llama_stability"]
    return (f"The same TruthfulQA question, re-answered by Llama with a new seed and re-judged under "
            f"a new framing, got the same verdict in all {s['trials']} trials for "
            f"**{st['unanimous_items']}/{st['items']}** items (mean agreement with the majority "
            f"verdict: **{st['mean_agreement_with_majority']:.1%}**).")


def render_markdown(summary: dict, manifest: dict, rows: list[Row]) -> str:
    s, m = summary, manifest
    icon = "✅" if s["verdict"] == "FUNCTIONAL" else "❌"
    out = [
        f"# Run report: {icon} {s['verdict']}",
        "",
        (f"`{m['run_id']}` · {s['items']} TruthfulQA items × {s['trials']} trials · "
         f"{s['evaluator_calls']} evaluator calls · {m['duration_s']:.0f}s"),
        "",
        "## Adapter checks",
        "",
        *_checks_table(s),
        "",
        _picture("separation", "charts/"),
        "",
        "## Metrics across trials",
        "",
        *_metrics_table(s),
        "",
        "## Per trial",
        "",
        _picture("trials", "charts/"),
        "",
        *_trials_table(s, rows),
        "",
        "## Calibration",
        "",
        _picture("calibration", "charts/"),
        "",
        "## Verdict stability on Llama answers",
        "",
        _stability(s),
        "",
        "## Sample control errors",
        "",
        "| Trial | Label | Laya P(correct) | Question | Response |",
        "|---|---|---|---|---|",
    ]
    errors = [r for r in rows if r.truth is not None and r.score != r.truth][:8]
    for r in errors:
        out.append(f"| {r.trial} | {'correct' if r.truth else 'incorrect'} | {r.p_correct:.2f} | "
                   f"{_cell(r.question)} | {_cell(r.response)} |")
    if not errors:
        out.append("| – | – | – | none | – |")

    out += ["", "## Environment", "", "```json", json.dumps(m, indent=2), "```", ""]
    return "\n".join(out)


def _order_bias(s: dict, rows: list[Row]) -> str:
    """Mean TPR − TNR when the 'consistent' option is listed first vs second."""
    first = {v.name: v.positive_first for v in VARIANTS}
    variant = {r.trial: r.variant for r in rows}
    gaps: dict[bool, list[float]] = {True: [], False: []}
    for t, pt in s["per_trial"].items():
        gaps[first[variant[int(t)]]].append(pt["control_tpr"] - pt["control_tnr"])
    if not gaps[True] or not gaps[False]:
        return ""
    g1, g2 = statistics.fmean(gaps[True]), statistics.fmean(gaps[False])
    text = (f" With the \"consistent\" option listed first, TPR − TNR averages {g1:+.2f}; "
            f"listed second, {g2:+.2f}.")
    if g1 > 0 > g2:
        text += " Laya leans toward whichever option comes first, so framing shifts its bias."
    return text


def render_readme_sections(summary: dict, manifest: dict, rows: list[Row],
                           charts: str) -> dict[str, str]:
    """The README's RESULTS (headline) and DETAILS (tables + charts) blocks."""
    s, agg = summary, summary["aggregate"]
    icon = "✅" if s["verdict"] == "FUNCTIONAL" else "❌"
    passed = sum(c["passed"] for c in s["checks"])

    def ms(key: str) -> str:
        return f"{agg[key]['mean']:.3f} ± {agg[key]['std']:.3f}"

    def rng(key: str) -> str:
        return f"{agg[key]['min']:.2f}–{agg[key]['max']:.2f}"

    verdict = (f"> **Verdict: {icon} {s['verdict']}**: {passed}/{len(s['checks'])} adapter checks "
               f"passed on {s['items']} TruthfulQA items × {s['trials']} trials "
               f"({s['evaluator_calls']} evaluator calls, run `{manifest['run_id']}`).")
    headline = (f"> Wrapped by `create_binary_classifier_evaluator`, Laya separates known-correct "
                f"from known-incorrect answers with **AUROC {ms('control_auroc')}** and "
                f"**balanced accuracy {ms('control_balanced_accuracy')}**, and judged "
                f"**{agg['llama_pass_rate']['mean']:.0%}** of Llama 3.2's answers truthful.")
    results = "\n".join([verdict, ">", headline, "", _picture("separation", charts)])

    per_trial = (f"Each trial uses a new Llama seed and frames the question to Laya differently "
                 f"(option keys, wording, and which option comes first). Across framings, TPR "
                 f"ranges {rng('control_tpr')} and TNR {rng('control_tnr')}, while balanced "
                 f"accuracy ranges {rng('control_balanced_accuracy')}.{_order_bias(s, rows)}")
    ece = agg["control_ece"]["mean"]
    calibrated = ("aren't calibrated: fit a temperature on your own labeled data before "
                  "thresholding on them" if ece > 0.1 else "are reasonably calibrated")
    calib = (f"Laya's P(correct) ranks answers with AUROC {agg['control_auroc']['mean']:.2f}, but "
             f"with mean ECE {ece:.3f} the probability values themselves {calibrated}.")
    details = [
        "### Adapter checks", "", *_checks_table(s), "",
        "### Metrics across trials", "", *_metrics_table(s), "",
        "### Per trial", "", per_trial, "",
        _picture("trials", charts), "", *_trials_table(s, rows), "",
        "### Calibration", "", calib, "",
        _picture("calibration", charts), "",
        "### Verdict stability on Llama answers", "", _stability(s),
    ]
    return {"RESULTS": results, "DETAILS": "\n".join(details)}


def update_readme(readme: Path, sections: dict[str, str]) -> None:
    """Replace the content between <!-- NAME:START --> and <!-- NAME:END --> markers."""
    text = readme.read_text(encoding="utf-8")
    for name, body in sections.items():
        pattern = re.compile(rf"(<!-- {name}:START -->\n).*?(\n<!-- {name}:END -->)", re.DOTALL)
        if not pattern.search(text):
            raise ValueError(f"README has no {name} markers")
        text = pattern.sub(lambda m, body=body: m.group(1) + body + m.group(2), text)
    readme.write_text(text, encoding="utf-8", newline="\n")


def _cell(text: str, width: int = 70) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= width else text[: width - 1] + "…"


def load_rows(path: Path) -> list[Row]:
    """Read rows.csv back into typed Rows, to re-render a report without re-running."""
    def opt_bool(v: str) -> bool | None:
        return None if v == "" else v == "True"

    with path.open(newline="", encoding="utf-8") as f:
        return [Row(
            trial=int(d["trial"]), variant=d["variant"], positive_key=d["positive_key"],
            seed=int(d["seed"]), item_id=int(d["item_id"]), category=d["category"],
            kind=d["kind"], question=d["question"], reference=d["reference"],
            response=d["response"], truth=opt_bool(d["truth"]), score=d["score"] == "True",
            key=d["key"], comment=d["comment"], laya_choice=d["laya_choice"],
            p_correct=float(d["p_correct"]), reference_seen=d["reference_seen"] or None,
            latency_ms=float(d["latency_ms"]),
        ) for d in csv.DictReader(f)]


def render_run(run_dir: Path, summary: dict, manifest: dict, rows: list[Row]) -> None:
    """Charts + REPORT.md for a run directory."""
    write_charts(run_dir / "charts", rows, summary)
    (run_dir / "REPORT.md").write_text(render_markdown(summary, manifest, rows), encoding="utf-8")


def publish_latest(results_dir: Path, run_dir: Path) -> None:
    latest = results_dir / "latest"
    if run_dir.resolve() == latest.resolve():
        return
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(run_dir, latest)


def write_run(results_dir: Path, summary: dict, manifest: dict, rows: list[Row]) -> Path:
    run_dir = results_dir / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (run_dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        writer.writerows(r.to_dict() for r in rows)
    render_run(run_dir, summary, manifest, rows)
    publish_latest(results_dir, run_dir)
    return run_dir

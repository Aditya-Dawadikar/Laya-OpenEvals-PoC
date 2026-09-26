"""Write a run to results/<run_id>/ and mirror it to results/latest/."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from .experiment import Row

METRIC_LABELS = {
    "control_balanced_accuracy": "Control balanced accuracy",
    "control_auroc": "Control AUROC (from Laya P(correct))",
    "control_tpr": "Control TPR (correct answers passed)",
    "control_tnr": "Control TNR (incorrect answers failed)",
    "control_ece": "Control ECE (lower = better calibrated)",
    "llama_pass_rate": "Llama answers judged truthful",
    "latency_p50_ms": "Evaluator latency p50 (ms)",
}


def _fmt(metric: str, v: float) -> str:
    return f"{v:.0f}" if metric.endswith("_ms") else f"{v:.3f}"


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
        "| | Check | Detail |",
        "|---|---|---|",
    ]
    out += [f"| {'✅' if c['passed'] else '❌'} | {c['name']} | {c['detail']} |" for c in s["checks"]]

    out += ["", "## Metrics across trials", "",
            "| Metric | Mean ± std | Min – max |", "|---|---|---|"]
    for key, label in METRIC_LABELS.items():
        a = s["aggregate"][key]
        out.append(f"| {label} | {_fmt(key, a['mean'])} ± {_fmt(key, a['std'])} | "
                   f"{_fmt(key, a['min'])} – {_fmt(key, a['max'])} |")

    out += ["", "## Per trial", "",
            "| Trial | Seed | Laya framing | Bal. acc | AUROC | ECE | Llama pass rate |",
            "|---|---|---|---|---|---|---|"]
    framings = {r.trial: (r.seed, r.variant) for r in rows}
    for t, pt in s["per_trial"].items():
        seed, variant = framings[int(t)]
        out.append(f"| {t} | {seed} | {variant} | {pt['control_balanced_accuracy']:.3f} | "
                   f"{pt['control_auroc']:.3f} | {pt['control_ece']:.3f} | {pt['llama_pass_rate']:.3f} |")

    st = s["llama_stability"]
    out += ["", "## Verdict stability on Llama answers", "",
            f"- Unanimous across all {s['trials']} trials: **{st['unanimous_items']}/{st['items']}** items",
            f"- Mean agreement with the majority verdict: **{st['mean_agreement_with_majority']:.1%}**",
            "", "## Sample control errors", "",
            "| Trial | Label | Laya P(correct) | Question | Response |", "|---|---|---|---|---|"]
    errors = [r for r in rows if r.truth is not None and r.score != r.truth][:8]
    for r in errors:
        out.append(f"| {r.trial} | {'correct' if r.truth else 'incorrect'} | {r.p_correct:.2f} | "
                   f"{_cell(r.question)} | {_cell(r.response)} |")
    if not errors:
        out.append("| – | – | – | none | – |")

    out += ["", "## Environment", "", "```json", json.dumps(m, indent=2), "```", ""]
    return "\n".join(out)


def _cell(text: str, width: int = 70) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= width else text[: width - 1] + "…"


def write_run(results_dir: Path, summary: dict, manifest: dict, rows: list[Row]) -> Path:
    run_dir = results_dir / manifest["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (run_dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        writer.writerows(r.to_dict() for r in rows)
    (run_dir / "REPORT.md").write_text(render_markdown(summary, manifest, rows), encoding="utf-8")

    latest = results_dir / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(run_dir, latest)
    return run_dir

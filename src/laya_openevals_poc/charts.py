"""SVG charts for the report and README, in a light and a dark variant (GitHub <picture>).

Blue is always "known-correct" and orange always "known-incorrect", across every chart.
Backgrounds are transparent so the charts sit on GitHub's own light/dark page.
"""
from __future__ import annotations

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory

from .experiment import Row

THEMES = {
    "light": {"correct": "#2a78d6", "incorrect": "#eb6834", "ink": "#0b0b0b",
              "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"correct": "#3987e5", "incorrect": "#d95926", "ink": "#ffffff",
             "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
CHARTS = ("separation", "trials", "calibration")


def _style(ax, c, *, xgrid: bool = False, ygrid: bool = True) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"])
    ax.tick_params(colors=c["muted"], length=0, labelsize=9)
    ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=c["grid"], linewidth=1)
    if xgrid:
        ax.xaxis.grid(True, color=c["grid"], linewidth=1)


def _title(fig, c, title: str, subtitle: str) -> None:
    """Title block at fixed inch offsets from the top, whatever the figure height."""
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / h, title, color=c["ink"], fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.44 / h, subtitle, color=c["ink2"], fontsize=9.5, va="top")


def _legend(fig, c, x: float, items: list[tuple[str, str, str]]) -> None:
    """One row of (glyph, color key, label) under the subtitle; text wears ink, not series color."""
    y = 1 - 0.82 / fig.get_figheight()
    for glyph, key, label in items:
        fig.text(x, y, glyph, color=c[key], fontsize=11, va="center")
        fig.text(x + 0.022, y, label, color=c["ink2"], fontsize=9, va="center")
        x += 0.03 + 0.0105 * len(label)


def _controls(rows: list[Row]) -> tuple[list[float], list[float]]:
    pos = [r.p_correct for r in rows if r.truth is True]
    neg = [r.p_correct for r in rows if r.truth is False]
    return pos, neg


def separation(rows: list[Row], summary: dict, c: dict) -> plt.Figure:
    """Two aligned histograms of Laya's P(correct): known-correct above, known-incorrect below."""
    pos, neg = _controls(rows)
    bins = [i / 20 for i in range(21)]
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.23, right=0.97, top=1 - 1.0 / 4.8, bottom=0.13, hspace=0.35)
    auc = summary["aggregate"]["control_auroc"]
    _title(fig, c, "Laya's P(correct) on answers with known labels",
           f"{len(pos) + len(neg)} TruthfulQA controls, all trials pooled · "
           f"AUROC {auc['mean']:.2f} ± {auc['std']:.2f}")
    for ax, vals, key, label in ((axes[0], pos, "correct", "Known-correct"),
                                 (axes[1], neg, "incorrect", "Known-incorrect")):
        _style(ax, c)
        ax.hist(vals, bins=bins, color=c[key], rwidth=0.82, zorder=2)
        ax.axvline(0.5, color=c["ink2"], linewidth=1, zorder=3)
        passed = sum(v > 0.5 for v in vals) / len(vals)
        verb = "passed" if key == "correct" else "failed"
        share = passed if key == "correct" else 1 - passed
        row = blended_transform_factory(fig.transFigure, ax.transAxes)  # page-left, panel-height
        ax.text(0.02, 0.62, label, transform=row, color=c["ink"], fontsize=10,
                fontweight="bold", ha="left")
        ax.text(0.02, 0.34, f"{share:.0%} {verb}", transform=row,
                color=c["ink2"], fontsize=9.5, ha="left")
        ax.yaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
    axes[0].text(0.505, 1.02, "decision boundary", transform=axes[0].get_xaxis_transform(),
                 color=c["ink2"], fontsize=8.5, va="bottom")
    axes[1].set_xlim(0, 1)
    axes[1].xaxis.set_major_locator(MultipleLocator(0.1))
    axes[1].set_xlabel("Laya P(correct)", color=c["ink2"], fontsize=9.5)
    return fig


def trials(rows: list[Row], summary: dict, c: dict) -> plt.Figure:
    """Dumbbell per trial: TPR (blue) and TNR (orange), balanced accuracy between them."""
    names = {r.trial: r.variant for r in rows}
    per = summary["per_trial"]
    order = sorted(per, key=int)
    n = len(order)
    h = 1.55 + 0.5 * n
    fig, ax = plt.subplots(figsize=(8, h))
    fig.subplots_adjust(left=0.37, right=0.97, top=1 - 1.12 / h, bottom=0.4 / h)
    bal = summary["aggregate"]["control_balanced_accuracy"]
    _title(fig, c, "Accuracy on known labels, trial by trial",
           f"Each trial is a new Llama seed and Laya framing · balanced accuracy "
           f"{bal['mean']:.2f} ± {bal['std']:.2f}")
    _legend(fig, c, 0.02, [("●", "correct", "TPR: correct answers passed"),
                          ("●", "incorrect", "TNR: incorrect answers failed"),
                          ("┃", "ink", "Balanced accuracy")])
    _style(ax, c, xgrid=True, ygrid=False)
    ax.spines["bottom"].set_visible(False)
    ax.axvline(0.5, color=c["ink2"], linewidth=1, zorder=1)
    for i, t in enumerate(order):
        y = n - 1 - i
        m = per[t]
        tpr, tnr, ba = m["control_tpr"], m["control_tnr"], m["control_balanced_accuracy"]
        ax.plot([tpr, tnr], [y, y], color=c["axis"], linewidth=2, zorder=2, solid_capstyle="round")
        ax.plot([ba, ba], [y - 0.3, y + 0.3], color=c["ink"], linewidth=2, zorder=3,
                solid_capstyle="round")
        for v, key in ((tpr, "correct"), (tnr, "incorrect")):
            ax.scatter(v, y, s=70, color=c[key], zorder=4, edgecolors="none")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"Trial {t}  ·  {names[int(t)]}" for t in reversed(order)],
                       color=c["ink"], fontsize=9.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.text(0.49, -0.55, "chance", color=c["ink2"], fontsize=8.5, ha="right", va="bottom")
    return fig


def calibration(rows: list[Row], summary: dict, c: dict) -> plt.Figure:
    """Reliability diagram: mean Laya P(correct) vs observed share correct, 10 bins."""
    ctrl = [r for r in rows if r.truth is not None]
    xs, ys, ns = [], [], []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        sel = [r for r in ctrl if lo <= r.p_correct < hi or (b == 9 and r.p_correct == 1.0)]
        if sel:
            xs.append(statistics.fmean(r.p_correct for r in sel))
            ys.append(statistics.fmean(float(r.truth) for r in sel))
            ns.append(len(sel))
    ece = summary["aggregate"]["control_ece"]["mean"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    fig.subplots_adjust(left=0.11, right=0.97, top=1 - 0.8 / 4.8, bottom=0.14)
    _title(fig, c, "Calibration of Laya's P(correct)",
           f"{len(ctrl)} known-label answers in 10 bins · dot area = answers per bin · "
           f"mean ECE {ece:.3f}")
    _style(ax, c, xgrid=True)
    ax.plot([0, 1], [0, 1], color=c["muted"], linewidth=1, zorder=1)
    ax.text(0.8, 0.765, "perfectly calibrated", color=c["ink2"], fontsize=8.5, ha="center",
            va="top", rotation=45, rotation_mode="anchor", transform_rotates_text=True)
    ax.plot(xs, ys, color=c["correct"], linewidth=2, zorder=2, solid_joinstyle="round")
    big = max(ns)
    ax.scatter(xs, ys, s=[24 + 220 * k / big for k in ns], color=c["correct"], zorder=3,
               edgecolors="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.05)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Laya P(correct)", color=c["ink2"], fontsize=9.5)
    ax.set_ylabel("Share actually correct", color=c["ink2"], fontsize=9.5)
    return fig


def write_charts(out_dir: Path, rows: list[Row], summary: dict) -> list[Path]:
    """Render every chart in both themes; returns the written files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "path", "svg.hashsalt": "laya-poc",
                         "font.family": "DejaVu Sans", "axes.facecolor": "none"})
    written = []
    for theme, c in THEMES.items():
        figs = {"separation": separation(rows, summary, c), "trials": trials(rows, summary, c),
                "calibration": calibration(rows, summary, c)}
        for name, fig in figs.items():
            path = out_dir / f"{name}-{theme}.svg"
            fig.savefig(path, format="svg", transparent=True, metadata={"Date": None})
            plt.close(fig)
            written.append(path)
    return written

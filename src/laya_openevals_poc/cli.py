"""`laya-poc run` / `laya-poc smoke` / `laya-poc report`."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime
from importlib import metadata

from dotenv import load_dotenv


def _openevals_source() -> str:
    """The exact openevals commit installed (the adapter under test)."""
    dist = metadata.distribution("openevals")
    raw = dist.read_text("direct_url.json")
    if raw:
        info = json.loads(raw)
        commit = info.get("vcs_info", {}).get("commit_id")
        if commit:
            return f"{info['url']}@{commit}"
        return info["url"]
    return f"openevals=={dist.version}"


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="laya-poc", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("run", "full experiment"), ("smoke", "tiny run: 6 items x 2 trials")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--samples", type=int, help="TruthfulQA items (default 50)")
        sp.add_argument("--trials", type=int, help="trials, each a new seed + Laya framing (default 5)")
        sp.add_argument("--seed", type=int, help="base seed (default 42)")
        sp.add_argument("--model", dest="ollama_model", help="Ollama model (default llama3.2:3b)")
    rp = sub.add_parser("report", help="re-render charts + REPORT.md from a saved run")
    rp.add_argument("run_dir", nargs="?", help="results/<run_id> (default results/latest)")
    rp.add_argument("--readme", action="store_true",
                    help="publish the run to results/latest and write it into README.md")
    return p.parse_args(argv)


def _report(run_dir_arg: str | None, readme: bool) -> int:
    from pathlib import Path

    from .config import REPO_ROOT, Settings
    from .report import load_rows, publish_latest, render_readme_sections, render_run, update_readme

    results_dir = Settings.from_env().results_dir
    run_dir = Path(run_dir_arg) if run_dir_arg else results_dir / "latest"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = load_rows(run_dir / "rows.csv")
    render_run(run_dir, summary, manifest, rows)
    print(f"Rendered {run_dir / 'REPORT.md'} and {run_dir / 'charts'}")
    if readme:
        publish_latest(results_dir, run_dir)
        update_readme(REPO_ROOT / "README.md",
                      render_readme_sections(summary, manifest, rows, "results/latest/charts/"))
        print("Updated README.md from results/latest")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    from .config import TRUTHFULQA_DATASET, TRUTHFULQA_REVISION, Settings, _pin_caches

    _pin_caches()
    args = _parse(argv)
    if args.cmd == "report":
        return _report(args.run_dir, args.readme)
    settings = Settings.from_env()
    if args.cmd == "smoke":
        settings = settings.override(samples=6, trials=2)
    settings = settings.override(samples=args.samples, trials=args.trials, seed=args.seed,
                                 ollama_model=args.ollama_model)

    from .data import load_truthfulqa
    from .experiment import run_experiment, summarize
    from .generator import OllamaGenerator
    from .judge import LayaBackend
    from .report import write_run

    started = time.perf_counter()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{args.cmd}"

    print(f"[1/4] Ollama at {settings.ollama_url}: ensuring {settings.ollama_model}")
    gen = OllamaGenerator(settings.ollama_url, settings.ollama_model,
                          settings.temperature, settings.max_tokens)
    ollama_version, digest = gen.version(), gen.ensure_model()

    print(f"[2/4] TruthfulQA @ {TRUTHFULQA_REVISION[:7]}: sampling {settings.samples} items")
    items = load_truthfulqa(settings.samples, settings.seed)

    print(f"[3/4] Loading Laya ({settings.laya_model})")
    backend = LayaBackend(settings.laya_model, settings.laya_subfolder, settings.laya_device)

    print(f"[4/4] Running {settings.trials} trials")
    rows = run_experiment(items, backend, gen.answer, settings.trials, settings.seed)
    summary = summarize(rows)

    manifest = {
        "run_id": run_id,
        "started_utc": run_id[:16],
        "duration_s": time.perf_counter() - started,
        "settings": settings.to_dict() | {"results_dir": None},
        "adapter": _openevals_source(),
        "generator": {"ollama": ollama_version, "model": settings.ollama_model, "digest": digest},
        "judge": {"laya": metadata.version("laya"), "checkpoint": settings.laya_model,
                  "subfolder": settings.laya_subfolder},
        "dataset": {"name": TRUTHFULQA_DATASET, "config": "generation", "split": "validation",
                    "revision": TRUTHFULQA_REVISION},
        "python": platform.python_version(),
        "torch": metadata.version("torch"),
        "platform": platform.platform(),
    }
    run_dir = write_run(settings.results_dir, summary, manifest, rows)

    print(f"\n{summary['verdict']}")
    for c in summary["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}  ({c['detail']})")
    agg = summary["aggregate"]
    print(f"  control balanced accuracy {agg['control_balanced_accuracy']['mean']:.3f} "
          f"± {agg['control_balanced_accuracy']['std']:.3f}   "
          f"AUROC {agg['control_auroc']['mean']:.3f} ± {agg['control_auroc']['std']:.3f}   "
          f"Llama pass rate {agg['llama_pass_rate']['mean']:.3f} ± {agg['llama_pass_rate']['std']:.3f}")
    print(f"\nReport: {run_dir / 'REPORT.md'}")
    return 0 if summary["verdict"] == "FUNCTIONAL" else 1


if __name__ == "__main__":
    sys.exit(main())

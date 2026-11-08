"""Difficulty calibration: run only the baseline harness (arm A) on every task, in a separate cache.

Used to choose a report split where the baseline is neither at the ceiling nor at zero. The runs are
kept apart from the experiment's cache so that the scored A runs are fresh: re-using the calibration
runs would select tasks on the very A scores the audit then compares against (regression to the mean
would favour B).

    python scripts/calibrate.py examples/polyglot/experiment.deepseek.yaml --cache-dir .calib-cache \
        --max-cost 5 --splits admission report retention proposal -j 4 --out calib-deepseek.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import yaml

from harness_audit.runner import Runner, load_experiment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--max-cost", type=float, required=True)
    ap.add_argument("--splits", nargs="+", default=["admission", "report", "retention", "proposal"])
    ap.add_argument("--only", nargs="*", help="task ids to run (default: all in the splits)")
    ap.add_argument("-j", "--concurrency", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.experiment.read_text())
    cfg["splits"] = a.splits
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", dir=a.experiment.parent, delete=False) as f:
        yaml.safe_dump(cfg, f)
    try:
        exp = load_experiment(f.name, seeds=1, aa=False)
    finally:
        Path(f.name).unlink()
    exp.tasks = [t for t in exp.tasks if not a.only or t.id in a.only]
    runner = Runner(exp, cache_dir=a.cache_dir, concurrency=a.concurrency)
    jobs = [j for j in runner.plan() if j.arm == "A"]
    recs = runner.execute(jobs, a.max_cost, progress=lambda s: print(s, flush=True))
    out = {
        r.task_id: {
            "score": r.score,
            "status": r.status,
            "cost_usd": r.usage.cost_usd if r.usage else None,
            "tool_calls": r.usage.tool_calls if r.usage else None,
        }
        for r in recs
    }
    a.out.write_text(json.dumps({"model": exp.config.model, "runs": out}, indent=2))
    print(f"spent ${runner.spent_usd:.4f}; wrote {a.out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from harness_audit.audit import run_audit
from harness_audit.evaluator import score_workspace
from harness_audit.harness import diff_harnesses, load_harness
from harness_audit.runner import BudgetError, Runner, load_experiment
from harness_audit.schema import Verdict
from harness_audit.tasks import load_tasks

from .fake_adapter import FakeAdapter

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture()
def exp_dir(tmp_path: Path) -> Path:
    d = tmp_path / "examples"
    shutil.copytree(EXAMPLES, d, ignore=shutil.ignore_patterns(".harness-audit-cache", "__pycache__", "out"))
    return d


def test_example_tasks_are_sound() -> None:
    """Buggy workspaces fail hidden tests and public tests; reference fixes pass both."""
    tasks = load_tasks(EXAMPLES / "tasks")
    assert len(tasks) == 20
    for t in tasks:
        assert score_workspace(t, t.workspace).score < 1.0, t.id
        public = subprocess.run([sys.executable, "-B", "test_public.py"], cwd=t.workspace, capture_output=True)
        assert public.returncode != 0, t.id


def test_reference_fixes_pass(tmp_path: Path) -> None:
    for t in load_tasks(EXAMPLES / "tasks"):
        work = tmp_path / t.id.replace("/", "_")
        shutil.copytree(t.workspace, work)
        module = next(p for p in work.glob("*.py") if p.name != "test_public.py")
        shutil.copy(t.evaluator_dir / "reference.py", module)
        assert score_workspace(t, work).score == 1.0, t.id
        assert subprocess.run([sys.executable, "test_public.py"], cwd=work, capture_output=True).returncode == 0, t.id


def test_harness_hash_and_diff(tmp_path: Path) -> None:
    a = load_harness(EXAMPLES / "harness_a")
    b = load_harness(EXAMPLES / "harness_b")
    assert a.hash != b.hash and b.skills == ["python-bugfix"] and a.skills == []
    arts, confounds = diff_harnesses(a, b)
    assert [(x.path, x.kind, x.change) for x in arts] == [("skills/python-bugfix/SKILL.md", "skill", "added")]
    assert arts[0].signatures and arts[0].signatures[0].tool == "Bash"
    assert confounds == []
    c = tmp_path / "c"
    shutil.copytree(EXAMPLES / "harness_b", c)
    cfg = yaml.safe_load((c / "harness.yaml").read_text())
    cfg["max_turns"] = 40
    (c / "harness.yaml").write_text(yaml.safe_dump(cfg))
    hc = load_harness(c)
    assert hc.hash != b.hash
    _, confounds = diff_harnesses(a, hc)
    assert confounds == ["max_turns: 20 -> 40"]


def test_dry_run_plan_and_estimate(exp_dir: Path) -> None:
    exp = load_experiment(exp_dir / "experiment.yaml", seeds=2)
    r = Runner(exp, adapter=FakeAdapter())
    plan = r.plan()
    # 17 tasks x (A, B) x 2 seeds + 8 report tasks x A2 x 2 seeds
    assert len(plan) == 17 * 2 * 2 + 8 * 2
    assert all(j.seed >= 1000 for j in plan if j.arm == "A2")
    est = r.estimate(plan)
    assert est.to_run == len(plan) and est.cached == 0
    assert 0 < est.typical_usd < est.worst_usd
    assert "to run" in r.describe(plan)


def test_execute_cache_and_audit(exp_dir: Path, tmp_path: Path) -> None:
    exp = load_experiment(exp_dir / "experiment.yaml", seeds=2)
    fake = FakeAdapter()
    r = Runner(exp, cache_dir=tmp_path / "cache", adapter=fake, workdir_base=tmp_path / "work")
    plan = r.plan()
    recs = r.execute(plan, max_cost=100.0)
    assert len(recs) == len(plan) and r.n_executed == len(plan) and fake.calls == len(plan)
    assert r.spent_usd == pytest.approx(0.01 * len(plan))
    assert all(x.score == 1.0 for x in recs if x.arm == "B")
    assert all(x.score < 1.0 for x in recs if x.arm == "A")
    res = run_audit(recs, r.manifest())
    assert res.claim("Activation").verdict == Verdict.PASS
    assert res.check("envelope").status.value == "pass"
    assert res.check("same_evaluator").status.value == "pass"
    # Second run: everything from cache, no adapter calls, no spend.
    fake2 = FakeAdapter()
    r2 = Runner(exp, cache_dir=tmp_path / "cache", adapter=fake2, workdir_base=tmp_path / "work")
    recs2 = r2.execute(r2.plan(), max_cost=0.001)
    assert fake2.calls == 0 and r2.n_cached == len(plan) and r2.spent_usd == 0
    assert [x.score for x in recs2] == [x.score for x in recs]
    assert r2.estimate(r2.plan()).to_run == 0


def test_budget_guard_stops_before_cap(exp_dir: Path, tmp_path: Path) -> None:
    exp = load_experiment(exp_dir / "experiment.yaml", seeds=1)
    fake = FakeAdapter(cost=0.05)
    r = Runner(exp, cache_dir=tmp_path / "cache", adapter=fake, concurrency=1)
    recs = r.execute(r.plan(), max_cost=0.35)
    # per_run_max_usd is 0.10 in the example: a job starts only if 0.10 is still available.
    assert r.spent_usd <= 0.35
    assert len(recs) == fake.calls == r.n_executed < len(r.plan())
    assert r.skipped


def test_unknown_cost_counts_as_reserved(exp_dir: Path, tmp_path: Path) -> None:
    exp = load_experiment(exp_dir / "experiment.yaml", seeds=1)
    fake = FakeAdapter(cost_known=False)
    r = Runner(exp, cache_dir=tmp_path / "cache", adapter=fake, concurrency=1)
    r.execute(r.plan(), max_cost=0.25)
    assert fake.calls == 2 and r.spent_usd == pytest.approx(0.2)


def test_changed_evaluator_rescores_from_cache(exp_dir: Path, tmp_path: Path) -> None:
    exp = load_experiment(exp_dir / "experiment.yaml", seeds=1)
    r = Runner(exp, cache_dir=tmp_path / "cache", adapter=FakeAdapter())
    r.execute(r.plan(), max_cost=100)
    hidden = exp_dir / "tasks/legacy/factorial/evaluator/test_hidden.py"
    hidden.write_text(hidden.read_text() + "\n\ndef test_extra():\n    assert False\n")
    exp2 = load_experiment(exp_dir / "experiment.yaml", seeds=1)
    fake = FakeAdapter()
    r2 = Runner(exp2, cache_dir=tmp_path / "cache", adapter=fake)
    recs = r2.execute(r2.plan(), max_cost=0.001)
    assert fake.calls == 0
    fact = [x for x in recs if x.task_id == "legacy/factorial" and x.arm == "B"]
    assert fact and fact[0].score == pytest.approx(6 / 7)


def test_invalid_budget(exp_dir: Path) -> None:
    exp = load_experiment(exp_dir / "experiment.yaml", seeds=1)
    with pytest.raises(BudgetError):
        Runner(exp, adapter=FakeAdapter()).execute([], max_cost=0)


def test_fatal_adapter_error_stops_the_run(exp_dir: Path, tmp_path: Path) -> None:
    from harness_audit.adapters import FatalAdapterError

    class NoKey(FakeAdapter):
        def preflight(self) -> None:
            raise FatalAdapterError("ANTHROPIC_API_KEY is not set")

    exp = load_experiment(exp_dir / "experiment.yaml", seeds=1)
    fake = NoKey()
    with pytest.raises(FatalAdapterError):
        Runner(exp, cache_dir=tmp_path / "c", adapter=fake).execute(Runner(exp, adapter=fake).plan(), max_cost=5)
    assert fake.calls == 0


def test_cli_run_dry_run_and_requires_max_cost(exp_dir: Path) -> None:
    from typer.testing import CliRunner

    from harness_audit.cli import app

    r = CliRunner().invoke(app, ["run", str(exp_dir / "experiment.yaml"), "--dry-run", "--seeds", "1"])
    assert r.exit_code == 0 and "estimated cost" in r.output
    r = CliRunner().invoke(app, ["run", str(exp_dir / "experiment.yaml"), "--seeds", "1"])
    assert r.exit_code == 2 and "--max-cost is required" in r.output


def test_cli_run_end_to_end_with_fake_adapter(exp_dir: Path) -> None:
    import json

    from typer.testing import CliRunner

    from harness_audit.cli import app

    cfg = yaml.safe_load((exp_dir / "experiment.yaml").read_text())
    cfg["adapter"] = "tests.fake_adapter:FakeAdapter"
    (exp_dir / "experiment.yaml").write_text(yaml.safe_dump(cfg))
    out = exp_dir / "out"
    args = ["run", str(exp_dir / "experiment.yaml"), "--seeds", "2", "--max-cost", "5", "--yes", "-o", str(out)]
    r = CliRunner().invoke(app, args)
    assert r.exit_code in (0, 1), r.output
    assert (out / "runs.jsonl").exists() and (out / "manifest.yaml").exists() and (out / "report/report.html").exists()
    verdicts = {c["claim"]: c["verdict"] for c in json.loads((out / "report/audit.json").read_text())["claims"]}
    assert verdicts["Activation"] == "PASS"
    assert "from cache" in r.output
    r2 = CliRunner().invoke(app, args)
    assert "on 0 new runs" in r2.output

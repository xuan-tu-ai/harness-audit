"""Out-of-harness scorer for example tasks: run hidden ``test_*`` functions in a subprocess.

Score = passed / total hidden tests, in [0, 1]. Hidden tests are plain ``test_*`` functions or\n``unittest.TestCase`` methods (as in Exercism-derived tasks); each test gets a 10 s alarm. The scorer never sees the agent's
transcript, only the final workspace, and its code plus the hidden tests define
the ``evaluator_hash``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .hashing import dir_hash
from .tasks import Task

RUNNER = r"""
import importlib.util, json, pathlib, signal, sys, unittest
work, hidden = sys.argv[1], sys.argv[2]
sys.path[:0] = [work, hidden]
passed, total, failures = 0, 0, []
PER_TEST_S = 10


class TestTimeout(BaseException):
    pass


def _alarm(signum, frame):
    raise TestTimeout(f"test took longer than {PER_TEST_S}s")


signal.signal(signal.SIGALRM, _alarm)


def flat(suite):
    for x in suite:
        if isinstance(x, unittest.TestSuite):
            yield from flat(x)
        else:
            yield x


def record(name, err):
    global passed, total
    total += 1
    if err is None:
        passed += 1
    else:
        failures.append({"test": name, "error": err[:200]})


for f in sorted(pathlib.Path(hidden).glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except BaseException as e:
        names = [n for n in getattr(mod, "__dict__", {}) if n.startswith("test_")] or ["<module>"]
        for _ in names:
            record(f.name, repr(e))
        continue
    # Plain test_* functions.
    for name in sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))):
        if isinstance(getattr(mod, name), type):
            continue
        signal.alarm(PER_TEST_S)
        try:
            getattr(mod, name)()
            record(f"{f.stem}.{name}", None)
        except BaseException as e:
            record(f"{f.stem}.{name}", repr(e))
        finally:
            signal.alarm(0)
    # unittest.TestCase methods (Exercism-style tests); each test method counts once.
    for case in flat(unittest.TestLoader().loadTestsFromModule(mod)):
        res = unittest.TestResult()
        signal.alarm(PER_TEST_S)
        try:
            case.run(res)
        except BaseException as e:
            res.errors.append((case, repr(e)))
        finally:
            signal.alarm(0)
        bad = res.failures + res.errors + [(c, "skipped: " + r) for c, r in res.skipped] + [
            (c, "unexpected success") for c in res.unexpectedSuccesses
        ]
        record(case.id(), None if not bad else str(bad[0][1]).strip().splitlines()[-1])
print(json.dumps({"passed": passed, "total": total, "failures": failures[:20]}))
"""

SCORER_VERSION = "hidden-tests-v2"


@dataclass(frozen=True)
class ScoreResult:
    score: float
    passed: int
    total: int
    failures: list[dict[str, str]]
    error: str | None = None


def evaluator_hash(task: Task) -> str:
    """Hash of the scorer program and the task's hidden evaluator directory."""
    return dir_hash(task.evaluator_dir, extra=(SCORER_VERSION + RUNNER).encode())


def score_workspace(task: Task, workspace: Path, timeout_s: float = 60.0) -> ScoreResult:
    """Copy the final workspace and the hidden tests to a scratch dir and run the tests."""
    with tempfile.TemporaryDirectory(prefix="harness-audit-score-") as tmp:
        work = Path(tmp) / "work"
        hidden = Path(tmp) / "hidden"
        shutil.copytree(workspace, work, ignore=shutil.ignore_patterns(".claude", "__pycache__"))
        shutil.copytree(task.evaluator_dir, hidden, ignore=shutil.ignore_patterns("__pycache__"))
        runner = Path(tmp) / "run_hidden.py"
        runner.write_text(RUNNER, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(runner), str(work), str(hidden)],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return ScoreResult(0.0, 0, 0, [], error=f"evaluator timed out after {timeout_s}s")
        try:
            out = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return ScoreResult(0.0, 0, 0, [], error=f"evaluator produced no result: {proc.stderr[-300:]}")
    total = int(out["total"])
    passed = int(out["passed"])
    return ScoreResult(passed / total if total else 0.0, passed, total, out["failures"])

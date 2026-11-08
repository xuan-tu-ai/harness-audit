"""Hidden-test scorer: plain test functions, unittest.TestCase methods, and per-test timeouts."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from harness_audit.evaluator import score_workspace
from harness_audit.tasks import load_task, load_tasks

POLYGLOT = Path(__file__).resolve().parent.parent / "examples" / "polyglot" / "tasks"


def make_task(tmp_path: Path, test_src: str, module_src: str) -> tuple[Path, Path]:
    root = tmp_path / "t"
    (root / "workspace").mkdir(parents=True)
    (root / "evaluator").mkdir()
    (root / "task.yaml").write_text("id: t\ngroup: g\nsplit: report\nprompt: p\n")
    (root / "evaluator" / "test_m.py").write_text(test_src)
    (root / "workspace" / "m.py").write_text(module_src)
    return root, root / "workspace"


def test_unittest_methods_and_functions_are_counted(tmp_path: Path) -> None:
    tests = (
        "import unittest\nfrom m import f\n"
        "def test_plain():\n    assert f(1) == 2\n"
        "class T(unittest.TestCase):\n"
        "    def test_ok(self):\n        self.assertEqual(f(2), 3)\n"
        "    def test_bad(self):\n        self.assertEqual(f(2), 4)\n"
        "    def test_raises(self):\n"
        "        with self.assertRaises(ValueError) as e:\n            f(-1)\n"
        "        self.assertEqual(e.exception.args[0], 'negative')\n"
    )
    root, ws = make_task(tmp_path, tests, "def f(x):\n    if x < 0:\n        raise ValueError('neg')\n    return x + 1\n")
    res = score_workspace(load_task(root), ws)
    assert (res.passed, res.total) == (2, 4)
    assert {f["test"].rsplit(".", 1)[-1] for f in res.failures} == {"test_bad", "test_raises"}


def test_a_hanging_test_fails_alone(tmp_path: Path) -> None:
    tests = "from m import f\ndef test_fast():\n    assert f(0) == 0\ndef test_slow():\n    assert f(1) == 1\n"
    root, ws = make_task(tmp_path, tests, "def f(x):\n    while x:\n        pass\n    return x\n")
    res = score_workspace(load_task(root), ws, timeout_s=60)
    assert (res.passed, res.total) == (1, 2) and "longer than" in res.failures[0]["error"]


@pytest.mark.parametrize("task", load_tasks(POLYGLOT), ids=lambda t: t.id)
def test_polyglot_reference_solutions_score_one_and_stubs_score_low(task, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    ws = tmp_path / "w"
    shutil.copytree(task.workspace, ws)
    assert score_workspace(task, ws).score < 0.6
    stub = next(p for p in ws.glob("*.py"))
    shutil.copy(task.evaluator_dir / "reference.py", stub)
    res = score_workspace(task, ws)
    assert res.score == 1.0, res.failures[:3]

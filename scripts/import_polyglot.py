"""Import the Python exercises of the Aider polyglot benchmark as harness-audit tasks.

Source: https://github.com/Aider-AI/polyglot-benchmark (exercises from the Exercism Python track,
MIT License, copyright Exercism). Usage::

    git clone --depth 1 https://github.com/Aider-AI/polyglot-benchmark /some/dir
    python scripts/import_polyglot.py /some/dir examples/polyglot/tasks [--splits examples/polyglot/splits.yaml]

Each task gets:

- ``workspace/``: the exercise stub and ``INSTRUCTIONS.md`` (introduction, instructions, Python
  append), as the agent sees them. No tests are given, as in the polyglot benchmark's first attempt.
- ``evaluator/``: the Exercism unit tests (renamed ``test_<module>.py``), any test helpers, and the
  Exercism example solution as ``reference.py`` (used only by the test suite to check the evaluator).

The prompt follows the polyglot benchmark's wording. Splits come from ``--splits`` (a mapping of
exercise slug to split); exercises not listed there get ``--default-split``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

PROMPT = (
    "INSTRUCTIONS.md in the current directory describes a programming exercise. "
    "Use those instructions to modify the supplied file: {files}. "
    "Don't change the names of existing functions or classes, as they may be referenced from other code like unit tests, etc. "
    "Only use standard libraries, don't suggest installing any packages. "
    "Hidden unit tests will check your solution."
)


def instructions(ex: Path) -> str:
    docs = ex / ".docs"
    parts = [docs / n for n in ("introduction.md", "instructions.md", "instructions.append.md")]
    return "\n\n".join(p.read_text(encoding="utf-8").strip() for p in parts if p.exists()) + "\n"


def import_one(ex: Path, out: Path, split: str) -> None:
    meta = json.loads((ex / ".meta" / "config.json").read_text(encoding="utf-8"))
    files = meta["files"]
    solution, tests, example = files["solution"], files["test"], files["example"]
    if len(solution) != 1 or len(example) != 1:
        raise SystemExit(f"{ex.name}: expected one solution and one example file")
    stub = solution[0]
    if out.exists():
        shutil.rmtree(out)
    ws, ev = out / "workspace", out / "evaluator"
    ws.mkdir(parents=True)
    ev.mkdir()
    shutil.copy(ex / stub, ws / stub)
    (ws / "INSTRUCTIONS.md").write_text(instructions(ex), encoding="utf-8")
    for t in tests:
        # Tests are renamed test_<module>.py; helper modules imported by the tests keep their names.
        name = "test_" + t.removesuffix("_test.py") + ".py" if t.endswith("_test.py") else t
        shutil.copy(ex / t, ev / name)
    shutil.copy(ex / example[0], ev / "reference.py")
    spec = {
        "id": f"exercism/{ex.name}",
        "group": "exercism",
        "split": split,
        "prompt": PROMPT.format(files=f"`{stub}`"),
    }
    (out / "task.yaml").write_text(yaml.safe_dump(spec, sort_keys=False, width=10_000), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("benchmark", type=Path, help="clone of Aider-AI/polyglot-benchmark")
    ap.add_argument("out", type=Path, help="tasks directory to write")
    ap.add_argument("--splits", type=Path, help="YAML mapping exercise slug -> split")
    ap.add_argument("--default-split", default="admission")
    ap.add_argument("--only-listed", action="store_true", help="import only exercises listed in --splits")
    a = ap.parse_args()
    splits = yaml.safe_load(a.splits.read_text()) if a.splits else {}
    practice = a.benchmark / "python" / "exercises" / "practice"
    n = 0
    for ex in sorted(p for p in practice.iterdir() if p.is_dir()):
        if a.only_listed and ex.name not in splits:
            continue
        import_one(ex, a.out / ex.name, splits.get(ex.name, a.default_split))
        n += 1
    print(f"imported {n} exercises into {a.out}")


if __name__ == "__main__":
    main()

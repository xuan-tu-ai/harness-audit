"""Task directories: ``task.yaml``, ``workspace/`` (given to the agent), ``evaluator/`` (hidden).

The evaluator directory is never copied into the agent's working directory, so
the final scorer stays outside the harness's editable surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from .hashing import dir_hash, sha256_bytes
from .io import InputError
from .schema import SPLITS, Split


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    group: str
    split: Split
    prompt: str


@dataclass(frozen=True)
class Task:
    spec: TaskSpec
    root: Path
    task_hash: str

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def evaluator_dir(self) -> Path:
        return self.root / "evaluator"


def load_task(root: str | Path) -> Task:
    base = Path(root)
    try:
        spec = TaskSpec.model_validate(yaml.safe_load((base / "task.yaml").read_text(encoding="utf-8")))
    except FileNotFoundError as e:
        raise InputError(f"{base}: missing task.yaml") from e
    except (yaml.YAMLError, ValidationError) as e:
        raise InputError(f"{base / 'task.yaml'}: {e}") from e
    for sub in ("workspace", "evaluator"):
        if not (base / sub).is_dir():
            raise InputError(f"{base}: missing {sub}/ directory")
    # Content hash of what the agent sees: prompt + workspace (not the id, split, or hidden evaluator).
    # A renamed copy keeps its hash; a changed evaluator re-scores cached runs instead of re-running them.
    content = spec.prompt.encode() + b"\0" + dir_hash(base / "workspace").encode()
    return Task(spec, base, sha256_bytes(content))


def load_tasks(root: str | Path, splits: list[str] | None = None) -> list[Task]:
    """Load every ``*/task.yaml`` under ``root`` (sorted by id), optionally filtered by split."""
    base = Path(root)
    if not base.is_dir():
        raise InputError(f"{base}: tasks directory not found")
    tasks = [load_task(p.parent) for p in sorted(base.glob("**/task.yaml"))]
    ids = [t.id for t in tasks]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise InputError(f"{base}: duplicate task ids {dup}")
    bad = [s for s in (splits or []) if s not in SPLITS]
    if bad:
        raise InputError(f"unknown splits {bad}")
    if splits is not None:
        tasks = [t for t in tasks if t.spec.split in splits]
    return sorted(tasks, key=lambda t: t.id)

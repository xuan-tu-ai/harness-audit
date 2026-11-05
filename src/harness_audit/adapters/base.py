"""The adapter interface. The core never imports an agent framework; adapters do."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..harness import Harness
from ..schema import Event, Usage
from ..tasks import Task


@dataclass(frozen=True)
class RunRequest:
    """Everything an adapter needs for one run.

    ``workdir`` already contains a fresh copy of the task workspace; the adapter may
    install harness files into it (e.g. ``.claude/skills``) and must leave the
    agent's final files there for the evaluator.
    """

    task: Task
    harness: Harness
    workdir: Path
    seed: int
    model: str
    max_budget_usd: float
    timeout_s: float


@dataclass
class AdapterResult:
    events: list[Event] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    status: str = "ok"  # ok | error | budget_exceeded | max_turns | timeout
    error: str | None = None
    model: str | None = None


class FatalAdapterError(RuntimeError):
    """A problem that affects every run (e.g. wrong credentials): the runner stops at once."""


@runtime_checkable
class Adapter(Protocol):
    """Run one task under one harness. ``id`` is part of the cache key."""

    id: str

    async def run(self, req: RunRequest) -> AdapterResult: ...

    def installed_paths(self) -> list[str]:
        """Workdir-relative paths the adapter created (excluded from the scored workspace)."""
        ...

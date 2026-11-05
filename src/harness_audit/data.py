"""Grouping helpers: per-task seed means and task pairing between arms.

This is the one place where runs are aggregated to tasks, so every claim uses
the same unit (the independent task) for its statistics.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .schema import Manifest, RunRecord, TaskEntry

Metric = Callable[[RunRecord], float | None]


def score_metric(r: RunRecord) -> float | None:
    return r.score


def usage_metric(field: str) -> Metric:
    """Metric reading ``r.usage.<field>`` (``None`` when missing)."""

    def _get(r: RunRecord) -> float | None:
        if r.usage is None:
            return None
        v = getattr(r.usage, field, None)
        return None if v is None else float(v)

    return _get


def select(records: Iterable[RunRecord], *, arm: str | None = None, split: str | None = None) -> list[RunRecord]:
    return [r for r in records if (arm is None or r.arm == arm) and (split is None or r.split == split)]


def task_means(records: Iterable[RunRecord], metric: Metric = score_metric) -> dict[str, float]:
    """Mean of ``metric`` over the replicates of each task (missing values skipped)."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in records:
        v = metric(r)
        if v is not None:
            acc[r.task_id].append(v)
    return {t: float(np.mean(v)) for t, v in acc.items() if v}


@dataclass
class Paired:
    """Per-task paired values for two arms on one split."""

    task_ids: list[str]
    a: np.ndarray
    b: np.ndarray
    unpaired: list[str]

    @property
    def diffs(self) -> np.ndarray:
        return self.b - self.a

    @property
    def m(self) -> int:
        return len(self.task_ids)


def paired(
    records: Iterable[RunRecord],
    split: str,
    arm_a: str,
    arm_b: str,
    metric: Metric = score_metric,
    task_filter: Callable[[str], bool] | None = None,
) -> Paired:
    """Pair arms ``arm_a`` and ``arm_b`` by task on ``split`` (seed means per task)."""
    recs = [r for r in records if r.split == split]
    if task_filter is not None:
        recs = [r for r in recs if task_filter(r.task_id)]
    ma = task_means((r for r in recs if r.arm == arm_a), metric)
    mb = task_means((r for r in recs if r.arm == arm_b), metric)
    common = sorted(set(ma) & set(mb))
    unpaired = sorted(set(ma) ^ set(mb))
    return Paired(
        common,
        np.array([ma[t] for t in common], dtype=float),
        np.array([mb[t] for t in common], dtype=float),
        unpaired,
    )


def seeds_per_task(records: Iterable[RunRecord]) -> dict[str, int]:
    acc: dict[str, set[int]] = defaultdict(set)
    for r in records:
        acc[r.task_id].add(r.seed)
    return {t: len(s) for t, s in acc.items()}


def registry(manifest: Manifest, records: Iterable[RunRecord]) -> list[TaskEntry]:
    """Task registry: manifest entries, completed with (task, split) pairs seen in records."""
    entries: dict[tuple[str, str], TaskEntry] = {(t.task_id, t.split): t for t in manifest.tasks}
    for r in records:
        key = (r.task_id, r.split)
        if key not in entries:
            entries[key] = TaskEntry(task_id=r.task_id, split=r.split, group=r.task_group, task_hash=r.task_hash)
        else:
            e = entries[key]
            if e.group is None and r.task_group is not None:
                e.group = r.task_group
            if e.task_hash is None and r.task_hash is not None:
                e.task_hash = r.task_hash
    return list(entries.values())


def tasks_in(reg: Iterable[TaskEntry], split: str) -> list[TaskEntry]:
    return [t for t in reg if t.split == split]


def group_of(reg: Iterable[TaskEntry]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for t in reg:
        if out.get(t.task_id) is None:
            out[t.task_id] = t.group
    return out


def proposal_groups(manifest: Manifest, reg: Iterable[TaskEntry]) -> set[str]:
    """Groups that formed the update: declared in the manifest, else those of the proposal split."""
    if manifest.proposal_groups is not None:
        return set(manifest.proposal_groups)
    return {t.group for t in tasks_in(reg, "proposal") if t.group is not None}

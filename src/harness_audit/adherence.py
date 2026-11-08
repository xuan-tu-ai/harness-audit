"""Adherence: did the trajectory follow the instructions of a skill or prompt? (report §6.2.2)

The companion report defines the *carried* grade of an instruction or skill as adherence:
"trajectories scored against a rubric extracted from the skill text". harness-audit reads a declared
rubric (a list of :class:`~harness_audit.schema.RubricItem`) against each run's trace. Items that a
trace cannot show (an instruction to reason about edge cases, say) are declared ``check: judge`` and
are scored only from ``rubric_judgment`` events that an external judge, human or model, writes into
the run. :class:`Judge` and :func:`apply_judge` are that plug; the audit itself never calls a model.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .schema import ChangedArtifact, Event, RubricItem, RunRecord

Outcome = Literal["met", "unmet", "n/a", "unjudged"]
JUDGMENT = "rubric_judgment"


@dataclass
class RubricResult:
    """Per-item outcomes of one run for one artifact."""

    items: dict[str, Outcome] = field(default_factory=dict)
    required: dict[str, bool] = field(default_factory=dict)

    @property
    def score(self) -> float | None:
        """Share of applicable, decided items that were met (None if none could be decided)."""
        decided = [o for o in self.items.values() if o in ("met", "unmet")]
        return round(sum(o == "met" for o in decided) / len(decided), 4) if decided else None

    @property
    def carried(self) -> bool | None:
        """Every required item that could be decided was met; None if no required item could be decided."""
        req = [o for i, o in self.items.items() if self.required[i] and o in ("met", "unmet")]
        if not req:
            return None
        return all(o == "met" for o in req)

    @property
    def unjudged_required(self) -> list[str]:
        return [i for i, o in self.items.items() if self.required[i] and o == "unjudged"]


def _span(e: Event, item: RubricItem, start: int = 0, *, use_ok: bool = True) -> int | None:
    """End position of the item's match in the call's summary, searching from ``start``; None if no match."""
    if e.type != "tool_call":
        return None
    if use_ok and item.ok is not None and e.ok is not item.ok:
        return None
    summary = e.summary or ""
    pairs = [(s.tool, s.pattern) for s in item.any_of] if item.any_of else [(item.tool, item.pattern)]
    ends = []
    for tool, pattern in pairs:
        if tool is not None and not re.fullmatch(tool, e.name or ""):
            continue
        m = re.compile(pattern).search(summary, start)
        if m:
            ends.append(m.end())
    return min(ends) if ends else None


def _matches(e: Event, item: RubricItem, *, use_ok: bool = True) -> bool:
    return _span(e, item, use_ok=use_ok) is not None


def evaluate(events: Iterable[Event], art: ChangedArtifact, start: float | None = 0.0) -> RubricResult:
    """Score one trajectory against the artifact's rubric.

    ``start`` is the time from which calls count (the moment the artifact was reached or loaded);
    ``None`` means the artifact never entered the run, so every trace item is unmet.
    """
    rubric = art.effective_rubric()
    out = RubricResult(required={i.id: i.required for i in rubric})
    evs = sorted(events, key=lambda e: e.t)
    calls = [] if start is None else [e for e in evs if e.type == "tool_call" and e.t >= start]
    judged = {
        e.name: e.ok
        for e in evs
        if e.type == JUDGMENT and e.ok is not None and e.name and (e.artifact is None or _same(e.artifact, art.path))
    }
    by_id = {i.id: i for i in rubric}
    first: dict[str, tuple[int, int] | None] = {}  # (call index, end of match in its summary) of the first hit

    def trigger(item: RubricItem) -> int | None:
        """Index of the first failed call matching any item listed in ``on_failure_of``."""
        idx = [
            k
            for k, e in enumerate(calls)
            if e.ok is False and any(_matches(e, by_id[r], use_ok=False) for r in item.on_failure_of)
        ]
        return min(idx) if idx else None

    def solve(iid: str) -> Outcome:
        if iid in out.items:
            return out.items[iid]
        item = by_id[iid]
        first[iid] = None
        if item.check == "judge":
            v = judged.get(iid)
            out.items[iid] = "unjudged" if v is None else ("met" if v else "unmet")
            return out.items[iid]
        # Earliest admissible position: a later call, or later text in the same call (one shell
        # command often chains several steps, e.g. `python test_public.py && python -c ...`).
        lo: tuple[int, int] = (-1, 0)
        if item.on_failure_of:
            t = trigger(item)
            if t is None:
                out.items[iid] = "n/a"
                return "n/a"
            lo = (t + 1, 0)  # strictly after the failed call
        for dep in item.after:
            o = solve(dep)
            if o == "unmet":
                out.items[iid] = "unmet"
                return "unmet"
            d = first.get(dep)
            if o == "met" and d is not None:
                lo = max(lo, d)
        hit: tuple[int, int] | None = None
        for k, e in enumerate(calls):
            if k < lo[0]:
                continue
            end = _span(e, item, lo[1] if k == lo[0] else 0)
            if end is not None:
                hit = (k, end)
                break
        first[iid] = hit
        out.items[iid] = "met" if hit is not None else "unmet"
        return out.items[iid]

    for i in rubric:
        solve(i.id)
    return out


def _same(a: str, b: str) -> bool:
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


class Judge(Protocol):
    """Decides ``check: judge`` rubric items for one run (an LLM or a person). Not used by the audit."""

    def judge(self, run: RunRecord, art: ChangedArtifact, items: list[RubricItem]) -> dict[str, bool | None]: ...


def apply_judge(records: list[RunRecord], arts: list[ChangedArtifact], judge: Judge) -> list[RunRecord]:
    """Return copies of ``records`` with ``rubric_judgment`` events added for every judge item decided."""
    out: list[RunRecord] = []
    for r in records:
        if r.events is None:
            out.append(r)
            continue
        new = list(r.events)
        t_end = max((e.t for e in r.events), default=0.0)
        for art in arts:
            items = [i for i in art.effective_rubric() if i.check == "judge"]
            if not items or not art.applies(r.task_id, r.task_group):
                continue
            for iid, v in judge.judge(r, art, items).items():
                if v is not None:
                    new.append(Event(t=t_end, type=JUDGMENT, artifact=art.path, name=iid, ok=bool(v)))
        out.append(r.model_copy(update={"events": new}))
    return out

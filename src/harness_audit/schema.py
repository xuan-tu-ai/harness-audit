"""Data models for run records, experiment manifests, and audit results.

The core schema is framework-independent: any agent framework can export
``RunRecord`` lines (JSONL) plus a ``Manifest`` (YAML) and be audited.
"""

from __future__ import annotations

import fnmatch
import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Split = Literal["proposal", "admission", "report", "retention"]
SPLITS: tuple[str, ...] = ("proposal", "admission", "report", "retention")

EventType = Literal["artifact_loaded", "artifact_reached", "artifact_carried", "tool_call"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Usage(_Model):
    """Inference resources of one scored run (the 'envelope')."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cost_usd: float | None = None
    wall_s: float | None = None
    tool_calls: int | None = None
    num_turns: int | None = None
    model_calls: int | None = None
    """Model requests in the run, including sub-agents, critics and verifiers inside the loop."""
    attempts: int | None = None
    """Independent tries of the task inside one scored run (best-of-n, pass@k); 1 for a single try."""


class Event(_Model):
    """One trace event used for Activation."""

    t: float = 0.0
    type: str
    artifact: str | None = None
    hash: str | None = None
    evidence: str | None = None
    name: str | None = None
    args_hash: str | None = None
    summary: str | None = None
    ok: bool | None = None
    """For ``tool_call``: whether the call succeeded (None = not recorded). For ``rubric_judgment``: the verdict."""


class RunRecord(_Model):
    """One run of one harness arm on one task with one replicate index."""

    run_id: str
    task_id: str
    task_group: str | None = None
    task_hash: str | None = None
    split: Split
    arm: str
    harness_hash: str
    seed: int = 0
    model: str
    score: float
    evaluator_hash: str
    evaluator_hash_after: str | None = None
    """Evaluator hash re-read after the agent ran; differs from ``evaluator_hash`` if the run changed the scorer."""
    status: str = "ok"
    usage: Usage | None = None
    events: list[Event] | None = None
    model_versions: list[str] | None = None
    """Exact model ids that served the main loop, as returned by the API (``model`` is what was requested)."""
    decoding: dict[str, Any] | None = None
    """Generation settings in effect: reasoning effort, thinking, temperature, turn limit, fallback model."""
    runtime: str | None = None
    """Agent runtime and version (e.g. ``claude-code/2.1.318``)."""

    @field_validator("score")
    @classmethod
    def _score_in_unit_interval(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {v}")
        return v


class Signature(_Model):
    """A declared trace feature: a tool call whose summary matches ``pattern``."""

    tool: str | None = None
    pattern: str


class RubricItem(_Model):
    """One instruction of a skill or prompt, checked against the trajectory (report §6.2.2, adherence).

    A trace item is met by a ``tool_call`` whose tool name fully matches ``tool`` (a regex) and whose
    summary matches ``pattern``, or by any of ``any_of``; ``ok`` also requires the call's outcome.
    ``after`` orders it after the first call that met each listed item. ``on_failure_of`` makes it
    conditional: it applies only in runs where a call matching one of the listed items failed, and is
    then met only by a matching call after that failure. A ``judge`` item cannot be read from the trace;
    it is met or unmet only through ``rubric_judgment`` events written by an external judge.
    """

    id: str
    text: str | None = Field(None, description="the instruction from the artifact this item checks")
    check: Literal["trace", "judge"] = "trace"
    tool: str | None = None
    pattern: str = ""
    any_of: list[Signature] = Field(default_factory=list)
    ok: bool | None = None
    after: list[str] = Field(default_factory=list)
    on_failure_of: list[str] = Field(default_factory=list)
    required: bool = True

    @field_validator("tool", "pattern")
    @classmethod
    def _valid_regex(cls, v: str | None) -> str | None:
        if v:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"invalid regex {v!r}: {e}") from e
        return v


def _check_rubric(items: list[RubricItem]) -> list[RubricItem]:
    ids = [i.id for i in items]
    dup = sorted({x for x in ids if ids.count(x) > 1})
    if dup:
        raise ValueError(f"duplicate rubric item ids: {dup}")
    known = set(ids)
    for i in items:
        bad = [r for r in i.after + i.on_failure_of if r not in known or r == i.id]
        if bad:
            raise ValueError(f"rubric item {i.id!r} refers to unknown or itself: {bad}")
    # `after` must be acyclic.
    state: dict[str, int] = {}
    deps = {i.id: i.after for i in items}

    def visit(n: str) -> None:
        if state.get(n) == 1:
            raise ValueError(f"rubric `after` has a cycle through {n!r}")
        if state.get(n) == 2:
            return
        state[n] = 1
        for d in deps[n]:
            visit(d)
        state[n] = 2

    for n in deps:
        visit(n)
    return items


ArtifactKind = Literal["skill", "system_prompt", "file", "tool", "code", "config"]


class ChangedArtifact(_Model):
    """One harness artifact that differs between A and B."""

    path: str
    kind: ArtifactKind = "file"
    change: Literal["added", "modified", "removed"] = "modified"
    hash_a: str | None = None
    hash_b: str | None = None
    signatures: list[Signature] = Field(default_factory=list)
    rubric: list[RubricItem] = Field(
        default_factory=list,
        description="instructions drawn from the artifact, checked against each trajectory; carried = every "
        "required, checkable item met. Without a rubric, `signatures` act as one required item.",
    )
    applies_to: list[str] | None = Field(
        None,
        description="task ids or task groups (glob patterns) where this update is meant to apply, declared before "
        "seeing outcomes; None means every scored run. The denominator of the load and reach rates (Eq. 11).",
    )

    @property
    def carried_requires_reached(self) -> bool:
        """Interpreted guidance in a system prompt has no 'reached' event of its own."""
        return self.kind not in ("system_prompt", "config")

    @field_validator("rubric")
    @classmethod
    def _rubric_ok(cls, v: list[RubricItem]) -> list[RubricItem]:
        return _check_rubric(v)

    def effective_rubric(self) -> list[RubricItem]:
        """The declared rubric, or the signatures as a single required item (the pre-rubric behaviour)."""
        if self.rubric:
            return self.rubric
        if self.signatures:
            return [RubricItem(id="signature", text="declared trace signature", any_of=self.signatures)]
        return []

    @property
    def executable(self) -> bool:
        """Executable paths condition carry on reach; interpreted guidance conditions it on load (Eq. 11)."""
        return self.kind in ("tool", "code")

    def applies(self, task_id: str, group: str | None) -> bool:
        """Whether a run on this task falls under the declared applicability rule."""
        if self.applies_to is None:
            return True
        keys = [task_id] + ([group] if group else [])
        return any(fnmatch.fnmatchcase(k, pat) for pat in self.applies_to for k in keys)


class Oracle(_Model):
    """How the scorer was kept out of the candidate's reach (Table 5, oracle integrity)."""

    outside_workspace: bool | None = Field(
        None, description="hidden tests, reference solutions and scoring code are not in the agent's workspace"
    )
    protected: list[str] = Field(
        default_factory=list,
        description="regex patterns; a traced tool call whose summary matches one touched the scorer",
    )


class SearchCost(_Model):
    """Compute spent finding the update (proposal, ranking, admission), stated apart from the scored runs."""

    usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    rollouts: int | None = None
    candidates: int | None = None
    notes: str | None = None


class Arms(_Model):
    baseline: str = "A"
    candidate: str = "B"
    replicate: str | None = "A2"


class SelectionConfig(_Model):
    """Inputs for the Selection claim and the admission gate (paper §3.2)."""

    K: int = Field(1, ge=1, description="max candidates compared per round")
    T: int = Field(1, ge=1, description="number of rounds")
    delta: float = Field(0.05, gt=0, lt=1)
    admission_data: Literal["admission", "report", "proposal", "none"] = "admission"
    report_uses: int = Field(1, ge=0, description="times the report split informed a decision")
    report_choice: Literal["fixed", "selected"] = "fixed"
    n_compared: int = Field(1, ge=1, description="rows the reported config was chosen among")


class Thresholds(_Model):
    alpha: float = 0.05
    envelope_ratio: float = 1.25
    regression: float = 0.02
    low_reach: float = 0.2
    n_boot: int = 10_000
    n_perm: int = 20_000
    rng_seed: int = 0


class TaskEntry(_Model):
    task_id: str
    split: Split
    group: str | None = None
    task_hash: str | None = None


class Manifest(_Model):
    """Description of one A → B experiment."""

    experiment: str = "experiment"
    baseline: str | None = None
    round: int | None = None
    arms: Arms = Field(default_factory=Arms)
    harness: dict[str, str] = Field(default_factory=dict)
    changed_artifacts: list[ChangedArtifact] = Field(default_factory=list)
    confounds: list[str] = Field(default_factory=list)
    proposal_groups: list[str] | None = None
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    oracle: Oracle = Field(default_factory=Oracle)
    search_cost: SearchCost | None = None
    thresholds: Thresholds = Field(default_factory=Thresholds)
    tasks: list[TaskEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_task_split(self) -> Manifest:
        seen: dict[tuple[str, str], int] = {}
        for t in self.tasks:
            seen[(t.task_id, t.split)] = seen.get((t.task_id, t.split), 0) + 1
        dups = [k for k, n in seen.items() if n > 1]
        if dups:
            raise ValueError(f"duplicate task registry entries: {dups[:5]}")
        return self


class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
    NOTE = "note"


class Verdict(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class CheckResult(_Model):
    """The outcome of one concrete check."""

    id: str
    title: str
    status: Status
    reason: str
    rubric: bool = True
    rubric_status: Status | None = None
    """Status the published rubric would give, when harness-audit is deliberately stricter."""
    required: bool = True
    criterion: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class ClaimResult(_Model):
    claim: Literal["Selection", "Activation", "Attribution", "Improvement"]
    verdict: Verdict
    rubric_grade: Literal["C", "O", "N", "U"]
    summary: str
    checks: list[CheckResult]


class AuditResult(_Model):
    tool: str = "harness-audit"
    version: str
    experiment: str
    primary_comparison: str
    n_records: int
    warnings: list[str] = Field(default_factory=list)
    claims: list[ClaimResult]

    def claim(self, name: str) -> ClaimResult:
        for c in self.claims:
            if c.claim == name:
                return c
        raise KeyError(name)

    def check(self, check_id: str) -> CheckResult:
        for c in self.claims:
            for ch in c.checks:
                if ch.id == check_id:
                    return ch
        raise KeyError(check_id)

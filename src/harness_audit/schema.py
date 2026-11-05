"""Data models for run records, experiment manifests, and audit results.

The core schema is framework-independent: any agent framework can export
``RunRecord`` lines (JSONL) plus a ``Manifest`` (YAML) and be audited.
"""

from __future__ import annotations

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
    status: str = "ok"
    usage: Usage | None = None
    events: list[Event] | None = None

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


ArtifactKind = Literal["skill", "system_prompt", "file", "tool", "code", "config"]


class ChangedArtifact(_Model):
    """One harness artifact that differs between A and B."""

    path: str
    kind: ArtifactKind = "file"
    change: Literal["added", "modified", "removed"] = "modified"
    hash_a: str | None = None
    hash_b: str | None = None
    signatures: list[Signature] = Field(default_factory=list)

    @property
    def carried_requires_reached(self) -> bool:
        """Interpreted guidance in a system prompt has no 'reached' event of its own."""
        return self.kind not in ("system_prompt", "config")


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

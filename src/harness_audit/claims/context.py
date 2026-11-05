"""Shared state for the claim modules and the check → verdict rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .. import data
from ..schema import CheckResult, ClaimResult, Manifest, RunRecord, Status, TaskEntry, Verdict
from ..stats import PairedEstimate, paired_estimate

PRIMARY_SPLIT = "report"


@dataclass
class Context:
    """Records, manifest and derived views shared by all claims."""

    records: list[RunRecord]
    manifest: Manifest
    warnings: list[str] = field(default_factory=list)

    @property
    def A(self) -> str:  # noqa: N802 - arm labels are conventionally capitalised
        return self.manifest.arms.baseline

    @property
    def B(self) -> str:  # noqa: N802
        return self.manifest.arms.candidate

    @property
    def A2(self) -> str | None:  # noqa: N802
        return self.manifest.arms.replicate

    @property
    def th(self):
        return self.manifest.thresholds

    @cached_property
    def reg(self) -> list[TaskEntry]:
        return data.registry(self.manifest, self.records)

    @cached_property
    def groups(self) -> dict[str, str | None]:
        return data.group_of(self.reg)

    @cached_property
    def proposal_groups(self) -> set[str]:
        return data.proposal_groups(self.manifest, self.reg)

    def estimate(self, diffs: np.ndarray, level: float | None = None, salt: int = 0) -> PairedEstimate:
        """Paired bootstrap CI + sign-flip p-value with the manifest's settings."""
        lvl = level if level is not None else 1.0 - self.th.alpha
        return paired_estimate(diffs, level=lvl, n_boot=self.th.n_boot, n_perm=self.th.n_perm, seed=self.th.rng_seed + salt)

    @cached_property
    def primary(self) -> data.Paired:
        """B vs A scores on the report split (the primary comparison)."""
        return data.paired(self.records, PRIMARY_SPLIT, self.A, self.B)

    @cached_property
    def primary_estimate(self) -> PairedEstimate | None:
        if self.primary.m == 0:
            return None
        k = self.manifest.selection.n_compared
        level = 1.0 - self.th.alpha / k
        return self.estimate(self.primary.diffs, level=level)

    def gain_is_significant(self) -> bool:
        est = self.primary_estimate
        return bool(est is not None and est.ci_low > 0 and est.p_value < self.th.alpha)


def verdict_of(checks: list[CheckResult], rubric_only: bool = False) -> Verdict:
    """FAIL if any fail; PARTIAL if any warn or a required check is missing; UNKNOWN if nothing evaluable."""
    if rubric_only:
        checks = [c.model_copy(update={"status": c.rubric_status}) if c.rubric_status else c for c in checks if c.rubric]
    graded = [c for c in checks if c.status != Status.NOTE]
    if not graded:
        return Verdict.UNKNOWN
    if any(c.status == Status.FAIL for c in graded):
        return Verdict.FAIL
    if any(c.status == Status.WARN for c in graded):
        return Verdict.PARTIAL
    required = [c for c in graded if c.required]
    if not required or all(c.status == Status.SKIP for c in required):
        return Verdict.UNKNOWN
    if any(c.status == Status.SKIP for c in required):
        return Verdict.PARTIAL
    return Verdict.PASS


_GRADE = {Verdict.PASS: "C", Verdict.PARTIAL: "O", Verdict.FAIL: "N", Verdict.UNKNOWN: "U"}


def make_claim(name: str, checks: list[CheckResult]) -> ClaimResult:
    """Build a ClaimResult with verdict, rubric grade and a one-line summary."""
    v = verdict_of(checks)
    g = _GRADE[verdict_of(checks, rubric_only=True)]
    if v == Verdict.PASS:
        summary = "All checks passed."
    elif v == Verdict.UNKNOWN:
        missing = [c.title for c in checks if c.status == Status.SKIP]
        summary = "Not enough data: " + "; ".join(missing) if missing else "Not enough data."
    else:
        worst = Status.FAIL if v == Verdict.FAIL else Status.WARN
        bad = [c for c in checks if c.status == worst] or [c for c in checks if c.status == Status.SKIP]
        summary = " ".join(c.reason for c in bad[:2])
    return ClaimResult(claim=name, verdict=v, rubric_grade=g, summary=summary, checks=checks)  # type: ignore[arg-type]


def fmt(x: float | None, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}"


def fmt_ci(lo: float, hi: float, nd: int = 3) -> str:
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"

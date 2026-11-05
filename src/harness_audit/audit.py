"""Entry point of the core: records + manifest → AuditResult."""

from __future__ import annotations

from collections import Counter

from . import __version__
from .claims import Context, audit_activation, audit_attribution, audit_improvement, audit_selection
from .claims.context import PRIMARY_SPLIT
from .schema import AuditResult, Manifest, RunRecord


def _consistency_warnings(records: list[RunRecord], manifest: Manifest) -> list[str]:
    out: list[str] = []
    arms = Counter(r.arm for r in records)
    for arm in (manifest.arms.baseline, manifest.arms.candidate):
        if arms.get(arm, 0) == 0:
            out.append(f"No records for arm {arm!r}.")
    known = {manifest.arms.baseline, manifest.arms.candidate, manifest.arms.replicate}
    extra = sorted(a for a in arms if a not in known)
    if extra:
        out.append(f"Records with unknown arms were ignored: {extra}.")
    reg = {t.task_id: {e.split for e in manifest.tasks if e.task_id == t.task_id} for t in manifest.tasks}
    mismatched = sorted({r.task_id for r in records if r.task_id in reg and r.split not in reg[r.task_id]})
    if mismatched:
        out.append(f"{len(mismatched)} task(s) appear in records under a split the manifest does not list: {mismatched[:5]}.")
    bad = Counter(r.status for r in records if r.status != "ok")
    if bad:
        out.append(
            "Runs with non-ok status are kept and scored as reported: "
            + ", ".join(f"{k}={v}" for k, v in sorted(bad.items()))
            + "."
        )
    if not any(r.split == PRIMARY_SPLIT for r in records):
        out.append("No records on the report split: the primary comparison is empty.")
    return out


def run_audit(records: list[RunRecord], manifest: Manifest) -> AuditResult:
    """Evaluate the four claims for the A → B comparison described by ``manifest``."""
    ctx = Context(records=records, manifest=manifest)
    ctx.warnings.extend(_consistency_warnings(records, manifest))
    claims = [audit_selection(ctx), audit_activation(ctx), audit_attribution(ctx), audit_improvement(ctx)]
    a, b = manifest.arms.baseline, manifest.arms.candidate
    return AuditResult(
        version=__version__,
        experiment=manifest.experiment,
        primary_comparison=f"arm {b} vs arm {a} on the '{PRIMARY_SPLIT}' split ({ctx.primary.m} paired tasks)",
        n_records=len(records),
        warnings=ctx.warnings,
        claims=claims,
    )

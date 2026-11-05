"""Improvement: how far does the difference reach?

Rubric (Harness-RSI ``data/claim-audit-rubric.md``, CC BY 4.0): ``C`` is scored on
**task distance** alone — held-out tasks, transfer to an unseen benchmark, or a
retention probe; ``O`` for partial overlap; ``N`` when reported only on the tasks
that produced the update. Baseline and round are recorded, not graded.

harness-audit additionally requires (both outside the rubric grade): the gain
holds on task groups other than those that formed the update, and the update
does not regress on the retention split beyond a user threshold.
"""

from __future__ import annotations

import math

from .. import data
from ..schema import CheckResult, ClaimResult, Status
from ..stats import hoeffding_epsilon
from .context import PRIMARY_SPLIT, Context, fmt_ci, make_claim


def audit_improvement(ctx: Context) -> ClaimResult:
    checks = [_distance(ctx), _heldout(ctx), _retention(ctx)]
    checks.append(
        CheckResult(
            id="context",
            title="Baseline and round",
            status=Status.PASS,
            rubric=False,
            required=False,
            reason=f"Baseline: {ctx.manifest.baseline or 'arm ' + ctx.A}; round: {ctx.manifest.round or 'n/a'} (recorded, not graded).",
            metrics={"baseline": ctx.manifest.baseline, "round": ctx.manifest.round},
        )
    )
    return make_claim("Improvement", checks)


def _distance(ctx: Context) -> CheckResult:
    crit = "held-out tasks / unseen benchmark / retention probe → C; partly overlapping → O; only update-forming tasks → N"
    report = data.tasks_in(ctx.reg, PRIMARY_SPLIT)
    proposal = data.tasks_in(ctx.reg, "proposal")
    pg = ctx.proposal_groups
    if not report:
        return CheckResult(
            id="task_distance", title="Task distance", status=Status.SKIP, criterion=crit, reason="No report tasks."
        )
    rgroups = {t.task_id: t.group for t in report}
    in_pg = sorted(t for t, g in rgroups.items() if g in pg)
    metrics = {
        "n_report": len(report),
        "proposal_groups": sorted(pg),
        "report_tasks_in_proposal_groups": len(in_pg),
        "report_tasks_in_heldout_groups": len(report) - len(in_pg),
    }
    if proposal:
        ids = {t.task_id for t in proposal}
        hashes = {t.task_hash for t in proposal if t.task_hash}
        over = sorted(t.task_id for t in report if t.task_id in ids or (t.task_hash and t.task_hash in hashes))
        metrics["overlap_with_proposal"] = over
        if len(over) == len(report):
            return CheckResult(
                id="task_distance",
                title="Task distance",
                status=Status.FAIL,
                criterion=crit,
                metrics=metrics,
                reason="Reported only on the tasks that formed the update.",
            )
        if over:
            return CheckResult(
                id="task_distance",
                title="Task distance",
                status=Status.WARN,
                criterion=crit,
                metrics=metrics,
                reason=f"{len(over)}/{len(report)} report tasks also formed the update (partial overlap).",
            )
        label = (
            "held-out groups"
            if not in_pg
            else (
                "held-out tasks in the proposal groups"
                if len(in_pg) == len(report)
                else "held-out tasks, some in held-out groups"
            )
        )
        return CheckResult(
            id="task_distance",
            title="Task distance",
            status=Status.PASS,
            criterion=crit,
            metrics=metrics,
            reason=f"Report tasks are disjoint from the update-forming tasks ({label}).",
        )
    if pg:
        if not in_pg:
            return CheckResult(
                id="task_distance",
                title="Task distance",
                status=Status.PASS,
                criterion=crit,
                metrics=metrics,
                reason="All report tasks come from groups that did not form the update.",
            )
        return CheckResult(
            id="task_distance",
            title="Task distance",
            status=Status.WARN,
            criterion=crit,
            metrics=metrics,
            reason="Proposal tasks are not listed and some report tasks share proposal groups; task-level separation is not shown.",
        )
    return CheckResult(
        id="task_distance",
        title="Task distance",
        status=Status.SKIP,
        criterion=crit,
        metrics=metrics,
        reason="Neither proposal tasks nor proposal groups are given; task distance is unknown.",
    )


def _group_table(ctx: Context) -> dict[str, dict[str, object]]:
    p = ctx.primary
    by_group: dict[str, list[int]] = {}
    for i, t in enumerate(p.task_ids):
        by_group.setdefault(ctx.groups.get(t) or "(none)", []).append(i)
    table: dict[str, dict[str, object]] = {}
    for gi, (g, idx) in enumerate(sorted(by_group.items())):
        est = ctx.estimate(p.diffs[idx], salt=100 + gi)
        table[g] = {
            "in_proposal_groups": g in ctx.proposal_groups,
            "m_tasks": len(idx),
            "score_A": round(float(p.a[idx].mean()), 4),
            "score_B": round(float(p.b[idx].mean()), 4),
            **{k: v for k, v in est.as_dict().items() if k in ("mean", "ci_low", "ci_high", "p_value")},
        }
    return table


def _heldout(ctx: Context) -> CheckResult:
    crit = "gain holds on task groups other than those that formed the update (harness-audit; not in the rubric grade)"
    p = ctx.primary
    if p.m == 0:
        return CheckResult(
            id="heldout_gain",
            title="Gain on held-out groups",
            status=Status.SKIP,
            rubric=False,
            criterion=crit,
            reason="No paired report runs.",
        )
    pg = ctx.proposal_groups
    table = _group_table(ctx)
    if not pg:
        return CheckResult(
            id="heldout_gain",
            title="Gain on held-out groups",
            status=Status.NOTE,
            rubric=False,
            required=False,
            criterion=crit,
            metrics={"per_group": table},
            reason="Proposal groups are unknown; transfer beyond them cannot be checked.",
        )
    held = [i for i, t in enumerate(p.task_ids) if ctx.groups.get(t) not in pg]
    same = [i for i, t in enumerate(p.task_ids) if ctx.groups.get(t) in pg]
    if not held:
        return CheckResult(
            id="heldout_gain",
            title="Gain on held-out groups",
            status=Status.NOTE,
            rubric=False,
            required=False,
            criterion=crit,
            metrics={"per_group": table},
            reason="No report tasks outside the proposal groups; transfer is untested.",
        )
    eh = ctx.estimate(p.diffs[held], salt=11)
    es = ctx.estimate(p.diffs[same], salt=12) if same else None
    metrics = {"per_group": table, "heldout": eh.as_dict(), "same_group": es.as_dict() if es else None}
    if eh.ci_low > 0 and eh.p_value < ctx.th.alpha:
        return CheckResult(
            id="heldout_gain",
            title="Gain on held-out groups",
            status=Status.PASS,
            rubric=False,
            criterion=crit,
            metrics=metrics,
            reason=f"Held-out groups gain {eh.mean:+.3f} (CI {fmt_ci(eh.ci_low, eh.ci_high)}, {eh.m} tasks).",
        )
    if es is not None and es.ci_low > 0 and es.p_value < ctx.th.alpha:
        why = (
            f"Gain confined to proposal groups: {es.mean:+.3f} there (CI {fmt_ci(es.ci_low, es.ci_high)}) vs "
            f"{eh.mean:+.3f} on held-out groups (CI {fmt_ci(eh.ci_low, eh.ci_high)})."
        )
    else:
        why = f"No gain distinguished from zero on held-out groups: {eh.mean:+.3f} (CI {fmt_ci(eh.ci_low, eh.ci_high)}, {eh.m} tasks)."
    return CheckResult(
        id="heldout_gain",
        title="Gain on held-out groups",
        status=Status.WARN,
        rubric=False,
        criterion=crit,
        metrics=metrics,
        reason=why,
    )


def _retention(ctx: Context) -> CheckResult:
    crit = "no regression on an earlier distribution beyond the user threshold (harness-audit; not in the rubric grade)"
    pr = data.paired(ctx.records, "retention", ctx.A, ctx.B)
    thr = ctx.th.regression
    if pr.m == 0:
        return CheckResult(
            id="retention",
            title="Retention (non-regression)",
            status=Status.WARN,
            rubric=False,
            criterion=crit,
            reason="Retention untested: no paired runs on the retention split.",
        )
    drop = pr.a - pr.b
    est = ctx.estimate(drop, salt=21)
    eps = hoeffding_epsilon(pr.m, 1, 1, ctx.manifest.selection.delta)
    cert = est.mean + eps
    metrics = {
        "drop": est.as_dict(),
        "threshold": thr,
        "score_A": round(float(pr.a.mean()), 4),
        "score_B": round(float(pr.b.mean()), 4),
        "hoeffding_upper_drop": round(cert, 6) if math.isfinite(cert) else None,
        "certified_non_regression": bool(cert <= thr),
    }
    ci = fmt_ci(est.ci_low, est.ci_high)
    if est.mean > thr and est.ci_low > 0:
        return CheckResult(
            id="retention",
            title="Retention (non-regression)",
            status=Status.FAIL,
            rubric=False,
            criterion=crit,
            metrics=metrics,
            reason=f"Regression on retention: score drops {est.mean:.3f} (CI {ci}) > threshold {thr:.3f} over {pr.m} tasks.",
        )
    if est.mean > thr:
        return CheckResult(
            id="retention",
            title="Retention (non-regression)",
            status=Status.WARN,
            rubric=False,
            criterion=crit,
            metrics=metrics,
            reason=f"Possible regression: score drops {est.mean:.3f} (CI {ci}), above threshold {thr:.3f} but not significant.",
        )
    cert_txt = " Certified at 1−δ." if cert <= thr else ""
    return CheckResult(
        id="retention",
        title="Retention (non-regression)",
        status=Status.PASS,
        rubric=False,
        criterion=crit,
        metrics=metrics,
        reason=f"No regression on {pr.m} retention tasks: drop {est.mean:+.3f} ≤ {thr:.3f} (CI {ci}).{cert_txt}",
    )

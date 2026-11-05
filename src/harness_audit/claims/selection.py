"""Selection: was the reported number measured, or was it picked?

Rubric (Harness-RSI ``data/claim-audit-rubric.md``, CC BY 4.0): four channels can
spend the reporting evidence — proposal, ranking, admission, report choice.
``C`` when none touched it and the partition is shown; ``O`` for partial overlap,
a held-out score used for promotion, or asserted split names; ``N`` when the
reported number is itself a choice or the update was generated from it.
The admission gate is the paper's §3.2 rule ``Δ̂ + sqrt(2·ln(K·T/δ)/m) < 0``.
"""

from __future__ import annotations

from .. import data
from ..schema import CheckResult, ClaimResult, Status, TaskEntry
from ..stats import hoeffding_gate
from .context import PRIMARY_SPLIT, Context, make_claim


def _overlap(a: list[TaskEntry], b: list[TaskEntry]) -> tuple[set[str], set[str]]:
    """Return (task ids of ``a`` also in ``b`` by id, ids of ``a`` matching ``b`` only by content hash)."""
    ids_b = {t.task_id for t in b}
    hashes_b = {t.task_hash for t in b if t.task_hash}
    by_id = {t.task_id for t in a if t.task_id in ids_b}
    by_hash = {t.task_id for t in a if t.task_id not in ids_b and t.task_hash and t.task_hash in hashes_b}
    return by_id, by_hash


def _disjoint_check(
    cid: str, title: str, report: list[TaskEntry], other: list[TaskEntry], other_name: str, full_is_fail: bool, criterion: str
) -> CheckResult:
    if not report:
        return CheckResult(id=cid, title=title, status=Status.SKIP, reason="No report tasks.", criterion=criterion)
    if not other:
        return CheckResult(
            id=cid,
            title=title,
            status=Status.PASS,
            criterion=criterion,
            reason=f"No {other_name} tasks are listed, so none overlap the report split.",
            metrics={"n_report": len(report), f"n_{other_name}": 0},
        )
    by_id, by_hash = _overlap(report, other)
    n_over = len(by_id) + len(by_hash)
    frac = n_over / len(report)
    metrics = {
        "n_report": len(report),
        f"n_{other_name}": len(other),
        "overlap_by_id": sorted(by_id),
        "overlap_by_content_hash": sorted(by_hash),
        "overlap_fraction": round(frac, 4),
    }
    if n_over == 0:
        return CheckResult(
            id=cid,
            title=title,
            status=Status.PASS,
            criterion=criterion,
            metrics=metrics,
            reason=f"Report and {other_name} splits share no task id or content hash.",
        )
    status = Status.FAIL if (full_is_fail and frac >= 1.0) else Status.WARN
    near = f" ({len(by_hash)} only by content hash)" if by_hash else ""
    return CheckResult(
        id=cid,
        title=title,
        status=status,
        criterion=criterion,
        metrics=metrics,
        reason=f"{n_over}/{len(report)} report tasks also appear in the {other_name} split{near}.",
    )


def audit_selection(ctx: Context) -> ClaimResult:
    sel = ctx.manifest.selection
    reg = ctx.reg
    report = data.tasks_in(reg, PRIMARY_SPLIT)
    proposal = data.tasks_in(reg, "proposal")
    admission = data.tasks_in(reg, "admission")
    checks: list[CheckResult] = []

    # Partition shown rather than asserted.
    if not report:
        checks.append(
            CheckResult(
                id="partition_shown",
                title="Partition shown",
                status=Status.SKIP,
                reason="No tasks in the report split.",
                criterion="split names asserted without the partition being shown → O",
            )
        )
    elif proposal:
        checks.append(
            CheckResult(
                id="partition_shown",
                title="Partition shown",
                status=Status.PASS,
                reason=f"Task registry lists {len(proposal)} proposal and {len(report)} report tasks.",
                criterion="split names asserted without the partition being shown → O",
                metrics={s: len(data.tasks_in(reg, s)) for s in ("proposal", "admission", "report", "retention")},
            )
        )
    else:
        what = "only proposal groups are declared" if ctx.manifest.proposal_groups else "no proposal tasks are listed"
        checks.append(
            CheckResult(
                id="partition_shown",
                title="Partition shown",
                status=Status.WARN,
                reason=f"Separation from the proposal data cannot be checked: {what}.",
                criterion="split names asserted without the partition being shown → O",
            )
        )

    checks.append(
        _disjoint_check(
            "split_disjoint_proposal",
            "Report ∩ proposal",
            report,
            proposal,
            "proposal",
            full_is_fail=True,
            criterion="proposal channel: update generated from reporting outcomes → N; partly coincides → O",
        )
    )
    checks.append(
        _disjoint_check(
            "split_disjoint_admission",
            "Report ∩ admission",
            report,
            admission,
            "admission",
            full_is_fail=False,
            criterion="admission channel: a held-out score also used for promotion → O",
        )
    )

    # Which data the acceptance decision consulted.
    ad = sel.admission_data
    if ad == "report":
        st, why = Status.WARN, "The acceptance rule consulted the report split (held-out score used for promotion)."
    elif ad == "none":
        st, why = Status.PASS, "Unconditional commit: no selection step spent the report split."
    elif ad == "proposal":
        st, why = Status.PASS, "Acceptance used proposal tasks, not the report split."
    else:
        st, why = Status.PASS, "Acceptance used a separate admission split."
    checks.append(
        CheckResult(
            id="admission_data",
            title="Admission data",
            status=st,
            reason=why,
            criterion="admission channel",
            metrics={"admission_data": ad},
        )
    )

    # Report choice: round / configuration selected on the report split.
    if sel.report_choice == "selected":
        checks.append(
            CheckResult(
                id="report_choice",
                title="Report choice",
                status=Status.FAIL,
                reason=f"The reported configuration was selected among {sel.n_compared} on the report split.",
                criterion="headline configuration picked on the reporting partition → N",
                metrics={"report_choice": "selected", "n_compared": sel.n_compared},
            )
        )
    else:
        checks.append(
            CheckResult(
                id="report_choice",
                title="Report choice",
                status=Status.PASS,
                reason="Round and configuration were fixed before scoring the report split.",
                criterion="report choice channel",
                metrics={"report_choice": "fixed"},
            )
        )

    # Reused holdout.
    if sel.report_uses > 1:
        checks.append(
            CheckResult(
                id="reused_holdout",
                title="Reused holdout",
                status=Status.WARN,
                reason=f"Reused holdout: the report split has informed {sel.report_uses} decisions.",
                criterion="adaptive reuse of a holdout (cf. Dwork et al., The Reusable Holdout, 2015)",
                metrics={"report_uses": sel.report_uses},
            )
        )
    else:
        checks.append(
            CheckResult(
                id="reused_holdout",
                title="Reused holdout",
                status=Status.PASS,
                reason=f"Report split used {sel.report_uses} time(s).",
                criterion="adaptive reuse of a holdout",
                metrics={"report_uses": sel.report_uses},
            )
        )

    checks.append(_gate_check(ctx, admission, proposal))
    return make_claim("Selection", checks)


def _gate_check(ctx: Context, admission: list[TaskEntry], proposal: list[TaskEntry]) -> CheckResult:
    sel = ctx.manifest.selection
    crit = "admit iff Δ̂ + sqrt(2·ln(K·T/δ)/m) < 0 on m fresh tasks (paper §3.2)"
    split = "admission" if data.paired(ctx.records, "admission", ctx.A, ctx.B).m > 0 else PRIMARY_SPLIT
    p = data.paired(ctx.records, split, ctx.A, ctx.B)
    if p.m == 0:
        return CheckResult(
            id="admission_gate",
            title="Admission gate",
            status=Status.SKIP,
            rubric=False,
            reason="No paired admission or report runs to evaluate the gate.",
            criterion=crit,
        )
    loss_diffs = (1.0 - p.b) - (1.0 - p.a)
    g = hoeffding_gate(loss_diffs, K=sel.K, T=sel.T, delta=sel.delta)
    metrics = {
        "split": split,
        "m_tasks": g.m,
        "delta_hat_loss": round(g.delta_hat, 6),
        "epsilon": round(g.epsilon, 6),
        "bound": round(g.bound, 6),
        "K": g.K,
        "T": g.T,
        "delta": g.delta,
        "passed": g.passed,
        "seeds_per_task": sorted(set(data.seeds_per_task(r for r in ctx.records if r.split == split).values())),
    }
    where = "admission split" if split == "admission" else "report split (no admission split present)"
    fresh_note = ""
    if split == "admission" and proposal:
        by_id, by_hash = _overlap(admission, proposal)
        if by_id or by_hash:
            fresh_note = f" {len(by_id) + len(by_hash)} admission tasks also formed the update, so they are not fresh."
            metrics["admission_overlap_proposal"] = sorted(by_id | by_hash)
    if g.passed and not fresh_note:
        return CheckResult(
            id="admission_gate",
            title="Admission gate",
            status=Status.PASS,
            rubric=False,
            criterion=crit,
            metrics=metrics,
            reason=f"Gate cleared on the {where}: Δ̂ {g.delta_hat:+.3f} + ε {g.epsilon:.3f} = {g.bound:+.3f} < 0 (m={g.m}).",
        )
    return CheckResult(
        id="admission_gate",
        title="Admission gate",
        status=Status.WARN,
        rubric=False,
        criterion=crit,
        metrics=metrics,
        reason=(
            f"Gate not cleared on the {where}: Δ̂ {g.delta_hat:+.3f} + ε {g.epsilon:.3f} = {g.bound:+.3f} ≥ 0 "
            f"(m={g.m} independent tasks).{fresh_note}"
        )
        if not g.passed
        else f"Gate cleared numerically, but{fresh_note}",
    )

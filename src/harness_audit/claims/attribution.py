"""Attribution: did the difference come from the update, not noise, compute, or the evaluator?

Rubric (Harness-RSI ``data/claim-audit-rubric.md``, CC BY 4.0): three conditions —
the arms differ only in the evolved component; the inference envelope of the
scored runs is controlled; the margin exceeds execution variation that was
measured ("a test distinguishes it from zero at the 5% level; an interval for the
difference excludes zero"). ``C`` when all three hold, ``O`` when one or two do,
``N`` for one run per task with no variance estimate and no resource accounting.

harness-audit is stricter in one place: a changed evaluator or model is ``FAIL``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .. import data
from ..schema import CheckResult, ClaimResult, RunRecord, Status
from .context import PRIMARY_SPLIT, Context, fmt_ci, make_claim, p_note

ENVELOPE_FIELDS = ("output_tokens", "num_turns", "tool_calls")
REPORTED_FIELDS = ("input_tokens", "cost_usd", "wall_s")


def _values(runs: list[RunRecord], attr: str) -> set[str]:
    return {getattr(r, attr) for r in runs}


def audit_attribution(ctx: Context) -> ClaimResult:
    runs = [r for r in ctx.records if r.split == PRIMARY_SPLIT and r.arm in (ctx.A, ctx.B)]
    a_runs = [r for r in runs if r.arm == ctx.A]
    b_runs = [r for r in runs if r.arm == ctx.B]
    checks: list[CheckResult] = []

    if not a_runs or not b_runs:
        checks.append(
            CheckResult(
                id="measured_variation",
                title="Measured variation",
                status=Status.SKIP,
                reason="The report split lacks runs for one of the arms.",
            )
        )
        return make_claim("Attribution", checks)

    # Evaluator and model held fixed (hard failures). Evaluators may differ between tasks,
    # but every run of one task, in either arm, must use the same one.
    per_task: dict[str, set[str]] = defaultdict(set)
    for r in runs:
        per_task[r.task_id].add(r.evaluator_hash)
    mixed = sorted(t for t, hs in per_task.items() if len(hs) > 1)
    ev = {arm: sorted(_values([r for r in runs if r.arm == arm], "evaluator_hash"))[:5] for arm in (ctx.A, ctx.B)}
    if mixed:
        checks.append(
            CheckResult(
                id="same_evaluator",
                title="Same evaluator",
                status=Status.FAIL,
                rubric_status=Status.WARN,
                reason=f"The evaluator differs between runs of the same task ({len(mixed)}/{len(per_task)} tasks); "
                "scores are not comparable.",
                criterion="the final evaluator stays outside the candidate's editable surface",
                metrics={"tasks_with_mixed_evaluators": mixed[:20], "evaluator_hashes_sample": ev},
            )
        )
    else:
        checks.append(
            CheckResult(
                id="same_evaluator",
                title="Same evaluator",
                status=Status.PASS,
                reason=f"Each task was scored by one evaluator in both arms ({len(per_task)} tasks).",
                metrics={"evaluator_hashes_sample": ev},
            )
        )
    models = {arm: sorted(_values([r for r in runs if r.arm == arm], "model")) for arm in (ctx.A, ctx.B)}
    if len(set(models[ctx.A]) | set(models[ctx.B])) > 1:
        checks.append(
            CheckResult(
                id="same_model",
                title="Same model",
                status=Status.FAIL,
                metrics={"models": models},
                rubric_status=Status.WARN,
                reason="The model differs between or within the arms; this is not a harness comparison.",
                criterion="the analysis holds the model M fixed",
            )
        )
    else:
        checks.append(
            CheckResult(
                id="same_model",
                title="Same model",
                status=Status.PASS,
                metrics={"models": models},
                reason=f"Both arms used {models[ctx.A][0]}.",
            )
        )

    checks.append(_isolated(ctx, a_runs, b_runs))
    p = ctx.primary
    if p.m < 2:
        checks.append(
            CheckResult(
                id="measured_variation",
                title="Measured variation",
                status=Status.FAIL,
                reason=f"Only {p.m} paired task(s); no variance estimate is possible.",
                criterion="one run per task, no variance estimate → N",
                metrics={"m_tasks": p.m},
            )
        )
        return make_claim("Attribution", checks)
    spt = data.seeds_per_task(runs)
    checks.append(
        CheckResult(
            id="measured_variation",
            title="Measured variation",
            status=Status.PASS,
            reason=f"{p.m} paired tasks with {min(spt.values())}–{max(spt.values())} replicate(s) each; unit = task.",
            criterion="repeated runs of frozen states, or a paired test over task outcomes",
            metrics={
                "m_tasks": p.m,
                "unpaired_tasks": p.unpaired,
                "replicates_min": min(spt.values()),
                "replicates_max": max(spt.values()),
            },
        )
    )
    if p.unpaired:
        ctx.warnings.append(f"{len(p.unpaired)} report task(s) ran in only one arm and were dropped from pairing.")

    checks.append(_envelope(ctx))
    checks.append(_margin(ctx))
    checks.append(_aa(ctx))
    return make_claim("Attribution", checks)


def _isolated(ctx: Context, a_runs: list[RunRecord], b_runs: list[RunRecord]) -> CheckResult:
    ha, hb = _values(a_runs, "harness_hash"), _values(b_runs, "harness_hash")
    metrics = {
        "harness_A": sorted(ha),
        "harness_B": sorted(hb),
        "changed_artifacts": [a.path for a in ctx.manifest.changed_artifacts],
        "confounds": ctx.manifest.confounds,
    }
    crit = "the arms differ only in the evolved component (same model, scaffold, decoding, tools, evaluator)"
    if ha & hb:
        return CheckResult(
            id="isolated_update",
            title="Arms differ only in the update",
            status=Status.FAIL,
            criterion=crit,
            metrics=metrics,
            reason="A and B ran the same harness; there is no update to attribute.",
        )
    issues = []
    if len(ha) > 1 or len(hb) > 1:
        issues.append(f"harness hash drifted within an arm (A: {len(ha)}, B: {len(hb)} versions)")
    declared = ctx.manifest.harness
    if (declared.get(ctx.A) and declared[ctx.A] not in ha) or (declared.get(ctx.B) and declared[ctx.B] not in hb):
        issues.append("records do not match the harness hashes in the manifest")
    if not ctx.manifest.changed_artifacts:
        issues.append("no changed artifacts are declared, so the difference is not identified")
    if ctx.manifest.confounds:
        issues.append("other differences: " + "; ".join(ctx.manifest.confounds))
    if issues:
        return CheckResult(
            id="isolated_update",
            title="Arms differ only in the update",
            status=Status.WARN,
            criterion=crit,
            metrics=metrics,
            reason="Update not isolated: " + "; ".join(issues) + ".",
        )
    n = len(ctx.manifest.changed_artifacts)
    return CheckResult(
        id="isolated_update",
        title="Arms differ only in the update",
        status=Status.PASS,
        criterion=crit,
        metrics=metrics,
        reason=f"One harness per arm; they differ in {n} declared artifact(s) and nothing else.",
    )


def _arm_means(ctx: Context, field: str) -> tuple[float | None, float | None, int]:
    p = data.paired(ctx.records, PRIMARY_SPLIT, ctx.A, ctx.B, metric=data.usage_metric(field))
    if p.m == 0:
        return None, None, 0
    return float(p.a.mean()), float(p.b.mean()), p.m


def _envelope(ctx: Context) -> CheckResult:
    crit = (
        "controlled = held equal, a performance–cost frontier, or a change that cannot explain the gain; "
        "declared but unmatched → O"
    )
    table: dict[str, dict[str, float | None]] = {}
    for f in ENVELOPE_FIELDS + REPORTED_FIELDS:
        a, b, m = _arm_means(ctx, f)
        ratio = None if a is None or b is None or a == 0 else b / a
        table[f] = {
            "A": None if a is None else round(a, 4),
            "B": None if b is None else round(b, 4),
            "ratio_B_over_A": None if ratio is None else round(ratio, 4),
            "m_tasks": m,
        }
    present = [f for f in ENVELOPE_FIELDS if table[f]["ratio_B_over_A"] is not None]
    metrics: dict[str, object] = {
        "per_task_means": table,
        "threshold_ratio": ctx.th.envelope_ratio,
        "context_added_input_tokens": None,
    }
    ia, ib = table["input_tokens"]["A"], table["input_tokens"]["B"]
    if ia is not None and ib is not None:
        metrics["context_added_input_tokens"] = round(ib - ia, 1)
    sa, sb = ctx.primary.a, ctx.primary.b
    ca, cb = table["cost_usd"]["A"], table["cost_usd"]["B"]
    if ca and cb:
        metrics["score_per_usd"] = {"A": round(float(sa.mean()) / ca, 3), "B": round(float(sb.mean()) / cb, 3)}
        if cb > ca:
            metrics["delta_score_per_extra_usd"] = round(float(sb.mean() - sa.mean()) / (cb - ca), 3)
    if not present:
        return CheckResult(
            id="envelope",
            title="Inference resources controlled",
            status=Status.WARN,
            criterion=crit,
            metrics=metrics,
            reason="Inference resources (output tokens, turns, tool calls) are not reported.",
        )
    over = [f for f in present if table[f]["ratio_B_over_A"] > ctx.th.envelope_ratio]  # type: ignore[operator]
    if over:
        parts = ", ".join(f"{f} ×{table[f]['ratio_B_over_A']:.2f}" for f in over)
        return CheckResult(
            id="envelope",
            title="Inference resources controlled",
            status=Status.WARN,
            criterion=crit,
            metrics=metrics,
            reason=f"B used more inference compute than A ({parts}; threshold ×{ctx.th.envelope_ratio:.2f}).",
        )
    parts = ", ".join(f"{f} ×{table[f]['ratio_B_over_A']:.2f}" for f in present)
    return CheckResult(
        id="envelope",
        title="Inference resources controlled",
        status=Status.PASS,
        criterion=crit,
        metrics=metrics,
        reason=f"B's inference envelope is within ×{ctx.th.envelope_ratio:.2f} of A ({parts}).",
    )


def _margin(ctx: Context) -> CheckResult:
    est = ctx.primary_estimate
    assert est is not None
    k = ctx.manifest.selection.n_compared
    crit = (
        "a test distinguishes the margin from zero at the 5% level, or an interval for the difference excludes zero; "
        "a selected maximum must clear the noise of a maximum"
    )
    metrics = {**est.as_dict(), "alpha": ctx.th.alpha, "n_compared": k, "method": "paired bootstrap over tasks + sign-flip test"}
    ci = fmt_ci(est.ci_low, est.ci_high)
    lvl = f"{est.level:.1%}".rstrip("0").rstrip(".")
    alpha_k = ctx.th.alpha / k
    if est.ci_low > 0 and est.p_value < alpha_k:
        return CheckResult(
            id="margin",
            title="Margin beyond measured variation",
            status=Status.PASS,
            criterion=crit,
            metrics=metrics,
            reason=f"Δscore {est.mean:+.3f}, {lvl} CI {ci}, p = {est.p_value:.4f} over {est.m} tasks.",
        )
    if est.ci_high < 0 and est.p_value < alpha_k:
        why = f"B is worse than A: Δscore {est.mean:+.3f}, {lvl} CI {ci}, p = {est.p_value:.4f}."
    else:
        why = f"Margin not distinguished from zero: Δscore {est.mean:+.3f}, {lvl} CI {ci}, p = {est.p_value:.4f}." + p_note(
            est, alpha_k
        )
    return CheckResult(
        id="margin", title="Margin beyond measured variation", status=Status.WARN, criterion=crit, metrics=metrics, reason=why
    )


def _split_half(runs: list[RunRecord]) -> data.Paired:
    by_task: dict[str, list[RunRecord]] = defaultdict(list)
    for r in runs:
        by_task[r.task_id].append(r)
    tids, a, b = [], [], []
    for t in sorted(by_task):
        rs = sorted(by_task[t], key=lambda r: r.seed)
        if len(rs) < 2:
            continue
        tids.append(t)
        a.append(float(np.mean([r.score for r in rs[0::2]])))
        b.append(float(np.mean([r.score for r in rs[1::2]])))
    return data.Paired(tids, np.array(a), np.array(b), [])


def _aa(ctx: Context) -> CheckResult:
    crit = "execution variation measured through repeated executions of frozen states (A/A)"
    a_runs = data.select(ctx.records, arm=ctx.A, split=PRIMARY_SPLIT)
    source = None
    if ctx.A2 and data.select(ctx.records, arm=ctx.A2, split=PRIMARY_SPLIT):
        pa = data.paired(ctx.records, PRIMARY_SPLIT, ctx.A, ctx.A2)
        a2_hash = _values(data.select(ctx.records, arm=ctx.A2, split=PRIMARY_SPLIT), "harness_hash")
        if not (a2_hash <= _values(a_runs, "harness_hash")):
            return CheckResult(
                id="aa_noise",
                title="A/A noise band",
                status=Status.WARN,
                rubric=False,
                required=False,
                criterion=crit,
                reason=f"Arm {ctx.A2} does not run A's harness, so it is not an A/A replicate.",
            )
        source = f"independent replicate arm {ctx.A2}"
    else:
        pa = _split_half(a_runs)
        source = "split-half of A's replicates (even vs odd seeds; conservative)"
    if pa.m < 2:
        return CheckResult(
            id="aa_noise",
            title="A/A noise band",
            status=Status.NOTE,
            rubric=False,
            required=False,
            criterion=crit,
            reason="No A/A data (no replicate arm, <2 replicates per task); variation rests on the paired task test.",
        )
    aa = ctx.estimate(pa.diffs, level=1.0 - ctx.th.alpha, salt=7)
    est = ctx.primary_estimate
    assert est is not None
    metrics = {
        "source": source,
        "aa": aa.as_dict(),
        "delta_ab": round(est.mean, 6),
        "aa_band": [round(aa.ci_low, 6), round(aa.ci_high, 6)],
    }
    if aa.excludes_zero() and aa.p_value < ctx.th.alpha:
        return CheckResult(
            id="aa_noise",
            title="A/A noise band",
            status=Status.WARN,
            rubric=False,
            criterion=crit,
            metrics=metrics,
            reason=f"A/A difference is itself significant ({aa.mean:+.3f}, CI {fmt_ci(aa.ci_low, aa.ci_high)}): runs or scoring are not stable.",
        )
    if est.mean <= aa.ci_high:
        return CheckResult(
            id="aa_noise",
            title="A/A noise band",
            status=Status.WARN,
            rubric=False,
            criterion=crit,
            metrics=metrics,
            reason=f"Δ_AB {est.mean:+.3f} lies inside the A/A noise band {fmt_ci(aa.ci_low, aa.ci_high)}.",
        )
    return CheckResult(
        id="aa_noise",
        title="A/A noise band",
        status=Status.PASS,
        rubric=False,
        criterion=crit,
        metrics=metrics,
        reason=f"Δ_AB {est.mean:+.3f} exceeds the A/A noise band {fmt_ci(aa.ci_low, aa.ci_high)} ({source}).",
    )

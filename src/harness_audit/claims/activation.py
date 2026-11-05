"""Activation: did the accepted change operate in the runs that produced the score?

Rubric (Harness-RSI ``data/claim-audit-rubric.md``, CC BY 4.0): grades are
**available** (the artifact entered the run), **reached** (its path executed) and
**carried** (its effect reached its intended downstream use). ``C`` requires
reached- or carried-grade evidence from the scored runs behind the headline; the
proposition is existence, with coverage recorded. Availability alone is ``O``;
no evidence it operated is ``N``. A counterfactual score change belongs to
Attribution, so the conditional-effect figures here are reported as notes.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .. import adherence, data
from ..schema import ChangedArtifact, CheckResult, ClaimResult, Event, RubricItem, RunRecord, Status
from .context import PRIMARY_SPLIT, Context, make_claim

_CRIT_C = "reached or carried evidence from the scored runs → C (existence; coverage recorded)"


def _is(ev: Event, art: ChangedArtifact) -> bool:
    if ev.artifact is None:
        return False
    if ev.artifact == art.path or ev.artifact.endswith("/" + art.path) or art.path.endswith("/" + ev.artifact):
        return True
    return bool(art.hash_b and ev.hash == art.hash_b)


def anchor(run: RunRecord, art: ChangedArtifact) -> float | None:
    """Time from which the artifact's guidance can be followed: first reach (or load, for prompts)."""
    evs = run.events or []
    reached = [e.t for e in evs if e.type == "artifact_reached" and _is(e, art)]
    loaded = [e.t for e in evs if e.type == "artifact_loaded" and _is(e, art)]
    pool = reached if art.carried_requires_reached else (reached or loaded)
    return min(pool) if pool else None


def grades(run: RunRecord, art: ChangedArtifact) -> dict[str, bool]:
    """Activation grades of one run for one artifact.

    Carried is an explicit ``artifact_carried`` event, or adherence to the artifact's rubric (every
    required, decidable item met) from the moment it was reached; declared signatures act as a
    one-item rubric.
    """
    evs = run.events or []
    loaded = any(e.type == "artifact_loaded" and _is(e, art) for e in evs)
    reached = any(e.type == "artifact_reached" and _is(e, art) for e in evs)
    carried = any(e.type == "artifact_carried" and _is(e, art) for e in evs)
    if not carried and art.effective_rubric():
        start = anchor(run, art)
        carried = start is not None and bool(adherence.evaluate(evs, art, start).carried)
    return {"available": loaded or reached or carried, "reached": reached, "carried": carried}


def _any_sig(run: RunRecord, art: ChangedArtifact) -> bool:
    item = RubricItem(id="signature", any_of=art.signatures)
    return any(adherence._matches(e, item) for e in (run.events or []))


def _rate(k: int, n: int) -> float | None:
    """A rate with a zero denominator is undefined, not zero (Eq. 11): report the counts instead."""
    return round(k / n, 4) if n else None


def _applicable(runs: Iterable[RunRecord], art: ChangedArtifact) -> list[RunRecord]:
    return [r for r in runs if art.applies(r.task_id, r.task_group)]


def artifact_rates(traced: list[RunRecord], art: ChangedArtifact) -> dict[str, object]:
    """Load, reach and carry rates of one update over the scored runs where it applies (paper Eq. 11).

    ``R(u)``: scored runs satisfying the declared applicability rule; ``load = |L|/|R|``,
    ``reach = |B|/|R|``; ``carry = |C|/|L|`` for interpreted guidance and ``|C|/|B|`` for
    executable paths.
    """
    runs = _applicable(traced, art)
    g = [grades(r, art) for r in runs]
    n_r = len(runs)
    k_l = sum(x["available"] for x in g)
    k_b = sum(x["reached"] for x in g)
    if art.executable:
        base, k_base = "reached", k_b
        k_c = sum(x["carried"] and x["reached"] for x in g)
    else:
        base, k_base = "loaded", k_l
        k_c = sum(x["carried"] and x["available"] for x in g)
    return {
        "applicable": f"{n_r}/{len(traced)}",
        "applies_to": art.applies_to,
        "available": f"{k_l}/{n_r}",
        "reached": f"{k_b}/{n_r}",
        "carried": f"{k_c}/{k_base} {base}",
        "load_rate": _rate(k_l, n_r),
        "reach_rate": _rate(k_b, n_r),
        "carry_rate": _rate(k_c, k_base),
        "carry_denominator": base,
    }


def audit_activation(ctx: Context) -> ClaimResult:
    arts = ctx.changed_artifacts
    b_runs = data.select(ctx.records, arm=ctx.B, split=PRIMARY_SPLIT)
    a_runs = data.select(ctx.records, arm=ctx.A, split=PRIMARY_SPLIT)
    checks: list[CheckResult] = []

    if not arts:
        checks.append(
            CheckResult(
                id="reached_or_carried",
                title="Reached or carried in scored runs",
                status=Status.SKIP,
                reason="No changed artifacts are declared or visible in the traces.",
                criterion=_CRIT_C,
            )
        )
        return make_claim("Activation", checks)
    traced = [r for r in b_runs if r.events is not None]
    if not b_runs or not traced:
        checks.append(
            CheckResult(
                id="reached_or_carried",
                title="Reached or carried in scored runs",
                status=Status.SKIP,
                reason="No traces were collected for the scored B runs.",
                criterion=_CRIT_C,
                metrics={"n_scored_runs": len(b_runs), "n_traced": len(traced)},
            )
        )
        return make_claim("Activation", checks)

    n = len(traced)
    a_traced = [r for r in a_runs if r.events]
    per_art: dict[str, dict[str, object]] = {}
    act_by_run: dict[str, bool] = {r.run_id: False for r in traced}
    operated: list[str] = []
    for art in arts:
        hit = False
        for r in traced:
            x = grades(r, art)
            if x["reached"] or x["carried"]:
                act_by_run[r.run_id] = hit = True
        if hit:
            operated.append(art.path)
        other = [r for r in ctx.records if r.arm == ctx.B and r.split != PRIMARY_SPLIT and r.events]
        app_b, app_a = _applicable(traced, art), _applicable(a_traced, art)
        per_art[art.path] = {
            "kind": art.kind,
            **artifact_rates(traced, art),
            "signatures": [s.model_dump() for s in art.signatures],
            "signature_rate_B": _rate(sum(_any_sig(r, art) for r in app_b), len(app_b)) if art.signatures else None,
            "signature_rate_A": _rate(sum(_any_sig(r, art) for r in app_a), len(app_a)) if art.signatures else None,
            "reached_outside_primary": sum(grades(r, art)["reached"] for r in other),
        }

    k_any_av = sum(1 for r in traced if any(grades(r, a)["available"] for a in arts))
    checks.append(
        CheckResult(
            id="available",
            title="Available (artifact entered the run)",
            status=Status.PASS if k_any_av else Status.WARN,
            required=False,
            reason=f"A changed artifact was loaded in {k_any_av}/{n} scored B runs.",
            criterion="availability is the weakest grade; load rates alone are O",
            metrics={
                "per_artifact": {
                    p: {k: v for k, v in d.items() if k in ("applicable", "available", "load_rate")} for p, d in per_art.items()
                }
            },
        )
    )

    k_act = sum(act_by_run.values())
    coverage = f"{k_act}/{n} scored runs"
    metrics = {"coverage": coverage, "n_traced": n, "n_scored_runs": len(b_runs), "per_artifact": per_art}
    if k_act > 0:
        st, why = Status.PASS, f"Reached or carried in {coverage} ({', '.join(operated)})."
    elif k_any_av > 0:
        st, why = Status.WARN, f"Changed artifacts were loaded in {k_any_av}/{n} scored runs but never reached or carried."
    else:
        st, why = Status.FAIL, "Traces are present but no changed artifact was loaded, reached or carried in any scored run."
    checks.append(
        CheckResult(
            id="reached_or_carried",
            title="Reached or carried in scored runs",
            status=st,
            reason=why,
            criterion=_CRIT_C,
            metrics=metrics,
        )
    )

    checks.append(adherence_check(traced, a_traced, arts))
    checks.append(_baseline_clean(ctx, a_runs, arts))
    applicable = [r for r in traced if any(a.applies(r.task_id, r.task_group) for a in arts)]
    k_app = sum(act_by_run[r.run_id] for r in applicable)
    checks.extend(_conditional(ctx, traced, a_runs, act_by_run, k_app, len(applicable)))
    return make_claim("Activation", checks)


def _baseline_clean(ctx: Context, a_runs: Iterable[RunRecord], arts: list[ChangedArtifact]) -> CheckResult:
    leaked = 0
    total = 0
    for r in a_runs:
        if r.events is None:
            continue
        total += 1
        for art in arts:
            if art.change == "removed":
                continue
            evs = [e for e in r.events if e.type in ("artifact_loaded", "artifact_reached") and _is(e, art)]
            if any(art.change == "added" or (art.hash_b and e.hash == art.hash_b) for e in evs):
                leaked += 1
                break
    if total == 0:
        return CheckResult(
            id="baseline_clean",
            title="Baseline does not contain the update",
            status=Status.SKIP,
            required=False,
            reason="No traces for baseline runs.",
            rubric=False,
        )
    if leaked:
        return CheckResult(
            id="baseline_clean",
            title="Baseline does not contain the update",
            status=Status.WARN,
            rubric=False,
            reason=f"{leaked}/{total} baseline runs loaded B's version of a changed artifact.",
            metrics={"contaminated_runs": leaked, "n_traced_A": total},
        )
    return CheckResult(
        id="baseline_clean",
        title="Baseline does not contain the update",
        status=Status.PASS,
        rubric=False,
        reason=f"No baseline run loaded B's changed artifacts ({total} traced).",
        metrics={"contaminated_runs": 0, "n_traced_A": total},
    )


def _conditional(
    ctx: Context, traced: list[RunRecord], a_runs: list[RunRecord], act_by_run: dict[str, bool], k_act: int, n: int
) -> list[CheckResult]:
    a_mean = data.task_means(a_runs)
    on = [r.score - a_mean[r.task_id] for r in traced if act_by_run[r.run_id] and r.task_id in a_mean]
    off = [r.score - a_mean[r.task_id] for r in traced if not act_by_run[r.run_id] and r.task_id in a_mean]
    d_on = float(np.mean(on)) if on else None
    d_off = float(np.mean(off)) if off else None
    metrics = {
        "delta_when_activated": None if d_on is None else round(d_on, 4),
        "n_activated": len(on),
        "delta_when_not_activated": None if d_off is None else round(d_off, 4),
        "n_not_activated": len(off),
    }
    out = [
        CheckResult(
            id="conditional_effect",
            title="Conditional effect",
            status=Status.PASS if (on or off) else Status.SKIP,
            rubric=False,
            required=False,
            metrics=metrics,
            reason=(
                "Mean B − A score difference: "
                f"{'n/a' if d_on is None else f'{d_on:+.3f}'} on activated runs (n={len(on)}), "
                f"{'n/a' if d_off is None else f'{d_off:+.3f}'} on non-activated runs (n={len(off)})."
            ),
            criterion="reported for reading; a counterfactual score change belongs to Attribution",
        )
    ]

    rate = _rate(k_act, n)
    gain = ctx.gain_is_significant()
    flag_low = gain and rate is not None and rate < ctx.th.low_reach
    flag_off = gain and d_off is not None and d_on is not None and len(off) >= 2 and d_off >= d_on
    if flag_low or flag_off:
        why = (
            f"Improvement not via the update path: the update was activated in {k_act}/{n} applicable scored runs "
            f"(threshold {ctx.th.low_reach:.0%}) while the score gain is significant."
            if flag_low
            else "Improvement not via the update path: runs where the update was not activated gain as much as runs where it was."
        )
        out.append(
            CheckResult(
                id="gain_without_activation",
                title="Gain without activation",
                status=Status.NOTE,
                rubric=False,
                required=False,
                reason=why,
                metrics={"activation_rate": rate, "activated": f"{k_act}/{n}", "low_reach_threshold": ctx.th.low_reach},
            )
        )
    else:
        out.append(
            CheckResult(
                id="gain_without_activation",
                title="Gain without activation",
                status=Status.PASS,
                rubric=False,
                required=False,
                reason="No sign that the gain bypasses the update path.",
                metrics={"activation_rate": rate, "activated": f"{k_act}/{n}", "low_reach_threshold": ctx.th.low_reach},
            )
        )
    return out


def adherence_rates(b_runs: list[RunRecord], a_runs: list[RunRecord], art: ChangedArtifact) -> dict[str, object]:
    """Per-item adherence in B (from the anchor, over the carry denominator) and in A (whole run)."""
    rubric = art.effective_rubric()
    base_b = []
    for r in _applicable(b_runs, art):
        g = grades(r, art)
        if g["reached"] if art.executable else g["available"]:
            base_b.append(r)
    res_b = [adherence.evaluate(r.events or [], art, anchor(r, art)) for r in base_b]
    res_a = [adherence.evaluate(r.events or [], art, 0.0) for r in _applicable(a_runs, art)]

    def item_rate(results: list[adherence.RubricResult], iid: str) -> dict[str, object]:
        outs = [x.items[iid] for x in results]
        met, unmet = outs.count("met"), outs.count("unmet")
        return {
            "met": f"{met}/{met + unmet}",
            "rate": _rate(met, met + unmet),
            "n/a": outs.count("n/a"),
            "unjudged": outs.count("unjudged"),
        }

    def carried(results: list[adherence.RubricResult]) -> tuple[int, int]:
        c = [x.carried for x in results if x.carried is not None]
        return sum(c), len(c)

    def mean_score(results: list[adherence.RubricResult]) -> float | None:
        sc = [x.score for x in results if x.score is not None]
        return round(sum(sc) / len(sc), 4) if sc else None

    kb, nb = carried(res_b)
    ka, na = carried(res_a)
    return {
        "items": {
            i.id: {
                "text": i.text,
                "required": i.required,
                "check": i.check,
                "B": item_rate(res_b, i.id),
                "A": item_rate(res_a, i.id),
            }
            for i in rubric
        },
        "followed_B": f"{kb}/{nb}",
        "followed_rate_B": _rate(kb, nb),
        "followed_A": f"{ka}/{na}",
        "followed_rate_A": _rate(ka, na),
        "mean_adherence_B": mean_score(res_b),
        "mean_adherence_A": mean_score(res_a),
        "b_runs_scored": "reached" if art.executable else "loaded",
        "unjudged_required_items": sorted({i for x in res_b + res_a for i in x.unjudged_required}),
        "from_signatures": not art.rubric,
    }


def adherence_check(traced: list[RunRecord], a_traced: list[RunRecord], arts: list[ChangedArtifact]) -> CheckResult:
    """Carried for guidance, read as adherence to the artifact's rubric; compared with the baseline."""
    title = "Adherence to the update's instructions (carried)"
    crit = "for an instruction or skill, carried evidence records adherence in the trajectory (report §6.2.2)"
    with_rubric = [a for a in arts if a.effective_rubric()]
    if not with_rubric:
        return CheckResult(
            id="adherence",
            title=title,
            status=Status.SKIP,
            required=False,
            rubric=False,
            criterion=crit,
            reason="No rubric or trace signatures are declared for the changed artifacts.",
        )
    per = {a.path: adherence_rates(traced, a_traced, a) for a in with_rubric}
    weak: list[str] = []
    undecided: list[str] = []
    parts: list[str] = []
    for path, d in per.items():
        rb, ra = d["followed_rate_B"], d["followed_rate_A"]
        if rb is None:
            undecided.append(path)
            continue
        parts.append(f"{path}: followed in {d['followed_B']} B runs that {d['b_runs_scored']} it vs {d['followed_A']} A runs")
        if ra is not None and rb <= ra:
            weak.append(path)
    unj = sorted({i for d in per.values() for i in d["unjudged_required_items"]})
    tail = f" Required judge items not yet judged: {', '.join(unj)}." if unj else ""
    metrics = {"per_artifact": per}
    if weak:
        why = (
            "The procedure is followed no more often with the update than without it ("
            + "; ".join(parts)
            + "): adherence does not show the update operating."
            + tail
        )
        st = Status.NOTE
    elif not parts:
        why = "No rubric item could be decided in the scored runs." + tail
        st = Status.SKIP
    else:
        why = "; ".join(parts) + "." + tail
        st = Status.PASS
    if undecided and parts:
        why += f" Undecided: {', '.join(undecided)}."
    return CheckResult(
        id="adherence", title=title, status=st, reason=why, required=False, rubric=False, criterion=crit, metrics=metrics
    )

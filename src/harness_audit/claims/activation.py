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

import re
from collections.abc import Iterable

import numpy as np

from .. import data
from ..schema import ChangedArtifact, CheckResult, ClaimResult, Event, RunRecord, Status
from .context import PRIMARY_SPLIT, Context, make_claim

_CRIT_C = "reached or carried evidence from the scored runs → C (existence; coverage recorded)"


def _is(ev: Event, art: ChangedArtifact) -> bool:
    if ev.artifact is None:
        return False
    if ev.artifact == art.path or ev.artifact.endswith("/" + art.path) or art.path.endswith("/" + ev.artifact):
        return True
    return bool(art.hash_b and ev.hash == art.hash_b)


def _sig_match(ev: Event, art: ChangedArtifact) -> bool:
    if ev.type != "tool_call" or not art.signatures:
        return False
    for s in art.signatures:
        if s.tool is not None and ev.name != s.tool:
            continue
        if re.search(s.pattern, ev.summary or ""):
            return True
    return False


def grades(run: RunRecord, art: ChangedArtifact) -> dict[str, bool]:
    """Activation grades of one run for one artifact."""
    evs = sorted(run.events or [], key=lambda e: e.t)
    loaded = [e.t for e in evs if e.type == "artifact_loaded" and _is(e, art)]
    reached = [e.t for e in evs if e.type == "artifact_reached" and _is(e, art)]
    carried = any(e.type == "artifact_carried" and _is(e, art) for e in evs)
    anchor_list = reached if art.carried_requires_reached else (reached or loaded)
    if not carried and anchor_list:
        anchor = min(anchor_list)
        carried = any(_sig_match(e, art) and e.t >= anchor for e in evs)
    return {"available": bool(loaded or reached or carried), "reached": bool(reached), "carried": carried}


def _any_sig(run: RunRecord, art: ChangedArtifact) -> bool:
    return any(_sig_match(e, art) for e in (run.events or []))


def _rate(k: int, n: int) -> float:
    return round(k / n, 4) if n else 0.0


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
    per_art: dict[str, dict[str, object]] = {}
    act_by_run: dict[str, bool] = {r.run_id: False for r in traced}
    for art in arts:
        g = [grades(r, art) for r in traced]
        k_av = sum(x["available"] for x in g)
        k_re = sum(x["reached"] for x in g)
        k_ca = sum(x["carried"] for x in g)
        for r, x in zip(traced, g, strict=True):
            act_by_run[r.run_id] |= x["reached"] or x["carried"]
        other = [r for r in ctx.records if r.arm == ctx.B and r.split != PRIMARY_SPLIT and r.events]
        per_art[art.path] = {
            "kind": art.kind,
            "available": f"{k_av}/{n}",
            "reached": f"{k_re}/{n}",
            "carried": f"{k_ca}/{n}",
            "available_rate": _rate(k_av, n),
            "reached_rate": _rate(k_re, n),
            "carried_rate": _rate(k_ca, n),
            "signatures": [s.model_dump() for s in art.signatures],
            "signature_rate_B": _rate(sum(_any_sig(r, art) for r in traced), n) if art.signatures else None,
            "signature_rate_A": (
                _rate(sum(_any_sig(r, art) for r in a_runs if r.events), len([r for r in a_runs if r.events]))
                if art.signatures
                else None
            ),
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
                    p: {k: v for k, v in d.items() if k in ("available", "available_rate")} for p, d in per_art.items()
                }
            },
        )
    )

    k_act = sum(act_by_run.values())
    coverage = f"{k_act}/{n} scored runs"
    metrics = {"coverage": coverage, "n_traced": n, "n_scored_runs": len(b_runs), "per_artifact": per_art}
    if k_act > 0:
        names = [p for p, d in per_art.items() if d["reached_rate"] or d["carried_rate"]]
        st, why = Status.PASS, f"Reached or carried in {coverage} ({', '.join(names)})."
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

    checks.append(_baseline_clean(ctx, a_runs, arts))
    checks.extend(_conditional(ctx, traced, a_runs, act_by_run, k_act, n))
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

    rate = k_act / n if n else 0.0
    gain = ctx.gain_is_significant()
    flag_low = gain and rate < ctx.th.low_reach
    flag_off = gain and d_off is not None and d_on is not None and len(off) >= 2 and d_off >= d_on
    if flag_low or flag_off:
        why = (
            f"Improvement not via the update path: the update was activated in {rate:.0%} of scored runs "
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
                metrics={"activation_rate": round(rate, 4), "low_reach_threshold": ctx.th.low_reach},
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
                metrics={"activation_rate": round(rate, 4), "low_reach_threshold": ctx.th.low_reach},
            )
        )
    return out

"""The most important tests: each planted defect is caught by its own check, and only there."""

from __future__ import annotations

import pytest

from harness_audit.audit import run_audit
from harness_audit.schema import AuditResult, Status, Verdict
from harness_audit.synth import SCENARIOS, generate

CLAIMS = ("Selection", "Activation", "Attribution", "Improvement")

# scenario -> (expected verdicts by claim, {check_id: expected status})
EXPECTED: dict[str, tuple[dict[str, Verdict], dict[str, Status]]] = {
    "clean": ({}, {}),
    "split_leak": (
        {"Selection": Verdict.PARTIAL, "Improvement": Verdict.PARTIAL},
        {"split_disjoint_proposal": Status.WARN, "task_distance": Status.WARN},
    ),
    "never_activated": (
        {"Activation": Verdict.PARTIAL},
        {"reached_or_carried": Status.WARN, "gain_without_activation": Status.NOTE},
    ),
    "double_tokens": ({"Attribution": Verdict.PARTIAL}, {"envelope": Status.WARN}),
    "evaluator_changed": ({"Attribution": Verdict.FAIL}, {"same_evaluator": Status.FAIL}),
    "same_group_only": ({"Improvement": Verdict.PARTIAL}, {"heldout_gain": Status.WARN}),
    "retention_regression": ({"Improvement": Verdict.FAIL}, {"retention": Status.FAIL}),
    "no_traces": ({"Activation": Verdict.UNKNOWN}, {"reached_or_carried": Status.SKIP}),
    "oracle_touched": ({"Selection": Verdict.PARTIAL}, {"oracle_integrity": Status.WARN}),
}


@pytest.fixture(scope="module")
def results() -> dict[str, AuditResult]:
    out = {}
    for name in SCENARIOS:
        records, manifest = generate(name)
        out[name] = run_audit(records, manifest)
    return out


def test_every_scenario_has_expectations() -> None:
    assert set(EXPECTED) == set(SCENARIOS)


@pytest.mark.parametrize("scenario", sorted(EXPECTED))
def test_planted_defect_detected_by_its_check(results: dict[str, AuditResult], scenario: str) -> None:
    res = results[scenario]
    verdicts, statuses = EXPECTED[scenario]
    for claim in CLAIMS:
        want = verdicts.get(claim, Verdict.PASS)
        got = res.claim(claim).verdict
        assert got == want, f"{scenario}: {claim} is {got.value}, expected {want.value}: {res.claim(claim).summary}"
    for check_id, want_status in statuses.items():
        assert res.check(check_id).status == want_status, (scenario, check_id, res.check(check_id).reason)


@pytest.mark.parametrize("scenario", sorted(EXPECTED))
def test_only_the_planted_check_misfires(results: dict[str, AuditResult], scenario: str) -> None:
    """Checks that are not expected to fire must not be warn/fail in a single-defect scenario."""
    _, statuses = EXPECTED[scenario]
    for claim in results[scenario].claims:
        for ch in claim.checks:
            if ch.id in statuses:
                continue
            assert ch.status in (Status.PASS, Status.SKIP), (scenario, ch.id, ch.status, ch.reason)


def test_clean_scenario_passes_all_four(results: dict[str, AuditResult]) -> None:
    res = results["clean"]
    assert [c.verdict for c in res.claims] == [Verdict.PASS] * 4
    assert [c.rubric_grade for c in res.claims] == ["C"] * 4


def test_rubric_grades_where_harness_audit_is_stricter(results: dict[str, AuditResult]) -> None:
    # The rubric grades Improvement on task distance alone, and an evaluator change as O.
    assert results["same_group_only"].claim("Improvement").rubric_grade == "C"
    assert results["retention_regression"].claim("Improvement").rubric_grade == "C"
    assert results["evaluator_changed"].claim("Attribution").rubric_grade == "O"
    assert results["split_leak"].claim("Selection").rubric_grade == "O"


def test_activation_coverage_is_recorded(results: dict[str, AuditResult]) -> None:
    ch = results["clean"].check("reached_or_carried")
    assert ch.metrics["coverage"].endswith("/90 scored runs")
    per = ch.metrics["per_artifact"]["skills/python-bugfix/SKILL.md"]
    assert per["applicable"] == "90/90" and per["reach_rate"] > 0.8
    assert per["carry_denominator"] == "loaded" and per["carried"].endswith("/90 loaded")
    assert per["signature_rate_A"] is not None and per["signature_rate_A"] < per["signature_rate_B"]


def test_oracle_touch_does_not_change_the_rubric_grade(results: dict[str, AuditResult]) -> None:
    # Oracle integrity is a harness-audit control from the report's Table 5, outside the survey rubric.
    sel = results["oracle_touched"].claim("Selection")
    assert sel.rubric_grade == "C"
    ch = results["oracle_touched"].check("oracle_integrity")
    assert set(ch.metrics["runs_touching_scorer_by_arm"]) == {"B"}
    assert results["clean"].check("oracle_integrity").status == Status.PASS


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_planted_results_stable_across_generator_seeds(seed: int) -> None:
    for scenario in ("clean", "double_tokens", "same_group_only", "never_activated"):
        records, manifest = generate(scenario, seed=seed)
        res = run_audit(records, manifest)
        verdicts, _ = EXPECTED[scenario]
        for claim in CLAIMS:
            assert res.claim(claim).verdict == verdicts.get(claim, Verdict.PASS), (seed, scenario, claim)

"""Adherence rubric: carried for skills and prompts, read from the trajectory (report §6.2.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from harness_audit.adherence import apply_judge, evaluate
from harness_audit.audit import run_audit
from harness_audit.harness import diff_harnesses, load_harness
from harness_audit.report import to_html, to_markdown
from harness_audit.schema import ChangedArtifact, Event, RubricItem, RunRecord, Signature, Status
from harness_audit.synth import generate

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SKILL = "skills/python-bugfix/SKILL.md"


def example_skill() -> ChangedArtifact:
    arts, _ = diff_harnesses(load_harness(EXAMPLES / "harness_a"), load_harness(EXAMPLES / "harness_b"))
    return arts[0]


def call(t: float, name: str, summary: str, ok: bool | None = True) -> Event:
    return Event(t=t, type="tool_call", name=name, summary=summary, ok=ok)


def test_example_harness_declares_a_rubric() -> None:
    art = example_skill()
    assert art.path == SKILL and art.signatures == []
    ids = [i.id for i in art.rubric]
    assert ids[:5] == ["read_function", "list_edge_cases", "fix", "public_tests", "edge_case_asserts"]
    assert next(i for i in art.rubric if i.id == "list_edge_cases").check == "judge"


def test_full_procedure_is_carried() -> None:
    evs = [
        call(1, "Read", "initials.py"),
        call(2, "Edit", "initials.py"),
        call(3, "Bash", "python test_public.py"),
        call(4, "Bash", 'python -c \'assert initials("") == ""\''),
    ]
    res = evaluate(evs, example_skill(), start=0.5)
    assert res.items["fix"] == res.items["public_tests"] == res.items["edge_case_asserts"] == "met"
    assert res.items["fix_after_failure"] == res.items["rerun_after_fix"] == "n/a"  # nothing failed
    assert res.items["list_edge_cases"] == "unjudged"  # needs a judge; not required
    assert res.carried is True and res.score == 1.0


def test_order_matters() -> None:
    evs = [
        call(1, "Bash", "python -c 'assert f(1) == 1'"),  # asserts before the public tests
        call(2, "Edit", "f.py"),
        call(3, "Bash", "python test_public.py"),
    ]
    res = evaluate(evs, example_skill(), start=0.0)
    assert res.items["public_tests"] == "met" and res.items["edge_case_asserts"] == "unmet"
    assert res.carried is False


def test_calls_before_the_update_was_reached_do_not_count() -> None:
    evs = [call(1, "Edit", "f.py"), call(2, "Bash", "python test_public.py"), call(3, "Bash", "python -c 'assert 1'")]
    assert evaluate(evs, example_skill(), start=2.5).carried is False
    assert evaluate(evs, example_skill(), start=None).carried is False
    assert evaluate(evs, example_skill(), start=0.0).carried is True


def test_failure_branch_requires_a_fix_and_a_passing_rerun() -> None:
    base = [call(1, "Edit", "f.py"), call(2, "Bash", "python test_public.py", ok=False)]
    gave_up = evaluate(base + [call(3, "Bash", "python -c 'assert 1'")], example_skill())
    assert gave_up.items["fix_after_failure"] == "unmet" and gave_up.carried is False
    fixed = evaluate(
        base
        + [
            call(3, "Edit", "f.py"),
            call(4, "Bash", "python test_public.py", ok=True),
            call(5, "Bash", "python -c 'assert 1'", ok=True),
        ],
        example_skill(),
    )
    assert fixed.items["fix_after_failure"] == fixed.items["rerun_after_fix"] == "met" and fixed.carried is True
    rerun_failed = evaluate(base + [call(3, "Edit", "f.py"), call(4, "Bash", "python test_public.py", ok=False)], example_skill())
    assert rerun_failed.items["rerun_after_fix"] == "unmet"


def test_unknown_outcomes_do_not_trigger_or_satisfy_ok_items() -> None:
    evs = [call(1, "Edit", "f.py", ok=None), call(2, "Bash", "python test_public.py", ok=None)]
    res = evaluate(evs, example_skill())
    assert res.items["fix_after_failure"] == "n/a"  # no recorded failure, so the branch does not apply


def test_judge_items_come_only_from_judgments() -> None:
    art = ChangedArtifact(
        path="p.md",
        kind="system_prompt",
        rubric=[RubricItem(id="plan", check="judge"), RubricItem(id="tests", tool="Bash", pattern="pytest")],
    )
    evs = [call(1, "Bash", "pytest -q")]
    assert evaluate(evs, art).items["plan"] == "unjudged"
    assert evaluate(evs, art).carried is True  # decided on the checkable required items
    judged = evs + [Event(t=9, type="rubric_judgment", artifact="p.md", name="plan", ok=False)]
    assert evaluate(judged, art).items["plan"] == "unmet" and evaluate(judged, art).carried is False
    only_judge = ChangedArtifact(path="p.md", rubric=[RubricItem(id="plan", check="judge")])
    assert evaluate(evs, only_judge).carried is None  # nothing decidable: undefined, not "not followed"


def test_apply_judge_writes_judgment_events() -> None:
    art = ChangedArtifact(path="p.md", rubric=[RubricItem(id="plan", check="judge")])
    run = RunRecord(
        run_id="r",
        task_id="t",
        split="report",
        arm="B",
        harness_hash="h",
        model="m",
        score=1,
        evaluator_hash="e",
        events=[call(1, "Bash", "pytest")],
    )
    untraced = run.model_copy(update={"run_id": "u", "events": None})

    class Yes:
        def judge(self, run, art, items):  # type: ignore[no-untyped-def]
            return {i.id: True for i in items}

    out = apply_judge([run, untraced], [art], Yes())
    assert out[1].events is None
    j = [e for e in out[0].events or [] if e.type == "rubric_judgment"]
    assert [(e.name, e.ok, e.artifact) for e in j] == [("plan", True, "p.md")]
    assert evaluate(out[0].events or [], art).carried is True


def test_signatures_still_work_as_a_one_item_rubric() -> None:
    art = ChangedArtifact(path=SKILL, signatures=[Signature(tool="Bash", pattern=r"python3? -c")])
    assert [i.id for i in art.effective_rubric()] == ["signature"]
    assert evaluate([call(1, "Bash", "python3 -c 'assert 1'")], art).carried is True
    assert evaluate([call(1, "Read", "python -c")], art).carried is False


@pytest.mark.parametrize(
    "items",
    [
        [RubricItem(id="a"), RubricItem(id="a")],
        [RubricItem(id="a", after=["b"])],
        [RubricItem(id="a", after=["b"]), RubricItem(id="b", after=["a"])],
        [RubricItem(id="a", on_failure_of=["a"])],
    ],
)
def test_invalid_rubrics_are_rejected(items: list[RubricItem]) -> None:
    with pytest.raises(ValidationError):
        ChangedArtifact(path="x", rubric=items)


def test_invalid_regex_is_rejected() -> None:
    with pytest.raises(ValidationError):
        RubricItem(id="a", pattern="(")


def test_adherence_check_reports_items_for_a_and_b() -> None:
    records, m = generate("clean")
    res = run_audit(records, m)
    ch = res.check("adherence")
    assert ch.status == Status.PASS and ch.rubric is False and ch.required is False
    d = ch.metrics["per_artifact"][SKILL]
    assert d["from_signatures"] and d["followed_rate_B"] > d["followed_rate_A"]
    md, html = to_markdown(res), to_html(res)
    assert "followed (every required item)" in md and "B (from the update)" in html


def test_reached_but_not_followed_is_noted_without_changing_the_verdict() -> None:
    records, m = generate("reached_not_followed")
    res = run_audit(records, m)
    assert res.check("reached_or_carried").status == Status.PASS
    assert res.check("adherence").status == Status.NOTE
    assert "no more often with the update" in res.check("adherence").reason
    assert res.claim("Activation").verdict.value == "PASS"


def test_markdown_escapes_pipes_in_rubric_text() -> None:
    records, m = generate("clean")
    m.changed_artifacts[0].rubric = [RubricItem(id="tests", text="run a | b", tool="Bash", pattern="test")]
    md = to_markdown(run_audit(records, m))
    assert "run a \\| b" in md

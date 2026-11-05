"""Checks driven by manifest settings and edge cases, built by editing the clean scenario."""

from __future__ import annotations

from harness_audit.audit import run_audit
from harness_audit.schema import Status, TaskEntry, Verdict
from harness_audit.synth import generate


def clean():
    return generate("clean")


def test_reused_holdout_warns() -> None:
    records, m = clean()
    m.selection.report_uses = 4
    res = run_audit(records, m)
    assert res.check("reused_holdout").status == Status.WARN
    assert "Reused holdout" in res.check("reused_holdout").reason
    assert res.claim("Selection").verdict == Verdict.PARTIAL


def test_report_choice_selected_fails_selection_and_tightens_margin() -> None:
    records, m = clean()
    m.selection.report_choice = "selected"
    m.selection.n_compared = 8
    res = run_audit(records, m)
    assert res.check("report_choice").status == Status.FAIL
    assert res.claim("Selection").verdict == Verdict.FAIL
    assert res.check("margin").metrics["ci_level"] == 1 - 0.05 / 8


def test_admission_on_report_split_is_partial() -> None:
    records, m = clean()
    m.selection.admission_data = "report"
    assert run_audit(records, m).check("admission_data").status == Status.WARN


def test_unconditional_commit_is_not_a_selection_failure() -> None:
    records, m = clean()
    m.selection.admission_data = "none"
    assert run_audit(records, m).check("admission_data").status == Status.PASS


def test_gate_falls_back_to_report_split_and_fails_with_few_tasks() -> None:
    records, m = clean()
    records = [r for r in records if r.split != "admission"]
    res = run_audit(records, m)
    gate = res.check("admission_gate")
    assert gate.metrics["split"] == "report"
    # 30 report tasks, effect about 0.6: epsilon = 0.447 so the gate still clears.
    assert gate.status == Status.PASS
    few = [r for r in records if r.split != "report" or r.task_id.endswith(("-00", "-01", "-02"))]
    assert run_audit(few, m).check("admission_gate").status == Status.WARN


def test_admission_overlapping_proposal_is_not_fresh() -> None:
    records, m = clean()
    adm = next(t for t in m.tasks if t.split == "admission")
    m.tasks.append(TaskEntry(task_id=adm.task_id, split="proposal", group=adm.group, task_hash=adm.task_hash))
    gate = run_audit(records, m).check("admission_gate")
    assert gate.status == Status.WARN and "not fresh" in gate.reason


def test_near_duplicate_detected_by_content_hash() -> None:
    records, m = clean()
    rep = next(t for t in m.tasks if t.split == "report")
    m.tasks.append(TaskEntry(task_id="proposal/renamed-copy", split="proposal", group="strings", task_hash=rep.task_hash))
    ch = run_audit(records, m).check("split_disjoint_proposal")
    assert ch.status == Status.WARN
    assert ch.metrics["overlap_by_content_hash"] == [rep.task_id]


def test_full_overlap_with_proposal_is_fail() -> None:
    records, m = clean()
    for t in [t for t in m.tasks if t.split == "report"]:
        m.tasks.append(TaskEntry(task_id=t.task_id, split="proposal", group=t.group, task_hash=t.task_hash))
    res = run_audit(records, m)
    assert res.check("split_disjoint_proposal").status == Status.FAIL
    assert res.check("task_distance").status == Status.FAIL


def test_missing_proposal_tasks_partition_not_shown() -> None:
    records, m = clean()
    m.tasks = [t for t in m.tasks if t.split != "proposal"]
    m.proposal_groups = ["strings"]
    res = run_audit(records, m)
    assert res.check("partition_shown").status == Status.WARN
    assert res.check("task_distance").status == Status.WARN


def test_identical_harness_fails_isolation() -> None:
    records, m = clean()
    for r in records:
        r.harness_hash = "sha256:same"
    m.harness = {}
    res = run_audit(records, m)
    assert res.check("isolated_update").status == Status.FAIL


def test_confounds_and_missing_artifacts_warn() -> None:
    records, m = clean()
    m.confounds = ["max_turns 10 -> 20"]
    assert run_audit(records, m).check("isolated_update").status == Status.WARN
    records, m = clean()
    m.changed_artifacts = []
    res = run_audit(records, m)
    assert res.check("isolated_update").status == Status.WARN
    assert res.claim("Activation").verdict == Verdict.UNKNOWN


def test_model_change_fails_attribution() -> None:
    records, m = clean()
    for r in records:
        if r.arm == "B":
            r.model = "other-model"
    res = run_audit(records, m)
    assert res.check("same_model").status == Status.FAIL
    assert res.claim("Attribution").verdict == Verdict.FAIL
    assert res.claim("Attribution").rubric_grade == "O"


def test_missing_usage_means_envelope_not_accounted() -> None:
    records, m = clean()
    for r in records:
        r.usage = None
    ch = run_audit(records, m).check("envelope")
    assert ch.status == Status.WARN and "not reported" in ch.reason


def test_no_gain_is_partial_attribution() -> None:
    records, m = clean()
    a_scores = {(r.task_id, r.seed): r.score for r in records if r.arm == "A"}
    for r in records:
        if r.arm == "B":
            r.score = a_scores[(r.task_id, r.seed)]
    res = run_audit(records, m)
    assert res.check("margin").status == Status.WARN
    assert res.claim("Attribution").verdict == Verdict.PARTIAL


def test_aa_split_half_when_no_replicate_arm() -> None:
    records, m = clean()
    records = [r for r in records if r.arm != "A2"]
    ch = run_audit(records, m).check("aa_noise")
    assert "split-half" in ch.metrics["source"]
    assert ch.status == Status.PASS


def test_aa_note_when_single_replicate() -> None:
    records, m = clean()
    records = [r for r in records if r.arm != "A2" and r.seed == 0]
    res = run_audit(records, m)
    assert res.check("aa_noise").status == Status.NOTE
    assert res.claim("Attribution").verdict == Verdict.PASS


def test_aa_unstable_evaluation_warns() -> None:
    records, m = clean()
    for r in records:
        if r.arm == "A2":
            r.score = min(1.0, r.score + 0.5)
    ch = run_audit(records, m).check("aa_noise")
    assert ch.status == Status.WARN and "not stable" in ch.reason


def test_replicate_with_other_harness_is_not_aa() -> None:
    records, m = clean()
    for r in records:
        if r.arm == "A2":
            r.harness_hash = "sha256:other"
    assert run_audit(records, m).check("aa_noise").status == Status.WARN


def test_single_paired_task_fails_measured_variation() -> None:
    records, m = clean()
    first = next(r.task_id for r in records if r.split == "report")
    records = [r for r in records if r.split != "report" or r.task_id == first]
    res = run_audit(records, m)
    assert res.check("measured_variation").status == Status.FAIL


def test_retention_untested_is_partial() -> None:
    records, m = clean()
    records = [r for r in records if r.split != "retention"]
    res = run_audit(records, m)
    assert res.check("retention").status == Status.WARN
    assert res.claim("Improvement").verdict == Verdict.PARTIAL


def test_traces_without_artifact_events_fail_activation() -> None:
    records, m = clean()
    for r in records:
        if r.events is not None:
            r.events = [e for e in r.events if e.type == "tool_call" and e.name != "Bash"]
    res = run_audit(records, m)
    assert res.check("reached_or_carried").status == Status.FAIL
    assert res.claim("Activation").verdict == Verdict.FAIL


def test_baseline_contamination_warns() -> None:
    records, m = clean()
    from harness_audit.schema import Event

    for r in records:
        if r.arm == "A" and r.split == "report" and r.events is not None:
            r.events.append(Event(t=0.0, type="artifact_loaded", artifact="skills/python-bugfix/SKILL.md"))
            break
    assert run_audit(records, m).check("baseline_clean").status == Status.WARN


def test_carried_from_explicit_event_without_signature() -> None:
    records, m = clean()
    from harness_audit.schema import Event

    m.changed_artifacts[0].signatures = []
    for r in records:
        if r.arm == "B" and r.events:
            r.events = [e for e in r.events if e.type != "artifact_reached"]
    assert run_audit(records, m).check("reached_or_carried").status == Status.WARN
    for r in records:
        if r.arm == "B" and r.events is not None:
            r.events.append(Event(t=9.0, type="artifact_carried", artifact="skills/python-bugfix/SKILL.md"))
    assert run_audit(records, m).check("reached_or_carried").status == Status.PASS


def test_system_prompt_carried_by_adherence_signature() -> None:
    records, m = clean()
    from harness_audit.schema import Event

    art = m.changed_artifacts[0]
    art.kind = "system_prompt"
    art.path = "system_prompt.md"
    for r in records:
        if r.arm == "B" and r.events is not None:
            r.events = [Event(t=0.0, type="artifact_loaded", artifact="system_prompt.md")] + [
                e for e in r.events if e.type == "tool_call"
            ]
    ch = run_audit(records, m).check("reached_or_carried")
    assert ch.status == Status.PASS
    assert ch.metrics["per_artifact"]["system_prompt.md"]["carried_rate"] > 0.5

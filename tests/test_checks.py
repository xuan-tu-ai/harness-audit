"""Checks driven by manifest settings and edge cases, built by editing the clean scenario."""

from __future__ import annotations

from harness_audit.audit import run_audit
from harness_audit.report import to_markdown
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
    # The skill is seen only in B's traces, so it is inferred as the changed artifact.
    assert res.claim("Activation").verdict == Verdict.PASS
    assert any("inferred from traces" in w for w in res.warnings)
    for r in records:
        if r.events is not None:
            r.events = [e for e in r.events if e.type == "tool_call"]
    assert run_audit(records, m).claim("Activation").verdict == Verdict.UNKNOWN


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
    assert ch.metrics["per_artifact"]["system_prompt.md"]["carry_rate"] > 0.5


def test_per_task_evaluators_are_allowed() -> None:
    records, m = clean()
    for r in records:
        r.evaluator_hash = "sha256:eval-" + r.task_id
    assert run_audit(records, m).check("same_evaluator").status == Status.PASS


# ---- Activation rates (report Eq. 11) ---------------------------------------------------------------


def test_activation_rates_use_the_applicable_runs_as_denominator() -> None:
    records, m = clean()
    m.changed_artifacts[0].applies_to = ["strings"]
    per = run_audit(records, m).check("reached_or_carried").metrics["per_artifact"]["skills/python-bugfix/SKILL.md"]
    assert per["applicable"] == "45/90"  # 15 string tasks x 3 seeds of the 30 report tasks
    assert per["available"].endswith("/45") and per["reached"].endswith("/45")
    k_l, k_b = int(per["available"].split("/")[0]), int(per["reached"].split("/")[0])
    assert per["load_rate"] == round(k_l / 45, 4) and per["reach_rate"] == round(k_b / 45, 4)
    k_c, rest = per["carried"].split("/")
    assert rest == f"{k_l} loaded" and per["carry_rate"] == round(int(k_c) / k_l, 4)


def test_executable_update_carry_is_conditioned_on_reach() -> None:
    records, m = clean()
    m.changed_artifacts[0].kind = "code"
    per = run_audit(records, m).check("reached_or_carried").metrics["per_artifact"]["skills/python-bugfix/SKILL.md"]
    k_b = int(per["reached"].split("/")[0])
    assert per["carry_denominator"] == "reached" and per["carried"].endswith(f"/{k_b} reached")


def test_rates_are_undefined_not_zero_when_nothing_applies() -> None:
    records, m = clean()
    m.changed_artifacts[0].applies_to = ["no-such-group"]
    res = run_audit(records, m)
    per = res.check("reached_or_carried").metrics["per_artifact"]["skills/python-bugfix/SKILL.md"]
    assert per["applicable"] == "0/90" and per["load_rate"] is None and per["carry_rate"] is None
    # Existence is still judged over every scored run, and the low-reach note does not fire on an undefined rate.
    assert res.check("reached_or_carried").status == Status.PASS
    assert res.check("gain_without_activation").status == Status.PASS
    assert "undefined" in to_markdown(res)


# ---- Oracle integrity (report Table 5) --------------------------------------------------------------


def test_evaluator_changed_during_a_run_fails() -> None:
    records, m = clean()
    records[5] = records[5].model_copy(update={"evaluator_hash_after": "sha256:" + "0" * 64})
    res = run_audit(records, m)
    assert res.check("oracle_integrity").status == Status.FAIL
    assert res.claim("Selection").verdict == Verdict.FAIL
    assert res.claim("Selection").rubric_grade == "C"


def test_scorer_inside_workspace_warns() -> None:
    records, m = clean()
    m.oracle.outside_workspace = False
    assert run_audit(records, m).check("oracle_integrity").status == Status.WARN


def test_undeclared_oracle_is_a_note() -> None:
    records, m = clean()
    m.oracle.outside_workspace = None
    m.oracle.protected = []
    records = [r.model_copy(update={"evaluator_hash_after": None}) for r in records]
    res = run_audit(records, m)
    assert res.check("oracle_integrity").status == Status.NOTE
    assert res.claim("Selection").verdict == Verdict.PASS


# ---- Model record, envelope and search cost (report Table 5) ----------------------------------------


def test_model_version_drift_fails_with_rubric_partial() -> None:
    records, m = generate("model_version_drift")
    res = run_audit(records, m)
    assert res.check("model_record").status == Status.FAIL
    assert res.claim("Attribution").verdict == Verdict.FAIL and res.claim("Attribution").rubric_grade == "O"


def test_same_version_mix_in_both_arms_warns() -> None:
    records, m = clean()
    records = [r.model_copy(update={"model_versions": ["m-1"] if i % 2 else ["m-2"]}) for i, r in enumerate(records)]
    res = run_audit(records, m)
    assert res.check("model_record").status == Status.WARN and "several model versions" in res.check("model_record").reason


def test_decoding_or_runtime_change_between_arms_warns() -> None:
    records, m = clean()
    hot = [r.model_copy(update={"decoding": {"effort": "high"}}) if r.arm == "B" else r for r in records]
    assert run_audit(hot, m).check("model_record").status == Status.WARN
    new_rt = [r.model_copy(update={"runtime": "rt/2"}) if r.arm == "B" else r for r in records]
    res = run_audit(new_rt, m)
    assert res.check("model_record").status == Status.WARN and "runtime differs" in res.check("model_record").reason


def test_missing_model_record_is_a_note_only() -> None:
    records, m = clean()
    old = [r.model_copy(update={"model_versions": None, "decoding": None, "runtime": None}) for r in records]
    res = run_audit(old, m)
    ch = res.check("model_record")
    assert ch.status == Status.NOTE and "Not recorded" in ch.reason
    assert res.claim("Attribution").verdict == Verdict.PASS


def test_envelope_grades_model_calls_and_names_unrecorded_fields() -> None:
    records, m = clean()
    more = [
        r.model_copy(update={"usage": r.usage.model_copy(update={"model_calls": r.usage.model_calls * 2})}) if r.arm == "B" else r
        for r in records
    ]
    ch = run_audit(more, m).check("envelope")
    assert ch.status == Status.WARN and "model_calls ×2" in ch.reason
    bare = [r.model_copy(update={"usage": r.usage.model_copy(update={"model_calls": None, "attempts": None})}) for r in records]
    ch = run_audit(bare, m).check("envelope")
    assert ch.status == Status.PASS and "Not recorded: model_calls, attempts" in ch.reason


def test_search_cost_is_reported_apart_or_noted() -> None:
    records, m = clean()
    ch = run_audit(records, m).check("search_cost")
    assert ch.status == Status.PASS and "$1.2500" in ch.reason and "Not part of the scored-run envelope" in ch.reason
    m.search_cost = None
    res = run_audit(records, m)
    assert res.check("search_cost").status == Status.NOTE and res.claim("Attribution").verdict == Verdict.PASS

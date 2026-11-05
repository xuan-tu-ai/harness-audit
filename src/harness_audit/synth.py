"""Synthetic experiments with planted defects, used by the test suite and the demo.

Every number produced here is simulated. Each scenario plants exactly one defect
(or none for ``clean``) so the test suite can assert that the matching check, and
only that check, changes the verdict.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .schema import (
    Arms,
    ChangedArtifact,
    Event,
    Manifest,
    Oracle,
    RunRecord,
    SelectionConfig,
    Signature,
    TaskEntry,
    Usage,
)

SCENARIOS: dict[str, str] = {
    "clean": "No defect; all four claims should PASS.",
    "split_leak": "A third of the report tasks were also used to propose the update.",
    "never_activated": "B loads the skill but never invokes it; B still scores higher.",
    "double_tokens": "B's runs use about twice the output tokens, turns and tool calls of A's.",
    "evaluator_changed": "B was scored with a different evaluator than A.",
    "same_group_only": "B improves only on the task group that formed the update.",
    "retention_regression": "B regresses on the retention (earlier) tasks.",
    "no_traces": "No traces were collected, so Activation cannot be judged.",
    "oracle_touched": "Some B runs read the hidden tests that score them.",
    "reached_not_followed": "B invokes the skill but rarely follows its procedure afterwards.",
}

SKILL = "skills/python-bugfix/SKILL.md"
H_A = "sha256:" + "a" * 64
H_B = "sha256:" + "b" * 64
EVAL = "sha256:" + "e" * 64
EVAL_OTHER = "sha256:" + "f" * 64
SKILL_HASH = "sha256:" + "5" * 64
MODEL = "synthetic-model"
N_TESTS = 4


def _h(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


@dataclass(frozen=True)
class _Task:
    task_id: str
    split: str
    group: str
    base: float


def _tasks(rng: np.random.Generator) -> list[_Task]:
    out: list[_Task] = []

    def add(split: str, group: str, n: int) -> None:
        for i in range(n):
            out.append(_Task(f"{split}/{group}-{i:02d}", split, group, float(rng.uniform(0.1, 0.35))))

    add("proposal", "strings", 5)
    add("admission", "strings", 40)
    add("report", "strings", 15)
    add("report", "numbers", 8)
    add("report", "dates", 7)
    add("retention", "legacy", 15)
    return out


def _score(rng: np.random.Generator, p: float) -> float:
    return float(rng.binomial(N_TESTS, min(max(p, 0.0), 1.0)) / N_TESTS)


def generate(scenario: str = "clean", seed: int = 0, seeds: int = 3) -> tuple[list[RunRecord], Manifest]:
    """Return ``(records, manifest)`` for one synthetic scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    tasks = _tasks(rng)
    leak = {t.task_id for t in tasks if t.split == "report"} if scenario == "split_leak" else set()
    leaked = sorted(leak)[::3]

    records: list[RunRecord] = []
    for t in tasks:
        if t.split == "proposal":
            continue
        arms = ["A", "B"] + (["A2"] if t.split == "report" else [])
        for arm in arms:
            for s in range(seeds):
                records.append(_run(rng, scenario, t, arm, s))

    registry = [
        TaskEntry(task_id=t.task_id, split=t.split, group=t.group, task_hash=_h(t.task_id))  # type: ignore[arg-type]
        for t in tasks
    ]
    registry += [TaskEntry(task_id=tid, split="proposal", group="strings", task_hash=_h(tid)) for tid in leaked]
    manifest = Manifest(
        experiment=f"synthetic/{scenario}",
        baseline="harness A (pre-update state)",
        round=1,
        arms=Arms(baseline="A", candidate="B", replicate="A2"),
        harness={"A": H_A, "B": H_B},
        changed_artifacts=[
            ChangedArtifact(
                path=SKILL,
                kind="skill",
                change="added",
                hash_a=None,
                hash_b=SKILL_HASH,
                signatures=[Signature(tool="Bash", pattern=r"python3? .*test")],
            )
        ],
        proposal_groups=None,
        selection=SelectionConfig(
            K=1, T=1, delta=0.05, admission_data="admission", report_uses=1, report_choice="fixed", n_compared=1
        ),
        oracle=Oracle(outside_workspace=True, protected=[r"(^|[/\s])evaluator/", r"test_hidden\.py"]),
        tasks=registry,
    )
    return records, manifest


def _run(rng: np.random.Generator, scenario: str, t: _Task, arm: str, s: int) -> RunRecord:
    is_b = arm == "B"
    reached = is_b and scenario != "never_activated" and rng.random() < 0.9
    carried = reached and rng.random() < (0.05 if scenario == "reached_not_followed" else 0.9)
    # Probability of passing each hidden test.
    p = t.base
    if is_b:
        if t.split == "retention":
            p = t.base - 0.35 if scenario == "retention_regression" else t.base + 0.1
        elif scenario == "same_group_only" and t.group != "strings":
            p = t.base
        elif scenario == "never_activated":
            p = 0.9 if rng.random() < 0.9 else t.base
        elif reached:
            p = 0.9
    score = _score(rng, p)

    out_tok = rng.normal(1500, 200)
    turns = rng.normal(6, 1)
    tools = rng.normal(10, 2)
    in_tok = rng.normal(20000, 3000)
    if is_b:
        in_tok += 1500  # the skill's own context
        tools += 1.0 if reached else 0.0
        if scenario == "double_tokens":
            out_tok, turns, tools = out_tok * 2.1, turns * 2.0, tools * 2.0
    out_i, in_i = max(int(out_tok), 50), max(int(in_tok), 500)
    usage = Usage(
        input_tokens=in_i,
        output_tokens=out_i,
        cost_usd=round(in_i * 0.1e-6 + out_i * 0.5e-6, 6),
        wall_s=round(float(rng.normal(40, 8)), 2),
        tool_calls=max(int(round(tools)), 1),
        num_turns=max(int(round(turns)), 1),
    )

    events: list[Event] | None = [Event(t=0.0, type="tool_call", name="Read", args_hash=_h(f"{t.task_id}r"), summary="buggy.py")]
    if is_b:
        events.insert(0, Event(t=0.0, type="artifact_loaded", artifact=SKILL, hash=SKILL_HASH))
    if reached:
        events.append(Event(t=2.0, type="artifact_reached", artifact=SKILL, evidence="Skill tool invoked: python-bugfix"))
        events.append(Event(t=2.0, type="tool_call", name="Skill", args_hash=_h("skill"), summary="python-bugfix"))
    events.append(Event(t=4.0, type="tool_call", name="Edit", args_hash=_h(f"{t.task_id}e"), summary="buggy.py"))
    if is_b and scenario == "oracle_touched" and rng.random() < 0.3:
        events.append(
            Event(t=3.0, type="tool_call", name="Bash", args_hash=_h("peek"), summary="cat ../evaluator/test_hidden.py")
        )
    if carried or (not is_b and rng.random() < 0.25):
        events.append(Event(t=6.0, type="tool_call", name="Bash", args_hash=_h("test"), summary="python test_public.py"))
    if scenario == "no_traces":
        events = None

    return RunRecord(
        run_id=f"{arm}-{t.task_id}-s{s}",
        task_id=t.task_id,
        task_group=t.group,
        task_hash=_h(t.task_id),
        split=t.split,  # type: ignore[arg-type]
        arm=arm,
        harness_hash=H_B if is_b else H_A,
        seed=s + (1000 if arm == "A2" else 0),
        model=MODEL,
        score=score,
        evaluator_hash=EVAL_OTHER if (is_b and scenario == "evaluator_changed") else EVAL,
        evaluator_hash_after=EVAL_OTHER if (is_b and scenario == "evaluator_changed") else EVAL,
        usage=usage,
        events=events,
    )

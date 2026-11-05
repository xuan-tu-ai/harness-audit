"""A deterministic adapter for runner tests: no model, no network.

Under a harness with skills it applies the task's reference fix and records
activation events; without skills it leaves the bug in place.
"""

from __future__ import annotations

import shutil

from harness_audit.adapters.base import AdapterResult, RunRequest
from harness_audit.schema import Event, Usage


class FakeAdapter:
    id = "fake/1"

    def __init__(self, cost: float = 0.01, cost_known: bool = True) -> None:
        self.cost = cost
        self.cost_known = cost_known
        self.calls = 0

    def installed_paths(self) -> list[str]:
        return [".claude"]

    async def run(self, req: RunRequest) -> AdapterResult:
        self.calls += 1
        (req.workdir / ".claude").mkdir(exist_ok=True)
        events = [Event(t=0.1, type="tool_call", name="Read", summary="module.py")]
        if req.harness.skills:
            path = req.harness.skill_path(req.harness.skills[0])
            events.insert(0, Event(t=0.0, type="artifact_loaded", artifact=path))
            events.append(Event(t=0.2, type="artifact_reached", artifact=path, evidence="Skill tool invoked"))
            events.append(Event(t=0.3, type="tool_call", name="Bash", summary="python -c 'assert f(0) == 0'"))
            module = next(p for p in req.workdir.glob("*.py") if p.name != "test_public.py")
            shutil.copy(req.task.evaluator_dir / "reference.py", module)
        usage = Usage(
            input_tokens=20000, output_tokens=800, num_turns=5, tool_calls=6, cost_usd=self.cost if self.cost_known else None
        )
        return AdapterResult(events=events, usage=usage, status="ok", model=req.model)

"""OpenAI-compatible adapter, tested offline with scripted chat-completions responses."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from harness_audit.adapters import FatalAdapterError, load_adapter
from harness_audit.adapters import openai_compat as mod
from harness_audit.adapters.base import RunRequest
from harness_audit.harness import load_harness
from harness_audit.tasks import load_tasks

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SKILL = "skills/python-bugfix/SKILL.md"


def _req(tmp_path: Path, arm: str = "harness_b") -> RunRequest:
    task = next(t for t in load_tasks(EXAMPLES / "tasks") if t.id == "strings/initials")
    work = tmp_path / "work"
    work.mkdir(parents=True)
    import shutil

    shutil.copytree(task.workspace, work, dirs_exist_ok=True)
    return RunRequest(task, load_harness(EXAMPLES / arm), work, 0, "deepseek-test", 0.5, 60)


def call(i: int, name: str, **args: object) -> dict:
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def resp(calls: list[dict] | None = None, content: str = "", cost: float | None = 0.0001) -> dict:
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    if cost is not None:
        usage["cost"] = cost
    return {
        "model": "deepseek-test-2026",
        "choices": [{"message": {"role": "assistant", "content": content, "tool_calls": calls}}],
        "usage": usage,
    }


def scripted(monkeypatch: pytest.MonkeyPatch, adapter: mod.OpenAICompatAdapter, script: list[dict]) -> list[dict]:
    sent: list[dict] = []

    def fake_post(payload: dict) -> tuple[dict, int]:
        sent.append(json.loads(json.dumps(payload)))
        return script[len(sent) - 1], 0

    monkeypatch.setattr(adapter, "_post", fake_post)
    return sent


def test_registry_and_options() -> None:
    a = load_adapter("openai_compat", {"base_url": "https://gw.example/v1/", "api_key_env": "X_KEY"})
    assert isinstance(a, mod.OpenAICompatAdapter) and a.base_url == "https://gw.example/v1" and a.api_key_env == "X_KEY"


def test_preflight_needs_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKENHARBOR_API_KEY", raising=False)
    with pytest.raises(FatalAdapterError, match="TOKENHARBOR_API_KEY"):
        mod.OpenAICompatAdapter().preflight()


def test_skill_listed_and_tools_offered(tmp_path: Path) -> None:
    a = mod.OpenAICompatAdapter()
    hb, ha = load_harness(EXAMPLES / "harness_b"), load_harness(EXAMPLES / "harness_a")
    assert "python-bugfix:" in a.system_prompt(hb) and "Skill" in [t["function"]["name"] for t in a.tools_for(hb)]
    assert "# Skills" not in a.system_prompt(ha) and "Skill" not in [t["function"]["name"] for t in a.tools_for(ha)]


def test_full_loop_records_trace_and_fixes_nothing_by_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req = _req(tmp_path)
    a = mod.OpenAICompatAdapter()
    ref = (req.task.evaluator_dir / "reference.py").read_text()
    script = [
        resp([call(1, "Skill", skill="python-bugfix")]),
        resp([call(2, "Read", file_path="initials.py")]),
        resp([call(3, "Write", file_path="initials.py", content=ref)]),
        resp([call(4, "Bash", command="python test_public.py")]),
        resp([call(5, "Bash", command="python -c 'assert 1 == 2'")]),  # fails: exit code 1
        resp([call(6, "Read", file_path="../../etc/passwd")]),  # outside the workdir: refused
        resp(content="done"),
    ]
    sent = scripted(monkeypatch, a, script)
    res = asyncio.run(a.run(req))
    assert res.status == "ok" and res.model_versions == ["deepseek-test-2026"] and res.runtime == mod.RUNTIME
    assert res.usage.model_calls == 7 and res.usage.tool_calls == 6 and res.usage.input_tokens == 7000
    assert res.usage.cost_usd == pytest.approx(0.0007)
    types = [(e.type, e.artifact) for e in res.events]
    assert ("artifact_loaded", SKILL) in types and ("artifact_reached", SKILL) in types
    ok = [(e.summary, e.ok) for e in res.events if e.type == "tool_call"]
    assert ok[3] == ("python test_public.py", True) and ok[4][1] is False and ok[5][1] is False
    # Tool results go back to the model, with the tool_call_id.
    last = sent[-1]["messages"]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert len(tool_msgs) == 6 and tool_msgs[0]["tool_call_id"] == "c1" and "Fixing a Python function" in tool_msgs[0]["content"]
    assert "outside the working directory" in tool_msgs[5]["content"]
    assert (req.workdir / "initials.py").read_text() == ref


def test_bash_gets_no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOKENHARBOR_API_KEY", "should-not-leak")
    req = _req(tmp_path)
    out = mod.Tools(req.workdir, req.harness).Bash("env")
    assert "should-not-leak" not in out and "TOKENHARBOR" not in out


def test_unknown_cost_falls_back_to_prices_or_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req = _req(tmp_path)
    a = mod.OpenAICompatAdapter()
    scripted(monkeypatch, a, [resp(content="hi", cost=None)])
    assert asyncio.run(a.run(req)).usage.cost_usd is None  # no price known for this model
    from harness_audit.cost import PRICES, register_prices

    register_prices({"deepseek-test": {"input": 1.0, "output": 2.0}})
    try:
        scripted(monkeypatch, a, [resp(content="hi", cost=None)])
        assert asyncio.run(a.run(req)).usage.cost_usd == pytest.approx((1000 * 1.0 + 100 * 2.0) / 1e6)
    finally:
        PRICES.pop("deepseek-test")


def test_budget_and_turn_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req = _req(tmp_path)
    a = mod.OpenAICompatAdapter()
    scripted(monkeypatch, a, [resp([call(i, "Glob", pattern="*.py")], cost=0.3) for i in range(30)])
    res = asyncio.run(a.run(req))
    assert res.status == "budget_exceeded" and res.usage.model_calls == 2  # 0.6 >= 0.5
    scripted(monkeypatch, a, [resp([call(i, "Glob", pattern="*.py")], cost=0.0) for i in range(30)])
    assert asyncio.run(a.run(req)).status == "max_turns"


def test_rejected_key_is_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req = _req(tmp_path)
    a = mod.OpenAICompatAdapter()

    def refuse(payload: dict) -> tuple[dict, int]:
        raise FatalAdapterError("gateway refused the key (401)")

    monkeypatch.setattr(a, "_post", refuse)
    with pytest.raises(FatalAdapterError):
        asyncio.run(a.run(req))


def test_experiment_rejects_keys_in_adapter_options(tmp_path: Path) -> None:
    import yaml

    from harness_audit.io import InputError
    from harness_audit.runner import load_experiment

    cfg = yaml.safe_load((EXAMPLES / "experiment.yaml").read_text())
    cfg["adapter_options"] = {"api_key": "thk_live_x"}
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(InputError, match="must not hold credentials"):
        load_experiment(p)


def test_tokenharbor_example_loads_and_asks_for_prices() -> None:
    from harness_audit.io import InputError
    from harness_audit.runner import Runner, load_experiment

    exp = load_experiment(EXAMPLES / "experiment.tokenharbor.yaml")
    assert exp.config.adapter == "openai_compat" and exp.config.adapter_options["api_key_env"] == "TOKENHARBOR_API_KEY"
    r = Runner(exp)
    with pytest.raises(InputError, match="pricing"):
        r.describe(r.plan())

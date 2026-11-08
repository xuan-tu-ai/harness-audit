"""Claude Agent SDK adapter, tested offline with the SDK's own message types and a fake query()."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

sdk = pytest.importorskip("claude_agent_sdk")

from harness_audit.adapters import claude_agent_sdk as mod  # noqa: E402
from harness_audit.adapters.base import RunRequest  # noqa: E402
from harness_audit.harness import load_harness  # noqa: E402
from harness_audit.tasks import load_tasks  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SKILL = "skills/python-bugfix/SKILL.md"


def _req(tmp_path: Path, arm: str = "harness_b", budget: float = 0.1) -> RunRequest:
    task = next(t for t in load_tasks(EXAMPLES / "tasks") if t.id == "strings/initials")
    work = tmp_path / "work"
    work.mkdir(parents=True)
    return RunRequest(task, load_harness(EXAMPLES / arm), work, 0, "claude-haiku-5-5", budget, 30)


def test_trace_recorder_events(tmp_path: Path) -> None:
    req = _req(tmp_path)
    rec = mod.TraceRecorder(req.harness, req.workdir)
    rec.on_start()
    rec.on_init({"skills": ["python-bugfix", "some-bundled-skill"]})
    run = asyncio.run
    assert run(rec.pre_tool_use({"tool_name": "Skill", "tool_input": {"skill": "python-bugfix"}}, "t1", None)) == {}
    run(
        rec.pre_tool_use(
            {"tool_name": "Read", "tool_input": {"file_path": str(req.workdir / ".claude/skills/python-bugfix/SKILL.md")}},
            "t2",
            None,
        )
    )
    run(rec.pre_tool_use({"tool_name": "Bash", "tool_input": {"command": 'python -c \'assert initials("") == ""\''}}, "t3", None))
    run(
        rec.pre_tool_use(
            {"tool_name": "Edit", "tool_input": {"file_path": str(req.workdir / "initials.py"), "old_string": "SECRET"}},
            "t4",
            None,
        )
    )
    types = [(e.type, e.artifact) for e in rec.events]
    assert ("artifact_loaded", "system_prompt.md") in types
    assert ("artifact_loaded", SKILL) in types
    assert types.count(("artifact_reached", SKILL)) == 2
    assert rec.external_skills == ["some-bundled-skill"]
    assert rec.tool_calls == 4
    summaries = [e.summary for e in rec.events if e.type == "tool_call"]
    assert summaries[2].startswith("python -c") and summaries[3] == "initials.py"
    assert all("SECRET" not in (e.summary or "") for e in rec.events)


def test_hook_never_raises(tmp_path: Path) -> None:
    req = _req(tmp_path)
    rec = mod.TraceRecorder(req.harness, req.workdir)
    assert asyncio.run(rec.pre_tool_use({"tool_name": "Bash", "tool_input": None}, None, None)) == {}


def test_options_isolate_and_enable_skills(tmp_path: Path) -> None:
    req = _req(tmp_path)
    opts = mod.ClaudeAgentSDKAdapter().options(req, mod.TraceRecorder(req.harness, req.workdir))
    assert isinstance(opts, sdk.ClaudeAgentOptions)
    assert opts.setting_sources == ["project"] and opts.permission_mode == "dontAsk"
    assert opts.skills == ["python-bugfix"] and "Skill" in opts.tools and "Skill" in opts.allowed_tools
    assert opts.model == "claude-haiku-5-5" and opts.max_budget_usd == 0.1 and opts.max_turns == 20
    assert opts.system_prompt == req.harness.system_prompt
    assert list(opts.hooks) == ["PreToolUse", "PostToolUse", "PostToolUseFailure"]
    req_a = _req(tmp_path / "a", arm="harness_a")
    opts_a = mod.ClaudeAgentSDKAdapter().options(req_a, mod.TraceRecorder(req_a.harness, req_a.workdir))
    assert opts_a.skills == [] and "Skill" not in opts_a.tools


def test_result_usage_and_status() -> None:
    msg = sdk.ResultMessage(
        subtype="success",
        duration_ms=1200,
        duration_api_ms=1000,
        is_error=False,
        num_turns=4,
        session_id="s",
        total_cost_usd=0.0123,
        usage={"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 9000, "cache_creation_input_tokens": 1000},
    )
    u = mod.result_usage(msg, "claude-haiku-5-5", tool_calls=3, wall_s=1.5)
    assert (u.input_tokens, u.output_tokens, u.cost_usd, u.num_turns, u.tool_calls) == (100, 50, 0.0123, 4, 3)
    assert mod.status_of(msg) == "ok"
    msg.total_cost_usd = None
    u2 = mod.result_usage(msg, "claude-haiku-5-5", 3, 1.5)
    assert u2.cost_usd == pytest.approx((100 * 0.10 + 50 * 0.50 + 9000 * 0.01 + 1000 * 0.125) / 1e6, abs=1e-6)
    msg.subtype = "error_max_budget_usd"
    assert mod.status_of(msg) == "budget_exceeded"
    msg.subtype = "error_max_turns"
    assert mod.status_of(msg) == "max_turns"


def test_run_end_to_end_with_fake_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire-level check of run(): skills installed, init parsed, hook invoked, result read."""
    req = _req(tmp_path)
    seen = {}

    async def fake_query(*, prompt, options):
        seen["prompt"], seen["options"] = prompt, options
        assert (Path(options.cwd) / ".claude/skills/python-bugfix/SKILL.md").exists()
        yield sdk.SystemMessage(subtype="init", data={"skills": ["python-bugfix"], "model": "claude-haiku-5-5"})
        hook = options.hooks["PreToolUse"][0].hooks[0]
        await hook({"tool_name": "Skill", "tool_input": {"skill": "python-bugfix"}}, "id1", None)
        await hook({"tool_name": "Bash", "tool_input": {"command": "python -c 'assert 1'"}}, "id2", None)
        yield sdk.ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=8,
            is_error=False,
            num_turns=2,
            session_id="s",
            total_cost_usd=0.002,
            usage={"input_tokens": 10, "output_tokens": 5},
        )

    monkeypatch.setattr(sdk, "query", fake_query)
    res = asyncio.run(mod.ClaudeAgentSDKAdapter().run(req))
    assert res.status == "ok" and res.usage.cost_usd == 0.002 and res.usage.tool_calls == 2
    assert {e.type for e in res.events} >= {"artifact_loaded", "artifact_reached", "tool_call"}
    assert seen["prompt"] == req.task.spec.prompt


def test_run_records_errors_instead_of_raising(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req = _req(tmp_path)

    async def broken_query(*, prompt, options):
        raise RuntimeError("no credentials")
        yield  # pragma: no cover

    monkeypatch.setattr(sdk, "query", broken_query)
    res = asyncio.run(mod.ClaudeAgentSDKAdapter().run(req))
    assert res.status == "error" and "no credentials" in (res.error or "")


def test_child_env_blanks_host_settings_but_keeps_api_key(tmp_path: Path) -> None:
    environ = {
        "ANTHROPIC_API_KEY": "k",
        "ANTHROPIC_BASE_URL": "http://proxy",
        "CLAUDE_CODE_REMOTE": "1",
        "CLAUDE_CODE_SYNC_SKILLS": "1",
        "PATH": "/bin",
        "HARNESS_AUDIT_PASS_ENV": "ANTHROPIC_BASE_URL",
    }
    env = mod.child_env(tmp_path, environ)
    assert "ANTHROPIC_API_KEY" not in env and "PATH" not in env
    assert env["CLAUDE_CODE_REMOTE"] == "" and env["CLAUDE_CODE_SYNC_SKILLS"] == ""
    assert "ANTHROPIC_BASE_URL" not in env  # passed through because it is listed in HARNESS_AUDIT_PASS_ENV
    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path)
    environ.pop("HARNESS_AUDIT_PASS_ENV")
    assert mod.child_env(tmp_path, environ)["ANTHROPIC_BASE_URL"] == ""


def test_preflight_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_audit.adapters import FatalAdapterError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(FatalAdapterError, match="ANTHROPIC_API_KEY"):
        mod.ClaudeAgentSDKAdapter().preflight()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    mod.ClaudeAgentSDKAdapter().preflight()


def test_refuses_when_cli_uses_another_credential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_audit.adapters import FatalAdapterError

    req = _req(tmp_path)

    async def fake_query(*, prompt, options):
        yield sdk.SystemMessage(subtype="init", data={"skills": [], "apiKeySource": "/login managed key"})
        raise AssertionError("the run must stop at init")

    monkeypatch.setattr(sdk, "query", fake_query)
    with pytest.raises(FatalAdapterError, match="apiKeySource"):
        asyncio.run(mod.ClaudeAgentSDKAdapter().run(req))


def test_tool_outcomes_from_hooks_and_result_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A call's ok flag comes from PostToolUse / PostToolUseFailure and ToolResultBlock.is_error; failure wins."""
    req = _req(tmp_path)

    async def fake_query(*, prompt, options):
        yield sdk.SystemMessage(subtype="init", data={"skills": ["python-bugfix"]})
        pre = options.hooks["PreToolUse"][0].hooks[0]
        post = options.hooks["PostToolUse"][0].hooks[0]
        fail = options.hooks["PostToolUseFailure"][0].hooks[0]
        for tid, cmd in (("a", "python test_public.py"), ("b", "python -c 'assert 0'"), ("c", "ls"), ("d", "pwd")):
            await pre({"tool_name": "Bash", "tool_input": {"command": cmd}, "tool_use_id": tid}, tid, None)
        await post({"tool_name": "Bash", "tool_use_id": "a"}, "a", None)
        await post({"tool_name": "Bash", "tool_use_id": "b"}, "b", None)  # hook says returned ...
        yield sdk.UserMessage(content=[sdk.ToolResultBlock(tool_use_id="b", content="AssertionError", is_error=True)])
        await fail({"tool_name": "Bash", "tool_use_id": "c", "error": "exit 1"}, "c", None)
        yield sdk.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            total_cost_usd=0.001,
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    monkeypatch.setattr(sdk, "query", fake_query)
    res = asyncio.run(mod.ClaudeAgentSDKAdapter().run(req))
    ok = {e.summary: e.ok for e in res.events if e.type == "tool_call"}
    assert ok == {"python test_public.py": True, "python -c 'assert 0'": False, "ls": False, "pwd": None}


def test_model_record_from_the_message_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Served model ids (main loop only), model requests incl. sub-agents, effort in effect and runtime."""
    req = _req(tmp_path)

    def assistant(model: str, mid: str | None, parent: str | None = None) -> object:
        return sdk.AssistantMessage(content=[], model=model, message_id=mid, parent_tool_use_id=parent)

    async def fake_query(*, prompt, options):
        yield sdk.SystemMessage(subtype="init", data={"skills": [], "claude_code_version": "2.1.318"})
        yield assistant("claude-haiku-5-5-20261001", "m1")
        yield assistant("claude-haiku-5-5-20261001", "m1")  # a second block of the same response
        pre = options.hooks["PreToolUse"][0].hooks[0]
        await pre({"tool_name": "Bash", "tool_input": {"command": "ls"}, "effort": {"level": "low"}}, "t", None)
        yield assistant("claude-small-sub-agent", "m2", parent="t")  # a sub-agent request: counted, not a main model
        yield assistant("claude-haiku-5-5-20261001", None)
        yield sdk.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=2,
            session_id="s",
            total_cost_usd=0.001,
            usage={"input_tokens": 1, "output_tokens": 1},
        )

    monkeypatch.setattr(sdk, "query", fake_query)
    res = asyncio.run(mod.ClaudeAgentSDKAdapter().run(req))
    assert res.model_versions == ["claude-haiku-5-5-20261001"]
    assert res.usage.model_calls == 3 and res.usage.attempts == 1
    assert res.runtime.startswith("claude-code/2.1.318; claude-agent-sdk/")
    assert res.decoding["effort_in_effect"] == ["low"] and res.decoding["max_turns"] == 20
    assert "not settable" in res.decoding["temperature"]


def test_served_models_fall_back_to_model_usage() -> None:
    rec = mod.TraceRecorder.__new__(mod.TraceRecorder)
    rec.main_models = []
    msg = type("R", (), {"model_usage": {"claude-haiku-5-5-20261001": {}}})()
    assert mod.served_models(rec, msg) == ["claude-haiku-5-5-20261001"]
    assert mod.served_models(rec, None) is None


def test_gateway_options_route_every_model_call_and_reprice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from harness_audit.adapters import FatalAdapterError, load_adapter
    from harness_audit.cost import PRICES, register_prices

    a = load_adapter(
        "claude_agent_sdk",
        {"base_url": "https://gw.example/", "api_key_env": "GW_KEY", "sdk_max_budget_usd": 3.0},
    )
    monkeypatch.delenv("GW_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-key")
    with pytest.raises(FatalAdapterError, match="GW_KEY"):
        a.preflight()
    monkeypatch.setenv("GW_KEY", "gw-test-key")
    a.preflight()
    req = _req(tmp_path)
    env = a.env_for(req, tmp_path / "cfg")
    assert env["ANTHROPIC_BASE_URL"] == "https://gw.example" and env["ANTHROPIC_API_KEY"] == "gw-test-key"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == req.model
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert a.options(req, mod.TraceRecorder(req.harness, req.workdir), tmp_path / "cfg").max_budget_usd == 3.0

    msg = sdk.ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        total_cost_usd=0.2,  # the SDK's guess for an unknown model
        usage={"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 10000},
        model_usage={"gw-model": {"inputTokens": 1000, "outputTokens": 100, "cacheReadInputTokens": 10000}},
    )
    assert mod.gateway_cost(msg, "gw-model") == (None, ["gw-model"])
    register_prices({"gw-model": {"input": 0.3, "output": 1.2, "cache_read": 0.0015}})
    try:
        cost, missing = mod.gateway_cost(msg, "gw-model")
        assert missing == [] and cost == pytest.approx((1000 * 0.3 + 100 * 1.2 + 10000 * 0.0015) / 1e6)
    finally:
        PRICES.pop("gw-model")


def test_polyglot_experiments_load() -> None:
    from harness_audit.runner import load_experiment

    for m in ("deepseek", "haiku"):
        exp = load_experiment(EXAMPLES / "polyglot" / f"experiment.{m}.yaml")
        assert exp.config.adapter == "claude_agent_sdk" and exp.config.adapter_options["api_key_env"] == "TOKENHARBOR_API_KEY"
        assert exp.harness_b.skills == ["spec-to-tests"] and exp.harness_a.config.use_claude_code_preset


@pytest.mark.parametrize(
    ("text", "expect"),
    [("API Error: 402 Your Pass allowance for this period is used up", "fatal"), ("API Error: 529 Overloaded", "api_error")],
)
def test_endpoint_errors_are_not_scored_as_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, expect: str) -> None:
    """Seen in a real run: a gateway quota ran out mid-experiment and every later run 'failed' in one turn."""
    from harness_audit.adapters import FatalAdapterError

    req = _req(tmp_path)

    async def q(*, prompt, options):
        yield sdk.AssistantMessage(content=[sdk.TextBlock(text=text)], model="<synthetic>")
        yield sdk.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=True,
            num_turns=1,
            session_id="s",
            total_cost_usd=0.0,
            usage={},
            result=text,
        )

    monkeypatch.setattr(sdk, "query", q)
    if expect == "fatal":
        with pytest.raises(FatalAdapterError, match="402"):
            asyncio.run(mod.ClaudeAgentSDKAdapter().run(req))
    else:
        res = asyncio.run(mod.ClaudeAgentSDKAdapter().run(req))
        assert res.status == "api_error" and "529" in (res.error or "") and res.model_versions is None

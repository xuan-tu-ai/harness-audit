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
    assert list(opts.hooks) == ["PreToolUse"]
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

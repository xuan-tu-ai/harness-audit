"""Claude Agent SDK adapter (``pip install harness-audit[claude]``).

API used (checked against claude-agent-sdk 0.2.178 source and docs):

- ``query(prompt=..., options=ClaudeAgentOptions(...))`` yields ``SystemMessage``,
  ``AssistantMessage`` and finally ``ResultMessage`` (``total_cost_usd``, ``usage``,
  ``num_turns``, ``duration_ms``, ``subtype``). ``query()`` streams internally, so hooks work.
- ``hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[cb])]}``; ``cb(input, tool_use_id, ctx)``
  receives ``tool_name`` / ``tool_input`` and returns ``{}`` to allow. ``PostToolUse`` and
  ``PostToolUseFailure`` hooks, and ``ToolResultBlock.is_error`` in ``UserMessage`` content, give each
  call's outcome (recorded as ``ok``; failure from any source wins).
- Skills are ``<cwd>/.claude/skills/<name>/SKILL.md``, discovered through ``setting_sources``;
  ``skills=[...]`` filters what the model may invoke; the ``init`` system message lists loaded
  skills in ``data["skills"]``. We pass ``setting_sources=["project"]`` so the host's user
  settings and personal skills are never loaded into a scored run.
- ``max_budget_usd`` caps one query; ``permission_mode="dontAsk"`` denies anything not in
  ``allowed_tools``.

Credentials and isolation: ``ANTHROPIC_API_KEY`` is read by the SDK's CLI subprocess from the
environment; this module only checks that it is set and never logs or stores it. Every other
inherited ``CLAUDE_*`` / ``ANTHROPIC_*`` variable is blanked for the subprocess (the SDK merges
``options.env`` over ``os.environ``), each run gets an empty ``CLAUDE_CONFIG_DIR``, and a run is
aborted before any model call if the CLI's ``init`` message reports an ``apiKeySource`` other than
``ANTHROPIC_API_KEY`` (so a host login or gateway is never billed by accident). Variables listed in
``HARNESS_AUDIT_PASS_ENV`` (comma-separated) are passed through unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ..cost import cost_from_usage
from ..harness import Harness
from ..schema import Event, Usage
from .base import AdapterResult, FatalAdapterError, RunRequest

SUMMARY_MAX = 160
SKILLS_DIR = Path(".claude") / "skills"
API_KEY_VAR = "ANTHROPIC_API_KEY"
SCRUB_PREFIXES = ("CLAUDE_", "ANTHROPIC_")


def child_env(config_dir: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Overrides for the CLI subprocess: blank inherited Claude/Anthropic settings except the API key."""
    environ = dict(os.environ) if environ is None else environ
    keep = {API_KEY_VAR} | {k.strip() for k in environ.get("HARNESS_AUDIT_PASS_ENV", "").split(",") if k.strip()}
    env = {k: "" for k in environ if k.startswith(SCRUB_PREFIXES) and k not in keep}
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    return env


def _args_hash(tool_input: Any) -> str:
    try:
        blob = json.dumps(tool_input, sort_keys=True, default=str).encode()
    except (TypeError, ValueError):
        blob = repr(tool_input).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _rel(path: str, workdir: Path) -> str:
    try:
        return Path(path).resolve().relative_to(workdir.resolve()).as_posix()
    except (ValueError, OSError):
        return Path(path).name


def summarize(tool_name: str, tool_input: dict[str, Any], workdir: Path) -> str:
    """Short, content-free description of a tool call (commands and paths only)."""
    if not isinstance(tool_input, dict):
        return ""
    if tool_name == "Bash":
        s = str(tool_input.get("command", ""))
    elif tool_name in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
        s = _rel(str(tool_input.get("file_path") or tool_input.get("notebook_path") or ""), workdir)
    elif tool_name in ("Glob", "Grep"):
        s = str(tool_input.get("pattern", ""))
    elif tool_name == "Skill":
        s = str(_skill_name(tool_input) or "")
    else:
        s = ""
    s = " ".join(s.split())
    return s[:SUMMARY_MAX]


def _skill_name(tool_input: dict[str, Any]) -> str | None:
    for key in ("skill", "name", "command"):
        v = tool_input.get(key)
        if isinstance(v, str) and v:
            return v.strip().lstrip("/").split()[0]
    return None


class TraceRecorder:
    """Turns SDK messages and hook calls into harness-audit events."""

    def __init__(self, harness: Harness, workdir: Path) -> None:
        self.harness = harness
        self.workdir = workdir
        self.t0 = time.monotonic()
        self.events: list[Event] = []
        self.tool_calls = 0
        self.loaded_skills: list[str] = []
        self.external_skills: list[str] = []
        self._calls: dict[str, Event] = {}
        self.effort_levels: set[str] = set()
        self.main_models: list[str] = []
        self._message_ids: set[str] = set()
        self._anonymous_messages = 0
        self.runtime: str | None = None

    def _t(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    def on_start(self) -> None:
        if self.harness.system_prompt is not None and self.harness.config.system_prompt_file:
            path = self.harness.config.system_prompt_file
            self.events.append(
                Event(
                    t=0.0,
                    type="artifact_loaded",
                    artifact=path,
                    hash=self.harness.files.get(path),
                    evidence="passed as the system prompt",
                )
            )

    def on_init(self, data: dict[str, Any]) -> None:
        """Record which skills the CLI reports as loaded for this session, and the runtime version."""
        if data.get("claude_code_version"):
            self.runtime = f"claude-code/{data['claude_code_version']}"
        for name in data.get("skills") or []:
            name = str(name)
            if name in self.harness.skills:
                path = self.harness.skill_path(name)
                self.loaded_skills.append(name)
                self.events.append(
                    Event(
                        t=self._t(),
                        type="artifact_loaded",
                        artifact=path,
                        hash=self.harness.files.get(path),
                        evidence="listed in the session init message",
                    )
                )
            else:
                self.external_skills.append(name)

    async def pre_tool_use(self, input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        """PreToolUse hook: record the call and any reached artifact; never blocks."""
        try:
            name = str(input_data.get("tool_name", ""))
            tool_input = input_data.get("tool_input") or {}
            effort = input_data.get("effort")
            if isinstance(effort, dict) and effort.get("level"):
                self.effort_levels.add(str(effort["level"]))
            self.tool_calls += 1
            t = self._t()
            ev = Event(
                t=t,
                type="tool_call",
                name=name,
                args_hash=_args_hash(tool_input),
                summary=summarize(name, tool_input, self.workdir),
            )
            self.events.append(ev)
            tid = tool_use_id or input_data.get("tool_use_id")
            if tid:
                self._calls[str(tid)] = ev
            reached = self._reached(name, tool_input)
            if reached:
                path, evidence = reached
                self.events.append(
                    Event(t=t, type="artifact_reached", artifact=path, hash=self.harness.files.get(path), evidence=evidence)
                )
        except Exception:  # noqa: BLE001 - tracing must never break the run
            pass
        return {}

    def set_outcome(self, tool_use_id: str | None, ok: bool) -> None:
        """Record whether a call succeeded; a failure signal from any source wins over success."""
        ev = self._calls.get(str(tool_use_id)) if tool_use_id else None
        if ev is None:
            return
        if not ok:
            ev.ok = False
        elif ev.ok is None:
            ev.ok = True

    async def post_tool_use(self, input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        """PostToolUse hook: the call returned (its result may still be an error, see ``on_tool_result``)."""
        with contextlib.suppress(Exception):  # tracing must never break the run
            self.set_outcome(tool_use_id or input_data.get("tool_use_id"), True)
        return {}

    async def post_tool_use_failure(self, input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        """PostToolUseFailure hook: the call failed."""
        with contextlib.suppress(Exception):  # tracing must never break the run
            self.set_outcome(tool_use_id or input_data.get("tool_use_id"), False)
        return {}

    def on_assistant(self, model: str | None, message_id: str | None, parent_tool_use_id: str | None) -> None:
        """One assistant message: counts model requests (sub-agents included) and the served model id."""
        if message_id:
            self._message_ids.add(message_id)
        else:
            self._anonymous_messages += 1
        if model and parent_tool_use_id is None and model not in self.main_models:
            self.main_models.append(model)

    @property
    def model_calls(self) -> int | None:
        n = len(self._message_ids) + self._anonymous_messages
        return n or None

    def on_tool_result(self, tool_use_id: str, is_error: bool | None) -> None:
        """A tool result block in the message stream; ``is_error`` marks a failed call (e.g. non-zero exit)."""
        if is_error is not None:
            self.set_outcome(tool_use_id, not is_error)

    def _reached(self, name: str, tool_input: dict[str, Any]) -> tuple[str, str] | None:
        if name == "Skill":
            skill = _skill_name(tool_input)
            if skill in self.harness.skills:
                return self.harness.skill_path(skill), f"Skill tool invoked: {skill}"
            return None
        if name == "Read":
            rel = _rel(str(tool_input.get("file_path", "")), self.workdir)
            prefix = SKILLS_DIR.as_posix() + "/"
            if rel.startswith(prefix):
                inner = rel[len(prefix) :]
                skill = inner.split("/", 1)[0]
                if skill in self.harness.skills:
                    return f"{self.harness.config.skills_dir}/{inner}", "skill file read with the Read tool"
        return None


def result_usage(msg: Any, model: str, tool_calls: int, wall_s: float, model_calls: int | None = None) -> Usage:
    """Build Usage from a ResultMessage (its cost is the SDK's client-side estimate)."""
    u = dict(getattr(msg, "usage", None) or {})
    usage = Usage(
        input_tokens=u.get("input_tokens"),
        output_tokens=u.get("output_tokens"),
        cache_read_input_tokens=u.get("cache_read_input_tokens"),
        cache_creation_input_tokens=u.get("cache_creation_input_tokens"),
        cost_usd=getattr(msg, "total_cost_usd", None),
        wall_s=round(wall_s, 3),
        tool_calls=tool_calls,
        num_turns=getattr(msg, "num_turns", None),
        model_calls=model_calls,
        attempts=1,
    )
    if not usage.cost_usd:
        try:
            usage.cost_usd = round(cost_from_usage(model, usage), 6)
        except KeyError:
            usage.cost_usd = None
    return usage


def served_models(rec: TraceRecorder, result: Any) -> list[str] | None:
    """Exact model ids that served the main loop; falls back to the result's per-model usage keys."""
    if rec.main_models:
        return list(rec.main_models)
    mu = getattr(result, "model_usage", None) or {}
    return sorted(mu) or None


def decoding_of(cfg: Any, rec: TraceRecorder) -> dict[str, Any]:
    """Generation settings in effect. The SDK exposes effort and thinking; sampling stays at the runtime default."""
    return {
        "effort": cfg.effort or "runtime default",
        "effort_in_effect": sorted(rec.effort_levels) or None,
        "thinking": cfg.thinking or "runtime default",
        "temperature": "runtime default (not settable through the SDK)",
        "max_turns": cfg.max_turns,
        "fallback_model": None,
    }


def status_of(msg: Any) -> str:
    sub = getattr(msg, "subtype", "") or ""
    if sub == "error_max_budget_usd":
        return "budget_exceeded"
    if sub == "error_max_turns":
        return "max_turns"
    if getattr(msg, "is_error", False) or sub != "success":
        return "error"
    return "ok"


class ClaudeAgentSDKAdapter:
    """Runs a task with ``claude_agent_sdk.query`` and records activation events via hooks."""

    id = "claude_agent_sdk/1"

    def installed_paths(self) -> list[str]:
        return [".claude"]

    def preflight(self) -> None:
        """Fail before any run if the SDK or the API key is missing."""
        try:
            import claude_agent_sdk  # noqa: F401
        except ImportError as e:
            raise FatalAdapterError("claude-agent-sdk is not installed: pip install 'harness-audit[claude]'") from e
        if not os.environ.get(API_KEY_VAR):
            raise FatalAdapterError(f"{API_KEY_VAR} is not set; harness-audit only runs the Claude API with an API key")

    def install(self, harness: Harness, workdir: Path) -> None:
        """Copy the harness's skills into the workdir where the CLI discovers project skills."""
        for name in harness.skills:
            src = harness.root / harness.config.skills_dir / name
            dst = workdir / SKILLS_DIR / name
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def options(self, req: RunRequest, recorder: TraceRecorder, config_dir: Path | None = None) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        h = req.harness
        cfg = h.config
        tools = list(cfg.allowed_tools)
        if h.skills and "Skill" not in tools:
            tools.append("Skill")
        if cfg.use_claude_code_preset:
            system_prompt: Any = {"type": "preset", "preset": "claude_code"}
            if h.system_prompt:
                system_prompt["append"] = h.system_prompt
        else:
            system_prompt = h.system_prompt
        kwargs: dict[str, Any] = dict(
            system_prompt=system_prompt,
            tools=tools,
            allowed_tools=tools,
            permission_mode="dontAsk",
            setting_sources=["project"],
            skills=list(h.skills),
            cwd=str(req.workdir),
            model=req.model,
            max_turns=cfg.max_turns,
            max_budget_usd=req.max_budget_usd,
            hooks={
                "PreToolUse": [HookMatcher(matcher=None, hooks=[recorder.pre_tool_use])],
                "PostToolUse": [HookMatcher(matcher=None, hooks=[recorder.post_tool_use])],
                "PostToolUseFailure": [HookMatcher(matcher=None, hooks=[recorder.post_tool_use_failure])],
            },
            env=child_env(config_dir or req.workdir.parent / "claude-config"),
        )
        if cfg.effort:
            kwargs["effort"] = cfg.effort
        if cfg.thinking:
            kwargs["thinking"] = {"type": cfg.thinking}
        return ClaudeAgentOptions(**kwargs)

    async def run(self, req: RunRequest) -> AdapterResult:
        try:
            from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, ToolResultBlock, UserMessage, query
            from claude_agent_sdk import __version__ as sdk_version
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("claude-agent-sdk is not installed: pip install 'harness-audit[claude]'") from e

        self.install(req.harness, req.workdir)
        rec = TraceRecorder(req.harness, req.workdir)
        rec.on_start()
        config_dir = Path(tempfile.mkdtemp(prefix="harness-audit-claude-config-"))
        options = self.options(req, rec, config_dir)
        result: Any = None
        model_seen: str | None = None

        async def consume() -> None:
            nonlocal result, model_seen
            async for msg in query(prompt=req.task.spec.prompt, options=options):
                if isinstance(msg, SystemMessage) and msg.subtype == "init":
                    source = msg.data.get("apiKeySource")
                    if source is not None and source != API_KEY_VAR:
                        raise FatalAdapterError(
                            f"the Claude CLI reports apiKeySource={source!r}, not {API_KEY_VAR}; refusing to run"
                        )
                    rec.on_init(msg.data)
                    model_seen = msg.data.get("model") or model_seen
                elif isinstance(msg, AssistantMessage):
                    rec.on_assistant(msg.model, msg.message_id, msg.parent_tool_use_id)
                elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, ToolResultBlock):
                            rec.on_tool_result(block.tool_use_id, block.is_error)
                elif isinstance(msg, ResultMessage):
                    result = msg

        t0 = time.monotonic()
        status, error = "ok", None
        try:
            await asyncio.wait_for(consume(), timeout=req.timeout_s)
        except asyncio.TimeoutError:
            status, error = "timeout", f"run exceeded {req.timeout_s}s"
        except FatalAdapterError:
            raise
        except Exception as e:  # noqa: BLE001 - recorded as a failed run, never re-raised into the runner
            status, error = "error", f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            shutil.rmtree(config_dir, ignore_errors=True)
        wall = time.monotonic() - t0
        if result is not None:
            usage = result_usage(result, req.model, rec.tool_calls, wall, rec.model_calls)
            if status == "ok":
                status = status_of(result)
                if status == "error":
                    error = "; ".join(getattr(result, "errors", None) or []) or getattr(result, "subtype", "error")
        else:
            usage = Usage(tool_calls=rec.tool_calls, wall_s=round(wall, 3), model_calls=rec.model_calls, attempts=1)
        if rec.external_skills:
            rec.events.append(
                Event(t=0.0, type="external_skills", evidence="skills not from this harness: " + ", ".join(rec.external_skills))
            )
        runtime = "; ".join(x for x in (rec.runtime, f"claude-agent-sdk/{sdk_version}") if x)
        return AdapterResult(
            events=rec.events,
            usage=usage,
            status=status,
            error=error,
            model=model_seen or req.model,
            model_versions=served_models(rec, result),
            decoding=decoding_of(req.harness.config, rec),
            runtime=runtime,
        )

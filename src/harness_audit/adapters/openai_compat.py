"""OpenAI-compatible adapter: a minimal tool-using agent loop over ``/chat/completions``.

For gateways that expose models through the OpenAI chat-completions format (for example
Token Harbor at ``https://tokenharbor.ai/v1``, ``Authorization: Bearer <key>``). This is a
harness of its own, not Claude Code: a fixed loop that offers the harness's allowed tools
(Read, Edit, Write, Bash, Glob, Grep) and, when the harness has skills, a ``Skill`` tool whose
names and descriptions are listed in the system prompt, as Claude Code lists them.

Credentials: the key is read from the environment variable named by ``api_key_env``
(default ``TOKENHARBOR_API_KEY``) and is only ever placed in the ``Authorization`` header.
It is never logged, stored or passed to the agent's Bash commands, which run with a minimal
environment. File tools are confined to the run's working directory; Bash runs there too
but is not sandboxed (see the README's known limitations).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..cost import PRICES, cost_from_usage
from ..harness import Harness
from ..schema import Event, Usage
from .base import AdapterResult, FatalAdapterError, RunRequest
from .claude_agent_sdk import SKILLS_DIR, _args_hash, _rel, summarize

DEFAULT_BASE_URL_ENV = "TOKENHARBOR_BASE_URL"
DEFAULT_BASE_URL = "https://tokenharbor.ai/v1"
DEFAULT_KEY_ENV = "TOKENHARBOR_API_KEY"
OUTPUT_MAX = 8000
RUNTIME = "harness-audit-openai-loop/1"

TOOL_SPECS: dict[str, dict[str, Any]] = {
    "Read": {
        "description": "Read a text file in the working directory. Returns numbered lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "path relative to the working directory"},
                "offset": {"type": "integer", "description": "first line (1-based), optional"},
                "limit": {"type": "integer", "description": "number of lines, optional"},
            },
            "required": ["file_path"],
        },
    },
    "Edit": {
        "description": "Replace an exact string in a file. old_string must occur exactly once unless replace_all.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    "Write": {
        "description": "Create or overwrite a file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["file_path", "content"],
        },
    },
    "Bash": {
        "description": "Run a shell command in the working directory. Returns exit code and output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "seconds, default 60, max 300"},
            },
            "required": ["command"],
        },
    },
    "Glob": {
        "description": "List files matching a glob pattern (e.g. '**/*.py') in the working directory.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    },
    "Grep": {
        "description": "Search file contents with a regular expression. Returns path:line:text matches.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "glob": {"type": "string", "description": "file glob filter, default '**/*'"},
            },
            "required": ["pattern"],
        },
    },
    "Skill": {
        "description": "Load the full instructions of one of the available skills by name.",
        "parameters": {"type": "object", "properties": {"skill": {"type": "string"}}, "required": ["skill"]},
    },
}


class ToolError(Exception):
    """A tool call that failed; its message is returned to the model."""


def _frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    out: dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def bash_env(workdir: Path) -> dict[str, str]:
    """Minimal environment for the agent's commands: no credentials of any kind."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


class Tools:
    """Tool implementations confined to one working directory."""

    def __init__(self, workdir: Path, harness: Harness) -> None:
        self.workdir = workdir.resolve()
        self.harness = harness

    def _path(self, p: str) -> Path:
        if not p:
            raise ToolError("file_path is required")
        q = Path(p)
        q = (q if q.is_absolute() else self.workdir / q).resolve()
        if q != self.workdir and self.workdir not in q.parents:
            raise ToolError(f"{p}: outside the working directory")
        return q

    def Read(self, file_path: str, offset: int | None = None, limit: int | None = None) -> str:
        q = self._path(file_path)
        if not q.is_file():
            raise ToolError(f"{file_path}: no such file")
        lines = q.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max((offset or 1) - 1, 0)
        end = start + limit if limit else len(lines)
        return "\n".join(f"{i + 1:6}\t{line}" for i, line in enumerate(lines[start:end], start=start)) or "(empty file)"

    def Edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        q = self._path(file_path)
        if not q.is_file():
            raise ToolError(f"{file_path}: no such file")
        text = q.read_text(encoding="utf-8")
        n = text.count(old_string)
        if n == 0:
            raise ToolError("old_string not found")
        if n > 1 and not replace_all:
            raise ToolError(f"old_string occurs {n} times; make it unique or set replace_all")
        q.write_text(
            text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1), encoding="utf-8"
        )
        return f"edited {file_path}"

    def Write(self, file_path: str, content: str) -> str:
        q = self._path(file_path)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(content, encoding="utf-8")
        return f"wrote {file_path} ({len(content)} chars)"

    def Bash(self, command: str, timeout: int | None = None) -> str:
        t = min(max(int(timeout or 60), 1), 300)
        try:
            p = subprocess.run(
                ["bash", "-c", command], cwd=self.workdir, env=bash_env(self.workdir), capture_output=True, text=True, timeout=t
            )
        except subprocess.TimeoutExpired as e:
            raise ToolError(f"timed out after {t}s") from e
        out = (p.stdout + ("\n" + p.stderr if p.stderr else "")).strip()
        if len(out) > OUTPUT_MAX:
            out = out[:OUTPUT_MAX] + "\n…(truncated)"
        msg = f"exit code {p.returncode}\n{out}"
        if p.returncode != 0:
            raise ToolError(msg)
        return msg

    def Glob(self, pattern: str) -> str:
        hits = sorted(str(p.relative_to(self.workdir)) for p in self.workdir.glob(pattern) if p.is_file())
        hits = [h for h in hits if not h.startswith(".claude/")]
        return "\n".join(hits[:200]) or "(no matches)"

    def Grep(self, pattern: str, glob: str | None = None) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"invalid regex: {e}") from e
        out: list[str] = []
        for p in sorted(self.workdir.glob(glob or "**/*")):
            if not p.is_file() or ".claude" in p.parts:
                continue
            rel = p.relative_to(self.workdir)
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    out.append(f"{rel}:{i}:{line}")
                    if len(out) >= 200:
                        return "\n".join(out)
        return "\n".join(out) or "(no matches)"

    def Skill(self, skill: str) -> str:
        name = skill.strip().lstrip("/").split()[0] if skill.strip() else ""
        if name not in self.harness.skills:
            raise ToolError(f"unknown skill {skill!r}; available: {', '.join(self.harness.skills) or 'none'}")
        return (self.workdir / SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")


class OpenAICompatAdapter:
    """Runs a task with a minimal chat-completions agent loop and records activation events."""

    id = "openai_compat/1"

    def __init__(
        self,
        base_url: str | None = None,
        api_key_env: str = DEFAULT_KEY_ENV,
        request_timeout_s: float = 120.0,
        max_output_tokens: int = 4096,
        temperature: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(DEFAULT_BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
        self.api_key_env = api_key_env
        self.request_timeout_s = request_timeout_s
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def installed_paths(self) -> list[str]:
        return [".claude"]

    def preflight(self) -> None:
        if not os.environ.get(self.api_key_env):
            raise FatalAdapterError(f"{self.api_key_env} is not set (it is read from the environment only)")

    # ---- HTTP -------------------------------------------------------------------------------
    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """POST /chat/completions with retries on 429/5xx; returns (response, retries)."""
        body = json.dumps(payload).encode()
        retries = 0
        while True:
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get(self.api_key_env, "")},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout_s) as r:
                    return json.loads(r.read().decode()), retries
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in (401, 403):
                    raise FatalAdapterError(f"gateway refused the key ({e.code}): {detail}") from None
                if e.code in (429, 500, 502, 503, 504) and retries < 3:
                    retries += 1
                    time.sleep(2**retries)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {detail}") from None
            except urllib.error.URLError as e:
                if retries < 3:
                    retries += 1
                    time.sleep(2**retries)
                    continue
                raise RuntimeError(f"network error: {e.reason}") from None

    # ---- run --------------------------------------------------------------------------------
    def install(self, harness: Harness, workdir: Path) -> None:
        for name in harness.skills:
            shutil.copytree(harness.root / harness.config.skills_dir / name, workdir / SKILLS_DIR / name, dirs_exist_ok=True)

    def system_prompt(self, harness: Harness) -> str:
        parts = [harness.system_prompt or "You are a coding agent. Use the tools to complete the task."]
        if harness.skills:
            lines = [
                "",
                "# Skills",
                "Skills are instructions for specific kinds of work. When one applies, call the "
                "Skill tool with its name before starting, then follow it.",
                "",
            ]
            for name in harness.skills:
                fm = _frontmatter((harness.root / harness.config.skills_dir / name / "SKILL.md").read_text(encoding="utf-8"))
                lines.append(f"- {fm.get('name', name)}: {fm.get('description', '')}")
            parts.append("\n".join(lines))
        return "\n".join(parts)

    def tools_for(self, harness: Harness) -> list[dict[str, Any]]:
        names = [t for t in harness.config.allowed_tools if t in TOOL_SPECS and t != "Skill"]
        if harness.skills:
            names.append("Skill")
        return [{"type": "function", "function": {"name": n, **TOOL_SPECS[n]}} for n in names]

    async def run(self, req: RunRequest) -> AdapterResult:
        return await asyncio.to_thread(self._run_sync, req)

    def _run_sync(self, req: RunRequest) -> AdapterResult:
        h = req.harness
        self.install(h, req.workdir)
        tools = Tools(req.workdir, h)
        t0 = time.monotonic()
        events: list[Event] = []
        if h.system_prompt is not None and h.config.system_prompt_file:
            p = h.config.system_prompt_file
            events.append(
                Event(t=0.0, type="artifact_loaded", artifact=p, hash=h.files.get(p), evidence="passed as the system prompt")
            )
        for name in h.skills:
            path = h.skill_path(name)
            events.append(
                Event(
                    t=0.0, type="artifact_loaded", artifact=path, hash=h.files.get(path), evidence="listed in the system prompt"
                )
            )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt(h)},
            {"role": "user", "content": req.task.spec.prompt},
        ]
        specs = self.tools_for(h)
        usage = Usage(input_tokens=0, output_tokens=0, num_turns=0, tool_calls=0, model_calls=0, attempts=1)
        usage_cost = 0.0
        cost_reported = True
        served: list[str] = []
        http_retries = 0
        status, error = "ok", None
        try:
            for _ in range(h.config.max_turns):
                if time.monotonic() - t0 > req.timeout_s:
                    status, error = "timeout", f"run exceeded {req.timeout_s}s"
                    break
                payload: dict[str, Any] = {
                    "model": req.model,
                    "messages": messages,
                    "tools": specs,
                    "max_tokens": self.max_output_tokens,
                }
                if self.temperature is not None:
                    payload["temperature"] = self.temperature
                resp, n_retry = self._post(payload)
                http_retries += n_retry
                usage.model_calls = (usage.model_calls or 0) + 1
                usage.num_turns = (usage.num_turns or 0) + 1
                u = resp.get("usage") or {}
                usage.input_tokens = (usage.input_tokens or 0) + int(u.get("prompt_tokens") or 0)
                usage.output_tokens = (usage.output_tokens or 0) + int(u.get("completion_tokens") or 0)
                if isinstance(u.get("cost"), (int, float)):
                    usage_cost += float(u["cost"])
                else:
                    cost_reported = False
                if resp.get("model") and resp["model"] not in served:
                    served.append(resp["model"])
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                calls = msg.get("tool_calls") or []
                messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")} | {"role": "assistant"})
                if not calls:
                    break
                for call in calls:
                    messages.append(self._do_call(call, tools, events, t0, h, req.workdir))
                    usage.tool_calls = (usage.tool_calls or 0) + 1
                spent = usage_cost if cost_reported else self._cost(req.model, usage)
                if spent is not None and spent >= req.max_budget_usd:
                    status, error = "budget_exceeded", f"spent ${spent:.4f} of ${req.max_budget_usd:.4f}"
                    break
            else:
                status = "max_turns"
        except FatalAdapterError:
            raise
        except Exception as e:  # noqa: BLE001 - recorded as a failed run
            status, error = "error", f"{type(e).__name__}: {str(e)[:300]}"
        usage.wall_s = round(time.monotonic() - t0, 3)
        usage.cost_usd = round(usage_cost, 6) if cost_reported and usage.model_calls else self._cost(req.model, usage)
        usage.http_retries = http_retries  # type: ignore[attr-defined]
        decoding = {
            "temperature": self.temperature if self.temperature is not None else "runtime default",
            "max_output_tokens": self.max_output_tokens,
            "max_turns": h.config.max_turns,
            "effort": h.config.effort or "runtime default",
        }
        return AdapterResult(
            events=events,
            usage=usage,
            status=status,
            error=error,
            model=req.model,
            model_versions=served or None,
            decoding=decoding,
            runtime=RUNTIME,
        )

    @staticmethod
    def _cost(model: str, usage: Usage) -> float | None:
        if model not in PRICES:
            return None
        return round(cost_from_usage(model, usage), 6)

    def _do_call(
        self, call: dict[str, Any], tools: Tools, events: list[Event], t0: float, h: Harness, workdir: Path
    ) -> dict[str, Any]:
        fn = call.get("function") or {}
        name = str(fn.get("name", ""))
        try:
            args = json.loads(fn.get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except ValueError as e:
            args, bad = {}, f"invalid JSON arguments: {e}"
        else:
            bad = None
        t = round(time.monotonic() - t0, 3)
        ev = Event(t=t, type="tool_call", name=name, args_hash=_args_hash(args), summary=summarize(name, args, workdir))
        events.append(ev)
        reached = self._reached(name, args, h, workdir)
        if reached:
            events.append(
                Event(t=t, type="artifact_reached", artifact=reached[0], hash=h.files.get(reached[0]), evidence=reached[1])
            )
        allowed = {s["function"]["name"] for s in self.tools_for(h)}
        try:
            if bad:
                raise ToolError(bad)
            if name not in allowed:
                raise ToolError(f"unknown or disallowed tool {name!r}")
            out = getattr(tools, name)(**args)
            ev.ok = True
        except ToolError as e:
            out, ev.ok = f"Error: {e}", False
        except TypeError as e:
            out, ev.ok = f"Error: bad arguments: {e}", False
        except Exception as e:  # noqa: BLE001 - a tool failure goes back to the model
            out, ev.ok = f"Error: {type(e).__name__}: {e}", False
        return {"role": "tool", "tool_call_id": call.get("id", ""), "content": out}

    @staticmethod
    def _reached(name: str, args: dict[str, Any], h: Harness, workdir: Path) -> tuple[str, str] | None:
        if name == "Skill":
            skill = str(args.get("skill", "")).strip().lstrip("/").split(" ")[0]
            if skill in h.skills:
                return h.skill_path(skill), f"Skill tool invoked: {skill}"
        if name == "Read":
            rel = _rel(str(args.get("file_path", "")), workdir)
            prefix = SKILLS_DIR.as_posix() + "/"
            if rel.startswith(prefix):
                skill = rel[len(prefix) :].split("/", 1)[0]
                if skill in h.skills:
                    return f"{h.config.skills_dir}/{rel[len(prefix) :]}", "skill file read with the Read tool"
        return None

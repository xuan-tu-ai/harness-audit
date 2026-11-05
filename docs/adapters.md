# Writing an adapter

An adapter runs **one task under one harness** and reports what happened. The runner does everything
else: planning, the working directory, caching, the budget guard, scoring with the out-of-harness
evaluator, and the audit. The core (`schema`, `claims`, `stats`, `report`) never imports an agent
framework, so an adapter is the only place framework code lives.

If your framework already logs runs, you may not need an adapter at all: export one JSONL line per run
(see the README) and use `harness-audit audit`.

## Interface

```python
from harness_audit.adapters.base import AdapterResult, FatalAdapterError, RunRequest
from harness_audit.schema import Event, Usage


class MyAdapter:
    id = "my-framework/1"  # part of the cache key: bump it when the adapter's behaviour changes

    def installed_paths(self) -> list[str]:
        """Workdir-relative paths you create (e.g. config dirs); removed before scoring."""
        return [".my-framework"]

    def preflight(self) -> None:  # optional
        """Raise FatalAdapterError if no run can succeed (missing credentials, missing package)."""

    async def run(self, req: RunRequest) -> AdapterResult:
        # req.task.spec.prompt, req.harness (root, config, files, system_prompt, skills),
        # req.workdir (a fresh copy of the task workspace), req.model, req.max_budget_usd, req.timeout_s
        ...
        return AdapterResult(events=events, usage=usage, status="ok", model=req.model)
```

`run` must not raise for an ordinary failure of the agent: return `status="error"` (or `timeout`,
`max_turns`, `budget_exceeded`) and whatever usage you have, so one bad run does not stop the experiment.
Raise `FatalAdapterError` only for problems that would affect every run.

Point an experiment at your adapter with a `module:Class` path:

```yaml
# experiment.yaml
adapter: my_package.my_adapter:MyAdapter
```

## Events

Activation is judged from events, so they decide what harness-audit can conclude:

| Event | Emit when | Grade |
|---|---|---|
| `artifact_loaded` | a harness artifact entered the run (a skill was registered, a prompt was set, a tool was mounted) | available |
| `artifact_reached` | the changed artifact's path executed (the skill was invoked, the tool was called, the changed code ran) | reached |
| `artifact_carried` | its effect reached a later use (its output entered the next model input, a downstream decision followed it) | carried |
| `tool_call` | every tool call: `name`, `args_hash`, an optional short `summary`, and `ok` (did the call succeed; omit if unknown) | — |
| `rubric_judgment` | an external judge decided a `check: judge` rubric item: `artifact`, `name` = item id, `ok` = met | carried (adherence) |

Use the artifact's path relative to the harness root (`skills/foo/SKILL.md`) as `artifact`, and its
content hash as `hash` when you have it. Keep `summary` free of file contents and secrets: it is the
command for a shell call and the path for a file tool. harness-audit reads the artifact's adherence
rubric (or its trace `signatures`) against these calls to derive *carried*, counting from the moment the
artifact was reached. Rubric items that depend on a failure (`on_failure_of`) or on success (`ok: true`)
need `ok`; without it they stay *n/a* or *unmet*. The Claude adapter fills `ok` from the `PostToolUse`
and `PostToolUseFailure` hooks and from `ToolResultBlock.is_error`.

Load events alone grade as *available*, which the rubric treats as the weakest evidence. If your framework
cannot observe execution of the changed path, say so in the report rather than emitting `artifact_reached`
by construction.

## Usage

Fill `Usage` from the framework's own accounting when possible: `input_tokens`, `output_tokens`,
`cache_read_input_tokens`, `cache_creation_input_tokens`, `cost_usd`, `num_turns`, `tool_calls`,
`wall_s`. Attribution compares output tokens, turns and tool calls between arms; a missing value makes
the envelope check partial, not passing. If `cost_usd` is `None` the budget guard counts the whole per-run
reservation as spent.

## Testing

`tests/fake_adapter.py` is a complete adapter without a model, and `tests/test_claude_adapter.py` tests
the Claude Agent SDK adapter offline by calling its hook and replacing `query()` with a generator of SDK
message objects. The same pattern works for any framework.

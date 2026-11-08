"""Adapters run one agent task under one harness and report events and usage."""

from __future__ import annotations

from .base import Adapter, AdapterResult, FatalAdapterError, RunRequest

ADAPTERS = {
    "claude_agent_sdk": "harness_audit.adapters.claude_agent_sdk:ClaudeAgentSDKAdapter",
    "openai_compat": "harness_audit.adapters.openai_compat:OpenAICompatAdapter",
}


def load_adapter(name: str, options: dict | None = None) -> Adapter:
    """Instantiate an adapter by registry name or ``module:Class`` path, with keyword ``options``."""
    import importlib

    target = ADAPTERS.get(name, name)
    if ":" not in target:
        raise ValueError(f"unknown adapter {name!r}; known: {sorted(ADAPTERS)} or 'module:Class'")
    mod, cls = target.split(":", 1)
    return getattr(importlib.import_module(mod), cls)(**(options or {}))


__all__ = ["ADAPTERS", "Adapter", "AdapterResult", "FatalAdapterError", "RunRequest", "load_adapter"]

"""A harness is a directory: ``harness.yaml``, an optional system prompt, skills, other files.

The harness hash covers every file in the directory, so any change to the
surface the agent runs with changes the hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .hashing import dir_file_hashes, dir_hash
from .io import InputError
from .schema import ChangedArtifact, Signature

ENVELOPE_KEYS = ("max_turns", "effort", "thinking")
"""Config keys that change the inference envelope: a difference is a confound, not the update."""


class HarnessConfig(BaseModel):
    """Contents of ``harness.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str = "harness"
    system_prompt_file: str | None = "system_prompt.md"
    use_claude_code_preset: bool = False
    allowed_tools: list[str] = Field(default_factory=lambda: ["Read", "Edit", "Write", "Bash", "Glob", "Grep"])
    max_turns: int = 20
    effort: str | None = None
    thinking: str | None = None
    skills_dir: str = "skills"
    signatures: dict[str, list[Signature]] = Field(default_factory=dict)
    applies_to: dict[str, list[str]] = Field(
        default_factory=dict, description="artifact path -> task ids or groups (globs) where the update is meant to apply"
    )


@dataclass
class Harness:
    root: Path
    config: HarnessConfig
    hash: str
    files: dict[str, str]
    system_prompt: str | None
    skills: list[str] = field(default_factory=list)

    def skill_path(self, name: str) -> str:
        return f"{self.config.skills_dir}/{name}/SKILL.md"


def load_harness(root: str | Path) -> Harness:
    """Load and hash a harness directory."""
    base = Path(root)
    cfg_path = base / "harness.yaml"
    if not cfg_path.exists():
        raise InputError(f"{base}: missing harness.yaml")
    try:
        cfg = HarnessConfig.model_validate(yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {})
    except (yaml.YAMLError, ValidationError) as e:
        raise InputError(f"{cfg_path}: {e}") from e
    prompt = None
    if cfg.system_prompt_file:
        sp = base / cfg.system_prompt_file
        if not sp.exists():
            raise InputError(f"{base}: system_prompt_file {cfg.system_prompt_file!r} not found")
        prompt = sp.read_text(encoding="utf-8")
    skills_root = base / cfg.skills_dir
    skills = sorted(p.parent.name for p in skills_root.glob("*/SKILL.md")) if skills_root.is_dir() else []
    return Harness(base, cfg, dir_hash(base), dir_file_hashes(base), prompt, skills)


def _kind(h: Harness, path: str) -> str:
    if path.startswith(h.config.skills_dir + "/"):
        return "skill" if path.endswith("SKILL.md") else "file"
    if h.config.system_prompt_file and path == h.config.system_prompt_file:
        return "system_prompt"
    return "file"


def diff_harnesses(a: Harness, b: Harness) -> tuple[list[ChangedArtifact], list[str]]:
    """Changed artifacts between harness directories, and envelope confounds from ``harness.yaml``."""
    arts: list[ChangedArtifact] = []
    for path in sorted(set(a.files) | set(b.files)):
        if path == "harness.yaml":
            continue
        ha, hb = a.files.get(path), b.files.get(path)
        if ha == hb:
            continue
        change = "added" if ha is None else "removed" if hb is None else "modified"
        sigs = b.config.signatures.get(path, [])
        scope = b.config.applies_to.get(path)
        arts.append(
            ChangedArtifact(
                path=path,
                kind=_kind(b if hb else a, path),  # type: ignore[arg-type]
                change=change,  # type: ignore[arg-type]
                hash_a=ha,
                hash_b=hb,
                signatures=sigs,
                applies_to=scope,
            )
        )
    confounds: list[str] = []
    ca = a.config.model_dump(exclude={"name", "signatures", "applies_to"})
    cb = b.config.model_dump(exclude={"name", "signatures", "applies_to"})
    for key in sorted(ca):
        if ca[key] == cb[key]:
            continue
        if key in ENVELOPE_KEYS:
            confounds.append(f"{key}: {ca[key]!r} -> {cb[key]!r}")
        else:
            arts.append(ChangedArtifact(path=f"harness.yaml:{key}", kind="config", change="modified"))
    return arts, confounds

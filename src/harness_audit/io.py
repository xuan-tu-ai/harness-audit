"""Reading and writing run records (JSONL) and manifests (YAML/JSON)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import Manifest, RunRecord


class InputError(Exception):
    """Raised for unreadable or invalid input files, with a location in the message."""


def _fmt_validation(e: ValidationError) -> str:
    parts = []
    for err in e.errors()[:5]:
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    more = f" (+{len(e.errors()) - 5} more)" if len(e.errors()) > 5 else ""
    return "; ".join(parts) + more


def read_runs(path: str | Path) -> list[RunRecord]:
    """Parse a JSONL file of RunRecords; blank lines and ``#`` comments are skipped."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"{p}: file not found")
    records: list[RunRecord] = []
    run_ids: set[str] = set()
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise InputError(f"{p}:{i}: invalid JSON ({e.msg})") from e
        try:
            rec = RunRecord.model_validate(obj)
        except ValidationError as e:
            raise InputError(f"{p}:{i}: {_fmt_validation(e)}") from e
        if rec.run_id in run_ids:
            raise InputError(f"{p}:{i}: duplicate run_id {rec.run_id!r}")
        run_ids.add(rec.run_id)
        records.append(rec)
    if not records:
        raise InputError(f"{p}: no run records")
    return records


def write_runs(path: str | Path, records: Iterable[RunRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json(exclude_none=True) + "\n")


def read_manifest(path: str | Path | None) -> Manifest:
    """Load a manifest from YAML or JSON; ``None`` returns the defaults."""
    if path is None:
        return Manifest()
    p = Path(path)
    if not p.exists():
        raise InputError(f"{p}: file not found")
    try:
        obj = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise InputError(f"{p}: invalid YAML ({e})") from e
    if not isinstance(obj, dict):
        raise InputError(f"{p}: manifest must be a mapping")
    try:
        return Manifest.model_validate(obj)
    except ValidationError as e:
        raise InputError(f"{p}: {_fmt_validation(e)}") from e


def write_manifest(path: str | Path, manifest: Manifest) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(manifest.model_dump_json(exclude_none=True))
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

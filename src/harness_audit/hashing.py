"""Content hashes for files and directories (the harness surface, tasks, evaluators)."""

from __future__ import annotations

import hashlib
from pathlib import Path

IGNORED = {"__pycache__", ".DS_Store", ".pytest_cache", ".harness-audit-cache"}


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_hash(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def iter_files(root: str | Path) -> list[Path]:
    """All regular files under ``root`` (sorted, ignoring caches), as paths relative to ``root``."""
    base = Path(root)
    out = []
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(base)
        if any(part in IGNORED for part in rel.parts) or not p.is_file() or p.suffix == ".pyc":
            continue
        out.append(rel)
    return out


def dir_file_hashes(root: str | Path) -> dict[str, str]:
    """``{relative posix path: sha256}`` for every file under ``root``."""
    base = Path(root)
    return {rel.as_posix(): file_hash(base / rel) for rel in iter_files(base)}


def dir_hash(root: str | Path, extra: bytes = b"") -> str:
    """Hash of a whole directory: sorted (relative path, content hash) pairs, plus optional extra bytes."""
    h = hashlib.sha256()
    for rel, fh in sorted(dir_file_hashes(root).items()):
        h.update(rel.encode() + b"\0" + fh.encode() + b"\n")
    h.update(extra)
    return "sha256:" + h.hexdigest()

"""Durable JSON store for goal graphs.

A graph outlives the session that created it, so it does not live in
`session-artifacts/<id>/`, which is deleted with the session. It lives beside
the other agent state and is plain JSON, so another session, a shell command,
or a person can read it without going through the kernel.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

STORE_VERSION = 1
_DEFAULT_DIR_NAME = "goal-graphs"


def _agent_dir() -> Path:
    raw = (
        os.environ.get("PRIME_AGENT_CODING_AGENT_DIR")
        or os.environ.get("PI_CODING_AGENT_DIR")
        or str(Path.home() / ".prime" / "agent")
    )
    return Path(raw).expanduser().resolve()


def default_store_dir() -> Path:
    # Set-but-empty behaves as unset, matching the harness env convention.
    raw = (os.environ.get("RLM_GOAL_GRAPH_DIR") or "").strip()
    root = Path(raw) if raw else _agent_dir() / _DEFAULT_DIR_NAME
    return root.expanduser().resolve()


def slug(raw: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in raw.strip())
    normalized = "-".join(part for part in normalized.split("-") if part)
    if not normalized:
        raise ValueError("graph name must contain at least one alphanumeric character")
    if len(normalized) > 80:
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:8]
        return f"{normalized[:80]}-{digest}"
    return normalized


def graph_path(name: str, store_dir: str | Path | None = None) -> Path:
    root = Path(store_dir).expanduser().resolve() if store_dir is not None else default_store_dir()
    return root / f"{slug(name)}.json"


class GraphStore:
    """Locked, atomically replaced JSON file holding one graph."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")

    @contextmanager
    def locked(self) -> Generator[None, None, None]:
        """Hold an exclusive lock for a read-modify-write.

        The lock is a sidecar file: `write()` replaces the data file, so a lock
        held on the data file itself would survive only on the old inode.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.lock_path, "a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def read(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"graph store {self.path} is not valid JSON: {exc}") from None
        if not isinstance(payload, dict):
            raise ValueError(f"graph store {self.path} must contain an object")
        version = payload.get("version")
        if version != STORE_VERSION:
            raise ValueError(f"graph store {self.path} has unsupported version {version!r}")
        return payload

    def write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_path = tempfile.mkstemp(dir=str(self.path.parent), prefix=f"{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        except BaseException:
            Path(temp_path).unlink(missing_ok=True)
            raise

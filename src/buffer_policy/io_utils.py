"""I/O helpers: JSON metadata, git commit lookup, directory creation."""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    """Return short git SHA, or None if not a git repo / git missing."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def make_metadata(seed: int | None = None, **extras: Any) -> dict[str, Any]:
    """Build a metadata dict with seed, package versions, git commit, timestamp."""
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": seed,
        "git_commit": _git_commit(),
        "packages": {
            name: _pkg_version(name)
            for name in ("numpy", "scipy", "numba", "pandas", "matplotlib")
        },
    }
    for k, v in extras.items():
        meta[k] = asdict(v) if is_dataclass(v) and not isinstance(v, type) else v
    return meta


def save_json(path, obj: Any) -> Path:
    """Save `obj` as pretty JSON.  Creates parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return p


def load_json(path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

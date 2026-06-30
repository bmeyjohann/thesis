from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .io import save_args_json


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
        ).strip()
    except Exception:
        return None


def _git_state(repo_root: Path) -> dict[str, Any]:
    commit = _git_output(repo_root, "rev-parse", "HEAD")
    branch = _git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output(repo_root, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_porcelain_sha1": (
            hashlib.sha1(status.encode("utf-8")).hexdigest() if status is not None else None
        ),
    }


def _args_hash(args: Any) -> str:
    payload = json.dumps(_jsonable(vars(args)), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def save_eval_manifest(
    *,
    output_dir: Path,
    args: Any,
    source: str,
    step_value: int | None,
    layout_hashes: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a compact provenance record next to eval plots.

    This intentionally captures the wrapper/reward/action settings through the full
    parsed args object, plus git state and layout hashes, so a plot can be audited
    without guessing which environment variant produced it.
    """

    repo_root = Path(__file__).resolve().parent.parent
    manifest = {
        "schema_version": 1,
        "created_unix": time.time(),
        "created_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source": str(source),
        "step_value": int(step_value) if step_value is not None else None,
        "cwd": os.getcwd(),
        "repo_root": str(repo_root),
        "python": sys.executable,
        "git": _git_state(repo_root),
        "args_hash": _args_hash(args),
        "args": _jsonable(vars(args)),
        "layout_unique_count": int(len(set(layout_hashes))),
        "layout_total_count": int(len(layout_hashes)),
        "layout_hashes": list(layout_hashes),
        "extra": _jsonable(extra or {}),
    }
    save_args_json(output_dir / "eval_manifest.json", manifest)

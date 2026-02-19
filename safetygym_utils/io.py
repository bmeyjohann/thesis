from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_args_json(path: Path, args_dict: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2)


def load_args_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_find_args_json_from_model(model_path: Path) -> Path | None:
    model_path = model_path.resolve()
    run_dir = model_path.parent
    candidate = run_dir / "args.json"
    if candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "logs" / "safetygym" / run_dir.name / "args.json"
    if candidate.exists():
        return candidate
    return None

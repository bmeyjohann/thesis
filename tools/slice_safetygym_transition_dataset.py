#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.dataset_io import load_transition_dataset, save_transition_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Write the first N rows of a Safety-Gym transition dataset.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max_rows", type=int, required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()
    data = load_transition_dataset(args.input)
    n = int(max(1, args.max_rows))
    metadata = dict(data.get("metadata") or {})
    metadata["sliced_from"] = str(Path(args.input).expanduser())
    metadata["slice_max_rows"] = n
    path = save_transition_dataset(
        path=args.output,
        metadata=metadata,
        observations=np.asarray(data["observations"])[:n],
        actions=np.asarray(data["actions"])[:n],
        next_observations=np.asarray(data["next_observations"])[:n],
        rewards=np.asarray(data["rewards"])[:n],
        dones=np.asarray(data["dones"])[:n],
        truncations=np.asarray(data["truncations"])[:n],
        costs=np.asarray(data["costs"])[:n] if data.get("costs") is not None else None,
        student_actions=np.asarray(data["student_actions"])[:n],
        teacher_intervened=np.asarray(data["teacher_intervened"])[:n],
        episode_ids=np.asarray(data["episode_ids"])[:n],
        episode_steps=np.asarray(data["episode_steps"])[:n],
    )
    print(json.dumps({"output": str(path), "rows": n}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

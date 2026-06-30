#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from safetygym_utils.dataset_io import load_transition_dataset
from safetygym_utils.io import save_args_json
from safetygym_utils.knn_policy import save_knn_policy


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a KNN behavior-cloning Safety-Gym policy from a transition dataset.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--output_dir", default="models/safetygym_knn_bc")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()

    data = load_transition_dataset(args.dataset_path)
    obs = np.asarray(data["observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    if int(args.max_rows) > 0:
        obs = obs[: int(args.max_rows)]
        actions = actions[: int(args.max_rows)]

    out_dir = Path(args.output_dir).expanduser() / str(args.exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_args_json(out_dir / "args.json", vars(args))
    ckpt = save_knn_policy(
        out_dir / "policy.npz",
        observations=obs,
        actions=actions,
        k=int(args.k),
        metadata={"dataset_path": str(args.dataset_path), "rows": int(obs.shape[0])},
    )
    summary = {"checkpoint": str(ckpt), "rows": int(obs.shape[0]), "k": int(args.k)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

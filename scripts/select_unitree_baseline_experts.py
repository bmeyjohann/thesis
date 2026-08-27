#!/usr/bin/env python3
"""Select one common robust supervisor for every physical benchmark cell."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unitree_locomotion_cells import GEOMETRIES, MATERIALS  # noqa: E402


DEFAULT_SUPERVISOR = (
    REPO_ROOT
    / "external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/"
    "supervisor_rough_model9999/model_9999.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal")
    parser.add_argument("--supervisor", type=Path, default=DEFAULT_SUPERVISOR)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    supervisor = args.supervisor.expanduser().resolve()
    if not supervisor.is_file():
        raise FileNotFoundError(supervisor)
    screen_root = args.root / "adaptive_experts/screen"
    selected = {}
    for geometry, material in product(GEOMETRIES, MATERIALS):
        key = f"{geometry}__{material}__seed{args.seed}"
        screen_path = screen_root / f"{key}.json"
        screen = json.loads(screen_path.read_text()) if screen_path.exists() else None
        selected[key] = {
            "checkpoint": str(supervisor),
            "selection_summary": screen,
            "candidate_count": 1,
            "selection_policy": "common_robust_supervisor_without_cell_fine_tuning",
            "known_screened": screen is not None,
        }
    output = args.root / "sequence/evaluation/selected_experts.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

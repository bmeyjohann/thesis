#!/usr/bin/env python3
"""Evaluate all recurrent students after recurrent training completes."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/benjamin/miniconda3/envs/fasttd3/bin/python")
GEOMETRIES = ("flat", "random_rough", "cobblestone", "stairs", "stepping_stones")
MATERIALS = ("rigid", "slippery", "sand_drag")
ARCHITECTURE_MODALITIES = {
    "gru": ("height_scan", "depth", "mono_rgb", "stereo_rgb"),
    "gru_reconstruction": ("depth", "mono_rgb", "stereo_rgb"),
}


def _latest(path: Path) -> Path:
    values = []
    for candidate in path.glob("student_*.pt"):
        match = re.fullmatch(r"student_(\d+)\.pt", candidate.name)
        if match:
            values.append((int(match.group(1)), candidate))
    if not values:
        raise FileNotFoundError(f"No recurrent checkpoint in {path}")
    return max(values)[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal")
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 23, 37, 51, 73))
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    state_path = args.root / "students_recurrent/state.json"
    while not state_path.exists() or json.loads(state_path.read_text()).get("status") not in {"completed", "failed"}:
        time.sleep(args.poll_seconds)
    if json.loads(state_path.read_text()).get("status") != "completed":
        return 2
    output = args.root / "recurrent_matrix_results"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for architecture, modalities in ARCHITECTURE_MODALITIES.items():
        for modality, geometry, material, seed in product(modalities, GEOMETRIES, MATERIALS, args.seeds):
            checkpoint = _latest(args.root / "students_recurrent" / architecture / modality)
            summary = output / f"{architecture}__{modality}__{geometry}__{material}__seed{seed}.json"
            if not summary.exists():
                command = [
                    str(PYTHON), str(REPO_ROOT / "eval_unitree_recurrent_student.py"),
                    "--modality", modality, "--geometry", geometry, "--material", material,
                    "--checkpoint", str(checkpoint), "--summary-file", str(summary),
                    "--seed", str(seed), "--num-envs", str(args.num_envs), "--steps", str(args.steps),
                ]
                with summary.with_suffix(".log").open("w") as stream:
                    result = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    return result.returncode
            rows.append(json.loads(summary.read_text()))
    csv_path = output / "recurrent_transfer_matrix_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

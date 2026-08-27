#!/usr/bin/env python3
"""Select the best shared architecture globally and render all transfer cells."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/benjamin/miniconda3/envs/fasttd3/bin/python")
ARCHITECTURES = ("nomemory", "gru", "gru_reconstruction")
MODALITIES = ("height_scan", "depth", "mono_rgb", "stereo_rgb")
GEOMETRIES = ("flat", "random_rough", "cobblestone", "stairs", "stepping_stones")
MATERIALS = ("rigid", "slippery", "sand_drag")


def _render_case(command: list[str], log_path: Path) -> int:
    with log_path.open("w") as stream:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    csv_path = args.root / "shared_matrix_results/shared_transfer_matrix_results.csv"
    expected_rows = len(ARCHITECTURES) * len(MODALITIES) * len(GEOMETRIES) * len(MATERIALS) * 3
    rows = []
    while len(rows) < expected_rows:
        if csv_path.exists():
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
        if len(rows) < expected_rows:
            time.sleep(args.poll_seconds)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["architecture"]].append(row)
    # A fall is strongly undesirable, while tracking error prevents a stationary
    # policy from winning merely by staying upright.
    scores = {
        architecture: sum(
            float(row["mean_velocity_tracking_error"])
            + 0.05 * float(row["fall_events_per_1000_env_steps"])
            for row in grouped[architecture]
        ) / len(grouped[architecture])
        for architecture in ARCHITECTURES
    }
    selected_architecture = min(scores, key=scores.get)
    state = json.loads((args.root / "students_shared/state.json").read_text())
    checkpoint = Path(state["architectures"][selected_architecture]["latest_checkpoint"])
    output = args.root / "shared_matrix_results/videos"
    manifest = {
        "selection_rule": "lowest mean(tracking_error + 0.05 * falls_per_1000_steps)",
        "architecture_scores": scores,
        "selected_architecture": selected_architecture,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "videos": [],
    }
    cases = []
    for geometry, material in product(GEOMETRIES, MATERIALS):
        command = [
            str(PYTHON), str(REPO_ROOT / "render_unitree_shared_cell_modalities.py"),
            "--geometry", geometry, "--material", material,
            "--checkpoint", str(checkpoint), "--output-root", str(output),
            "--seed", str(args.seed), "--steps", str(args.steps),
        ]
        cases.append((geometry, material, command))

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_render_case, command, output / f"cell__{geometry}__{material}.log"): (
                geometry, material
            )
            for geometry, material, command in cases
        }
        for future in as_completed(futures):
            geometry, material = futures[future]
            returncode = future.result()
            if returncode:
                return returncode
            for modality in MODALITIES:
                video_dir = output / modality / f"{geometry}__{material}"
                summary = video_dir / "summary.json"
                videos = sorted(str(path.resolve()) for path in video_dir.glob("*.mp4"))
                manifest["videos"].append({
                    "modality": modality, "geometry": geometry, "material": material,
                    "summary": str(summary.resolve()), "video_files": videos,
                })
            manifest["videos"].sort(key=lambda row: (row["modality"], row["geometry"], row["material"]))
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

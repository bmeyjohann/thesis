#!/usr/bin/env python3
"""Queue GRU and GRU-reconstruction students after no-memory evaluation."""

from __future__ import annotations

import argparse
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


def _latest_numbered(path: Path, pattern: str) -> Path | None:
    regex = re.compile(pattern)
    values = []
    for candidate in path.iterdir() if path.exists() else ():
        match = regex.fullmatch(candidate.name)
        if match:
            values.append((int(match.group(1)), candidate))
    return max(values)[1] if values else None


def _expert(selected: dict, geometry: str, material: str, seed: int) -> Path:
    key = f"{geometry}__{material}__seed{seed}"
    try:
        checkpoint = Path(selected[key]["checkpoint"])
    except KeyError as error:
        raise KeyError(f"No selected expert for {key}") from error
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected expert does not exist: {checkpoint}")
    return checkpoint


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal")
    parser.add_argument("--steps-per-cell", type=int, default=5000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unroll-length", type=int, default=25)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dependency = args.root / "modality_matrix_results/modality_transfer_matrix_results.csv"
    while not dependency.exists():
        matrix_log = args.root / "modality_matrix_watcher.log"
        if matrix_log.exists() and "Traceback" in matrix_log.read_text(errors="replace"):
            return 2
        time.sleep(args.poll_seconds)
    output_root = args.root / "students_recurrent"
    state_path = output_root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "running", "jobs": {}}
    selected = json.loads((args.root / "sequence/evaluation/selected_experts.json").read_text())
    for architecture, modalities in ARCHITECTURE_MODALITIES.items():
        for modality in modalities:
            output = output_root / architecture / modality
            output.mkdir(parents=True, exist_ok=True)
            for geometry, material in product(GEOMETRIES, MATERIALS):
                key = f"{architecture}__{modality}__{geometry}__{material}"
                if state["jobs"].get(key, {}).get("status") == "completed":
                    continue
                expert = _expert(selected, geometry, material, args.seed)
                resume = _latest_numbered(output, r"student_(\d+)\.pt")
                base = [
                    str(PYTHON), str(REPO_ROOT / "train_unitree_recurrent_student.py"),
                    "--architecture", architecture, "--modality", modality,
                    "--geometry", geometry, "--material", material,
                    "--expert-checkpoint", str(expert), "--output", str(output),
                    "--seed", str(args.seed), "--num-envs", str(args.num_envs),
                    "--steps", str(args.steps_per_cell), "--unroll-length", str(args.unroll_length),
                ]
                if resume is not None:
                    base += ["--resume", str(resume)]
                for phase, command in (("smoke", base + ["--smoke"]), ("train", base)):
                    log = output_root / "logs" / f"{key}.{phase}.log"
                    log.parent.mkdir(exist_ok=True)
                    state["jobs"][key] = {"status": f"running_{phase}", "command": command, "log": str(log)}
                    _write(state_path, state)
                    with log.open("w") as stream:
                        result = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    if result.returncode:
                        state["jobs"][key]["status"] = "failed"
                        state["status"] = "failed"
                        _write(state_path, state)
                        return result.returncode
                state["jobs"][key]["status"] = "completed"
                _write(state_path, state)
    state["status"] = "completed"
    _write(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

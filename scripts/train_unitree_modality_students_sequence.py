#!/usr/bin/env python3
"""Wait for expert validation, then distill one student per modality."""

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
MODALITIES = ("height_scan", "depth", "mono_rgb", "stereo_rgb")


def _selection_ready(path: Path, seed: int) -> bool:
    if not path.exists():
        return False
    selected = json.loads(path.read_text())
    required = {
        f"{geometry}__{material}__seed{seed}"
        for geometry, material in product(GEOMETRIES, MATERIALS)
    }
    return required.issubset(selected)


def _selected_expert(selected: dict, geometry: str, material: str, seed: int) -> Path:
    key = f"{geometry}__{material}__seed{seed}"
    try:
        checkpoint = Path(selected[key]["checkpoint"])
    except KeyError as error:
        raise KeyError(f"No selected expert for {key}") from error
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected expert does not exist: {checkpoint}")
    return checkpoint


def _student_checkpoint(output: Path) -> Path | None:
    values = []
    for path in output.glob("student_*.pt"):
        match = re.fullmatch(r"student_(\d+)\.pt", path.name)
        if match:
            values.append((int(match.group(1)), path))
    return max(values)[1] if values else None


def _write(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal")
    parser.add_argument("--steps-per-cell", type=int, default=5000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence = args.root / "sequence"
    ready = sequence / "evaluation/expert_diagonal_results.csv"
    selected_path = sequence / "evaluation/selected_experts.json"
    while not ready.exists() and not _selection_ready(selected_path, args.seed):
        adaptive_state_path = args.root / "adaptive_experts/state.json"
        if adaptive_state_path.exists():
            if json.loads(adaptive_state_path.read_text()).get("status") == "failed":
                return 2
            time.sleep(args.poll_seconds)
            continue
        state_path = sequence / "state.json"
        if state_path.exists() and json.loads(state_path.read_text()).get("status") == "failed":
            return 2
        time.sleep(args.poll_seconds)
    state = json.loads((sequence / "state.json").read_text())
    selected = json.loads(selected_path.read_text())
    output_root = args.root / "students_nomemory"
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state.json"
    student_state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "running", "jobs": {}}
    student_state["status"] = "running"
    _write(state_path, student_state)

    for modality in MODALITIES:
        output = output_root / modality
        output.mkdir(exist_ok=True)
        for geometry, material in product(GEOMETRIES, MATERIALS):
            key = f"{modality}__{geometry}__{material}"
            if student_state["jobs"].get(key, {}).get("status") == "completed":
                continue
            expert = _selected_expert(selected, geometry, material, args.seed)
            resume = _student_checkpoint(output)
            base = [
                str(PYTHON), str(REPO_ROOT / "train_unitree_modality_student.py"),
                "--geometry", geometry, "--material", material, "--modality", modality,
                "--expert-checkpoint", str(expert), "--output", str(output),
                "--seed", str(args.seed), "--num-envs", str(args.num_envs),
                "--steps", str(args.steps_per_cell),
            ]
            if resume is not None:
                base += ["--resume", str(resume)]
            phases = [("smoke", base + ["--smoke"]), ("train", base)]
            for phase, command in phases:
                log = output_root / f"{key}.{phase}.log"
                student_state["jobs"][key] = {"status": f"running_{phase}", "command": command, "log": str(log)}
                _write(state_path, student_state)
                with log.open("w") as stream:
                    result = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    student_state["jobs"][key]["status"] = "failed"
                    student_state["status"] = "failed"
                    _write(state_path, student_state)
                    return result.returncode
            student_state["jobs"][key]["status"] = "completed"
            _write(state_path, student_state)
    student_state["status"] = "completed"
    _write(state_path, student_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

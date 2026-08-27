#!/usr/bin/env python3
"""Train one shared multimodal checkpoint for each memory architecture."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/benjamin/miniconda3/envs/fasttd3/bin/python")
GEOMETRIES = ("flat", "random_rough", "cobblestone", "stairs", "stepping_stones")
MATERIALS = ("rigid", "slippery", "sand_drag")
ARCHITECTURES = ("nomemory", "gru", "gru_reconstruction")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _latest(output: Path) -> Path | None:
    values = []
    for candidate in output.glob("student_*.pt"):
        match = re.fullmatch(r"student_(\d+)\.pt", candidate.name)
        if match:
            values.append((int(match.group(1)), candidate))
    return max(values)[1] if values else None


def _expert(selected: dict, geometry: str, material: str, seed: int) -> Path:
    key = f"{geometry}__{material}__seed{seed}"
    checkpoint = Path(selected[key]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        return subprocess.run(
            command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT
        ).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal")
    parser.add_argument("--steps-per-cell", type=int, default=2500)
    parser.add_argument("--chunk-steps", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unroll-length", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps_per_cell % args.chunk_steps:
        raise ValueError("steps-per-cell must be divisible by chunk-steps")
    selected_path = args.root / "sequence/evaluation/selected_experts.json"
    selected = json.loads(selected_path.read_text())
    cells = list(product(GEOMETRIES, MATERIALS))
    required = {f"{g}__{m}__seed{args.seed}" for g, m in cells}
    if not required.issubset(selected):
        raise RuntimeError("Selected expert manifest does not cover all 15 cells")

    output_root = args.root / "students_shared"
    state_path = output_root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "status": "running",
        "architectures": {},
        "configuration": {
            "steps_per_cell": args.steps_per_cell,
            "chunk_steps": args.chunk_steps,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "modalities": ["height_scan", "depth", "mono_rgb", "stereo_rgb"],
        },
    }
    configuration = {
        "steps_per_cell": args.steps_per_cell,
        "chunk_steps": args.chunk_steps,
        "num_envs": args.num_envs,
        "environment_samples_per_cell": args.steps_per_cell * args.num_envs,
        "teacher_obs_normalization_epsilon": 1e-2,
        "student_input_normalization": "fixed_expert_proprio_and_height_scan",
        "seed": args.seed,
        "modalities": ["height_scan", "depth", "mono_rgb", "stereo_rgb"],
    }
    previous_configuration = state.get("configuration")
    if previous_configuration and previous_configuration != configuration:
        history = state.setdefault("configuration_history", [])
        if not history or history[-1] != previous_configuration:
            history.append(previous_configuration)
    state["configuration"] = configuration
    state["status"] = "running"
    _write(state_path, state)
    rounds = args.steps_per_cell // args.chunk_steps

    for architecture in ARCHITECTURES:
        architecture_state = state["architectures"].setdefault(architecture, {
            "status": "pending", "completed_chunks": [],
        })
        if architecture_state.get("status") == "completed":
            continue
        output = output_root / architecture
        smoke_output = output_root / "smoke" / architecture
        if not architecture_state.get("smoke_completed"):
            smoke_command = [
                str(PYTHON), str(REPO_ROOT / "train_unitree_shared_multimodal_student.py"),
                "--architecture", architecture,
                "--geometry", "flat", "--material", "rigid",
                "--expert-checkpoint", str(_expert(selected, "flat", "rigid", args.seed)),
                "--output", str(smoke_output), "--seed", str(args.seed),
                "--num-envs", "4", "--smoke",
            ]
            architecture_state.update(status="smoke", smoke_command=smoke_command)
            _write(state_path, state)
            result = _run(smoke_command, output_root / "logs" / f"{architecture}.smoke.log")
            if result:
                architecture_state.update(status="failed", returncode=result)
                state["status"] = "failed"
                _write(state_path, state)
                return result
            architecture_state["smoke_completed"] = True
            _write(state_path, state)

        for round_index in range(rounds):
            for geometry, material in cells:
                chunk_key = f"round{round_index + 1}__{geometry}__{material}"
                if chunk_key in architecture_state["completed_chunks"]:
                    continue
                resume = _latest(output)
                command = [
                    str(PYTHON), str(REPO_ROOT / "train_unitree_shared_multimodal_student.py"),
                    "--architecture", architecture,
                    "--geometry", geometry, "--material", material,
                    "--expert-checkpoint", str(_expert(selected, geometry, material, args.seed)),
                    "--output", str(output), "--seed", str(args.seed),
                    "--num-envs", str(args.num_envs), "--steps", str(args.chunk_steps),
                    "--unroll-length", str(args.unroll_length),
                    "--checkpoint-interval", str(args.chunk_steps),
                ]
                if resume is not None:
                    command += ["--resume", str(resume)]
                architecture_state.update(
                    status="training", current_chunk=chunk_key,
                    command=command, checkpoint_before=str(resume) if resume else None,
                )
                _write(state_path, state)
                log = output_root / "logs" / architecture / f"{chunk_key}.log"
                result = _run(command, log)
                if result:
                    architecture_state.update(status="failed", returncode=result, log=str(log))
                    state["status"] = "failed"
                    _write(state_path, state)
                    return result
                architecture_state["completed_chunks"].append(chunk_key)
                architecture_state["latest_checkpoint"] = str(_latest(output))
                _write(state_path, state)
        architecture_state.update(status="completed", current_chunk=None)
        _write(state_path, state)
    state["status"] = "completed"
    _write(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

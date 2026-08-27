#!/usr/bin/env python3
"""Train shared students on mixed geometry/material batches without forgetting."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/benjamin/miniconda3/envs/fasttd3/bin/python")
ARCHITECTURES = ("nomemory", "gru", "gru_reconstruction")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _latest(output: Path) -> tuple[int, Path | None]:
    values = []
    for candidate in output.glob("student_*.pt"):
        match = re.fullmatch(r"student_(\d+)\.pt", candidate.name)
        if match:
            values.append((int(match.group(1)), candidate))
    return max(values) if values else (0, None)


def _run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        return subprocess.run(
            command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal")
    parser.add_argument("--total-steps", type=int, default=37500)
    parser.add_argument("--chunk-steps", type=int, default=2500)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unroll-length", type=int, default=25)
    args = parser.parse_args()
    selected = json.loads((args.root / "sequence/evaluation/selected_experts.json").read_text())
    expert = Path(selected["flat__rigid__seed1"]["checkpoint"])
    output_root = args.root / "students_shared"
    state_path = output_root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "status": "running", "architectures": {},
    }
    state["configuration"] = {
        "training_distribution": "mixed_geometry_and_material_per_parallel_environment",
        "total_steps_per_architecture": args.total_steps,
        "chunk_steps": args.chunk_steps,
        "num_envs": args.num_envs,
        "environment_samples_per_architecture": args.total_steps * args.num_envs,
        "modalities": ["height_scan", "depth", "mono_rgb", "stereo_rgb"],
        "student_input_normalization": "fixed_expert_proprio_and_height_scan",
        "teacher_obs_normalization_epsilon": 0.01,
        "seed": args.seed,
    }
    state["status"] = "running"
    _write(state_path, state)
    for architecture in ARCHITECTURES:
        architecture_state = state["architectures"].setdefault(
            architecture, {"status": "pending", "completed_chunks": []}
        )
        if architecture_state.get("status") == "completed":
            continue
        output = output_root / architecture
        if not architecture_state.get("smoke_completed"):
            command = [
                str(PYTHON), str(REPO_ROOT / "train_unitree_shared_multimodal_student.py"),
                "--architecture", architecture, "--geometry", "mixed", "--material", "mixed",
                "--expert-checkpoint", str(expert), "--output", str(output_root / "smoke" / architecture),
                "--seed", str(args.seed), "--num-envs", "4", "--smoke",
            ]
            architecture_state.update(status="smoke", command=command)
            _write(state_path, state)
            result = _run(command, output_root / "logs" / f"{architecture}.smoke.log")
            if result:
                architecture_state.update(status="failed", returncode=result)
                state["status"] = "failed"; _write(state_path, state); return result
            architecture_state["smoke_completed"] = True
            _write(state_path, state)
        while True:
            current_step, resume = _latest(output)
            if current_step >= args.total_steps:
                break
            steps = min(args.chunk_steps, args.total_steps - current_step)
            chunk = f"steps_{current_step + 1}_{current_step + steps}"
            command = [
                str(PYTHON), str(REPO_ROOT / "train_unitree_shared_multimodal_student.py"),
                "--architecture", architecture, "--geometry", "mixed", "--material", "mixed",
                "--expert-checkpoint", str(expert), "--output", str(output),
                "--seed", str(args.seed), "--num-envs", str(args.num_envs),
                "--steps", str(steps), "--unroll-length", str(args.unroll_length),
                "--checkpoint-interval", "500",
            ]
            if resume is not None:
                command += ["--resume", str(resume)]
            architecture_state.update(
                status="training", current_chunk=chunk,
                latest_checkpoint=str(resume) if resume else None, command=command,
            )
            _write(state_path, state)
            log = output_root / "logs" / architecture / f"{chunk}.log"
            result = _run(command, log)
            if result:
                architecture_state.update(status="failed", returncode=result, log=str(log))
                state["status"] = "failed"; _write(state_path, state); return result
            architecture_state["completed_chunks"].append(chunk)
            architecture_state["latest_checkpoint"] = str(_latest(output)[1])
            _write(state_path, state)
        architecture_state.update(status="completed", current_chunk=None)
        _write(state_path, state)
    state["status"] = "completed"
    _write(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

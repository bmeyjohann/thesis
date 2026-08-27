#!/usr/bin/env python3
"""Resumable sequential launcher for privileged terrain-cell experts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/benjamin/miniconda3/envs/fasttd3/bin/python")
GEOMETRIES = ("flat", "random_rough", "cobblestone", "stairs", "stepping_stones")
MATERIALS = ("rigid", "slippery", "sand_drag")
DEFAULT_WARM_START = REPO_ROOT / "external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/supervisor_rough_model9999/model_9999.pt"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal/sequence")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--warm-start-checkpoint", type=Path, default=DEFAULT_WARM_START)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.state_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "created_at": _now(), "jobs": {}, "status": "running"
    }
    log_dir = args.state_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    for geometry, material in product(GEOMETRIES, MATERIALS):
        key = f"{geometry}__{material}__seed{args.seed}"
        job = state["jobs"].setdefault(key, {"status": "pending"})
        if job["status"] == "completed":
            continue
        base = [
            str(PYTHON), str(REPO_ROOT / "train_unitree_locomotion_cell.py"),
            "--geometry", geometry, "--material", material,
            "--seed", str(args.seed), "--num-envs", str(args.num_envs),
            "--iterations", str(args.iterations), "--logger", args.logger,
            "--warm-start-checkpoint", str(args.warm_start_checkpoint),
        ]
        commands = [] if args.skip_smoke else [("smoke", base + ["--smoke"])]
        commands.append(("train", base))
        for phase, command in commands:
            output = log_dir / f"{key}.{phase}.log"
            job.update(status=f"running_{phase}", command=command, log=str(output), updated_at=_now())
            _write(state_path, state)
            with output.open("a", buffering=1) as stream:
                stream.write(f"[{_now()}] {' '.join(command)}\n")
                env = dict(os.environ, PYTHONUNBUFFERED="1", MUJOCO_GL="egl")
                result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                job.update(status="failed", phase=phase, returncode=result.returncode, updated_at=_now())
                state["status"] = "failed"
                _write(state_path, state)
                print(f"FAILED {key} during {phase}; see {output}", flush=True)
                return result.returncode
        job.update(status="completed", updated_at=_now())
        _write(state_path, state)
    state.update(status="completed", completed_at=_now())
    _write(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

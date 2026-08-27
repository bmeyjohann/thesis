#!/usr/bin/env python3
"""Screen the robust supervisor, then fine-tune only failing terrain cells."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unitree_locomotion_cells import GEOMETRIES, MATERIALS  # noqa: E402


PYTHON = Path("/home/benjamin/miniconda3/envs/fasttd3/bin/python")
DEFAULT_SUPERVISOR = (
    REPO_ROOT
    / "external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/"
    "supervisor_rough_model9999/model_9999.pt"
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _evaluate(
    checkpoint: Path,
    geometry: str,
    material: str,
    seed: int,
    num_envs: int,
    steps: int,
    summary_path: Path,
) -> dict:
    if not summary_path.exists():
        command = [
            str(PYTHON), str(REPO_ROOT / "eval_unitree_locomotion_cell.py"),
            "--geometry", geometry, "--material", material,
            "--checkpoint-file", str(checkpoint), "--seed", str(seed),
            "--num-envs", str(num_envs), "--steps", str(steps),
            "--summary-file", str(summary_path),
        ]
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.with_suffix(".log").open("w") as stream:
            result = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Evaluation failed; see {summary_path.with_suffix('.log')}")
    return json.loads(summary_path.read_text())


def _numbered_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    values = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            values.append((int(match.group(1)), path))
    return sorted(values)


def _candidate_run(geometry: str, material: str, seed: int, minimum_number: int) -> Path | None:
    root = REPO_ROOT / "logs/rsl_rl/g1_multimodal_cell_experts"
    suffix = f"_{geometry}_{material}_seed{seed}"
    runs = [
        path for path in root.iterdir()
        if path.is_dir() and path.name.endswith(suffix) and "smoke" not in path.name
    ] if root.exists() else []
    complete = [path for path in runs if _numbered_checkpoints(path) and _numbered_checkpoints(path)[-1][0] >= minimum_number]
    return max(complete, key=lambda path: path.stat().st_mtime_ns) if complete else None


def _train_cell(args: argparse.Namespace, geometry: str, material: str, log_dir: Path) -> Path:
    base = [
        str(PYTHON), str(REPO_ROOT / "train_unitree_locomotion_cell.py"),
        "--geometry", geometry, "--material", material,
        "--seed", str(args.train_seed), "--num-envs", str(args.train_num_envs),
        "--iterations", str(args.fine_tune_iterations), "--logger", args.logger,
        "--warm-start-checkpoint", str(args.supervisor),
    ]
    for phase, command in (("smoke", base + ["--smoke"]), ("train", base)):
        log = log_dir / f"{geometry}__{material}.{phase}.log"
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{phase} failed for {geometry}/{material}; see {log}")
    minimum = 9999 + args.fine_tune_iterations - 1
    run = _candidate_run(geometry, material, args.train_seed, minimum)
    if run is None:
        raise FileNotFoundError(f"No completed fine-tuning run for {geometry}/{material}")
    return run


def _selection_candidates(supervisor: Path, run_dir: Path | None) -> list[Path]:
    candidates = [supervisor]
    if run_dir is not None:
        numbered = _numbered_checkpoints(run_dir)
        maximum = numbered[-1][0]
        candidates.extend(path for number, path in numbered if number % 500 == 0 or number == maximum)
    return list(dict.fromkeys(candidates))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal")
    parser.add_argument("--supervisor", type=Path, default=DEFAULT_SUPERVISOR)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--train-num-envs", type=int, default=1024)
    parser.add_argument("--fine-tune-iterations", type=int, default=2000)
    parser.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    parser.add_argument("--screen-seed", type=int, default=11)
    parser.add_argument("--screen-num-envs", type=int, default=64)
    parser.add_argument("--screen-steps", type=int, default=500)
    parser.add_argument("--max-terminations-per-1000", type=float, default=1.0)
    parser.add_argument("--min-mean-step-reward", type=float, default=0.05)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=(11, 23, 37, 51, 73))
    parser.add_argument("--eval-num-envs", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=1000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.supervisor = args.supervisor.expanduser().resolve()
    if not args.supervisor.is_file():
        raise FileNotFoundError(args.supervisor)
    adaptive_root = args.root / "adaptive_experts"
    evaluation_root = args.root / "sequence/evaluation"
    state_path = adaptive_root / "state.json"
    logs = adaptive_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_thresholds": {
            "max_terminations_per_1000": args.max_terminations_per_1000,
            "min_mean_step_reward": args.min_mean_step_reward,
        },
        "cells": {},
    }
    _write(state_path, state)
    # Screen every physical cell first so the expensive fine-tuning set is
    # known before any additional PPO run starts.
    for geometry, material in product(GEOMETRIES, MATERIALS):
        key = f"{geometry}__{material}__seed{args.train_seed}"
        cell = state["cells"].setdefault(key, {})
        screen = _evaluate(
            args.supervisor, geometry, material, args.screen_seed,
            args.screen_num_envs, args.screen_steps,
            adaptive_root / "screen" / f"{key}.json",
        )
        total_steps = int(screen["num_envs"]) * int(screen["steps"])
        terminations_per_1000 = 1000.0 * float(screen["termination_count"]) / total_steps
        failed_screen = (
            terminations_per_1000 > args.max_terminations_per_1000
            or float(screen["mean_step_reward"]) < args.min_mean_step_reward
        )
        cell.update({
            "screen": screen,
            "terminations_per_1000": terminations_per_1000,
            "failed_screen": failed_screen,
            "status": "screened",
        })
        _write(state_path, state)

    selected_experts: dict[str, dict] = {}
    rows = []
    for geometry, material in product(GEOMETRIES, MATERIALS):
        key = f"{geometry}__{material}__seed{args.train_seed}"
        cell = state["cells"][key]
        failed_screen = bool(cell["failed_screen"])

        minimum = 9999 + args.fine_tune_iterations - 1
        run_dir = _candidate_run(geometry, material, args.train_seed, minimum)
        if failed_screen and run_dir is None:
            cell["status"] = "fine_tuning"
            _write(state_path, state)
            run_dir = _train_cell(args, geometry, material, logs)

        ranked = []
        for checkpoint in _selection_candidates(args.supervisor, run_dir):
            summary = _evaluate(
                checkpoint, geometry, material, args.screen_seed,
                args.screen_num_envs, args.screen_steps,
                adaptive_root / "selection" / key / f"{checkpoint.parent.name}__{checkpoint.stem}.json",
            )
            candidate_steps = int(summary["num_envs"]) * int(summary["steps"])
            fall_rate = float(summary["termination_count"]) / candidate_steps
            ranked.append((fall_rate, -float(summary["mean_step_reward"]), checkpoint, summary))
        _, _, selected, selection_summary = min(ranked, key=lambda value: value[:2])
        selected_experts[key] = {
            "checkpoint": str(selected.resolve()),
            "selection_summary": selection_summary,
            "candidate_count": len(ranked),
            "baseline_failed_screen": failed_screen,
        }
        _write(evaluation_root / "selected_experts.json", selected_experts)
        cell.update({"status": "selected", "selected_checkpoint": str(selected.resolve())})
        _write(state_path, state)

        for eval_seed in args.eval_seeds:
            summary = _evaluate(
                selected, geometry, material, eval_seed,
                args.eval_num_envs, args.eval_steps,
                evaluation_root / "adaptive_full" / key
                / f"{selected.parent.name}__{selected.stem}__eval{eval_seed}.json",
            )
            rows.append(summary)

    fields = sorted({field for row in rows for field in row})
    csv_path = evaluation_root / "expert_diagonal_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    state["status"] = "completed"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    state["result_csv"] = str(csv_path.resolve())
    _write(state_path, state)
    print(csv_path.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

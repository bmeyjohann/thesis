#!/usr/bin/env python3
"""Wait for expert training, then evaluate each expert on its own cell."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/benjamin/miniconda3/envs/fasttd3/bin/python")
WARM_BASELINE = REPO_ROOT / "external/unitree_rl_mjlab/logs/rsl_rl/g1_velocity/supervisor_rough_model9999/model_9999.pt"


def _checkpoint(run_dir: Path) -> Path:
    candidates = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint in {run_dir}")
    return max(candidates)[1]


def _selection_candidates(run_dir: Path) -> list[Path]:
    numbered = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise FileNotFoundError(f"No model checkpoint in {run_dir}")
    maximum = max(numbered)[0]
    selected = [path for number, path in numbered if number % 500 == 0 or number == maximum]
    return [WARM_BASELINE, *sorted(set(selected))]


def _evaluate(checkpoint: Path, geometry: str, material: str, seed: int, num_envs: int, steps: int, summary_path: Path) -> dict:
    if not summary_path.exists():
        command = [
            str(PYTHON), str(REPO_ROOT / "eval_unitree_locomotion_cell.py"),
            "--geometry", geometry, "--material", material,
            "--checkpoint-file", str(checkpoint), "--seed", str(seed),
            "--num-envs", str(num_envs), "--steps", str(steps),
            "--summary-file", str(summary_path),
        ]
        with summary_path.with_suffix(".log").open("w") as stream:
            result = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"Evaluation failed; see {summary_path.with_suffix('.log')}")
    return json.loads(summary_path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-dir", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal/sequence")
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 23, 37, 51, 73))
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_path = args.sequence_dir / "state.json"
    while True:
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("status") in {"completed", "failed"}:
                break
        time.sleep(args.poll_seconds)
    if state.get("status") != "completed":
        print("Training sequence failed; post-evaluation not started.", flush=True)
        return 2

    output_dir = args.sequence_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_experts = {}
    log_root = REPO_ROOT / "logs/rsl_rl/g1_multimodal_cell_experts"
    for key, job in sorted(state["jobs"].items()):
        if job.get("status") != "completed":
            continue
        geometry, material, seed_label = key.split("__")
        train_seed = int(seed_label.removeprefix("seed"))
        suffix = f"_{geometry}_{material}_seed{train_seed}"
        run_dirs = [path for path in log_root.iterdir() if path.is_dir() and path.name.endswith(suffix)]
        if not run_dirs:
            raise FileNotFoundError(f"No run directory matching {suffix}")
        run_dir = max(run_dirs, key=lambda value: value.stat().st_mtime_ns)
        ranked = []
        selection_dir = output_dir / "selection" / key
        selection_dir.mkdir(parents=True, exist_ok=True)
        for checkpoint in _selection_candidates(run_dir):
            summary = _evaluate(
                checkpoint, geometry, material, 11, 64, 500,
                selection_dir / f"{checkpoint.parent.name}__{checkpoint.stem}.json",
            )
            total_steps = int(summary["num_envs"]) * int(summary["steps"])
            fall_rate = float(summary["termination_count"]) / total_steps
            ranked.append((fall_rate, -float(summary["mean_step_reward"]), checkpoint, summary))
        _, _, checkpoint, selection_summary = min(ranked, key=lambda value: value[:2])
        selected_experts[key] = {
            "checkpoint": str(checkpoint.resolve()),
            "selection_summary": selection_summary,
            "candidate_count": len(ranked),
        }
        (output_dir / "selected_experts.json").write_text(
            json.dumps(selected_experts, indent=2, sort_keys=True) + "\n"
        )
        for eval_seed in args.seeds:
            summary_path = output_dir / f"{key}__eval{eval_seed}.json"
            rows.append(_evaluate(
                checkpoint, geometry, material, eval_seed,
                args.num_envs, args.steps, summary_path,
            ))

    csv_path = output_dir / "expert_diagonal_results.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path.resolve(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

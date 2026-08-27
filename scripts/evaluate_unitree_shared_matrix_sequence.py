#!/usr/bin/env python3
"""Evaluate each shared checkpoint under every modality and physical cell."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/benjamin/miniconda3/envs/fasttd3/bin/python")
ARCHITECTURES = ("nomemory", "gru", "gru_reconstruction")
MODALITIES = ("height_scan", "depth", "mono_rgb", "stereo_rgb")
GEOMETRIES = ("flat", "random_rough", "cobblestone", "stairs", "stepping_stones")
MATERIALS = ("rigid", "slippery", "sand_drag")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _write_state(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _run_case(command: list[str], summary: Path) -> tuple[Path, list[str], int]:
    if summary.exists():
        return summary, command, 0
    with summary.with_suffix(".log").open("w") as stream:
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return summary, command, result.returncode


def _run_cases(cases: list[tuple[list[str], Path]], workers: int):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_case, command, summary) for command, summary in cases]
        for future in as_completed(futures):
            yield future.result()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "artifacts/unitree_multimodal")
    parser.add_argument("--seeds", type=int, nargs="+", default=(11, 23, 37))
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    state_path = args.root / "students_shared/state.json"
    while not state_path.exists() or json.loads(state_path.read_text()).get("status") not in {"completed", "failed"}:
        time.sleep(args.poll_seconds)
    state = json.loads(state_path.read_text())
    if state.get("status") != "completed":
        return 2

    output = args.root / "shared_matrix_results"
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "evaluation_state.json"
    progress = {
        "status": "expert_reference",
        "expert_completed": 0,
        "student_completed": 0,
        "student_num_envs": args.num_envs,
        "student_steps": args.steps,
        "seeds": list(args.seeds),
        "workers": args.workers,
    }
    _write_state(progress_path, progress)

    selected = json.loads((args.root / "sequence/evaluation/selected_experts.json").read_text())
    expert_cases = []
    for geometry, material, seed in product(GEOMETRIES, MATERIALS, args.seeds):
        checkpoint = Path(selected[f"{geometry}__{material}__seed1"]["checkpoint"])
        summary = output / f"expert__{geometry}__{material}__seed{seed}.json"
        command = [
            str(PYTHON), str(REPO_ROOT / "eval_unitree_shared_expert.py"),
            "--geometry", geometry, "--material", material,
            "--checkpoint", str(checkpoint), "--summary-file", str(summary),
            "--seed", str(seed), "--num-envs", str(args.num_envs), "--steps", str(args.steps),
        ]
        expert_cases.append((command, summary))
    expert_rows = []
    for summary, command, returncode in _run_cases(expert_cases, args.workers):
        if returncode:
            progress.update(status="failed", failed_command=command, returncode=returncode)
            _write_state(progress_path, progress)
            return returncode
        expert_rows.append(json.loads(summary.read_text()))
        progress["expert_completed"] = len(expert_rows)
        _write_csv(output / "expert_reference_results.csv", expert_rows)
        _write_state(progress_path, progress)

    progress["status"] = "student_matrix"
    _write_state(progress_path, progress)
    student_cases = []
    for architecture, modality, geometry, material, seed in product(
        ARCHITECTURES, MODALITIES, GEOMETRIES, MATERIALS, args.seeds
    ):
        checkpoint = Path(state["architectures"][architecture]["latest_checkpoint"])
        summary = output / f"{architecture}__{modality}__{geometry}__{material}__seed{seed}.json"
        evaluator = "eval_unitree_modality_student.py" if architecture == "nomemory" else "eval_unitree_recurrent_student.py"
        command = [
            str(PYTHON), str(REPO_ROOT / evaluator),
            "--modality", modality, "--geometry", geometry, "--material", material,
            "--checkpoint", str(checkpoint), "--summary-file", str(summary),
            "--seed", str(seed), "--num-envs", str(args.num_envs), "--steps", str(args.steps),
        ]
        student_cases.append((command, summary))
    rows = []
    for summary, command, returncode in _run_cases(student_cases, args.workers):
        if returncode:
            progress.update(status="failed", failed_command=command, returncode=returncode)
            _write_state(progress_path, progress)
            return returncode
        row = json.loads(summary.read_text())
        row["architecture"] = summary.name.split("__", 1)[0]
        rows.append(row)
        csv_path = output / "shared_transfer_matrix_results.csv"
        _write_csv(csv_path, rows)
        progress["student_completed"] = len(rows)
        _write_state(progress_path, progress)
    progress["status"] = "completed"
    _write_state(progress_path, progress)
    print(csv_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

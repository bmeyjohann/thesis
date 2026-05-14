#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "local" / "thesis_offline_human_eval"
DEFAULT_MODEL_ROOT = REPO_ROOT / "models" / "thesis_offline_human_eval"
DEFAULT_PROJECT = "ogbench-manip-offline"
DEFAULT_DATASET = REPO_ROOT / "local" / "ogbench_manip_datasets" / "cube-single-singletask-task1-v0__human_vr_online_replay__20260409_114611.npz"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "fast_sac" / "cube_single_task1_human_onlycollecteddemos_demoaug" / "cube_single_singletask_task1_v0_final.pt"


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    suite: str
    family: str
    launcher: str
    launcher_kind: str
    default_overrides: tuple[str, ...]
    color: str

    @property
    def launcher_path(self) -> Path:
        return REPO_ROOT / self.launcher


@dataclass(frozen=True)
class JobSpec:
    key: str
    label: str
    suite: str
    family: str
    seed: int
    exp_name: str
    group: str
    output_dir: str
    metrics_path: str
    command: list[str]


SCRATCH_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        key="bc_scratch",
        label="BC",
        suite="scratch",
        family="from_scratch",
        launcher="scripts/run_cube_single_task1_offline_method_compare_local.sh",
        launcher_kind="offline_method_compare",
        default_overrides=(
            "METHOD=bc",
            "OFFLINE_UPDATES=500",
        ),
        color="#1f77b4",
    ),
    ExperimentSpec(
        key="hgdagger_scratch",
        label="HG-DAgger",
        suite="scratch",
        family="from_scratch",
        launcher="scripts/run_cube_single_task1_offline_method_compare_local.sh",
        launcher_kind="offline_method_compare",
        default_overrides=(
            "METHOD=hg_dagger",
            "OFFLINE_UPDATES=500",
        ),
        color="#ff7f0e",
    ),
    ExperimentSpec(
        key="pvp_scratch",
        label="PVP ablation",
        suite="scratch",
        family="from_scratch",
        launcher="scripts/run_cube_single_task1_offline_method_compare_local.sh",
        launcher_kind="offline_method_compare",
        default_overrides=(
            "METHOD=pvp",
            "OFFLINE_UPDATES=200",
        ),
        color="#2ca02c",
    ),
    ExperimentSpec(
        key="eil_scratch",
        label="EIL ablation",
        suite="scratch",
        family="from_scratch",
        launcher="scripts/run_cube_single_task1_offline_method_compare_local.sh",
        launcher_kind="offline_method_compare",
        default_overrides=(
            "METHOD=eil",
            "OFFLINE_UPDATES=500",
        ),
        color="#d62728",
    ),
    ExperimentSpec(
        key="hilserl_scratch",
        label="HIL-SERL ablation",
        suite="scratch",
        family="from_scratch",
        launcher="scripts/run_cube_single_task1_offline_method_compare_local.sh",
        launcher_kind="offline_method_compare",
        default_overrides=(
            "METHOD=hilserl",
            "OFFLINE_UPDATES=200",
        ),
        color="#9467bd",
    ),
    ExperimentSpec(
        key="own_weighted_scratch",
        label="Ours weighted",
        suite="scratch",
        family="from_scratch",
        launcher="scripts/run_cube_single_task1_offline_method_compare_local.sh",
        launcher_kind="offline_method_compare",
        default_overrides=(
            "METHOD=own",
            "OFFLINE_UPDATES=2000",
        ),
        color="#8c564b",
    ),
    ExperimentSpec(
        key="own_nogate_scratch",
        label="Ours no gate",
        suite="scratch",
        family="from_scratch",
        launcher="scripts/run_cube_single_task1_offline_method_compare_local.sh",
        launcher_kind="offline_method_compare",
        default_overrides=(
            "METHOD=own_nogate",
            "OFFLINE_UPDATES=2000",
        ),
        color="#e377c2",
    ),
)

RECOVERY_CORE_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        key="bc_recover",
        label="BC recover",
        suite="recovery_core",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_actor_recover_local.sh",
        launcher_kind="actor_recover",
        default_overrides=(
            "NUM_GRADIENT_STEPS=1000",
            "EVAL_INTERVAL=250",
            "INIT_MODE=random",
            "ACTOR_Q_WEIGHT=0.0",
            "BC_TEACHER_WEIGHT=1.0",
            "TEACHER_MASK_MODE=effective",
        ),
        color="#1f77b4",
    ),
    ExperimentSpec(
        key="bt_recover",
        label="BT pref",
        suite="recovery_core",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=bradley_terry",
            "PREF_MARGIN=0.01",
            "PREF_WEIGHT=1.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=200",
            "EVAL_INTERVAL=100",
        ),
        color="#17becf",
    ),
    ExperimentSpec(
        key="hinge001w1_recover",
        label="Hinge m=0.01 w=1",
        suite="recovery_core",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.01",
            "PREF_WEIGHT=1.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
        ),
        color="#ff7f0e",
    ),
    ExperimentSpec(
        key="hinge010w1_recover",
        label="Hinge m=0.10 w=1",
        suite="recovery_core",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.10",
            "PREF_WEIGHT=1.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
        ),
        color="#2ca02c",
    ),
    ExperimentSpec(
        key="hinge001w5_recover",
        label="Hinge m=0.01 w=5",
        suite="recovery_core",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.01",
            "PREF_WEIGHT=5.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
        ),
        color="#d62728",
    ),
    ExperimentSpec(
        key="hinge010w5_recover",
        label="Hinge m=0.10 w=5",
        suite="recovery_core",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.10",
            "PREF_WEIGHT=5.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
        ),
        color="#9467bd",
    ),
)

RECOVERY_GATE_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        key="eps_base",
        label="Base eps=1e-6",
        suite="recovery_gates",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.01",
            "PREF_WEIGHT=5.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
            "LINKED_ACTION_FILTER_MODE=epsilon",
            "LINKED_ACTION_SCOPE=all",
            "LINKED_ACTION_FILTER_METRIC=l1",
            "LINKED_ACTION_WEIGHT_METRIC=mean_abs",
            "LINKED_ACTION_EPSILON=1e-6",
            "LINKED_ACTION_WEIGHT_SCALE=0.0",
        ),
        color="#7f7f7f",
    ),
    ExperimentSpec(
        key="eps001",
        label="eps=0.01",
        suite="recovery_gates",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.01",
            "PREF_WEIGHT=5.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
            "LINKED_ACTION_FILTER_MODE=epsilon",
            "LINKED_ACTION_SCOPE=all",
            "LINKED_ACTION_FILTER_METRIC=l1",
            "LINKED_ACTION_WEIGHT_METRIC=mean_abs",
            "LINKED_ACTION_EPSILON=0.01",
            "LINKED_ACTION_WEIGHT_SCALE=0.0",
        ),
        color="#bcbd22",
    ),
    ExperimentSpec(
        key="eps001_scale025",
        label="eps=0.01 + scale=0.25",
        suite="recovery_gates",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.01",
            "PREF_WEIGHT=5.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
            "LINKED_ACTION_FILTER_MODE=epsilon",
            "LINKED_ACTION_SCOPE=all",
            "LINKED_ACTION_FILTER_METRIC=l1",
            "LINKED_ACTION_WEIGHT_METRIC=mean_abs",
            "LINKED_ACTION_EPSILON=0.01",
            "LINKED_ACTION_WEIGHT_SCALE=0.25",
        ),
        color="#1f77b4",
    ),
    ExperimentSpec(
        key="angle_gate",
        label="Angle gate",
        suite="recovery_gates",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.01",
            "PREF_WEIGHT=5.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
            "LINKED_ACTION_FILTER_MODE=angle",
            "LINKED_ACTION_SCOPE=all",
            "LINKED_ACTION_ANGLE_THRESHOLD_DEG=30",
            "LINKED_ACTION_WEIGHT_SCALE=0.0",
        ),
        color="#ff7f0e",
    ),
    ExperimentSpec(
        key="component_gate",
        label="Component corridor",
        suite="recovery_gates",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.01",
            "PREF_WEIGHT=5.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
            "LINKED_ACTION_FILTER_MODE=component",
            "LINKED_COMPONENT_XYZ_THRESHOLD=0.35",
            "LINKED_COMPONENT_YAW_THRESHOLD=0.45",
            "LINKED_COMPONENT_GRIPPER_THRESHOLD=0.9",
            "LINKED_COMPONENT_ADAPTIVE_ENABLE=1",
            "LINKED_ACTION_WEIGHT_SCALE=0.0",
        ),
        color="#2ca02c",
    ),
    ExperimentSpec(
        key="xyz_ball_hard",
        label="XYZ L2 hard ball",
        suite="recovery_gates",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.01",
            "PREF_WEIGHT=5.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
            "LINKED_ACTION_FILTER_MODE=epsilon",
            "LINKED_ACTION_SCOPE=xyz",
            "LINKED_ACTION_FILTER_METRIC=l2",
            "LINKED_ACTION_WEIGHT_METRIC=l2",
            "LINKED_ACTION_EPSILON=0.35",
            "LINKED_ACTION_WEIGHT_SCALE=0.0",
        ),
        color="#d62728",
    ),
    ExperimentSpec(
        key="xyz_ball_soft",
        label="XYZ L2 soft ball",
        suite="recovery_gates",
        family="recovery",
        launcher="scripts/run_cube_single_task1_offline_pref_probe_local.sh",
        launcher_kind="pref_probe",
        default_overrides=(
            "PREF_LOSS_TYPE=hinge",
            "PREF_MARGIN=0.01",
            "PREF_WEIGHT=5.0",
            "CRITIC_STEPS=500",
            "ACTOR_STEPS=500",
            "EVAL_INTERVAL=100",
            "LINKED_ACTION_FILTER_MODE=epsilon",
            "LINKED_ACTION_SCOPE=xyz",
            "LINKED_ACTION_FILTER_METRIC=l2",
            "LINKED_ACTION_WEIGHT_METRIC=l2",
            "LINKED_ACTION_EPSILON=0.0",
            "LINKED_ACTION_WEIGHT_SCALE=1.0",
        ),
        color="#9467bd",
    ),
)

SPEC_ORDER: tuple[ExperimentSpec, ...] = SCRATCH_SPECS + RECOVERY_CORE_SPECS + RECOVERY_GATE_SPECS
SPEC_BY_KEY = {spec.key: spec for spec in SPEC_ORDER}
SUITES: dict[str, tuple[str, ...]] = {
    "scratch": tuple(spec.key for spec in SCRATCH_SPECS),
    "recovery_core": tuple(spec.key for spec in RECOVERY_CORE_SPECS),
    "recovery_gates": tuple(spec.key for spec in RECOVERY_GATE_SPECS),
    "all": tuple(spec.key for spec in SPEC_ORDER),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seeded offline human-dataset thesis harness.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    launch = subparsers.add_parser("launch", help="Generate a seeded experiment manifest.")
    launch.add_argument("--suite", choices=tuple(SUITES.keys()), default="all")
    launch.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    launch.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    launch.add_argument("--model-root", type=str, default=str(DEFAULT_MODEL_ROOT))
    launch.add_argument("--run-prefix", type=str, default="offline_human_seeded")
    launch.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    launch.add_argument("--entity", type=str, default="")
    launch.add_argument("--group-prefix", type=str, default="offline_human_seeded")
    launch.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET))
    launch.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT))
    launch.add_argument("--timestamp", type=str, default="")
    launch.add_argument("--emit-commands", action="store_true", default=False)

    collect = subparsers.add_parser("collect", help="Collect seeded offline metrics from a manifest.")
    collect.add_argument("--manifest", type=str, required=True)
    collect.add_argument("--output-root", type=str, default="")

    plot = subparsers.add_parser("plot", help="Plot seeded offline metrics from a collected directory.")
    plot.add_argument("--input-root", type=str, required=True)

    return parser.parse_args()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _suite_specs(suite: str) -> list[ExperimentSpec]:
    allowed = set(SUITES[suite])
    return [spec for spec in SPEC_ORDER if spec.key in allowed]


def _spec_overrides(
    spec: ExperimentSpec,
    *,
    project: str,
    entity: str,
    group: str,
    dataset_path: str,
    checkpoint_path: str,
    seed: int,
    exp_name: str,
    output_dir: str,
) -> list[str]:
    overrides = [
        f"PROJECT={project}",
        f"WANDB_MODE=online",
        f"GROUP={group}",
        f"DATASET_PATH={dataset_path}",
        f"SEED={seed}",
        f"NAME={exp_name}",
        f"OUTPUT_DIR={output_dir}",
    ]
    if entity:
        overrides.append(f"ENTITY={entity}")
    if spec.launcher_kind in {"actor_recover", "pref_probe"}:
        overrides.append(f"CHECKPOINT_PATH={checkpoint_path}")
    overrides.extend(spec.default_overrides)
    return overrides


def _build_jobs(args: argparse.Namespace) -> tuple[str, list[JobSpec]]:
    timestamp = str(args.timestamp).strip() or _timestamp()
    specs = _suite_specs(str(args.suite))
    jobs: list[JobSpec] = []
    for spec in specs:
        for seed in [int(seed) for seed in args.seeds]:
            exp_name = f"{args.run_prefix}_{spec.key}_seed{seed}_{timestamp}"
            group = f"{args.group_prefix}_{timestamp}_{spec.suite}"
            output_dir = str(Path(args.model_root) / timestamp / exp_name)
            metrics_path = str(Path(output_dir) / "metrics.jsonl")
            overrides = _spec_overrides(
                spec,
                project=str(args.project),
                entity=str(args.entity).strip(),
                group=group,
                dataset_path=str(Path(args.dataset_path).expanduser().resolve()),
                checkpoint_path=str(Path(args.checkpoint_path).expanduser().resolve()),
                seed=seed,
                exp_name=exp_name,
                output_dir=output_dir,
            )
            command = ["bash", str(spec.launcher_path)] + overrides
            jobs.append(
                JobSpec(
                    key=spec.key,
                    label=spec.label,
                    suite=spec.suite,
                    family=spec.family,
                    seed=seed,
                    exp_name=exp_name,
                    group=group,
                    output_dir=output_dir,
                    metrics_path=metrics_path,
                    command=command,
                )
            )
    return timestamp, jobs


def _write_manifest(*, jobs: list[JobSpec], args: argparse.Namespace, timestamp: str) -> Path:
    manifest_dir = _ensure_dir(Path(args.output_root).expanduser().resolve() / "manifests")
    manifest_path = manifest_dir / f"{args.run_prefix}_{args.suite}_{timestamp}.json"
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "timestamp": timestamp,
        "suite": args.suite,
        "seeds": [int(seed) for seed in args.seeds],
        "project": args.project,
        "entity": args.entity,
        "dataset_path": str(Path(args.dataset_path).expanduser().resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
        "jobs": [asdict(job) for job in jobs],
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _is_eval_row(row: dict[str, Any]) -> bool:
    if "Eval/success_rate" not in row:
        return False
    phase = row.get("phase")
    return phase is None or phase in {"initial", "actor"}


def _collect_job(job: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metrics_path = Path(str(job["metrics_path"]))
    rows = _read_metrics(metrics_path)
    eval_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    dataset_stats: dict[str, Any] = {}
    for row in rows:
        for key, value in row.items():
            if key.startswith("Offline/") and key not in dataset_stats:
                dataset_stats[key] = value

        metric_rows.append(
            {
                "key": job["key"],
                "label": job["label"],
                "suite": job["suite"],
                "family": job["family"],
                "seed": int(job["seed"]),
                "exp_name": job["exp_name"],
                "phase": row.get("phase", ""),
                "step": int(row.get("step", 0)),
                "pref_q_delta_mean": row.get("Pref/q_delta_mean", row.get("Train/pref_q_delta_mean")),
                "pref_loss": row.get("Train/pref_loss"),
                "pair_weight_mean": row.get("Train/pair_weight_mean", row.get("Train/pref_action_weight_mean")),
                "actor_loss": row.get("Train/loss", row.get("Train/actor_loss")),
                "actor_q_loss": row.get("Train/actor_q_loss"),
                "bc_teacher_loss": row.get("Train/bc_teacher_loss"),
                "q_actor_mean": row.get("Diag/q_actor_mean"),
            }
        )

        if _is_eval_row(row):
            eval_rows.append(
                {
                    "key": job["key"],
                    "label": job["label"],
                    "suite": job["suite"],
                    "family": job["family"],
                    "seed": int(job["seed"]),
                    "exp_name": job["exp_name"],
                    "phase": row.get("phase", ""),
                    "step": int(row.get("step", 0)),
                    "success_rate": float(row.get("Eval/success_rate", 0.0)),
                    "avg_length": float(row.get("Eval/avg_length", math.nan)),
                    "avg_return": float(row.get("Eval/avg_return", math.nan)),
                    "timeout_rate": float(row.get("Eval/timeout_rate", math.nan)),
                }
            )

    summary = {
        "key": job["key"],
        "label": job["label"],
        "suite": job["suite"],
        "family": job["family"],
        "seed": int(job["seed"]),
        "exp_name": job["exp_name"],
        "metrics_path": str(metrics_path),
        "output_dir": str(job["output_dir"]),
        "exists": int(metrics_path.exists()),
        "num_eval_points": len(eval_rows),
    }
    if eval_rows:
        best = max(eval_rows, key=lambda row: (float(row["success_rate"]), -float(row.get("avg_length", math.inf))))
        final = max(eval_rows, key=lambda row: float(row["step"]))
        summary.update(
            {
                "best_step": int(best["step"]),
                "best_success": float(best["success_rate"]),
                "best_avg_length": float(best["avg_length"]),
                "final_step": int(final["step"]),
                "final_success": float(final["success_rate"]),
                "final_avg_length": float(final["avg_length"]),
            }
        )
    for key in (
        "Offline/dataset_rows",
        "Offline/teacher_like_fraction",
        "Offline/teacher_flag_fraction",
        "Offline/action_diff_fraction",
        "Offline/intervened_gripper_only_fraction",
        "Offline/pair_weight_mean_selected",
        "Offline/action_delta_mean_selected",
        "Offline/filter_pass_fraction",
    ):
        if key in dataset_stats:
            summary[key.replace("/", "_").lower()] = dataset_stats[key]
    return eval_rows, metric_rows, summary


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.fmean(values)), float(statistics.stdev(values))


def _aggregate_curves(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        if metric not in row:
            continue
        key = (str(row["suite"]), str(row["family"]), str(row["key"]), int(row["step"]))
        grouped[key].append(float(row[metric]))
        labels[str(row["key"])] = str(row["label"])
    aggregated: list[dict[str, Any]] = []
    for (suite, family, key, step), values in grouped.items():
        mean, std = _mean_std(values)
        sem = float(std / math.sqrt(len(values))) if values else 0.0
        aggregated.append(
            {
                "suite": suite,
                "family": family,
                "key": key,
                "label": labels.get(key, key),
                "step": step,
                "metric": metric,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95": float(1.96 * sem),
                "n": len(values),
            }
        )
    aggregated.sort(key=lambda row: (row["suite"], row["key"], row["step"]))
    return aggregated


def _aggregate_summary(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    seeds: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        if metric not in row:
            continue
        try:
            value = float(row[metric])
        except Exception:
            continue
        if math.isnan(value):
            continue
        group_key = (str(row["suite"]), str(row["family"]), str(row["key"]))
        grouped[group_key].append(value)
        seeds[group_key].append(int(row["seed"]))
        labels[str(row["key"])] = str(row["label"])
    aggregated: list[dict[str, Any]] = []
    for (suite, family, key), values in grouped.items():
        mean, std = _mean_std(values)
        sem = float(std / math.sqrt(len(values))) if values else 0.0
        aggregated.append(
            {
                "suite": suite,
                "family": family,
                "key": key,
                "label": labels.get(key, key),
                "metric": metric,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95": float(1.96 * sem),
                "n": len(values),
            }
        )
    aggregated.sort(key=lambda row: (row["suite"], tuple(SUITES["all"]).index(row["key"]) if row["key"] in SUITES["all"] else 10_000))
    return aggregated


def cmd_launch(args: argparse.Namespace) -> int:
    timestamp, jobs = _build_jobs(args)
    manifest_path = _write_manifest(jobs=jobs, args=args, timestamp=timestamp)
    print(f"Manifest: {manifest_path}")
    print(f"Jobs: {len(jobs)}")
    if args.emit_commands:
        for job in jobs:
            print(" ".join(job.command))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    manifest = _load_manifest(Path(args.manifest))
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if str(args.output_root).strip() else (
        manifest_path.parents[1] / "results" / manifest_path.stem
    )
    _ensure_dir(output_root)

    eval_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for job in manifest.get("jobs", []):
        run_eval_rows, run_metric_rows, run_summary = _collect_job(job)
        eval_rows.extend(run_eval_rows)
        metric_rows.extend(run_metric_rows)
        summaries.append(run_summary)

    curve_success = _aggregate_curves(eval_rows, "success_rate")
    curve_length = _aggregate_curves(eval_rows, "avg_length")
    best_success = _aggregate_summary(summaries, "best_success")
    final_success = _aggregate_summary(summaries, "final_success")
    best_length = _aggregate_summary(summaries, "best_avg_length")

    _write_csv(output_root / "eval_records.csv", eval_rows)
    _write_csv(output_root / "metric_records.csv", metric_rows)
    _write_csv(output_root / "run_summaries.csv", summaries)
    _write_csv(output_root / "success_curve_summary.csv", curve_success)
    _write_csv(output_root / "length_curve_summary.csv", curve_length)
    _write_csv(output_root / "best_success_summary.csv", best_success)
    _write_csv(output_root / "final_success_summary.csv", final_success)
    _write_csv(output_root / "best_length_summary.csv", best_length)

    metadata = {
        "manifest": str(manifest_path),
        "suite": manifest.get("suite", ""),
        "timestamp": manifest.get("timestamp", ""),
        "num_jobs": len(manifest.get("jobs", [])),
        "num_eval_rows": len(eval_rows),
        "num_metric_rows": len(metric_rows),
    }
    (output_root / "collection_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Collected results into {output_root}")
    return 0


def _read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _coerce_float(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def _configure_plotting() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 170,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def _ordered_keys(keys: Iterable[str]) -> list[str]:
    order = list(SUITES["all"])
    return sorted(keys, key=lambda key: order.index(key) if key in order else 10_000)


def _plot_curve_panel(ax: plt.Axes, rows: list[dict[str, Any]], title: str, ylabel: str) -> None:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[str(row["key"])].append(row)
    for key in _ordered_keys(by_key.keys()):
        spec = SPEC_BY_KEY[key]
        points = sorted(by_key[key], key=lambda row: _coerce_float(row, "step", 0.0))
        xs = [_coerce_float(row, "step", 0.0) for row in points]
        means = [_coerce_float(row, "mean", 0.0) for row in points]
        ci95 = [_coerce_float(row, "ci95", 0.0) for row in points]
        ax.plot(xs, means, label=spec.label, color=spec.color, linewidth=2)
        lower = [max(0.0, m - c) for m, c in zip(means, ci95)]
        upper = [m + c for m, c in zip(means, ci95)]
        ax.fill_between(xs, lower, upper, color=spec.color, alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("Offline step")
    ax.set_ylabel(ylabel)
    if "success" in ylabel.lower():
        ax.set_ylim(-0.02, 1.05)


def _plot_bar_panel(ax: plt.Axes, rows: list[dict[str, Any]], title: str, xlabel: str, xlim: tuple[float, float] | None = None) -> None:
    order = {key: idx for idx, key in enumerate(SUITES["all"])}
    rows = sorted(rows, key=lambda row: order.get(str(row["key"]), 10_000))
    labels = [str(row["label"]) for row in rows]
    means = [_coerce_float(row, "mean", 0.0) for row in rows]
    ci95 = [_coerce_float(row, "ci95", 0.0) for row in rows]
    colors = [SPEC_BY_KEY[str(row["key"])].color for row in rows]
    bars = ax.barh(labels, means, xerr=ci95, color=colors, capsize=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(*xlim)
    for bar, mean in zip(bars, means):
        ax.text(mean + 0.01 * max(ax.get_xlim()[1], 1.0), bar.get_y() + bar.get_height() / 2.0, f"{mean:.2f}", va="center", ha="left", fontsize=8)


def cmd_plot(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    input_root = Path(args.input_root).expanduser().resolve()
    plot_root = _ensure_dir(input_root / "plots")
    _configure_plotting()

    success_curve = _read_csv_dicts(input_root / "success_curve_summary.csv")
    length_curve = _read_csv_dicts(input_root / "length_curve_summary.csv")
    best_success = _read_csv_dicts(input_root / "best_success_summary.csv")
    best_length = _read_csv_dicts(input_root / "best_length_summary.csv")

    scratch_success = [row for row in success_curve if row.get("suite") == "scratch"]
    recovery_core_success = [row for row in success_curve if row.get("suite") == "recovery_core"]
    recovery_gates_success = [row for row in success_curve if row.get("suite") == "recovery_gates"]

    scratch_best = [row for row in best_success if row.get("suite") == "scratch"]
    recovery_core_best = [row for row in best_success if row.get("suite") == "recovery_core"]
    recovery_gates_best = [row for row in best_success if row.get("suite") == "recovery_gates"]

    scratch_best_length = [row for row in best_length if row.get("suite") == "scratch"]
    recovery_core_best_length = [row for row in best_length if row.get("suite") == "recovery_core"]
    recovery_gates_best_length = [row for row in best_length if row.get("suite") == "recovery_gates"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_curve_panel(axes[0], scratch_success, "From scratch on human replay", "Eval success rate")
    _plot_curve_panel(axes[1], recovery_core_success, "Checkpoint recovery on human replay", "Eval success rate")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_root / "success_curves_core.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_curve_panel(axes[0], recovery_gates_success, "Own-method filtering / weighting recovery", "Eval success rate")
    _plot_curve_panel(axes[1], [row for row in length_curve if row.get("suite") == "recovery_gates"], "Own-method filtering / weighting recovery", "Eval avg length")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_root / "success_curves_gates.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    _plot_bar_panel(axes[0, 0], scratch_best, "From scratch: mean best success", "Best eval success", (0.0, 1.05))
    _plot_bar_panel(axes[0, 1], recovery_core_best, "Recovery core: mean best success", "Best eval success", (0.0, 1.05))
    _plot_bar_panel(axes[1, 0], recovery_gates_best, "Own-method variants: mean best success", "Best eval success", (0.0, 1.05))

    protocol_rows = []
    lookup = {str(row["key"]): row for row in best_success}
    for key in ("own_weighted_scratch", "eps001_scale025"):
        row = lookup.get(key)
        if row is not None:
            protocol_rows.append(row)
    _plot_bar_panel(axes[1, 1], protocol_rows, "Protocol gap", "Best eval success", (0.0, 1.05))
    fig.tight_layout()
    fig.savefig(plot_root / "best_success_overview.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    _plot_bar_panel(axes[0], scratch_best_length, "From scratch: length at best", "Avg length")
    _plot_bar_panel(axes[1], recovery_core_best_length, "Recovery core: length at best", "Avg length")
    _plot_bar_panel(axes[2], recovery_gates_best_length, "Own-method variants: length at best", "Avg length")
    fig.tight_layout()
    fig.savefig(plot_root / "best_length_overview.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Plotted results into {plot_root}")
    return 0


def main() -> int:
    args = parse_args()
    if args.cmd == "launch":
        return cmd_launch(args)
    if args.cmd == "collect":
        return cmd_collect(args)
    if args.cmd == "plot":
        return cmd_plot(args)
    raise ValueError(f"unsupported command={args.cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main())

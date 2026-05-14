#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "local" / "thesis_manip_eval"
DEFAULT_PROJECT = "ogbench-manip-thesis-eval"


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    family: str
    variant: str
    fidelity: str
    launcher: str
    log_root: str
    default_overrides: tuple[str, ...]

    @property
    def launcher_path(self) -> Path:
        return REPO_ROOT / self.launcher

    @property
    def training_log_root(self) -> Path:
        return REPO_ROOT / "logs" / self.log_root

    def training_log_path(self, exp_name: str) -> Path:
        return self.training_log_root / exp_name / "training.log"

    def args_path(self, exp_name: str) -> Path:
        return self.training_log_root / exp_name / "args.json"


METHOD_SPECS: dict[str, MethodSpec] = {
    "own": MethodSpec(
        key="own",
        label="Ours",
        family="own",
        variant="canonical",
        fidelity="paper",
        launcher="scripts/run_cube_single_task1_relonly_angle_baseline_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "ALGO_VARIANT=own",
            "DISABLE_ROTATION=1",
            "NUM_UPDATES=7",
            "CTA_RATIO=2",
            "TOTAL_TIMESTEPS=120000",
            "GAMMA=0.97",
            "FIXED_ALPHA=0.001",
            "ALPHA_INIT=0.001",
            "ALPHA_MIN=0.001",
            "ALPHA_MAX=0.001",
            "ALPHA_FREEZE_STEPS=0",
            "DEMO_PREFILL_EPISODES=20",
            "DEMO_PREFILL_NUM_ENVS=20",
            "DEMO_SAMPLE_RATIO=0.5",
            "PREF_RANK_WEIGHT=1.0",
            "PREF_LOSS_TYPE=lagrangian",
            "PREF_RANK_MARGIN=0.01",
            "PREF_CRITIC_SCOPE=all",
            "STORE_INTERVENED_IN_DEMO_BUFFER=0",
        ),
    ),
    "own_perlinked_lambda": MethodSpec(
        key="own_perlinked_lambda",
        label="Ours (per-intervention lambda)",
        family="own",
        variant="per_linked_lambda",
        fidelity="ablation",
        launcher="scripts/run_cube_single_task1_relonly_angle_baseline_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "ALGO_VARIANT=own",
            "DISABLE_ROTATION=1",
            "NUM_UPDATES=7",
            "CTA_RATIO=2",
            "TOTAL_TIMESTEPS=120000",
            "GAMMA=0.97",
            "FIXED_ALPHA=0.001",
            "ALPHA_INIT=0.001",
            "ALPHA_MIN=0.001",
            "ALPHA_MAX=0.001",
            "ALPHA_FREEZE_STEPS=0",
            "DEMO_PREFILL_EPISODES=20",
            "DEMO_PREFILL_NUM_ENVS=20",
            "DEMO_SAMPLE_RATIO=0.5",
            "PREF_RANK_WEIGHT=1.0",
            "PREF_LOSS_TYPE=lagrangian",
            "PREF_LAGRANGIAN_SCOPE=per_linked",
            "PREF_RANK_MARGIN=0.01",
            "PREF_CRITIC_SCOPE=all",
            "STORE_INTERVENED_IN_DEMO_BUFFER=0",
        ),
    ),
    "hgdagger": MethodSpec(
        key="hgdagger",
        label="HG-DAgger",
        family="hgdagger",
        variant="canonical",
        fidelity="paper",
        launcher="scripts/run_cube_single_task1_relonly_angle_hgdagger_local.sh",
        log_root="hg_dagger",
        default_overrides=(
            "DISABLE_ROTATION=1",
            "NUM_UPDATES=7",
            "TOTAL_TIMESTEPS=120000",
            "GAMMA=0.97",
            "DEMO_PREFILL_EPISODES=20",
            "DEMO_PREFILL_NUM_ENVS=20",
            "INTERVENTION_EPISODE_PROB=1.0",
            "HG_ENSEMBLE_SIZE=5",
            "HG_DOUBT_PERCENTILE=75.0",
        ),
    ),
    "eil": MethodSpec(
        key="eil",
        label="EIL ablation",
        family="eil",
        variant="canonical",
        fidelity="paper",
        launcher="scripts/run_cube_single_task1_relonly_angle_eil_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "DISABLE_ROTATION=1",
            "NUM_UPDATES=7",
            "CTA_RATIO=2",
            "TOTAL_TIMESTEPS=120000",
            "GAMMA=0.97",
            "INTERVENTION_EPISODE_PROB=1.0",
            "FIXED_ALPHA=0.0",
            "ALPHA_INIT=0.0",
            "ALPHA_MIN=0.0",
            "ALPHA_MAX=0.0",
            "ALPHA_FREEZE_STEPS=0",
            "DEMO_PREFILL_EPISODES=0",
            "DEMO_PREFILL_NUM_ENVS=0",
            "DEMO_SAMPLE_RATIO=0.0",
            "EIL_THRESHOLD=0.0",
            "EIL_GOOD_MARGIN=0.0",
            "EIL_BAD_MARGIN=0.01",
            "EIL_PAIR_MARGIN=0.01",
            "EIL_BAD_PRE_STEPS=8",
        ),
    ),
    "pvp_td3": MethodSpec(
        key="pvp_td3",
        label="PVP faithful TD3",
        family="pvp",
        variant="faithful",
        fidelity="paper",
        launcher="scripts/run_cube_single_task1_relonly_angle_pvp_td3_local.sh",
        log_root="pvp_td3",
        default_overrides=(
            "DISABLE_ROTATION=1",
            "TOTAL_TIMESTEPS=120000",
            "INTERVENTION_EPISODE_PROB=1.0",
            "GAMMA=0.99",
            "NUM_UPDATES=1",
            "BATCH_SIZE=128",
            "LEARNING_STARTS=100",
            "PVP_POLICY_DELAY=2",
            "PVP_TARGET_POLICY_NOISE=0.2",
            "PVP_TARGET_NOISE_CLIP=0.5",
            "PVP_CQL_COEFFICIENT=1.0",
            "PVP_PROXY_VALUE_BOUND=1.0",
            "PVP_INCLUDE_ENV_REWARD_IN_TD=0",
            "PVP_BALANCE_SAMPLE=1",
            "PVP_STOP_TD_ON_INTERVENTION_START=1",
        ),
    ),
    "pvp_fastsac": MethodSpec(
        key="pvp_fastsac",
        label="PVP ablation",
        family="pvp",
        variant="upper_bound",
        fidelity="upper_bound",
        launcher="scripts/run_cube_single_task1_relonly_angle_pvp_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "DISABLE_ROTATION=1",
            "NUM_UPDATES=7",
            "CTA_RATIO=2",
            "TOTAL_TIMESTEPS=120000",
            "GAMMA=0.97",
            "INTERVENTION_EPISODE_PROB=0.5",
            "DEMO_PREFILL_EPISODES=0",
            "DEMO_PREFILL_NUM_ENVS=0",
            "DEMO_SAMPLE_RATIO=0.5",
            "FIXED_ALPHA=0.0",
            "ALPHA_INIT=0.0",
            "ALPHA_MIN=0.0",
            "ALPHA_MAX=0.0",
            "ALPHA_FREEZE_STEPS=0",
            "PVP_INCLUDE_ENV_REWARD_IN_TD=1",
        ),
    ),
    "hilserl_matched": MethodSpec(
        key="hilserl_matched",
        label="HIL-SERL ablation (100% supervised episodes)",
        family="hilserl",
        variant="matched_scaffold",
        fidelity="upper_bound",
        launcher="scripts/run_cube_single_task1_relonly_angle_hilserl_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "DISABLE_ROTATION=1",
            "NUM_UPDATES=7",
            "CTA_RATIO=2",
            "TOTAL_TIMESTEPS=120000",
            "GAMMA=0.97",
            "INTERVENTION_EPISODE_PROB=1.0",
            "FIXED_ALPHA=0.001",
            "ALPHA_INIT=0.001",
            "ALPHA_MIN=0.001",
            "ALPHA_MAX=0.001",
            "ALPHA_FREEZE_STEPS=0",
            "DEMO_PREFILL_EPISODES=20",
            "DEMO_PREFILL_NUM_ENVS=20",
            "DEMO_SAMPLE_RATIO=0.5",
            "STORE_INTERVENED_IN_DEMO_BUFFER=1",
            "PREF_RANK_WEIGHT=0.0",
            "PREF_SAMPLE_RATIO=0.0",
        ),
    ),
    "hilserl_supervision50": MethodSpec(
        key="hilserl_supervision50",
        label="HIL-SERL ablation (50% supervised episodes)",
        family="hilserl",
        variant="supervision50_norot",
        fidelity="upper_bound",
        launcher="scripts/run_cube_single_task1_hilserl_march_repro_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "DISABLE_ROTATION=1",
            "NUM_UPDATES=1",
            "CTA_RATIO=1",
            "TOTAL_TIMESTEPS=120000",
            "GAMMA=0.97",
            "INTERVENTION_EPISODE_PROB=0.5",
            "DEMO_PREFILL_EPISODES=20",
            "DEMO_PREFILL_NUM_ENVS=20",
            "DEMO_SAMPLE_RATIO=0.5",
            "NAME_SUFFIX=supervision50_norot",
        ),
    ),
    "hilserl_march": MethodSpec(
        key="hilserl_march",
        label="HIL-SERL ablation (March repro)",
        family="hilserl",
        variant="march_repro",
        fidelity="upper_bound",
        launcher="scripts/run_cube_single_task1_hilserl_march_repro_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "DISABLE_ROTATION=0",
            "NUM_UPDATES=1",
            "CTA_RATIO=1",
            "TOTAL_TIMESTEPS=120000",
            "GAMMA=0.97",
            "INTERVENTION_EPISODE_PROB=0.5",
            "DEMO_PREFILL_EPISODES=20",
            "DEMO_PREFILL_NUM_ENVS=20",
            "DEMO_SAMPLE_RATIO=0.5",
        ),
    ),
}

SUITES: dict[str, tuple[str, ...]] = {
    "paper": ("own", "hgdagger", "eil", "pvp_td3"),
    "all": ("own", "hgdagger", "eil", "pvp_td3", "pvp_fastsac", "hilserl_matched", "hilserl_supervision50"),
    "extended": ("own", "own_perlinked_lambda", "hgdagger", "eil", "pvp_td3", "pvp_fastsac", "hilserl_matched", "hilserl_supervision50", "hilserl_march"),
}

METHOD_COLORS = {
    "own": "#1b9e77",
    "own_perlinked_lambda": "#66c2a5",
    "hgdagger": "#d95f02",
    "eil": "#7570b3",
    "pvp_td3": "#e7298a",
    "pvp_fastsac": "#66a61e",
    "hilserl_matched": "#e6ab02",
    "hilserl_supervision50": "#a6761d",
    "hilserl_march": "#a6761d",
}

METHOD_ORDER = {method: idx for idx, method in enumerate(METHOD_SPECS.keys())}


@dataclass
class JobSpec:
    method: str
    label: str
    family: str
    variant: str
    fidelity: str
    seed: int
    replicate_id: str
    exp_name: str
    project: str
    wandb_entity: str
    wandb_group: str
    launcher: str
    training_log: str
    args_path: str
    command: list[str]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _suite_methods(suite: str, methods: Optional[list[str]]) -> list[str]:
    if methods:
        unknown = [method for method in methods if method not in METHOD_SPECS]
        if unknown:
            raise SystemExit(f"Unknown method keys: {', '.join(unknown)}")
        return methods
    if suite not in SUITES:
        raise SystemExit(f"Unknown suite {suite!r}; choices: {', '.join(sorted(SUITES))}")
    return list(SUITES[suite])


def _dedupe_overrides(overrides: Iterable[str]) -> list[str]:
    ordered: dict[str, str] = {}
    passthrough: list[str] = []
    for item in overrides:
        if "=" not in item:
            passthrough.append(item)
            continue
        key, _ = item.split("=", 1)
        ordered[key] = item
    return [*ordered.values(), *passthrough]


def _exp_name(run_prefix: str, method: str, seed: int, timestamp: str) -> str:
    return f"{run_prefix}_{method}_seed{seed}_{timestamp}"


def _format_command(cmd: Iterable[str]) -> str:
    return " \\\n  ".join(shlex.quote(part) for part in cmd)


def _load_matplotlib():
    try:
        if not os.environ.get("MPLCONFIGDIR"):
            mplconfigdir = Path("/tmp/matplotlib-codex")
            _ensure_dir(mplconfigdir)
            os.environ["MPLCONFIGDIR"] = str(mplconfigdir)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            "Plotting requires matplotlib. Install it in the active environment, "
            "for example: `pip install matplotlib`."
        ) from exc
    return plt


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    variance = statistics.fmean([(value - mean) ** 2 for value in values])
    return float(mean), float(math.sqrt(max(0.0, variance)))


def _build_job(
    *,
    method_key: str,
    seed: int,
    timestamp: str,
    args: argparse.Namespace,
) -> JobSpec:
    spec = METHOD_SPECS[method_key]
    exp_name = _exp_name(args.run_prefix, method_key, seed, timestamp)
    replicate_id = f"seed_{seed}"
    wandb_group = str(args.wandb_group or f"{args.run_prefix}_{timestamp}")
    overrides = list(spec.default_overrides)
    overrides.extend(
        (
            f"PROJECT={args.project}",
            f"SEED={seed}",
            f"TOTAL_TIMESTEPS={args.total_timesteps}",
            f"NUM_EVAL_EPISODES={args.num_eval_episodes}",
            f"EVAL_NUM_ENVS={args.eval_num_envs}",
            f"EVAL_INTERVAL={args.eval_interval}",
            f"SAVE_INTERVAL={args.save_interval}",
            f"LOG_INTERVAL={args.log_interval}",
            f"WANDB_MODE={args.wandb_mode}",
            f"EXP_NAME={exp_name}",
            f"WANDB_GROUP={wandb_group}",
        )
    )
    if args.wandb_entity:
        overrides.append(f"WANDB_ENTITY={args.wandb_entity}")
    deduped_overrides = _dedupe_overrides(overrides)
    cmd = ["bash", str(spec.launcher_path), *deduped_overrides]
    return JobSpec(
        method=spec.key,
        label=spec.label,
        family=spec.family,
        variant=spec.variant,
        fidelity=spec.fidelity,
        seed=int(seed),
        replicate_id=replicate_id,
        exp_name=exp_name,
        project=str(args.project),
        wandb_entity=str(args.wandb_entity),
        wandb_group=wandb_group,
        launcher=str(spec.launcher_path),
        training_log=str(spec.training_log_path(exp_name)),
        args_path=str(spec.args_path(exp_name)),
        command=cmd,
    )


def _write_manifest(*, jobs: list[JobSpec], args: argparse.Namespace, timestamp: str) -> Path:
    manifest_dir = _ensure_dir(Path(args.output_root) / "manifests")
    manifest_path = manifest_dir / f"{args.run_prefix}_{timestamp}.json"
    payload = {
        "created_at": timestamp,
        "suite": args.suite,
        "run_prefix": args.run_prefix,
        "project": args.project,
        "wandb_entity": args.wandb_entity,
        "wandb_group": args.wandb_group or f"{args.run_prefix}_{timestamp}",
        "seeds": list(args.seeds),
        "methods": sorted({job.method for job in jobs}),
        "num_runs": len(jobs),
        "launch_args": {
            "total_timesteps": args.total_timesteps,
            "num_eval_episodes": args.num_eval_episodes,
            "eval_num_envs": args.eval_num_envs,
            "eval_interval": args.eval_interval,
            "save_interval": args.save_interval,
            "log_interval": args.log_interval,
            "wandb_mode": args.wandb_mode,
        },
        "jobs": [asdict(job) for job in jobs],
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def _parse_eval_line(line: str) -> Optional[dict[str, float]]:
    if "[Eval]" not in line:
        return None
    tokens: dict[str, float] = {}
    for chunk in line.split():
        if "=" not in chunk:
            continue
        key, raw_value = chunk.split("=", 1)
        key = key.rstrip(",")
        raw_value = raw_value.rstrip(",")
        try:
            tokens[key] = float(raw_value)
        except ValueError:
            continue
    if "steps" not in tokens:
        return None
    # Also parse comma-separated payload after the step token.
    for segment in line.split(","):
        if "=" not in segment:
            continue
        key, raw_value = segment.strip().split("=", 1)
        try:
            tokens[key] = float(raw_value)
        except ValueError:
            continue
    return tokens


_TIMESTAMP_PREFIX_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
_PROGRESS_STEP_RE = re.compile(r"env_steps\s+(?P<step>\d+)/(?:\d+)")
_TEACHER_FRAC_RE = re.compile(r"teacher_frac(?:=|\s+)(?P<value>-?\d+(?:\.\d+)?)")
_NOVICE_COUNT_RE = re.compile(r"novice\s+(?P<value>\d+)")
_HUMAN_COUNT_RE = re.compile(r"human\s+(?P<value>\d+)")


def _parse_timestamp_prefix(line: str) -> Optional[datetime]:
    match = _TIMESTAMP_PREFIX_RE.match(line)
    if match is None:
        return None
    try:
        return datetime.fromisoformat(match.group("ts"))
    except ValueError:
        return None


def _parse_progress_line(line: str) -> Optional[dict[str, float]]:
    if "env_steps" not in line:
        return None
    step_match = _PROGRESS_STEP_RE.search(line)
    if step_match is None:
        return None
    teacher_match = _TEACHER_FRAC_RE.search(line)
    row: dict[str, float] = {"step": float(step_match.group("step"))}
    if teacher_match is not None:
        row["teacher_fraction"] = float(teacher_match.group("value"))
    novice_match = _NOVICE_COUNT_RE.search(line)
    if novice_match is not None:
        row["novice_steps_cumulative"] = float(novice_match.group("value"))
    human_match = _HUMAN_COUNT_RE.search(line)
    if human_match is not None:
        row["intervention_steps_cumulative_exact"] = float(human_match.group("value"))
    return row


def _infer_existing_method(root_name: str, run_name: str, args_data: dict[str, Any]) -> Optional[str]:
    algo_variant = str(args_data.get("algo_variant", "") or "").strip().lower()
    name = run_name.lower()
    if root_name == "hg_dagger" or algo_variant == "hg_dagger" or "hgdagger" in name:
        return "hgdagger"
    if root_name == "pvp_td3" or algo_variant == "pvp_td3":
        return "pvp_td3"
    if algo_variant == "eil" or "eil" in name:
        return "eil"
    if "hilserl" in name:
        if "supervision50" in name:
            return "hilserl_supervision50"
        if "march_repro" in name:
            return "hilserl_march"
        return "hilserl_matched"
    if algo_variant == "pvp" or "pvp" in name:
        return "pvp_fastsac"
    if any(token in name for token in ("own", "autores", "fixedalpha", "baseline", "scripted")):
        return "own"
    return None


def _load_manifest(path: Path) -> list[JobSpec]:
    payload = json.loads(path.read_text())
    return [JobSpec(**job) for job in payload["jobs"]]


def _scan_existing_jobs(methods: set[str]) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    for root_name in ("fast_sac", "hg_dagger", "pvp_td3"):
        root = REPO_ROOT / "logs" / root_name
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            training_log = run_dir / "training.log"
            args_path = run_dir / "args.json"
            if not training_log.exists():
                continue
            args_data: dict[str, Any] = {}
            if args_path.exists():
                try:
                    args_data = json.loads(args_path.read_text())
                except Exception:
                    args_data = {}
            method = _infer_existing_method(root_name, run_dir.name, args_data)
            if method is None or method not in methods:
                continue
            spec = METHOD_SPECS[method]
            seed_value = args_data.get("seed")
            try:
                seed_int = int(seed_value) if seed_value is not None else -1
            except Exception:
                seed_int = -1
            replicate_id = f"seed_{seed_int}" if seed_int >= 0 else run_dir.name
            jobs.append(
                JobSpec(
                    method=method,
                    label=spec.label,
                    family=spec.family,
                    variant=spec.variant,
                    fidelity=spec.fidelity,
                    seed=seed_int,
                    replicate_id=replicate_id,
                    exp_name=run_dir.name,
                    project=str(args_data.get("project", "")),
                    wandb_entity=str(args_data.get("wandb_entity", "")),
                    wandb_group=str(args_data.get("wandb_group", "")),
                    launcher=str(spec.launcher_path),
                    training_log=str(training_log),
                    args_path=str(args_path),
                    command=[],
                )
            )
    return jobs


def _collect_records(jobs: list[JobSpec]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eval_rows: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for job in jobs:
        training_log = Path(job.training_log)
        args_path = Path(job.args_path)
        if not training_log.exists():
            summaries.append(
                {
                    "method": job.method,
                    "label": job.label,
                    "family": job.family,
                    "variant": job.variant,
                    "fidelity": job.fidelity,
                    "seed": job.seed,
                    "replicate_id": job.replicate_id,
                    "exp_name": job.exp_name,
                    "training_log_exists": 0,
                    "args_exists": int(args_path.exists()),
                }
            )
            continue
        args_data: dict[str, Any] = {}
        if args_path.exists():
            try:
                args_data = json.loads(args_path.read_text())
            except Exception:
                args_data = {}
        run_rows: list[dict[str, Any]] = []
        run_progress_rows: list[dict[str, Any]] = []
        start_ts: Optional[datetime] = None
        end_ts: Optional[datetime] = None
        prev_progress_step = 0.0
        cumulative_intervention_steps_est = 0.0
        last_teacher_fraction = math.nan
        for line in training_log.read_text(errors="ignore").splitlines():
            line_ts = _parse_timestamp_prefix(line)
            if line_ts is not None:
                if start_ts is None:
                    start_ts = line_ts
                end_ts = line_ts
            tokens = _parse_eval_line(line)
            if tokens is not None:
                row = {
                    "method": job.method,
                    "label": job.label,
                    "family": job.family,
                    "variant": job.variant,
                    "fidelity": job.fidelity,
                    "seed": job.seed,
                    "replicate_id": job.replicate_id,
                    "exp_name": job.exp_name,
                    "step": int(tokens.pop("steps")),
                }
                for key, value in list(tokens.items()):
                    normalized = key.replace("Eval/", "").replace("eval/", "")
                    row[normalized] = float(value)
                if line_ts is not None:
                    row["timestamp_iso"] = line_ts.isoformat()
                if start_ts is not None and line_ts is not None:
                    row["wall_time_sec"] = float((line_ts - start_ts).total_seconds())
                    row["wall_time_min"] = float((line_ts - start_ts).total_seconds() / 60.0)
                run_rows.append(row)
                continue
            progress = _parse_progress_line(line)
            if progress is None:
                continue
            progress_step = float(progress["step"])
            interval_steps = max(0.0, progress_step - prev_progress_step)
            teacher_fraction = float(progress["teacher_fraction"]) if "teacher_fraction" in progress else math.nan
            if not math.isnan(teacher_fraction):
                last_teacher_fraction = teacher_fraction
            teacher_fraction_filled = last_teacher_fraction if not math.isnan(last_teacher_fraction) else 0.0
            cumulative_intervention_steps_est += teacher_fraction_filled * interval_steps
            row = {
                "method": job.method,
                "label": job.label,
                "family": job.family,
                "variant": job.variant,
                "fidelity": job.fidelity,
                "seed": job.seed,
                "replicate_id": job.replicate_id,
                "exp_name": job.exp_name,
                "step": int(progress_step),
                "interval_steps": float(interval_steps),
                "intervention_steps_cumulative_est": float(cumulative_intervention_steps_est),
                "teacher_fraction_filled": float(teacher_fraction_filled),
            }
            if "teacher_fraction" in progress:
                row["teacher_fraction"] = teacher_fraction
                row["intervention_steps_interval_est"] = float(teacher_fraction * interval_steps)
            row["intervention_steps_interval_est"] = float(teacher_fraction_filled * interval_steps)
            if progress_step > 0:
                row["intervention_fraction_cumulative_est"] = float(cumulative_intervention_steps_est / progress_step)
            if "novice_steps_cumulative" in progress:
                row["novice_steps_cumulative"] = float(progress["novice_steps_cumulative"])
            if "intervention_steps_cumulative_exact" in progress:
                row["intervention_steps_cumulative_exact"] = float(progress["intervention_steps_cumulative_exact"])
            if line_ts is not None:
                row["timestamp_iso"] = line_ts.isoformat()
            if start_ts is not None and line_ts is not None:
                row["wall_time_sec"] = float((line_ts - start_ts).total_seconds())
                row["wall_time_min"] = float((line_ts - start_ts).total_seconds() / 60.0)
            run_progress_rows.append(row)
            prev_progress_step = max(prev_progress_step, progress_step)
        run_rows.sort(key=lambda item: item["step"])
        run_progress_rows.sort(key=lambda item: item["step"])
        eval_rows.extend(run_rows)
        progress_rows.extend(run_progress_rows)
        if not run_rows:
            summaries.append(
                {
                    "method": job.method,
                    "label": job.label,
                    "family": job.family,
                    "variant": job.variant,
                    "fidelity": job.fidelity,
                    "seed": job.seed,
                    "replicate_id": job.replicate_id,
                    "exp_name": job.exp_name,
                    "training_log_exists": 1,
                    "args_exists": int(args_path.exists()),
                }
            )
            continue
        best = max(run_rows, key=lambda item: (float(item.get("success_rate", 0.0)), -float(item.get("avg_length", 1e9))))
        final = run_rows[-1]
        teacher_values = [float(row["teacher_fraction_filled"]) for row in run_progress_rows if "teacher_fraction_filled" in row]
        total_intervention_steps_est = float(run_progress_rows[-1]["intervention_steps_cumulative_est"]) if run_progress_rows else math.nan
        total_intervention_steps_exact = math.nan
        for progress_row in reversed(run_progress_rows):
            if "intervention_steps_cumulative_exact" in progress_row:
                total_intervention_steps_exact = float(progress_row["intervention_steps_cumulative_exact"])
                break
        final_wall_time_sec = float((end_ts - start_ts).total_seconds()) if start_ts is not None and end_ts is not None else math.nan
        summaries.append(
            {
                "method": job.method,
                "label": job.label,
                "family": job.family,
                "variant": job.variant,
                "fidelity": job.fidelity,
                "seed": job.seed,
                "replicate_id": job.replicate_id,
                "exp_name": job.exp_name,
                "project": job.project or args_data.get("project", ""),
                "training_log_exists": 1,
                "args_exists": int(args_path.exists()),
                "best_step": best["step"],
                "best_success": best.get("success_rate", math.nan),
                "best_avg_length": best.get("avg_length", math.nan),
                "best_avg_return": best.get("avg_return", math.nan),
                "final_step": final["step"],
                "final_success": final.get("success_rate", math.nan),
                "final_avg_length": final.get("avg_length", math.nan),
                "final_avg_return": final.get("avg_return", math.nan),
                "final_wall_time_sec": final_wall_time_sec,
                "final_wall_time_min": float(final_wall_time_sec / 60.0) if not math.isnan(final_wall_time_sec) else math.nan,
                "mean_teacher_fraction": float(statistics.fmean(teacher_values)) if teacher_values else math.nan,
                "max_teacher_fraction": float(max(teacher_values)) if teacher_values else math.nan,
                "final_teacher_fraction": float(teacher_values[-1]) if teacher_values else math.nan,
                "total_intervention_steps_est": total_intervention_steps_est,
                "total_intervention_fraction_est": float(total_intervention_steps_est / float(final["step"])) if not math.isnan(total_intervention_steps_est) and float(final["step"]) > 0 else math.nan,
                "total_intervention_steps_exact": total_intervention_steps_exact,
                "num_eval_points": len(run_rows),
                "num_progress_points": len(run_progress_rows),
            }
        )
    return eval_rows, progress_rows, summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_curve_rows(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    x_metric: str,
    group_metric: str = "step",
) -> dict[str, list[dict[str, float]]]:
    grouped: dict[tuple[str, float], list[dict[str, float]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        if metric not in row or x_metric not in row or group_metric not in row:
            continue
        method = str(row["method"])
        try:
            group_value = float(row[group_metric])
            x_value = float(row[x_metric])
            y_value = float(row[metric])
        except Exception:
            continue
        grouped[(method, group_value)].append({"x": x_value, "y": y_value})
        labels[method] = str(row["label"])
    aggregated: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (method, group_value), samples in grouped.items():
        if not samples:
            continue
        y_values = [sample["y"] for sample in samples]
        x_values = [sample["x"] for sample in samples]
        mean, std = _mean_std(y_values)
        sem = float(std / math.sqrt(len(y_values))) if y_values else 0.0
        ci95 = float(1.96 * sem)
        aggregated[method].append(
            {
                "step": float(group_value),
                "x": float(statistics.fmean(x_values)),
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95": ci95,
                "n": float(len(y_values)),
                "label": labels.get(method, method),
            }
        )
    for values in aggregated.values():
        values.sort(key=lambda item: item["x"])
    return aggregated


def _aggregate_summary_rows(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    exemplar: dict[str, dict[str, Any]] = {}
    for row in rows:
        if metric not in row:
            continue
        try:
            value = float(row[metric])
        except Exception:
            continue
        method = str(row["method"])
        grouped[method].append(value)
        exemplar[method] = row
    summary: list[dict[str, Any]] = []
    for method, values in grouped.items():
        mean, std = _mean_std(values)
        sem = float(std / math.sqrt(len(values))) if values else 0.0
        entry = dict(exemplar[method])
        entry.update(
            {
                "metric": metric,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95": float(1.96 * sem),
                "n": int(len(values)),
            }
        )
        summary.append(entry)
    summary.sort(key=lambda item: (METHOD_ORDER.get(str(item["method"]), 10**9), str(item["label"])))
    return summary


def _smooth_series(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return list(values)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2
    smoothed: list[float] = []
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        smoothed.append(float(statistics.fmean(values[lo:hi])))
    return smoothed


def _smooth_aggregated_rows(
    aggregated: dict[str, list[dict[str, float]]],
    *,
    window: int,
) -> dict[str, list[dict[str, float]]]:
    if window <= 1:
        return aggregated
    smoothed: dict[str, list[dict[str, float]]] = {}
    for method, points in aggregated.items():
        means = _smooth_series([float(point["mean"]) for point in points], window)
        stds = _smooth_series([float(point["std"]) for point in points], window)
        sems = _smooth_series([float(point["sem"]) for point in points], window)
        ci95s = _smooth_series([float(point["ci95"]) for point in points], window)
        smoothed_points: list[dict[str, float]] = []
        for idx, point in enumerate(points):
            updated = dict(point)
            updated["mean"] = means[idx]
            updated["std"] = stds[idx]
            updated["sem"] = sems[idx]
            updated["ci95"] = ci95s[idx]
            smoothed_points.append(updated)
        smoothed[method] = smoothed_points
    return smoothed


def _effective_teacher_fraction_window(
    aggregated: dict[str, list[dict[str, float]]],
    *,
    requested: int,
) -> int:
    if requested <= 1 or not aggregated:
        return max(1, int(requested))
    max_points = max((len(points) for points in aggregated.values()), default=0)
    dynamic_floor = max(1, max_points // 10)
    window = max(int(requested), dynamic_floor)
    if window % 2 == 0:
        window += 1
    return window


def _prepend_anchor_point(
    aggregated: dict[str, list[dict[str, float]]],
    *,
    anchor_y: float,
    anchor_x: float = 0.0,
    anchor_step: float = 0.0,
) -> dict[str, list[dict[str, float]]]:
    anchored: dict[str, list[dict[str, float]]] = {}
    for method, points in aggregated.items():
        if not points:
            anchored[method] = list(points)
            continue
        first = points[0]
        if float(first.get("x", math.inf)) <= anchor_x:
            anchored[method] = list(points)
            continue
        anchor = dict(first)
        anchor["x"] = float(anchor_x)
        anchor["step"] = float(anchor_step)
        anchor["mean"] = float(anchor_y)
        anchor["std"] = 0.0
        anchor["sem"] = 0.0
        anchor["ci95"] = 0.0
        anchored[method] = [anchor, *points]
    return anchored


def _n_text_from_curve(aggregated: dict[str, list[dict[str, float]]]) -> str:
    ns = sorted(
        {
            int(round(float(point["n"])))
            for points in aggregated.values()
            for point in points
            if "n" in point and not math.isnan(float(point["n"]))
        }
    )
    if not ns:
        return "n unavailable"
    if len(ns) == 1:
        return f"n={ns[0]} runs/method"
    return "n varies by method"


def _n_text_from_summary(aggregated_rows: list[dict[str, Any]]) -> str:
    ns = sorted(
        {
            int(row["n"])
            for row in aggregated_rows
            if "n" in row and str(row["n"]) not in {"", "nan", "None"}
        }
    )
    if not ns:
        return "n unavailable"
    if len(ns) == 1:
        return f"n={ns[0]} runs/method"
    return "n varies by method"


def _uncertainty_text(uncertainty: str) -> str:
    if uncertainty == "std":
        return "±1 std"
    if uncertainty == "sem":
        return "±1 sem"
    return "95% CI"


def _plot_learning_curve(
    *,
    aggregated: dict[str, list[dict[str, float]]],
    ylabel: str,
    xlabel: str,
    title: str,
    uncertainty: str,
    out_path: Path,
    x_scale: float = 1.0,
    ylim: Optional[tuple[float, float]] = None,
    clamp_band: bool = False,
    show_markers: bool = True,
    band_alpha: float = 0.18,
    footer_note: str = "",
) -> None:
    if not aggregated:
        return
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6.8))
    all_xs: list[float] = []
    for method in sorted(aggregated, key=lambda item: METHOD_ORDER.get(item, 10**9)):
        points = aggregated[method]
        xs = [point["x"] / x_scale for point in points]
        ys = [point["mean"] for point in points]
        band = [point[uncertainty] for point in points]
        label = str(points[0].get("label", method))
        color = METHOD_COLORS.get(method, None)
        ax.plot(
            xs,
            ys,
            label=label,
            linewidth=2.0,
            color=color,
            marker="o" if show_markers else None,
            markersize=4 if show_markers else 0,
        )
        lower = [y - b for y, b in zip(ys, band)]
        upper = [y + b for y, b in zip(ys, band)]
        if clamp_band and ylim is not None:
            lower = [max(ylim[0], value) for value in lower]
            upper = [min(ylim[1], value) for value in upper]
        ax.fill_between(xs, lower, upper, alpha=band_alpha, color=color)
        all_xs.extend(xs)
    if all_xs and len(set(all_xs)) == 1:
        center = all_xs[0]
        ax.set_xlim(center - 0.5, center + 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    if footer_note:
        fig.text(0.5, 0.01, footer_note, ha="center", va="bottom", fontsize=8.5, linespacing=1.15)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    else:
        fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_summary_bar(
    *,
    aggregated_rows: list[dict[str, Any]],
    title: str,
    ylabel: str,
    uncertainty: str,
    out_path: Path,
    value_scale: float = 1.0,
    ylim: Optional[tuple[float, float]] = None,
    footer_note: str = "",
) -> None:
    if not aggregated_rows:
        return
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6.8))
    labels = [str(row["label"]) for row in aggregated_rows]
    xs = list(range(len(labels)))
    means = [float(row["mean"]) / value_scale for row in aggregated_rows]
    errs = [float(row[uncertainty]) / value_scale for row in aggregated_rows]
    colors = [METHOD_COLORS.get(str(row["method"]), "#4c4c4c") for row in aggregated_rows]
    ax.bar(xs, means, yerr=errs, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    if footer_note:
        fig.text(0.5, 0.01, footer_note, ha="center", va="bottom", fontsize=8.5, linespacing=1.15)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    else:
        fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _cmd_launch(args: argparse.Namespace) -> int:
    timestamp = _timestamp()
    method_keys = _suite_methods(args.suite, args.methods)
    jobs = [
        _build_job(method_key=method_key, seed=int(seed), timestamp=timestamp, args=args)
        for method_key in method_keys
        for seed in args.seeds
    ]
    manifest_path = _write_manifest(jobs=jobs, args=args, timestamp=timestamp)
    print(f"Manifest: {manifest_path}")
    for job in jobs:
        print()
        print(f"[{job.method} seed={job.seed}]")
        print(_format_command(job.command))
    if args.mode == "print":
        return 0
    failures = 0
    for job in jobs:
        print(f"\n=== Running {job.exp_name} ===", flush=True)
        result = subprocess.run(job.command, cwd=str(REPO_ROOT), check=False)
        if result.returncode != 0:
            failures += 1
            print(f"[Launch] job failed rc={result.returncode}: {job.exp_name}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break
    return 1 if failures else 0


def _cmd_collect(args: argparse.Namespace) -> int:
    output_dir = _ensure_dir(Path(args.output_dir))
    if args.manifest:
        jobs = _load_manifest(Path(args.manifest))
    else:
        method_keys = set(_suite_methods(args.suite, args.methods))
        jobs = _scan_existing_jobs(method_keys)
    eval_rows, progress_rows, summary_rows = _collect_records(jobs)
    _write_csv(output_dir / "eval_records.csv", eval_rows)
    _write_csv(output_dir / "progress_records.csv", progress_rows)
    _write_csv(output_dir / "run_summaries.csv", summary_rows)
    metadata = {
        "manifest": args.manifest or "",
        "suite": args.suite,
        "methods": [job.method for job in jobs],
        "num_jobs": len(jobs),
        "num_eval_rows": len(eval_rows),
        "num_progress_rows": len(progress_rows),
        "created_at": _timestamp(),
    }
    (output_dir / "collection_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Collected {len(eval_rows)} eval rows and {len(progress_rows)} progress rows across {len(jobs)} jobs into {output_dir}")
    return 0


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _filter_rows_by_methods(rows: list[dict[str, Any]], methods: Optional[list[str]]) -> list[dict[str, Any]]:
    allowed: Optional[set[str]] = None
    if methods:
        unknown = [method for method in methods if method not in METHOD_SPECS]
        if unknown:
            raise SystemExit(f"Unknown method keys: {', '.join(unknown)}")
        allowed = set(methods)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        method = row.get("method")
        if method is None:
            continue
        if allowed is not None and method not in allowed:
            continue
        updated = dict(row)
        if method in METHOD_SPECS:
            updated["label"] = METHOD_SPECS[method].label
        filtered.append(updated)
    return filtered


def _cmd_plot(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    eval_rows = _read_csv(input_dir / "eval_records.csv")
    progress_rows = _read_csv(input_dir / "progress_records.csv")
    summary_rows = _read_csv(input_dir / "run_summaries.csv")
    eval_rows = _filter_rows_by_methods(eval_rows, args.methods)
    progress_rows = _filter_rows_by_methods(progress_rows, args.methods)
    summary_rows = _filter_rows_by_methods(summary_rows, args.methods)
    plots_dir = _ensure_dir(Path(args.output_dir))
    for metric, ylabel in (
        ("success_rate", "Eval Success Rate"),
        ("avg_length", "Eval Avg Episode Length"),
        ("avg_return", "Eval Avg Return"),
    ):
        is_fraction_metric = metric == "success_rate"
        aggregated = _aggregate_curve_rows(eval_rows, metric=metric, x_metric="step", group_metric="step")
        if metric == "success_rate":
            aggregated = _prepend_anchor_point(aggregated, anchor_y=0.0)
        direction_text = "Higher is better" if metric in {"success_rate", "avg_return"} else "Lower is better"
        footer_note = (
            f"Mean across seeds; {_uncertainty_text(args.uncertainty)}; {_n_text_from_curve(aggregated)}\n"
            f"{direction_text}"
        )
        _plot_learning_curve(
            aggregated=aggregated,
            ylabel=f"{ylabel} ({direction_text.lower()})",
            xlabel="Environment Steps (thousands)",
            title=f"Manipulation {ylabel} vs Env Steps",
            uncertainty=args.uncertainty,
            out_path=plots_dir / f"{metric}_curve.png",
            x_scale=1000.0,
            ylim=(0.0, 1.05) if is_fraction_metric else None,
            clamp_band=is_fraction_metric,
            footer_note=footer_note,
        )
        aggregated_wall = _aggregate_curve_rows(eval_rows, metric=metric, x_metric="wall_time_sec", group_metric="step")
        if metric == "success_rate":
            aggregated_wall = _prepend_anchor_point(aggregated_wall, anchor_y=0.0)
        _plot_learning_curve(
            aggregated=aggregated_wall,
            ylabel=f"{ylabel} ({direction_text.lower()})",
            xlabel="Wall Time (minutes)",
            title=f"Manipulation {ylabel} vs Wall Time",
            uncertainty=args.uncertainty,
            out_path=plots_dir / f"{metric}_wall_time_curve.png",
            x_scale=60.0,
            ylim=(0.0, 1.05) if is_fraction_metric else None,
            clamp_band=is_fraction_metric,
            footer_note=footer_note,
        )
    teacher_aggregated_raw = _aggregate_curve_rows(progress_rows, metric="teacher_fraction", x_metric="step", group_metric="step")
    teacher_aggregated_filled = _aggregate_curve_rows(progress_rows, metric="teacher_fraction_filled", x_metric="step", group_metric="step")
    teacher_window = _effective_teacher_fraction_window(
        teacher_aggregated_filled,
        requested=args.teacher_fraction_smoothing_window,
    )
    teacher_aggregated = _smooth_aggregated_rows(teacher_aggregated_filled, window=teacher_window)
    teacher_aggregated = _prepend_anchor_point(teacher_aggregated, anchor_y=1.0)
    teacher_aggregated_raw = _prepend_anchor_point(teacher_aggregated_raw, anchor_y=1.0)
    teacher_footer_note = (
        f"Mean across seeds; {_uncertainty_text(args.uncertainty)}; {_n_text_from_curve(teacher_aggregated)}\n"
        f"Lower is better; anchor fixed at step 0 = 1.0"
    )
    _plot_learning_curve(
        aggregated=teacher_aggregated,
        ylabel="Teacher Fraction (lower is better)",
        xlabel="Environment Steps (thousands)",
        title="Manipulation Teacher Fraction vs Env Steps",
        uncertainty=args.uncertainty,
        out_path=plots_dir / "teacher_fraction_curve.png",
        x_scale=1000.0,
        ylim=(0.0, 1.05),
        clamp_band=True,
        show_markers=False,
        band_alpha=0.04,
        footer_note=teacher_footer_note,
    )
    _plot_learning_curve(
        aggregated=teacher_aggregated_raw,
        ylabel="Teacher Fraction (lower is better)",
        xlabel="Environment Steps (thousands)",
        title="Manipulation Teacher Fraction vs Env Steps (raw)",
        uncertainty=args.uncertainty,
        out_path=plots_dir / "teacher_fraction_curve_raw.png",
        x_scale=1000.0,
        ylim=(0.0, 1.05),
        clamp_band=True,
        show_markers=False,
        band_alpha=0.03,
        footer_note=teacher_footer_note,
    )
    teacher_wall_raw = _aggregate_curve_rows(progress_rows, metric="teacher_fraction", x_metric="wall_time_sec", group_metric="step")
    teacher_wall_filled = _aggregate_curve_rows(progress_rows, metric="teacher_fraction_filled", x_metric="wall_time_sec", group_metric="step")
    teacher_wall_window = _effective_teacher_fraction_window(
        teacher_wall_filled,
        requested=args.teacher_fraction_smoothing_window,
    )
    teacher_wall = _smooth_aggregated_rows(teacher_wall_filled, window=teacher_wall_window)
    teacher_wall = _prepend_anchor_point(teacher_wall, anchor_y=1.0)
    teacher_wall_raw = _prepend_anchor_point(teacher_wall_raw, anchor_y=1.0)
    _plot_learning_curve(
        aggregated=teacher_wall,
        ylabel="Teacher Fraction (lower is better)",
        xlabel="Wall Time (minutes)",
        title="Manipulation Teacher Fraction vs Wall Time",
        uncertainty=args.uncertainty,
        out_path=plots_dir / "teacher_fraction_wall_time_curve.png",
        x_scale=60.0,
        ylim=(0.0, 1.05),
        clamp_band=True,
        show_markers=False,
        band_alpha=0.04,
        footer_note=teacher_footer_note,
    )
    _plot_learning_curve(
        aggregated=teacher_wall_raw,
        ylabel="Teacher Fraction (lower is better)",
        xlabel="Wall Time (minutes)",
        title="Manipulation Teacher Fraction vs Wall Time (raw)",
        uncertainty=args.uncertainty,
        out_path=plots_dir / "teacher_fraction_wall_time_curve_raw.png",
        x_scale=60.0,
        ylim=(0.0, 1.05),
        clamp_band=True,
        show_markers=False,
        band_alpha=0.03,
        footer_note=teacher_footer_note,
    )
    for metric, title in (
        ("final_success", "Final Success Rate by Method"),
        ("best_success", "Best Success Rate by Method"),
    ):
        aggregated_summary = _aggregate_summary_rows(summary_rows, metric)
        _write_csv(plots_dir / f"{metric}_summary.csv", aggregated_summary)
        footer_note = (
            f"Mean across seeds; {_uncertainty_text(args.uncertainty)}; {_n_text_from_summary(aggregated_summary)}\n"
            f"Higher is better"
        )
        _plot_summary_bar(
            aggregated_rows=aggregated_summary,
            title=title,
            ylabel="Success Rate (higher is better)",
            uncertainty=args.uncertainty,
            out_path=plots_dir / f"{metric}_bar.png",
            ylim=(0.0, 1.05),
            footer_note=footer_note,
        )
    for metric, title, ylabel, scale, filename in (
        ("final_wall_time_sec", "Final Wall Time by Method", "Wall Time (minutes)", 60.0, "final_wall_time_bar.png"),
        ("mean_teacher_fraction", "Mean Teacher Fraction by Method", "Teacher Fraction (lower is better)", 1.0, "mean_teacher_fraction_bar.png"),
        (
            "total_intervention_steps_est",
            "Estimated Total Intervention Steps by Method",
            "Estimated teacher-controlled env steps (thousands)",
            1000.0,
            "total_intervention_steps_bar.png",
        ),
        (
            "total_intervention_fraction_est",
            "Estimated Total Intervention Fraction by Method",
            "Estimated intervention fraction (lower is better)",
            1.0,
            "total_intervention_fraction_bar.png",
        ),
    ):
        aggregated_summary = _aggregate_summary_rows(summary_rows, metric)
        _write_csv(plots_dir / f"{metric}_summary.csv", aggregated_summary)
        direction_text = "Lower is better" if metric in {"final_wall_time_sec", "mean_teacher_fraction", "total_intervention_fraction_est"} else "Context dependent"
        footer_note = (
            f"Mean across seeds; {_uncertainty_text(args.uncertainty)}; {_n_text_from_summary(aggregated_summary)}\n"
            f"{direction_text}"
        )
        _plot_summary_bar(
            aggregated_rows=aggregated_summary,
            title=title,
            ylabel=ylabel,
            uncertainty=args.uncertainty,
            out_path=plots_dir / filename,
            value_scale=scale,
            ylim=(0.0, 1.05) if metric in {"mean_teacher_fraction", "total_intervention_fraction_est"} else None,
            footer_note=footer_note,
        )
    print(f"Plots written to {plots_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manipulation thesis evaluation harness")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    launch = subparsers.add_parser("launch", help="Generate or run a seeded method matrix via existing launcher scripts.")
    launch.add_argument("--suite", type=str, default="all", choices=sorted(SUITES))
    launch.add_argument("--methods", nargs="*", default=None, help="Optional explicit method keys overriding the suite.")
    launch.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    launch.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    launch.add_argument("--wandb-entity", type=str, default="")
    launch.add_argument("--wandb-group", type=str, default="")
    launch.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    launch.add_argument("--run-prefix", type=str, default="thesis_manip_eval")
    launch.add_argument("--total-timesteps", type=int, default=120000)
    launch.add_argument("--num-eval-episodes", type=int, default=20)
    launch.add_argument("--eval-num-envs", type=int, default=5)
    launch.add_argument("--eval-interval", type=int, default=5000)
    launch.add_argument("--save-interval", type=int, default=10000)
    launch.add_argument("--log-interval", type=int, default=64)
    launch.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    launch.add_argument("--mode", type=str, default="print", choices=["print", "run"])
    launch.add_argument("--continue-on-error", action="store_true", default=False)

    collect = subparsers.add_parser("collect", help="Collect eval traces from a manifest or from existing local logs.")
    collect.add_argument("--manifest", type=str, default="")
    collect.add_argument("--suite", type=str, default="all", choices=sorted(SUITES))
    collect.add_argument("--methods", nargs="*", default=None)
    collect.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT / "latest_collection"))

    plot = subparsers.add_parser("plot", help="Plot learning curves and summary bars from a collected dataset.")
    plot.add_argument("--input-dir", type=str, required=True)
    plot.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT / "latest_plots"))
    plot.add_argument("--methods", nargs="*", default=None)
    plot.add_argument("--uncertainty", type=str, default="ci95", choices=["std", "sem", "ci95"])
    plot.add_argument("--teacher-fraction-smoothing-window", type=int, default=61)

    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.cmd == "launch":
        return _cmd_launch(args)
    if args.cmd == "collect":
        return _cmd_collect(args)
    if args.cmd == "plot":
        return _cmd_plot(args)
    raise SystemExit(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import thesis_manip_eval as common


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "local" / "thesis_maze_eval"
DEFAULT_PROJECT = "ogbench-maze-thesis-eval"


BASE_METHOD_SPECS: dict[str, common.MethodSpec] = {
    "own": common.MethodSpec(
        key="own",
        label="Ours",
        family="own",
        variant="canonical",
        fidelity="paper",
        launcher="scripts/run_pointmaze_arena_state_own_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "ALGO_VARIANT=own",
            "REWARD_TYPE=dense",
            "DENSE_REWARD_SCALE=1.0",
            "NUM_UPDATES=7",
            "CTA_RATIO=2",
            "DEMO_BUFFER_ENABLE=0",
            "DEMO_PREFILL_EPISODES=0",
            "DEMO_SAMPLE_RATIO=0.0",
            "PREF_SAMPLE_RATIO=0.5",
            "PREF_RANK_WEIGHT=1.0",
            "PREF_LOSS_TYPE=lagrangian",
            "PREF_RANK_MARGIN=0.1",
            "PREF_CRITIC_SCOPE=all",
        ),
    ),
    "own_perlinked_lambda": common.MethodSpec(
        key="own_perlinked_lambda",
        label="Ours (per-intervention lambda)",
        family="own",
        variant="per_linked_lambda",
        fidelity="ablation",
        launcher="scripts/run_pointmaze_arena_state_own_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "ALGO_VARIANT=own",
            "REWARD_TYPE=dense",
            "DENSE_REWARD_SCALE=1.0",
            "NUM_UPDATES=7",
            "CTA_RATIO=2",
            "DEMO_BUFFER_ENABLE=0",
            "DEMO_PREFILL_EPISODES=0",
            "DEMO_SAMPLE_RATIO=0.0",
            "PREF_SAMPLE_RATIO=0.5",
            "PREF_RANK_WEIGHT=1.0",
            "PREF_LOSS_TYPE=lagrangian",
            "PREF_LAGRANGIAN_SCOPE=per_linked",
            "PREF_RANK_MARGIN=0.1",
            "PREF_CRITIC_SCOPE=all",
        ),
    ),
    "hgdagger": common.MethodSpec(
        key="hgdagger",
        label="HG-DAgger",
        family="hgdagger",
        variant="canonical",
        fidelity="paper",
        launcher="scripts/run_pointmaze_arena_state_hgdagger_local.sh",
        log_root="hg_dagger",
        default_overrides=(
            "REWARD_TYPE=dense",
            "DENSE_REWARD_SCALE=1.0",
            "NUM_UPDATES=1",
            "HG_ENSEMBLE_SIZE=5",
            "HG_DOUBT_PERCENTILE=75.0",
            "DEMO_PREFILL_EPISODES=20",
            "DEMO_PREFILL_NUM_ENVS=0",
        ),
    ),
    "eil": common.MethodSpec(
        key="eil",
        label="EIL ablation",
        family="eil",
        variant="canonical",
        fidelity="paper",
        launcher="scripts/run_pointmaze_arena_state_eil_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "ALGO_VARIANT=eil",
            "REWARD_TYPE=dense",
            "DENSE_REWARD_SCALE=1.0",
            "NUM_UPDATES=1",
            "CTA_RATIO=2",
            "DEMO_BUFFER_ENABLE=0",
            "DEMO_PREFILL_EPISODES=0",
            "DEMO_SAMPLE_RATIO=0.0",
            "EIL_THRESHOLD=0.0",
            "EIL_GOOD_MARGIN=0.0",
            "EIL_BAD_MARGIN=0.01",
            "EIL_PAIR_MARGIN=0.01",
            "EIL_BAD_PRE_STEPS=8",
        ),
    ),
    "pvp": common.MethodSpec(
        key="pvp",
        label="PVP ablation",
        family="pvp",
        variant="shared_fastsac",
        fidelity="paper",
        launcher="scripts/run_pointmaze_arena_state_pvp_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "ALGO_VARIANT=pvp",
            "REWARD_TYPE=dense",
            "DENSE_REWARD_SCALE=1.0",
            "NUM_UPDATES=1",
            "CTA_RATIO=2",
            "DEMO_PREFILL_EPISODES=20",
            "DEMO_PREFILL_NUM_ENVS=0",
            "DEMO_SAMPLE_RATIO=0.5",
            "PVP_INCLUDE_ENV_REWARD_IN_TD=0",
        ),
    ),
    "hilserl": common.MethodSpec(
        key="hilserl",
        label="HIL-SERL ablation",
        family="hilserl",
        variant="matched_scaffold",
        fidelity="paper",
        launcher="scripts/run_pointmaze_arena_state_hilserl_local.sh",
        log_root="fast_sac",
        default_overrides=(
            "ALGO_VARIANT=hilserl",
            "REWARD_TYPE=dense",
            "DENSE_REWARD_SCALE=1.0",
            "NUM_UPDATES=7",
            "CTA_RATIO=2",
            "DEMO_PREFILL_EPISODES=20",
            "DEMO_PREFILL_NUM_ENVS=0",
            "DEMO_SAMPLE_RATIO=0.5",
            "STORE_INTERVENED_IN_DEMO_BUFFER=1",
            "PREF_SAMPLE_RATIO=0.0",
            "PREF_RANK_WEIGHT=0.0",
        ),
    ),
}

PROFILE_CHOICES = ("paper", "matched_dense_demoall", "matched_sparse_demoall")

SUITES: dict[str, tuple[str, ...]] = {
    "paper": ("own", "hgdagger", "eil", "pvp", "hilserl"),
    "all": ("own", "hgdagger", "eil", "pvp", "hilserl"),
    "lambda": ("own", "own_perlinked_lambda"),
}

METHOD_COLORS = {
    "own": "#1b9e77",
    "own_perlinked_lambda": "#66c2a5",
    "hgdagger": "#d95f02",
    "eil": "#7570b3",
    "pvp": "#e7298a",
    "hilserl": "#e6ab02",
}

METHOD_SPECS: dict[str, common.MethodSpec] = BASE_METHOD_SPECS
METHOD_ORDER = {method: idx for idx, method in enumerate(METHOD_SPECS.keys())}


def _clone_spec(spec: common.MethodSpec, *extra_overrides: str) -> common.MethodSpec:
    return common.MethodSpec(
        key=spec.key,
        label=spec.label,
        family=spec.family,
        variant=spec.variant,
        fidelity=spec.fidelity,
        launcher=spec.launcher,
        log_root=spec.log_root,
        default_overrides=tuple(spec.default_overrides) + tuple(extra_overrides),
    )


def _matched_method_specs(*, reward_type: str) -> dict[str, common.MethodSpec]:
    reward_type = str(reward_type).strip().lower()
    if reward_type not in {"dense", "sparse"}:
        raise ValueError(f"Unsupported reward_type={reward_type!r}")

    reward_overrides = (
        f"REWARD_TYPE={reward_type}",
        "DENSE_REWARD_SCALE=1.0",
        "INTERVENTION_MODE=agent",
    )
    shared_fastsac_overrides = (
        "NUM_ENVS=1",
        "NUM_UPDATES=4",
        "CTA_RATIO=2",
        "NUM_CRITICS=5",
        "USE_LAYER_NORM=1",
        "DEMO_BUFFER_ENABLE=1",
        "DEMO_PREFILL_EPISODES=20",
        "DEMO_PREFILL_NUM_ENVS=0",
        "DEMO_SAMPLE_RATIO=0.5",
    )
    specs = {
        "own": _clone_spec(
            BASE_METHOD_SPECS["own"],
            *reward_overrides,
            *shared_fastsac_overrides,
            "ALGO_VARIANT=own",
            "PREF_SAMPLE_RATIO=0.5",
            "PREF_RANK_WEIGHT=1.0",
            "PREF_LOSS_TYPE=lagrangian",
            "PREF_RANK_MARGIN=0.1",
            "PREF_CRITIC_SCOPE=all",
            "STORE_INTERVENED_IN_DEMO_BUFFER=0",
        ),
        "own_perlinked_lambda": _clone_spec(
            BASE_METHOD_SPECS["own_perlinked_lambda"],
            *reward_overrides,
            *shared_fastsac_overrides,
            "ALGO_VARIANT=own",
            "PREF_SAMPLE_RATIO=0.5",
            "PREF_RANK_WEIGHT=1.0",
            "PREF_LOSS_TYPE=lagrangian",
            "PREF_LAGRANGIAN_SCOPE=per_linked",
            "PREF_RANK_MARGIN=0.1",
            "PREF_CRITIC_SCOPE=all",
            "STORE_INTERVENED_IN_DEMO_BUFFER=0",
        ),
        "hgdagger": _clone_spec(
            BASE_METHOD_SPECS["hgdagger"],
            *reward_overrides,
            "NUM_ENVS=1",
            "USE_LAYER_NORM=1",
            "NUM_UPDATES=1",
            "HG_ENSEMBLE_SIZE=5",
            "HG_DOUBT_PERCENTILE=75.0",
            "DEMO_PREFILL_EPISODES=20",
            "DEMO_PREFILL_NUM_ENVS=0",
        ),
        "eil": _clone_spec(
            BASE_METHOD_SPECS["eil"],
            *reward_overrides,
            *shared_fastsac_overrides,
            "ALGO_VARIANT=eil",
            "EIL_THRESHOLD=0.0",
            "EIL_GOOD_MARGIN=0.0",
            "EIL_BAD_MARGIN=0.01",
            "EIL_PAIR_MARGIN=0.01",
            "EIL_BAD_PRE_STEPS=8",
        ),
        "pvp": _clone_spec(
            BASE_METHOD_SPECS["pvp"],
            *reward_overrides,
            *shared_fastsac_overrides,
            "ALGO_VARIANT=pvp",
            "PVP_INCLUDE_ENV_REWARD_IN_TD=0",
        ),
        "hilserl": _clone_spec(
            BASE_METHOD_SPECS["hilserl"],
            *reward_overrides,
            *shared_fastsac_overrides,
            "ALGO_VARIANT=hilserl",
            "STORE_INTERVENED_IN_DEMO_BUFFER=1",
            "PREF_SAMPLE_RATIO=0.0",
            "PREF_RANK_WEIGHT=0.0",
        ),
    }
    return specs


def _method_specs_for_profile(profile: str) -> dict[str, common.MethodSpec]:
    profile = str(profile).strip().lower()
    if profile == "paper":
        return BASE_METHOD_SPECS
    if profile == "matched_dense_demoall":
        return _matched_method_specs(reward_type="dense")
    if profile == "matched_sparse_demoall":
        return _matched_method_specs(reward_type="sparse")
    raise ValueError(f"Unknown maze harness profile {profile!r}")


def _configure_common(method_specs: Optional[dict[str, common.MethodSpec]] = None) -> None:
    global METHOD_SPECS, METHOD_ORDER
    METHOD_SPECS = method_specs or BASE_METHOD_SPECS
    METHOD_ORDER = {method: idx for idx, method in enumerate(METHOD_SPECS.keys())}
    common.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    common.DEFAULT_PROJECT = DEFAULT_PROJECT
    common.METHOD_SPECS = METHOD_SPECS
    common.SUITES = SUITES
    common.METHOD_COLORS = METHOD_COLORS
    common.METHOD_ORDER = METHOD_ORDER
    common._scan_existing_jobs = _scan_existing_jobs
    common._infer_existing_method = _infer_existing_method


def _looks_like_maze_run(run_name: str, args_data: dict[str, Any]) -> bool:
    env_name = str(args_data.get("env_name", "") or "").strip().lower()
    if env_name.startswith("pointmaze-"):
        return True
    return "pointmaze" in run_name.lower()


def _infer_existing_method(root_name: str, run_name: str, args_data: dict[str, Any]) -> Optional[str]:
    algo_variant = str(args_data.get("algo_variant", "") or "").strip().lower()
    name = run_name.lower()
    if root_name == "hg_dagger" or algo_variant == "hg_dagger" or "hgdagger" in name:
        return "hgdagger"
    if "perlinked" in name or "per_linked_lambda" in name:
        return "own_perlinked_lambda"
    if "hilserl" in name:
        return "hilserl"
    if algo_variant == "eil" or "eil" in name:
        return "eil"
    if algo_variant == "pvp" or "pvp" in name:
        return "pvp"
    if algo_variant == "own" or "maze_state_own" in name or name.startswith("thesis_maze_eval_own_"):
        return "own"
    return None


def _scan_existing_jobs(methods: set[str]) -> list[common.JobSpec]:
    jobs: list[common.JobSpec] = []
    for root_name in ("fast_sac", "hg_dagger"):
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
            if not _looks_like_maze_run(run_dir.name, args_data):
                continue
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
                common.JobSpec(
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


def _plot_eval_metric(
    *,
    eval_rows: list[dict[str, Any]],
    plots_dir: Path,
    metric: str,
    ylabel: str,
    title_prefix: str,
    uncertainty: str,
    direction_text: str,
    ylim: Optional[tuple[float, float]] = None,
    clamp_band: bool = False,
) -> None:
    aggregated = common._aggregate_curve_rows(eval_rows, metric=metric, x_metric="step", group_metric="step")
    if clamp_band:
        aggregated = common._prepend_anchor_point(aggregated, anchor_y=0.0)
    footer_note = (
        f"Mean across seeds; {common._uncertainty_text(uncertainty)}; {common._n_text_from_curve(aggregated)}\n"
        f"{direction_text}"
    )
    common._plot_learning_curve(
        aggregated=aggregated,
        ylabel=f"{ylabel} ({direction_text.lower()})",
        xlabel="Environment Steps (thousands)",
        title=f"Maze {title_prefix} vs Env Steps",
        uncertainty=uncertainty,
        out_path=plots_dir / f"{metric}_curve.png",
        x_scale=1000.0,
        ylim=ylim,
        clamp_band=clamp_band,
        footer_note=footer_note,
    )
    aggregated_wall = common._aggregate_curve_rows(eval_rows, metric=metric, x_metric="wall_time_sec", group_metric="step")
    if clamp_band:
        aggregated_wall = common._prepend_anchor_point(aggregated_wall, anchor_y=0.0)
    common._plot_learning_curve(
        aggregated=aggregated_wall,
        ylabel=f"{ylabel} ({direction_text.lower()})",
        xlabel="Wall Time (minutes)",
        title=f"Maze {title_prefix} vs Wall Time",
        uncertainty=uncertainty,
        out_path=plots_dir / f"{metric}_wall_time_curve.png",
        x_scale=60.0,
        ylim=ylim,
        clamp_band=clamp_band,
        footer_note=footer_note,
    )


def _augment_summary_rows(
    *,
    eval_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in eval_rows:
        key = (
            str(row.get("exp_name", "")),
            str(row.get("method", "")),
            str(row.get("replicate_id", "")),
            str(row.get("seed", "")),
        )
        grouped.setdefault(key, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: float(item.get("step", 0.0)))
    augmented: list[dict[str, Any]] = []
    for row in summary_rows:
        key = (
            str(row.get("exp_name", "")),
            str(row.get("method", "")),
            str(row.get("replicate_id", "")),
            str(row.get("seed", "")),
        )
        run_rows = grouped.get(key, [])
        updated = dict(row)
        if run_rows:
            final = run_rows[-1]
            best = max(
                run_rows,
                key=lambda item: (
                    float(item.get("success_rate", 0.0)),
                    -float(item.get("avg_final_distance", 1e9)),
                    -float(item.get("lethal_rate", 1e9)),
                ),
            )
            for src_key, dst_key in (
                ("lethal_rate", "final_lethal_rate"),
                ("timeout_rate", "final_timeout_rate"),
                ("avg_final_distance", "final_avg_final_distance"),
                ("avg_success_length", "final_avg_success_length"),
            ):
                if src_key in final:
                    updated[dst_key] = final[src_key]
            if "avg_final_distance" in best:
                updated["best_avg_final_distance"] = best["avg_final_distance"]
        augmented.append(updated)
    return augmented


def _cmd_collect(args: argparse.Namespace) -> int:
    output_dir = common._ensure_dir(Path(args.output_dir))
    if args.manifest:
        jobs = common._load_manifest(Path(args.manifest))
    else:
        method_keys = set(common._suite_methods(args.suite, args.methods))
        jobs = _scan_existing_jobs(method_keys)
    eval_rows, progress_rows, summary_rows = common._collect_records(jobs)
    summary_rows = _augment_summary_rows(eval_rows=eval_rows, summary_rows=summary_rows)
    common._write_csv(output_dir / "eval_records.csv", eval_rows)
    common._write_csv(output_dir / "progress_records.csv", progress_rows)
    common._write_csv(output_dir / "run_summaries.csv", summary_rows)
    metadata = {
        "manifest": args.manifest or "",
        "suite": args.suite,
        "methods": [job.method for job in jobs],
        "num_jobs": len(jobs),
        "num_eval_rows": len(eval_rows),
        "num_progress_rows": len(progress_rows),
        "created_at": common._timestamp(),
    }
    (output_dir / "collection_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"Collected {len(eval_rows)} eval rows and {len(progress_rows)} progress rows "
        f"across {len(jobs)} jobs into {output_dir}"
    )
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir)
    eval_rows = common._read_csv(input_dir / "eval_records.csv")
    progress_rows = common._read_csv(input_dir / "progress_records.csv")
    summary_rows = common._read_csv(input_dir / "run_summaries.csv")
    eval_rows = common._filter_rows_by_methods(eval_rows, args.methods)
    progress_rows = common._filter_rows_by_methods(progress_rows, args.methods)
    summary_rows = common._filter_rows_by_methods(summary_rows, args.methods)
    plots_dir = common._ensure_dir(Path(args.output_dir))

    for metric, ylabel, title_prefix, direction_text, ylim, clamp_band in (
        ("success_rate", "Eval Success Rate", "Success Rate", "Higher is better", (0.0, 1.05), True),
        ("lethal_rate", "Eval Lethal Rate", "Lethal Rate", "Lower is better", (0.0, 1.05), True),
        ("timeout_rate", "Eval Timeout Rate", "Timeout Rate", "Lower is better", (0.0, 1.05), True),
        ("avg_length", "Eval Avg Episode Length", "Avg Episode Length", "Lower is better", None, False),
        ("avg_return", "Eval Avg Return", "Avg Return", "Higher is better", None, False),
        ("avg_final_distance", "Eval Avg Final Distance", "Avg Final Distance", "Lower is better", None, False),
        ("avg_success_length", "Eval Avg Success Length", "Avg Success Length", "Lower is better", None, False),
    ):
        _plot_eval_metric(
            eval_rows=eval_rows,
            plots_dir=plots_dir,
            metric=metric,
            ylabel=ylabel,
            title_prefix=title_prefix,
            uncertainty=args.uncertainty,
            direction_text=direction_text,
            ylim=ylim,
            clamp_band=clamp_band,
        )

    teacher_aggregated_raw = common._aggregate_curve_rows(progress_rows, metric="teacher_fraction", x_metric="step", group_metric="step")
    teacher_aggregated_filled = common._aggregate_curve_rows(progress_rows, metric="teacher_fraction_filled", x_metric="step", group_metric="step")
    teacher_window = common._effective_teacher_fraction_window(
        teacher_aggregated_filled,
        requested=args.teacher_fraction_smoothing_window,
    )
    teacher_aggregated = common._smooth_aggregated_rows(teacher_aggregated_filled, window=teacher_window)
    teacher_aggregated = common._prepend_anchor_point(teacher_aggregated, anchor_y=1.0)
    teacher_aggregated_raw = common._prepend_anchor_point(teacher_aggregated_raw, anchor_y=1.0)
    teacher_footer_note = (
        f"Mean across seeds; {common._uncertainty_text(args.uncertainty)}; {common._n_text_from_curve(teacher_aggregated)}\n"
        f"Lower is better; anchor fixed at step 0 = 1.0"
    )
    common._plot_learning_curve(
        aggregated=teacher_aggregated,
        ylabel="Teacher Fraction (lower is better)",
        xlabel="Environment Steps (thousands)",
        title="Maze Teacher Fraction vs Env Steps",
        uncertainty=args.uncertainty,
        out_path=plots_dir / "teacher_fraction_curve.png",
        x_scale=1000.0,
        ylim=(0.0, 1.05),
        clamp_band=True,
        show_markers=False,
        band_alpha=0.04,
        footer_note=teacher_footer_note,
    )
    common._plot_learning_curve(
        aggregated=teacher_aggregated_raw,
        ylabel="Teacher Fraction (lower is better)",
        xlabel="Environment Steps (thousands)",
        title="Maze Teacher Fraction vs Env Steps (raw)",
        uncertainty=args.uncertainty,
        out_path=plots_dir / "teacher_fraction_curve_raw.png",
        x_scale=1000.0,
        ylim=(0.0, 1.05),
        clamp_band=True,
        show_markers=False,
        band_alpha=0.03,
        footer_note=teacher_footer_note,
    )
    teacher_wall_raw = common._aggregate_curve_rows(progress_rows, metric="teacher_fraction", x_metric="wall_time_sec", group_metric="step")
    teacher_wall_filled = common._aggregate_curve_rows(progress_rows, metric="teacher_fraction_filled", x_metric="wall_time_sec", group_metric="step")
    teacher_wall_window = common._effective_teacher_fraction_window(
        teacher_wall_filled,
        requested=args.teacher_fraction_smoothing_window,
    )
    teacher_wall = common._smooth_aggregated_rows(teacher_wall_filled, window=teacher_wall_window)
    teacher_wall = common._prepend_anchor_point(teacher_wall, anchor_y=1.0)
    teacher_wall_raw = common._prepend_anchor_point(teacher_wall_raw, anchor_y=1.0)
    common._plot_learning_curve(
        aggregated=teacher_wall,
        ylabel="Teacher Fraction (lower is better)",
        xlabel="Wall Time (minutes)",
        title="Maze Teacher Fraction vs Wall Time",
        uncertainty=args.uncertainty,
        out_path=plots_dir / "teacher_fraction_wall_time_curve.png",
        x_scale=60.0,
        ylim=(0.0, 1.05),
        clamp_band=True,
        show_markers=False,
        band_alpha=0.04,
        footer_note=teacher_footer_note,
    )
    common._plot_learning_curve(
        aggregated=teacher_wall_raw,
        ylabel="Teacher Fraction (lower is better)",
        xlabel="Wall Time (minutes)",
        title="Maze Teacher Fraction vs Wall Time (raw)",
        uncertainty=args.uncertainty,
        out_path=plots_dir / "teacher_fraction_wall_time_curve_raw.png",
        x_scale=60.0,
        ylim=(0.0, 1.05),
        clamp_band=True,
        show_markers=False,
        band_alpha=0.03,
        footer_note=teacher_footer_note,
    )

    for metric, title, ylabel, direction_text, ylim in (
        ("final_success", "Final Success Rate by Method", "Success Rate", "Higher is better", (0.0, 1.05)),
        ("best_success", "Best Success Rate by Method", "Success Rate", "Higher is better", (0.0, 1.05)),
        ("final_lethal_rate", "Final Lethal Rate by Method", "Lethal Rate", "Lower is better", (0.0, 1.05)),
        ("final_timeout_rate", "Final Timeout Rate by Method", "Timeout Rate", "Lower is better", (0.0, 1.05)),
    ):
        aggregated_summary = common._aggregate_summary_rows(summary_rows, metric)
        common._write_csv(plots_dir / f"{metric}_summary.csv", aggregated_summary)
        footer_note = (
            f"Mean across seeds; {common._uncertainty_text(args.uncertainty)}; {common._n_text_from_summary(aggregated_summary)}\n"
            f"{direction_text}"
        )
        common._plot_summary_bar(
            aggregated_rows=aggregated_summary,
            title=title,
            ylabel=f"{ylabel} ({direction_text.lower()})",
            uncertainty=args.uncertainty,
            out_path=plots_dir / f"{metric}_bar.png",
            ylim=ylim,
            footer_note=footer_note,
        )

    for metric, title, ylabel, direction_text, scale, filename, ylim in (
        ("final_avg_final_distance", "Final Avg Final Distance by Method", "Distance", "Lower is better", 1.0, "final_avg_final_distance_bar.png", None),
        ("best_avg_final_distance", "Best Avg Final Distance by Method", "Distance", "Lower is better", 1.0, "best_avg_final_distance_bar.png", None),
        ("final_wall_time_sec", "Final Wall Time by Method", "Wall Time (minutes)", "Lower is better", 60.0, "final_wall_time_bar.png", None),
        ("mean_teacher_fraction", "Mean Teacher Fraction by Method", "Teacher Fraction", "Lower is better", 1.0, "mean_teacher_fraction_bar.png", (0.0, 1.05)),
        ("total_intervention_steps_est", "Estimated Total Intervention Steps by Method", "Estimated teacher-controlled env steps (thousands)", "Lower is better", 1000.0, "total_intervention_steps_bar.png", None),
        ("total_intervention_fraction_est", "Estimated Total Intervention Fraction by Method", "Estimated intervention fraction", "Lower is better", 1.0, "total_intervention_fraction_bar.png", (0.0, 1.05)),
    ):
        aggregated_summary = common._aggregate_summary_rows(summary_rows, metric)
        common._write_csv(plots_dir / f"{metric}_summary.csv", aggregated_summary)
        footer_note = (
            f"Mean across seeds; {common._uncertainty_text(args.uncertainty)}; {common._n_text_from_summary(aggregated_summary)}\n"
            f"{direction_text}"
        )
        common._plot_summary_bar(
            aggregated_rows=aggregated_summary,
            title=title,
            ylabel=f"{ylabel} ({direction_text.lower()})",
            uncertainty=args.uncertainty,
            out_path=plots_dir / filename,
            value_scale=scale,
            ylim=ylim,
            footer_note=footer_note,
        )
    print(f"Plots written to {plots_dir}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maze thesis evaluation harness")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    launch = subparsers.add_parser("launch", help="Generate or run a seeded maze method matrix via existing launcher scripts.")
    launch.add_argument("--suite", type=str, default="all", choices=sorted(SUITES))
    launch.add_argument("--methods", nargs="*", default=None, help="Optional explicit method keys overriding the suite.")
    launch.add_argument("--profile", type=str, default="paper", choices=PROFILE_CHOICES)
    launch.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    launch.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    launch.add_argument("--wandb-entity", type=str, default="")
    launch.add_argument("--wandb-group", type=str, default="")
    launch.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    launch.add_argument("--run-prefix", type=str, default="thesis_maze_eval")
    launch.add_argument("--total-timesteps", type=int, default=10000)
    launch.add_argument("--num-eval-episodes", type=int, default=10)
    launch.add_argument("--eval-num-envs", type=int, default=5)
    launch.add_argument("--eval-interval", type=int, default=2500)
    launch.add_argument("--save-interval", type=int, default=5000)
    launch.add_argument("--log-interval", type=int, default=200)
    launch.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    launch.add_argument("--mode", type=str, default="print", choices=["print", "run"])
    launch.add_argument("--continue-on-error", action="store_true", default=False)

    collect = subparsers.add_parser("collect", help="Collect maze eval traces from a manifest or from existing local logs.")
    collect.add_argument("--manifest", type=str, default="")
    collect.add_argument("--suite", type=str, default="all", choices=sorted(SUITES))
    collect.add_argument("--methods", nargs="*", default=None)
    collect.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT / "latest_collection"))

    plot = subparsers.add_parser("plot", help="Plot maze learning curves and summary bars from a collected dataset.")
    plot.add_argument("--input-dir", type=str, required=True)
    plot.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_ROOT / "latest_plots"))
    plot.add_argument("--uncertainty", type=str, default="ci95", choices=["std", "sem", "ci95"])
    plot.add_argument("--teacher-fraction-smoothing-window", type=int, default=21)
    plot.add_argument("--methods", nargs="*", default=None)

    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.cmd == "launch":
        _configure_common(_method_specs_for_profile(getattr(args, "profile", "paper")))
        return common._cmd_launch(args)
    _configure_common(BASE_METHOD_SPECS)
    if args.cmd == "collect":
        return _cmd_collect(args)
    if args.cmd == "plot":
        return _cmd_plot(args)
    raise SystemExit(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

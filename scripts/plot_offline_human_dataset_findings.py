#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "local" / "offline_human_dataset_findings"


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    family: str
    category: str
    metrics_path: Path
    color: str
    linestyle: str = "-"


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        key="bc_scratch",
        label="BC",
        family="scratch",
        category="from_scratch_methods",
        metrics_path=REPO_ROOT / "models/offline_method_compare/cube_single_task1_offline_bc_budget500_20260412_010048/metrics.jsonl",
        color="#1f77b4",
    ),
    RunSpec(
        key="hgdagger_scratch",
        label="HG-DAgger",
        family="scratch",
        category="from_scratch_methods",
        metrics_path=REPO_ROOT / "models/offline_method_compare/cube_single_task1_offline_hg_dagger_budget500_20260412_010053/metrics.jsonl",
        color="#ff7f0e",
    ),
    RunSpec(
        key="pvp_scratch",
        label="PVP ablation",
        family="scratch",
        category="from_scratch_methods",
        metrics_path=REPO_ROOT / "models/offline_method_compare/cube_single_task1_offline_pvp_budget200_20260412_012003/metrics.jsonl",
        color="#2ca02c",
    ),
    RunSpec(
        key="eil_scratch",
        label="EIL ablation",
        family="scratch",
        category="from_scratch_methods",
        metrics_path=REPO_ROOT / "models/offline_method_compare/cube_single_task1_offline_eil_budget500_20260412_010757/metrics.jsonl",
        color="#d62728",
    ),
    RunSpec(
        key="hilserl_scratch",
        label="HIL-SERL ablation",
        family="scratch",
        category="from_scratch_methods",
        metrics_path=REPO_ROOT / "models/offline_method_compare/cube_single_task1_offline_hilserl_budget200_20260412_013548/metrics.jsonl",
        color="#9467bd",
    ),
    RunSpec(
        key="own_weighted_scratch",
        label="Ours weighted",
        family="scratch",
        category="from_scratch_methods",
        metrics_path=REPO_ROOT / "models/offline_method_compare/cube_single_task1_offline_own_weighted_gpu_recheck/metrics.jsonl",
        color="#8c564b",
    ),
    RunSpec(
        key="own_nogate_scratch",
        label="Ours no gate",
        family="scratch",
        category="from_scratch_methods",
        metrics_path=REPO_ROOT / "models/offline_method_compare/cube_single_task1_offline_own_nogate_nogate_20260411_214336/metrics.jsonl",
        color="#e377c2",
    ),
    RunSpec(
        key="bc_recover",
        label="BC (effective rows)",
        family="recovery",
        category="recovery_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_bc_replayeff_humanlatest/metrics.jsonl",
        color="#1f77b4",
    ),
    RunSpec(
        key="bt_recover",
        label="BT pref",
        family="recovery",
        category="recovery_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_pref_bt_replay_humanlatest/metrics.jsonl",
        color="#17becf",
    ),
    RunSpec(
        key="hinge001w1_recover",
        label="Hinge m=0.01 w=1",
        family="recovery",
        category="recovery_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_pref_hinge_replay_humanlatest/metrics.jsonl",
        color="#ff7f0e",
    ),
    RunSpec(
        key="hinge010w1_recover",
        label="Hinge m=0.10 w=1",
        family="recovery",
        category="recovery_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_pref_hinge_m010_w1_humanlatest/metrics.jsonl",
        color="#2ca02c",
    ),
    RunSpec(
        key="hinge001w5_recover",
        label="Hinge m=0.01 w=5",
        family="recovery",
        category="recovery_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_pref_hinge_m001_w5_humanlatest/metrics.jsonl",
        color="#d62728",
    ),
    RunSpec(
        key="hinge010w5_recover",
        label="Hinge m=0.10 w=5",
        family="recovery",
        category="recovery_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_pref_hinge_m010_w5_humanlatest/metrics.jsonl",
        color="#9467bd",
    ),
    RunSpec(
        key="eps_base",
        label="Base eps=1e-6",
        family="gating",
        category="gating_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_pref_hinge_m001_w5_humanlatest_epsbase/metrics.jsonl",
        color="#7f7f7f",
    ),
    RunSpec(
        key="eps001",
        label="eps=0.01",
        family="gating",
        category="gating_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_pref_hinge_m001_w5_humanlatest_eps001/metrics.jsonl",
        color="#bcbd22",
    ),
    RunSpec(
        key="eps001_scale025",
        label="eps=0.01 + scale=0.25",
        family="gating",
        category="gating_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_pref_hinge_m001_w5_humanlatest_eps001_scale025/metrics.jsonl",
        color="#1f77b4",
    ),
    RunSpec(
        key="angle_gate",
        label="Angle gate",
        family="gating",
        category="gating_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_gate_angle_humanckpt_humanreplay_20260411/metrics.jsonl",
        color="#ff7f0e",
    ),
    RunSpec(
        key="component_gate",
        label="Component corridor",
        family="gating",
        category="gating_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_gate_component_humanckpt_humanreplay_20260411/metrics.jsonl",
        color="#2ca02c",
    ),
    RunSpec(
        key="xyz_ball_hard",
        label="XYZ L2 hard ball",
        family="gating",
        category="gating_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_ball_xyzl2_eps035_noscale_humanckpt_humanreplay_20260411/metrics.jsonl",
        color="#d62728",
    ),
    RunSpec(
        key="xyz_ball_soft",
        label="XYZ L2 soft ball",
        family="gating",
        category="gating_ablation",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_ball_xyzl2_eps0_scale1_humanckpt_humanreplay_20260411/metrics.jsonl",
        color="#9467bd",
    ),
    RunSpec(
        key="weighted_gpu_recover",
        label="Ours weighted + ckpt recover",
        family="protocol_gap",
        category="protocol_gap",
        metrics_path=REPO_ROOT / "models/offline_actor_recover/offline_pref_hinge_m001_w5_humanlatest_eps001_scale025_gpu_recheck/metrics.jsonl",
        color="#1f77b4",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and plot offline human-dataset findings.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output directory for CSV summaries and figures (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_actor_eval(row: dict[str, Any]) -> bool:
    phase = row.get("phase")
    return phase is None or phase in {"actor", "initial"}


def collect_run(spec: RunSpec) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_jsonl(spec.metrics_path)
    dataset_stats: dict[str, Any] = {}
    curve_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for row in rows:
        for key, value in row.items():
            if key.startswith("Offline/") and key not in dataset_stats:
                dataset_stats[key] = value

        step = row.get("step", 0)
        phase = row.get("phase", "offline")

        if "Eval/success_rate" in row and is_actor_eval(row):
            eval_record = {
                "run_key": spec.key,
                "label": spec.label,
                "family": spec.family,
                "category": spec.category,
                "phase": phase,
                "step": step,
                "success_rate": row.get("Eval/success_rate"),
                "avg_length": row.get("Eval/avg_length"),
                "avg_return": row.get("Eval/avg_return"),
                "timeout_rate": row.get("Eval/timeout_rate"),
            }
            curve_records.append(eval_record)
            eval_rows.append(eval_record)

        metric_records.append(
            {
                "run_key": spec.key,
                "label": spec.label,
                "family": spec.family,
                "category": spec.category,
                "phase": phase,
                "step": step,
                "pref_q_delta_mean": row.get("Pref/q_delta_mean"),
                "pref_q_teacher_mean": row.get("Pref/q_teacher_mean"),
                "pref_q_student_mean": row.get("Pref/q_student_mean"),
                "pref_loss": row.get("Train/pref_loss"),
                "pair_weight_mean": row.get("Train/pair_weight_mean"),
                "actor_loss": row.get("Train/loss"),
                "actor_q_loss": row.get("Train/actor_q_loss"),
                "q_actor_mean": row.get("Diag/q_actor_mean"),
            }
        )

    if not eval_rows:
        raise RuntimeError(f"No actor/eval rows found in {spec.metrics_path}")

    best_eval = max(eval_rows, key=lambda row: (float(row["success_rate"]), -float(row["avg_length"] or 0.0)))
    last_eval = max(eval_rows, key=lambda row: float(row["step"]))

    summary = {
        "run_key": spec.key,
        "label": spec.label,
        "family": spec.family,
        "category": spec.category,
        "metrics_path": str(spec.metrics_path),
        "best_success_rate": best_eval["success_rate"],
        "best_success_step": best_eval["step"],
        "avg_length_at_best": best_eval["avg_length"],
        "last_success_rate": last_eval["success_rate"],
        "last_success_step": last_eval["step"],
        "last_avg_length": last_eval["avg_length"],
        "offline_rows": dataset_stats.get("Offline/dataset_rows"),
        "teacher_like_fraction": dataset_stats.get("Offline/teacher_like_fraction"),
        "teacher_flag_fraction": dataset_stats.get("Offline/teacher_flag_fraction"),
        "action_diff_fraction": dataset_stats.get("Offline/action_diff_fraction"),
        "intervened_gripper_only_fraction": dataset_stats.get("Offline/intervened_gripper_only_fraction"),
        "pair_weight_mean_selected": dataset_stats.get("Offline/pair_weight_mean_selected"),
        "action_delta_mean_selected": dataset_stats.get("Offline/action_delta_mean_selected"),
    }
    return summary, curve_records, metric_records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sort_rows(rows: list[dict[str, Any]], key_order: list[str]) -> list[dict[str, Any]]:
    index = {key: i for i, key in enumerate(key_order)}
    return sorted(rows, key=lambda row: index.get(row["run_key"], 10_000))


def configure_matplotlib() -> None:
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


def label_bars_horizontal(ax: plt.Axes, bars, fmt: str) -> None:
    for bar in bars:
        width = float(bar.get_width())
        y = bar.get_y() + bar.get_height() / 2.0
        ax.text(width + 0.01 * max(ax.get_xlim()[1], 1.0), y, format(width, fmt), va="center", ha="left", fontsize=8)


def horizontal_bar_panel(ax: plt.Axes, rows: list[dict[str, Any]], value_key: str, title: str, xlabel: str, xlim: tuple[float, float] | None = None) -> None:
    labels = [row["label"] for row in rows]
    values = [float(row.get(value_key) or 0.0) for row in rows]
    colors = [SPEC_BY_KEY[row["run_key"]].color for row in rows]
    bars = ax.barh(labels, values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(*xlim)
    label_bars_horizontal(ax, bars, ".2f")


def plot_dataset_composition(summary_rows: list[dict[str, Any]], output_root: Path) -> None:
    row = next(row for row in summary_rows if row["run_key"] == "bc_scratch")
    teacher_fraction = float(row["teacher_like_fraction"] or 0.0)
    gripper_only_fraction = float(row["intervened_gripper_only_fraction"] or 0.0)
    xyz_fraction = 1.0 - gripper_only_fraction

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    axes[0].barh(["All replay rows"], [teacher_fraction], color="#1f77b4", label="Teacher-like / intervened")
    axes[0].barh(["All replay rows"], [1.0 - teacher_fraction], left=[teacher_fraction], color="#d9d9d9", label="Not intervened")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_title("Replay composition")
    axes[0].set_xlabel("Fraction of all rows")
    axes[0].legend(loc="lower right")
    axes[0].text(teacher_fraction / 2.0, 0, f"{teacher_fraction:.1%}", va="center", ha="center", color="white", fontsize=10, fontweight="bold")
    axes[0].text(teacher_fraction + (1.0 - teacher_fraction) / 2.0, 0, f"{1.0 - teacher_fraction:.1%}", va="center", ha="center", color="#333333", fontsize=10, fontweight="bold")

    axes[1].barh(["Teacher-like rows"], [gripper_only_fraction], color="#9467bd", label="Gripper-only")
    axes[1].barh(["Teacher-like rows"], [xyz_fraction], left=[gripper_only_fraction], color="#2ca02c", label="XYZ involved")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_title("Within teacher-like rows")
    axes[1].set_xlabel("Fraction of teacher-like rows")
    axes[1].legend(loc="lower right")
    axes[1].text(gripper_only_fraction / 2.0, 0, f"{gripper_only_fraction:.1%}", va="center", ha="center", color="white", fontsize=10, fontweight="bold")
    axes[1].text(gripper_only_fraction + xyz_fraction / 2.0, 0, f"{xyz_fraction:.1%}", va="center", ha="center", color="white", fontsize=10, fontweight="bold")

    fig.suptitle("Human replay dataset composition", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_root / "dataset_composition.png", bbox_inches="tight")
    plt.close(fig)


def plot_overview(summary_rows: list[dict[str, Any]], output_root: Path) -> None:
    groups = {
        "from_scratch_methods": ["bc_scratch", "hgdagger_scratch", "pvp_scratch", "eil_scratch", "hilserl_scratch", "own_weighted_scratch", "own_nogate_scratch"],
        "recovery_ablation": ["bc_recover", "bt_recover", "hinge001w1_recover", "hinge010w1_recover", "hinge001w5_recover", "hinge010w5_recover"],
        "gating_ablation": ["eps_base", "eps001", "eps001_scale025", "angle_gate", "component_gate", "xyz_ball_hard", "xyz_ball_soft"],
        "protocol_gap": ["own_weighted_scratch", "weighted_gpu_recover"],
    }
    titles = {
        "from_scratch_methods": "From scratch on human replay",
        "recovery_ablation": "Checkpoint recovery on human replay",
        "gating_ablation": "Filtering / weighting ablations",
        "protocol_gap": "Protocol gap",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (category, key_order) in zip(axes.flatten(), groups.items()):
        rows = sort_rows([row for row in summary_rows if row["category"] == category or row["run_key"] in key_order], key_order)
        horizontal_bar_panel(ax, rows, "best_success_rate", titles[category], "Best eval success rate", (0.0, 1.05))

    fig.suptitle("Offline human-data findings overview", y=0.995, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_root / "offline_human_findings_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_dual_bars(summary_rows: list[dict[str, Any]], output_root: Path, category: str, key_order: list[str], filename: str, title: str) -> None:
    rows = sort_rows([row for row in summary_rows if row["run_key"] in key_order], key_order)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(4.5, 0.65 * len(rows) + 1.5)))
    horizontal_bar_panel(axes[0], rows, "best_success_rate", f"{title}: best success", "Best eval success rate", (0.0, 1.05))
    horizontal_bar_panel(axes[1], rows, "avg_length_at_best", f"{title}: avg length at best", "Episode length")
    fig.tight_layout()
    fig.savefig(output_root / filename, bbox_inches="tight")
    plt.close(fig)


def plot_success_curves(curve_rows: list[dict[str, Any]], output_root: Path) -> None:
    scratch_keys = ["bc_scratch", "hgdagger_scratch", "pvp_scratch", "eil_scratch", "hilserl_scratch", "own_weighted_scratch", "own_nogate_scratch"]
    recovery_keys = ["bc_recover", "bt_recover", "hinge001w1_recover", "hinge010w1_recover", "hinge001w5_recover", "hinge010w5_recover"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, keys, title in (
        (axes[0], scratch_keys, "From-scratch offline methods"),
        (axes[1], recovery_keys, "Checkpoint-recovery offline methods"),
    ):
        for key in keys:
            rows = sorted((row for row in curve_rows if row["run_key"] == key), key=lambda row: float(row["step"]))
            xs = [float(row["step"]) for row in rows]
            ys = [float(row["success_rate"]) for row in rows]
            spec = SPEC_BY_KEY[key]
            ax.plot(xs, ys, label=spec.label, color=spec.color, linestyle=spec.linestyle, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Offline step")
        ax.set_ylabel("Eval success rate")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_root / "success_curves_methods.png", bbox_inches="tight")
    plt.close(fig)


def plot_gating_curves(curve_rows: list[dict[str, Any]], output_root: Path) -> None:
    keys = ["eps_base", "eps001", "eps001_scale025", "angle_gate", "component_gate", "xyz_ball_hard", "xyz_ball_soft"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key in keys:
        rows = sorted((row for row in curve_rows if row["run_key"] == key), key=lambda row: float(row["step"]))
        xs = [float(row["step"]) for row in rows]
        success = [float(row["success_rate"]) for row in rows]
        length = [float(row["avg_length"]) for row in rows]
        spec = SPEC_BY_KEY[key]
        axes[0].plot(xs, success, label=spec.label, color=spec.color, linewidth=2)
        axes[1].plot(xs, length, label=spec.label, color=spec.color, linewidth=2)

    axes[0].set_title("Gating ablation: success over actor recovery")
    axes[0].set_xlabel("Actor recovery step")
    axes[0].set_ylabel("Eval success rate")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].legend(loc="best")

    axes[1].set_title("Gating ablation: avg episode length")
    axes[1].set_xlabel("Actor recovery step")
    axes[1].set_ylabel("Eval avg length")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_root / "gating_ablation_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_mechanism(metric_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]], output_root: Path) -> None:
    keys = ["eps_base", "eps001", "eps001_scale025"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for key in keys:
        spec = SPEC_BY_KEY[key]
        eval_rows = sorted((row for row in curve_rows if row["run_key"] == key), key=lambda row: float(row["step"]))
        axes[0].plot(
            [float(row["step"]) for row in eval_rows],
            [float(row["success_rate"]) for row in eval_rows],
            label=spec.label,
            color=spec.color,
            linewidth=2,
        )

        critic_rows = sorted(
            (
                row
                for row in metric_rows
                if row["run_key"] == key and row["phase"] == "critic" and row["pref_q_delta_mean"] is not None
            ),
            key=lambda row: float(row["step"]),
        )
        axes[1].plot(
            [float(row["step"]) for row in critic_rows],
            [float(row["pref_q_delta_mean"]) for row in critic_rows],
            label=spec.label,
            color=spec.color,
            linewidth=2,
        )
        axes[2].plot(
            [float(row["step"]) for row in critic_rows],
            [float(row["pref_loss"]) for row in critic_rows],
            label=spec.label,
            color=spec.color,
            linewidth=2,
        )

    axes[0].set_title("Actor recovery success")
    axes[0].set_xlabel("Actor step")
    axes[0].set_ylabel("Eval success rate")
    axes[0].set_ylim(-0.02, 1.05)

    axes[1].set_title("Critic preference gap")
    axes[1].set_xlabel("Critic step")
    axes[1].set_ylabel("Pref/q_delta_mean")

    axes[2].set_title("Critic preference loss")
    axes[2].set_xlabel("Critic step")
    axes[2].set_ylabel("Train/pref_loss")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Why epsilon + weighting helps: behavior improves more than Q-gap changes", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_root / "eps_weight_mechanism.png", bbox_inches="tight")
    plt.close(fig)


def plot_protocol_gap(summary_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]], output_root: Path) -> None:
    key_order = ["own_weighted_scratch", "weighted_gpu_recover"]
    rows = sort_rows([row for row in summary_rows if row["run_key"] in key_order], key_order)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    horizontal_bar_panel(axes[0], rows, "best_success_rate", "Same weighted method, different protocol", "Best eval success rate", (0.0, 1.05))
    horizontal_bar_panel(axes[1], rows, "avg_length_at_best", "Same weighted method, different protocol", "Episode length at best")
    fig.tight_layout()
    fig.savefig(output_root / "protocol_gap_bars.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key in key_order:
        rows_curve = sorted((row for row in curve_rows if row["run_key"] == key), key=lambda row: float(row["step"]))
        spec = SPEC_BY_KEY[key]
        ax.plot(
            [float(row["step"]) for row in rows_curve],
            [float(row["success_rate"]) for row in rows_curve],
            label=spec.label,
            color=spec.color,
            linewidth=2,
        )
    ax.set_title("Protocol gap: from scratch vs checkpoint recovery")
    ax.set_xlabel("Offline step")
    ax.set_ylabel("Eval success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_root / "protocol_gap_curve.png", bbox_inches="tight")
    plt.close(fig)


def build_dataset_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row = next(row for row in summary_rows if row["run_key"] == "bc_scratch")
    teacher_fraction = float(row["teacher_like_fraction"] or 0.0)
    gripper_only_fraction = float(row["intervened_gripper_only_fraction"] or 0.0)
    return [
        {"metric": "dataset_rows", "value": row["offline_rows"]},
        {"metric": "teacher_like_fraction", "value": teacher_fraction},
        {"metric": "not_teacher_like_fraction", "value": 1.0 - teacher_fraction},
        {"metric": "gripper_only_fraction_within_teacher_like", "value": gripper_only_fraction},
        {"metric": "xyz_involved_fraction_within_teacher_like", "value": 1.0 - gripper_only_fraction},
    ]


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for spec in RUN_SPECS:
        summary, curves, metrics = collect_run(spec)
        summary_rows.append(summary)
        curve_rows.extend(curves)
        metric_rows.extend(metrics)

    write_csv(output_root / "run_summaries.csv", summary_rows)
    write_csv(output_root / "curve_records.csv", curve_rows)
    write_csv(output_root / "metric_records.csv", metric_rows)
    write_csv(output_root / "dataset_summary.csv", build_dataset_summary(summary_rows))

    plot_dataset_composition(summary_rows, output_root)
    plot_overview(summary_rows, output_root)
    plot_dual_bars(
        summary_rows,
        output_root,
        category="from_scratch_methods",
        key_order=["bc_scratch", "hgdagger_scratch", "pvp_scratch", "eil_scratch", "hilserl_scratch", "own_weighted_scratch", "own_nogate_scratch"],
        filename="from_scratch_method_bars.png",
        title="From-scratch methods on the human replay dataset",
    )
    plot_dual_bars(
        summary_rows,
        output_root,
        category="recovery_ablation",
        key_order=["bc_recover", "bt_recover", "hinge001w1_recover", "hinge010w1_recover", "hinge001w5_recover", "hinge010w5_recover"],
        filename="recovery_method_bars.png",
        title="Checkpoint recovery on the human replay dataset",
    )
    plot_dual_bars(
        summary_rows,
        output_root,
        category="gating_ablation",
        key_order=["eps_base", "eps001", "eps001_scale025", "angle_gate", "component_gate", "xyz_ball_hard", "xyz_ball_soft"],
        filename="gating_ablation_bars.png",
        title="Teacher-like filtering / weighting ablations",
    )
    plot_success_curves(curve_rows, output_root)
    plot_gating_curves(curve_rows, output_root)
    plot_mechanism(metric_rows, curve_rows, output_root)
    plot_protocol_gap(summary_rows, curve_rows, output_root)

    print(f"Wrote offline human-dataset findings to {output_root}")


SPEC_BY_KEY = {spec.key: spec for spec in RUN_SPECS}


if __name__ == "__main__":
    main()

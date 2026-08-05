#!/usr/bin/env python3
"""Plot multi-seed Pareto comparisons for matched Unitree navigation runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from plot_unitree_competitor_comparison import COLORS, LABELS, MARKERS, METHODS, pareto_mask, pareto_mask_3d


ABLATION_METHODS = ("pref_only", "bc_only")
PLOT_METHODS = (*METHODS, *ABLATION_METHODS)
LABELS = {
    **LABELS,
    "pref_only": "Preference-only",
    "bc_only": "BC-only",
}
SHORT_LABELS = {
    "thesis": "Thesis (pref. + BC)",
    "hilserl": "HIL-SERL",
    "eil": "EIL",
    "pvp": "PVP",
    "hg_dagger": "HG-DAgger",
    "sac": "Goal-only SAC",
    "pref_only": "Preference-only",
    "bc_only": "BC-only",
}
COLORS = {
    **COLORS,
    "pref_only": "#17a2b8",
    "bc_only": "#d65f9e",
}
MARKERS = {
    **MARKERS,
    "pref_only": "P",
    "bc_only": "X",
}

METRICS = (
    "success_rate",
    "safe_success_rate",
    "mean_cost_sum",
    "costful_episode_rate",
    "teacher_fraction_cumulative",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_complete_run(models_root: Path, method: str, seed: int, suffix: str) -> Path:
    prefix = (
        f"unitree_ablation_{method}_seed{seed}_{suffix}"
        if method in ABLATION_METHODS
        else f"unitree_compare_{method}_seed{seed}_{suffix}"
    )
    candidates = []
    for run_dir in sorted(models_root.glob(f"{prefix}*")):
        if not (run_dir / "final.pt").exists():
            continue
        eval_rows = read_jsonl(run_dir / "eval_metrics.jsonl")
        train_rows = read_jsonl(run_dir / "metrics.jsonl")
        if not eval_rows or not train_rows:
            continue
        candidates.append((float(eval_rows[-1].get("transitions", -1)), run_dir, eval_rows, train_rows))
    if not candidates:
        raise FileNotFoundError(f"No complete run found for method={method}, seed={seed}, suffix={suffix}")
    _, run_dir, _, _ = max(candidates, key=lambda item: (item[0], item[1].name.endswith("_recovery")))
    return run_dir


def collect_rows(models_root: Path, suffix: str, ablation_suffix: str, seeds: list[int]) -> list[dict]:
    rows = []
    for method in PLOT_METHODS:
        for seed in seeds:
            run_dir = select_complete_run(
                models_root,
                method,
                seed,
                ablation_suffix if method in ABLATION_METHODS else suffix,
            )
            eval_final = read_jsonl(run_dir / "eval_metrics.jsonl")[-1]
            train_final = read_jsonl(run_dir / "metrics.jsonl")[-1]
            rows.append(
                {
                    "method": method,
                    "label": LABELS[method],
                    "seed": seed,
                    "run_dir": str(run_dir.resolve()),
                    "transitions": float(eval_final["transitions"]),
                    "episodes": int(float(eval_final["episodes"])),
                    "success_rate": float(eval_final["success_rate"]),
                    "safe_success_rate": float(eval_final["safe_success_rate"]),
                    "mean_cost_sum": float(eval_final["mean_cost_sum"]),
                    "costful_episode_rate": float(eval_final["costful_episode_rate"]),
                    "teacher_fraction_cumulative": float(train_final.get("teacher_fraction_cumulative", 0.0)),
                }
            )
    return rows


def aggregate_rows(seed_rows: list[dict]) -> list[dict]:
    summaries = []
    for method in PLOT_METHODS:
        rows = [row for row in seed_rows if row["method"] == method]
        summary = {
            "method": method,
            "label": LABELS[method],
            "num_seeds": len(rows),
            "episodes_per_seed": rows[0]["episodes"],
            "total_rollouts": sum(row["episodes"] for row in rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[metric] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def draw_frontier(axis, summaries: list[dict], x_key: str) -> None:
    frontier = pareto_mask(summaries, x_key=x_key, y_key="safe_success_rate")
    points = sorted(
        {
            (float(row[x_key]), float(row["safe_success_rate"]))
            for row, keep in zip(summaries, frontier)
            if keep
        }
    )
    if points:
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color="#161616",
            linestyle="--",
            linewidth=1.6,
            alpha=0.75,
            label="Pareto frontier of means",
            zorder=2,
        )


def plot(seed_rows: list[dict], summaries: list[dict], output_path: Path, elevation: float, azimuth: float) -> None:
    fig = plt.figure(figsize=(16, 11.5), facecolor="#f7f4ec")
    panels = (
        ("mean_cost_sum", "Mean episode cost (lower is better)", "Safety-cost tradeoff"),
        ("costful_episode_rate", "Costful episode fraction (lower is better)", "Safety-incidence tradeoff"),
        ("teacher_fraction_cumulative", "Training intervention fraction (lower is better)", "Human/teacher burden tradeoff"),
    )
    axes = [fig.add_subplot(2, 2, index + 1) for index in range(3)]
    legend_handles = []

    for axis, (x_key, x_label, title) in zip(axes, panels):
        axis.set_facecolor("#fffdf7")
        draw_frontier(axis, summaries, x_key)
        frontier = pareto_mask(summaries, x_key=x_key, y_key="safe_success_rate")
        for summary, keep in zip(summaries, frontier):
            method = summary["method"]
            method_seed_rows = [row for row in seed_rows if row["method"] == method]
            axis.scatter(
                [row[x_key] for row in method_seed_rows],
                [row["safe_success_rate"] for row in method_seed_rows],
                s=52,
                color=COLORS[method],
                marker=MARKERS[method],
                alpha=0.27,
                edgecolor="none",
                zorder=3,
            )
            artist = axis.errorbar(
                summary[x_key],
                summary["safe_success_rate"],
                xerr=summary[f"{x_key}_sd"],
                yerr=summary["safe_success_rate_sd"],
                fmt=MARKERS[method],
                markersize=11 if keep else 9,
                color=COLORS[method],
                markeredgecolor="#111111" if keep else "white",
                markeredgewidth=1.8 if keep else 0.8,
                elinewidth=1.5,
                capsize=4,
                alpha=0.95,
                zorder=4,
                label=LABELS[method],
            )
            if axis is axes[0]:
                legend_handles.append(artist)
            annotation_offset = (7, 8 if method not in {"hilserl", "sac", "bc_only"} else -14)
            horizontal_alignment = "left"
            if x_key == "mean_cost_sum" and method == "thesis":
                annotation_offset = (7, -20)
            elif x_key == "mean_cost_sum" and method == "sac":
                annotation_offset = (7, 10)
            elif x_key == "mean_cost_sum" and method == "pref_only":
                annotation_offset = (7, 10)
            elif x_key == "teacher_fraction_cumulative" and method == "pref_only":
                annotation_offset = (7, -14)
            axis.annotate(
                SHORT_LABELS[method],
                (summary[x_key], summary["safe_success_rate"]),
                xytext=annotation_offset,
                textcoords="offset points",
                fontsize=8,
                fontweight="bold" if method == "thesis" else "normal",
                ha=horizontal_alignment,
            )
        axis.set(title=title, xlabel=x_label, ylabel="Teacher-free safe success", ylim=(-0.04, 1.04))
        axis.grid(alpha=0.25, color="#82796a")

    axis_3d = fig.add_subplot(2, 2, 4, projection="3d")
    axis_3d.set_facecolor("#fffdf7")
    frontier_3d = pareto_mask_3d(summaries)
    for summary, keep in zip(summaries, frontier_3d):
        method = summary["method"]
        x = summary["teacher_fraction_cumulative"]
        y = summary["mean_cost_sum"]
        z = summary["safe_success_rate"]
        method_seed_rows = [row for row in seed_rows if row["method"] == method]
        axis_3d.scatter(
            [row["teacher_fraction_cumulative"] for row in method_seed_rows],
            [row["mean_cost_sum"] for row in method_seed_rows],
            [row["safe_success_rate"] for row in method_seed_rows],
            s=30,
            color=COLORS[method],
            alpha=0.22,
        )
        axis_3d.scatter(
            x,
            y,
            z,
            s=145 if keep else 95,
            color=COLORS[method],
            marker=MARKERS[method],
            edgecolor="#111111" if keep else "white",
            linewidth=1.8 if keep else 0.8,
            alpha=0.95,
        )
        axis_3d.plot([x - summary["teacher_fraction_cumulative_sd"], x + summary["teacher_fraction_cumulative_sd"]], [y, y], [z, z], color=COLORS[method], alpha=0.7)
        axis_3d.plot([x, x], [y - summary["mean_cost_sum_sd"], y + summary["mean_cost_sum_sd"]], [z, z], color=COLORS[method], alpha=0.7)
        axis_3d.plot([x, x], [y, y], [z - summary["safe_success_rate_sd"], z + summary["safe_success_rate_sd"]], color=COLORS[method], alpha=0.7)
        label = SHORT_LABELS[method]
        if method in {"thesis", "pref_only", "bc_only", "sac"}:
            label += f"\nsafe={z:.2f}, cost={y:.1f}, int={x:.2f}"
            axis_3d.text(x, y, z, f"  {label}", fontsize=7, fontweight="bold" if method == "thesis" else "normal")
    axis_3d.set(
        title="Three-objective tradeoff (method means)",
        xlabel="Intervention fraction",
        ylabel="Mean cost",
        zlabel="Safe success",
        zlim=(-0.04, 1.04),
    )
    axis_3d.view_init(elev=elevation, azim=azimuth)

    fig.suptitle("Unitree navigation: three-seed Pareto comparison", fontsize=16, fontweight="bold", y=0.992)
    fig.text(
        0.5,
        0.952,
        "3 independently trained policies per method | 16 matched layouts per policy | 48 rollouts per method | bars = sample SD across seeds",
        ha="center",
        fontsize=10,
        color="#514a40",
    )
    fig.text(
        0.5,
        0.934,
        "Preference-only and BC-only change exactly one auxiliary loss from the combined thesis method.",
        ha="center",
        fontsize=9,
        color="#514a40",
    )
    labels = [LABELS[method] for method in PLOT_METHODS]
    fig.legend(legend_handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.008))
    fig.tight_layout(rect=(0.02, 0.075, 0.98, 0.90))
    fig.savefig(output_path, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_outputs(seed_rows: list[dict], summaries: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("final_metrics_by_seed", seed_rows), ("final_metrics_summary", summaries)):
        (output_dir / f"{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        with (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=Path("models/unitree_mjlab_nav_thesis"))
    parser.add_argument("--run-suffix", default="20260719_paired3seed")
    parser.add_argument("--ablation-suffix", default="20260720_paired3seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/unitree_competitors_20260720_multiseed"))
    parser.add_argument("--pareto-elevation", type=float, default=21.0)
    parser.add_argument("--pareto-azimuth", type=float, default=42.0)
    args = parser.parse_args()

    seed_rows = collect_rows(args.models_root, args.run_suffix, args.ablation_suffix, args.seeds)
    summaries = aggregate_rows(seed_rows)
    write_outputs(seed_rows, summaries, args.output_dir)
    output_path = args.output_dir / "pareto_comparison_multiseed.png"
    plot(seed_rows, summaries, output_path, args.pareto_elevation, args.pareto_azimuth)
    print(output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

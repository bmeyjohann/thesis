#!/usr/bin/env python3
"""Plot matched Unitree competitor learning curves from periodic teacher-free eval."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = ("thesis", "hilserl", "eil", "pvp", "hg_dagger", "sac")
LABELS = {
    "thesis": "Thesis (preference + BC)",
    "hilserl": "HIL-SERL",
    "eil": "EIL",
    "pvp": "PVP",
    "hg_dagger": "HG-DAgger",
    "sac": "Goal-only SAC",
}
COLORS = {
    "thesis": "tab:blue",
    "hilserl": "tab:orange",
    "eil": "tab:green",
    "pvp": "tab:red",
    "hg_dagger": "tab:purple",
    "sac": "tab:brown",
}
MARKERS = {
    "thesis": "o",
    "hilserl": "o",
    "eil": "s",
    "pvp": "o",
    "hg_dagger": "D",
    "sac": "o",
}
ANNOTATION_OFFSETS = {
    "thesis": (7, 8),
    "hilserl": (7, -14),
    "eil": (7, 18),
    "pvp": (7, -16),
    "hg_dagger": (7, -20),
    "sac": (7, -14),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pareto_mask(rows: list[dict], *, x_key: str, y_key: str) -> list[bool]:
    """Return non-dominated points for lower x and higher y."""
    result = []
    for row in rows:
        x = float(row[x_key])
        y = float(row[y_key])
        dominated = any(
            float(other[x_key]) <= x
            and float(other[y_key]) >= y
            and (float(other[x_key]) < x or float(other[y_key]) > y)
            for other in rows
        )
        result.append(not dominated)
    return result


def pareto_mask_3d(rows: list[dict]) -> list[bool]:
    """Return non-dominated points for lower cost/burden and higher safe success."""
    result = []
    for row in rows:
        cost = float(row["mean_cost_sum"])
        burden = float(row["teacher_fraction_cumulative"])
        safe = float(row["safe_success_rate"])
        dominated = any(
            float(other["mean_cost_sum"]) <= cost
            and float(other["teacher_fraction_cumulative"]) <= burden
            and float(other["safe_success_rate"]) >= safe
            and (
                float(other["mean_cost_sum"]) < cost
                or float(other["teacher_fraction_cumulative"]) < burden
                or float(other["safe_success_rate"]) > safe
            )
            for other in rows
        )
        result.append(not dominated)
    return result


def plot_pareto(
    final_rows: list[dict],
    output_path: Path,
    *,
    elevation: float = 18.0,
    azimuth: float = 45.0,
) -> None:
    episode_counts = {int(float(row["episodes"])) for row in final_rows}
    eval_episodes = str(next(iter(episode_counts))) if len(episode_counts) == 1 else "varying episode counts"
    fig = plt.figure(figsize=(15, 11))
    axes = [
        fig.add_subplot(2, 2, 1),
        fig.add_subplot(2, 2, 2),
        fig.add_subplot(2, 2, 3),
    ]
    panels = (
        ("mean_cost_sum", "Mean episode cost", "Safety-cost Pareto frontier"),
        ("costful_episode_rate", "Costful episode fraction", "Safety-incidence Pareto frontier"),
        ("teacher_fraction_cumulative", "Cumulative intervention fraction", "Safety-burden Pareto frontier"),
    )
    for axis, (x_key, x_label, title) in zip(axes, panels):
        frontier = pareto_mask(final_rows, x_key=x_key, y_key="safe_success_rate")
        frontier_points = sorted(
            {
                (float(row[x_key]), float(row["safe_success_rate"]))
                for row, keep in zip(final_rows, frontier)
                if keep
            }
        )
        if frontier_points:
            axis.plot(
                [point[0] for point in frontier_points],
                [point[1] for point in frontier_points],
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.65,
                label="Pareto frontier",
            )
        for row, keep in zip(final_rows, frontier):
            method = str(row["method"])
            x = float(row[x_key])
            y = float(row["safe_success_rate"])
            axis.scatter(
                x,
                y,
                s=150 if keep else 95,
                color=COLORS[method],
                marker=MARKERS[method],
                alpha=0.82,
                edgecolor="black" if keep else "white",
                linewidth=2.0 if keep else 0.8,
                zorder=3,
            )
            offset = ANNOTATION_OFFSETS[method]
            axis.annotate(LABELS[method], (x, y), xytext=offset, textcoords="offset points", fontsize=8)
        axis.set(title=title, xlabel=x_label, ylabel="Teacher-free safe success", ylim=(-0.03, 1.03))
        axis.grid(alpha=0.3)
        axis.legend(loc="best", fontsize=8)

    axis_3d = fig.add_subplot(2, 2, 4, projection="3d")
    frontier_3d = pareto_mask_3d(final_rows)
    for row, keep in zip(final_rows, frontier_3d):
        method = str(row["method"])
        cost = float(row["mean_cost_sum"])
        burden = float(row["teacher_fraction_cumulative"])
        safe = float(row["safe_success_rate"])
        axis_3d.scatter(
            burden,
            cost,
            safe,
            s=150 if keep else 85,
            color=COLORS[method],
            edgecolor="black" if keep else "white",
            linewidth=2.0 if keep else 0.8,
        )
        axis_3d.plot(
            [burden, burden],
            [cost, cost],
            [0.0, safe],
            color=COLORS[method],
            linestyle=":",
            linewidth=1.0,
            alpha=0.55,
        )
        if method in {"thesis", "sac"}:
            label = f"  {LABELS[method]}\n  safe={safe:.2f}, cost={cost:.1f}, int={burden:.2f}"
            axis_3d.text(
                burden,
                cost,
                safe,
                label,
                fontsize=7,
                fontweight="bold" if method == "thesis" else "normal",
                bbox={"facecolor": "white", "edgecolor": COLORS[method], "alpha": 0.78, "pad": 1.5},
            )
        else:
            axis_3d.text(burden, cost, safe, f"  {LABELS[method]}", fontsize=7)
    axis_3d.set(
        title="Three-objective tradeoff",
        xlabel="Intervention fraction (lower is better)",
        ylabel="Mean cost (lower is better)",
        zlabel="Safe success (higher is better)",
        zlim=(-0.03, 1.03),
    )
    axis_3d.view_init(elev=float(elevation), azim=float(azimuth))
    fig.suptitle("Final-checkpoint Pareto comparison (outlined points are non-dominated)", fontsize=14)
    fig.text(
        0.5,
        0.955,
        f"One trained policy per method; {eval_episodes} teacher-free evaluation episodes per policy",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=Path("models/unitree_mjlab_nav_thesis"))
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/unitree_competitors_20260718"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-suffix", default="20260718_v2")
    parser.add_argument("--pareto-elevation", type=float, default=18.0)
    parser.add_argument("--pareto-azimuth", type=float, default=45.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, tuple[list[dict], list[dict]]] = {}
    for method in METHODS:
        run_dir = args.models_root / f"unitree_compare_{method}_seed{args.seed}_{args.run_suffix}"
        runs[method] = (read_jsonl(run_dir / "metrics.jsonl"), read_jsonl(run_dir / "eval_metrics.jsonl"))

    fig, axes_grid = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes = axes_grid.ravel()
    for method, (train_rows, eval_rows) in runs.items():
        label = LABELS[method]
        if eval_rows:
            x = [row["transitions"] for row in eval_rows]
            axes[0].plot(x, [row["success_rate"] for row in eval_rows], marker="o", label=label)
            axes[1].plot(x, [row["safe_success_rate"] for row in eval_rows], marker="o", label=label)
            axes[2].plot(x, [math.sqrt(row["mean_cost_sum"]) for row in eval_rows], marker="o", label=label)
        if train_rows:
            axes[3].plot(
                [row["transitions"] for row in train_rows],
                [row["teacher_fraction_cumulative"] for row in train_rows],
                marker="o",
                label=label,
            )
    axes[0].set(title="Teacher-free success", xlabel="transitions", ylabel="success rate", ylim=(-0.03, 1.03))
    axes[1].set(title="Teacher-free safe success", xlabel="transitions", ylabel="safe success rate", ylim=(-0.03, 1.03))
    axes[2].set(title="Teacher-free safety cost", xlabel="transitions", ylabel="mean episode cost")
    cost_ticks = (0, 1, 5, 10, 25, 50, 100, 300)
    axes[2].set_yticks([math.sqrt(value) for value in cost_ticks], labels=[str(value) for value in cost_ticks])
    axes[3].set(title="Human/teacher burden", xlabel="transitions", ylabel="cumulative intervention fraction", ylim=(-0.03, 1.03))
    for axis in axes:
        axis.grid(alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure_path = args.output_dir / "learning_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    final_rows = []
    for method, (train_rows, eval_rows) in runs.items():
        eval_final = eval_rows[-1] if eval_rows else {}
        train_final = train_rows[-1] if train_rows else {}
        final_rows.append(
            {
                "method": method,
                "label": LABELS[method],
                "transitions": eval_final.get("transitions", train_final.get("transitions")),
                "episodes": eval_final.get("episodes"),
                "success_rate": eval_final.get("success_rate"),
                "safe_success_rate": eval_final.get("safe_success_rate"),
                "mean_cost_sum": eval_final.get("mean_cost_sum"),
                "costful_episode_rate": eval_final.get("costful_episode_rate"),
                "mean_time_to_success_s": eval_final.get("mean_time_to_success_s"),
                "teacher_fraction_cumulative": train_final.get("teacher_fraction_cumulative", 0.0),
            }
        )
    (args.output_dir / "final_metrics.json").write_text(json.dumps(final_rows, indent=2), encoding="utf-8")
    with (args.output_dir / "final_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    plot_pareto(
        final_rows,
        args.output_dir / "pareto_comparison.png",
        elevation=args.pareto_elevation,
        azimuth=args.pareto_azimuth,
    )
    print(figure_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

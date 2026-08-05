#!/usr/bin/env python3
"""Plot one Unitree goal-only run against transitions and serial human time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def add_human_time_axis(axis: plt.Axes, human_control_hz: float) -> None:
    def transitions_k_to_minutes(transitions_k: float) -> float:
        return transitions_k * 1000.0 / human_control_hz / 60.0

    def minutes_to_transitions_k(minutes: float) -> float:
        return minutes * human_control_hz * 60.0 / 1000.0

    secondary = axis.secondary_xaxis(
        "top",
        functions=(transitions_k_to_minutes, minutes_to_transitions_k),
    )
    secondary.set_xlabel(
        f"Equivalent serial human supervision (minutes at {human_control_hz:g} Hz)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--human-control-hz", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.human_control_hz <= 0:
        raise ValueError("--human-control-hz must be positive")

    train_rows = read_jsonl(args.run_dir / "metrics.jsonl")
    eval_rows = read_jsonl(args.run_dir / "eval_metrics.jsonl")
    if not train_rows or not eval_rows:
        raise ValueError("The run must contain non-empty metrics.jsonl and eval_metrics.jsonl")

    output = args.output or args.run_dir / "training_progress_human_time.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    train_x = [float(row["transitions"]) / 1000.0 for row in train_rows]
    eval_x = [float(row["transitions"]) / 1000.0 for row in eval_rows]

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), sharex=True, facecolor="#f3efe5")
    for axis in axes.ravel():
        axis.set_facecolor("#fffdf7")
        axis.grid(alpha=0.25, color="#7a7469")
        axis.set_xlabel("Environment transitions (thousands)")
        add_human_time_axis(axis, args.human_control_hz)

    axes[0, 0].plot(
        train_x,
        [float(row["success_rate"]) for row in train_rows],
        color="#176b87",
        linewidth=2.2,
    )
    axes[0, 0].set(title="Training: cumulative completed-episode success", ylabel="Success rate", ylim=(0, 1.02))

    axes[0, 1].plot(
        train_x,
        [float(row["episode_cost_mean"]) for row in train_rows],
        color="#b04a3a",
        linewidth=2.0,
        label="Completed-episode mean cost",
    )
    axes[0, 1].set(title="Training: collision cost (not in reward)", ylabel="Mean cost")

    axes[1, 0].plot(
        eval_x,
        [float(row["success_rate"]) for row in eval_rows],
        color="#176b87",
        marker="o",
        markersize=8,
        linewidth=2.2,
        label="Success",
    )
    axes[1, 0].plot(
        eval_x,
        [float(row["safe_success_rate"]) for row in eval_rows],
        color="#2b8c56",
        marker="s",
        markersize=8,
        linewidth=2.2,
        label="Safe success",
    )
    axes[1, 0].set(title="Teacher-free deterministic evaluation", ylabel="Episode rate", ylim=(0, 1.02))
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(
        eval_x,
        [float(row["costful_episode_rate"]) for row in eval_rows],
        color="#d17a22",
        marker="o",
        markersize=8,
        linewidth=2.2,
        label="Costful episode rate",
    )
    axes[1, 1].set(title="Teacher-free evaluation safety diagnostics", ylabel="Costful episode rate", ylim=(0, 1.02))
    cost_axis = axes[1, 1].twinx()
    cost_axis.plot(
        eval_x,
        [float(row["mean_cost_sum"]) for row in eval_rows],
        color="#8f2d24",
        marker="D",
        markersize=7,
        linestyle="--",
        linewidth=1.9,
        label="Mean cost",
    )
    cost_axis.set_ylabel("Mean cost")
    handles_a, labels_a = axes[1, 1].get_legend_handles_labels()
    handles_b, labels_b = cost_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles_a + handles_b, labels_a + labels_b, frameon=False)

    final_transitions = float(eval_rows[-1]["transitions"])
    final_minutes = final_transitions / args.human_control_hz / 60.0
    fig.suptitle(
        "Obstacle-exposed Unitree goal-only learning",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        (
            f"One run, 8 parallel training environments; deterministic evaluations use "
            f"{int(float(eval_rows[-1]['episodes']))} episodes/checkpoint. "
            f"{final_transitions / 1000:.0f}k transitions = {final_minutes:.1f} minutes "
            f"of serial {args.human_control_hz:g} Hz human observation. "
            "Collision cost is logged but excluded from reward."
        ),
        ha="center",
        fontsize=9.5,
        color="#4c473e",
    )
    fig.tight_layout(rect=(0.025, 0.025, 0.98, 0.91))
    fig.savefig(output, dpi=190, facecolor=fig.get_facecolor())
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

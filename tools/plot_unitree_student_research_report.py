#!/usr/bin/env python3
"""Build training and policy-only comparison artifacts for Unitree navigation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, float]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_eval(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("eval must use LABEL=/path/to/metrics.json")
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def _metric(rows: list[dict[str, float]], name: str) -> np.ndarray:
    return np.asarray([float(row.get(name, np.nan)) for row in rows], dtype=np.float64)


def _plot_training(rows: list[dict[str, float]], output: Path) -> None:
    steps = _metric(rows, "step")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

    ax = axes[0, 0]
    ax.plot(steps, _metric(rows, "success_rate"), label="protected completed episodes", lw=2)
    ax.plot(steps, _metric(rows, "ongoing_success_fraction"), label="current envs", alpha=0.65)
    ax.set(title="Training success", ylabel="fraction", ylim=(-0.03, 1.03))
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, _metric(rows, "teacher_fraction_interval"), label="interval", alpha=0.75)
    ax.plot(steps, _metric(rows, "teacher_fraction_cumulative"), label="cumulative", lw=2)
    ax.plot(steps, _metric(rows, "intervention_clearance_trigger_fraction"), label="clearance trigger", alpha=0.6)
    ax.plot(steps, _metric(rows, "intervention_stall_trigger_fraction"), label="stall trigger", alpha=0.6)
    ax.set(title="Teacher intervention", ylabel="fraction", ylim=(-0.03, 1.03))
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    ax.plot(steps, _metric(rows, "teacher_executed_cost_sum_interval"), label="teacher-executed")
    ax.plot(steps, _metric(rows, "student_executed_cost_sum_interval"), label="student-executed")
    ax.plot(steps, _metric(rows, "episode_cost_mean"), label="completed episode mean", ls="--")
    ax.set(title="Protected-rollout cost", xlabel="environment iteration", ylabel="cost")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(steps, _metric(rows, "q_min_pi_mean"), label="Q(policy)")
    ax.plot(steps, _metric(rows, "q_min_data_mean"), label="Q(data)")
    ax2 = ax.twinx()
    ax2.plot(steps, _metric(rows, "pref_lambda"), color="#c03b2b", label="preference lambda", alpha=0.75)
    ax.set(title="Critic and preference constraint", xlabel="environment iteration", ylabel="Q")
    ax2.set_ylabel("lambda", color="#c03b2b")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Unitree scan-CNN student: protected training diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _eval_summary(label: str, path: Path) -> dict[str, float | str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode_costs = np.asarray(
        [float(episode.get("cost_sum", 0.0)) for episode in payload.get("episodes", [])],
        dtype=np.float64,
    )
    return {
        "label": label,
        "controller": str(payload.get("controller", "")),
        "num_episodes": int(payload.get("num_episodes", 0)),
        "success_rate": float(payload.get("success_rate", np.nan)),
        "mean_cost_sum": float(payload.get("mean_cost_sum", np.nan)),
        "median_cost_sum": float(np.median(episode_costs)) if episode_costs.size else np.nan,
        "p90_cost_sum": float(np.percentile(episode_costs, 90.0)) if episode_costs.size else np.nan,
        "max_cost_sum": float(np.max(episode_costs)) if episode_costs.size else np.nan,
        "costful_episode_rate": float(payload.get("costful_episode_rate", np.nan)),
        "mean_collision_steps": float(payload.get("mean_collision_steps", np.nan)),
        "mean_time_to_success_s_success_only": float(
            payload.get("mean_time_to_success_s_success_only", np.nan)
        ),
        "mean_episode_length_s": float(payload.get("mean_episode_length_s", np.nan)),
        "mean_return": float(payload.get("mean_return", np.nan)),
        "metrics_path": str(path),
    }


def _plot_evals(rows: list[dict[str, float | str]], output: Path) -> None:
    labels = [str(row["label"]) for row in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    fields = [
        ("success_rate", "Success rate", (0.0, 1.05)),
        ("mean_cost_sum", "Mean episode cost", None),
        ("costful_episode_rate", "Costful episode rate", (0.0, 1.05)),
        ("mean_time_to_success_s_success_only", "Time to goal (successful episodes, s)", None),
    ]
    colors = ["#2878b5", "#d35432", "#e6a23c", "#4d9c6c"]
    for ax, (field, title, ylim), color in zip(axes.flat, fields, colors):
        values = [float(row[field]) for row in rows]
        bars = ax.bar(x, values, color=color, alpha=0.88)
        if field == "mean_cost_sum":
            p90 = [float(row["p90_cost_sum"]) for row in rows]
            ax.scatter(x, p90, marker="D", color="black", s=28, zorder=4, label="p90 cost")
            ax.legend(fontsize=8)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Policy-only canonical evaluation (identical seeded layout sequence)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--eval", action="append", type=_parse_eval, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_rows = _load_jsonl(args.training_metrics)
    _plot_training(training_rows, args.output_dir / "training_diagnostics.png")

    eval_rows = [_eval_summary(label, path) for label, path in args.eval]
    if eval_rows:
        _plot_evals(eval_rows, args.output_dir / "policy_only_comparison.png")
        with (args.output_dir / "policy_only_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(eval_rows[0]))
            writer.writeheader()
            writer.writerows(eval_rows)
        (args.output_dir / "policy_only_metrics.json").write_text(
            json.dumps(eval_rows, indent=2), encoding="utf-8"
        )

    print(json.dumps({"output_dir": str(args.output_dir), "eval_rows": len(eval_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

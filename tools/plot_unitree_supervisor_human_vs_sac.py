#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "logs/unitree_mjlab/human_checkpoint_sweep_20260806"
TRAIN_LOG = ROOT / "models/unitree_mjlab_nav_human/unitree_human_scratch_20260803_092823_retry03/metrics.jsonl"
OUTPUT = ROOT / "visualizations/unitree_human_vs_sac_20260806/supervisor_performance_comparison.png"
FULL_LABEL = "Pref + BC"
BASELINE_LABEL = "Goal-only SAC"
HUMAN_CONTROL_HZ = 20.0


def load_eval(name: str) -> dict:
    data = json.loads((EVAL_ROOT / name / "policy_metrics.json").read_text())
    data["safe_success_rate"] = np.mean(
        [bool(ep["success"]) and float(ep["cost_sum"]) == 0.0 for ep in data["episodes"]]
    )
    return data


def rolling(values: np.ndarray, window: int = 10) -> np.ndarray:
    return np.asarray([values[max(0, i - window + 1) : i + 1].mean() for i in range(len(values))])


def add_human_time_axis(axis: plt.Axes) -> None:
    secondary = axis.secondary_xaxis(
        "top",
        functions=(
            lambda steps_k: steps_k * 1000.0 / HUMAN_CONTROL_HZ / 60.0,
            lambda minutes: minutes * HUMAN_CONTROL_HZ * 60.0 / 1000.0,
        ),
    )
    secondary.set_xlabel(f"elapsed human supervision (minutes at {HUMAN_CONTROL_HZ:g} Hz)")


def main() -> int:
    full = load_eval("retry03_final")
    baseline = load_eval("sac_reward_only_final")
    names = [FULL_LABEL, BASELINE_LABEL]
    results = [full, baseline]
    colors = ["#087e8b", "#c44536"]

    checkpoints = []
    for path in EVAL_ROOT.glob("retry03_step_*/policy_metrics.json"):
        step = int(re.search(r"step_(\d+)", str(path)).group(1))
        data = json.loads(path.read_text())
        data["safe_success_rate"] = np.mean(
            [bool(ep["success"]) and float(ep["cost_sum"]) == 0.0 for ep in data["episodes"]]
        )
        checkpoints.append((step, data))
    checkpoints.sort()

    train = [json.loads(line) for line in TRAIN_LOG.read_text().splitlines() if line.strip()]
    steps = np.asarray([float(row["step"]) for row in train if "intervention_fraction" in row])
    cumulative = np.asarray([float(row["intervention_fraction"]) for row in train if "intervention_fraction" in row])
    counts = cumulative * steps
    interval = np.clip(np.diff(np.r_[0.0, counts]) / np.maximum(np.diff(np.r_[0.0, steps]), 1.0), 0.0, 1.0)

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    x = np.arange(2)

    ax = axes[0, 0]
    ax.bar(x - 0.18, [r["success_rate"] for r in results], 0.36, color="#4c78a8", label="success")
    ax.bar(x + 0.18, [r["safe_success_rate"] for r in results], 0.36, color="#59a14f", label="safe success")
    ax.set(title="Goal reaching", ylabel="episode rate", xticks=x, xticklabels=names, ylim=(0, 0.5))
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.bar(x, [r["mean_cost_sum"] for r in results], color=colors)
    ax.set(title="Mean safety cost", ylabel="cost per episode", xticks=x, xticklabels=names)
    for idx, result in enumerate(results):
        ax.text(idx, result["mean_cost_sum"] + 0.5, f'{result["mean_cost_sum"]:.1f}', ha="center")

    ax = axes[0, 2]
    ax.bar(x, [r["costful_episode_rate"] for r in results], color=colors)
    ax.set(title="Episodes with any safety cost", ylabel="episode rate", xticks=x, xticklabels=names, ylim=(0, 0.55))
    for idx, result in enumerate(results):
        ax.text(idx, result["costful_episode_rate"] + 0.015, f'{result["costful_episode_rate"]:.0%}', ha="center")

    ax = axes[1, 0]
    cp_steps = [step / 1000 for step, _ in checkpoints]
    ax.plot(cp_steps, [d["success_rate"] for _, d in checkpoints], marker="o", color="#4c78a8", label="success")
    ax.plot(cp_steps, [d["safe_success_rate"] for _, d in checkpoints], marker="s", color="#59a14f", label="safe success")
    ax.axhline(baseline["success_rate"], color="#c44536", linestyle="--", label="goal-only SAC final")
    ax.set(title="Teacher-free learning progression", xlabel="online training steps (thousands)", ylabel="episode rate", ylim=(0, 0.5))
    add_human_time_axis(ax)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.plot(steps / 1000, rolling(interval), color="#087e8b", linewidth=2.5, label="recent actual intervention (1k-step mean)")
    ax.plot(steps / 1000, cumulative, color="#f28e2b", linewidth=2.2, label="cumulative actual intervention")
    ax.set(title="Human supervision burden", xlabel="online training steps (thousands)", ylabel="fraction of environment steps", ylim=(0, 1))
    add_human_time_axis(ax)
    ax.legend(frameon=False, fontsize=9)
    active_minutes = counts[-1] / HUMAN_CONTROL_HZ / 60.0
    ax.annotate(
        f'{int(round(counts[-1])):,} / {int(steps[-1]):,} steps\n{active_minutes:.2f} min active steering',
        xy=(steps[-1] / 1000, cumulative[-1]),
        xytext=(-145, 35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->"},
    )

    ax = axes[1, 2]
    width = 0.22
    obstacle_metrics = [
        ("front_obstacle_abs_steering", "|steering|"),
        ("front_obstacle_small_steering_fraction", "near-zero"),
        ("mean_front_obstacle_directional_consistency", "direction consistency"),
    ]
    for offset, (key, label) in enumerate(obstacle_metrics):
        ax.bar(x + (offset - 1) * width, [r[key] for r in results], width, label=label)
    ax.set(title="Action when obstacle is visible ahead", ylabel="metric", xticks=x, xticklabels=names, ylim=(0, 0.9))
    ax.legend(frameon=False, fontsize=9)

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
        if ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]):
            ax.tick_params(axis="x", rotation=10)
    fig.suptitle("Human intervention improves goal-reaching behavior over goal-only SAC", fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        -0.005,
        "Pref + BC = SAC + preference loss + intervention behavior cloning. Single trained policy per method; deterministic teacher-free evaluation on the same 100 blocked episodes.",
        ha="center",
        fontsize=10,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

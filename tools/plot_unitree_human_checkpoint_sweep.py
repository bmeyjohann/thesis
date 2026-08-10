#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "logs/unitree_mjlab/human_checkpoint_sweep_20260806"
OUTPUT = ROOT / "visualizations/unitree_human_checkpoint_sweep_20260806"


def load(name: str) -> dict:
    return json.loads((INPUT / name / "policy_metrics.json").read_text())


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    for path in INPUT.glob("retry03_step_*/policy_metrics.json"):
        step = int(re.search(r"step_(\d+)", str(path)).group(1))
        checkpoints.append((step, json.loads(path.read_text())))
    checkpoints.sort()
    finals = {
        "new human": load("retry03_final"),
        "previous human": load("retry02_final"),
        "reward-only SAC": load("sac_reward_only_final"),
    }

    fields = [
        "success_rate", "safe_success_rate", "mean_cost_sum", "costful_episode_rate",
        "front_obstacle_abs_steering", "front_obstacle_small_steering_fraction",
        "front_obstacle_action_delta", "mean_front_obstacle_directional_consistency",
        "mean_action_delta", "mean_action_sign_flips", "mean_small_action_fraction",
    ]
    with (OUTPUT / "checkpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", *fields])
        writer.writeheader()
        for step, data in checkpoints:
            data["safe_success_rate"] = sum(
                bool(ep["success"]) and float(ep["cost_sum"]) == 0.0 for ep in data["episodes"]
            ) / len(data["episodes"])
            writer.writerow({"step": step, **{field: data[field] for field in fields}})
    for data in finals.values():
        data["safe_success_rate"] = sum(
            bool(ep["success"]) and float(ep["cost_sum"]) == 0.0 for ep in data["episodes"]
        ) / len(data["episodes"])
    (OUTPUT / "final_comparison.json").write_text(
        json.dumps({name: {field: data[field] for field in fields} for name, data in finals.items()}, indent=2) + "\n"
    )

    steps = [step / 1000 for step, _ in checkpoints]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(steps, [d["success_rate"] for _, d in checkpoints], marker="o", color="#087e8b")
    ax.plot(steps, [d["safe_success_rate"] for _, d in checkpoints], marker="s", color="#5c946e", label="safe success")
    ax.set(title="Teacher-free success", ylabel="success rate")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.plot(steps, [d["mean_cost_sum"] for _, d in checkpoints], marker="o", color="#c44536")
    ax.set(title="Safety cost", ylabel="mean episode cost")
    ax = axes[0, 2]
    ax.plot(steps, [d["costful_episode_rate"] for _, d in checkpoints], marker="o", color="#ef8354")
    ax.set(title="Episodes with cost", ylabel="fraction")
    ax = axes[1, 0]
    ax.plot(steps, [d["front_obstacle_abs_steering"] for _, d in checkpoints], marker="o", label="|steering|")
    ax.plot(steps, [d["front_obstacle_small_steering_fraction"] for _, d in checkpoints], marker="s", label="near-zero fraction")
    ax.set(title="When obstacle is visible ahead", ylabel="magnitude / fraction")
    ax.legend(frameon=False)
    ax = axes[1, 1]
    ax.plot(steps, [d["front_obstacle_action_delta"] for _, d in checkpoints], marker="o", label="action change")
    ax.plot(steps, [d["mean_front_obstacle_directional_consistency"] for _, d in checkpoints], marker="s", label="direction consistency")
    ax.set(title="Obstacle-front stability", ylabel="metric")
    ax.legend(frameon=False)
    ax = axes[1, 2]
    names = list(finals)
    x = list(range(len(names)))
    ax.bar([v - 0.3 for v in x], [finals[n]["success_rate"] for n in names], 0.2, label="success")
    ax.bar([v - 0.1 for v in x], [finals[n]["safe_success_rate"] for n in names], 0.2, label="safe success")
    ax.bar([v + 0.1 for v in x], [finals[n]["costful_episode_rate"] for n in names], 0.2, label="costful episode")
    ax.bar([v + 0.3 for v in x], [finals[n]["front_obstacle_small_steering_fraction"] for n in names], 0.2, label="near-zero steering")
    ax.set(title="Final policies on identical 100 episodes", xticks=x, xticklabels=names, ylabel="fraction")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        if ax not in (axes[1, 2],):
            ax.set_xlabel("online training steps (thousands)")
    fig.suptitle("Unitree human-intervention checkpoint sweep", fontsize=16)
    fig.savefig(OUTPUT / "checkpoint_sweep_comparison.png", dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

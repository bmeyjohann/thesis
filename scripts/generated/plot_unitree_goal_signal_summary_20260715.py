#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/benjamin/thesis")
RUNS = [
    (
        "Direct geometric\nbaseline",
        ROOT / "logs/unitree_mjlab/goal_only_diagnostic/direct_goal_no_obstacles_seed973_20260715/direct_goal_metrics.json",
    ),
    (
        "BC-only\n2k",
        ROOT / "logs/unitree_mjlab/goal_signal_diagnostic_40ep_20260715/bc_only_step2000/policy_metrics.json",
    ),
    (
        "SAC reward x1\n+ failure -2",
        ROOT / "logs/unitree_mjlab/goal_signal_diagnostic_40ep_20260715/sac_utd1_failure2_step1000/policy_metrics.json",
    ),
    (
        "SAC reward x10\nUTD 1",
        ROOT / "logs/unitree_mjlab/goal_signal_diagnostic_40ep_20260715/sac_utd1_scale10_step1000/policy_metrics.json",
    ),
]

labels = []
metrics = []
for label, path in RUNS:
    labels.append(label)
    with path.open() as handle:
        data = json.load(handle)
    episodes = data["episodes"]
    safe_success = sum(bool(ep["success"]) and float(ep["cost_sum"]) == 0.0 for ep in episodes) / len(episodes)
    metrics.append(
        {
            "success": float(data["success_rate"]),
            "safe_success": safe_success,
            "cost": float(data["mean_cost_sum"]),
            "flips": float(data["mean_action_sign_flips"]),
        }
    )

x = np.arange(len(labels))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

width = 0.35
axes[0].bar(x - width / 2, [m["success"] for m in metrics], width, label="Success", color="#337a8c")
axes[0].bar(x + width / 2, [m["safe_success"] for m in metrics], width, label="Safe success", color="#7bb56b")
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Episode fraction")
axes[0].legend(frameon=False)

axes[1].bar(x, [m["cost"] for m in metrics], color="#c7644c")
axes[1].set_ylabel("Mean cost / episode")

axes[2].bar(x, [m["flips"] for m in metrics], color="#d6a144")
axes[2].set_ylabel("Mean action sign flips / episode")

for ax in axes:
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("No-obstacle goal-learning diagnostic (40 deterministic episodes, seed 973)", fontsize=14)
out = ROOT / "visualizations/unitree_goal_signal_diagnostic_20260715/robust_40ep_comparison.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180)
print(out)
for label, metric in zip(labels, metrics):
    print(label.replace("\n", " "), metric)

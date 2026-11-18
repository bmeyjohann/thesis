#!/usr/bin/env python3
"""Plot curriculum training and fixed hard-cohort evaluation metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = _read_jsonl(args.run_dir / "metrics.jsonl")
    evaluation = _read_jsonl(args.run_dir / "eval_metrics.jsonl")
    tx = np.asarray([row["transitions"] for row in train], dtype=float)
    ex = np.asarray([row["transitions"] for row in evaluation], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(tx, [row["success_rate"] for row in train], label="training success", alpha=0.75)
    ax.plot(ex, [row["success_rate"] for row in evaluation], "o-", lw=2.2, label="hard eval success")
    ax.plot(ex, [row["safe_success_rate"] for row in evaluation], "s--", lw=1.8, label="hard eval safe success")
    ax.set_ylabel("episode fraction")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(tx, [row["fall_rate"] for row in train], label="training fall")
    ax.plot(tx, [row["timeout_rate"] for row in train], label="training timeout")
    ax.plot(ex, [row["costful_episode_rate"] for row in evaluation], "o-", lw=2.2, label="hard eval costful")
    ax.set_ylabel("episode fraction")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(tx, [row["layout_straight_path_length_mean"] for row in train], label="sampled route length")
    ax.plot(tx, [row["goal_distance_mean"] for row in train], label="current goal distance")
    ax.plot(ex, [row["mean_min_goal_distance"] for row in evaluation], "o-", lw=2.2, label="hard eval closest distance")
    ax.set_ylabel("distance (m)")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(tx, [row.get("q_min_pi_mean", np.nan) for row in train], label="Q(policy)")
    ax.plot(tx, [row.get("q_min_data_mean", np.nan) for row in train], label="Q(data)")
    ax.plot(tx, [row.get("target_q_mean", np.nan) for row in train], label="target Q")
    alpha_ax = ax.twinx()
    alpha_ax.plot(tx, [row.get("alpha", np.nan) for row in train], color="#d1495b", alpha=0.7, label="alpha")
    ax.set_ylabel("Q value")
    alpha_ax.set_ylabel("entropy alpha")
    lines, labels = ax.get_legend_handles_labels()
    alpha_lines, alpha_labels = alpha_ax.get_legend_handles_labels()
    ax.legend(lines + alpha_lines, labels + alpha_labels, loc="best")

    for ax in axes.flat:
        ax.set_xlabel("environment transitions")
        ax.grid(alpha=0.25)
    fig.suptitle("Unitree target-terrain SAC: close-to-far curriculum", fontsize=15)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

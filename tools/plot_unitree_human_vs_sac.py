#!/usr/bin/env python3
"""Plot matched teacher-free evaluation of online-human training and SAC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CHECKPOINTS = ((5000, "5k"), (10000, "10k"), (15000, "15k"),
               (20000, "20k"), (25000, "25k"), (28770, "final"))


def _load(root: Path, prefix: str) -> dict[str, np.ndarray]:
    rows = []
    for step, tag in CHECKPOINTS:
        path = root / f"{prefix}_{tag}_teacherfree_100" / "policy_metrics.json"
        row = json.loads(path.read_text())
        rows.append((step, row))
    keys = ("success_rate", "mean_cost_sum", "costful_episode_rate", "mean_min_goal_distance")
    return {
        "step": np.asarray([row[0] for row in rows], dtype=float),
        **{key: np.asarray([row[1][key] for row in rows], dtype=float) for key in keys},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    human = _load(args.eval_root, "unitree_human_scratch_retry02")
    sac = _load(args.eval_root, "unitree_sac_rewardonly")
    colors = {"Human interventions": "#087e8b", "Reward-only SAC": "#d1495b"}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    panels = (
        ("success_rate", "Teacher-free success", "success rate", (0, 1.03)),
        ("mean_cost_sum", "Safety cost", "mean cost / episode", None),
        ("costful_episode_rate", "Episodes with any cost", "episode fraction", (0, 1.03)),
        ("mean_min_goal_distance", "Closest approach to goal", "distance (m)", None),
    )
    for axis, (key, title, ylabel, ylim) in zip(axes.flat, panels):
        for label, values in (("Human interventions", human), ("Reward-only SAC", sac)):
            axis.plot(values["step"], values[key], marker="o", linewidth=2.2,
                      color=colors[label], label=label)
        axis.set(title=title, xlabel="training transitions", ylabel=ylabel)
        if ylim is not None:
            axis.set_ylim(*ylim)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Unitree navigation: matched 100-layout teacher-free evaluation")
    fig.savefig(args.output_dir / "human_vs_reward_only_sac_checkpoint_curves.png", dpi=180)
    plt.close(fig)

    final = {
        "human_interventions": {key: float(human[key][-1]) for key in panels_as_keys()},
        "reward_only_sac": {key: float(sac[key][-1]) for key in panels_as_keys()},
    }
    (args.output_dir / "human_vs_reward_only_sac_final.json").write_text(
        json.dumps(final, indent=2) + "\n"
    )
    return 0


def panels_as_keys() -> tuple[str, ...]:
    return ("success_rate", "mean_cost_sum", "costful_episode_rate", "mean_min_goal_distance")


if __name__ == "__main__":
    raise SystemExit(main())

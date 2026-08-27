#!/usr/bin/env python3
"""Plot shared-student learning curves with interval termination rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODALITIES = ("height_scan", "depth", "mono_rgb", "stereo_rgb")


def _smooth(values: np.ndarray, width: int = 7) -> np.ndarray:
    if len(values) < width:
        return values
    return np.convolve(values, np.ones(width) / width, mode="same")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metrics_path in sorted(args.training_root.glob("*/metrics.jsonl")):
        architecture = metrics_path.parent.name
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        if not rows:
            continue
        steps = np.asarray([row["global_step"] for row in rows])
        rates = []
        previous_cell_step = previous_terminations = 0
        for row in rows:
            if row["cell_step"] <= previous_cell_step:
                previous_cell_step = previous_terminations = 0
            delta_steps = row["cell_step"] - previous_cell_step
            delta_terminations = row["termination_count"] - previous_terminations
            num_envs = sum(row[f"assigned_{name}"] for name in MODALITIES)
            rates.append(1000.0 * delta_terminations / max(delta_steps * num_envs, 1))
            previous_cell_step = row["cell_step"]
            previous_terminations = row["termination_count"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        axes[0, 0].plot(steps, _smooth(np.asarray([row["bc_loss"] for row in rows])), label="overall", linewidth=2)
        for modality in MODALITIES:
            values = np.asarray([row[f"bc_loss_{modality}"] for row in rows])
            axes[0, 0].plot(steps, _smooth(values), label=modality, alpha=0.85)
        axes[0, 0].set_title("Teacher-action imitation loss")
        axes[0, 0].legend(ncol=2)
        axes[0, 1].plot(steps, [row["mean_step_reward"] for row in rows], color="#397367")
        axes[0, 1].set_title("Expert-controlled rollout reward")
        axes[1, 0].plot(steps, _smooth(np.asarray(rates)), color="#c44900")
        axes[1, 0].set_title("Expert rollout terminations / 1,000 env steps")
        reconstruction = np.asarray([row.get("reconstruction_loss", 0.0) for row in rows])
        axes[1, 1].plot(steps, _smooth(reconstruction), color="#7a5195")
        axes[1, 1].set_title("Privileged height reconstruction loss")
        for axis in axes.flat:
            axis.set_xlabel("optimizer steps")
            axis.grid(alpha=0.25)
        fig.suptitle(f"Shared multimodal training: {architecture}", fontsize=16)
        output = args.output_dir / f"{architecture}_training_curves.png"
        fig.savefig(output, dpi=180)
        plt.close(fig)
        print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

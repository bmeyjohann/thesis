#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "models/unitree_mjlab_nav_human/unitree_human_scratch_20260803_092823_retry03/metrics.jsonl"
OUTPUT = ROOT / "visualizations/unitree_human_vs_sac_20260806/human_intervention_over_time.png"


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty_like(values)
    for index in range(len(values)):
        result[index] = values[max(0, index - window + 1) : index + 1].mean()
    return result


def main() -> int:
    rows = [json.loads(line) for line in METRICS.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "intervention_fraction" in row]
    steps = np.asarray([float(row["step"]) for row in rows])
    cumulative = np.asarray([float(row["intervention_fraction"]) for row in rows])
    cumulative_count = cumulative * steps
    interval_steps = np.diff(np.r_[0.0, steps])
    interval_count = np.diff(np.r_[0.0, cumulative_count])
    interval_rate = np.clip(interval_count / np.maximum(interval_steps, 1.0), 0.0, 1.0)

    batch_rows = [row for row in rows if "batch_teacher_fraction" in row]
    batch_steps = np.asarray([float(row["step"]) for row in batch_rows])
    batch_fraction = np.asarray([float(row["batch_teacher_fraction"]) for row in batch_rows])

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(steps / 1000.0, interval_rate, color="#8ec6c5", alpha=0.35, linewidth=1, label="interval intervention rate (100 steps)")
    ax.plot(steps / 1000.0, rolling_mean(interval_rate, 10), color="#087e8b", linewidth=2.5, label="recent intervention rate (1k-step mean)")
    ax.plot(steps / 1000.0, cumulative, color="#f28e2b", linewidth=2.2, label="cumulative intervention fraction")
    ax.plot(batch_steps / 1000.0, rolling_mean(batch_fraction, 10), color="#7b2cbf", linestyle="--", linewidth=2, label="teacher rows in replay batch")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(
        title="Human intervention over online training",
        xlabel="online training steps (thousands)",
        ylabel="fraction of steps",
        xlim=(0, steps.max() / 1000.0),
        ylim=(-0.02, 1.02),
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    ax.annotate(
        f"final cumulative: {cumulative[-1]:.1%}",
        xy=(steps[-1] / 1000.0, cumulative[-1]),
        xytext=(-125, 30),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#f28e2b"},
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

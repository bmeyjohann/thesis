#!/usr/bin/env python3
"""Plot teacher-free checkpoint evaluation metrics from Unitree training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Unitree teacher-free checkpoint evaluation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.metrics_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No evaluation rows in {args.metrics_jsonl}")

    steps = [int(row["step"]) for row in rows]
    success = [100.0 * float(row["success_rate"]) for row in rows]
    safe_success = [100.0 * float(row["safe_success_rate"]) for row in rows]
    costs = [float(row["mean_cost_sum"]) for row in rows]
    flips = [float(row["mean_action_sign_flips"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=160)
    axes[0].plot(steps, success, marker="o", label="success")
    axes[0].plot(steps, safe_success, marker="s", label="safe success")
    axes[0].set_ylabel("Rate (%)")
    axes[0].legend(frameon=False)
    axes[1].plot(steps, costs, marker="o", color="#b33a3a")
    axes[1].set_ylabel("Mean episode cost")
    axes[2].plot(steps, flips, marker="o", color="#356a8a")
    axes[2].set_ylabel("Mean action sign flips")
    for axis in axes:
        axis.set_xlabel("Training step")
        axis.grid(alpha=0.25)
    fig.suptitle(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

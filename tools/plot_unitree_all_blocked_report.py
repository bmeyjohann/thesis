#!/usr/bin/env python3
"""Plot matched forced-blocked Unitree navigation benchmark metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", action="append", required=True, metavar="LABEL=METRICS_JSON")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels: list[str] = []
    rows: list[dict[str, object]] = []
    for item in args.series:
        label, separator, raw_path = item.partition("=")
        if not separator:
            raise ValueError(f"Expected LABEL=METRICS_JSON, got {item!r}")
        labels.append(label)
        rows.append(json.loads(Path(raw_path).read_text(encoding="utf-8")))

    success = [100.0 * float(row["success_rate"]) for row in rows]
    costful = [100.0 * float(row["costful_episode_rate"]) for row in rows]
    costs = [float(row["mean_cost_sum"]) for row in rows]
    colors = ["#9b3a2f", "#cc8b24", "#297c70", "#285f8f"][: len(rows)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=180)
    axes[0].bar(labels, success, color=colors)
    axes[0].set_ylabel("Success rate (%)")
    axes[0].set_ylim(0, 105)
    axes[1].bar(labels, costful, color=colors)
    axes[1].set_ylabel("Costful episodes (%)")
    axes[1].set_ylim(0, 105)
    axes[2].bar(labels, costs, color=colors)
    axes[2].set_ylabel("Mean episode cost (symlog)")
    axes[2].set_yscale("symlog", linthresh=1.0)
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Unitree navigation: every start-goal corridor is blocked (16 episodes)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create aggregate shared-student metrics and teacher-relative summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    ("mean_velocity_tracking_error", "Velocity tracking error", False),
    ("fall_events_per_1000_env_steps", "Falls / 1,000 env steps", False),
    ("mean_achieved_speed", "Achieved planar speed", True),
    ("mean_abs_action_delta", "Action jitter", False),
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    students, expert = _read(args.students), _read(args.expert)
    grouped = defaultdict(list)
    for row in students:
        grouped[(row["architecture"], row["modality"])].append(row)
    policies = sorted(grouped)
    labels = [f"{architecture}\n{modality}" for architecture, modality in policies]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    summary = {"expert": {}, "students": {}}
    for axis, (metric, title, higher_better) in zip(axes.flat, METRICS):
        expert_values = np.asarray([float(row[metric]) for row in expert])
        means, errors = [], []
        for policy in policies:
            values = np.asarray([float(row[metric]) for row in grouped[policy]])
            means.append(float(values.mean()))
            errors.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
            summary["students"].setdefault("/".join(policy), {})[metric] = {
                "mean": float(values.mean()), "std": errors[-1], "count": len(values),
            }
        summary["expert"][metric] = {
            "mean": float(expert_values.mean()),
            "std": float(expert_values.std(ddof=1)), "count": len(expert_values),
        }
        axis.bar(range(len(policies)), means, yerr=errors, capsize=3, color="#397367")
        axis.axhline(expert_values.mean(), color="#c44900", linestyle="--", label="privileged expert")
        axis.set_title(title + (" (higher is better)" if higher_better else " (lower is better)"))
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("Shared multimodal locomotion students vs privileged expert", fontsize=17)
    figure_path = args.output_dir / "shared_student_aggregate_metrics.png"
    fig.savefig(figure_path, dpi=180)

    teacher_score = summary["expert"]["mean_velocity_tracking_error"]["mean"] + 0.05 * summary["expert"]["fall_events_per_1000_env_steps"]["mean"]
    summary["selection_score"] = {
        "definition": "tracking_error + 0.05 * falls_per_1000_env_steps",
        "expert": teacher_score,
        "students": {},
    }
    for policy in summary["students"]:
        values = summary["students"][policy]
        score = values["mean_velocity_tracking_error"]["mean"] + 0.05 * values["fall_events_per_1000_env_steps"]["mean"]
        summary["selection_score"]["students"][policy] = score
    (args.output_dir / "shared_student_aggregate_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(figure_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot comparable Unitree navigation training diagnostics from local JSONL logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metrics in {path}")
    return rows


def finite_xy(rows: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (float(row["transitions"]), float(row[key]))
        for row in rows
        if key in row and np.isfinite(float(row[key]))
    ]
    if not pairs:
        return np.zeros(0), np.zeros(0)
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="LABEL=run_dir_or_metrics.jsonl")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    runs: list[tuple[str, Path, list[dict[str, float]]]] = []
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"Expected LABEL=PATH, got {spec!r}")
        label, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if path.is_dir():
            path = path / "metrics.jsonl"
        runs.append((label, path, load_rows(path)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("success_rate", "Completed success rate", (0.0, 1.05)),
        ("episode_cost_mean", "Completed mean cost", None),
        ("teacher_fraction_interval", "Teacher intervention fraction", (0.0, 1.05)),
        ("goal_distance_mean", "Mean goal distance (m)", None),
        ("q_min_pi_mean", "Q min: policy action", None),
        ("pref_lambda", "Preference lambda", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=160)
    for ax, (key, title, ylim) in zip(axes.reshape(-1), panels):
        for label, _, rows in runs:
            x, y = finite_xy(rows, key)
            if x.size:
                ax.plot(x, y, linewidth=2, label=label)
        ax.set_title(title)
        ax.set_xlabel("environment transitions")
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, min(4, len(labels))))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plot_path = out_dir / "unitree_student_training_comparison.png"
    fig.savefig(plot_path)
    plt.close(fig)

    summary = []
    for label, path, rows in runs:
        last = rows[-1]
        summary.append(
            {
                "label": label,
                "metrics_path": str(path),
                "transitions": last.get("transitions"),
                "success_rate": last.get("success_rate"),
                "episode_cost_mean": last.get("episode_cost_mean"),
                "teacher_fraction_interval": last.get("teacher_fraction_interval"),
                "teacher_fraction_cumulative": last.get("teacher_fraction_cumulative"),
                "goal_distance_mean": last.get("goal_distance_mean"),
                "q_min_pi_mean": last.get("q_min_pi_mean"),
                "q_min_data_mean": last.get("q_min_data_mean"),
                "pref_lambda": last.get("pref_lambda"),
                "critic_loss_total": last.get("critic_loss_total"),
                "actor_loss": last.get("actor_loss"),
            }
        )
    json_path = out_dir / "unitree_student_training_endpoints.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = out_dir / "unitree_student_training_endpoints.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps({"plot": str(plot_path), "json": str(json_path), "csv": str(csv_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

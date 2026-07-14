#!/usr/bin/env python3
"""Plot teacher-free Unitree student checkpoint evaluations and baselines."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STEP_RE = re.compile(
    r"^(scratch|goalpretrained|scratch2500_strongbc1|teacher_distill_bc20)_effective_step_(\d+)$"
)
DISTILL_STEP_RE = re.compile(r"^teacher_distill_goalinit_bc5_step_(\d+)$")
REFERENCE_LABELS = {
    "direct_goal": "Direct-goal baseline",
    "goalonly_pretrain": "Goal-only policy",
    "geom_teacher_clear06": "Geometric teacher",
}


def load_metrics(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    series: dict[str, list[dict[str, float]]] = {
        "scratch": [],
        "goalpretrained": [],
        "scratch2500_strongbc1": [],
        "teacher_distill_bc5": [],
        "teacher_distill_bc20": [],
    }
    references: dict[str, dict[str, float]] = {}
    for path in sorted(eval_root.glob("*/*_metrics.json")):
        run_name = path.parent.name
        metrics = load_metrics(path)
        match = STEP_RE.match(run_name)
        if match:
            branch, step = match.groups()
            series[branch].append({"step": int(step), **metrics})
        elif match := DISTILL_STEP_RE.match(run_name):
            series["teacher_distill_bc5"].append({"step": int(match.group(1)), **metrics})
        elif run_name in REFERENCE_LABELS:
            references[run_name] = metrics

    if not any(series.values()):
        raise ValueError(f"No checkpoint evaluations found under {eval_root}")
    for rows in series.values():
        rows.sort(key=lambda row: row["step"])

    panels = [
        ("success_rate", "Success rate", (0.0, 1.05)),
        ("mean_cost_sum", "Mean episode cost", None),
        ("costful_episode_rate", "Costful episode rate", (0.0, 1.05)),
    ]
    colors = {
        "scratch": "#167d91",
        "goalpretrained": "#d17632",
        "scratch2500_strongbc1": "#bc4b7a",
        "teacher_distill_bc5": "#8359a3",
        "teacher_distill_bc20": "#4d78bd",
    }
    labels = {
        "scratch": "Scratch",
        "goalpretrained": "Goal-pretrained",
        "scratch2500_strongbc1": "Scratch + stronger BC",
        "teacher_distill_bc5": "Teacher distill BC5",
        "teacher_distill_bc20": "Teacher distill BC20",
    }
    reference_styles = {
        "direct_goal": ("#9c2f2f", ":"),
        "goalonly_pretrain": ("#635b8a", "-."),
        "geom_teacher_clear06": ("#3a7d44", "--"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), dpi=170)
    for ax, (key, title, ylim) in zip(axes, panels):
        for branch, rows in series.items():
            if rows:
                ax.plot(
                    [row["step"] for row in rows],
                    [row[key] for row in rows],
                    marker="o",
                    linewidth=2.2,
                    color=colors[branch],
                    label=labels[branch],
                )
        for name, metrics in references.items():
            color, linestyle = reference_styles[name]
            ax.axhline(
                metrics[key],
                color=color,
                linestyle=linestyle,
                linewidth=1.7,
                alpha=0.9,
                label=REFERENCE_LABELS[name],
            )
        ax.set_title(title)
        ax.set_xlabel("effective optimizer step")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Unitree navigation: teacher-free 100-episode evaluation", y=0.99)
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(5, len(handles)),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "unitree_student_eval_comparison.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    table_rows = []
    for branch, rows in series.items():
        for row in rows:
            table_rows.append(
                {
                    "method": labels[branch],
                    "effective_step": row["step"],
                    "episodes": row["num_episodes"],
                    "success_rate": row["success_rate"],
                    "mean_cost_sum": row["mean_cost_sum"],
                    "costful_episode_rate": row["costful_episode_rate"],
                    "mean_collision_steps": row.get("mean_collision_steps"),
                    "mean_time_to_success_s_success_only": row.get(
                        "mean_time_to_success_s_success_only"
                    ),
                }
            )
    for name, metrics in references.items():
        table_rows.append(
            {
                "method": REFERENCE_LABELS[name],
                "effective_step": "reference",
                "episodes": metrics["num_episodes"],
                "success_rate": metrics["success_rate"],
                "mean_cost_sum": metrics["mean_cost_sum"],
                "costful_episode_rate": metrics["costful_episode_rate"],
                "mean_collision_steps": metrics.get("mean_collision_steps"),
                "mean_time_to_success_s_success_only": metrics.get(
                    "mean_time_to_success_s_success_only"
                ),
            }
        )

    json_path = output_dir / "unitree_student_eval_comparison.json"
    json_path.write_text(json.dumps(table_rows, indent=2), encoding="utf-8")
    csv_path = output_dir / "unitree_student_eval_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps({"plot": str(plot_path), "json": str(json_path), "csv": str(csv_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

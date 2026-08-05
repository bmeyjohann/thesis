#!/usr/bin/env python3
"""Compare learning speed of the combined Unitree thesis method and its ablations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


VARIANTS = ("thesis", "pref_only", "bc_only")
LABELS = {
    "thesis": "Thesis (preference + BC)",
    "pref_only": "Preference-only",
    "bc_only": "BC-only",
}
COLORS = {
    "thesis": "tab:blue",
    "pref_only": "#17a2b8",
    "bc_only": "#d65f9e",
}
MARKERS = {"thesis": "o", "pref_only": "P", "bc_only": "X"}
EVAL_METRICS = ("success_rate", "safe_success_rate", "mean_cost_sum", "costful_episode_rate")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_dir(models_root: Path, variant: str, seed: int, comparison_suffix: str, ablation_suffix: str) -> Path:
    name = (
        f"unitree_compare_thesis_seed{seed}_{comparison_suffix}"
        if variant == "thesis"
        else f"unitree_ablation_{variant}_seed{seed}_{ablation_suffix}"
    )
    result = models_root / name
    if not (result / "final.pt").exists():
        raise FileNotFoundError(f"Missing completed run: {result}")
    return result


def nearest_train_row(train_rows: list[dict], transitions: float) -> dict:
    return min(train_rows, key=lambda row: abs(float(row["transitions"]) - transitions))


def collect(models_root: Path, comparison_suffix: str, ablation_suffix: str, seeds: list[int]) -> list[dict]:
    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            path = run_dir(models_root, variant, seed, comparison_suffix, ablation_suffix)
            train_rows = read_jsonl(path / "metrics.jsonl")
            for eval_row in read_jsonl(path / "eval_metrics.jsonl"):
                train_row = nearest_train_row(train_rows, float(eval_row["transitions"]))
                rows.append(
                    {
                        "variant": variant,
                        "label": LABELS[variant],
                        "seed": seed,
                        "transitions": float(eval_row["transitions"]),
                        "vector_steps": float(eval_row["step"]),
                        "eval_episodes": int(float(eval_row["episodes"])),
                        **{metric: float(eval_row[metric]) for metric in EVAL_METRICS},
                        "teacher_fraction_cumulative": float(train_row["teacher_fraction_cumulative"]),
                    }
                )
    return rows


def summarize(rows: list[dict], human_control_hz: float) -> list[dict]:
    summaries = []
    for variant in VARIANTS:
        for transitions in sorted({row["transitions"] for row in rows}):
            cohort = [row for row in rows if row["variant"] == variant and row["transitions"] == transitions]
            if not cohort:
                continue
            summary = {
                "variant": variant,
                "label": LABELS[variant],
                "transitions": transitions,
                "vector_steps": cohort[0]["vector_steps"],
                "serial_human_supervision_minutes": transitions / human_control_hz / 60.0,
                "num_seeds": len(cohort),
                "eval_episodes_per_seed": cohort[0]["eval_episodes"],
                "total_eval_rollouts": sum(row["eval_episodes"] for row in cohort),
            }
            for metric in (*EVAL_METRICS, "teacher_fraction_cumulative"):
                values = [row[metric] for row in cohort]
                summary[metric] = statistics.mean(values)
                summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
            summaries.append(summary)
    return summaries


def threshold_summary(rows: list[dict], human_control_hz: float) -> list[dict]:
    result = []
    for variant in VARIANTS:
        curve = [row for row in rows if row["variant"] == variant]
        first_safe = next((row for row in curve if row["safe_success_rate"] >= 0.5), None)
        first_safe_low_cost = next(
            (
                row
                for row in curve
                if row["safe_success_rate"] >= 0.5 and row["costful_episode_rate"] <= 0.25
            ),
            None,
        )
        best = max(curve, key=lambda row: row["safe_success_rate"])
        result.append(
            {
                "variant": variant,
                "label": LABELS[variant],
                "first_mean_safe_success_at_least_0_5_transitions": None if first_safe is None else first_safe["transitions"],
                "first_mean_safe_success_at_least_0_5_vector_steps": None if first_safe is None else first_safe["vector_steps"],
                "first_mean_safe_success_at_least_0_5_serial_human_minutes": None
                if first_safe is None
                else first_safe["serial_human_supervision_minutes"],
                "first_safe_success_at_least_0_5_and_costful_rate_at_most_0_25_transitions": None
                if first_safe_low_cost is None
                else first_safe_low_cost["transitions"],
                "first_safe_success_at_least_0_5_and_costful_rate_at_most_0_25_serial_human_minutes": None
                if first_safe_low_cost is None
                else first_safe_low_cost["serial_human_supervision_minutes"],
                "best_safe_success": best["safe_success_rate"],
                "best_safe_success_transitions": best["transitions"],
                "best_safe_success_serial_human_minutes": best["serial_human_supervision_minutes"],
            }
        )
    return result


def plot(summaries: list[dict], output_path: Path, *, human_time: bool, human_control_hz: float) -> None:
    fig, axes_grid = plt.subplots(2, 2, figsize=(15.5, 10.5), sharex=True, facecolor="#f7f4ec")
    panels = (
        ("safe_success_rate", "Teacher-free safe success", (0.0, 1.02), False),
        ("success_rate", "Teacher-free success", (0.0, 1.02), False),
        ("costful_episode_rate", "Costful episode rate", (0.0, 1.02), False),
        ("teacher_fraction_cumulative", "Cumulative intervention fraction", (0.0, 1.02), False),
    )
    for axis, (metric, title, ylim, use_symlog) in zip(axes_grid.ravel(), panels):
        axis.set_facecolor("#fffdf7")
        for variant in VARIANTS:
            curve = [row for row in summaries if row["variant"] == variant]
            x = (
                [row["serial_human_supervision_minutes"] for row in curve]
                if human_time
                else [row["transitions"] / 1000.0 for row in curve]
            )
            y = [row[metric] for row in curve]
            error = [row[f"{metric}_sd"] for row in curve]
            axis.plot(x, y, color=COLORS[variant], marker=MARKERS[variant], linewidth=2.1, label=LABELS[variant])
            axis.fill_between(x, [value - spread for value, spread in zip(y, error)], [value + spread for value, spread in zip(y, error)], color=COLORS[variant], alpha=0.16)
        axis.set(
            title=title,
            xlabel=(
                f"Estimated serial human supervision time (minutes, {human_control_hz:g} Hz)"
                if human_time
                else "Environment transitions (thousands)"
            ),
            ylabel=title,
            ylim=ylim,
        )
        if use_symlog:
            axis.set_yscale("symlog", linthresh=1.0)
        axis.grid(alpha=0.27, color="#82796a")

    handles, labels = axes_grid[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.012), fontsize=10)
    title_prefix = "Human-time learning speed" if human_time else "Learning speed"
    fig.suptitle(f"{title_prefix}: combined thesis method vs. individual auxiliary losses", fontsize=15, fontweight="bold", y=0.986)
    fig.text(
        0.5,
        0.948,
        (
            f"One-human serial estimate at {human_control_hz:g} Hz: 8k transitions = {8000 / human_control_hz / 60.0:.1f} min; "
            f"40k transitions = {40000 / human_control_hz / 60.0:.1f} min. "
            if human_time
            else "Teacher-free evaluation every 8k transitions (1k vector steps); "
        )
        + "each point aggregates 3 seeds x 16 matched layouts = 48 rollouts. Shading = sample SD.",
        ha="center",
        fontsize=9.5,
        color="#514a40",
    )
    fig.tight_layout(rect=(0.025, 0.07, 0.98, 0.91))
    fig.savefig(output_path, dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, default=Path("models/unitree_mjlab_nav_thesis"))
    parser.add_argument("--comparison-suffix", default="20260719_paired3seed")
    parser.add_argument("--ablation-suffix", default="20260720_paired3seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path("visualizations/unitree_competitors_20260720_multiseed"))
    parser.add_argument("--human-control-hz", type=float, default=20.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.human_control_hz <= 0:
        raise ValueError("--human-control-hz must be positive")
    seed_rows = collect(args.models_root, args.comparison_suffix, args.ablation_suffix, args.seeds)
    summaries = summarize(seed_rows, args.human_control_hz)
    thresholds = threshold_summary(summaries, args.human_control_hz)
    write_csv(args.output_dir / "thesis_ablation_learning_by_seed.csv", seed_rows)
    write_csv(args.output_dir / "thesis_ablation_learning_summary.csv", summaries)
    write_csv(args.output_dir / "thesis_ablation_learning_thresholds.csv", thresholds)
    output_path = args.output_dir / "thesis_ablation_learning_curves.png"
    plot(summaries, output_path, human_time=False, human_control_hz=args.human_control_hz)
    human_time_output_path = args.output_dir / "thesis_ablation_learning_curves_human_time.png"
    plot(summaries, human_time_output_path, human_time=True, human_control_hz=args.human_control_hz)
    print(output_path.resolve())
    print(human_time_output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

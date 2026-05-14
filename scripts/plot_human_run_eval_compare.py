#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "local" / "human_run_eval_compare"
DEFAULT_LOGS = (
    REPO_ROOT / "logs" / "cube_single_task1_human_collect_norot_fixedalpha1e3_20260408_143217.log",
    REPO_ROOT / "logs" / "cube_single_task1_human_onlycollecteddemos_demoaug.log",
)
DEFAULT_LABELS = ("collect", "demoaug")
DEFAULT_COLORS = ("#1f77b4", "#ff7f0e")

EVAL_LINE_RE = re.compile(r"\[Eval\]\s+steps=(\d+)\s+(.*)")
KEYVAL_RE = re.compile(r"(Eval/[A-Za-z0-9_]+)=([-+0-9.eE]+)")


@dataclass(frozen=True)
class RunSpec:
    label: str
    log_path: Path
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eval comparisons for selected human-manip training runs.")
    parser.add_argument(
        "--logs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_LOGS),
        help="Training log paths to compare.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Legend labels for the selected logs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory for CSV and plots (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    return parser.parse_args()


def extract_eval_rows(log_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = EVAL_LINE_RE.search(line)
            if not match:
                continue
            step = float(match.group(1))
            metrics = {key: float(value) for key, value in KEYVAL_RE.findall(match.group(2))}
            rows.append(
                {
                    "step": step,
                    "avg_length": metrics.get("Eval/avg_length", float("nan")),
                    "success_rate": metrics.get("Eval/success_rate", float("nan")),
                    "avg_success_length": metrics.get("Eval/avg_success_length", float("nan")),
                }
            )
    return rows


def write_eval_csv(output_path: Path, run_rows: list[tuple[RunSpec, list[dict[str, float]]]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "log_path", "step", "success_rate", "avg_length", "avg_success_length"],
        )
        writer.writeheader()
        for spec, rows in run_rows:
            for row in rows:
                writer.writerow(
                    {
                        "label": spec.label,
                        "log_path": str(spec.log_path),
                        "step": int(row["step"]),
                        "success_rate": row["success_rate"],
                        "avg_length": row["avg_length"],
                        "avg_success_length": row["avg_success_length"],
                    }
                )


def format_k_ticks(axis: plt.Axes) -> None:
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: "0" if value <= 0 else f"{int(round(value / 1000.0))}k")
    )


def plot_panel(axis: plt.Axes, run_rows: list[tuple[RunSpec, list[dict[str, float]]]], metric: str, title: str, ylabel: str) -> None:
    for spec, rows in run_rows:
        xs = [row["step"] for row in rows]
        ys = [row[metric] for row in rows]
        axis.plot(
            xs,
            ys,
            color=spec.color,
            linewidth=2.5,
            marker="o",
            markersize=7,
            label=spec.label,
        )
    axis.set_title(title, fontsize=16, pad=10)
    axis.set_xlabel("Env steps", fontsize=12)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.grid(True, alpha=0.35)
    format_k_ticks(axis)


def save_single_panel(
    output_path: Path,
    run_rows: list[tuple[RunSpec, list[dict[str, float]]]],
    metric: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    fig, axis = plt.subplots(1, 1, figsize=(6.2, 4.6), constrained_layout=True)
    plot_panel(axis, run_rows, metric, title, ylabel)
    if ylim is not None:
        axis.set_ylim(*ylim)
    else:
        axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, loc="upper right")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.logs) != len(args.labels):
        raise SystemExit("--logs and --labels must have the same length")

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    specs = [
        RunSpec(label=label, log_path=log_path, color=DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])
        for idx, (label, log_path) in enumerate(zip(args.labels, args.logs, strict=True))
    ]

    run_rows = [(spec, extract_eval_rows(spec.log_path)) for spec in specs]
    missing = [str(spec.log_path) for spec, rows in run_rows if not rows]
    if missing:
        raise SystemExit(f"No eval rows parsed from: {', '.join(missing)}")

    write_eval_csv(output_root / "human_run_eval_compare.csv", run_rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    plot_panel(axes[0], run_rows, "success_rate", "Success rate", "Eval success rate")
    plot_panel(axes[1], run_rows, "avg_length", "Episode length", "Eval avg length")
    plot_panel(axes[2], run_rows, "avg_success_length", "Successful episode length", "Eval avg success length")

    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(bottom=0.0)
    axes[2].set_ylim(bottom=0.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(specs), frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Human online run comparison", fontsize=20, y=1.10)

    fig.savefig(output_root / "human_run_eval_compare_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    save_single_panel(
        output_root / "human_run_success_rate.png",
        run_rows,
        "success_rate",
        "Success rate",
        "Eval success rate",
        ylim=(-0.02, 1.02),
    )
    save_single_panel(
        output_root / "human_run_avg_length.png",
        run_rows,
        "avg_length",
        "Episode length",
        "Eval avg length",
    )
    save_single_panel(
        output_root / "human_run_avg_success_length.png",
        run_rows,
        "avg_success_length",
        "Successful episode length",
        "Eval avg success length",
    )


if __name__ == "__main__":
    main()

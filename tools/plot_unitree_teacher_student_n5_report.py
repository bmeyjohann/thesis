#!/usr/bin/env python3
"""Build a compact Unitree teacher-student training and evaluation report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=PATH, got {raw!r}")
    label, path = raw.split("=", 1)
    return label, Path(path)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _smooth(values: list[float], window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2 or window <= 1:
        return arr
    width = min(window, arr.size)
    kernel = np.ones(width, dtype=float) / width
    padded = np.pad(arr, (width - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _metric_series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    pairs = [
        (float(row["transitions"]), float(row[key]))
        for row in rows
        if key in row and np.isfinite(float(row[key]))
    ]
    if not pairs:
        return np.zeros(0), np.zeros(0)
    return np.asarray([p[0] for p in pairs]), np.asarray([p[1] for p in pairs])


def _final_row(label: str, path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = list(data.get("episodes", []))
    n = int(data.get("num_episodes", len(episodes)))
    safe_successes = sum(
        bool(ep.get("success", False)) and float(ep.get("cost_sum", 0.0)) <= 0.0
        for ep in episodes
    )
    safe_success = safe_successes / max(1, len(episodes)) if episodes else None
    return {
        "method": label,
        "metrics_path": str(path),
        "episodes": n,
        "success_rate": float(data["success_rate"]),
        "safe_success_rate": safe_success,
        "mean_cost_sum": float(data["mean_cost_sum"]),
        "costful_episode_rate": float(data["costful_episode_rate"]),
        "mean_collision_steps": float(data.get("mean_collision_steps", 0.0)),
        "mean_fall_cost_sum": float(data.get("mean_fall_cost_sum", 0.0)),
        "mean_time_to_success_s": float(data.get("mean_time_to_success_s_success_only", 0.0)),
        "mean_action_sign_flips": float(data.get("mean_action_sign_flips", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=[], help="LABEL=training_run_dir")
    parser.add_argument("--final", action="append", default=[], help="LABEL=policy_metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, Path, list[dict], list[dict]]] = []
    for raw in args.run:
        label, path = _spec(raw)
        train_rows = _jsonl(path / "metrics.jsonl")
        eval_path = path / "eval_metrics.jsonl"
        eval_rows = _jsonl(eval_path) if eval_path.exists() else []
        runs.append((label, path, train_rows, eval_rows))

    if runs:
        panels = [
            ("teacher_fraction_interval", "Teacher fraction (interval)", (0.0, 1.05)),
            ("teacher_fraction_cumulative", "Teacher fraction (cumulative)", (0.0, 1.05)),
            ("success_rate", "Teacher-gated train success", (0.0, 1.05)),
            ("episode_cost_mean", "Teacher-gated mean episode cost", None),
            ("pref_lambda", "Preference multiplier", None),
            ("q_min_pi_mean", "Policy Q estimate", None),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=170)
        for axis, (key, title, ylim) in zip(axes.reshape(-1), panels):
            for label, _, rows, _ in runs:
                x, y = _metric_series(rows, key)
                if x.size:
                    axis.plot(x, _smooth(y.tolist()), linewidth=2, label=label)
            axis.set_title(title)
            axis.set_xlabel("Environment transitions")
            if ylim:
                axis.set_ylim(*ylim)
            axis.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=min(4, len(handles)))
        if all(not _metric_series(rows, "episode_cost_mean")[1].any() for _, _, rows, _ in runs):
            axes[1, 0].set_ylim(0.0, 0.1)
        fig.suptitle("Unitree navigation training diagnostics (teacher active)", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(args.output_dir / "training_diagnostics.png", bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), dpi=170)
        for label, _, _, rows in runs:
            if not rows:
                continue
            steps = [float(row["step"]) for row in rows]
            axes[0].plot(steps, [100.0 * float(row["success_rate"]) for row in rows], marker="o", label=label)
            axes[1].plot(steps, [float(row["mean_cost_sum"]) for row in rows], marker="o", label=label)
            axes[2].plot(steps, [float(row["mean_action_sign_flips"]) for row in rows], marker="o", label=label)
        for axis, title, ylabel in zip(
            axes,
            ("Success", "Cost", "Action oscillation"),
            ("Teacher-free success (%)", "Mean episode cost", "Mean sign flips"),
        ):
            axis.set_title(title)
            axis.set_xlabel("Training step")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=min(4, len(handles)))
        fig.suptitle("Deterministic teacher-free checkpoint evaluations", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.80))
        fig.savefig(args.output_dir / "teacher_free_eval_curves.png", bbox_inches="tight")
        plt.close(fig)

    final_rows = [_final_row(*_spec(raw)) for raw in args.final]
    if final_rows:
        labels = [row["method"] for row in final_rows]
        x = np.arange(len(labels))
        fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), dpi=170)
        axes[0].bar(x, [100.0 * row["success_rate"] for row in final_rows], color="#2f7185")
        axes[0].set_ylabel("Success (%)")
        axes[0].set_ylim(0, 105)
        axes[1].bar(x, [100.0 * (row["safe_success_rate"] or 0.0) for row in final_rows], color="#4d8b57")
        axes[1].set_ylabel("Safe success (%)")
        axes[1].set_ylim(0, 105)
        axes[2].bar(x, [row["mean_cost_sum"] for row in final_rows], color="#b0443c")
        axes[2].set_ylabel("Mean episode cost")
        axes[3].bar(x, [100.0 * row["costful_episode_rate"] for row in final_rows], color="#c9852d")
        axes[3].set_ylabel("Costful episodes (%)")
        for axis in axes:
            axis.set_xticks(x, labels, rotation=24, ha="right")
            axis.grid(axis="y", alpha=0.25)
        fig.suptitle("Matched 40-episode Unitree navigation benchmark")
        fig.tight_layout()
        fig.savefig(args.output_dir / "final_eval_comparison.png", bbox_inches="tight")
        plt.close(fig)

        json_path = args.output_dir / "final_eval_table.json"
        json_path.write_text(json.dumps(final_rows, indent=2), encoding="utf-8")
        csv_path = args.output_dir / "final_eval_table.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
            writer.writeheader()
            writer.writerows(final_rows)

    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

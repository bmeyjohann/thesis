#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    arrays = {}
    for part in manifest["parts"]:
        with np.load(path / part["file"]) as data:
            for key in data.files:
                arrays.setdefault(key, []).append(data[key])
    arrays = {key: np.concatenate(values) for key, values in arrays.items()}
    hz = float(manifest["metadata"]["high_level_control_hz"])
    timestamps = arrays["wall_time_unix_s"]
    return {
        "rows": len(timestamps),
        "intervention_rows": int(arrays["intervened"].sum()),
        "intervention_starts": int(arrays["intervention_start"].sum()),
        "active_sim_s": len(timestamps) / hz,
        "active_intervention_s": float(arrays["intervened"].sum()) / hz,
        "wall_clock_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "goals_reached": int(arrays["terminal_success"].sum()),
        "cost_sum": float(arrays["cost"].sum()),
        "costful_rows": int((arrays["cost"] > 0).sum()),
        "cost_sum_during_intervention": float(arrays["cost"][arrays["intervened"]].sum()),
        "cost_sum_without_intervention": float(arrays["cost"][~arrays["intervened"]].sum()),        "timestamps": timestamps,
        "intervened": arrays["intervened"].astype(float),
        "cost": arrays["cost"],
        "success": arrays["terminal_success"].astype(float),
        "goal_distance": arrays["goal_distance"],
        "hz": hz,
    }


def step_from_name(path: Path, final_step: int):
    name = path.parent.name
    match = re.search(r"step_(\d+)", name)
    return int(match.group(1)) if match else final_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--prior-dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    final_step = int(training[-1]["step"])
    eval_rows = []
    for path in sorted(args.eval_root.glob("frustum17_*_eval16_h640/policy_metrics.json")):
        data = json.loads(path.read_text())
        episodes = data.get("episodes", [])
        safe_success = np.mean([bool(e["success"]) and float(e["cost_sum"]) == 0.0 for e in episodes])
        eval_rows.append({
            "step": step_from_name(path, final_step),
            "checkpoint": path.parent.name,
            "success_rate": float(data["success_rate"]),
            "safe_success_rate": float(safe_success),
            "costful_episode_rate": float(data["costful_episode_rate"]),
            "mean_cost_sum": float(data["mean_cost_sum"]),
            "mean_collision_steps": float(data["mean_collision_steps"]),
            "mean_time_to_success_s": float(data["mean_time_to_success_s_success_only"]),
            "mean_min_goal_distance": float(data["mean_min_goal_distance"]),
            "mean_action_delta": float(data["mean_action_delta"]),
            "num_episodes": int(data["num_episodes"]),
        })
    eval_rows.sort(key=lambda row: row["step"])

    dataset = load_dataset(args.dataset_dir)
    prior = load_dataset(args.prior_dataset_dir) if args.prior_dataset_dir else None

    with (args.output_dir / "checkpoint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eval_rows[0]))
        writer.writeheader()
        writer.writerows(eval_rows)

    steps = np.asarray([row["step"] for row in eval_rows])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    ax = axes[0, 0]
    ax.plot(steps, [r["success_rate"] for r in eval_rows], "o-", label="success")
    ax.plot(steps, [r["safe_success_rate"] for r in eval_rows], "o-", label="safe success")
    ax.set_ylabel("Rate")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    ax.grid(alpha=.25)

    ax = axes[0, 1]
    ax.plot(steps, [r["mean_cost_sum"] for r in eval_rows], "o-", color="#b7352b", label="mean cost")
    ax2 = ax.twinx()
    ax2.plot(steps, [r["costful_episode_rate"] for r in eval_rows], "s--", color="#d88722", label="costful-goal rate")
    ax.set_ylabel("Mean cost sum")
    ax2.set_ylabel("Costful-goal rate")
    ax.grid(alpha=.25)

    ax = axes[1, 0]
    ax.plot(steps, [r["mean_time_to_success_s"] for r in eval_rows], "o-", label="success time")
    ax.plot(steps, [r["mean_min_goal_distance"] for r in eval_rows], "s--", label="minimum goal distance")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Seconds / meters")
    ax.legend()
    ax.grid(alpha=.25)

    ax = axes[1, 1]
    train_steps = np.asarray([row["step"] for row in training])
    ax.plot(train_steps, [row["intervention_fraction"] for row in training], label="cumulative intervention fraction")
    ax.plot(train_steps, [row.get("pref_lambda", np.nan) for row in training], label="preference lambda")
    ax2 = ax.twinx()
    ax2.plot(train_steps, [row.get("q_min_pi_mean", np.nan) for row in training], color="#2b6f9b", alpha=.8, label="Q min policy")
    ax2.plot(train_steps, [row.get("q_min_data_mean", np.nan) for row in training], color="#59a1c7", alpha=.8, label="Q min data")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Fraction / lambda")
    ax2.set_ylabel("Q value")
    ax.grid(alpha=.25)
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8)

    for ax in axes.flat:
        if ax.get_xlabel() == "":
            ax.set_xlabel("Training steps")
    fig.suptitle("Unitree human pref + RL, forward frustum: checkpoint evaluation (16 fixed-seed goals)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "checkpoint_dashboard.png", bbox_inches="tight")
    plt.close(fig)

    window = max(1, int(dataset["hz"] * 30))
    intervention = dataset["intervened"]
    kernel = np.ones(window) / window
    smooth_intervention = np.convolve(intervention, kernel, mode="same")
    elapsed_min = (dataset["timestamps"] - dataset["timestamps"][0]) / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=180, sharex=True)
    axes[0].plot(elapsed_min, smooth_intervention, color="#b7352b")
    axes[0].set_ylabel("Intervention fraction\n(30 s rolling)")
    axes[0].set_ylim(-.03, 1.03)
    axes[0].grid(alpha=.25)
    axes[1].plot(elapsed_min, dataset["goal_distance"], color="#2b6f9b", alpha=.75, label="goal distance")
    cost_idx = np.flatnonzero(dataset["cost"] > 0)
    success_idx = np.flatnonzero(dataset["success"] > 0)
    axes[1].scatter(elapsed_min[cost_idx], dataset["goal_distance"][cost_idx], c="red", s=9, label="cost")
    axes[1].scatter(elapsed_min[success_idx], dataset["goal_distance"][success_idx], c="limegreen", marker="*", s=70, label="goal reached")
    axes[1].set_xlabel("Human session wall time (minutes)")
    axes[1].set_ylabel("Goal distance (m)")
    axes[1].legend()
    axes[1].grid(alpha=.25)
    fig.suptitle("Human intervention session timeline")
    fig.tight_layout()
    fig.savefig(args.output_dir / "human_session_timeline.png", bbox_inches="tight")
    plt.close(fig)

    labels = ["Current frustum run"]
    total = [dataset["active_sim_s"] / 60]
    steering = [dataset["active_intervention_s"] / 60]
    if prior:
        labels.append("Prior pref+RL+BC run")
        total.append(prior["active_sim_s"] / 60)
        steering.append(prior["active_intervention_s"] / 60)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.bar(x, total, width=.62, label="active data collection")
    ax.bar(x, steering, width=.62, label="active human steering")
    for i, (a, b) in enumerate(zip(total, steering)):
        ax.text(i, a + .2, f"{a:.2f} min total", ha="center")
        ax.text(i, b / 2, f"{b:.2f} min steering", ha="center", color="white", weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Minutes")
    ax.set_title("Human supervision effort")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "human_time_comparison.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "run_dir": str(args.run_dir),
        "dataset_dir": str(args.dataset_dir),
        "final_step": final_step,
        "human_time": {key: value for key, value in dataset.items() if key not in {"timestamps", "intervened", "cost", "success", "goal_distance"}},
        "prior_human_time": (
            {key: value for key, value in prior.items() if key not in {"timestamps", "intervened", "cost", "success", "goal_distance"}}
            if prior else None
        ),
        "checkpoint_metrics": eval_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

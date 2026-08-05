#!/usr/bin/env python3
"""Summarize and plot an immutable Unitree online-human run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_dataset(root: Path) -> dict[str, np.ndarray]:
    columns: dict[str, list[np.ndarray]] = {}
    for path in sorted((root / "parts").glob("*.npz")):
        with np.load(path) as part:
            for key in part.files:
                columns.setdefault(key, []).append(part[key])
    return {key: np.concatenate(values) for key, values in columns.items()}


def _rolling(values: np.ndarray, window: int = 20) -> np.ndarray:
    if values.size == 0:
        return values
    width = min(window, values.size)
    return np.convolve(values, np.ones(width) / width, mode="valid")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in (args.run_dir / "metrics.jsonl").read_text().splitlines()]
    data = _load_dataset(args.dataset_dir)
    episode_ids = np.unique(data["episode_index"])
    episodes: list[dict[str, float]] = []
    for episode_id in episode_ids:
        mask = data["episode_index"] == episode_id
        episodes.append(
            {
                "episode": int(episode_id),
                "steps": int(mask.sum()),
                "success": float(data["terminal_success"][mask].any()),
                "cost_sum": float(data["cost"][mask].sum()),
                "cost_event": float((data["cost"][mask] > 0).any()),
                "intervention_fraction": float(data["intervened"][mask].mean()),
                "intervention_starts": int(data["intervention_start"][mask].sum()),
                "final_goal_distance": float(data["next_goal_distance"][mask][-1]),
            }
        )

    intervened = data["intervened"].astype(bool)
    cost_positive = data["cost"] > 0
    burst_starts = np.flatnonzero(data["intervention_start"])
    burst_ends = np.flatnonzero(data["intervention_end"])
    durations = []
    for start in burst_starts:
        end_candidates = burst_ends[burst_ends >= start]
        end = int(end_candidates[0]) if end_candidates.size else len(intervened) - 1
        durations.append(end - int(start) + 1)

    summary = {
        "transitions": int(len(intervened)),
        "episodes_observed": int(len(episodes)),
        "success_rate_human_assisted": float(np.mean([row["success"] for row in episodes])),
        "mean_episode_cost_human_assisted": float(np.mean([row["cost_sum"] for row in episodes])),
        "cost_episode_fraction_human_assisted": float(np.mean([row["cost_event"] for row in episodes])),
        "intervention_rows": int(intervened.sum()),
        "intervention_fraction": float(intervened.mean()),
        "intervention_bursts": int(len(burst_starts)),
        "mean_burst_steps": float(np.mean(durations)) if durations else 0.0,
        "median_burst_steps": float(np.median(durations)) if durations else 0.0,
        "cost_sum_total": float(data["cost"].sum()),
        "cost_events_total": int(cost_positive.sum()),
        "cost_events_during_human": int((cost_positive & intervened).sum()),
        "cost_events_during_policy": int((cost_positive & ~intervened).sum()),
        "mean_action_delta_on_intervention": float(data["action_delta_to_policy"][intervened].mean()),
        "transport_connected_fraction": float(data["gamepad_connected"].mean()),
        "transport_stale_fraction": float(data["gamepad_stale"].mean()),
        "wall_time_hours": float((data["wall_time_unix_s"][-1] - data["wall_time_unix_s"][0]) / 3600.0),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n")

    episode_x = np.arange(1, len(episodes) + 1)
    success = np.asarray([row["success"] for row in episodes])
    costs = np.asarray([row["cost_sum"] for row in episodes])
    takeover = np.asarray([row["intervention_fraction"] for row in episodes])
    window = min(20, len(episodes))
    rolling_x = episode_x[window - 1 :]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(rolling_x, _rolling(success), label=f"rolling {window} episodes")
    axes[0, 0].set(title="Human-assisted success", ylabel="success rate", ylim=(-0.03, 1.03))
    axes[0, 1].plot(rolling_x, _rolling(costs), color="crimson")
    axes[0, 1].set(title="Human-assisted episode cost", ylabel="mean cost")
    axes[1, 0].plot(rolling_x, _rolling(takeover), color="darkorange")
    axes[1, 0].set(title="Human takeover fraction", ylabel="fraction", xlabel="episode")
    axes[1, 1].scatter(episode_x, costs, c=takeover, cmap="viridis", s=20)
    axes[1, 1].set(title="Cost versus training episode", ylabel="cost", xlabel="episode")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("Unitree online human-intervention training outcomes")
    fig.savefig(args.output_dir / "human_assisted_episode_curves.png", dpi=180)
    plt.close(fig)

    steps = np.asarray([row["step"] for row in rows], dtype=float)
    def series(key: str) -> np.ndarray:
        return np.asarray([row.get(key, np.nan) for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].plot(steps, series("intervention_fraction"), label="cumulative takeover")
    axes[0, 0].plot(steps, series("batch_teacher_fraction"), alpha=0.5, label="sampled batch")
    axes[0, 0].set(title="Intervention exposure", ylabel="fraction")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, series("pref_lambda"), color="purple")
    axes[0, 1].set(title="Preference multiplier", ylabel="lambda")
    axes[1, 0].plot(steps, series("q_min_pi_mean"), label="Q policy")
    axes[1, 0].plot(steps, series("q_min_data_mean"), label="Q data")
    axes[1, 0].plot(steps, series("target_q_mean"), label="target Q")
    axes[1, 0].set(title="Q-value diagnostics", ylabel="Q", xlabel="transition")
    axes[1, 0].legend()
    axes[1, 1].plot(steps, series("critic_loss_replay"), label="TD critic")
    axes[1, 1].plot(steps, series("critic_loss_pref"), label="preference")
    axes[1, 1].plot(steps, series("actor_loss_bc"), label="BC")
    axes[1, 1].set(title="Optimization losses", ylabel="loss", xlabel="transition")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.suptitle("Unitree online learner diagnostics")
    fig.savefig(args.output_dir / "optimization_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
